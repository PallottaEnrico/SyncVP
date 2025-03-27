from abc import abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import Conv1d, Identity

from models.ddpm.utils import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
)

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class AttentionBlock1D(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.heads_dim = dim // num_heads
        self.dim = dim

        self.W_q = Conv1d(dim, dim, 1)
        self.W_k = Conv1d(dim, dim, 1)
        self.W_v = Conv1d(dim, dim, 1)
        self.W_out = Conv1d(dim, dim, 1)

    def forward(self, x, y, skip=False):
        """
        Forward pass of the MultiHeadCrossAttention module
        :param x: tensor of shape (batch_size, dim, seq_len)
        :param y: tensor of shape (batch_size, dim, seq_len)
        :param skip: whether to use skip connection
        :return: x tensor of shape (batch_size, dim, seq_len) attended to y
        """
        batch_size = x.size(0)

        q = self.W_q(x)
        k = self.W_k(y)
        v = self.W_v(y)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.heads_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.heads_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.heads_dim).transpose(1, 2)

        orig_x = x
        x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).contiguous().view(batch_size, self.dim, -1)
        if skip:
            x = orig_x + self.W_out(x)
        else:
            x = self.W_out(x)

        return x


class VideoDualCrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, **kwargs):
        super().__init__()

        if kwargs['normalize']:
            self.norm_1 = normalization(dim)
            self.norm_2 = normalization(dim)
        else:
            self.norm_1 = Identity()
            self.norm_2 = Identity()

        self.cross_attn1 = MultiHeadCrossAttention(dim, num_heads=num_heads)
        self.cross_attn2 = MultiHeadCrossAttention(dim, num_heads=num_heads)

        self.skip = kwargs['skip_conn']

        self.split_attn = kwargs['split_attn']

    def forward(self, x, y):
        if self.split_attn:
            unit = x.size(2) // 3
            x = [x[:, :, :unit * 2], x[:, :, unit * 2:unit * 2 + unit // 2], x[:, :, unit * 2 + unit // 2:]]
            y = [y[:, :, :unit * 2], y[:, :, unit * 2:unit * 2 + unit // 2], y[:, :, unit * 2 + unit // 2:]]

            for i in range(3):
                x[i] = self.norm_1(x[i])
                y[i] = self.norm_2(y[i])
                x[i] = self.cross_attn1(x[i], y[i], self.skip)
                y[i] = self.cross_attn2(y[i], x[i], self.skip)

            x = torch.cat(x, dim=2)
            y = torch.cat(y, dim=2)
        else:
            x = self.norm_1(x)
            y = self.norm_2(y)
            x = self.cross_attn1(x, y, self.skip)
            y = self.cross_attn2(y, x, self.skip)

        return x, y


class MultiHeadSingleCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.heads_dim = dim // num_heads
        self.dim = dim

        self.W_q_x = Conv1d(dim, dim, 1)
        self.W_q_y = Conv1d(dim, dim, 1)
        self.W_v_x = Conv1d(dim, dim, 1)
        self.W_v_y = Conv1d(dim, dim, 1)

        self.W_x_out = Conv1d(dim, dim, 1)
        self.W_y_out = Conv1d(dim, dim, 1)

    def forward(self, x, y, skip=False):
        """
        Forward pass of the MultiHeadCrossAttention module
        :param x: tensor of shape (batch_size, dim, seq_len)
        :param y: tensor of shape (batch_size, dim, seq_len)
        :param skip: whether to use skip connection
        :return: x tensor of shape (batch_size, dim, seq_len) attended to y
        """
        batch_size = x.size(0)

        q_x = self.W_q_x(x)
        q_y = self.W_q_y(y)
        v_x = self.W_v_x(x)
        v_y = self.W_v_y(y)

        # Reshape for multi-head attention
        q_x = q_x.view(batch_size, -1, self.num_heads, self.heads_dim).transpose(1, 2)
        q_y = q_y.view(batch_size, -1, self.num_heads, self.heads_dim).transpose(1, 2)
        v_x = v_x.view(batch_size, -1, self.num_heads, self.heads_dim).transpose(1, 2)
        v_y = v_y.view(batch_size, -1, self.num_heads, self.heads_dim).transpose(1, 2)

        orig_x = x
        orig_y = y

        att_scores = torch.matmul(q_x, q_y.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.heads_dim, dtype=torch.float32))

        x = torch.matmul(F.softmax(att_scores, dim=-1), v_y)
        y = torch.matmul(F.softmax(att_scores.transpose(-2, -1), dim=-1), v_x)

        x = x.transpose(1, 2).contiguous().view(batch_size, self.dim, -1)
        y = y.transpose(1, 2).contiguous().view(batch_size, self.dim, -1)

        if skip:
            x = orig_x + self.W_x_out(x)
            y = orig_y + self.W_y_out(y)
        else:
            x = self.W_x_out(x)
            y = self.W_y_out(y)

        return x, y


class SingleVideoDualCrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, **kwargs):
        super().__init__()

        if kwargs['normalize']:
            self.norm_1 = normalization(dim)
            self.norm_2 = normalization(dim)
        else:
            self.norm_1 = Identity()
            self.norm_2 = Identity()

        self.cross_attn = MultiHeadSingleCrossAttention(dim, num_heads=num_heads)

        self.skip = kwargs['skip_conn']

        self.split_attn = kwargs['split_attn']

        self.shapes = {
            2048: (32, 16),
            1536: (32, 8),
            512: (16, 8),
            384: (16, 4),
            128: (8, 4),
            32: (4, 2),
        }

    def forward(self, x, y):
        if self.split_attn:
            # TODO: add implementation for non-squared resolution
            spatial_dim, temporal_dim = self.shapes[x.size(2)]
            spatial = spatial_dim ** 2
            temporal = spatial + spatial_dim * temporal_dim

            x = [x[:, :, :spatial], x[:, :, spatial:temporal], x[:, :, temporal:]]
            y = [y[:, :, :spatial], y[:, :, spatial:temporal], y[:, :, temporal:]]

            for i in range(3):
                x[i] = self.norm_1(x[i])
                y[i] = self.norm_2(y[i])
                x[i], y[i] = self.cross_attn(x[i], y[i], self.skip)

            x = torch.cat(x, dim=2)
            y = torch.cat(y, dim=2)
        else:
            x = self.norm_1(x)
            y = self.norm_2(y)
            x, y = self.cross_attn(x, y, self.skip)

        return x, y

class NoCross(nn.Module):
    def __init__(self, *args, **kwargs):  # ignore extra arguments on purpose
        super().__init__()

    def forward(self, x, y):
        return x, y
