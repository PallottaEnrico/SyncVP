import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from models.ddpm.utils import (
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)

from models.ddpm.modules import (
    Downsample,
    Upsample,
    ResBlock,
    AttentionBlock,
    AttentionBlock1D,
    TimestepEmbedSequential,
    SpatialTransformer
)

class DiffusionWrapper(nn.Module):
    def __init__(self, model, conditioning_key=None):
        super().__init__()
        self.diffusion_model = model
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, cond, t):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, cond, t)
        else:
            raise NotImplementedError()

        return out





class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        frames=16
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.frames = frames
        if isinstance(self.image_size, int):
            self.ae_emb_dim = (image_size * image_size) + (frames * image_size) + (frames * image_size)
        else:
            self.ae_emb_dim = (image_size[0] * image_size[1]) + (frames * image_size[0]) + (frames * image_size[1])

        self.register_buffer("zeros", torch.zeros(1, self.in_channels, self.ae_emb_dim))

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)


        self.input_blocks = nn.ModuleList(
                [
                    TimestepEmbedSequential(
                        conv_nd(dims, in_channels*2, model_channels, 3, padding=1)
                    )
                ]
            )


        self.input_attns = nn.ModuleList([nn.Identity()])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

                self.input_attns.append(
                            AttentionBlock1D(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ))


            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

                self.input_attns.append(
                            AttentionBlock1D(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ))


        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.mid_attn = AttentionBlock1D(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        self.output_attns = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_attns.append(AttentionBlock1D(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )


    def forward(self, x, cond=None, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        h_xys = []
        h_xts = []
        h_yts = []

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        if cond != None:
            if cond.size(2) != h.size(2):
                # Zero pad the cond tensor to match the size of h along the dimension 2
                cond = F.pad(cond, (0, h.size(2) - cond.size(2)), "constant", 0)
            h = torch.cat([h, cond], dim=1)
        else:
            h = torch.cat([h, self.zeros.repeat(h.size(0), 1, 1)], dim=1)

        # define res1, res2 and t to split latent vector h into 3 parts
        if isinstance(self.image_size, int):
            res1 = self.image_size
            res2 = self.image_size
        else:
            res1 = self.image_size[0]
            res2 = self.image_size[1]
        t = self.timesteps


        h_xy = h[:, :, 0:res1*res2].view(h.size(0), h.size(1), res1, res2)
        h_yt = h[:, :, res1*res2:res2*(res1+t)].view(h.size(0), h.size(1), t, res2)
        h_xt = h[:, :, res2*(res1+t):res2*(res1+t)+res1*t].view(h.size(0), h.size(1), t, res1)

        for module, input_attn in zip(self.input_blocks, self.input_attns):
            h_xy = module(h_xy, emb, context)
            h_yt = module(h_yt, emb, context)
            h_xt = module(h_xt, emb, context)

            res1 = h_xy.size(-2)
            res2 = h_xy.size(-1)
            t    = h_xt.size(-2)

            h_xy = h_xy.view(h_xy.size(0), h_xy.size(1), -1)
            h_yt = h_yt.view(h_yt.size(0), h_yt.size(1), -1)
            h_xt = h_xt.view(h_xt.size(0), h_xt.size(1), -1)

            h = torch.cat([h_xy, h_yt, h_xt], dim=-1)
            h = input_attn(h)

            h_xy = h[:, :, 0:res1 * res2].view(h.size(0), h.size(1), res1, res2)
            h_yt = h[:, :, res1 * res2:res2 * (res1 + t)].view(h.size(0), h.size(1), t, res2)
            h_xt = h[:, :, res2 * (res1 + t):res2 * (res1 + t) + res1 * t].view(h.size(0), h.size(1), t, res1)

            h_xys.append(h_xy)
            h_yts.append(h_yt)
            h_xts.append(h_xt)

        h_xy = self.middle_block(h_xy, emb, context)
        h_yt = self.middle_block(h_yt, emb, context)
        h_xt = self.middle_block(h_xt, emb, context)

        res1 = h_xy.size(-2)
        res2 = h_xy.size(-1)
        t    = h_xt.size(-2)

        h_xy = h_xy.view(h_xy.size(0), h_xy.size(1), -1)
        h_yt = h_yt.view(h_yt.size(0), h_yt.size(1), -1)
        h_xt = h_xt.view(h_xt.size(0), h_xt.size(1), -1)

        h = torch.cat([h_xy, h_yt, h_xt], dim=-1)
        h = self.mid_attn(h)

        h_xy = h[:, :, 0:res1*res2].view(h.size(0), h.size(1), res1, res2)
        h_yt = h[:, :, res1*res2:res2*(res1+t)].view(h.size(0), h.size(1), t, res2)
        h_xt = h[:, :, res2*(res1+t):res2*(res1+t)+res1*t].view(h.size(0), h.size(1), t, res1)

        for module, output_attn in zip(self.output_blocks, self.output_attns):
            h_xy = th.cat([h_xy, h_xys.pop()], dim=1)
            h_xy = module(h_xy, emb, context)
            # Handle shape mismatch in case of number of frames in the video is not a power of 2
            h_yt_pop = h_yts.pop()
            if h_yt.shape[2] != h_yt_pop.shape[2]:
                h_yt = F.pad(h_yt, (0, 0, 0, h_yt_pop.shape[2] - h_yt.shape[2]), "constant", 0)
            h_yt = th.cat([h_yt, h_yt_pop], dim=1)
            h_yt = module(h_yt, emb, context)
            h_xt_pop = h_xts.pop()
            if h_xt.shape[2] != h_xt_pop.shape[2]:
                h_xt = F.pad(h_xt, (0, 0, 0, h_xt_pop.shape[2] - h_xt.shape[2]), "constant", 0)
            h_xt = th.cat([h_xt, h_xt_pop], dim=1)
            h_xt = module(h_xt, emb, context)

            res1 = h_xy.size(-2)
            res2 = h_xy.size(-1)
            t   = h_xt.size(-2)

            h_xy = h_xy.view(h_xy.size(0), h_xy.size(1), -1)
            h_yt = h_yt.view(h_yt.size(0), h_yt.size(1), -1)
            h_xt = h_xt.view(h_xt.size(0), h_xt.size(1), -1)

            h = torch.cat([h_xy, h_yt, h_xt], dim=-1)
            h = output_attn(h)

            h_xy = h[:, :, 0:res1 * res2].view(h.size(0), h.size(1), res1, res2)
            h_yt = h[:, :, res1 * res2:res2 * (res1 + t)].view(h.size(0), h.size(1), t, res2)
            h_xt = h[:, :, res2 * (res1 + t):res2 * (res1 + t) + res1 * t].view(h.size(0), h.size(1), t, res1)

        h_xy = self.out(h_xy)
        h_yt = self.out(h_yt)
        h_xt = self.out(h_xt)

        h_xy = h_xy.view(h_xy.size(0), h_xy.size(1), -1)
        h_yt = h_yt.view(h_yt.size(0), h_yt.size(1), -1)
        h_xt = h_xt.view(h_xt.size(0), h_xt.size(1), -1)

        h = torch.cat([h_xy, h_yt, h_xt], dim=-1)
        h = h.type(x.dtype)

        return h