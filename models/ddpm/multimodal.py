import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from models.ddpm.modules import SingleVideoDualCrossAttentionBlock, NoCross
from models.ddpm.unet import UNetModel, DiffusionWrapper
from models.ddpm.utils import timestep_embedding

class MMDiffusionWrapper(Module):
    def __init__(self, model, conditioning_key=None):
        super().__init__()
        self.diffusion_model = model
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x_rgb, x_depth, cond_rgb, cond_depth, t):
        if self.conditioning_key is None:
            out = self.diffusion_model(x_rgb, x_depth, cond_rgb, cond_depth, t)
        else:
            raise NotImplementedError()

        return out



class SingleModalityUnet(UNetModel):
    def __init__(self, unet_config, frames):
        super().__init__(**unet_config, frames=frames)

    def forward(self, x, cond=None, timesteps=None, context=None, y=None, **kwargs):
        assert (y is not None) == (self.num_classes is not None),\
            "must specify y if and only if the model is class-conditional"

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
        elif self.cond_model:
            h = torch.cat([h, self.zeros.repeat(h.size(0), 1, 1)], dim=1)

        # define res1, res2 and t to split latent vector h into 3 parts
        if isinstance(self.image_size, int):
            res1 = self.image_size
            res2 = self.image_size
        else:
            res1 = self.image_size[0]
            res2 = self.image_size[1]
        t = self.frames

        h_xy = h[:, :, 0:res1 * res2].view(h.size(0), h.size(1), res1, res2)
        h_yt = h[:, :, res1 * res2:res2 * (res1 + t)].view(h.size(0), h.size(1), t, res2)
        h_xt = h[:, :, res2 * (res1 + t):res2 * (res1 + t) + res1 * t].view(h.size(0), h.size(1), t, res1)

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

            h = yield h

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

        h = yield h

        h_xy = h[:, :, 0:res1 * res2].view(h.size(0), h.size(1), res1, res2)
        h_yt = h[:, :, res1 * res2:res2 * (res1 + t)].view(h.size(0), h.size(1), t, res2)
        h_xt = h[:, :, res2 * (res1 + t):res2 * (res1 + t) + res1 * t].view(h.size(0), h.size(1), t, res1)

        for module, output_attn in zip(self.output_blocks, self.output_attns):
            h_xy = torch.cat([h_xy, h_xys.pop()], dim=1)
            h_xy = module(h_xy, emb, context)
            h_yt = torch.cat([h_yt, h_yts.pop()], dim=1)
            h_yt = module(h_yt, emb, context)
            h_xt = torch.cat([h_xt, h_xts.pop()], dim=1)
            h_xt = module(h_xt, emb, context)

            res1 = h_xy.size(-2)
            res2 = h_xy.size(-1)
            t   = h_xt.size(-2)

            h_xy = h_xy.view(h_xy.size(0), h_xy.size(1), -1)
            h_yt = h_yt.view(h_yt.size(0), h_yt.size(1), -1)
            h_xt = h_xt.view(h_xt.size(0), h_xt.size(1), -1)

            h = torch.cat([h_xy, h_yt, h_xt], dim=-1)
            h = output_attn(h)

            h = yield h

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

        yield h


class MultiModalUnet(Module):
    def __init__(self, unet_config1, unet_config2, frames, cross_attn_configs, shared=False):
        super().__init__()

        self.diff_rgb = DiffusionWrapper(SingleModalityUnet(unet_config1, frames=frames))
        if shared:
            self.diff_depth = self.diff_rgb
        else:
            self.diff_depth = DiffusionWrapper(SingleModalityUnet(unet_config2, frames=frames))

        cross_attn_block = SingleVideoDualCrossAttentionBlock # customize this if needed

        if cross_attn_configs['cross_attn'] == 'deep':
            down_attn_class = NoCross
            mid_attn_class = cross_attn_block
            up_attn_class = NoCross
        elif cross_attn_configs['cross_attn'] == 'all':
            down_attn_class = cross_attn_block
            mid_attn_class = cross_attn_block
            up_attn_class = cross_attn_block
        elif cross_attn_configs['cross_attn'] == 'none':
            down_attn_class = NoCross
            mid_attn_class = NoCross
            up_attn_class = NoCross
        else:
            raise ValueError(f"Unknown cross attention configuration: {cross_attn_configs['cross_attn']}")

        # Create a num channels list copying them from the self.diff_rgb.diffusion_model attention blocks
        self.cross_attns = ModuleList([])
        model_channels = unet_config1['model_channels']
        ch = model_channels

        ca_levels = [len(unet_config1['channel_mult']) - i for i in range(1, 3)] if len(
            unet_config1['channel_mult']) > 2 else [1]
        # down path
        for level, mult in enumerate(unet_config1['channel_mult']):
            for _ in range(unet_config1['num_res_blocks']):
                ch = model_channels * mult
                if level in ca_levels and cross_attn_configs['cross_attn'] == 'deep':
                    self.cross_attns.append(mid_attn_class(ch, **cross_attn_configs))
                else:
                    self.cross_attns.append(down_attn_class(ch, **cross_attn_configs))
            if level != len(unet_config1['channel_mult']) - 1:
                self.cross_attns.append(down_attn_class(ch, **cross_attn_configs))

        # mid layer
        self.cross_attns.append(mid_attn_class(ch, **cross_attn_configs))

        # up path
        ca_levels = [i for i in range(2)] if len(unet_config1['channel_mult']) > 2 else [0]
        for level, mult in enumerate(unet_config1['channel_mult'][::-1]):
            for _ in range(unet_config1['num_res_blocks'] + 1):
                ch = model_channels * mult
                if level in ca_levels and cross_attn_configs['cross_attn'] == 'deep':
                    self.cross_attns.append(mid_attn_class(ch, **cross_attn_configs))
                else:
                    self.cross_attns.append(up_attn_class(ch, **cross_attn_configs))

    def load_single_modality_models(self, model_rgb: str, model_depth: str):
        diff_rgb_ckpt = torch.load(model_rgb)
        diff_depth_ckpt = torch.load(model_depth)
        self.diff_rgb.load_state_dict(diff_rgb_ckpt, strict=False)
        self.diff_depth.load_state_dict(diff_depth_ckpt, strict=False)

        del diff_rgb_ckpt, diff_depth_ckpt

    def forward(self, x_rgb, x_depth, cond_rgb=None, cond_depth=None, t=None):
        gen_rgb = self.diff_rgb(x_rgb, cond_rgb, t)
        gen_depth = self.diff_depth(x_depth, cond_depth, t)

        # Apply cross attention only at second call
        h_rgb = next(gen_rgb)
        h_depth = next(gen_depth)

        for cross_attn in self.cross_attns:
            # YIELDS give us intermediate activations on which we apply cross attention
            h_rgb = gen_rgb.send(h_rgb)
            h_depth = gen_depth.send(h_depth)

            h_rgb, h_depth = cross_attn(h_rgb, h_depth)

        h_rgb, h_depth = gen_rgb.send(h_rgb), gen_depth.send(h_depth)

        return h_rgb, h_depth
