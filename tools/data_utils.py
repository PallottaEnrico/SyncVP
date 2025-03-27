import random
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch.cuda.amp import autocast
from torch.utils.data.sampler import Sampler


def resize_crop(video, resolution):
    _, _, h, w = video.shape

    if h > w:
        half = (h - w) // 2
        cropsize = (0, half, w, half + w)  # left, upper, right, lower
    elif w >= h:
        half = (w - h) // 2
        cropsize = (half, 0, half + h, h)

    video = video[:, :, cropsize[1]:cropsize[3], cropsize[0]:cropsize[2]]
    video = F.interpolate(video, size=resolution, mode='bilinear', align_corners=False)

    video = video.permute(1, 0, 2, 3).contiguous()  # [t, c, h, w]
    return video


class InfiniteSampler(Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


def prepare_input(it, x, diffusion_model, device, autoencoder_model, args, autoencoder_cond_model=None, p=None):
    x = x.to(device)
    x = rearrange(x / 127.5 - 1, 'b t c h w -> b c t h w')  # videos

    diffusion_model.zero_grad()

    if p is None:
        p = np.random.random()

    if p < args.cond_prob:
        c, x = x[:, :, :args.cond_frames], x[:, :, args.cond_frames:]
        mask = (c + 1).contiguous().view(c.size(0), -1) ** 2
        mask = torch.where(mask.sum(dim=-1) > 0, 1, 0).view(-1, 1, 1)

        with autocast():
            with torch.no_grad():
                z = autoencoder_model.module.extract(x).detach()
                if autoencoder_cond_model is not None:
                    c = autoencoder_cond_model.module.extract(c).detach()
                else:
                    c = autoencoder_model.module.extract(c).detach()
                c = c * mask + torch.zeros_like(c).to(c.device) * (1 - mask)
    else:
        c, x_tmp = x[:, :, :args.cond_frames], x[:, :, args.cond_frames:]
        mask = (c + 1).contiguous().view(c.size(0), -1) ** 2
        mask = torch.where(mask.sum(dim=-1) > 0, 1, 0).view(-1, 1, 1, 1, 1)

        clip_length = x_tmp.size(2)
        prefix = random.randint(0, args.cond_frames)
        x = x[:, :, prefix:prefix + clip_length, :, :] * mask + x_tmp * (1 - mask)
        with autocast():
            with torch.no_grad():
                z = autoencoder_model.module.extract(x).detach()
                c = torch.zeros_like(z).to(device)

    return z, c, p
