import sys;

sys.path.extend(['.', 'src'])
import numpy as np
import torch
from tools.utils import AverageMeter
from einops import rearrange
from losses.ddpm import DDPM, MMDDPM

from evals.fvd.fvd import get_fvd_logits, frechet_distance
from evals.fvd.download import load_i3d_pretrained
from evals.ssim.ssim import calculate_ssim, SSIM
import lpips
import os

import torchvision
import PIL


def log_videos(video_grid, gts, predictions, it, logger=None, modality=None):
    if not modality:
        modality = logger.logdir.split('_')[-2].lower() # rgb / depth
    gt_video = save_image_grid(gts.cpu().numpy(), os.path.join(logger.logdir, f'ground-truth_{modality}_{it}.gif'),
                               drange=[0, 255], grid_size=video_grid)
    gt_video = np.expand_dims(gt_video, 0).transpose(0, 1, 4, 2, 3)
    pred_video = save_image_grid(predictions.cpu().numpy(),
                                 os.path.join(logger.logdir, f'predicted_{modality}_{it}.gif'),
                                 drange=[0, 255], grid_size=video_grid)
    pred_video = np.expand_dims(pred_video, 0).transpose(0, 1, 4, 2, 3)

    logger.video_summary(f'ground-truth_{modality}', gt_video, it)
    logger.video_summary(f'predicted_{modality}', pred_video, it)


def save_image_grid(img, fname, drange, grid_size, normalize=True):
    if normalize:
        lo, hi = drange
        img = np.asarray(img, dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, T, H, W = img.shape
    img = img[:gw * gh]
    img = img.reshape(gh, gw, C, T, H, W)
    img = img.transpose(3, 0, 4, 1, 5, 2)
    img = img.reshape(T, gh * H, gw * W, C)

    print(f'Saving Video with {T} frames, img shape {H}, {W}')
    frame_duration = 100  # 100 ms per frame
    if C == 1:
        img = np.repeat(img, 3, axis=3)

    torchvision.io.write_video(f'{fname[:-3]}mp4', torch.from_numpy(img), fps=16)
    imgs = [PIL.Image.fromarray(img[i], 'RGB') for i in range(len(img))]
    imgs[0].save(fname, quality=95, save_all=True, append_images=imgs[1:], duration=T * frame_duration, loop=0)

    return img


def lpips_video(pred_videos, real_videos, lpips_model):
    """
    :param pred_videos: [B, T, H, W, C]
    :param real_videos: [B, T, C, H, W]
    :param lpips_model: pretrained lpips model
    :return:
    """
    lpips_loss = []

    pred_videos = rearrange(pred_videos, 'b t h w c -> b t c h w')

    # normalize [0, 255] to [-1, 1]
    pred_videos = pred_videos / 127.5 - 1
    real_videos = real_videos / 127.5 - 1

    for pred, real in zip(pred_videos, real_videos):
        lpips_loss.append(lpips_model(pred, real).mean().item())

    return lpips_loss


def compute_psnr(gt, pred, batch_size):
    gt = gt.view(batch_size, -1)
    pred = pred.view(batch_size, -1)

    mse = ((gt * 0.5 - pred * 0.5) ** 2).mean(dim=-1)
    psnr = (-10 * torch.log10(mse)).mean()

    return psnr


def l2_depth(pred_depth, real_depth):
    """
    Compute L2 distance between predicted and real depth maps.
    :param pred_depth: [B, T, H, W, C] Tensor
    :param real_depth: [B, T, H, W, C] Tensor
    :return: [B] numpy array
    """
    # Min max normalization
    pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
    real_depth = (real_depth - real_depth.min()) / (real_depth.max() - real_depth.min())

    # Flatten tensors keeping B dimension
    pred_depth = pred_depth.contiguous().view(pred_depth.size(0), -1)
    real_depth = real_depth.contiguous().view(real_depth.size(0), -1)

    l2 = (pred_depth - real_depth).pow(2).mean(dim=-1).sqrt()

    return l2.cpu().numpy()

def eval_autoencoder(rank, model, loader, it, samples, logger=None):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['fvd'] = AverageMeter()
    losses['ssim'] = AverageMeter()
    losses['lpips'] = AverageMeter()
    losses['psnr'] = AverageMeter()

    ssim = SSIM()
    lpips_model = lpips.LPIPS(net='alex')

    gt_embeddings = []
    rec_embeddings = []

    reconstructions = []
    gts = []

    model.eval()
    i3d = load_i3d_pretrained(device)

    save_samples = 1
    # save as many images(video gif) as possible but not more than 100 (10x10 matrix)
    for i in range(2, 11):
        save_samples = i * i if samples >= i * i else save_samples
    video_grid = (int(np.sqrt(save_samples)), int(np.sqrt(save_samples)))

    with torch.no_grad():
        for n, (gt, _) in enumerate(loader):
            k = gt.size(0)  # batch size
            if n >= samples // k:
                break
            gt = gt.to(device)
            rec, _ = model(rearrange(gt / 127.5 - 1, 'b t c h w -> b c t h w'))

            losses['psnr'].update(compute_psnr(gt / 127.5 - 1, rec, k))

            gt = rearrange(gt, 'b t c h w -> b t h w c')
            rec = rearrange((rec.clamp(-1, 1) + 1) * 127.5, '(b t) c h w -> b t h w c', b=gt.size(0))

            gt = gt.type(torch.uint8).cpu()
            rec = rec.type(torch.uint8).cpu()

            losses['ssim'].update(calculate_ssim(rearrange(rec, 'b t h w c -> b t c h w'),
                                                 rearrange(gt, 'b t h w c -> b t c h w'),
                                                 ssim).mean())
            losses['lpips'].update(np.average(lpips_video(rec, rearrange(gt, 'b t h w c -> b t c h w'), lpips_model)))

            gt_embeddings.append(get_fvd_logits(gt.numpy(), i3d=i3d, device=device))
            rec_embeddings.append(get_fvd_logits(rec.numpy(), i3d=i3d, device=device))

            if len(reconstructions) < save_samples:
                gts.append(rearrange(gt, 'b t h w c -> b c t h w'))
                reconstructions.append(rearrange(rec, 'b t h w c -> b c t h w'))

    gts = torch.cat(gts)
    reconstructions = torch.cat(reconstructions)

    if rank == 0:
        log_videos(video_grid, gts, reconstructions, it, logger)

    gt_embeddings = torch.cat(gt_embeddings)
    rec_embeddings = torch.cat(rec_embeddings)

    fvd = frechet_distance(rec_embeddings.clone().detach(), gt_embeddings.clone().detach())

    # reset model to train mode
    model.train()

    return fvd.item(), losses['ssim'].average, losses['lpips'].average, losses['psnr'].average


def eval_diffusion(rank, ema_model, ae, ae_cond, loader, it, samples=16, logger=None, frames=16, cond_frames=8,
                   trajectories=1, w=0.):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['ssim'] = AverageMeter()
    losses['lpips'] = AverageMeter()

    ssim = SSIM()

    lpips_model = lpips.LPIPS(net='alex')

    diffusion_model = DDPM(ema_model,
                           channels=ema_model.module.diffusion_model.in_channels,
                           image_size=ema_model.module.diffusion_model.image_size,
                           sampling_timesteps=100,
                           w=w).to(device)

    gt_embeddings = []
    pred_embeddings = []

    gts = []
    predictions = []

    i3d = load_i3d_pretrained(device)

    save_samples = 1
    # save as many images(video gif) as possible but not more than 100 (10x10 matrix)
    for i in range(2, 11):
        save_samples = i * i if samples >= i * i else save_samples
    video_grid = (int(np.sqrt(save_samples)), int(np.sqrt(save_samples)))

    with torch.no_grad():
        for n, (x, _) in enumerate(loader):
            k = x.size(0)  # batch size
            if n >= samples // k:  # useful to break earlier in case num samples is not the full dataset
                break

            if cond_frames > 0:
                c_init = x[:, :cond_frames]
                c_init = ae_cond.module.extract(
                    rearrange(c_init / 127.5 - 1, 'b t c h w -> b c t h w').to(device).detach())
            else:
                c_init = None

            _ssim = []
            _lpips = []

            for i in range(trajectories):
                z = diffusion_model.sample(batch_size=k, cond=c_init)
                pred = ae.module.decode_from_sample(z).clamp(-1, 1).cpu()
                pred = (1 + rearrange(pred, '(b t) c h w -> b t h w c', b=k)) * 127.5
                pred = pred.type(torch.uint8)
                # Autoregressive prediction of new frames until we reach the desired number of frames
                while pred.size(1) < frames - cond_frames:
                    c = pred[:, -cond_frames:]  # last n frames
                    c = ae_cond.module.extract(rearrange(c / 127.5 - 1, 'b t h w c-> b c t h w').to(device).detach())
                    z = diffusion_model.sample(batch_size=k, cond=c)
                    new_pred = ae.module.decode_from_sample(z).clamp(-1, 1).cpu()
                    new_pred = (1 + rearrange(new_pred, '(b t) c h w -> b t h w c', b=k)) * 127.5
                    new_pred = new_pred.type(torch.uint8)
                    pred = torch.cat([pred, new_pred], dim=1)
                # discard eventual extra frames
                if pred.size(1) > frames - cond_frames:
                    pred = pred[:, :frames - cond_frames]

                # real = rearrange(real, 'b t c h w -> b t h w c')
                real = rearrange(x, 'b t c h w -> b t h w c')
                real = real.type(torch.uint8)

                cond = x[:, :cond_frames]

                # Following previous work, SSIM and LPIPS are computed only on the predicted frames
                if cond_frames > 0:
                    _ssim.append(calculate_ssim(rearrange(pred, 'b t h w c -> b t c h w'), x[:, cond_frames:], ssim))
                _lpips.append(lpips_video(pred, x[:, cond_frames:], lpips_model))

                # FVD is instead computed on the whole video (gt condition + predicted frames)
                pred = torch.cat([rearrange(cond.type(torch.uint8), 'b t c h w -> b t h w c'), pred], dim=1)
                if i == 0:
                    gt_embeddings.append(get_fvd_logits(real.numpy(), i3d=i3d, device=device))
                pred_embeddings.append(get_fvd_logits(pred.cpu().numpy(), i3d=i3d, device=device))

                if i == 0 and len(predictions) < save_samples:  # store some samples for visualization
                    gts.append(rearrange(x.type(torch.uint8), 'b t c h w -> b c t h w'))
                    predictions.append(rearrange(pred, 'b t h w c -> b c t h w'))

            # for each video take the best prediction among the different generated trajectories
            # if needed for legacy support (numpy<1.26.4)
            ssim_best = np.max(_ssim, axis=0) if len(_ssim) > 1 else np.array(_ssim[0]) if len(_ssim) > 0 else np.array(
                [-1])
            lpips_best = np.min(_lpips, axis=0) if len(_lpips) > 1 else np.array(_lpips[0])
            losses['ssim'].update(ssim_best.mean())
            losses['lpips'].update(lpips_best.mean())

    gts = torch.cat(gts)
    predictions = torch.cat(predictions)

    if rank == 0:
        log_videos(video_grid, gts, predictions, it, logger)

    gt_embeddings = torch.cat(gt_embeddings)
    pred_embeddings = torch.cat(pred_embeddings)
    fvd = frechet_distance(pred_embeddings.clone().detach(), gt_embeddings.clone().detach())

    return fvd, losses['ssim'].average, losses['lpips'].average


def eval_multimodal_diffusion(rank, ema_model, ae_rgb, ae_depth, ae_cond_rgb, ae_cond_depth, loader, it, samples=16, logger=None,
                              frames=16, cond_frames=8, trajectories=1, w=0., no_depth_cond=False, same_noise=False):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['ssim'] = AverageMeter()
    losses['ssim-depth'] = AverageMeter()
    losses['lpips'] = AverageMeter()
    losses['lpips-depth'] = AverageMeter()
    losses['l2'] = AverageMeter()

    ssim_object = SSIM()
    lpips_model = lpips.LPIPS(net='alex')

    diffusion_model = MMDDPM(ema_model,
                             channels=ema_model.module.diffusion_model.diff_rgb.diffusion_model.in_channels,
                             image_size=ema_model.module.diffusion_model.diff_rgb.diffusion_model.image_size,
                             sampling_timesteps=100,
                             w=w,
                             **{'same_noise': same_noise}).to(device)

    gt_embeddings_rgb = []
    pred_embeddings_rgb = []
    gt_embeddings_depth = []
    pred_embeddings_depth = []

    gts_rgb = []
    predictions_rgb = []
    gts_depth = []
    predictions_depth = []

    i3d = load_i3d_pretrained(device)

    save_samples = 1
    # save as many images(video gif) as possible but not more than 100 (10x10 matrix)
    for i in range(2, 10):
        save_samples = i * i if samples >= i * i else save_samples
    video_grid = (int(np.sqrt(save_samples)), int(np.sqrt(save_samples)))

    with torch.no_grad():
        for n, (x_rgb, x_depth, _) in enumerate(loader):
            k = x_rgb.size(0)
            if n >= samples // k:
                break
            c_rgb_init = x_rgb[:, :cond_frames]
            c_depth_init = x_depth[:, :cond_frames]

            if cond_frames > 0:
                c_rgb_init = ae_cond_rgb.module.extract(
                    rearrange(c_rgb_init / 127.5 - 1, 'b t c h w -> b c t h w').to(device).detach())
                if not no_depth_cond:
                    c_depth_init = ae_cond_depth.module.extract(
                        rearrange(c_depth_init / 127.5 - 1, 'b t c h w -> b c t h w').to(device).detach())
                else:
                    c_depth_init = torch.zeros_like(c_rgb_init)
            else:
                c_rgb_init = None
                c_depth_init = None

            _ssim = []
            _ssim_depth = []
            _lpips = []
            _lpips_depth = []
            _l2 = []

            for i in range(trajectories):
                z_rgb, z_depth = diffusion_model.sample(batch_size=k, cond_rgb=c_rgb_init, cond_depth=c_depth_init)

                pred_rgb = ae_rgb.module.decode_from_sample(z_rgb).clamp(-1, 1).cpu()
                pred_depth = ae_depth.module.decode_from_sample(z_depth).clamp(-1, 1).cpu()

                pred_rgb = (1 + rearrange(pred_rgb, '(b t) c h w -> b t h w c', b=k)) * 127.5
                pred_rgb = pred_rgb.type(torch.uint8)
                pred_depth = (1 + rearrange(pred_depth, '(b t) c h w -> b t h w c', b=k)) * 127.5
                pred_depth = pred_depth.type(torch.uint8)

                # Autoregressive prediction of new frames until we reach the desired number of frames
                while pred_rgb.size(1) < frames - cond_frames:
                    c_rgb = pred_rgb[:, -cond_frames:]
                    c_depth = pred_depth[:, -cond_frames:]
                    c_rgb = ae_cond_rgb.module.extract(
                        rearrange(c_rgb / 127.5 - 1, 'b t h w c-> b c t h w').to(device).detach())
                    c_depth = ae_cond_depth.module.extract(
                        rearrange(c_depth / 127.5 - 1, 'b t h w c-> b c t h w').to(device).detach())

                    z_rgb, z_depth = diffusion_model.sample(batch_size=k, cond_rgb=c_rgb, cond_depth=c_depth)

                    new_pred_rgb = ae_rgb.module.decode_from_sample(z_rgb).clamp(-1, 1).cpu()
                    new_pred_depth = ae_depth.module.decode_from_sample(z_depth).clamp(-1, 1).cpu()
                    new_pred_rgb = (1 + rearrange(new_pred_rgb, '(b t) c h w -> b t h w c', b=k)) * 127.5
                    new_pred_rgb = new_pred_rgb.type(torch.uint8)
                    new_pred_depth = (1 + rearrange(new_pred_depth, '(b t) c h w -> b t h w c', b=k)) * 127.5
                    new_pred_depth = new_pred_depth.type(torch.uint8)

                    pred_rgb = torch.cat([pred_rgb, new_pred_rgb], dim=1)
                    pred_depth = torch.cat([pred_depth, new_pred_depth], dim=1)
                # Discard eventual extra frames
                if pred_rgb.size(1) > frames - cond_frames:
                    pred_rgb = pred_rgb[:, :frames - cond_frames]
                    pred_depth = pred_depth[:, :frames - cond_frames]

                gt_rgb = rearrange(x_rgb[:k], 'b t c h w -> b t h w c')
                gt_rgb = gt_rgb.type(torch.uint8)
                gt_depth = rearrange(x_depth[:k], 'b t c h w -> b t h w c')
                gt_depth = gt_depth.type(torch.uint8)

                # Following previous work, SSIM and LPIPS are computed only on the predicted frames
                if cond_frames > 0:
                    _ssim.append(calculate_ssim(rearrange(pred_rgb, 'b t h w c -> b t c h w'), x_rgb[:, cond_frames:],
                                                ssim_object))
                    _ssim_depth.append(
                        calculate_ssim(rearrange(pred_depth, 'b t h w c -> b t c h w'), x_depth[:, cond_frames:],
                                       ssim_object))

                _lpips.append(lpips_video(pred_rgb, x_rgb[:, cond_frames:], lpips_model))
                _lpips_depth.append(lpips_video(pred_depth, x_depth[:, cond_frames:], lpips_model))

                # Compute L2 distance between predicted and real depth maps
                _l2.append(l2_depth(pred_depth, gt_depth[:, cond_frames:]))

                # Concat cond_frames with predicted frames for FVD computation
                pred_rgb = torch.cat(
                    [rearrange(x_rgb[:k, :cond_frames].type(torch.uint8), 'b t c h w -> b t h w c'), pred_rgb],
                    dim=1)
                pred_depth = torch.cat(
                    [rearrange(x_depth[:k, :cond_frames].type(torch.uint8), 'b t c h w -> b t h w c'), pred_depth],
                    dim=1)

                if i == 0:
                    gt_embeddings_rgb.append(get_fvd_logits(gt_rgb.numpy(), i3d=i3d, device=device))
                    gt_embeddings_depth.append(get_fvd_logits(gt_depth.numpy(), i3d=i3d, device=device))

                pred_embeddings_rgb.append(get_fvd_logits(pred_rgb.numpy(), i3d=i3d, device=device))
                pred_embeddings_depth.append(get_fvd_logits(pred_depth.numpy(), i3d=i3d, device=device))

                if len(predictions_rgb) < save_samples and i == 0:  # store some samples for visualization
                    gts_rgb.append(rearrange(x_rgb[:k].type(torch.uint8), 'b t c h w -> b c t h w'))
                    predictions_rgb.append(rearrange(pred_rgb, 'b t h w c -> b c t h w'))
                    gts_depth.append(rearrange(x_depth[:k].type(torch.uint8), 'b t c h w -> b c t h w'))
                    predictions_depth.append(rearrange(pred_depth, 'b t h w c -> b c t h w'))

            ssim_best = np.max(_ssim, axis=0) if cond_frames > 0 else np.array([-1])
            ssim_depth_best = np.max(_ssim_depth, axis=0) if cond_frames > 0 else np.array([-1])
            lpips_best = np.min(_lpips, axis=0)
            lpips_depth_best = np.min(_lpips_depth, axis=0)
            l2_best = np.min(_l2, axis=0)

            losses['ssim'].update(ssim_best.mean())
            losses['ssim-depth'].update(ssim_depth_best.mean())
            losses['lpips'].update(lpips_best.mean())
            losses['lpips-depth'].update(lpips_depth_best.mean())
            losses['l2'].update(l2_best.mean())

    gts_rgb = torch.cat(gts_rgb)
    gts_depth = torch.cat(gts_depth)
    predictions_rgb = torch.cat(predictions_rgb)
    predictions_depth = torch.cat(predictions_depth)

    gt_embeddings_rgb = torch.cat(gt_embeddings_rgb)
    gt_embeddings_depth = torch.cat(gt_embeddings_depth)
    if rank == 0:
        log_videos(video_grid, gts_rgb, predictions_rgb, it, logger, modality='rgb')
        log_videos(video_grid, gts_depth, predictions_depth, it, logger, modality='depth')

    pred_embeddings_rgb = torch.cat(pred_embeddings_rgb)
    pred_embeddings_depth = torch.cat(pred_embeddings_depth)

    fvd_rgb_pred = frechet_distance(pred_embeddings_rgb.clone().detach(), gt_embeddings_rgb.clone().detach())
    fvd_depth_pred = frechet_distance(pred_embeddings_depth.clone().detach(), gt_embeddings_depth.clone().detach())

    return (fvd_rgb_pred.item(), fvd_depth_pred.item(), losses['ssim'].average, losses['ssim-depth'].average,
            losses['lpips'].average, losses['lpips-depth'].average, losses['l2'].average)
