import copy

from models.autoencoder.autoencoder_vit import ViTAutoencoder
from models.ddpm.unet import DiffusionWrapper, UNetModel
from models.ddpm.multimodal import MultiModalUnet, MMDiffusionWrapper
from tools.dataloader import get_loaders
from tools.utils import Logger, file_name
import torch
from evals.eval import eval_autoencoder, eval_diffusion, eval_multimodal_diffusion


def setup(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fn = "eval_" + file_name(args)
    logger = Logger(fn, path=args.output, resume=args.resume)
    logger.log(args)
    logger.log(f'Log path: {logger.logdir}')

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    log_(f"Loading dataset {args.data} with resolution {args.res}")

    args.frames = args.future_frames
    train_loader, _, test_loader = get_loaders(0, copy.deepcopy(args))

    args.frames += args.cond_frames

    if args.train:
        return log_, logger, train_loader, device
    return log_, logger, test_loader, device


def autoencoder_eval(args):
    log_, logger, test_loader, device = setup(args)
    log_(f"Loading autoencoder model")
    model = ViTAutoencoder(args.embed_dim, args.ddconfig)
    model.load_state_dict(torch.load(args.ae_model))
    model = model.to(device)

    if args.train:
        samples = args.samples if args.samples > 0 else 256  # Train might be too big
    else:
        samples = args.samples if args.samples > 0 else len(test_loader) * args.batch_size

    log_(f"Evaluation on {samples} samples")
    fvd, ssim, lpips, psnr = eval_autoencoder(0, model, test_loader, 0, samples, logger)
    log_(f"FVD: {fvd}, SSIM: {ssim}, LPIPS: {lpips * 1000}, PSNR: {psnr}")


class DummyWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


def diffusion_eval(args):
    log_, logger, test_loader, device = setup(args)

    log_(f"Loading autoencoder model")
    autoencoder_model = ViTAutoencoder(args.embed_dim, args.ddconfig)
    autoencoder_model.load_state_dict(torch.load(args.ae_model))
    autoencoder_model = DummyWrapper(autoencoder_model).to(device)
    autoencoder_model.eval()

    if args.ae_cond_model != '':
        autoencoder_cond_model = ViTAutoencoder(args.embed_dim, args.ae_cond_ddconfig)
        autoencoder_cond_model.load_state_dict(torch.load(args.ae_cond_model))
        autoencoder_cond_model = DummyWrapper(autoencoder_cond_model).to(device)
        autoencoder_cond_model.eval()
    else:
        autoencoder_cond_model = autoencoder_model

    log_(f"Loading diffusion model")
    unet = UNetModel(**args.unetconfig, frames=args.ddconfig.frames)
    diffusion_model = DiffusionWrapper(unet)
    diffusion_model.load_state_dict(torch.load(args.diffusion_model))
    diffusion_model = DummyWrapper(diffusion_model).to(device)
    diffusion_model.eval()

    # Count diffusion model parameters
    log_(f"Diffusion model has {sum(p.numel() for p in diffusion_model.parameters())} parameters")

    if args.train:
        samples = args.samples if args.samples > 0 else 256
    else:
        samples = args.samples if args.samples > 0 else len(test_loader) * args.batch_size

    trajectories = args.traj if args.traj > 0 else 1
    log_(f"Evaluation on {samples} samples with {trajectories} generated samples")

    fvd, ssim, lpips = eval_diffusion(0, diffusion_model, autoencoder_model, autoencoder_cond_model, test_loader, 0,
                                      samples, logger, args.frames, args.cond_frames, trajectories, args.ddpmconfig.w)
    log_(f"FVD: {fvd}, SSIM: {ssim}, LPIPS: {lpips * 1000}")


def multimodal_diffusion_eval(args):
    log_, logger, test_loader, device = setup(args)

    log_(f"Loading autoencoder models")
    autoencoder_model_rgb = ViTAutoencoder(args.embed_dim, args.ddconfig).to(device)
    autoencoder_model_depth = ViTAutoencoder(args.embed_dim, args.ddconfig).to(device)

    autoencoder_model_rgb.load_state_dict(torch.load(args.ae_model))
    autoencoder_model_depth.load_state_dict(torch.load(args.ae_model_depth))
    autoencoder_model_rgb = DummyWrapper(autoencoder_model_rgb).to(device)
    autoencoder_model_depth = DummyWrapper(autoencoder_model_depth).to(device)
    autoencoder_model_rgb.eval()
    autoencoder_model_depth.eval()

    if args.ae_cond_model != '':
        autoencoder_cond_model_rgb = ViTAutoencoder(args.embed_dim, args.ae_cond_ddconfig).to(device)
        autoencoder_cond_model_depth = ViTAutoencoder(args.embed_dim, args.ae_cond_ddconfig).to(device)
        autoencoder_cond_model_rgb.load_state_dict(torch.load(args.ae_cond_model))
        autoencoder_cond_model_depth.load_state_dict(torch.load(args.ae_cond_model_depth))
        autoencoder_cond_model_rgb = DummyWrapper(autoencoder_cond_model_rgb).to(device)
        autoencoder_cond_model_depth = DummyWrapper(autoencoder_cond_model_depth).to(device)
        autoencoder_cond_model_rgb.eval()
        autoencoder_cond_model_depth.eval()
    else:
        autoencoder_cond_model_rgb = autoencoder_model_rgb
        autoencoder_cond_model_depth = autoencoder_model_depth

    log_(f"Loading multi modal diffusion model")
    unet = MultiModalUnet(args.unetconfig, args.unetconfig, args.ddconfig.frames, args.cross_attn_configs,
                          args.shared)
    diffusion_model = MMDiffusionWrapper(unet).to(device)
    diffusion_model.load_state_dict(torch.load(args.diffusion_model))
    diffusion_model = DummyWrapper(diffusion_model).to(device)
    diffusion_model.eval()

    # Count diffusion model parameters
    log_(f"Diffusion model has {sum(p.numel() for p in diffusion_model.parameters())} parameters")

    if args.train:
        samples = args.samples if args.samples > 0 else 256
    else:
        samples = args.samples if args.samples > 0 else len(test_loader) * args.batch_size

    trajectories = args.traj if args.traj > 0 else 1

    log_(f"Evaluation on {samples} samples with {trajectories} generated samples")

    if args.no_depth_cond:
        log_(f"!!! Depth conditioning disabled !!!")

    fvd_rgb, fvd_depth, ssim_rgb, ssim_depth, lpips_rgb, lpips_depth, l2 = eval_multimodal_diffusion(
        0,
        ema_model=diffusion_model,
        ae_rgb=autoencoder_model_rgb,
        ae_depth=autoencoder_model_depth,
        ae_cond_rgb=autoencoder_cond_model_rgb,
        ae_cond_depth=autoencoder_cond_model_depth,
        loader=test_loader,
        it=0,
        samples=samples,
        logger=logger,
        frames=args.frames,
        cond_frames=args.cond_frames,
        trajectories=trajectories,
        w=args.ddpmconfig.w,
        no_depth_cond=args.no_depth_cond,
        same_noise=args.same_noise
    )

    log_(f"RGB - FVD Avg: {fvd_rgb}, SSIM: {ssim_rgb}, LPIPS: {lpips_rgb * 1000}")
    log_(f"Depth - FVD Avg: {fvd_depth}, SSIM: {ssim_depth}, LPIPS: {lpips_depth * 1000}, L2: {l2 * 100}")
