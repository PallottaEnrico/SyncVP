import os
from glob import glob
import time
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from tools.dataloader import get_loaders
from models.autoencoder.autoencoder_vit import ViTAutoencoder
from losses.perceptual import LPIPSWithDiscriminator
from torch.cuda.amp import GradScaler, autocast
from tools.utils import AverageMeter, setup_distibuted_training, setup_logger
from evals.eval import eval_autoencoder
from einops import rearrange
import copy

# ----------------------------------------------------------------------------

_num_moments = 3  # [num_scalars, sum_of_scalars, sum_of_squares]
_reduce_dtype = torch.float32  # Data type to use for initial per-tensor reduction.
_counter_dtype = torch.float64  # Data type to use for the internal counters.
_rank = 0  # Rank of the current process.
_sync_device = None  # Device to use for multiprocess communication. None = single-process.
_sync_called = False  # Has _sync() been called yet?
_counters = dict()  # Running counters on each device, updated by report(): name => device => torch.Tensor
_cumulative = dict()  # Cumulative counters on the CPU, updated by _sync(): name => torch.Tensor


# ----------------------------------------------------------------------------

def init_multiprocessing(rank, sync_device):
    r"""Initializes `torch_utils.training_stats` for collecting statistics
    across multiple processes.
    This function must be called after
    `torch.distributed.init_process_group()` and before `Collector.update()`.
    The call is not necessary if multi-process collection is not needed.
    Args:
        rank:           Rank of the current process.
        sync_device:    PyTorch device to use for inter-process
                        communication, or None to disable multi-process
                        collection. Typically `torch.device('cuda', rank)`.
    """
    global _rank, _sync_device
    assert not _sync_called
    _rank = rank
    _sync_device = sync_device


# ----------------------------------------------------------------------------

def autoencoder_training(rank, args):
    device = torch.device('cuda', rank)

    # Set up distributed training. -----------------------------------------------
    setup_distibuted_training(args, rank)

    sync_device = torch.device('cuda', rank) if args.n_gpus > 1 else None
    init_multiprocessing(rank=rank, sync_device=sync_device)
    torch.cuda.set_device(rank)

    # Set up logger and saving directory.
    log_, logger = setup_logger(args, rank)

    # Dataloaders
    if rank == 0:
        log_(f"Loading dataset {args.data} with resolution {args.res}")

    train_loader, val_loader, _ = get_loaders(rank, copy.deepcopy(args))
    # Model definition
    if rank == 0:
        log_(f"Generating autoencoder model")

    torch.cuda.set_device(rank)
    model = ViTAutoencoder(args.embed_dim, args.ddconfig)
    model = model.to(device)

    criterion = LPIPSWithDiscriminator(disc_start=args.lossconfig.params.disc_start,
                                       timesteps=args.ddconfig.frames,
                                       perceptual_weight=args.perceptual_weight).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.9))

    scheduler = CosineAnnealingLR(opt, T_max=args.max_iter, eta_min=args.lr / 100)

    d_opt = torch.optim.AdamW(
        list(criterion.discriminator_2d.parameters()) + list(criterion.discriminator_3d.parameters()),
        lr=args.lr, betas=(0.5, 0.9))

    last_it = 0

    if args.resume and rank == 0:
        model_ckpt = torch.load(os.path.join(args.ae_folder, 'model_last.pth'))
        model.load_state_dict(model_ckpt)
        if not args.no_sched:
            opt_ckpt = torch.load(os.path.join(args.ae_folder, 'opt.pth'))
            opt.load_state_dict(opt_ckpt)
            del opt_ckpt

        # Extract last it number from the model name considering that the last model is model_last.pth
        all_models = glob(os.path.join(args.ae_folder, 'model_*0.pth'))
        if len(all_models) > 0 and args.lossconfig.params.disc_start > 0 and not args.no_sched:
            last_it = max([int(os.path.basename(x).split('_')[1].split('.')[0]) for x in all_models])
        else:
            last_it = 0
        log_(f"Resuming from iteration {last_it}")

        if os.path.exists(os.path.join(args.ae_folder, 'scheduler.pth')) and not args.no_sched:
            sched_ckpt = torch.load(os.path.join(args.ae_folder, 'scheduler.pth'))
            scheduler.load_state_dict(sched_ckpt)
            del sched_ckpt

        log_(f"Lr: {scheduler.get_last_lr()[0]}")

        del model_ckpt

    if rank == 0:
        torch.save(model.state_dict(), os.path.join(logger.logdir, 'net_init.pth'))

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[device],
                                                      broadcast_buffers=False,
                                                      find_unused_parameters=False)

    criterion = torch.nn.parallel.DistributedDataParallel(criterion,
                                                          device_ids=[device],
                                                          broadcast_buffers=False,
                                                          find_unused_parameters=False)

    losses = dict()
    losses['ae_loss'] = AverageMeter()
    losses['d_loss'] = AverageMeter()
    check = time.time()

    accum_iter = 3
    disc_opt = False

    if args.amp:
        scaler = GradScaler()
        scaler_d = GradScaler()

        if os.path.exists(os.path.join(args.ae_folder, 'scaler.pth')) and os.path.exists(
                os.path.join(args.ae_folder, 'scaler_d.pth')):
            scaler.load_state_dict(torch.load(os.path.join(args.ae_folder, 'scaler.pth')))
            scaler_d.load_state_dict(torch.load(os.path.join(args.ae_folder, 'scaler_d.pth')))

    model.train()
    disc_start = criterion.module.discriminator_iter_start

    for it, (x, _) in enumerate(train_loader):
        it += last_it
        if it > args.max_iter:
            break
        batch_size = x.size(0)

        x = x.to(device)
        x = rearrange(x / 127.5 - 1, 'b t c h w -> b c t h w')  # normalize to [-1, 1]

        if not disc_opt:
            with (autocast()):
                x_tilde, vq_loss = model(x)

                if it % accum_iter == 0:
                    model.zero_grad()

                ae_loss = criterion(vq_loss, x,
                                    rearrange(x_tilde, '(b t) c h w -> b c t h w', b=batch_size),
                                    optimizer_idx=0,
                                    global_step=it)

                ae_loss = ae_loss / accum_iter

            scaler.scale(ae_loss).backward()

            if it % accum_iter == accum_iter - 1:
                scaler.step(opt)
                scaler.update()

            scheduler.step()

            losses['ae_loss'].update(ae_loss.item(), 1)
        else:
            if it % accum_iter == 0:
                criterion.zero_grad()

            with autocast():
                with torch.no_grad():
                    x_tilde, vq_loss = model(x)

                d_loss = criterion(vq_loss, x,
                                   rearrange(x_tilde, '(b t) c h w -> b c t h w', b=batch_size),
                                   optimizer_idx=1,
                                   global_step=it)
                d_loss = d_loss / accum_iter

            scaler_d.scale(d_loss).backward()

            if it % accum_iter == accum_iter - 1:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler_d.unscale_(d_opt)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(criterion.module.discriminator_2d.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(criterion.module.discriminator_3d.parameters(), 1.0)

                scaler_d.step(d_opt)
                scaler_d.update()

            losses['d_loss'].update(d_loss.item() * 3, 1)

        if it % accum_iter == accum_iter - 1 and it // accum_iter >= disc_start:
            if disc_opt:
                disc_opt = False
            else:
                disc_opt = True

        if it % args.log_freq == 0:
            if it % args.eval_freq == 0:
                fvd, ssim, lpips, psnr = eval_autoencoder(rank, model, val_loader, it, args.eval_samples, logger)
                lpips *= 1000
            if logger is not None and rank == 0:
                logger.scalar_summary('train/ae_loss', losses['ae_loss'].average, it)
                logger.scalar_summary('train/d_loss', losses['d_loss'].average, it)
                if it % args.eval_freq == 0:
                    logger.scalar_summary('test/psnr', psnr, it)
                    logger.scalar_summary('test/fvd', fvd, it)
                    logger.scalar_summary('test/ssim', ssim, it)
                    logger.scalar_summary('test/lpips', lpips, it)

                    log_('[It %d] [Time %.3f] [AELoss %f] [DLoss %f] [PSNR %.2f] [FVD %.2f] [SSIM %.2f] [LPIPS %.2f]' %
                         (it, time.time() - check, losses['ae_loss'].average, losses['d_loss'].average, psnr, fvd, ssim,
                          lpips))
                else:
                    log_('[It %d] [Time %.3f] [AELoss %f] [DLoss %f]' %
                         (it, time.time() - check, losses['ae_loss'].average, losses['d_loss'].average))

                torch.save(model.module.state_dict(), os.path.join(logger.logdir, 'model_last.pth'))
                torch.save(criterion.module.state_dict(), os.path.join(logger.logdir, 'loss_last.pth'))
                torch.save(opt.state_dict(), os.path.join(logger.logdir, 'opt.pth'))
                torch.save(d_opt.state_dict(), os.path.join(logger.logdir, 'd_opt.pth'))
                torch.save(scaler.state_dict(), os.path.join(logger.logdir, 'scaler.pth'))
                torch.save(scaler_d.state_dict(), os.path.join(logger.logdir, 'scaler_d.pth'))
                torch.save(scheduler.state_dict(), os.path.join(logger.logdir, 'scheduler.pth'))

            losses = dict()
            losses['ae_loss'] = AverageMeter()
            losses['d_loss'] = AverageMeter()

        if it % args.eval_freq == 0 and rank == 0:
            torch.save(model.module.state_dict(), os.path.join(logger.logdir, f'model_{it}.pth'))
            torch.save(criterion.module.state_dict(), os.path.join(logger.logdir, f'loss_{it}.pth'))
            torch.save(opt.state_dict(), os.path.join(logger.logdir, f'opt_{it}.pth'))
            torch.save(d_opt.state_dict(), os.path.join(logger.logdir, f'd_opt_{it}.pth'))
            torch.save(scaler.state_dict(), os.path.join(logger.logdir, f'scaler_{it}.pth'))
            torch.save(scaler_d.state_dict(), os.path.join(logger.logdir, f'scaler_d_{it}.pth'))

    if rank == 0:
        torch.save(model.state_dict(), os.path.join(logger.logdir, 'net_meta.pth'))
