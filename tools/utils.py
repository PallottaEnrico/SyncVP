import os
import random
import sys
from datetime import datetime, timedelta
from glob import glob

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import gdown


class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""

    def __init__(self, fn, path='./results', resume=False):
        self.path = path
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        logdir = self._make_dir(fn)
        if not os.path.exists(logdir):
            os.mkdir(logdir)

        if len(os.listdir(logdir)) != 0 and not resume:
            raise ValueError(
                f"Results directory {logdir} already exists and is not empty. Set resume=True to resume training.")

        self.set_dir(logdir)

    def _make_dir(self, fn):
        # today = datetime.today().strftime("%y%m%d")
        logdir = os.path.join(self.path, fn + '/')
        return logdir

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.writer = SummaryWriter(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
        self.log_file.flush()

        print('[%s] %s' % (datetime.now(), string))
        sys.stdout.flush()

    def log_dirname(self, string):
        self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
        self.log_file.flush()

        print('%s (%s)' % (string, self.logdir))
        sys.stdout.flush()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        self.writer.add_image(tag, images, step)

    def video_summary(self, tag, videos, step):
        self.writer.add_video(tag, videos, step, fps=16)

    def histo_summary(self, tag, values, step):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(tag, values, step, bins='auto')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_name(args):
    fn = f'{args.exp}_{args.id}_{args.data}'
    fn += f'_{args.seed}'
    return fn


def download(id, fname, root=os.path.expanduser('~/.cache/video-diffusion')):
    os.makedirs(root, exist_ok=True)
    destination = os.path.join(root, fname)

    if os.path.exists(destination):
        return destination

    gdown.download(id=id, output=destination, quiet=False)
    return destination


def setup_distibuted_training(args, rank):
    temp_dir = './'
    if args.n_gpus >= 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':  # Windows
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank,
                                                 world_size=args.n_gpus)
        else:  # Unix
            init_method = f'file://{init_file}'
            os.environ[
                'TORCH_NCCL_BLOCKING_WAIT'] = '0'  # eval is performed on a single GPU, this avoids timeout from other waiting processes
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank,
                                                 world_size=args.n_gpus,
                                                 timeout=timedelta(seconds=10000))


def setup_logger(args, rank):
    if rank == 0:
        fn = file_name(args)
        logger = Logger(fn, path=args.output, resume=args.resume)
        logger.log(args)
        logger.log(f'Log path: {logger.logdir}')
    else:
        logger = None

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    return log_, logger


def resume_training(diffusion_model, opt, scheduler, save_dir, log, args, last_it):
    # Extract last it number from the model name considering that the last model is model_last.pth
    all_models = glob(os.path.join(save_dir, 'ema_model_*0.pth'))
    if len(all_models) > 0:
        last_it = max([int(os.path.basename(x).split('_')[2].split('.')[0]) for x in all_models])
    log(f"Resuming from iteration {last_it}")

    diff_model_ckpt = torch.load(os.path.join(save_dir, f'ema_model_{last_it}.pth'))
    diffusion_model.load_state_dict(diff_model_ckpt)
    del diff_model_ckpt

    if not args.no_sched:
        opt_ckpt = torch.load(os.path.join(save_dir, 'opt_last.pth'))
        opt.load_state_dict(opt_ckpt)
        del opt_ckpt

    if os.path.exists(os.path.join(save_dir, 'scheduler_last.pth')) and not args.no_sched:
        sched_ckpt = torch.load(os.path.join(save_dir, 'scheduler_last.pth'))
        scheduler.load_state_dict(sched_ckpt)
        del sched_ckpt
    log(f"Lr: : {scheduler.get_lr()[0]}")

    return last_it
