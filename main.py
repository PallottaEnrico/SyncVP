import sys;
sys.path.extend(['.'])

import yaml
import os
import argparse
import torch
from exps.diffusion import diffusion_training, multimodal_diffusion_training
from exps.autoencoder import autoencoder_training
from tools.utils import set_random_seed
from tools.config_utils import ddpm_config_setup, autoencoder_config_setup, mmddpm_config_setup

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, help="Path to config file for all next arguments")

""" General arguments """
parser.add_argument('--exp', type=str, default='ddpm', help='Type of training to run [autoencoder, ddpm, mmddpm]')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--id', type=str, default='', help='experiment identifier')
parser.add_argument('--n_gpus', type=int, default=2, help='Number of GPUs to use')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
parser.add_argument('--output', type=str, default='./results', help='Output directory where to store exp results')

""" Args about Data """
parser.add_argument('--data', type=str, default='UCF101',
                    help='Dataset identifier, must be implemented in get_loaders()')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training (will be split over gpus if n_gpus > 1)')
parser.add_argument('--data_folder', type=str, default='', help='path to datasets root folder')

""" Args about latent autoencoder """
parser.add_argument('--ae_config', type=str, default='configs/8x128x128.yaml',
                    help='the path of autoencoder config for the main modality')
parser.add_argument('--ae_cond_config', type=str, default='',
                    help='conditional frames may have a different autoencoder config if specified')

parser.add_argument('--ae_model', type=str, default='', help='autoencoder pretrained weights for the main modality')
parser.add_argument('--ae_cond_model', type=str, default='',
                    help='autoencoder pretrained weights for the conditional frames, if not specified, the same as ae_model')

# for GAN resume
parser.add_argument('--ae_folder', type=str, default='', help='the folder of the autoencoder training before GAN')

# Second modality autoencoder - here we assume depth as default
parser.add_argument('--ae_model_depth', type=str, default='',
                    help='autoencoder pretrained weights for the depth modality')
parser.add_argument('--ae_cond_model_depth', type=str, default='',
                    help='autoencoder pretrained weights for the depth conditional frames, if not specified, the same as ae_model_depth')

""" Args about diffusion models """
parser.add_argument('--diffusion_config', type=str, default='',
                    help='the path of diffusion model config, whether it is a single modality or multimodal')
parser.add_argument('--diffusion_model', type=str, default='',
                    help='path for pretrained diffusion model (e.g. needed for resume)')

# Modality specific models needed to start training the multimodal diffusion model
parser.add_argument('--diffusion_rgb_model', type=str, default='', help='the path of pretrained model for rgb')
parser.add_argument('--diffusion_depth_model', type=str, default='', help='the path of pretrained model for depth')

""" Lr scheduler settings """
parser.add_argument('--no_sched', action='store_true', help='load scheduler or start from new one)')
parser.add_argument('--scale_lr', action='store_true', help='scale learning rate for batch size')


def main():
    """ Additional args ends here. """
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)

        # Update argparse arguments with YAML values
        for key, value in yaml_config.items():
            setattr(args, key, value)

    """ FIX THE RANDOMNESS """
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if os.path.exists('.torch_distributed_init'):
        os.remove('.torch_distributed_init')

    """ RUN THE EXP """
    if args.exp == 'ddpm':
        args = ddpm_config_setup(args)
        runner = diffusion_training
    elif args.exp == 'mmddpm':
        args = mmddpm_config_setup(args)
        runner = multimodal_diffusion_training
    elif args.exp == 'autoencoder':
        args = autoencoder_config_setup(args)
        runner = autoencoder_training
    else:
        raise ValueError("Unknown experiment.")

    if args.n_gpus == 1:
        runner(rank=0, args=args)
    else:
        torch.multiprocessing.spawn(fn=runner, args=(args,), nprocs=args.n_gpus)


if __name__ == '__main__':
    main()
