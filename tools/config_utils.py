from omegaconf import OmegaConf

def ddpm_config_setup(args):
    config = OmegaConf.load(args.diffusion_config)
    ae_config = OmegaConf.load(args.ae_config)

    args.unetconfig = config.model.params.unet_config
    args.lr = config.model.base_learning_rate if 'base_learning_rate' in config.model else 1e-4
    args.max_iter = config.model.max_iter
    args.res = ae_config.model.params.ddconfig.resolution
    args.frames = ae_config.model.params.ddconfig.frames
    args.ddconfig = ae_config.model.params.ddconfig
    args.embed_dim = ae_config.model.params.embed_dim
    args.ddpmconfig = config.model.params
    args.cond_model = True
    args.cond_frames = args.frames
    args.cond_prob = config.model.cond_prob
    args.log_freq = config.model.log_freq
    args.eval_freq = config.model.eval_freq
    args.max_size = config.model.max_size if 'max_size' in config.model else None
    args.eval_samples = config.model.eval_samples if 'eval_samples' in config.model else 16
    args.resume = config.model.resume if 'resume' in config.model else False

    if args.ae_cond_config != '':
        ae_cond_config = OmegaConf.load(args.ae_cond_config)
        args.ae_cond_ddconfig = ae_cond_config.model.params.ddconfig
        args.cond_frames = ae_cond_config.model.params.ddconfig.frames

    return args


def mmddpm_config_setup(args):
    config = OmegaConf.load(args.diffusion_config)
    ae_config = OmegaConf.load(args.ae_config)

    args.unetconfig = config.model.params.unet_config
    args.lr = config.model.base_learning_rate if 'base_learning_rate' in config.model else 1e-5
    args.max_iter = config.model.max_iter
    args.res = ae_config.model.params.ddconfig.resolution
    args.frames = ae_config.model.params.ddconfig.frames
    args.ddconfig = ae_config.model.params.ddconfig
    args.embed_dim = ae_config.model.params.embed_dim
    args.ddpmconfig = config.model.params
    args.cond_model = True
    args.cond_prob = config.model.cond_prob
    args.log_freq = config.model.log_freq
    args.eval_freq = config.model.eval_freq
    args.max_size = config.model.max_size if 'max_size' in config.model else None
    args.eval_samples = config.model.eval_samples if 'eval_samples' in config.model else 16
    args.resume = config.model.resume if 'resume' in config.model else False
    args.same_noise = config.model.same_noise if 'same_noise' in config.model else True

    if args.ae_cond_config != '':
        ae_cond_config = OmegaConf.load(args.ae_cond_config)
        args.ae_cond_ddconfig = ae_cond_config.model.params.ddconfig
        args.cond_frames = ae_cond_config.model.params.ddconfig.frames if args.cond_model else 0
    else:
        args.ae_cond_ddconfig = ae_config.model.params.ddconfig
        args.cond_frames = ae_config.model.params.ddconfig.frames if args.cond_model else 0

    args.cross_attn_configs = {
        'normalize': config.model.normalize if 'normalize' in config.model else False,
        'skip_conn': config.model.skip_conn if 'skip_conn' in config.model else False,
        'cross_attn': config.model.cross_attn if 'cross_attn' in config.model else 'deep',
        'single_attn': config.model.single_attn if 'single_attn' in config.model else False,
        'split_attn': config.model.split_attn if 'split_attn' in config.model else False,
        'num_heads': config.model.num_heads if 'num_heads' in config.model else 8,
    }

    args.shared = config.model.shared if 'shared' in config.model else False

    args.modality_guidance = config.model.modality_guidance if 'modality_guidance' in config.model else False

    return args


def autoencoder_config_setup(args):
    config = OmegaConf.load(args.ae_config)
    args.ddconfig = config.model.params.ddconfig
    args.embed_dim = config.model.params.embed_dim
    args.lossconfig = config.model.params.lossconfig
    args.lr = config.model.base_learning_rate if 'base_learning_rate' in config.model else 1e-4
    args.res = config.model.params.ddconfig.resolution
    args.perceptual_weight = config.model.params.perceptual_weight if 'perceptual_weight' in config.model.params else 0.0
    args.frames = config.model.params.ddconfig.frames
    args.resume = config.model.resume if 'resume' in config.model else False
    args.amp = config.model.amp
    args.max_iter = config.model.max_iter
    args.log_freq = config.model.log_freq if 'log_freq' in config.model else 1000
    args.eval_freq = config.model.eval_freq if 'eval_freq' in config.model else 10000
    args.eval_samples = config.model.eval_samples if 'eval_samples' in config.model else 128
    # Limits the dataset size
    args.max_size = config.model.max_size if 'max_size' in config.model else None
    args.cond_model = False

    return args
