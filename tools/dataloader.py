import os
from torch.utils.data import DataLoader
from tools.data_utils import InfiniteSampler
from tools.datasets import CityscapesDataset, MultiModalCityscapesDataset


def get_loaders(rank, args):
    """
    Load dataloaders for a video dataset, center-cropped to a resolution.
    """
    dataset, modality = args.data.split('_')
    modality = modality.lower()

    # TODO: Add support to other datasets
    if dataset == "CITYSCAPES":
        video_len = 30
        dataset_class = CityscapesDataset
        data_dir = os.path.join(args.data_folder, dataset, modality)
        if modality == 'rgbd':
            dataset_class = MultiModalCityscapesDataset
            modalities = ['rgb', 'depth']
            data_dir = [os.path.join(args.data_folder, dataset, mod) for mod in modalities]
    else:
        raise NotImplementedError()

    if args.cond_model:  # false for autoencoders, true for video prediction
        args.frames += args.cond_frames

    trainset = dataset_class(data_dir, split='train', video_len=video_len, resolution=args.res, n_frames=args.frames, seed=args.seed)

    valset = None
    try:
        valset = dataset_class(data_dir, split='val', video_len=video_len, resolution=args.res, n_frames=args.frames, seed=args.seed)
    except Exception as e:
        print(e)
        print("Using test set as default")
        pass

    testset = dataset_class(data_dir, split='test', video_len=video_len, resolution=args.res, n_frames=args.frames, seed=args.seed)

    trainset_sampler = InfiniteSampler(dataset=trainset, rank=rank, num_replicas=args.n_gpus, seed=args.seed)
    trainloader = DataLoader(trainset, sampler=trainset_sampler, batch_size=args.batch_size // args.n_gpus,
                             pin_memory=False,
                             num_workers=args.num_workers)

    testloader = DataLoader(testset, batch_size=args.batch_size // args.n_gpus, pin_memory=False,
                            num_workers=args.num_workers, shuffle=False)

    if valset is not None:
        val_sampler = InfiniteSampler(valset, rank=rank, num_replicas=args.n_gpus, seed=args.seed)
        validationloader = DataLoader(valset, sampler=val_sampler, batch_size=args.batch_size // args.n_gpus,
                                      pin_memory=False,
                                      num_workers=args.num_workers)
    else:
        # Defaulting to test set
        validationloader = testloader

    return trainloader, validationloader, testloader
