from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import random
from tools.data_utils import resize_crop
import numpy as np
import torch
import cv2
import os
from glob import glob


class ImageFramesDataset(Dataset, ABC):
    def __init__(self, path, split, resolution, video_len, n_frames=16, max_size=None, seed=42):
        """
        Abstract class for a dataset which videos are stored as sequences of images (e.g. PNG Files).
        :param path: path to dataset
        :param split: train/val/test set
        :param resolution: Images resolution
        :param video_len: Length of original raw video, e.g. Cityscapes = 30 frames video
        :param n_frames: Length of video loaded by the __getitem__ method
        :param max_size: limit the total number of videos
        :param seed: random seed
        """
        self.path = os.path.join(path, split)  # train/val/test
        assert os.path.exists(self.path), f"{self.path} does not exist"
        self.name = path.split('/')[-1]
        self.resolution = resolution
        self.video_len = video_len
        self.nframes = n_frames

        self.video_list = []

        random.seed(seed)

    def __len__(self):
        return len(self.video_list)

    def _preprocess(self, video):
        video = resize_crop(video, self.resolution)
        return video

    @abstractmethod
    def load_video_paths(self):
        """
        Each dataset can be organized differently, implement this method for your dataset folder structure.
        :return: A list of lists of frames, e.g. [[0.png, 1.png, ...], [0.png, 1.png, ...]]
        """
        pass

    def __getitem__(self, idx):
        prefix = np.random.randint(self.video_len - self.nframes + 1)

        video = np.array([cv2.imread(image) for image in self.video_list[idx][prefix:prefix + self.nframes]])

        video = torch.tensor(video).permute(3, 0, 1, 2).contiguous()

        # Permute channels convert BGR to RGB
        video = video[[2, 1, 0], ...]

        return self._preprocess(video), idx


class MultiModalImageFramesDataset(Dataset, ABC):
    def __init__(self, path, split, resolution, video_len, n_frames=16, max_size=None, seed=42):
        """
        Abstract class for a dataset which videos are stored as sequences of images (e.g. PNG Files).
        :param path: list of paths to each modality
        :param split: train/val/test set
        :param resolution: Images resolution
        :param video_len: Length of original raw video, e.g. Cityscapes = 30 frames video
        :param n_frames: Length of video loaded by the __getitem__ method
        :param max_size: limit the total number of videos
        :param seed: random seed
        """
        assert len(
            path) == 2, "This dataset class only supports two modality setting. If you need only one use an implementation of ImageFramesDataset"
        self.path_rgb, self.path_depth = os.path.join(path[0], split), os.path.join(path[1], split)
        assert os.path.exists(self.path_rgb), f"{self.path_rgb} does not exist"
        assert os.path.exists(self.path_depth), f"{self.path_depth} does not exist"
        self.name = path[0].split('/')[-1]
        self.resolution = resolution
        self.video_len = video_len
        self.nframes = n_frames

        self.video_list_rgb = []
        self.video_list_depth = []

        random.seed(seed)

    def __len__(self):
        return len(self.video_list_rgb)

    def _preprocess(self, video):
        video = resize_crop(video, self.resolution)
        return video

    def __getitem__(self, idx):
        prefix = np.random.randint(self.video_len - self.nframes + 1)

        video_rgb = np.array([cv2.imread(image) for image in self.video_list_rgb[idx][prefix:prefix + self.nframes]])
        video_depth = np.array(
            [cv2.imread(image) for image in self.video_list_depth[idx][prefix:prefix + self.nframes]])

        video_rgb = torch.tensor(video_rgb).permute(3, 0, 1, 2).contiguous()
        video_depth = torch.tensor(video_depth).permute(3, 0, 1, 2).contiguous()

        # Permute channels convert BGR to RGB
        video_rgb = video_rgb[[2, 1, 0], ...]

        return self._preprocess(video_rgb), self._preprocess(video_depth), idx


class CityscapesDataset(ImageFramesDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load the list of videos paths
        self.load_video_paths()

        if 'max_size' in kwargs:
            self.video_list = self.video_list[:kwargs['max_size']]

    def load_video_paths(self):
        # Load the list of videos paths
        self.video_list = sorted(glob(os.path.join(self.path, '*', '*.png')))
        # Every video is made of 30 frames so reorganize them in a list of videos
        self.video_list = [self.video_list[i:i + self.video_len] for i in
                           range(0, len(self.video_list), self.video_len)]


class MultiModalCityscapesDataset(MultiModalImageFramesDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load the list of videos paths
        self.load_video_paths()

        if 'max_size' in kwargs:
            self.video_list_rgb = self.video_list_rgb[:kwargs['max_size']]
            self.video_list_depth = self.video_list_depth[:kwargs['max_size']]

    def load_video_paths(self):
        # Load the list of videos paths
        self.video_list_rgb = sorted(glob(os.path.join(self.path_rgb, '*', '*.png')))
        self.video_list_rgb = [self.video_list_rgb[i:i + self.video_len] for i in
                               range(0, len(self.video_list_rgb), self.video_len)]

        self.video_list_depth = sorted(glob(os.path.join(self.path_depth, '*', '*.png')))
        self.video_list_depth = [self.video_list_depth[i:i + self.video_len] for i in
                                 range(0, len(self.video_list_depth), self.video_len)]
