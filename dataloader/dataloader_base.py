import torch
import torch.utils.data as data
import pandas as pd
from torchvision import transforms
import numpy as np
from dataloader.transforms import _transforms_video as T
from torch.utils.data import WeightedRandomSampler
#import av
import os


class DataloaderBase(data.Dataset):
    def __init__(self, df,
                 config,
                 transform=None):
        super(DataloaderBase, self).__init__()
        self.config = config
        self.transform = transform
        self.df = df
        self.num_samples = len(self.df)

    def __len__(self):
        return self.num_samples

    def replace_labels(self, label_uniques):
        new_values = np.int16(np.linspace(0, len(label_uniques),
                              len(label_uniques), endpoint=False))
        dictionary = dict(zip(np.array(label_uniques),
                              new_values))
        return dictionary

    def map_labels(self, labels):
        mapping = self.replace_labels(self.labels.drop_duplicates())
        labels = pd.Series([mapping[k] for k in self.df['labels'].to_list()])
        return labels

    # def load_video(self, path):
    #     # use av backend
    #     container = av.open(path)
    #     array_3d = np.zeros((
    #                     container.streams.video[0].height,
    #                     container.streams.video[0].width,
    #                     container.streams.video[0].frames, 3), dtype=np.uint8)
    #     i = 0
    #     for frame in container.decode(video=0):
    #         array_3d[:, :, i, :] = np.array(frame.to_image())
    #         i += 1
    #     return array_3d


class DataloaderTest(DataloaderBase):
    def __init__(self, df, transform=None):
        super(DataloaderTest, self).__init__(df, transform)

    @staticmethod
    def pad_volume(x, pad_size):
        flip = torch.flip(x, [2])
        x = torch.cat((x, flip), 2)[:, :, 0:pad_size, :]
        return x

    @staticmethod
    def create_patches(x, nr_frames, transform):
        inp_frames = x.shape[2]
        nr_patches = int(np.ceil(inp_frames / nr_frames))
        pad_size = nr_patches * nr_frames
        x = DataloaderTest.pad_volume(x, pad_size)
        crop_size = DataloaderTest.get_crop_size(transform)
        patches = torch.zeros(nr_patches, x.shape[3], nr_frames,
                              crop_size[0], crop_size[1])
        for i in range(0, nr_patches):
            patch = x[:, :, i*nr_frames:(i+1)*nr_frames, :]
            if transform:
                patch = transform(patch)
            patches[i, :, :, :, :] = patch
        return patches

    @staticmethod
    def get_crop_size(transform):
        for i in transform.transforms:
            if hasattr(i, 'crop_size'):
                crop_size = i.crop_size
        return crop_size


class DataloaderTrain(DataloaderBase):
    def __init__(self, df, config, transform=None):
        super(DataloaderTrain, self).__init__(df,
                                              config,
                                              transform)
        if config['task_type'] in ['classification']:
            self.class_counts = self.df['labels'].value_counts().to_list()
            self.class_weights = [self.num_samples/self.class_counts[i] for
                                  i in range(len(self.class_counts))]  # [::-1]
            self.weights = [self.class_weights[self.df['labels'].to_list()[i]]
                            for i in range(int(self.num_samples))]

            self.sampler = WeightedRandomSampler(
                torch.DoubleTensor(self.weights), int(self.num_samples))


def getVideoTrainTransforms(nr_frames=32,
                            crop_size=(224, 224)):
    transform_train = transforms.Compose(
            [
             transforms.Lambda(lambda x: x.permute(2, 0, 1, 3)),
             T.ToTensorVideo(),
             T.FrameSample(nr_frames),
             T.NormalizeVideo(
                    mean=(0.43216, 0.394666, 0.37645),
                    std=(0.22803, 0.22145, 0.216989)
                ),
             T.RandomCropVideo(crop_size),

            ])
    return transform_train


def getVideoTestTransforms(nr_frames=32,
                           uni_percent_frame=0.5,
                           crop_size=(224, 224)):
    transform_test = transforms.Compose(
            [
             transforms.Lambda(lambda x: x.permute(2, 0, 1, 3)),
             T.ToTensorVideo(),
             T.UniformFrameSample(nr_frames, uni_percent_frame),
             #  T.FrameSample(nr_frames),
             T.NormalizeVideo(
                    mean=(0.43216, 0.394666, 0.37645),
                    std=(0.22803, 0.22145, 0.216989)
                ),
             T.CenterCropVideo(crop_size)
            ])
    return transform_test


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        # if self.concat is True:
        #     q = torch.unsqueeze(q, 1)
        #     k = torch.unsqueeze(k, 1)
        #     x = torch.cat((q, k), dim=1)
        #     return x
        # else:
        return [q, k]
