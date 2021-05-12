import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import monai
from monai.apps import DecathlonDataset
from monai.data import list_data_collate
from torchvision import transforms
from monai.data import PILReader
from monai.transforms import (
    Activations,
    ToPILd,
    AsChannelFirstd,
    RepeatChanneld,
    AddChanneld,
    AsDiscrete,
    CenterSpatialCropd,
    ScaleIntensityd,
    Compose,
    RandZoomd,
    ToNumpyd,
    CopyItemsd,
    RandTorchVisiond,
    LoadImaged,
    LoadImage,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    Spacingd,
    RandRotate90d,
    ToTensord,
    ConcatItemsd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    RandAffined,
    CropForegroundd,
    Resized,
    GaussianSmoothd,
    Lambdad,
    RandSpatialCropSamplesd)
from dataloader.dataloader_base_monai import base_monai_loader
from dataloader.dataloader_base import TwoCropsTransform


def show_img(array):
    array = array.numpy()
    #array = check_data['inputs'][0,:,0,:, :].numpy()
    array = np.transpose(array, (1, 2, 0))
    img = (array - np.min(array)) / \
        (np.amax(array) - np.amin(array) + 1e-8)  # + 1e-8
    img = img * 255.0
    img = np.uint8(img)
    plt.figure(figsize=(2,2))
    plt.imshow(img)
    plt.show()


class base_monai_representation_loader(base_monai_loader):
    def __init__(self, image_path, csv_path_file, config,
                 use_complete_data=False):
        super(base_monai_representation_loader, self).__init__(
            image_path,
            csv_path_file,
            config,
            use_complete_data)
        self.features = self.get_input_features(self.csv)
        self.csv = self.set_feature_path(self.csv,
                                         self.features,
                                         self.image_path)
        self.data = self.csv.to_dict('records')

    def get_input_features(self, csv):
        features = [col for col in
                    self.csv.columns.tolist() if col.startswith('inputs')]
        return features

class train_monai_representation_loader(base_monai_representation_loader):
    def __init__(self, image_path, csv_path_file, config,
                 use_complete_data):
        super(train_monai_representation_loader, self).__init__(image_path,
                                                                csv_path_file,
                                                                config,
                                                                use_complete_data)

    def __call__(self):
        # define transforms for image
        train_transforms = [
                # transforms on images
                # Consider change to pure torchvision. it is 2X
                LoadImaged(keys='inputs', image_only=True, dtype='uint8'),
                ToPILd(keys='inputs'),
                RandTorchVisiond(keys='inputs',
                                 name="RandomResizedCrop",
                                 size=(self.config['loaders']['Resize_height'],
                                       self.config['loaders']['Resize_width']),
                                 scale=(0.2, 1.)),
                RandTorchVisiond(keys='inputs',
                                 name="RandomHorizontalFlip"),
                RandTorchVisiond(
                    keys='inputs', name='RandomApply',
                    transforms=[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8),
                RandTorchVisiond(keys='inputs',
                                 name="RandomGrayscale",
                                 p=0.2),
                RandTorchVisiond(keys='inputs',
                                 name="ToTensor"),
                RandTorchVisiond(keys='inputs',
                                 name="Normalize",
                                 mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010))
        ]



                # OLD
                #transforms.ToPILImage()
                # AsChannelFirstd(keys='inputs'),
                # RandZoomd(keys='inputs', prob=0.5,
                #           min_zoom=1, max_zoom=1.3,
                #           mode="bilinear",
                #           align_corners=True),
                # ScaleIntensityd(keys='inputs'),
                #ToTensord(keys='inputs')
                # RandTorchVisiond(keys='inputs',
                #                  name="RandomResizedCrop",
                #                  size=(self.config['loaders']['Resize_height'],
                #                        self.config['loaders']['Resize_width']),
                #                  scale=(0.2, 1.)),
                # RandTorchVisiond(keys='inputs',
                #                  name="RandomHorizontalFlip"),
                # RandTorchVisiond(
                #     keys='inputs', name='RandomApply',
                #     transforms=[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                #     p=0.8),
                # RandTorchVisiond(keys='inputs',
                #                  name="RandomGrayscale",
                #                  p=0.2),

                # RandTorchVisiond(keys='inputs',
                #                  name="Normalize",
                #                  mean=(0.4914, 0.4822, 0.4465),
                #                  std=(0.2023, 0.1994, 0.2010))
                #                  ]

        train_transforms = Compose(train_transforms)
        # CHECK: for debug ###
        # check_ds = monai.data.Dataset(data=self.data,
        #                               transform=train_transforms)
        # check_loader = DataLoader(
        #     check_ds,
        #     batch_size=self.config['loaders']['batchSize'],
        #     num_workers=self.config['loaders']['numWorkers'],
        #     collate_fn=list_data_collate)
        # check_data = monai.utils.misc.first(check_loader)
        # # create a training data loader

        train_loader = monai.data.CacheDataset(
            data=self.data,
            transform=TwoCropsTransform(train_transforms))
        return train_loader


class val_monai_representation_loader(base_monai_representation_loader):
    def __init__(self, image_path, csv_path_file, config, use_complete_data):
        super(val_monai_representation_loader, self).__init__(
            image_path,
            csv_path_file,
            config,
            use_complete_data)

    def __call__(self):
        # Monai
        val_transforms = [
                # transforms on images
                LoadImaged(keys='inputs'),
                AsChannelFirstd(keys='inputs'),
                ScaleIntensityd(keys='inputs'),
                ToTensord(keys='inputs'),
                RandTorchVisiond(keys='inputs',
                                 name="Normalize",
                                 mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010)),
                                  ]
        val_transforms = Compose(val_transforms)
        # torchvision

        # CHECK: for debug ###
        # check_ds = monai.data.Dataset(data=self.data,
        #                              transform=val_transforms)
        # check_loader = DataLoader(
        #     check_ds,
        #     batch_size=self.config['loaders']['batchSize'],
        #     num_workers=self.config['loaders']['numWorkers'],
        #     collate_fn=list_data_collate)
        # check_data = monai.utils.misc.first(check_loader)
        val_loader = monai.data.Dataset(data=self.data,
                                        transform=val_transforms)
        # val_loader = monai.data.CacheDataset(data=self.data,
        #                                      transform=val_transforms,
        #                                      cache_rate=0.8)
        return val_loader


### GRAVEYARD####

    # train_transforms = [
    #             # transforms on images
    #             LoadImaged(keys='inputs'),
    #             AsChannelFirstd(keys='inputs'),
    #             RandSpatialCropSamplesd(
    #                 keys='inputs',
    #                 roi_size=[
    #                     self.config['loaders']['Resize_height'],
    #                     self.config['loaders']['Resize_width']],
    #                 random_size=True,
    #                 num_samples=2),
    #             Resized(
    #                 keys='inputs',
    #                 spatial_size=(self.config['loaders']['Resize_height'],
    #                               self.config['loaders']['Resize_width'])
    #                               ),
    #             ScaleIntensityd(keys='inputs'),
    #             ToTensord(keys='inputs'),
    #             RandTorchVisiond(keys='inputs',
    #                              name="RandomResizedCrop",
    #                              size=(self.config['loaders']['Resize_height'],
    #                                    self.config['loaders']['Resize_width']),
    #                              scale=(0.2, 1.)),
    #             RandTorchVisiond(keys='inputs',
    #                              name="RandomHorizontalFlip"),
    #             # slow
    #             RandTorchVisiond(
    #                 keys='inputs', name='RandomApply',
    #                 transforms=[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
    #                 p=0.8),
    #             # RandTorchVisiond(
    #             #     keys='inputs', name='ColorJitter',
    #             #     brightness=0.4, contrast=0.4,
    #             #     saturation=0.4, hue=0.1),
    #             RandTorchVisiond(keys='inputs',
    #                              name="RandomGrayscale",
    #                              p=0.2),

    #             RandTorchVisiond(keys='inputs',
    #                              name="Normalize",
    #                              mean=(0.4914, 0.4822, 0.4465),
    #                              std=(0.2023, 0.1994, 0.2010))
    #                               ]