import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import monai
from monai.apps import DecathlonDataset
from monai.data import list_data_collate
from monai.transforms import CopyItemsd
from dataloader.Segmentation._3D.dataloader_monai_segmentation_3D import ConvertToMultiChannel
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AsChannelLastd,
    AddChanneld,
    AsDiscrete,
    CenterSpatialCropd,
    ScaleIntensityd,
    Compose,
    Lambda,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    Lambdad,
    RandCropByPosNegLabeld,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    RandRotate90d,
    ToTensord,
    ConcatItemsd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    RandAffined,
    CropForegroundd,
    Transpose)


class base_monai_segmentation_loader():
    def __init__(self, image_path, csv_path_file, config):
        self.image_path = image_path
        self.csv = pd.read_csv(csv_path_file)
        self.features = self.get_input_features(self.csv)
        self.csv = self.set_feature_path(self.csv,
                                         self.features,
                                         self.image_path)
        self.data = self.csv.to_dict('records')
        self.image_data = self.csv[self.features+[config['labels_names']]].to_dict('records')
        self.config = config

    def get_input_features(self, csv):
        features = [col for col in
                    self.csv.columns.tolist() if col.startswith('inputs')]
        return features

    def set_feature_path(self, csv, features, image_path):
        feature_paths = features + [config['labels_names']]
        for feature in feature_paths:
            csv[feature] = csv[feature].apply(
                    lambda x: os.path.join(image_path, x))
        return csv


class train_monai_segmentation_loader(base_monai_segmentation_loader):
    def __init__(self, image_path, csv_path_file, config):
        super(train_monai_segmentation_loader, self).__init__(image_path,
                                                              csv_path_file,
                                                              config)

    def __call__(self):
        # define transforms for image and segmentation
        trans_pre = [
                LoadImaged(keys=self.features + [self.config["labels_names"]]),
                ConvertToMultiChannel(keys=self.config["labels_names"]),
                AsChannelFirstd(keys=self.features),
                RandCropByPosNegLabeld(
                    keys=self.features + [self.config["labels_names"]],
                    label_key=self.config["labels_names"],
                    spatial_size=[self.config['loaders']['height'],
                                  self.config['loaders']['width']],
                    pos=1, neg=1, num_samples=1),
                NormalizeIntensityd(keys=self.features, nonzero=True,
                                    channel_wise=True),
                ToTensord(keys=self.features + [self.config["labels_names"]]),
            ]

        train_transforms = Compose(trans_pre)

        # # CHECK: for debug
        # check_ds = monai.data.Dataset(data=self.data,
        #                              transform=train_transforms)
        # check_loader = DataLoader(
        #     check_ds,
        #     batch_size=self.config['loaders']['batchSize'],
        #     num_workers=self.config['loaders']['numWorkers'],
        #     collate_fn=list_data_collate)
        # check_data = monai.utils.misc.first(check_loader)

        # create a training data loader
        train_loader = monai.data.CacheDataset(data=self.image_data,
                                               transform=train_transforms)
        return train_loader


class val_monai_segmentation_loader(base_monai_segmentation_loader):
    def __init__(self, image_path, csv_path_file, config):
        super(val_monai_segmentation_loader, self).__init__(image_path,
                                                            csv_path_file,
                                                            config)

    def __call__(self):
        trans_pre = [
                LoadImaged(keys=self.features + [self.config["labels_names"]]),
                ConvertToMultiChannel(keys=self.config["labels_names"]),
                AsChannelFirstd(keys=self.features),
                RandCropByPosNegLabeld(
                    keys=self.features + [self.config["labels_names"]],
                    label_key=self.config["labels_names"],
                    spatial_size=[self.config['loaders']['height'],
                                  self.config['loaders']['width']],
                    pos=1, neg=1, num_samples=1),
                NormalizeIntensityd(keys=self.features, nonzero=True,
                                    channel_wise=True),
                ToTensord(keys=self.features + [self.config["labels_names"]]),

            ]

        val_transforms = Compose(trans_pre)
        val_loader = monai.data.Dataset(data=self.image_data,
                                        transform=val_transforms)

        return val_loader


class val_monai_loader_sliding_window(base_monai_segmentation_loader):
    def __init__(self, image_path, csv_path_file, config):
        super(val_monai_loader_sliding_window, self).__init__(image_path,
                                                              csv_path_file,
                                                              config)

    def __call__(self):
        trans_pre = [
                LoadImaged(keys=self.features + [self.config["labels_names"]]),
                ConvertToMultiChannel(keys=self.config["labels_names"]),
                AsChannelFirstd(keys=self.features),
                NormalizeIntensityd(keys=self.features, nonzero=True,
                                    channel_wise=True),
                ToTensord(keys=self.features + [self.config["labels_names"]]),

            ]

        val_transforms = Compose(trans_pre)

        val_loader = monai.data.CacheDataset(data=self.image_data,
                                        transform=val_transforms)

        return val_loader
