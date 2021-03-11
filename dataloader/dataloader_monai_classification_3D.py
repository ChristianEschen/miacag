import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import monai
from monai.apps import DecathlonDataset
from monai.data import list_data_collate
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    RepeatChanneld,
    AddChanneld,
    AsDiscrete,
    CenterSpatialCropd,
    ScaleIntensityd,
    Compose,
    CopyItemsd,
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
    Spacingd,
    RandRotate90d,
    ToTensord,
    ConcatItemsd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    RandAffined,
    CropForegroundd,
    Resized,
    Lambdad,
    RandSpatialCropSamplesd)
from dataloader.dataloader_base_video import VideoDataloaderTest

class base_monai_classification_loader():
    def __init__(self, image_path, csv_path_file, config):
        self.image_path = image_path
        self.csv = pd.read_csv(csv_path_file)
        self.features = self.get_input_features(self.csv)
        self.csv = self.set_feature_path(self.csv,
                                         self.features,
                                         self.image_path)
        self.data = self.csv.to_dict('records')
        self.config = config

    def get_input_features(self, csv):
        features = [col for col in
                    csv.columns.tolist() if col.startswith('image')]
        return features

    def set_feature_path(self, csv, features, image_path):
        feature_paths = features #+ ['labels']
        for feature in feature_paths:
            csv[feature] = csv[feature].apply(
                    lambda x: os.path.join(image_path, x))
        return csv


class train_monai_classification_loader(base_monai_classification_loader):
    def __init__(self, image_path, csv_path_file, config):
        super(train_monai_classification_loader, self).__init__(image_path,
                                                                csv_path_file,
                                                                config)

    def __call__(self):
        # define transforms for image
        train_transforms_pre = [
                LoadImaged(keys=self.features),
                Lambdad(keys=self.features,
                        func=lambda x: x.transpose(2, 1, 0, 3)),
                Resized(keys=self.features,
                        spatial_size=(self.config['loaders']['Resize_height'],
                                      self.config['loaders']['Resize_width'],
                                      self.config['loaders']['Resize_depth'])),
                RandSpatialCropd(keys=self.features,
                                 roi_size=[
                                     self.config['loaders']['Crop_height'],
                                     self.config['loaders']['Crop_width'],
                                     self.config['loaders']['Crop_depth']],
                                 random_size=False)
            ]
        if self.config['model']['in_channels'] != 1:
            train_transforms_post = [
                    ConcatItemsd(keys=self.features, name='inputs'),
                    ScaleIntensityd(keys='inputs'),
                    NormalizeIntensityd(keys='inputs',
                                        subtrahend=(0.43216, 0.394666, 0.37645),
                                        divisor=(0.22803, 0.22145, 0.216989),
                                        channel_wise=True),
                    Lambdad(keys='inputs',
                            func=lambda x: x.transpose(0, 3, 2, 1) if
                            self.config['loaders']['model_dim_order'] == 'CH_D_H_W'
                            else x),
                    ToTensord(keys=self.features)]
        else:
            train_transforms_post = [
                    CopyItemsd(keys=self.features, times=1, names='inputs'),
                    RepeatChanneld('inputs', 3),
                    ScaleIntensityd(keys='inputs'),
                    NormalizeIntensityd(keys='inputs',
                                        subtrahend=(0.43216, 0.394666, 0.37645),
                                        divisor=(0.22803, 0.22145, 0.216989),
                                        channel_wise=True),
                    Lambdad(keys='inputs',
                            func=lambda x: x.transpose(0, 3, 2, 1) if
                            self.config['loaders']['model_dim_order'] == 'CH_D_H_W'
                            else x),
                    ToTensord(keys=self.features)]
        train_transforms = Compose(train_transforms_pre +
                                   train_transforms_post)
        # CHECK: for debug ###
        check_ds = monai.data.Dataset(data=self.data,
                                     transform=train_transforms)
        check_loader = DataLoader(
            check_ds,
            batch_size=self.config['loaders']['batchSize'],
            num_workers=self.config['loaders']['numWorkers'],
            collate_fn=list_data_collate)
        check_data = monai.utils.misc.first(check_loader)

        # create a training data loader
        train_loader = monai.data.CacheDataset(data=self.data,
                                               transform=train_transforms)

        return train_loader


class val_monai_classification_loader(base_monai_classification_loader):
    def __init__(self, image_path, csv_path_file, config):
        super(val_monai_classification_loader, self).__init__(image_path,
                                                              csv_path_file,
                                                              config)

    def __call__(self):
        val_transforms_pre = [
                LoadImaged(keys=self.features),
                Lambdad(keys=self.features,
                        func=lambda x: x.transpose(2, 1, 0, 3)),
                Resized(keys=self.features,
                        spatial_size=(self.config['loaders']['Resize_height'],
                                      self.config['loaders']['Resize_width'],
                                      self.config['loaders']['Resize_depth'])),
                # CenterSpatialCropd(keys=self.features,
                #                    roi_size=[self.config['loaders']['height'],
                #                              self.config['loaders']['width'],
                #                              self.config['loaders']['depth']]),
                RandSpatialCropSamplesd(
                    keys=self.features,
                    roi_size=[self.config['loaders']['Crop_height'],
                              self.config['loaders']['Crop_width'],
                              self.config['loaders']['Crop_depth']],
                    random_size=False,
                    num_samples=self.config[
                        'loaders']['val_method']['samples']),
            ]
        if self.config['model']['in_channels'] != 1:
            val_transforms_post = [
                ConcatItemsd(keys=self.features, name='inputs'),
                ScaleIntensityd(keys='inputs'),
                NormalizeIntensityd(keys='inputs',
                                    subtrahend=(0.43216, 0.394666, 0.37645),
                                    divisor=(0.22803, 0.22145, 0.216989),
                                    channel_wise=True),
                Lambdad(keys='inputs',
                        func=lambda x: x.transpose(0, 3, 2, 1) if
                        self.config['loaders']['model_dim_order'] == 'CH_D_H_W'
                        else x),
                ToTensord(keys=self.features)]
        else:
            val_transforms_post = [
                CopyItemsd(keys=self.features, times=1, names='inputs'),
                RepeatChanneld('inputs', 3),
                ScaleIntensityd(keys='inputs'),
                NormalizeIntensityd(keys='inputs',
                                    subtrahend=(0.43216, 0.394666, 0.37645),
                                    divisor=(0.22803, 0.22145, 0.216989),
                                    channel_wise=True),
                Lambdad(keys='inputs',
                        func=lambda x: x.transpose(0, 3, 2, 1) if
                        self.config['loaders']['model_dim_order'] == 'CH_D_H_W'
                        else x),
                ToTensord(keys=self.features)]
        val_transforms = Compose(val_transforms_pre +
                                 val_transforms_post)
        val_loader = monai.data.CacheDataset(data=self.data,
                                             transform=val_transforms)

        return val_loader


class SlidingClassificationDataset(monai.data.Dataset):
    def __init__(self, data, config, transform_pre, transform_post):
        self.data = data
        self.config = config
        self.transform_pre = transform_pre
        self.transform_post = transform_post

    def __len__(self):
        return len(self.data)

    def pad_volume(self, image, pad_size):
        flip = torch.flip(image, [3])
        image = torch.cat((image, flip), 3)[:, :, :, 0:pad_size]
        return image

    def create_patches(self, data, config, transform):
        image = data['inputs']
        image = torch.from_numpy(image)
        inp_frames = image.shape[3]
        nr_patches = int(np.ceil(inp_frames / config['loaders']['Crop_depth']))
        pad_size = nr_patches * config['loaders']['Crop_depth']
        image = self.pad_volume(image, pad_size)
        patches = torch.zeros(nr_patches,
                              image.shape[0],
                              config['loaders']['Crop_height'],
                              config['loaders']['Crop_width'],
                              config['loaders']['Crop_depth'])
        for i in range(0, nr_patches):
            patch = image[:, :, :, i*config['loaders']['Crop_depth']:
                          (i+1)*config['loaders']['Crop_depth']]
            data['inputs'] = patch.numpy()
            if transform:
                data = transform(data)
            patches[i, :, :, :, :] = patch
        data['inputs'] = patches.numpy()
        if config['loaders']['model_dim_order'] == 'CH_D_H_W':
            data['inputs'] = data['inputs'].transpose(0, 1, 4, 3, 2)
        return data

    def __getitem__(self, idx):
        data = self.transform_pre(self.data[idx])
        data = self.create_patches(data, self.config, self.transform_post)
        sample = {'inputs': data['inputs'],
                  'labels': data['labels'],
                  'file': data['image1_meta_dict']['filename_or_obj'],
                  'shape': data['image1'].shape[3]}
        return sample

class test_monai_classification_loader(base_monai_classification_loader):
    def __init__(self, image_path, csv_path_file, config):
        super(test_monai_classification_loader, self).__init__(image_path,
                                                               csv_path_file,
                                                               config)

    def __call__(self):
        test_transforms_pre_pre = [
                LoadImaged(keys=self.features),
                Lambdad(keys=self.features,
                        func=lambda x: x.transpose(2, 1, 0, 3)),
                Resized(keys=self.features,
                        spatial_size=(self.config['loaders']['Resize_height'],
                                      self.config['loaders']['Resize_width'],
                                      self.config['loaders']['Resize_depth'])),
            ]
        if self.config['model']['in_channels'] != 1:
            test_transforms_intermed = [
                ConcatItemsd(keys=self.features, name='inputs')]
        else:
            test_transforms_intermed = [
                CopyItemsd(keys=self.features, times=1, names='inputs'),
                RepeatChanneld('inputs', 3)]
        test_transforms_pre = Compose(test_transforms_pre_pre +
                                      test_transforms_intermed)
        test_transforms_post = Compose(
            [

                ScaleIntensityd(keys='inputs'),
                NormalizeIntensityd(keys='inputs',
                                    subtrahend=(0.43216, 0.394666, 0.37645),
                                    divisor=(0.22803, 0.22145, 0.216989),
                                    channel_wise=True),
                Lambdad(keys='inputs',
                        func=lambda x: x.transpose(0, 3, 2, 1) if
                        self.config['loaders']['model_dim_order'] == 'CH_D_H_W'
                        else x),
                ToTensord(keys=self.features),
            ])

        # CHECK: for debug ###
        # check_ds = monai.data.GridPatchDataset(
        #     dataset=self.data,
        #     #transform=test_transforms,
        #     patch_size = (self.config['loaders']['height'],
        #                       self.config['loaders']['width'],
        #                       self.config['loaders']['depth']))
        # check_loader = DataLoader(
        #     check_ds,
        #     batch_size=self.config['loaders']['batchSize'],
        #     num_workers=self.config['loaders']['numWorkers'],
        #     collate_fn=list_data_collate)
        # check_data = monai.utils.misc.first(check_loader)

        # create a training data loader
        # test_loader = monai.data.Dataset(data=self.data,
        #                                  transform=test_transforms)
        test_loader = SlidingClassificationDataset(
            data=self.data,
            config=self.config,
            transform_pre=test_transforms_pre,
            transform_post=test_transforms_post)
        return test_loader
