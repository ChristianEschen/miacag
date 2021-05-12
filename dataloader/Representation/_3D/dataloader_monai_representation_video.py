import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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

def show_img(array):
    array = array.numpy()
    #array = check_data['inputs'][0,:,0,:, :].numpy()
    array = np.transpose(array, (1, 2, 0))
    img = (array - np.min(array)) / \
        (np.amax(array) - np.amin(array) + 1e-8)  # + 1e-8
    img = img * 255.0
    img = np.uint8(img)
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.show()
class base_monai_representation_loader(base_monai_loader):
    def __init__(self, image_path, csv_path_file, config):
        super(base_monai_representation_loader, self).__init__(image_path,
                                                               csv_path_file,
                                                               config)
        self.features = self.get_input_features(self.csv)
        self.csv = self.set_feature_path(self.csv,
                                         self.features,
                                         self.image_path)
        self.data = self.csv.to_dict('records')


class train_monai_representation_loader(base_monai_representation_loader):
    def __init__(self, image_path, csv_path_file, config):
        super(train_monai_representation_loader, self).__init__(image_path,
                                                                csv_path_file,
                                                                config)

    def __call__(self):
        # define transforms for image
        train_transforms_pre = [
                # transforms on images
                LoadImaged(keys=self.features),
                Lambdad(keys=self.features,
                        func=lambda x: x.transpose(2, 1, 0, 3)),
                # Resized(keys=self.features,
                #         spatial_size=(self.config['loaders']['Resize_height'],
                #                       self.config['loaders']['Resize_width'],
                #                       self.config['loaders']['Resize_depth'])),
                RandSpatialCropSamplesd(
                    keys=self.features,
                    roi_size=[
                        self.config['loaders']['Crop_height'],
                        self.config['loaders']['Crop_width'],
                        self.config['loaders']['Crop_depth']],
                    random_size=True,
                    num_samples=2),
                Resized(keys=self.features,
                        spatial_size=(self.config['loaders']['Crop_height'],
                                      self.config['loaders']['Crop_width'],
                                      self.config['loaders']['Crop_depth'])),
                RandFlipd(keys=self.features, spatial_axis=1)
            ]
        if self.config['model']['in_channels'] != 1:
            train_transforms_post = [
                    ConcatItemsd(keys=self.features, name='inputs'),
                    ScaleIntensityd(keys='inputs'),
                    NormalizeIntensityd(keys='inputs',
                                        subtrahend=(0.43216, 0.394666, 0.37645),
                                        divisor=(0.22803, 0.22145, 0.216989),
                                        channel_wise=True),
                    # [B,N,C,T,H,W]
                    Lambdad(keys='inputs',
                            func=lambda x: x.transpose(0, 3, 2, 1) if
                            self.config['loaders']['model_dim_order'] == 'CH_D_H_W'
                            else x),
                    ToTensord(keys=self.features)]
        else:
            train_transforms_post = [
                    CopyItemsd(keys=self.features, times=1, names='inputs'),
                    RepeatChanneld('inputs', 3),
                    RandGaussianSmoothd(
                        keys='inputs', sigma_x=[0.1, 0.2],
                        sigma_y=[0.1, 0.2], sigma_z=[0.0, 0.0]),
                    RandAdjustContrastd(
                        keys='inputs',
                        gamma=(0.5, 4.5)),
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
 
        # # create a training data loader
        # dataset = monai.data.Dataset(data=self.data,
        #                              transform=train_transforms)
     #   train_loader = monai.data.DistributedSampler(dataset=dataset)
        # train_loader = monai.data.Dataset(data=self.data,
        #                                        transform=train_transforms)
        train_loader = monai.data.CacheDataset(data=self.data,
                                               transform=train_transforms,
                                               cache_rate=1)
        return train_loader


class val_monai_representation_loader(base_monai_representation_loader):
    def __init__(self, image_path, csv_path_file, config):
        super(val_monai_representation_loader, self).__init__(image_path,
                                                              csv_path_file,
                                                              config)

    def __call__(self):
        val_transforms_pre = [
                LoadImaged(keys=self.features),
                Lambdad(keys=self.features,
                        func=lambda x: x.transpose(2, 1, 0, 3)),
                # Resized(keys=self.features,
                #         spatial_size=(self.config['loaders']['Resize_height'],
                #                       self.config['loaders']['Resize_width'],
                #                       self.config['loaders']['Resize_depth'])),
                # CenterSpatialCropd(keys=self.features,
                #                    roi_size=[self.config['loaders']['height'],
                #                              self.config['loaders']['width'],
                #                              self.config['loaders']['depth']]),
                RandSpatialCropSamplesd(
                    keys=self.features,
                    roi_size=[
                        self.config['loaders']['Crop_height'],
                        self.config['loaders']['Crop_width'],
                        self.config['loaders']['Crop_depth']],
                    random_size=False,
                    num_samples=2),
                RandFlipd(keys=self.features, spatial_axis=1),
                RandGaussianSmoothd(
                    keys=self.features, sigma_x=[0.1, 0.2],
                    sigma_y=[0.1, 0.2], sigma_z=[0.0, 0.0]),
                RandAdjustContrastd(
                    keys=self.features,
                    gamma=(0.5, 4.5))
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
        # CHECK: for debug ###
        # check_ds = monai.data.Dataset(data=self.data,
        #                              transform=val_transforms)
        # check_loader = DataLoader(
        #     check_ds,
        #     batch_size=self.config['loaders']['batchSize'],
        #     num_workers=self.config['loaders']['numWorkers'],
        #     collate_fn=list_data_collate)
        # check_data = monai.utils.misc.first(check_loader)

        # val_loader = monai.data.DistributedSampler(data=self.data,
        #                                            transform=val_transforms)
        val_loader = monai.data.Dataset(data=self.data,
                                             transform=val_transforms)
        # val_loader = monai.data.CacheDataset(data=self.data,
        #                                      transform=val_transforms,
        #                                      cache_rate=0.8)
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


class test_monai_classification_loader(base_monai_representation_loader):
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
