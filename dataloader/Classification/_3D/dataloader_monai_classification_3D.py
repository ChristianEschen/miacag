import os
import torch
from torch.utils.data import DataLoader
from dataloader.transforms._transforms_video import RandomCropVideo
import pandas as pd
import numpy as np
import monai
from monai.apps import DecathlonDataset
from monai.data import list_data_collate, pad_list_data_collate
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    RepeatChanneld,
    AddChanneld,
    EnsureChannelFirstD,
    AsDiscrete,
    CenterSpatialCropd,
    ScaleIntensityd,
    RandTorchVisiond,
    Compose,
    SqueezeDimd,
    RandLambdad,
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
from dataloader.dataloader_base_monai import base_monai_loader


class base_monai_classification_loader(base_monai_loader):
    def __init__(self, df, config):
        super(base_monai_classification_loader, self).__init__(
            df,
            config)

        self.features = self.get_input_features(self.df)
        self.data = self.df[self.features + ['labels']]
        self.data = self.data.to_dict('records')


class train_monai_classification_loader(base_monai_classification_loader):
    def __init__(self, df, config):
        super(train_monai_classification_loader, self).__init__(
            df,
            config)

    def __call__(self):
        # define transforms for image
        train_transforms = [
                LoadImaged(keys=self.features),
                EnsureChannelFirstD(keys=self.features),
                self.resampleORresize(),
                self.getMaybePad(),
                RandSpatialCropd(keys=self.features,
                                 roi_size=[
                                     self.config['loaders']['Crop_height'],
                                     self.config['loaders']['Crop_width'],
                                     self.config['loaders']['Crop_depth']],
                                 random_size=False),
                self.getCopy1to3Channels(),
                ConcatItemsd(keys=self.features, name='inputs'),
                ScaleIntensityd(keys='inputs'),
                NormalizeIntensityd(keys='inputs',
                                    channel_wise=True),
                ToTensord(keys='inputs')
                ]
        train_transforms = Compose(train_transforms)
        # CHECK: for debug ###
        # check_ds = monai.data.Dataset(data=self.data,
        #                              transform=train_transforms)
        # check_loader = DataLoader(
        #     check_ds,
        #     batch_size=self.config['loaders']['batchSize'],
        #     num_workers=self.config['num_workers'],
        #     collate_fn=list_data_collate)
        # check_data = monai.utils.misc.first(check_loader)
        # img = check_data['inputs'][0,0,:,:,16].numpy()
        # import matplotlib.pyplot as plt
        # fig_train = plt.figure()
        # plt.imshow(img, cmap="gray", interpolation="None")
        # plt.show()
        # create a training data loader
        train_loader = monai.data.Dataset(data=self.data,
                                               transform=train_transforms)

        return train_loader


class val_monai_classification_loader(base_monai_classification_loader):
    def __init__(self, df, config):
        super(val_monai_classification_loader, self).__init__(df,
                                                              config)

    def __call__(self):
        val_transforms = [
                LoadImaged(keys=self.features),
                EnsureChannelFirstD(keys=self.features),
                self.resampleORresize(),
                self.getMaybePad(),
                RandSpatialCropd(
                    keys=self.features,
                    roi_size=[
                                     self.config['loaders']['Crop_height'],
                                     self.config['loaders']['Crop_width'],
                                     self.config['loaders']['Crop_depth']],
                    random_size=False,),
                self.getCopy1to3Channels(),
                ConcatItemsd(keys=self.features, name='inputs'),
                ScaleIntensityd(keys='inputs'),
                NormalizeIntensityd(keys='inputs',
                                    channel_wise=True),
                ToTensord(keys='inputs')]

        val_transforms = Compose(val_transforms)
        # CHECK: for debug ###
        # check_ds = monai.data.Dataset(
        #     data=self.data, transform=val_transforms)
        # check_loader = DataLoader(
        #     check_ds,
        #     batch_size=self.config['loaders']['batchSize'],
        #     num_workers=self.config['num_workers'],
        #     collate_fn=list_data_collate)
        # check_data = monai.utils.misc.first(check_loader)
        # import matplotlib.pyplot as plt
        # inp = check_data['inputs'][0,:,:,:,16].cpu().numpy()
        # inp =((inp-np.min(inp))/(np.max(inp)-np.min(inp))*255.0).astype(np.uint8)
        # imgplot = plt.imshow(np.transpose(inp, (1, 2, 0)))
        # plt.show()

        val_loader = monai.data.Dataset(data=self.data,
                                        transform=val_transforms)

        return val_loader


class val_monai_classification_loader_SW(
        base_monai_classification_loader):
    def __init__(self, df, config):
        super(val_monai_classification_loader_SW, self).__init__(image_path,
                                                              df_path_file,
                                                              config)

    def __call__(self):

        val_transforms = [
                LoadImaged(keys=self.features),
                EnsureChannelFirstD(keys=self.features),
                Resized(
                    keys=self.features,
                    spatial_size=(
                                self.config['loaders']['Resize_height'],
                                self.config['loaders']['Resize_width'],
                                self.config['loaders']['Resize_depth'])),
                self.getCopy1to3Channels(),
                ConcatItemsd(keys=self.features, name='inputs'),
                ScaleIntensityd(keys='inputs'),
                NormalizeIntensityd(keys='inputs',
                                    channel_wise=True),
                ToTensord(keys='inputs')]
        val_transforms = Compose(val_transforms)
        # CHECK: for debug ###
        # check_ds = monai.data.Dataset(data=self.data,
        #                              transform=val_transforms)
        # check_loader = DataLoader(
        #     check_ds,
        #     batch_size=self.config['loaders']['batchSize'],
        #     num_workers=self.config['loaders']['num_workers'],
        #     collate_fn=pad_list_data_collate)
        # check_data = monai.utils.misc.first(check_loader)
        val_loader = monai.data.Dataset(data=self.data,
                                        transform=val_transforms)

        return val_loader

# class SlidingClassificationDataset(monai.data.Dataset):
#     def __init__(self, data, config, transform_pre, transform_post):
#         self.data = data
#         self.config = config
#         self.transform_pre = transform_pre
#         self.transform_post = transform_post

#     def __len__(self):
#         return len(self.data)

#     def pad_volume(self, image, pad_size):
#         flip = torch.flip(image, [3])
#         image = torch.cat((image, flip), 3)[:, :, :, 0:pad_size]
#         return image

#     def create_patches(self, data, config, transform):
#         image = data['inputs']
#         image = torch.from_numpy(image)
#         inp_frames = image.shape[3]
#         nr_patches = int(np.ceil(inp_frames / config['loaders']['Crop_depth']))
#         pad_size = nr_patches * config['loaders']['Crop_depth']
#         image = self.pad_volume(image, pad_size)
#         patches = torch.zeros(nr_patches,
#                               image.shape[0],
#                               config['loaders']['Crop_height'],
#                               config['loaders']['Crop_width'],
#                               config['loaders']['Crop_depth'])
#         for i in range(0, nr_patches):
#             patch = image[:, :, :, i*config['loaders']['Crop_depth']:
#                           (i+1)*config['loaders']['Crop_depth']]
#             data['inputs'] = patch.numpy()
#             if transform:
#                 data = transform(data)
#             patches[i, :, :, :, :] = patch
#         data['inputs'] = patches.numpy()
#         if config['loaders']['model_dim_order'] == 'CH_D_H_W':
#             data['inputs'] = data['inputs'].transpose(0, 1, 4, 3, 2)
#         return data

#     def __getitem__(self, idx):
#         data = self.transform_pre(self.data[idx])
#         data = self.create_patches(data, self.config, self.transform_post)
#         sample = {'inputs': data['inputs'],
#                   'labels': data['labels'],
#                   'file': data['image1_meta_dict']['filename_or_obj'],
#                   'shape': data['image1'].shape[3]}
#         return sample

# class test_monai_classification_loader(base_monai_classification_loader):
#     def __init__(self, df, config):
#         super(test_monai_classification_loader, self).__init__(image_path,
#                                                                df_path_file,
#                                                                config)

#     def __call__(self):
#         test_transforms_pre_pre = [
#                 LoadImaged(keys=self.features),
#                 Lambdad(keys=self.features,
#                         func=lambda x: x.transpose(2, 1, 0, 3)),
#                 Resized(keys=self.features,
#                         spatial_size=(self.config['loaders']['Resize_height'],
#                                       self.config['loaders']['Resize_width'],
#                                       self.config['loaders']['Resize_depth'])),
#             ]
#         if self.config['model']['in_channels'] != 1:
#             test_transforms_intermed = [
#                 ConcatItemsd(keys=self.features, name='inputs')]
#         else:
#             test_transforms_intermed = [
#                 CopyItemsd(keys=self.features, times=1, names='inputs'),
#                 RepeatChanneld('inputs', 3)]
#         test_transforms_pre = Compose(test_transforms_pre_pre +
#                                       test_transforms_intermed)
#         test_transforms_post = Compose(
#             [

#                 ScaleIntensityd(keys='inputs'),
#                 NormalizeIntensityd(keys='inputs',
#                                     subtrahend=(0.43216, 0.394666, 0.37645),
#                                     divisor=(0.22803, 0.22145, 0.216989),
#                                     channel_wise=True),
#                 Lambdad(keys='inputs',
#                         func=lambda x: x.transpose(0, 3, 2, 1) if
#                         self.config['loaders']['model_dim_order'] == 'CH_D_H_W'
#                         else x),
#                 ToTensord(keys=self.features),
#             ])

#         # CHECK: for debug ###
#         # check_ds = monai.data.GridPatchDataset(
#         #     dataset=self.data,
#         #     #transform=test_transforms,
#         #     patch_size = (self.config['loaders']['height'],
#         #                       self.config['loaders']['width'],
#         #                       self.config['loaders']['depth']))
#         # check_loader = DataLoader(
#         #     check_ds,
#         #     batch_size=self.config['loaders']['batchSize'],
#         #     num_workers=self.config['loaders']['num_workers'],
#         #     collate_fn=list_data_collate)
#         # check_data = monai.utils.misc.first(check_loader)

#         # create a training data loader
#         # test_loader = monai.data.Dataset(data=self.data,
#         #                                  transform=test_transforms)
#         test_loader = SlidingClassificationDataset(
#             data=self.data,
#             config=self.config,
#             transform_pre=test_transforms_pre,
#             transform_post=test_transforms_post)
#         return test_loader
