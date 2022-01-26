from torch.utils.data import DataLoader
import monai
from monai.data import list_data_collate, pad_list_data_collate
import torch.distributed as dist
from monai.transforms import (
    AsChannelFirstd,
    RepeatChanneld,
    AddChanneld,
    EnsureChannelFirstD,
    RandAffined,
    DeleteItemsd,
    AsDiscrete,
    CenterSpatialCropd,
    ScaleIntensityd,
    RandTorchVisiond,
    RandRotated,
    RandZoomd,
    Compose,
    Rotate90d,
    SqueezeDimd,
    ToDeviced,
    RandLambdad,
    CopyItemsd,
    LoadImaged,
    EnsureTyped,
    RandSpatialCropSamplesd,
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
        self.set_data_path(self.features)
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
                DeleteItemsd(keys=self.features[0]+"_meta_dict.[0-9]\\|[0-9]", use_re=True),
                self.getMaybePad(),
                self.getCopy1to3Channels(),
                ScaleIntensityd(keys=self.features),
                NormalizeIntensityd(keys=self.features,
                                    channel_wise=True),
                EnsureTyped(keys=self.features, data_type='tensor'),
                self.maybeToGpu(self.features),
                RandAffined(
                    keys=self.features,
                    mode="bilinear",
                    prob=0.2,
                    spatial_size=(self.config['loaders']['Crop_height'],
                                  self.config['loaders']['Crop_width'],
                                  self.config['loaders']['Crop_depth']),
                    translate_range=(5, 5, 1),
                    rotate_range=(0, 0, 0.17),
                    scale_range=(0.15, 0.15, 0),
                    padding_mode="zeros"),
                RandZoomd(keys=self.features,
                          prob=0.2,
                          min_zoom=(1, 1, 0.5),
                          max_zoom=(1, 1, 1.5),
                          mode='nearest'),
                ConcatItemsd(keys=self.features, name='inputs'),
                DeleteItemsd(keys=self.features),
                ]
        train_transforms = Compose(train_transforms)
        # CHECK: for debug ###
        # check_ds = monai.data.Dataset(data=self.data,
        #                              transform=train_transforms)
        # check_loader = DataLoader(
        #     check_ds,
        #     batch_size=self.config['loaders']['batchSize'],
        #     num_workers=0,
        #     collate_fn=list_data_collate
        #     )
        # check_data = monai.utils.misc.first(check_loader)
        # img = check_data['inputs'].cpu().numpy()
        # import matplotlib.pyplot as plt
        # import numpy as np
        # for i in range(9, img.shape[-1]):
        #     img2d = img[0,0,:,:,i]
        #     fig_train = plt.figure()
        #     plt.imshow(img2d, cmap="gray", interpolation="None")
        #     plt.show()

        self.data_par_train = monai.data.partition_dataset(
            data=self.data,
            num_partitions=dist.get_world_size(),
            shuffle=True,
            even_divisible=True,
        )[dist.get_rank()]

        # create a training data loader
        if self.config['cache_num'] != 'None':
            train_ds = monai.data.SmartCacheDataset(
                data=self.data_par_train,
                transform=train_transforms,
                copy_cache=True,
                cache_num=self.config['cache_num'],
                num_init_workers=int(self.config['num_workers']/2),
                replace_rate=self.config['cache_rate'],
                num_replace_workers=int(self.config['num_workers']/2))
        else:
            train_ds = monai.data.CacheDataset(
                data=self.data_par_train,
                transform=train_transforms,
                copy_cache=True,
                num_workers=self.config['num_workers'])


        return train_ds


class val_monai_classification_loader(base_monai_classification_loader):
    def __init__(self, df, config):
        super(val_monai_classification_loader, self).__init__(df,
                                                              config)

    def __call__(self):
        val_transforms = [
                LoadImaged(keys=self.features),
                EnsureChannelFirstD(keys=self.features),
                DeleteItemsd(keys=self.features[0]+"_meta_dict.[0-9]\\|[0-9]", use_re=True),
                self.resampleORresize(),
                self.getMaybePad(),
                self.getCopy1to3Channels(),
                ScaleIntensityd(keys=self.features),
                NormalizeIntensityd(keys=self.features,
                                    channel_wise=True),
                EnsureTyped(keys=self.features, data_type='tensor'),
                self.maybeToGpu(self.features),
                RandSpatialCropd(keys=self.features,
                                 roi_size=[
                                     self.config['loaders']['Crop_height'],
                                     self.config['loaders']['Crop_width'],
                                     self.config['loaders']['Crop_depth']],
                                 random_size=False),
                ConcatItemsd(keys=self.features, name='inputs'),
                DeleteItemsd(keys=self.features),
                ]
        val_transforms = Compose(val_transforms)
        if self.config['use_DDP'] == 'True':
            self.data_par_val = monai.data.partition_dataset(
                data=self.data,
                num_partitions=dist.get_world_size(),
                shuffle=True,
                even_divisible=True,
            )[dist.get_rank()]
            if self.config['cache_num'] != 'None':
                val_ds = monai.data.SmartCacheDataset(
                    data=self.data_par_val,
                    transform=val_transforms,
                    copy_cache=True,
                    cache_num=self.config['cache_num'],
                    num_init_workers=int(self.config['num_workers']/2),
                    replace_rate=self.config['cache_rate'],
                    num_replace_workers=int(self.config['num_workers']/2))
            else:
                val_ds = monai.data.CacheDataset(
                    data=self.data_par_val,
                    transform=val_transforms,
                    copy_cache=True,
                    num_workers=self.config['num_workers'])
        else:
            val_ds = monai.data.Dataset(
                data=self.data,
                transform=val_transforms)

        return val_ds


class val_monai_classification_loader_SW(base_monai_classification_loader):
    def __init__(self, df, config):
        super(val_monai_classification_loader_SW, self).__init__(df,
                                                              config)

    def __call__(self):
        val_transforms = [
                LoadImaged(keys=self.features),
                EnsureChannelFirstD(keys=self.features),
                self.resampleORresize(),
                self.getMaybePad(),
                self.getCopy1to3Channels(),
                ScaleIntensityd(keys=self.features),
                NormalizeIntensityd(keys=self.features,
                                    channel_wise=True),
                EnsureTyped(keys=self.features, data_type='tensor'),

                ConcatItemsd(keys=self.features, name='inputs'),
                ]
        val_transforms = Compose(val_transforms)
        if self.config['use_DDP'] == 'True':
            self.data_par_val = monai.data.partition_dataset(
                data=self.data,
                num_partitions=dist.get_world_size(),
                shuffle=True,
                even_divisible=True,
            )[dist.get_rank()]
            if self.config['cache_num'] != 'None':
                val_ds = monai.data.SmartCacheDataset(
                    data=self.data_par_val,
                    transform=val_transforms,
                    copy_cache=True,
                    cache_num=self.config['cache_num'],
                    num_init_workers=int(self.config['num_workers']/2),
                    replace_rate=self.config['cache_rate'],
                    num_replace_workers=int(self.config['num_workers']/2))
            else:
                val_ds = monai.data.CacheDataset(
                    data=self.data_par_val,
                    transform=val_transforms,
                    copy_cache=True,
                    num_workers=self.config['num_workers'])
        else:
            val_ds = monai.data.Dataset(
                data=self.data,
                transform=val_transforms)

            self.data = monai.data.partition_dataset(
                data=self.data,
                num_partitions=dist.get_world_size(),
                shuffle=False,
                even_divisible=True,
            )[dist.get_rank()]
        val_ds = monai.data.Dataset(
            data=self.data,
            transform=val_transforms)
        return val_ds
