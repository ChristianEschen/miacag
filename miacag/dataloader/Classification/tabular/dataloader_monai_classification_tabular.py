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
from miacag.dataloader.dataloader_base_monai import base_monai_loader


# class base_monai_classification_loader(base_monai_loader):
#     def __init__(self, df, config):
#         super(base_monai_classification_loader, self).__init__(
#             df,
#             config)

#         self.features = self.get_input_features(self.df)
#         self.set_data_path(self.features)
#         self.data = self.df[self.features + [config['labels_names'], 'rowid']]
#         self.data = self.data.to_dict('records')


class train_monai_classification_loader(base_monai_loader):
    def __init__(self, df, config):
        super(base_monai_loader, self).__init__(
            df,
            config)
        self.getSampler()
        self.features = self.get_input_features(
            self.df,
            ['PositionerPrimaryAngle',
                'PositionerSecondaryAngle',
                'DistanceSourceToPatient',
                'DistanceSourceToDetector'])
        self.data = self.df[self.features + [config['labels_names'], 'rowid']]
        self.data = self.data.dropna()
        for col in self.features:
            self.data[col] \
                = (self.data[col] - self.data[col].mean() + 1e-10) \
                / (self.data[col].std(ddof=0) + 1e-10)
        self.data = self.data.to_dict('records')

    def __call__(self):
        # define transforms for image
        train_transforms = [
            ToTensord(keys=self.features),
            AddChanneld(keys=self.features),
            ConcatItemsd(keys=self.features, name='inputs'),
            self.maybeToGpu(['inputs']),
           ]
        train_transforms = Compose(train_transforms)
        train_transforms.set_random_state(seed=0)
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
                replace_rate=0.1,
                num_replace_workers=int(self.config['num_workers']/2))
        else:
            train_ds = monai.data.CacheDataset(
                data=self.data_par_train,
                transform=train_transforms,
                copy_cache=True,
                num_workers=self.config['num_workers'])


        return train_ds


class val_monai_classification_loader(base_monai_loader):
    def __init__(self, df, config):
        super(val_monai_classification_loader, self).__init__(df,
                                                              config)

        self.features = self.get_input_features(
            self.df,
            ['PositionerPrimaryAngle',
                'PositionerSecondaryAngle',
                'DistanceSourceToPatient',
                'DistanceSourceToDetector'])
        self.data = self.df[self.features + [config['labels_names']]]
        self.data = self.data.dropna()
        for col in self.features:
            self.data[col] \
                = (self.data[col] - self.data[col].mean() + 1e-10) \
                / (self.data[col].std(ddof=0) + 1e-10)
        self.data = self.data.to_dict('records')


    def __call__(self):
        val_transforms = [
            ToTensord(keys=self.features),
            AddChanneld(keys=self.features),
            ConcatItemsd(keys=self.features, name='inputs'),
            self.maybeToGpu(['inputs']),
           ]
       # if self.config['loaders']['mode'] != 'testing':
        
        #CHECK: for debug ###
        # check_ds = monai.data.Dataset(data=self.data,
        #                              transform=val_transforms)
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
        val_transforms = Compose(val_transforms)
        val_transforms.set_random_state(seed=0)
        if self.config['use_DDP'] == 'True':
            self.data_par_val = monai.data.partition_dataset(
                data=self.data,
                num_partitions=dist.get_world_size(),
                shuffle=False,
                even_divisible=True if self.config['loaders']['mode'] != 'testing' else False,
            )[dist.get_rank()]
            rowids = [i["rowid"] for i in self.data_par_val]
            if self.config['loaders']['mode'] != 'testing':
                if self.config['cache_num'] != 'None':
                    val_ds = monai.data.SmartCacheDataset(
                        data=self.data_par_val,
                        transform=val_transforms,
                        copy_cache=True,
                        cache_num=self.config['cache_num'],
                        num_init_workers=int(self.config['num_workers']/2),
                        replace_rate=self.config['replace_rate'],
                        num_replace_workers=int(self.config['num_workers']/2))
                else:
                    val_ds = monai.data.CacheDataset(
                        data=self.data_par_val,
                        transform=val_transforms,
                        copy_cache=True,
                        num_workers=self.config['num_workers'])
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


class val_monai_classification_loader_SW(base_monai_loader):
    def __init__(self, df, config):
        super(val_monai_classification_loader_SW, self).__init__(df,
                                                              config)
        self.features = self.get_input_features(self.df)
        self.set_data_path(self.features)
        self.data = self.df[self.features + [config['labels_names'], 'rowid']]
        self.data = self.data.to_dict('records')

    def __call__(self):
        val_transforms = [
            ToTensord(keys=self.features),
            AddChanneld(keys=self.features),
            ConcatItemsd(keys=self.features, name='inputs'),
            self.maybeToGpu(['inputs']),
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
                    replace_rate=self.config['replace_rate'],
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
