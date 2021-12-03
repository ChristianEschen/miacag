from torch.utils.data import DataLoader
import monai
from monai.data import list_data_collate, pad_list_data_collate
from monai.transforms import (
    AsChannelFirstd,
    RepeatChanneld,
    AddChanneld,
    EnsureChannelFirstD,
    DeleteItemsd,
    AsDiscrete,
    CenterSpatialCropd,
    ScaleIntensityd,
    RandTorchVisiond,
    Compose,
    SqueezeDimd,
    ToDeviced,
    RandLambdad,
    CopyItemsd,
    LoadImaged,
    EnsureTyped,
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
                DeleteItemsd(keys=self.features[0]+"_meta_dict.[0-9]\\|[0-9]", use_re=True),
                self.getCopy1to3Channels(),
                ScaleIntensityd(keys=self.features),
                NormalizeIntensityd(keys=self.features,
                                    channel_wise=True),
                EnsureTyped(keys=self.features, data_type='tensor'),
                #ToDeviced(keys=self.features, device="cuda:0"),
                #self.maybeToGpu(),
                RandSpatialCropd(keys=self.features,
                                 roi_size=[
                                     self.config['loaders']['Crop_height'],
                                     self.config['loaders']['Crop_width'],
                                     self.config['loaders']['Crop_depth']],
                                 random_size=False),
               # CopyItemsd(keys=self.features, times=1, names='inputs'),
                ConcatItemsd(keys=self.features, name='inputs'),
                DeleteItemsd(keys=self.features),
                ]
        train_transforms = Compose(train_transforms)
        # CHECK: for debug ###
        # check_ds = monai.data.CacheDataset(data=self.data,
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
        if self.config['cache_num'] != 'None':
            train_loader = monai.data.SmartCacheDataset(
                data=self.data,
                transform=train_transforms,
                copy_cache=True,
                cache_num=self.config['cache_num'],
                num_init_workers=int(self.config['num_workers']/2),
                replace_rate=0.25,
                num_replace_workers=int(self.config['num_workers']/2))
        else:
            train_loader = monai.data.CacheDataset(
                data=self.data,
                transform=train_transforms,
                copy_cache=True,
                num_workers=self.config['num_workers'])

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
                DeleteItemsd(keys=self.features[0]+"_meta_dict.[0-9]\\|[0-9]", use_re=True),
                self.getCopy1to3Channels(),
                ScaleIntensityd(keys=self.features),
                NormalizeIntensityd(keys=self.features,
                                    channel_wise=True),
                EnsureTyped(keys=self.features, data_type='tensor'),
                # ToDeviced(keys=self.features, device="cuda:0"),
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
        # val_loader = monai.data.CacheDataset(
        #     data=self.data,
        #     transform=val_transforms,
        #     copy_cache=False,
        #     #  replace_rate=0.25,
        #     num_workers=self.config['num_workers'])
        val_loader = monai.data.Dataset(
            data=self.data,
            transform=val_transforms)
        return val_loader
