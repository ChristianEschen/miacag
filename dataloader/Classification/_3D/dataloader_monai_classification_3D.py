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
                ToTensord(keys='inputs'),
                DeleteItemsd(keys=self.features[0]+"_meta_dict.0008\\|[0-9]", use_re=True),
                DeleteItemsd(keys=self.features[0]+"_meta_dict.0020\\|[0-9]", use_re=True),
                DeleteItemsd(keys=self.features[0]+"_meta_dict.0028\\|[0-9]", use_re=True),
                #DeleteItemsd(keys=self.features[0]+"_meta_dict.\\[0-9]|[0-9]", use_re=True),
                ]
        train_transforms = Compose(train_transforms)
        # CHECK: for debug ###
        check_ds = monai.data.Dataset(data=self.data,
                                     transform=train_transforms)
        check_loader = DataLoader(
            check_ds,
            batch_size=self.config['loaders']['batchSize'],
            num_workers=self.config['num_workers'],
            collate_fn=list_data_collate)
        check_data = monai.utils.misc.first(check_loader)
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
                ToTensord(keys='inputs'),
                DeleteItemsd(keys=self.features[0]+"_meta_dict.0008\\|[0-9]", use_re=True),
                DeleteItemsd(keys=self.features[0]+"_meta_dict.0020\\|[0-9]", use_re=True),
                DeleteItemsd(keys=self.features[0]+"_meta_dict.0028\\|[0-9]", use_re=True),
                #DeleteItemsd(keys=self.features[0]+"_meta_dict.\\[0-9]|[0-9]", use_re=True)
                ]

        val_transforms = Compose(val_transforms)
        val_loader = monai.data.Dataset(data=self.data,
                                        transform=val_transforms)

        return val_loader
