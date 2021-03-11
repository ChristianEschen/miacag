import os
import pandas as pd
import numpy as np
import monai
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    RandFlipd,
    RandCropByPosNegLabeld,
    CopyItemsd,
    RandZoomd,
    # ScaleIntensityRanged,
    RandAdjustContrastd,
    RandRotate90d,
    Spacingd,
    Identityd,
    SpatialPadd,
    Lambdad,
    ToTensord,
    ConcatItemsd,
    CropForegroundd,
    CastToTyped,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd)


class base_monai_segmentation_loader():
    def __init__(self, image_path, csv_path_file, config):
        self.image_path = image_path
        self.csv = pd.read_csv(csv_path_file)
        self.features = self.get_input_features(self.csv)
        self.csv = self.set_feature_path(self.csv,
                                         self.features,
                                         self.image_path)
        self.data = self.csv.to_dict('records')
        self.image_data = self.csv[self.features+['seg']].to_dict('records')
        self.config = config

    def get_input_features(self, csv):
        features = [col for col in
                    self.csv.columns.tolist() if col.startswith('image')]
        return features

    def set_feature_path(self, csv, features, image_path):
        feature_paths = features + ['seg']
        for feature in feature_paths:
            csv[feature] = csv[feature].apply(
                    lambda x: os.path.join(image_path, x))
        return csv

    def getMaybeForegroundCropper(self):
        if self.config['loaders']['CropForeGround'] is False:
            fCropper = Identityd(keys=self.features + ["seg"])
        else:
            fCropper = CropForegroundd(keys=self.features + ["seg"],
                                       source_key="seg")
        return fCropper

    def getMaybeClip(self):
        if self.config['loaders']['isCT'] is False:
            clip = Identityd(keys=self.features + ["seg"])
        else:
            clip = Lambdad(keys=self.features,
                           func=lambda x: np.clip(
                            x, self.config['loaders']['minPercentile'],
                            self.config['loaders']['maxPercentile']))
        return clip

    def getNormalization(self):
        if self.config['loaders']['isCT'] is False:
            Normalizer = NormalizeIntensityd(self.features, nonzero=True,
                                             channel_wise=True)
        else:
            Normalizer = NormalizeIntensityd(
                keys=self.features,
                subtrahend=self.config['loaders']['subtrahend'],
                divisor=self.config['loaders']['divisor'],
                channel_wise=True)

        return Normalizer

    def getMaybeConcat(self):
        if self.config['model']['in_channels'] != 1:
            concat = ConcatItemsd(
                keys=self.features, name='inputs')
        else:
            concat = CopyItemsd(keys=self.features,
                                times=1, names='inputs')
        return concat

    def getMaybeRandCrop(self):
        if self.config['loaders']['val_method'] != "sliding_window":
            randCrop = RandCropByPosNegLabeld(
                    keys=self.features + ["seg"],
                    label_key="seg",
                    spatial_size=[self.config['loaders']['height'],
                                  self.config['loaders']['width'],
                                  self.config['loaders']['depth']],
                    pos=1, neg=1, num_samples=1)
        else:
            randCrop = Identityd(keys=self.features + ["seg"])
        return randCrop


class train_monai_segmentation_loader(base_monai_segmentation_loader):
    def __init__(self, image_path, csv_path_file, config):
        super(train_monai_segmentation_loader, self).__init__(image_path,
                                                              csv_path_file,
                                                              config)

    def __call__(self):
        train_transforms = [
                LoadImaged(keys=self.features + ["seg"]),
                AddChanneld(keys=self.features),
                ConvertToMultiChannel(keys="seg"),
                self.getMaybeForegroundCropper(),
                Spacingd(
                     keys=self.features + ["seg"],
                     pixdim=(self.config['loaders']['pixdim_height'],
                             self.config['loaders']['pixdim_width'],
                             self.config['loaders']['pixdim_depth']),
                     mode=tuple([
                         'bilinear' for i in
                         range(len(self.features))]+['nearest'])),
                SpatialPadd(keys=self.features + ["seg"],
                            spatial_size=[self.config['loaders']['height'],
                                          self.config['loaders']['width'],
                                          self.config['loaders']['depth']]),
                self.getMaybeClip(),
                self.getNormalization(),
                NormalizeIntensityd(keys=self.features, nonzero=True,
                                    channel_wise=True),
                RandCropByPosNegLabeld(
                    keys=self.features + ["seg"],
                    label_key="seg",
                    spatial_size=[self.config['loaders']['height'],
                                  self.config['loaders']['width'],
                                  self.config['loaders']['depth']],
                    pos=1, neg=1, num_samples=1),
                RandRotate90d(keys=self.features + ["seg"], prob=0.1, max_k=3,
                              spatial_axes=(0, 1), allow_missing_keys=False),
                RandZoomd(
                    keys=self.features + ["seg"],
                    min_zoom=0.9,
                    max_zoom=1.2,
                    mode=("trilinear", "nearest"),
                    align_corners=(True, None),
                    prob=0.16,
                ),
                CastToTyped(keys=self.features + ["seg"],
                            dtype=(np.float32, np.uint8)),
                RandGaussianNoised(keys=self.features, std=0.01, prob=0.15),
                RandGaussianSmoothd(
                    keys=self.features,
                    sigma_x=(0.5, 1.15),
                    sigma_y=(0.5, 1.15),
                    sigma_z=(0.5, 1.15),
                    prob=0.15,
                ),
                RandAdjustContrastd(keys=self.features,
                                    gamma=(0.7, 1.5), prob=0.1),
                RandScaleIntensityd(keys=self.features,
                                    factors=0.3,
                                    prob=0.15),
                RandFlipd(self.features + ["seg"],
                          spatial_axis=[0, 1, 2], prob=0.5),
                self.getMaybeConcat(),
                Lambdad(keys=['inputs', 'seg'],
                        func=lambda x: np.squeeze(x, axis=3) if
                        self.config['model']['dimensions'] == 2
                        else x),
                ToTensord(keys=["inputs", "seg"]),
                            ]
        train_transforms = Compose(train_transforms)
        # CHECK: for debug
        # check_ds = monai.data.Dataset(data=self.data,
        #                               transform=train_transforms)
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
        val_transforms = [
                LoadImaged(keys=self.features + ["seg"]),
                AddChanneld(keys=self.features),
                ConvertToMultiChannel(keys="seg"),
                self.getMaybeForegroundCropper(),
                Spacingd(
                     keys=self.features + ["seg"],
                     pixdim=(self.config['loaders']['pixdim_height'],
                             self.config['loaders']['pixdim_width'],
                             self.config['loaders']['pixdim_depth']),
                     mode=tuple([
                         'bilinear' for i in
                         range(len(self.features))]+['nearest'])),
                SpatialPadd(keys=self.features + ["seg"],
                            spatial_size=[self.config['loaders']['height'],
                                          self.config['loaders']['width'],
                                          self.config['loaders']['depth']]),
                self.getMaybeClip(),
                self.getNormalization(),
                NormalizeIntensityd(keys=self.features, nonzero=True,
                                    channel_wise=True),
                self.getMaybeRandCrop(),
                CastToTyped(keys=self.features + ["seg"],
                            dtype=(np.float32, np.uint8)),
                self.getMaybeConcat(),
                Lambdad(keys=['inputs', 'seg'],
                        func=lambda x: np.squeeze(x, axis=3) if
                        self.config['model']['dimensions'] == 2
                        else x),
                ToTensord(keys=["inputs", "seg"]),
                            ]
        val_transforms = Compose(val_transforms)

        val_loader = monai.data.CacheDataset(data=self.image_data,
                                             transform=val_transforms)

        return val_loader


class ConvertToMultiChannel(MapTransform):
    """
    Convert labels to multi channels based.
    Background is also encoded
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            for label_idx in np.unique(d[key]):
                label_bool = d[key] == int(label_idx)
                result.append(label_bool)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d
