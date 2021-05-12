import os
import numpy as np
import pandas as pd
from dataloader.dataloader_base import DataloaderTrain
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

class base_monai_loader(DataloaderTrain):
    def __init__(self, image_path, csv_path_file,
                 config, use_complete_data):
        super(base_monai_loader, self).__init__(image_path,
                                                csv_path_file,
                                                config,
                                                use_complete_data)

    def get_input_flow(self, csv):
        features = [col for col in
                    csv.columns.tolist() if col.startswith('flow')]
        return features

    def set_flow_path(self, csv, features, image_path):
        feature_paths = features
        for feature in feature_paths:
            csv[feature] = csv[feature].apply(
                    lambda x: os.path.join(image_path, x))
        return csv

    def get_input_features(self, csv):
        features = [col for col in
                    csv.columns.tolist() if col.startswith('image')]
        return features

    def set_feature_path(self, csv, features, image_path):
        feature_paths = features
        for feature in feature_paths:
            csv[feature] = csv[feature].apply(
                    lambda x: os.path.join(image_path, x))
        return csv

    def getMaybeForegroundCropper(self):
        if self.config['loaders']['CropForeGround'] is False:
            fCropper = Identityd(keys=self.features + ["labels"])
        else:
            fCropper = CropForegroundd(keys=self.features + ["labels"],
                                       source_key="labels")
        return fCropper

    def getMaybeClip(self):
        if self.config['loaders']['isCT'] is False:
            clip = Identityd(keys=self.features + ["labels"])
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
        if self.config['loaders']['val_method']['type'] == "patches":
            randCrop = RandCropByPosNegLabeld(
                    keys=self.features + ["labels"],
                    label_key="labels",
                    spatial_size=[self.config['loaders']['depth'],
                                  self.config['loaders']['height'],
                                  self.config['loaders']['width']],
                    pos=1, neg=1, num_samples=1)
        elif self.config['loaders']['val_method']['type'] == "sliding_window":
            randCrop = Identityd(keys=self.features + ["labels"])
        else:
            raise ValueError("Invalid val_method %s" % repr(
             self.config['loaders']['val_method']['type']))
        return randCrop

    def getMaybePad(self):
        if self.config['loaders']['mode'] == 'training':
            pad = SpatialPadd(
                keys=self.features + ["labels"],
                spatial_size=[self.config['loaders']['depth'],
                              self.config['loaders']['height'],
                              self.config['loaders']['width']])
        elif self.config['loaders']['mode'] == 'testing':
            pad = Identityd(keys=self.features + ["labels"])
        else:
            raise ValueError("Invalid mode %s" % repr(
             self.config['loaders']['mode']))
        return pad

    def maybeReorder_z_dim(self):
        if self.config['loaders']['format'] == 'nifty':
            if self.config['model']['dimension'] == '2D+T':
                permute = Lambdad(
                    keys=self.features,
                    func=lambda x: x.transpose(2, 3, 0, 1))
            elif self.config['model']['dimension'] == '3D':
                permute = Lambdad(
                    keys=self.features + ["labels"],
                    func=lambda x: np.transpose(x, (0, 3, 1, 2)))
            else:
                raise ValueError('data model dimension not understood')
        else:
            permute = Identityd(keys=self.features + ["labels"])
        return permute
