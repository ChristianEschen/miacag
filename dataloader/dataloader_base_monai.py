import os
import numpy as np
import pandas as pd
from dataloader.dataloader_base import DataloaderTrain
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    RepeatChanneld,
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
    Resized,
    ToTensord,
    ConcatItemsd,
    CropForegroundd,
    CastToTyped,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    ToDeviced)


class base_monai_loader(DataloaderTrain):
    def __init__(self, df,
                 config):
        super(base_monai_loader, self).__init__(df,
                                                config)

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

    def get_input_features(self, csv, features='image_path'):
        if features == 'image_path':
            features = [col for col in
                        csv.columns.tolist() if col.startswith(features)]
        else:
            features = features
        return features

    # def set_feature_path(self, csv, features, image_path):
    #     for feature in features:
    #         csv[feature] = csv[feature].apply(
    #                 lambda x: os.path.join(image_path, x))
    #     return csv

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

    def getCopy1to3Channels(self):
        copy = RepeatChanneld(keys=self.features, repeats=3)
        return copy

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
            if self.config['task_type'] in ['representation_learning',
                                            "classification"]:
                keys_ = self.features
            elif self.config['task_type'] == "segmentation":
                keys_ = self.features + ["labels"]
            else:
                raise ValueError('not implemented')
            pad = SpatialPadd(
                keys=keys_,
                spatial_size=[self.config['loaders']['Crop_height'],
                              self.config['loaders']['Crop_width'],
                              self.config['loaders']['Crop_depth']])
        elif self.config['loaders']['mode'] == 'testing':
            if self.config['task_type'] == "classification":
                keys_ = self.features
                pad = SpatialPadd(
                    keys=keys_,
                    spatial_size=[self.config['loaders']['Crop_height'],
                                  self.config['loaders']['Crop_width'],
                                  self.config['loaders']['Crop_depth']])
            elif self.config['task_type'] == "segmentation":
                pad = Identityd(keys=self.features + ["labels"])
            else:
                raise ValueError('not implemented')
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
        elif self.config['loaders']['format'] == 'dicom':
            permute = Lambdad(
                    keys=self.features,
                    func=lambda x: x.transpose(1, 2, 0))
        else:
            permute = Identityd(keys=self.features + ["labels"])
        return permute

    def resampleORresize(self):
        if self.config['task_type'] in [
            'representation_learning', "classification"]:
            keys_ = self.features
            mode_ = tuple([
                         'bilinear' for i in
                         range(len(self.features))])
            if len(mode_) == 1:
                mode_ = mode_[0]
        elif self.config['task_type'] == "segmentation":
            keys_ = self.features + ["labels"]
            mode_ = tuple([
                         'bilinear' for i in
                         range(len(self.features))]+['nearest'])
        else:
            raise ValueError('not implemented')
        if self.config['loaders']['spatial_resize'] is True:
            resample = Spacingd(
                     keys=keys_,
                     pixdim=(self.config['loaders']['pixdim_height'],
                             self.config['loaders']['pixdim_width'],
                             self.config['loaders']['pixdim_depth']),
                     mode=mode_)
        else:
            resample = Resized(
                    keys=keys_,
                    spatial_size=(
                                self.config['loaders']['Resize_height'],
                                self.config['loaders']['Resize_width'],
                                self.config['loaders']['Resize_depth']))
        return resample

    def maybeToGpu(self, keys):
        if self.config['cpu'] == 'True':
            if self.config['task_type'] in ['representation_learning',
                                            "classification"]:
                device = ToDeviced(keys=keys, device="cpu")
            else:
                device = ToDeviced(
                    keys=keys + ["labels"], device="cpu")
        else:
            device = ToDeviced(
                keys=keys,
                device="cuda:{}".format(os.environ['LOCAL_RANK']))

        return device
