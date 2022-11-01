from fnmatch import translate
import os
import numpy as np
import pandas as pd
from miacag.dataloader.dataloader_base import DataloaderBase
from monai.transforms import Transform
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
    RandAffined,
    DeleteItemsd,
    # ScaleIntensityRanged,
    RandAdjustContrastd,
    RandRotate90d,
    RandSpatialCropd,
    CenterSpatialCropd,
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
    DataStatsd,
    ToDeviced)


class base_monai_loader(DataloaderBase):
    def __init__(self, df,
                 config):
        super(base_monai_loader, self).__init__(df,
                                                config)

    def get_input_flow(self, csv):
        features = [col for col in
                    csv.columns.tolist() if col.startswith('flow')]
        return features

    def set_flow_path(self, csv, features, DcmPathFlatten):
        feature_paths = features
        for feature in feature_paths:
            csv[feature] = csv[feature].apply(
                    lambda x: os.path.join(DcmPathFlatten, x))
        return csv

    def set_data_path(self, features):
        for feature in features:
            self.df[feature] = self.df[feature].apply(
                        lambda x: os.path.join(self.config['DataSetPath'], x))

    def get_input_features(self, csv, features='DcmPathFlatten'):
        if features == 'DcmPathFlatten':
            features = [col for col in
                        csv.columns.tolist() if col.startswith(features)]
        else:
            features = features
        return features

    def getMaybeForegroundCropper(self):
        if self.config['loaders']['CropForeGround'] is False:
            fCropper = Identityd(keys=self.features + [self.config["labels_names"]])
        else:
            fCropper = CropForegroundd(keys=self.features + [self.config["labels_names"]],
                                       source_key=self.config["labels_names"])
        return fCropper

    def getMaybeClip(self):
        if self.config['loaders']['isCT'] is False:
            clip = Identityd(keys=self.features + [self.config["labels_names"]])
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

    def getClipChannels(self):
        clip = Lambdad(
                    keys=self.features,
                    func=lambda x: x[0:3, :, :, :])
        return clip

    def getMaybeRandCrop(self):
        if self.config['loaders']['val_method']['type'] == "patches":
            randCrop = RandCropByPosNegLabeld(
                    keys=self.features + [self.config["labels_names"]],
                    label_key=self.config["labels_names"],
                    spatial_size=[self.config['loaders']['depth'],
                                  self.config['loaders']['height'],
                                  self.config['loaders']['width']],
                    pos=1, neg=1, num_samples=1)
        elif self.config['loaders']['val_method']['type'] == "sliding_window":
            randCrop = Identityd(keys=self.features + [self.config["labels_names"]])
        else:
            raise ValueError("Invalid val_method %s" % repr(
             self.config['loaders']['val_method']['type']))
        return randCrop

    def getMaybePad(self):
        if self.config['loaders']['mode'] == 'training':
            if self.config['task_type'] in ["classification",
                                            "regression",
                                             "mil_classification"]:
                keys_ = self.features
            elif self.config['task_type'] == "segmentation":
                keys_ = self.features + [self.config["labels_names"]]
            else:
                raise ValueError('not implemented')
            pad = SpatialPadd(
                keys=keys_,
                spatial_size=[self.config['loaders']['Crop_height'],
                              self.config['loaders']['Crop_width'],
                              self.config['loaders']['Crop_depth']])
        elif self.config['loaders']['mode'] in ['testing', 'prediction']:
            if self.config['task_type'] in ["classification",
                                            "regression",
                                            "mil_classification"]:
                keys_ = self.features
                pad = SpatialPadd(
                    keys=keys_,
                    spatial_size=[self.config['loaders']['Crop_height'],
                                  self.config['loaders']['Crop_width'],
                                  self.config['loaders']['Crop_depth']])
            elif self.config['task_type'] == "segmentation":
                pad = Identityd(keys=self.features + [self.config["labels_names"]])
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
                    keys=self.features + [self.config["labels_names"]],
                    func=lambda x: np.transpose(x, (0, 3, 1, 2)))
            else:
                raise ValueError('data model dimension not understood')
        elif self.config['loaders']['format'] == 'dicom':
            permute = Lambdad(
                    keys=self.features,
                    func=lambda x: x.transpose(1, 2, 0))
        else:
            permute = Identityd(keys=self.features + [self.config["labels_names"]])
        return permute

    def resampleORresize(self):
        if self.config['task_type'] in ["classification", "regression", "mil_classification"]:
            keys_ = self.features
            mode_ = tuple([
                         'bilinear' for i in
                         range(len(self.features))])
            if len(mode_) == 1:
                mode_ = mode_[0]
        elif self.config['task_type'] == "segmentation":
            keys_ = self.features + [self.config["labels_names"]]
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
            if self.config['task_type'] in ["mil_classification",
                                            "classification",
                                            "regression"]:
                device = ToDeviced(keys=keys, device="cpu")
            else:
                device = ToDeviced(
                    keys=keys + [self.config["labels_names"]], device="cpu")

        else:
            if self.config['use_DDP'] == 'False':
                device = Identityd(keys=keys + [self.config["labels_names"]])
            else:
                device = ToDeviced(
                    keys=keys,
                    device="cuda:{}".format(os.environ['LOCAL_RANK']))

        return device

    def maybeCenterCrop(self, features):
        if self.config['loaders']['mode'] == 'training':
            crop = CenterSpatialCropd(
                keys=features,
                roi_size=[
                    self.config['loaders']['Crop_height'],
                    self.config['loaders']['Crop_width'],
                    self.config['loaders']['Crop_depth']])
        else:
            # before it was equal true. correct now?
            if (self.config['loaders']['val_method']['saliency'] == 'True' and
                    self.config['loaders']['mode'] == 'prediction'):

                crop = CenterSpatialCropd(
                    keys=features,
                    roi_size=[
                        self.config['loaders']['Crop_height'],
                        self.config['loaders']['Crop_width'],
                        self.config['loaders']['Crop_depth']])
            else:
                crop = RandSpatialCropd(
                    keys=features,
                    roi_size=[
                        self.config['loaders']['Crop_height'],
                        self.config['loaders']['Crop_width'],
                        self.config['loaders']['Crop_depth']],
                    random_size=False)
        return crop

    def maybeCenterCropMIL(self, features):
        if self.config['loaders']['mode'] == 'training':

            crop = MyCenterCropd(
                keys=features,
                roi_size=[
                    -1,
                    self.config['loaders']['Crop_height'],
                    self.config['loaders']['Crop_width'],
                    self.config['loaders']['Crop_depth']])

        else:
        #    if self.config['loaders']['val_method']['saliency'] == 'True':
            # crop = MyCenterCropd(
            #     keys=features,
            #     roi_size=[
            #         -1,
            #         -1,
            #         self.config['loaders']['Crop_height'],
            #         self.config['loaders']['Crop_width'],
            #         self.config['loaders']['Crop_depth']])
            crop = CenterSpatialCropd(
                    keys=features,
                    roi_size=[
                        self.config['loaders']['Crop_height'],
                        self.config['loaders']['Crop_width'],
                        self.config['loaders']['Crop_depth']])
            # else:
            #     crop = RandSpatialCropd(
            #         keys=features,
            #         roi_size=[
            #             -1,
            #             self.config['loaders']['Crop_height'],
            #             self.config['loaders']['Crop_width'],
            #             self.config['loaders']['Crop_depth']],
            #         random_size=False)
        return crop

    def maybeTranslate(self):
        if self.config['loaders']['translate'] == 'True':
            translation = RandAffined(
                    keys=self.features,
                    mode="bilinear",
                    prob=0.2,
                    spatial_size=(self.config['loaders']['Resize_height'],
                                  self.config['loaders']['Resize_width'],
                                  self.config['loaders']['Resize_depth']),
                    translate_range=(
                         int(0.22*self.config['loaders']['Resize_height']),
                         int(0.22*self.config['loaders']['Resize_width']),
                         int(0.5*self.config['loaders']['Resize_depth'])),
                    padding_mode="zeros")
        else:
            translation = Identityd(keys=self.features)
        return translation

    def maybeSpatialScaling(self):
        if self.config['loaders']['spatial_scaling'] == 'True':

            spatial_scale = RandAffined(
                        keys=self.features,
                        mode="bilinear",
                        prob=0.2,
                        spatial_size=(self.config['loaders']['Resize_height'],
                                      self.config['loaders']['Resize_width'],
                                      self.config['loaders']['Resize_depth']),
                        scale_range=(0.15, 0.15, 0),
                        padding_mode="zeros")
        else:
            spatial_scale = Identityd(keys=self.features)
        return spatial_scale

    def maybeTemporalScaling(self):
        if self.config['loaders']['temporal_scaling'] == 'True':

            temporal_scaling = RandZoomd(
                keys=self.features,
                prob=0.2,
                min_zoom=(1, 1, 0.5),
                max_zoom=(1, 1, 1.5),
                mode='nearest')
        else:
            temporal_scaling = Identityd(keys=self.features)
        return temporal_scaling

    def maybeRotate(self):
        if self.config['loaders']['rotate'] == 'True':
            rotate = RandAffined(
                        keys=self.features,
                        mode="bilinear",
                        prob=0.2,
                        rotate_range=(0, 0, 0.17),
                        spatial_size=(self.config['loaders']['Resize_height'],
                                      self.config['loaders']['Resize_width'],
                                      self.config['loaders']['Resize_depth']),
                        padding_mode="zeros")
        else:
            rotate = Identityd(keys=self.features)
        return rotate

    def maybeNormalize(self):
        if self.config['model']['backbone'] in [
            'x3d_s', 'slowfast8x8', "mvit_base_16x4", 'mvit_base_32x3', 'debug_3d']:
            normalize = NormalizeIntensityd(
                keys=self.features,
                subtrahend=(0.45, 0.45, 0.45),#(0.43216, 0.394666, 0.37645),
                divisor=(0.225, 0.225, 0.225),#(0.22803, 0.22145, 0.216989),
                channel_wise=True)
        elif self.config['model']['backbone'] == 'r2plus1_18':
            normalize = NormalizeIntensityd(
                keys=self.features,
                subtrahend=(0.43216, 0.394666, 0.37645),
                divisor=(0.22803, 0.22145, 0.216989),
                channel_wise=True)
        elif self.config['model']['backbone'] in ['r50']:
            normalize = NormalizeIntensityd(
                keys=self.features,
                subtrahend=(0.485, 0.456, 0.406),
                divisor=(0.229, 0.224, 0.225),
                channel_wise=True)
        else:
            raise ValueError('not implemented')

        return normalize

    def CropTemporal(self):
        if self.config['task_type'] in ["classification",
                                        "regression"]:
            crop = self.CropTemporalNormal()
        elif self.config['task_type'] in ["mil_classification"]:
            if self.config['model']['dimension'] == '2D+T':
                crop = self.CropTemporalMIL()
            elif self.config['model']['dimension'] == '2D':
                crop = self.CropTemporalMIL2d()
            else:
                raise ValueError('model dimension is not implemented')
        else:
            raise ValueError('task type is not implemented')
        return crop

    def CropTemporalNormal(self):
        crop = RandSpatialCropd(
                keys=self.features,
                roi_size=[
                    self.config['loaders']['Crop_height'],
                    self.config['loaders']['Crop_width'],
                    self.config['loaders']['Crop_depth']],
                random_size=False)
        return crop

    def CropTemporalMIL(self):
        crop = RandSpatialCropd(
            keys=self.features,
            roi_size=[
                -1,
                self.config['loaders']['Crop_height'],
                self.config['loaders']['Crop_width'],
                self.config['loaders']['Crop_depth']],
            random_size=False)
        return crop

    def CropTemporalMIL2d(self):
        crop = RandSpatialCropd(
            keys=self.features,
            roi_size=[
                -1,
                self.config['loaders']['Crop_height'],
                self.config['loaders']['Crop_width']],
            random_size=False)
        return crop

    def maybeDeleteFeatures(self):
       # if self.config['loaders']['mode'] != 'predictions':
        if (self.config['loaders']['val_method']['saliency'] == 'True' and
                self.config['loaders']['mode'] == 'prediction'):
            deleter = DeleteItemsd(keys=self.features)
        else:
            deleter = Identityd(keys=self.features)

        return deleter

    def maybeDeleteMeta(self):
        if (self.config['loaders']['val_method']['saliency'] == 'True' and
                self.config['loaders']['mode'] in ['prediction']):
            
            deleter = Identityd(keys=self.features)
        else:
            deleter = DeleteItemsd(
                keys=self.features[0]+"_meta_dict.[0-9]\\|[0-9]", use_re=True)
        return deleter


class LabelEncodeIntegerGraded(Transform):
    """
    Convert an integer label to encoded array representation of length num_classes,
    with 1 filled in up to label index, and 0 otherwise. For example for num_classes=5,
    embedding of 2 -> (1,1,0,0,0)
    Args:
        num_classes: the number of classes to convert to encoded format.
        keys: keys of the corresponding items to be transformed
            Defaults to ``['label']``.
    """

    def __init__(self, num_classes, keys=["label"]):
        super().__init__()
        self.keys = keys
        self.num_classes = num_classes

    def __call__(self, data):

        d = dict(data)
        for key in self.keys:
            label = int(d[key])

            lz = np.zeros(self.num_classes, dtype=np.float32)
            lz[:label] = 1.0
            # alternative oneliner lz=(np.arange(self.num_classes)<int(label)).astype(np.float32) #same oneliner
            d[key] = lz

        return d

import contextlib
from copy import deepcopy
from enum import Enum
from itertools import chain
from math import ceil, floor
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from monai.config import IndexSelection, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.transforms.croppad.array import (
    BorderPad,
    BoundingRect,
    CenterSpatialCrop,
    CropForeground,
    DivisiblePad,
    RandCropByLabelClasses,
    RandCropByPosNegLabel,
    ResizeWithPadOrCrop,
    SpatialCrop,
    SpatialPad,
)
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform, Randomizable
from monai.transforms.utils import (
    allow_missing_keys_mode,
    generate_label_classes_crop_centers,
    generate_pos_neg_label_crop_centers,
    is_positive,
    map_binary_to_indices,
    map_classes_to_indices,
    weighted_patch_samples,
)
from monai.utils import ImageMetaKey as Key
from monai.utils import Method, NumpyPadMode, PytorchPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple
from monai.utils.enums import PostFix, TraceKeys

__all__ = [
    "PadModeSequence",
    "SpatialPadd",
    "BorderPadd",
    "DivisiblePadd",
    "SpatialCropd",
    "CenterSpatialCropd",
    "CenterScaleCropd",
    "RandScaleCropd",
    "RandSpatialCropd",
    "RandSpatialCropSamplesd",
    "CropForegroundd",
    "RandWeightedCropd",
    "RandCropByPosNegLabeld",
    "ResizeWithPadOrCropd",
    "BoundingRectd",
    "RandCropByLabelClassesd",
    "SpatialPadD",
    "SpatialPadDict",
    "BorderPadD",
    "BorderPadDict",
    "DivisiblePadD",
    "DivisiblePadDict",
    "SpatialCropD",
    "SpatialCropDict",
    "CenterSpatialCropD",
    "CenterSpatialCropDict",
    "CenterScaleCropD",
    "CenterScaleCropDict",
    "RandScaleCropD",
    "RandScaleCropDict",
    "RandSpatialCropD",
    "RandSpatialCropDict",
    "RandSpatialCropSamplesD",
    "RandSpatialCropSamplesDict",
    "CropForegroundD",
    "CropForegroundDict",
    "RandWeightedCropD",
    "RandWeightedCropDict",
    "RandCropByPosNegLabelD",
    "RandCropByPosNegLabelDict",
    "ResizeWithPadOrCropD",
    "ResizeWithPadOrCropDict",
    "BoundingRectD",
    "BoundingRectDict",
    "RandCropByLabelClassesD",
    "RandCropByLabelClassesDict",
]

PadModeSequence = Union[Sequence[Union[NumpyPadMode, PytorchPadMode, str]], NumpyPadMode, PytorchPadMode, str]
DEFAULT_POST_FIX = PostFix.meta()

class MyCenterCropd(Randomizable, MapTransform, InvertibleTransform):
    backend = CenterSpatialCrop.backend

    def __init__(
        self, keys: KeysCollection, roi_size: Union[Sequence[int], int], allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.cropper = CenterSpatialCrop(roi_size)
    
    def randomize(self, img_size: Sequence[int]) -> None:
        self._size = fall_back_tuple(self.roi_size, img_size)
        if self.random_size:
            max_size = img_size if self.max_roi_size is None else fall_back_tuple(self.max_roi_size, img_size)
            if any(i > j for i, j in zip(self._size, max_size)):
                raise ValueError(f"min ROI size: {self._size} is bigger than max ROI size: {max_size}.")
            self._size = [self.R.randint(low=self._size[i], high=max_size[i] + 1) for i in range(len(img_size))]
        if self.random_center:
            valid_size = get_valid_patch_size(img_size, self._size)
            self._slices = (slice(None),) + get_random_patch(img_size, valid_size, self.R)


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            orig_size = d[key].shape[1:]
            d[key] = self.cropper(d[key])
            self.push_transform(d, key, orig_size=orig_size)
        return d


    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.array(transform[TraceKeys.ORIG_SIZE])
            current_size = np.array(d[key].shape[1:])
            pad_to_start = np.floor((orig_size - current_size) / 2).astype(int)
            # in each direction, if original size is even and current size is odd, += 1
            pad_to_start[np.logical_and(orig_size % 2 == 0, current_size % 2 == 1)] += 1
            pad_to_end = orig_size - current_size - pad_to_start
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            inverse_transform = BorderPad(pad)
            # Apply inverse transform
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d
