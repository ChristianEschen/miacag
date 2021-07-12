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
from dataloader.dataloader_base_monai import base_monai_loader


class base_monai_segmentation_loader(base_monai_loader):
    def __init__(self, image_path, csv_path_file, config,
                 use_complete_data=True):
        super(base_monai_segmentation_loader, self).__init__(
            image_path,
            csv_path_file,
            config,
            use_complete_data)
        self.features = self.get_input_features(self.csv)
        self.csv = self.set_feature_path(self.csv,
                                         self.features,
                                         self.image_path)
        self.csv = self.set_feature_path(self.csv,
                                         ['labels'],
                                         self.image_path)
        self.data = self.csv.to_dict('records')
        self.image_data = self.csv[self.features+['labels']].to_dict('records')
        self.config = config


class train_monai_segmentation_loader(base_monai_segmentation_loader):
    def __init__(self, image_path, csv_path_file, config):
        super(train_monai_segmentation_loader, self).__init__(image_path,
                                                              csv_path_file,
                                                              config)

    def __call__(self):
        train_transforms = [
                LoadImaged(keys=self.features + ["labels"]),
                AddChanneld(keys=self.features),
                ConvertToMultiChannel(keys="labels"),
                self.maybeReorder_z_dim(),
                self.getMaybeForegroundCropper(),
                Spacingd(
                     keys=self.features + ["labels"],
                     pixdim=(self.config['loaders']['pixdim_depth'],
                             self.config['loaders']['pixdim_height'],
                             self.config['loaders']['pixdim_width']),
                     mode=tuple([
                         'bilinear' for i in
                         range(len(self.features))]+['nearest'])),
                self.getMaybePad(),
                self.getMaybeClip(),
                self.getNormalization(),
                NormalizeIntensityd(keys=self.features, nonzero=True,
                                    channel_wise=True),
                RandCropByPosNegLabeld(
                    keys=self.features + ["labels"],
                    label_key="labels",
                    spatial_size=[self.config['loaders']['depth'],
                                  self.config['loaders']['height'],
                                  self.config['loaders']['width']],
                    pos=1, neg=1, num_samples=1),
                RandRotate90d(keys=self.features + ["labels"], prob=0.1, max_k=3,
                              spatial_axes=(1, 2)),
                RandZoomd(
                    keys=self.features + ["labels"],
                    min_zoom=0.9,
                    max_zoom=1.2,
                    mode=tuple([
                            "trilinear" for i in
                            range(len(self.features))]) + ("nearest",),
                    align_corners=tuple([
                                    True for i in
                                    range(len(self.features))]) + (None,),
                    prob=0.16,
                ),
                CastToTyped(keys=self.features + ["labels"],
                            dtype=tuple([
                                    np.float32 for i in
                                    range(len(self.features))]) + (np.uint8,)),
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
                # RandFlipd(self.features + ["labels"],
                #           spatial_axis=[0, 1, 2], prob=0.5),
                self.getMaybeConcat(),
                ToTensord(keys=["inputs", "labels"]),
                            ]
        train_transforms = Compose(train_transforms)
        # CHECK: for debug
        check_ds = monai.data.Dataset(data=self.image_data,
                                      transform=train_transforms)
        from torch.utils.data import DataLoader
        from monai.data import list_data_collate
        check_loader = DataLoader(
            check_ds,
            batch_size=self.config['loaders']['batchSize'],
            num_workers=self.config['loaders']['numWorkers'],
            collate_fn=list_data_collate)
        check_data = monai.utils.misc.first(check_loader)

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
                LoadImaged(keys=self.features + ["labels"]),
                AddChanneld(keys=self.features),
                ConvertToMultiChannel(keys="labels"),
                self.maybeReorder_z_dim(),
                self.getMaybeForegroundCropper(),
                Spacingd(
                     keys=self.features + ["labels"],
                     pixdim=(self.config['loaders']['pixdim_depth'],
                             self.config['loaders']['pixdim_height'],
                             self.config['loaders']['pixdim_width']),
                     mode=tuple([
                         'bilinear' for i in
                         range(len(self.features))]+['nearest'])),
                self.getMaybePad(),
                self.getMaybeClip(),
                self.getNormalization(),
                NormalizeIntensityd(keys=self.features, nonzero=True,
                                    channel_wise=True),
                self.getMaybeRandCrop(),
                CastToTyped(keys=self.features + ["labels"],
                            dtype=tuple([
                                    np.float32 for i in
                                    range(len(self.features))]) + (np.uint8,)),
                self.getMaybeConcat(),
                ToTensord(keys=["inputs", "labels"]),
                            ]
        val_transforms = Compose(val_transforms)
        # CHECK: for debug
        check_ds = monai.data.Dataset(data=self.image_data,
                                      transform=val_transforms)
        from torch.utils.data import DataLoader
        from monai.data import list_data_collate
        check_loader = DataLoader(
            check_ds,
            batch_size=self.config['loaders']['batchSize'],
            num_workers=self.config['loaders']['numWorkers'],
            collate_fn=list_data_collate)
        check_data = monai.utils.misc.first(check_loader)
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
