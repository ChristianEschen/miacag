from torch.utils.data import DataLoader
import monai
import pandas as pd
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
    GridPatchd,
    ToDeviced,
    RandLambdad,
    CopyItemsd,
    LoadImaged,
    EnsureTyped,
    RandSpatialCropSamplesd,
    RandSpatialCropSamples,
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
from miacag.dataloader.dataloader_base_monai import \
    base_monai_loader, LabelEncodeIntegerGraded
from monai.data import GridPatchDataset, PatchDataset, PatchIter
import os
from miacag.dataloader.Classification._2D.dataset_mil_2d import \
    Dataset, CacheDataset, SmartCacheDataset, PersistentDataset
    #artition_dataset_classes, partition_dataset


def reorder_rows(df):
    temp = pd.pivot_table(
        df, index=['StudyInstanceUID', 'PatientID'],
        values=['DcmPathFlatten', 'SOPInstanceUID', 'SeriesInstanceUID'],
        aggfunc=lambda x: list(x))
    temp = temp.reset_index(level=[0, 1])
    df = df.drop(columns=[
        'DcmPathFlatten', 'SOPInstanceUID', 'SeriesInstanceUID'])
    df = temp.merge(
        df, on=["PatientID", "StudyInstanceUID"],
        how="inner")
    df = df.drop_duplicates(['StudyInstanceUID', 'PatientID'])
    return df


class train_monai_classification_loader(base_monai_loader):
    def __init__(self, df, config):
        super(base_monai_loader, self).__init__(
            df,
            config)
        if config['weighted_sampler'] == 'True':
            self.getSampler()
        self.features = self.get_input_features(self.df)
        self.set_data_path(self.features)
        self.df = reorder_rows(self.df)
        self.data = self.df[
            self.features + config['labels_names'] +
            ['rowid', "SOPInstanceUID"]]
        self.data = self.data.to_dict('records')

    def __call__(self):
        # define transforms for image

        train_transforms = [
                LoadImaged(keys=self.features),
                EnsureChannelFirstD(keys=self.features),
                # LabelEncodeIntegerGraded(
                #     keys=self.config['labels_names'],
                #     num_classes=self.config['model']['num_classes']),
                self.resampleORresize(),
                DeleteItemsd(keys=self.features[0]+"_meta_dict.[0-9]\\|[0-9]", use_re=True),
                self.getMaybePad(),
                self.getCopy1to3Channels(),
                self.getClipChannels(),
                ScaleIntensityd(keys=self.features),
                self.maybeNormalize(),
                EnsureTyped(keys=self.features, data_type='tensor'),
                self.maybeToGpu(self.features),
                self.maybeTranslate(),
                self.maybeSpatialScaling(),
                self.maybeTemporalScaling(),
                self.maybeRotate(),
                self.CropTemporalMIL2d(),
                ConcatItemsd(keys=self.features, name='inputs'),
                DeleteItemsd(keys=self.features),
                ]

        train_transforms = Compose(train_transforms)
        train_transforms.set_random_state(seed=0)
        # CHECK: for debug ###
      
        # check_ds = Dataset(config=self.config,
        #                    features=self.features,
        #                    data=self.data,
        #                    transform=train_transforms)
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
        if len(self.config['labels_names'][0]) == 1:
            classes = [i[self.config['labels_names'][0]] for i in self.data]
            self.data_par_train = monai.data.partition_dataset_classes(
                data=self.data,
                classes=classes,
                num_partitions=dist.get_world_size(),
                shuffle=True,
                even_divisible=True,
            )[dist.get_rank()]
        else:
            self.data_par_train = monai.data.partition_dataset(
                data=self.data,
                num_partitions=dist.get_world_size(),
                shuffle=True,
                even_divisible=True,
            )[dist.get_rank()]

        # create a training data loader
        if self.config['cache_num'] != 'None':
            train_ds = SmartCacheDataset(
                config=self.config,
                features=self.features,
                data=self.data_par_train,
                transform=train_transforms,
                copy_cache=True,
                cache_num=self.config['cache_num'],
                num_init_workers=int(self.config['num_workers']/2),
                replace_rate=self.config['replace_rate'],
                num_replace_workers=int(self.config['num_workers']/2))
        else:
            train_ds = CacheDataset(
                config=self.config,
                features=self.features,
                data=self.data_par_train,
                transform=train_transforms,
                copy_cache=True,
                num_workers=self.config['num_workers'])
        return train_ds


class val_monai_classification_loader(base_monai_loader):
    def __init__(self, df, config):
        super(val_monai_classification_loader, self).__init__(df,
                                                              config)

        self.features = self.get_input_features(self.df)
        self.set_data_path(self.features)
        self.df = reorder_rows(self.df)

        self.data = self.df[
            self.features + config['labels_names'] +
            ['rowid', "SOPInstanceUID", "PatientID",
             "StudyInstanceUID", "SeriesInstanceUID"]]
        self.data = self.data.to_dict('records')


    def __call__(self):
        val_transforms = [
                LoadImaged(keys=self.features),
                EnsureChannelFirstD(keys=self.features),
                self.resampleORresize(),
                self.maybeDeleteMeta(),
               # DeleteItemsd(keys=self.features[0]+"_meta_dict.[0-9]\\|[0-9]", use_re=True),
                self.getMaybePad(),
                self.getCopy1to3Channels(),
                self.getClipChannels(),
                ScaleIntensityd(keys=self.features),
                self.maybeNormalize(),
                EnsureTyped(keys=self.features, data_type='tensor'),
                self.maybeToGpu(self.features),
                self.CropTemporalMIL2d(),
                ConcatItemsd(keys=self.features, name='inputs'),
               # self.maybeDeleteFeatures()
                ]

        val_transforms = Compose(val_transforms)
        val_transforms.set_random_state(seed=0)
        #if self.config['use_DDP'] == 'True':
        self.data_par_val = monai.data.partition_dataset(
            data=self.data,
            num_partitions=dist.get_world_size(),
            shuffle=False,
            even_divisible=True if self.config['loaders']['mode'] not in ['testing', 'prediction'] else False,
        )[dist.get_rank()]
        rowids = [i["rowid"] for i in self.data_par_val]
        if self.config['loaders']['mode'] not in ['prediction', 'testing']:
            if self.config['cache_num'] != 'None':
                val_ds = SmartCacheDataset(
                    config=self.config,
                    features=self.features,
                    data=self.data_par_val,
                    transform=val_transforms,
                    copy_cache=True,
                    cache_num=self.config['cache_num'],
                    num_init_workers=int(self.config['num_workers']/2),
                    replace_rate=self.config['replace_rate'],
                    num_replace_workers=int(self.config['num_workers']/2))
            else:
                val_ds = CacheDataset(
                    config=self.config,
                    features=self.features,
                    data=self.data_par_val,
                    transform=val_transforms,
                    copy_cache=True,
                    num_workers=self.config['num_workers'])
        else:
            if self.config['cache_test'] == "True":
                val_ds = CacheDataset(
                        config=self.config,
                        features=self.features,
                        data=self.data_par_val,
                        transform=val_transforms,
                        copy_cache=True,
                        num_workers=self.config['num_workers'])
            elif self.config['cache_test'] == "False":
                val_ds = Dataset(
                        config=self.config,
                        features=self.features,
                        data=self.data_par_val, transform=val_transforms
                    )
            elif self.config['cache_test'] == "persistant":
                cachDir = os.path.join(
                    self.config['model']['pretrain_model'],
                    'persistent_cache')
                val_ds = PersistentDataset(
                        config=self.config,
                        features=self.features,
                        data=self.data_par_val, transform=val_transforms,
                        cache_dir=cachDir
                    )
            else:
                raise ValueError(
                    'this type of test is not implemented! :',
                    self.config['cache_test'])

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
                LoadImaged(keys=self.features),
                EnsureChannelFirstD(keys=self.features),
                self.resampleORresize(),
                self.getMaybePad(),
                self.getCopy1to3Channels(),
                self.getClipChannels(),
                ScaleIntensityd(keys=self.features),
                NormalizeIntensityd(keys=self.features,
                                    channel_wise=True),
                EnsureTyped(keys=self.features, data_type='tensor'),

                ConcatItemsd(keys=self.features, name='inputs'),
                ]
        val_transforms = Compose(val_transforms)
        if self.config['use_DDP'] == 'True':
            self.data_par_val = partition_dataset(
                data=self.data,
                num_partitions=dist.get_world_size(),
                shuffle=True,
                even_divisible=True,
            )[dist.get_rank()]
            if self.config['cache_num'] != 'None':
                val_ds = SmartCacheDataset(
                    self.config,
                    self.features,
                    data=self.data_par_val,
                    transform=val_transforms,
                    copy_cache=True,
                    cache_num=self.config['cache_num'],
                    num_init_workers=int(self.config['num_workers']/2),
                    replace_rate=self.config['replace_rate'],
                    num_replace_workers=int(self.config['num_workers']/2))
            else:
                val_ds = CacheDataset(
                    self.config,
                    self.features,
                    data=self.data_par_val,
                    transform=val_transforms,
                    copy_cache=True,
                    num_workers=self.config['num_workers'])
        else:
            val_ds = Dataset(
                config=self.config,
                features=self.features,
                data=self.data,
                transform=val_transforms)

            self.data = partition_dataset(
                data=self.data,
                num_partitions=dist.get_world_size(),
                shuffle=False,
                even_divisible=True,
            )[dist.get_rank()]
        val_ds = Dataset(
            config=self.config,
            features=self.features,  
            data=self.data,
            transform=val_transforms)
        return val_ds
