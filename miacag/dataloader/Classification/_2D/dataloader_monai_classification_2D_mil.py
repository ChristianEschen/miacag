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
    Dataset, CacheDataset, SmartCacheDataset, PersistentDataset, MILDataset#, MILDataset_old
from scipy.ndimage import convolve1d
from miacag.model_utils.utils_regression import get_lds_kernel_window
from collections import Counter
from miacag.dataloader.dataloader_base import _get_weights_classification
import random
import numpy as np
from sklearn.utils import shuffle


# compute weights for weighted sampler for config['labels_names']
def _compute_weights(df, config):
    labels = df[config['labels_names']].values
    label_counts = Counter(labels)

    # Compute inverse frequency weights
    weights = [len(labels) / label_counts[label] for label in labels]

    # Optionally normalize the weights to sum to 1
    # total_weight = sum(weights)
    # weights = [w / total_weight for w in weights]
    return weights

def _prepare_weights(df, reweight, target_name, max_target=0.99, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    assert reweight in {'none', 'inverse', 'sqrt_inv'}
    assert reweight != 'none' if lds else True, \
        "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

    value_dict = {x: 0 for x in range(max_target)}
    labels = df[target_name].values * 100

    # Handling NaN values explicitly
    labels = [int(i) if not np.isnan(i) else np.nan for i in labels]

    for label in labels:
        if not np.isnan(label):
            value_dict[min(max_target - 1, int(label))] += 1

    # Adjust weight calculation to handle NaNs
    num_per_label = [value_dict[min(max_target - 1, int(label))] if not np.isnan(label) else np.nan for label in labels]

    if reweight == 'sqrt_inv':
        value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
    elif reweight == 'inverse':
        value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}

    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
        smoothed_value = convolve1d(
            np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
        num_per_label = [smoothed_value[min(max_target - 1, int(label))] if not np.isnan(label) else np.nan for label in labels]

    # Calculate weights while handling NaNs
    weights = [np.float32(1 / x) if not np.isnan(x) else np.nan for x in num_per_label]

    # Adjust scaling to exclude NaNs
    valid_weights = [w for w in weights if not np.isnan(w)]
    scaling = len(valid_weights) / np.sum(valid_weights)
    weights = [scaling * x if not np.isnan(x) else np.nan for x in weights]

    return weights


# def _prepare_weights(df, reweight, target_name, max_target=0.99, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
#     assert reweight in {'none', 'inverse', 'sqrt_inv'}
#     assert reweight != 'none' if lds else True, \
#         "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

#     value_dict = {x: 0 for x in range(max_target)}
#     labels = df[target_name].values*100
#     #   labels = generate_values()*100
#     labels = [int(i) for i in labels]
#     for label in labels:
#         value_dict[min(max_target - 1, int(label))] += 1
#     if reweight == 'sqrt_inv':
#         value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
#     elif reweight == 'inverse':
#         value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
#     num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
#     if not len(num_per_label) or reweight == 'none':
#         return None
#     print(f"Using re-weighting: [{reweight.upper()}]")
    
#     if lds:
#         lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
#         print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
#         smoothed_value = convolve1d(
#             np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
#         num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]
#     #   plt.hist(labels)
#     #   plt.show()
#     #  plt.hist(smoothed_value)
#     #  plt.show()
#     weights = [np.float32(1 / x) for x in num_per_label]
#     scaling = len(weights) / np.sum(weights)
#     weights = [scaling * x for x in weights]
#     #plt.hist(weights)
#     #  plt.show()
#     return weights
def generate_debug_label(len_vector, num_classes):
    values = list(range(0, num_classes))

    # Start by adding one of each required value.
    vector = values[:]

    # Fill the rest of the vector with random choices from the required values.
    while len(vector) < len_vector:
        vector.append(random.choice(values))
    return vector

def generate_weights_for_single_label(df, config):
    if config['labels_names'][0].startswith('koronar') or config['labels_names'][0].startswith('treatment'):
        # generate random values for labels with number of classes specificed by:
        num_classes = len(config['labels_dict']) - 1
        #generate values covering every labels with number of classes (num_classes)
        labels_debug = generate_debug_label(len(df), num_classes)
        # insert to df with labels_names
        df[config['labels_names'][0]] = labels_debug
        labels_to_weight = config['labels_names']
        

    elif config['labels_names'][0].startswith('duration'):
        num_classes = 2
        #generate values covering every labels with number of classes (num_classes)
        labels_debug = generate_debug_label(len(df), num_classes)
        # insert to df with labels_names
        
        labels_to_weight = ['event']
        df[[labels_to_weight][0]] = labels_debug
    else:
        ValueError('labels_names not recognized')
    return df, labels_to_weight
    
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
        # if config['weighted_sampler'] == 'True':
        #     self.getSampler()
        self.features = self.get_input_features(self.df)
        self.set_data_path(self.features)
        self.df = reorder_rows(self.df)
        # shuffle rows in df
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        # dropnans with labels
        self.df = self.df.dropna(subset=config["labels_names"], how='all')
        self.df = self.df.fillna(value=np.nan)
        if self.config['labels_names'][0].startswith('treatment'):
            # drop rows where the column self.config['labels_names'][0] is equal to 4
            self.df = self.df[self.df[self.config['labels_names'][0]] != 4]
        w_label_names = []
        if config['labels_names'][0].startswith('sten'):
            
            self.max_target = 100
        elif config['labels_names'][0].startswith('ffr'):
            self.max_target = 100

        elif config['labels_names'][0].startswith('timi'):
            self.max_target = 3
        else:
            self.max_target = 100    
        for label_name in config['labels_names']:
           # if label_name.startswith('sten') or label_name.startswith('ffr') or :
            if self.config['debugging'] == True:
                self.df = self.df.loc[self.df.index.repeat(2)].reset_index(drop=True)
                self.df[config['labels_names'][0]].iloc[0]= np.NaN
                self.df[config['labels_names'][0]].iloc[1]= np.NaN
          #      self.df[config['labels_names'][1]].iloc[0]= np.NaN
            self.weights = _prepare_weights(self.df, reweight="inverse", target_name=label_name, max_target=100, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
            w_label_names.append('weights_' + label_name)
            self.df['weights_' + label_name] = self.weights
        #self.data = self.df[self.features + config['labels_na
        if self.config['loss']['name'][0] == 'NNL':
            event = ['event']
        else:
            event = []
        
        if self.config['labels_names'][0].startswith('koronar') or self.config['labels_names'][0].startswith('duration') or self.config['labels_names'][0].startswith('treatment'):
            if self.config['debugging'] == True:
                self.df = self.df.loc[self.df.index.repeat(5)].reset_index(drop=True)
            
            self.df, labels_to_weight = generate_weights_for_single_label(self.df, self.config)

            self.weights = _get_weights_classification(self.df, labels_to_weight , config)
          #  self.weights_cat = self._compute_weights(self.df, config)
        self.data = self.df[
            self.features + config['labels_names'] +
            ['rowid', "SOPInstanceUID", 'SeriesInstanceUID',
             "StudyInstanceUID", "PatientID"] + event+ w_label_names + ['duration_transformed']]
        # replicate rows 10 times in self.data
        # self.data = pd.DataFrame(np.repeat(
       # self.data = self.data.loc[self.data.index.repeat(100)].reset_index(drop=True)

        # unroll "DcmPathFlatten" column in dataframe to multiple rows
   #     self.data = self.data.explode('DcmPathFlatten')
        self.data = self.data.to_dict('records')



    def __call__(self):
        # define transforms for image

        train_transforms = [
                LoadImaged(keys=self.features, prune_meta_pattern="(^0008|^6000|^0010|^5004|^5006|^5|^0)"),
                EnsureChannelFirstD(keys=self.features),
                # LabelEncodeIntegerGraded(
                #     keys=self.config['labels_names'],
                #     num_classes=self.config['model']['num_classes']),
                self.resampleORresize(),
                DeleteItemsd(keys=self.features[0]+"_meta_dict.[0-9]\\|[0-9]", use_re=True),
                self.getMaybePad(),
                self.getCopy1to3Channels(),
               # self.getClipChannels(),
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
        
        # if len(self.config['labels_names'][0]) == 1:
        #     classes = [i[self.config['labels_names'][0]] for i in self.data]
        #     self.data = monai.data.partition_dataset_classes(
        #         data=self.data,
        #         classes=classes,
        #         num_partitions=dist.get_world_size(),
        #         shuffle=True,
        #         even_divisible=True,
        #     )[dist.get_rank()]
        # else:
        #     self.data = monai.data.partition_dataset(
        #         data=self.data,
        #         num_partitions=dist.get_world_size(),
        #         shuffle=True,
        #         even_divisible=True,
        #     )[dist.get_rank()]

        # create a training data loader'
        if self.config['cache_num'] not in ['standard', 'None']:
            train_ds = SmartCacheDataset(
                config=self.config,
                features=self.features,
                data=self.data,
                transform=train_transforms,
                copy_cache=True,
                cache_num=self.config['cache_num'],
                num_init_workers=int(self.config['num_workers']/2),
                replace_rate=self.config['replace_rate'],
                num_replace_workers=int(self.config['num_workers']/2))
        
        elif self.config['cache_num'] == 'standard':
            if self.config['mil_old']:
                train_ds = MILDataset_old(
                            config=self.config,
                            features=self.features,
                            data=self.data, transform=train_transforms,
                            phase='train'
                        )
            else:
                 train_ds = MILDataset(
                            config=self.config,
                            features=self.features,
                            data=self.data, transform=train_transforms,
                            phase='train'
                        )
        else:
            train_ds = CacheDataset(
                config=self.config,
                features=self.features,
                data=self.data,
                transform=train_transforms,
                copy_cache=True,
                num_workers=self.config['num_workers'])
            
        # for idx in range(len( train_ds)):
        #     try:
        #         _ =  train_ds[idx]
        #     except IndexError as e:
        #         print(f"IndexError at {idx}: {e}")
        return train_ds


class val_monai_classification_loader(base_monai_loader):
    def __init__(self, df, config):
        super(val_monai_classification_loader, self).__init__(df,
                                                              config)

        # self.features = self.get_input_features(self.df)
        # self.set_data_path(self.features)
        # self.df = reorder_rows(self.df)
        # if self.config['loss']['name'][0] == 'NNL':
        #     event = ['event']
        # else:
        #     event = []
        # self.data = self.df[
        #     self.features + config['labels_names'] +
        #     ['rowid', "SOPInstanceUID", "PatientID",
        #      "StudyInstanceUID", "SeriesInstanceUID","PatientID"] + event]
        # self.data = self.data.to_dict('records')

        # if config['weighted_sampler'] == 'True':
        #     self.getSampler()
        self.features = self.get_input_features(self.df)
        self.set_data_path(self.features)
        self.df = reorder_rows(self.df)
        self.df = self.df.dropna(subset=config["labels_names"], how='all')
        self.df = self.df.fillna(value=np.nan)
        if self.config['labels_names'][0].startswith('treatment'):
            # drop rows where the column self.config['labels_names'][0] is equal to 4
            self.df = self.df[self.df[self.config['labels_names'][0]] != 4]
        w_label_names = []
        if config['labels_names'][0].startswith('sten'):
            
            self.max_target = 100
        elif config['labels_names'][0].startswith('ffr'):
            
            self.max_target = 100
        elif config['labels_names'][0].startswith('timi'):
            self.max_target = 3
        else:
            self.max_target = 100
        for label_name in config['labels_names']:
            if self.config['debugging'] == True:
                self.df = self.df.loc[self.df.index.repeat(8)].reset_index(drop=True)
                self.df[config['labels_names'][0]].iloc[0]= np.NaN
                self.df[config['labels_names'][0]].iloc[1]= np.NaN
                #self.df[config['labels_names'][1]].iloc[0]= np.NaN
                # self.df[config['labels_names'][0]].iloc[5]= None
                # self.df[config['labels_names'][1]].iloc[5]= None
                # self.df[config['labels_names'][0]].iloc[6]= None
                # self.df[config['labels_names'][1]].iloc[7]= None
                
            self.weights = _prepare_weights(self.df, reweight="inverse", target_name=label_name, max_target=100, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
            w_label_names.append('weights_' + label_name)
            self.df['weights_' + label_name] = self.weights
        #self.data = self.df[self.features + config['labels_na
        if self.config['loss']['name'][0] == 'NNL':
            event = ['event']
        else:
            event = []
            
        if self.config['labels_names'][0].startswith('koronar') or self.config['labels_names'][0].startswith('duration') or self.config['labels_names'][0].startswith('treatment'):
            if self.config['debugging'] == True:
                self.df = self.df.loc[self.df.index.repeat(5)].reset_index(drop=True)
            
            self.df, labels_to_weight = generate_weights_for_single_label(self.df, self.config)

            self.weights = _get_weights_classification(self.df, labels_to_weight , config)

          #  self.weights_cat = self._compute_weights(self.df, config)
        self.data = self.df[
            self.features + config['labels_names'] +
            ['rowid', "SOPInstanceUID", 'SeriesInstanceUID',
             "StudyInstanceUID", "PatientID"] + event+ w_label_names + ['duration_transformed']]
   #     self.data = self.data.explode('DcmPathFlatten')
        self.data = self.data.to_dict('records')


    def __call__(self):
        val_transforms = [
                LoadImaged(keys=self.features, prune_meta_pattern="(^0008|^6000|^0010|^5004|^5006|^5|^0)"),
                EnsureChannelFirstD(keys=self.features),
                self.resampleORresize(),
                self.maybeDeleteMeta(),
               # DeleteItemsd(keys=self.features[0]+"_meta_dict.[0-9]\\|[0-9]", use_re=True),
           #     self.getMaybePad(),
                self.getCopy1to3Channels(),
              #  self.getClipChannels(),
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
        # self.data = monai.data.partition_dataset(
        #     data=self.data,
        #     num_partitions=dist.get_world_size(),
        #     shuffle=False,
        #     even_divisible=True if self.config['loaders']['mode'] not in ['testing', 'prediction'] else False,
        # )[dist.get_rank()]
        rowids = [i["rowid"] for i in self.data]
        if self.config['loaders']['mode'] not in ['prediction', 'testing']:
            if self.config['cache_num'] not in ['standard', 'None']:
                val_ds = SmartCacheDataset(
                    config=self.config,
                    features=self.features,
                    data=self.data,
                    transform=val_transforms,
                    copy_cache=True,
                    cache_num=self.config['cache_num'],
                    num_init_workers=int(self.config['num_workers']/2),
                    replace_rate=self.config['replace_rate'],
                    num_replace_workers=int(self.config['num_workers']/2))
            elif self.config['cache_num'] == 'standard':
                if self.config['mil_old']:
                    val_ds = MILDataset_old(
                                config=self.config,
                                features=self.features,
                                data=self.data, transform=val_transforms,
                                phase='val'
                            )
                else:
                    val_ds = MILDataset(
                                config=self.config,
                                features=self.features,
                                data=self.data, transform=val_transforms,
                                phase='val'
                            )
            else:
                val_ds = CacheDataset(
                    config=self.config,
                    features=self.features,
                    data=self.data,
                    transform=val_transforms,
                    copy_cache=True,
                    num_workers=self.config['num_workers'])
        else:
            if self.config['cache_test'] == "True":
                val_ds = CacheDataset(
                        config=self.config,
                        features=self.features,
                        data=self.data,
                        transform=val_transforms,
                        copy_cache=True,
                        num_workers=self.config['num_workers'])
            elif self.config['cache_test'] == "False":
                val_ds = MILDataset(
                        config=self.config,
                        features=self.features,
                        data=self.data, transform=val_transforms,
                        phase='val'
                    )
            elif self.config['cache_num'] == 'standard':
                val_ds = MILDataset(
                            config=self.config,
                            features=self.features,
                            data=self.data, transform=val_transforms,
                            phase='val'
                        )
            elif self.config['cache_test'] == "persistant":
                cachDir = os.path.join(
                    self.config['model']['pretrain_model'],
                    'persistent_cache')
                val_ds = PersistentDataset(
                        config=self.config,
                        features=self.features,
                        data=self.data, transform=val_transforms,
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
                LoadImaged(keys=self.features, prune_meta_pattern="(^0008|^6000|^0010|^5004|^5006|^5|^0)"),
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
            self.data = partition_dataset(
                data=self.data,
                num_partitions=dist.get_world_size(),
                shuffle=True,
                even_divisible=True,
            )[dist.get_rank()]
            if self.config['cache_num'] != 'None':
                val_ds = SmartCacheDataset(
                    self.config,
                    self.features,
                    data=self.data,
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
                    data=self.data,
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
