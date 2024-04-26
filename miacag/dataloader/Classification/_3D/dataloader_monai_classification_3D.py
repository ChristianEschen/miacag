from torch.utils.data import DataLoader
import monai
from monai.data import list_data_collate, pad_list_data_collate
import torch.distributed as dist
from monai.transforms import (
    EnsureChannelFirstd, #FIXME
    RepeatChanneld,
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
    Identityd,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    DataStatsd,
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
import os
import numpy as np
from scipy.ndimage import convolve1d
from miacag.model_utils.utils_regression import get_lds_kernel_window
import numpy as np
from miacag.dataloader.dataloader_base import _get_weights_classification
from sklearn.preprocessing import StandardScaler
from miacag.utils.survival_utils import LabTransDiscreteTime
import random
import pandas as pd
import torch
# import matplotlib
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# matplotlib.use( 'tkagg' )



class RandReplicateSliceTransform(MapTransform):
    """
    A transformation class to select a random 2D slice from a 3D volume and replicate it.
    The slice selection is normally distributed around the center of the volume.
    """
    def __init__(self, keys, num_slices=32, prob=0.2):
        super().__init__(keys)
        self.num_slices = num_slices
        self.prob = prob  

    def __call__(self, data):
        d = dict(data)  # Create a shallow copy to ensure source data is not modified

        apply_transform = np.random.rand() < self.prob 
        if apply_transform:
            for key in self.keys:
                x = d[key]
                mid_slice = float(x.shape[-1] // 2)  # Assuming x is in the shape [C, D, H, W] or [D, H, W]
                std_dev = float(np.array(range(0,x.shape[-1])).std()) # Standard deviation

                if torch.is_tensor(x):
                    # Normal distribution centered at the middle slice, for tensors
                    slice_idx  = int(torch.normal(mean=torch.tensor([mid_slice]), std=torch.tensor([std_dev])).clamp(0,x.shape[-1]-1).item())
                #   slice_idx = 32
                    selected_slice = x[:, :, :, slice_idx]
                    replicated_slices = torch.unsqueeze(selected_slice, -1).repeat(1, 1, 1, self.num_slices)
                else:
                    raise ValueError("not implemeted for np")
                    
                    # Normal distribution for numpy arrays
                    slice_idx = int(np.random.normal(mid_slice, std_dev))
                    slice_idx = np.clip(slice_idx, 0, x.shape[-1] - 1)
                    slice_idx = 32

                    selected_slice = x[:, :, :, slice_idx]
                    replicated_slices = np.repeat(selected_slice[np.newaxis, :, :], self.num_slices, axis=0)
                
                d[key] = replicated_slices
        
        return d
def generate_values(n=100):
    # Generate values around 0.7 using a normal distribution
    values_around_70 = np.random.normal(0.7, 0.05, n//2)
    
    # Clip values to ensure they are between 0 and 1
    values_around_70 = np.clip(values_around_70, 0, 1)
    
    # Create an array of zeros
    zeros = np.zeros(n//2)
    
    # Combine both arrays
    combined_values = np.concatenate([zeros, values_around_70])
    
    # Shuffle the combined array to mix the values
    np.random.shuffle(combined_values)
    
    return combined_values
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
def generate_debug_label(len_vector, num_classes):
    values = list(range(0, num_classes))

    # Start by adding one of each required value.
    vector = values[:]

    # Fill the rest of the vector with random choices from the required values.
    while len(vector) < len_vector:
        vector.append(random.choice(values))
    return vector
def generate_weights_for_single_label(df, config):
   # raise NotImplementedError('This function is not implemented yet')
    if config['labels_names'][0].startswith('koronar') or config['labels_names'][0].startswith('treatment') or config['labels_names'][0].startswith('labels_'):
        # generate random values for labels with number of classes specificed by:
        num_classes = len(config['labels_dict']) - 1
        #generate values covering every labels with number of classes (num_classes)
      #  labels_debug = generate_debug_label(len(df), num_classes)
        # insert to df with labels_names
     #   df[config['labels_names'][0]] = labels_debug
        labels_to_weight = config['labels_names']
        

    elif config['labels_names'][0].startswith('duration'):
        num_classes = 2
        #generate values covering every labels with number of classes (num_classes)
      #  labels_debug = generate_debug_label(len(df), num_classes)
        # insert to df with labels_names
        
        labels_to_weight = ['event']
     #   df[[labels_to_weight][0]] = labels_debug
    else:
        ValueError('labels_names not recognized')
    return df, labels_to_weight
    
    
class train_monai_classification_loader(base_monai_loader):
    def __init__(self, df, config):
        super(base_monai_loader, self).__init__(
            df,
            config)
        # if config['weighted_sampler'] == 'True':
        #     self.getSampler()
        self.features = self.get_input_features(self.df)
        self.set_data_path(self.features)
        #######################################################################
        # # old
        # w_label_names = []
        # for label_name in config['labels_names']:
            
        #     self.weights = self._prepare_weights(reweight="inverse", target_name=label_name, max_target=100, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
        #     w_label_names.append('weights_' + label_name)
        #     self.df['weights_' + label_name] = self.weights
        # self.data = self.df[self.features + config['labels_names'] + ['rowid'] + ['event'] +["labels_predictions"] + w_label_names]
        ########################################################################

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
            self.weights = _prepare_weights(self.df, reweight="inverse", target_name=label_name, max_target=100, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
            w_label_names.append('weights_' + label_name)
            self.df['weights_' + label_name] = self.weights
        #self.data = self.df[self.features + config['labels_na
        if self.config['loss']['name'][0] == 'NNL':
            # start with censor after followup
            self.df['event'] = self.df.apply(lambda row: 0 if row['duration_transformed'] > self.config['loss']['censur_date'] else row['event'], axis=1)
           # self.df = pd.DataFrame({"duration_transformed": np.random.randint(1, 551, size=100), "event": np.random.randint(0, 2, size=100)})

            self.labtrans = LabTransDiscreteTime(
                cuts=config['model']["num_classes"][0], #np.array((0, 100, 200, 300, 400, 500)),
                scheme='quantiles')

            get_target = lambda df: (self.df[self.config['labels_names'][0]].values, self.df['event'].values)

            target_trains = self.labtrans.fit_transform(*get_target(self.df))

            self.df[self.config['labels_names'][0]] = target_trains[0]
            
            self.df['event'] = target_trains[1]
            #self.config['labtrans'] = self.labtrans
            self.config['cuts'] = self.labtrans.cuts
            event = ['event']
        else:
            event = []
        
        if self.config['labels_names'][0].startswith('koronar') or self.config['labels_names'][0].startswith('duration') or self.config['labels_names'][0].startswith('treatment') or self.config['labels_names'][0].startswith('labels_'):
            self.df, labels_to_weight = generate_weights_for_single_label(self.df, self.config)

            self.weights = _get_weights_classification(self.df, labels_to_weight , config)
          #  self.weights_cat = self._compute_weights(self.df, config)


        self.data = self.df[
            self.features + config['labels_names'] +
            ['rowid', "SOPInstanceUID", 'SeriesInstanceUID',
             "StudyInstanceUID", "PatientID", 'labels_predictions', "treatment_transformed"] + ["koronarpatologi_transformed"] + event+ w_label_names + ['duration_transformed']]
        self.data.fillna(value=np.nan, inplace=True)
        self.data = self.data.to_dict('records')

    def transformations(self):
        self.transforms = [
            LoadImaged(keys=self.features, prune_meta_pattern="(^0008|^6000|^0010|^5004|^5006|^5|^0)"),
            EnsureChannelFirstD(keys=self.features),
            self.resampleORresize(),
            # DeleteItemsd(
            #     keys=self.features[0]+"_meta_dict.[0-9]\\|[0-9]", use_re=True), #FIXME
            self.getMaybePad(),
            self.getCopy1to3Channels(),
            EnsureTyped(keys=self.features, data_type='tensor'),
            self.maybeToGpu(self.features),
          #  RandReplicateSliceTransform(keys=self.features, num_slices=self.config['loaders']['Crop_depth'], prob=0.5),
            self.maybeTranslate(),
            self.maybeSpatialScaling(),
            self.maybeTemporalScaling(),
            self.maybeRotate(),
            self.CropTemporal(),
            ScaleIntensityd(keys=self.features),
            self.maybeNormalize(config=self.config, features=self.features),
            ConcatItemsd(keys=self.features, name='inputs'),
            DeleteItemsd(keys=self.features),
            ]
        self.transforms = Compose(self.transforms)
        self.transforms.set_random_state(seed=0)
        return self.transforms

    def __call__(self):
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
        if type(self.config['cache_num']) == int:
            train_ds = monai.data.SmartCacheDataset(
                data=self.data_par_train,
                transform=self.transformations(),
                copy_cache=True,
                cache_num=self.config['cache_num'],
                num_init_workers=int(self.config['num_workers']/2),
                replace_rate=self.config['replace_rate'],
                num_replace_workers=int(self.config['num_workers']/2))
        elif self.config['cache_num'] in [None, 'None', False, 'False']:
            train_ds = monai.data.Dataset(
                data=self.data_par_train,
                transform=self.transformations())
        else:
            raise ValueError('Not implemented')
            # train_ds = monai.data.CacheDataset(
            #     data=self.data_par_train,
            #     transform=self.transformations(),
            #     copy_cache=True,
            #     num_workers=self.config['num_workers'])
        return train_ds

    def _prepare_weights(self, reweight, target_name, max_target=0.99, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
            assert reweight in {'none', 'inverse', 'sqrt_inv'}
            assert reweight != 'none' if lds else True, \
                "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

            value_dict = {x: 0 for x in range(max_target)}
            labels = self.df[target_name].values*100
         #   labels = generate_values()*100
            labels = [int(i) for i in labels]
            for label in labels:
                value_dict[min(max_target - 1, int(label))] += 1
            if reweight == 'sqrt_inv':
                value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
            elif reweight == 'inverse':
                value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
            num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
            if not len(num_per_label) or reweight == 'none':
                return None
            print(f"Using re-weighting: [{reweight.upper()}]")
            
            if lds:
                lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
                print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
                smoothed_value = convolve1d(
                    np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
                num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]
         #   plt.hist(labels)
         #   plt.show()
          #  plt.hist(smoothed_value)
          #  plt.show()
            weights = [np.float32(1 / x) for x in num_per_label]
            scaling = len(weights) / np.sum(weights)
            weights = [scaling * x for x in weights]
          #plt.hist(weights)
          #  plt.show()
            return weights

class val_monai_classification_loader(base_monai_loader):
    def __init__(self, df, config):
        super(val_monai_classification_loader, self).__init__(df,
                                                              config)

        self.features = self.get_input_features(self.df)
        self.set_data_path(self.features)
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
            self.df[label_name] = self.df[label_name].fillna(0)

            self.weights = _prepare_weights(self.df, reweight="inverse", target_name=label_name, max_target=100, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
            w_label_names.append('weights_' + label_name)
            self.df['weights_' + label_name] = self.weights
        #self.data = self.df[self.features + config['labels_na
        if self.config['loss']['name'][0] == 'NNL':
            self.df['event'] = self.df.apply(lambda row: 0 if row['duration_transformed'] > self.config['loss']['censur_date'] else row['event'], axis=1)

            if self.config["loaders"]["mode"] != 'training':
                cuts=self.config['cuts']
            else:
                cuts=self.config['cuts']
            self.labtrans = LabTransDiscreteTime(
                cuts=cuts,
                #cuts=config['model']["num_classes"][0], #np.array((0, 100, 200, 300, 400, 500)),
                )
            
            get_target = lambda df: (self.df[self.config['labels_names'][0]].values, self.df['event'].values)
            
            target_trains = self.labtrans.fit_transform(*get_target(self.df))
            
            self.df[self.config['labels_names'][0]] = target_trains[0]
            
            self.df['event'] = target_trains[1]
           # self.config['labtrans'] = self.labtrans
           # self.config['cuts'] = self.labtrans.cuts
            event = ['event']
        else:
            event = []
        
        if self.config['labels_names'][0].startswith('koronar') or self.config['labels_names'][0].startswith('duration') or self.config['labels_names'][0].startswith('treatment') or self.config['labels_names'][0].startswith('labels_'):
            self.df, labels_to_weight = generate_weights_for_single_label(self.df, self.config)

            self.weights = _get_weights_classification(self.df, labels_to_weight , config)
          #  self.weights_cat = self._compute_weights(self.df, config)


        self.data = self.df[
            self.features + config['labels_names'] +
            ['rowid', "SOPInstanceUID", 'SeriesInstanceUID',
             "StudyInstanceUID", "PatientID", 'labels_predictions'] + event+ w_label_names + ['duration_transformed']]
        ###############################################################################################333
        # OLD
     #   self.data = self.df[self.features + config['labels_names'] + ['rowid']]
        # w_label_names = []

        # for label_name in config['labels_names']:
        #     # replace nans with zeros
        #     self.weights = self._prepare_weights(reweight="inverse", target_name=label_name, max_target=100, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
        #     w_label_names.append('weights_' + label_name)
        #     self.df['weights_' + label_name] = self.weights
        # OLD
        #########################################################################################################
        # self.data = self.df[self.features + config['labels_names'] + ['rowid'] +
        #                     ['event'] +["labels_predictions"] +["treatment_transformed"] +
        #                     ["duration_transformed"] + ["koronarpatologi_transformed"] + w_label_names ]
        
        self.data.fillna(value=np.nan, inplace=True)
        
        
        
        self.data = self.data.to_dict('records')

    def _prepare_weights(self, reweight, target_name, max_target=0.99, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
            assert reweight in {'none', 'inverse', 'sqrt_inv'}
            assert reweight != 'none' if lds else True, \
                "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

            value_dict = {x: 0 for x in range(max_target)}
            labels = self.df[target_name].values*100
            for label in labels:
                value_dict[min(max_target - 1, int(label))] += 1
            if reweight == 'sqrt_inv':
                value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
            elif reweight == 'inverse':
                value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
            num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
            if not len(num_per_label) or reweight == 'none':
                return None
            print(f"Using re-weighting: [{reweight.upper()}]")

            if lds:
                lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
                print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
                smoothed_value = convolve1d(
                    np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
                num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

            weights = [np.float32(1 / x) for x in num_per_label]
            scaling = len(weights) / np.sum(weights)
            weights = [scaling * x for x in weights]
            return weights
    def tansformations(self):
        self.transforms = [
                LoadImaged(keys=self.features, prune_meta_pattern="(^0008|^6000|^0010|^5004|^5006|^5|^0)"),
                EnsureChannelFirstD(keys=self.features),
                self.resampleORresize(),
               # self.maybeDeleteMeta(), #FIXME
                self.getMaybePad(),
                self.getCopy1to3Channels(),
                EnsureTyped(keys=self.features, data_type='tensor'),
                self.maybeToGpu(self.features),
                self.maybeCenterCrop(self.features)
                    if self.config['loaders']['mode'] == 'training'
                    else monai.transforms.GridPatchd(keys=self.features,
                                                patch_size=(
                                                    self.config['loaders']['Crop_height'],
                                                    self.config['loaders']['Crop_width'],
                                                    self.config['loaders']['Crop_depth']),
                                                pad_mode="constant",
                                                ),
                ScaleIntensityd(keys=self.features) if self.config['loaders']['mode'] == 'training' else Identityd(keys=self.features),
                self.maybeNormalize(config=self.config, features=self.features) if self.config['loaders']['mode'] == 'training' else Identityd(keys=self.features),
                ConcatItemsd(keys=self.features, name='inputs'),
                self.maybeDeleteFeatures(),
                ]

        self.transforms = Compose(self.transforms, log_stats=False)
        self.transforms.set_random_state(seed=0)
        return self.transforms

    def __call__(self):
        self.data_par_val = monai.data.partition_dataset(
            data=self.data,
            num_partitions=dist.get_world_size(),
            shuffle=False,
            even_divisible=True if self.config['loaders']['mode'] not in ['testing', 'prediction'] else False,
        )[dist.get_rank()]
        if self.config['loaders']['mode'] not in ['prediction', 'testing']:
            # if self.config['cache_num'] != 'None':
            #     val_ds = monai.data.SmartCacheDataset(
            #         data=self.data_par_val,
            #         transform=self.tansformations(),
            #         copy_cache=True,
            #         cache_num=self.config['cache_num'],
            #         num_init_workers=int(self.config['num_workers']/2),
            #         replace_rate=self.config['replace_rate'],
            #         num_replace_workers=int(self.config['num_workers']/2))
            # else:
            if self.config['cache_test'] == "True":
                val_ds = monai.data.CacheDataset(
                    data=self.data_par_val,
                    transform=self.tansformations(),
                    copy_cache=True,
                    num_workers=self.config['num_workers'],
                    cache_num=self.config['cache_num_val'])
            else:
                val_ds = monai.data.Dataset(
                        data=self.data_par_val,
                    transform=self.tansformations())
        else:
            # if self.config['cache_test'] == "True":
            #     val_ds = monai.data.CacheDataset(
            #             data=self.data_par_val,
            #             transform=self.tansformations(),
            #             copy_cache=True,
            #             num_workers=self.config['num_workers'])
                val_ds = monai.data.Dataset(
                        data=self.data_par_val,
                    transform=self.tansformations())
            # elif self.config['cache_test'] == "persistant":
            #     cachDir = os.path.join(
            #         self.config['model']['pretrain_model'],
            #         'persistent_cache')
            #     val_ds = monai.data.PersistentDataset(
            #             data=self.data_par_val, transform=self.tansformations(),
            #             cache_dir=cachDir
            #         )
            # else:
            #     raise ValueError(
            #         'this type of test is not implemented! :',
            #         self.config['cache_test'])
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
                ScaleIntensityd(keys=self.features),
                NormalizeIntensityd(keys=self.features,
                                    channel_wise=True),
                EnsureTyped(keys=self.features, data_type='tensor'),

                ConcatItemsd(keys=self.features, name='inputs'),
                ]
        val_transforms = Compose(val_transforms, log_stats=False)
        val_ds = monai.data.Dataset(
            data=self.data,
            transform=val_transforms)
        return val_ds


## graveyard
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