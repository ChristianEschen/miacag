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
import yaml
from sklearn.impute import SimpleImputer
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import copy
# import matplotlib
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# matplotlib.use( 'tkagg' )
def z_score_normalize(data, keys):
    normalized_data = data.copy()  # Copy the original data to avoid in-place modification
    for key in keys:
        if key in data:
            value = data[key]
            mean = value.mean()
            std = value.std()
            #normalized_data[key] = (value - value.min()) / (value.max() - value.min() + std +1e-6)

            normalized_data[key] = (value - mean) / (std +1e-6)
    return normalized_data

def get_num_features(config):
    column_names = config['loaders']['tabular_data_names']
    column_types = config['loaders']['tabular_data_names_one_hot']  # 0: numeric, 1: categorical
    num_features = []
    for count, col_type in enumerate(column_types):
        if col_type == 0:  # Numeric
            num_features.append(column_names[count])
    return num_features

def z_score_denormalize(data, config):
    column_names = config['loaders']['tabular_data_names']
    column_types = config['loaders']['tabular_data_names_one_hot']  # 0: numeric, 1: categorical
    num_features = []
    for count, col_type in enumerate(column_types):
        if col_type == 0:  # Numeric

            num_features.append(column_names[count]
                                )
    scaler = preprocessing.StandardScaler().fit(data[num_features])
    return scaler, num_features

def impute_data(df, config):
    # Extract the configuration details
    column_names = config['loaders']['tabular_data_names']
    column_types = config['loaders']['tabular_data_names_one_hot']  # 0: numeric, 1: categorical
    
    # Iterate over the columns and their types
    for col_name, col_type in zip(column_names, column_types):
        if col_name in df.columns:
            if col_type == 0:  # Numeric
                median_value = df[col_name].median()
                df[col_name].fillna(median_value, inplace=True)
            elif col_type == 1:  # Categorical
                df[col_name].fillna('Ukendt', inplace=True)
        else:
            print(f"Warning: {col_name} not found in DataFrame")
    
    return df


def impute_missing(df, config):
    # Extract the configuration details
    column_names = config['loaders']['tabular_data_names']
    column_types = config['loaders']['tabular_data_names_one_hot']  # 0: numeric, 1: categorical
    
    # Iterate over the columns and their types
    for col_name, col_type in zip(column_names, column_types):
        if col_name in df.columns:
            # if col_type == 0:  # Numeric
            #     median_value = df[col_name].median()
            #     df[col_name].fillna(median_value, inplace=True)
            if col_type == 1:  # Categorical
                df[col_name].fillna('Ukendt', inplace=True)
        else:
            print(f"Warning: {col_name} not found in DataFrame")
    
    return df
def check_nan_after_imputation(df):
    if df.isnull().any().any():
        print("NaNs remain in the following columns:", df.columns[df.isnull().any()])
    else:
        print("No NaNs remain in the DataFrame.")

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
    

def determine_unique_values(df, config):
    indices_of_ones = [index for index, value in enumerate(config['loaders']['tabular_data_names_one_hot'][:config['loaders']['new_columns_nr']]) if value == 1]
    num_columns_cat = [config['loaders']['tabular_data_names'][:config['loaders']['new_columns_nr']][index] for index in indices_of_ones]
    columns_cat_missing = [col + '_missing' for col in num_columns_cat]
    unique_values = []
    for column in config['loaders']['tabular_data_names']:
        if column in num_columns_cat:
            unique_values.append(max(len(df[column].unique()), 2))
        elif column in columns_cat_missing:
            unique_values.append(2)
        else:
            unique_values.append(0)
    return unique_values
def add_misisng_indicator_column_names(df, imputer, config, phase='train'):

    tabular_feature_missing_indicator = imputer.indicator_.features_
    missing_names = []
    is_categorical = []
    uniques_if_categorical = []
    for i in range(0, len(config['loaders']['tabular_data_names'])):
        for tab_indic in tabular_feature_missing_indicator:
            if i == tab_indic:
                if config['loaders']['tabular_data_names_one_hot'][i] == 1:
                    is_categorical.append(1)
                else:
                    is_categorical.append(0)
                missing_names.append(config['loaders']['tabular_data_names'][i] +'_missing')
        if phase == 'train':

            if config['loaders']['tabular_data_names_one_hot'][i] == 1:
                uniques_if_categorical.append(len(df[config['loaders']['tabular_data_names'][i]].unique()))
            else:
                uniques_if_categorical.append(0)
    if phase == 'train':
        new_columns_nr = len(missing_names)
        config['loaders']['new_columns_nr'] = len(config['loaders']['tabular_data_names'])
        config['loaders']['new_data_names'] = [0]*len(config['loaders']['tabular_data_names']) + [1]*new_columns_nr
    else:
        if config['is_already_trained']:
            config['loaders']['new_columns_nr'] = len(config['loaders']['tabular_data_names'])-len(list(tabular_feature_missing_indicator))
            config['loaders']['tabular_data_names'] = config['loaders']['tabular_data_names'][0: len(config['loaders']['tabular_data_names'])-len(list(tabular_feature_missing_indicator))]
        else:
            config['loaders']['tabular_data_names'] = config['loaders']['tabular_data_names'][0:config['loaders']['new_columns_nr']]
    #if 

    imputed_missing_cols = imputer.transform(df[config['loaders']['tabular_data_names']])
    
    imputed_missing_cols_names = config['loaders']['tabular_data_names'] + missing_names

    for count, miss_name in enumerate(imputed_missing_cols_names):
        df[miss_name] = imputed_missing_cols[:, count]
    
    # update config
    if phase == 'train':
        
        config['loaders']['tabular_data_names'] = imputed_missing_cols_names
        config['loaders']['tabular_data_names_one_hot'] = config['loaders']['tabular_data_names_one_hot'] + is_categorical
        config['loaders']['tabular_data_names_embed_dim'] = uniques_if_categorical + is_categorical
    else:
        config['loaders']['tabular_data_names'] = imputed_missing_cols_names
    if config['loaders']['mode'] in ['prediction', 'testing']:
        config['loaders']['tabular_data_names'] = imputed_missing_cols_names
      #  config['loaders']['tabular_data_names_one_hot'] = config['loaders']['tabular_data_names_one_hot'] + is_categorical
       # config['loaders']['tabular_data_names_embed_dim'] = uniques_if_categorical + is_categorical
    return df, config


class train_monai_classification_loader(base_monai_loader):
    def __init__(self, df, config):
        super(base_monai_loader, self).__init__(
            df,
            config)
        # if config['weighted_sampler'] == 'True':
        #     getSampler()
        self.features = self.get_input_features(self.df)
        self.set_data_path(self.features)
        # shuffle the data
        self.df = self.df.sample(frac=1).reset_index(drop=True)
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
        # define hard cases based on presence if any labels_names are between 0.001 and 0.999 and add to new column
        self.df['weights'] = self.df[config['labels_names']].apply(lambda x: any([0.001 < i < 0.999 for i in x]), axis=1).astype(int)
        self.df['weights'] = _get_weights_classification(self.df, ["weights"],config)
        for label_name in config['labels_names']:
            self.weights = _prepare_weights(self.df, reweight="inverse", target_name=label_name, max_target=100, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
            w_label_names.append('weights_' + label_name)
            self.df['weights_' + label_name] = self.weights
        #self.data = self.df[self.features + config['labels_na
        if self.config['loss']['name'][0] == 'NNL':
            # start with censor after followup
            # drop rows with nans for event and duration_transformed
            self.df = self.df.dropna(subset=['event', 'duration_transformed'])
            self.df['event'] = self.df.apply(lambda row: 0 if row['duration_transformed'] > self.config['loss']['censur_date'] else row['event'], axis=1)
           # self.df = pd.DataFrame({"duration_transformed": np.random.randint(1, 551, size=100), "event": np.random.randint(0, 2, size=100)})

            self.labtrans = LabTransDiscreteTime(
                cuts=config['model']["num_classes"][0], #np.array((0, 100, 200, 300, 400, 500)),
             #   scheme='quantiles'
                )

            get_target = lambda df: (self.df[self.config['labels_names'][0]].values, self.df['event'].values)
            print('get_target', get_target(self.df))

            target_trains = self.labtrans.fit_transform(*get_target(self.df))
            

            durations_train, events_train = get_target(self.df)
            self.df[self.config['labels_names'][0]], self.df['event'] = self.labtrans.transform(durations_train, events_train)


            
           # self.df['event'] = target_trains[1]
            #self.config['labtrans'] = self.labtrans
            self.config['cuts'] = self.labtrans.cuts
            event = ['event']
            self.df['weights'] = 1
        else:
            event = []
        
        if self.config['labels_names'][0].startswith('koronar') or self.config['labels_names'][0].startswith('duration') or self.config['labels_names'][0].startswith('treatment') or self.config['labels_names'][0].startswith('labels_'):
            self.df, labels_to_weight = generate_weights_for_single_label(self.df, self.config)

            self.weights = _get_weights_classification(self.df, labels_to_weight , config)
          #  self.weights_cat = self._compute_weights(self.df, config)
            #self.df[w_label_names]= self.weights
            self.df['weights'] = self.weights
       # else:
           # self.df['weights'] = 1

        self.data = self.df[
            self.features + config['labels_names'] + ["weights"] +
            ['rowid', "SOPInstanceUID", 'SeriesInstanceUID',
             "StudyInstanceUID", "PatientID"] + event+ w_label_names + ['duration_transformed'] + config['loaders']['tabular_data_names']]
        self.data.fillna(value=np.nan, inplace=True)
        if len(config['loaders']['tabular_data_names']) > 0:

            num_columns = [config['loaders']['tabular_data_names'][i] for i in range(len(config['loaders']['tabular_data_names'])) if config['loaders']['tabular_data_names_one_hot'][i] == 0]
            num_columns_cat = [config['loaders']['tabular_data_names'][i] for i in range(len(config['loaders']['tabular_data_names'])) if config['loaders']['tabular_data_names_one_hot'][i] != 0]
            # self.data.at[2, "PatientSex"]=np.nan
            # self.data.at[10, "sten_proc_15_dist_lcx_transformed"]=np.nan
            # self.data.at[6, "sten_proc_1_prox_rca_transformed"]=np.nan

            self.imputer =SimpleImputer(add_indicator=True, strategy='most_frequent')
            

            

 
            self.imputer.fit(self.data[config['loaders']['tabular_data_names']])
            self.data, self.config = add_misisng_indicator_column_names(self.data, self.imputer, config, phase='train')

            self.enc = defaultdict(LabelEncoder)
            fit = self.data[num_columns_cat].apply(lambda x: self.enc[x.name].fit_transform(x))
            fit.apply(lambda x: self.enc[x.name].inverse_transform(x))
            self.data[num_columns_cat] = self.data[num_columns_cat].apply(lambda x: self.enc[x.name].transform(x))


            self.config['loaders']['tabular_data_names_embed_dim'] = determine_unique_values(self.data, config)

            num_columns = [config['loaders']['tabular_data_names'][i] for i in range(len(config['loaders']['tabular_data_names'])) if config['loaders']['tabular_data_names_one_hot'][i] == 0]

       
            ####
            with open(os.path.join(self.config['output'], 'imputer.pkl'),'wb') as f:
                pickle.dump(self.imputer,f)
            with open(os.path.join(self.config['output'], 'encode_cat.pkl'),'wb') as f:
                pickle.dump(self.enc,f)
            self.scaler = preprocessing.StandardScaler().fit(self.data[num_columns])
            self.data[num_columns] = self.scaler.transform(self.data[num_columns])
            with open(os.path.join(self.config['output'], 'preprocessing.pkl'),'wb') as f:
                pickle.dump(self.scaler, f)
            

        if config['labels_names'][0].startswith("sten"):
            self.data = self.data.dropna(subset=config['labels_names'])
        if config['loaders']['only_tabular']:
            self.data= self.data.drop_duplicates(subset=['StudyInstanceUID', 'PatientID'])

        check_nan_after_imputation(self.data)
        self.data[config['loaders']['tabular_data_names']].isna().sum()
        self.data = self.data.to_dict('records')
        
        # def find_nan_keys(d):
        #     nan_keys = [key for key, value in d.items() if isinstance(value, float) and math.isnan(value)]
        #     return nan_keys

        # Find NaN keys
        my_dict = self.data
        #nan_keys = find_nan_keys(my_dict)
     #   print("Keys with NaN:", nan_key)

    def transformations(self):
        self.transforms = [
            LoadImaged(keys=self.features, prune_meta_pattern="(^0008|^6000|^0010|^5004|^5006|^5|^0)"),
            EnsureChannelFirstD(keys=self.features),
            self.resampleORresize(),
            # DeleteItemsd(
            #     keys=self.features[0]+"_meta_dict.[0-9]\\|[0-9]", use_re=True), #FIXME
            self.getMaybePad(),
            self.getCopy1to3Channels(),

            ScaleIntensityd(keys=self.features),
            self.maybeNormalize(config=self.config, features=self.features),
            EnsureTyped(keys=self.features, data_type='tensor'),
            self.maybeToGpu(self.features) if isinstance(self.config['cache_num'], int) else Identityd(keys=self.features),
          #  RandReplicateSliceTransform(keys=self.features, num_slices=self.config['loaders']['Crop_depth'], prob=0.5),
            self.maybeTranslate(),
            self.maybeSpatialScaling(),
            self.maybeTemporalScaling(),
            self.maybeRotate(),
            self.CropTemporal(),

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
       #     self.data_par_train = self.data

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
            if self.config['loaders']['only_tabular']:
                train_ds = monai.data.Dataset(
                    data=self.data_par_train)
            else:
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
        if config['labels_names'][0].startswith('sten') or config['labels_names'][0].startswith('ffr') or config['labels_names'][0].startswith('timi'):
            for label_name in config['labels_names']:
                if config['labels_names'][0].startswith('ffr'):
                    print('consider if we should implement replacement of ffr nan values just for validation...')
                #self.df[label_name] = self.df[label_name].fillna(0)

                self.weights = _prepare_weights(self.df, reweight="inverse", target_name=label_name, max_target=100, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
                w_label_names.append('weights_' + label_name)
                self.df['weights_' + label_name] = self.weights

        if self.config['loss']['name'][0] == 'NNL':
            self.df = self.df.dropna(subset=['event', 'duration_transformed'])

            self.df['event'] = self.df.apply(lambda row: 0 if row['duration_transformed'] > self.config['loss']['censur_date'] else row['event'], axis=1)

            if config["is_already_trained"]:
                with open(os.path.join(self.config['base_model'], 'config.yaml'), 'r') as stream:
                    data_loaded = yaml.safe_load(stream)
                cuts=data_loaded['cuts']
                self.config['cuts'] = cuts
            else:
                if self.config['loaders']['val_method']['saliency'] == True:
                    with open(os.path.join(self.config['output_directory'], 'config.yaml'), 'r') as stream:
                        data_loaded = yaml.safe_load(stream)
                    cuts=data_loaded['cuts']
                    self.config['cuts'] = cuts
                else:
                    cuts=self.config['cuts']

            self.labtrans = LabTransDiscreteTime(
                cuts=cuts,
                predefined_cuts=True,
             #   scheme='quantiles',
                #cuts=config['model']["num_classes"][0], #np.array((0, 100, 200, 300, 400, 500)),
                )
            if self.config['loaders']['mode'] not in ['testing', 'prediction']:
                get_target = lambda df: (self.df[self.config['labels_names'][0]].values, self.df['event'].values)
                print('get_target', get_target(self.df))

                

                durations_train, events_train = get_target(self.df)
                self.df[self.config['labels_names'][0]], self.df['event'] = self.labtrans.transform(durations_train, events_train)


            
            # OLD
            # get_target = lambda df: (self.df[self.config['labels_names'][0]].values, self.df['event'].values)
            
            # target_trains = self.labtrans.transform(*get_target(self.df))
            
            # self.df[self.config['labels_names'][0]] = target_trains[0]
            
            # self.df['event'] = target_trains[1]
           # self.config['labtrans'] = self.labtrans
           # self.config['cuts'] = self.labtrans.cuts
            event = ['event']
        else:
            event = []
        
        if self.config['labels_names'][0].startswith('koronar') or self.config['labels_names'][0].startswith('duration') or self.config['labels_names'][0].startswith('treatment') or self.config['labels_names'][0].startswith('labels_'):
            self.df, labels_to_weight = generate_weights_for_single_label(self.df, self.config)

            self.weights = _get_weights_classification(self.df, labels_to_weight , config)
          #  self.weights_cat = self._compute_weights(self.df, config)
         #   w_label_names = ["event_weights"]
           # self.df[w_label_names[0]]= self.weights




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
        self.data = self.df
        self.data.fillna(value=np.nan, inplace=True)
        if len(config['loaders']['tabular_data_names']) > 0:
        #    self.data = impute_missing(self.data, config)

            if self.config['is_already_trained']:
               # if self.config['is_already_tested']:
                with open(os.path.join(os.path.dirname(self.config['base_model']), 'imputer.pkl'), 'rb') as f:
                    self.imputer = pickle.load(f)
                config_trained = yaml.load(open(os.path.join(self.config['base_model'], "config.yaml"), 'r'), Loader=yaml.FullLoader)
                self.config["loaders"]["tabular_data_names"] = config_trained["loaders"]["tabular_data_names"]
                self.config["loaders"]["tabular_data_names_one_hot"] = config_trained["loaders"]["tabular_data_names_one_hot"]
                self.config["loaders"]["tabular_data_names_embed_dim"] = config_trained["loaders"]["tabular_data_names_embed_dim"]
                

            else:
                with open(os.path.join(self.config['output'], 'imputer.pkl'), 'rb') as f:
                    
                    self.imputer = pickle.load(f)
            self.data, self.config = add_misisng_indicator_column_names(self.data, self.imputer, config, phase='val')
            

            
            num_columns = [config['loaders']['tabular_data_names'][i] for i in range(len(config['loaders']['tabular_data_names'])) if config['loaders']['tabular_data_names_one_hot'][i] == 0]
            num_columns_cat = [config['loaders']['tabular_data_names'][i] for i in range(len(config['loaders']['tabular_data_names'])) if config['loaders']['tabular_data_names_one_hot'][i] != 0]

            
            if self.config['is_already_trained']:
               # if self.config['is_already_tested']:
                with open(os.path.join(os.path.dirname(self.config['base_model']), 'preprocessing.pkl'), 'rb') as f:
                    self.scalar = pickle.load(f)
            else:
                with open(os.path.join(self.config['output'], 'preprocessing.pkl'), 'rb') as f:
                    
                    self.scalar = pickle.load(f)
            if self.config['is_already_trained']:
               # if self.config['is_already_tested']:
                with open(os.path.join(os.path.dirname(self.config['base_model']), 'encode_cat.pkl'), 'rb') as f:
                    self.enc = pickle.load(f)
            else:
                with open(os.path.join(self.config['output'], 'encode_cat.pkl'), 'rb') as f:
                    
                    self.enc = pickle.load(f)
            indices_of_ones = [index for index, value in enumerate(self.config['loaders']['tabular_data_names_one_hot'][:self.config['loaders']['new_columns_nr']]) if value == 1]
            num_columns_cat = [self.config['loaders']['tabular_data_names'][:self.config['loaders']['new_columns_nr']][index] for index in indices_of_ones]
            
            #self.data[num_columns_cat] = self.enc.transform(self.data[num_columns_cat])
            self.data[num_columns_cat] = self.data[num_columns_cat].apply(lambda x: self.enc[x.name].transform(x))

            self.data[num_columns] = self.scalar.transform(self.data[num_columns])
            # if self.config['is_already_trained']:
            #    # if self.config['is_already_tested']:
            #     self.config['loaders']['tabular_data_names_embed_dim'] = determine_unique_values(self.data, config)
        # if config['labels_names][0] startswith "sten_"
        if config['labels_names'][0].startswith("sten"):
            self.data = self.data.dropna(subset=config['labels_names'])
        if config['loaders']['only_tabular']:
            self.data= self.data.drop_duplicates(subset=['StudyInstanceUID', 'PatientID'])

        self.data = self.df[
            self.features + config['labels_names'] +
            ['rowid', "SOPInstanceUID", 'SeriesInstanceUID',
             "StudyInstanceUID", "PatientID"] + event+ w_label_names + ['duration_transformed']+ config['loaders']['tabular_data_names']]
        
        self.data[config['loaders']['tabular_data_names']].isna().sum()
        check_nan_after_imputation(self.data)
        
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
             #   self.maybeToGpu(self.features),
                self.maybeCenterCrop(self.features)
                if self.config['loaders']['mode'] == 'training'
                else monai.transforms.GridPatchd(keys=self.features,
                                            patch_size=(
                                                self.config['loaders']['Crop_height'],
                                                self.config['loaders']['Crop_width'],
                                                self.config['loaders']['Crop_depth']),
                                            pad_mode="constant",),
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
       # self.data_par_val = self.data ## ???
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
            # if self.config['cache_test'] == "True":
            #     val_ds = monai.data.CacheDataset(
            #         data=self.data_par_val,
            #         transform=self.tansformations(),
            #         copy_cache=True,
            #         num_workers=self.config['num_workers'],
            #         cache_num=self.config['cache_num_val'])
            # else:
            if self.config['loaders']['only_tabular']:
                val_ds = monai.data.Dataset(
                    data=self.data_par_val)
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
            if self.config['loaders']['only_tabular']:
                val_ds = monai.data.Dataset(
                    data=self.data_par_val)
            else:
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