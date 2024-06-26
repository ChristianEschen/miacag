from torch.utils.data import DataLoader
import monai
import pandas as pd
from monai.data import list_data_collate, pad_list_data_collate
import torch.distributed as dist
from monai.transforms import (
   # AsChannelFirstd,
    RepeatChanneld,
    #AddChanneld,
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
    SpatialCropd,
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
    # Define a function to aggregate and sort the lists
    def sort_and_aggregate(group):
        sorted_group = group.sort_values('TimeStamp')
        return pd.Series({
            "TimeStamp": sorted_group['TimeStamp'].tolist(),
            'DcmPathFlatten': sorted_group['DcmPathFlatten'].tolist(),
            'SOPInstanceUID': sorted_group['SOPInstanceUID'].tolist(),
            'SeriesInstanceUID': sorted_group['SeriesInstanceUID'].tolist()
        })

    # Group by 'StudyInstanceUID' and 'PatientID', then apply the custom aggregation function
    temp = df.groupby(['StudyInstanceUID', 'PatientID']).apply(sort_and_aggregate)

    # Reset the index to turn the group keys into columns
    temp = temp.reset_index()

    # Merge the aggregated and sorted data back with the original DataFrame
    # Dropping columns that are already aggregated
    df = df.drop(columns=['DcmPathFlatten', 'SOPInstanceUID', 'SeriesInstanceUID', "TimeStamp"])
    df = temp.merge(df, on=["PatientID", "StudyInstanceUID"], how="inner")

    # Drop duplicates based on 'StudyInstanceUID' and 'PatientID'
    df = df.drop_duplicates(['StudyInstanceUID', 'PatientID'])

    return df
def replace_paths(df):
    new_paths = [
        "/home/alatar/miacag/data/angio/sample_data_imagenet/2_extra/0002.dcm",
        "/home/alatar/miacag/data/angio/sample_data_imagenet/3_extra_v2/0003.dcm",
        "/home/alatar/miacag/data/angio/sample_data_imagenet/5_extra_v2/0012.DCM"
    ]
    # set random seed
    random.seed(0)
    for i, row in df.iterrows():
        len_row = len(list(row["DcmPathFlatten"]))
        percent_80 = int(len_row * 0.8)
        indexes_to_replace = random.sample(range(0,len_row),percent_80)
        print('random indexes to replace', indexes_to_replace)
        # copy row["DcmPathFlatten"]
        ifor_val = row["DcmPathFlatten"].copy()
        for idx in indexes_to_replace:
            ifor_val[idx] = random.choice(new_paths)
            
        
        df.at[i,'DcmPathFlatten'] =ifor_val
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
        df["DcmPathFlatten"] = df["DcmPathFlatten"].apply(lambda x: [x])
       # self.df = reorder_rows(self.df)

        if self.config['debugging'] == True:
            self.df= replace_paths(self.df )
            self.df = self.df.loc[self.df.index.repeat(10)].reset_index(drop=True)
            self.df = self.df.sample(frac=1).reset_index(drop=True)


       # self.df= replace_paths(self.df )
     #   self.df['DcmPathFlatten'] = self.df['DcmPathFlatten'].apply(replace_paths)
     #   self.df = self.df.sample(frac=1).reset_index(drop=True)

        # replicate the rows in df 10 times
       # self.df = self.df.loc[self.df.index.repeat(100)].reset_index(drop=True)
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
            self.weights = _prepare_weights(self.df, reweight="inverse", target_name=label_name, max_target=100, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
            w_label_names.append('weights_' + label_name)
            self.df['weights_' + label_name] = self.weights
        #self.data = self.df[self.features + config['labels_na
        if self.config['loss']['name'][0] == 'NNL':
            event = ['event']
        else:
            event = []
        
        if self.config['labels_names'][0].startswith('koronar') or self.config['labels_names'][0].startswith('duration') or self.config['labels_names'][0].startswith('treatment'):
            self.df, labels_to_weight = generate_weights_for_single_label(self.df, self.config)

            self.weights = _get_weights_classification(self.df, labels_to_weight , config)
          #  self.weights_cat = self._compute_weights(self.df, config)


        self.data = self.df[
            self.features + config['labels_names'] +
            ['rowid', "SOPInstanceUID", 'SeriesInstanceUID',
             "StudyInstanceUID", "PatientID", 'labels_predictions'] + event+ w_label_names + ['duration_transformed']]
        # replicate rows 10 times in self.data
        # self.data = pd.DataFrame(np.repeat(
       # self.data = self.data.loc[self.data.index.repeat(100)].reset_index(drop=True)

        # unroll "DcmPathFlatten" column in dataframe to multiple rows
        #self.data = self.data.explode('DcmPathFlatten')
        self.data = self.data.to_dict('records')

    def get_transforms_train(self):
        if self.config['task_type'] == 'mil_classification':
        # this is the deterministic cache transforms
            # 2d or 2d+t
            if self.config['model']['dimension'] == "2D+T":
                patcher = monai.transforms.RandGridPatchd(keys=self.features,
                                                patch_size=(
                                                    self.config['loaders']['Crop_height'],
                                                    self.config['loaders']['Crop_width'],
                                                    self.config['loaders']['Crop_depth']),
                                                    pad_mode="constant",

                                                )
            else:
                patcher = monai.transforms.RandGridPatchd(keys=self.features,
                                                patch_size=(
                                                    self.config['loaders']['Crop_height'],
                                                    self.config['loaders']['Crop_width'],
                                                    self.config['loaders']['Crop_depth']),
                                                pad_mode="constant",
                                                )
                
            train_transforms = [
                    LoadImaged(keys=self.features, prune_meta_pattern="(^0008|^6000|^0010|^5004|^5006|^5|^0)", image_only=True),
                    EnsureChannelFirstD(keys=self.features),
                    self.resampleORresize(),
                    self.getCopy1to3Channels(),
                    self.getClipChannels(),
                    self.getMaybePad(),

                    CenterSpatialCropd(
                        keys=self.features,
                        roi_size=[
                            self.config['loaders']['Crop_height'],
                            self.config['loaders']['Crop_width'],
                            self.config['loaders']['Crop_depth']*4]),
                    EnsureTyped(keys=self.features, data_type='tensor'),
                    self.maybeToGpu(self.features),
                    self.maybeSpatialScaling(),
                    self.maybeTemporalScaling(),
                    self.maybeRotate(),
                    # self.maybeTranslate(),
                    patcher
                    ]


        
        else:

            train_transforms = [
                    LoadImaged(keys=self.features, prune_meta_pattern="(^0008|^6000|^0010|^5004|^5006|^5|^0)"),
                    EnsureChannelFirstD(keys=self.features),
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
                    self.CropTemporal(),
                    ConcatItemsd(keys=self.features, name='inputs'),
                    DeleteItemsd(keys=self.features),
                    ]
        return train_transforms
        
    def __call__(self):
        # define transforms for image

        # 3D transforms
        train_transforms = self.get_transforms_train()
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
        self.data = monai.data.partition_dataset(
                data=self.data,
                num_partitions=dist.get_world_size(),
                shuffle=True,
                even_divisible=True,
            )[dist.get_rank()]

        # create a training data loader'
        if self.config['cache_num'] not in ['standard', 'None']:
            from miacag.dataloader.Classification.dataset_mil import SmartCacheDataset

      #      from miacag.dataloader.Classification.dataset_mil_2024_concat import SmartCacheDataset
            
            train_ds = SmartCacheDataset(
                config=self.config,
                features=self.features,
                data=self.data,
                transform=train_transforms,
                copy_cache=False,
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
                if self.config['model']['dimension'] == "2D+T":
                    from miacag.dataloader.Classification.dataset_mil import MILDataset3D as MILDataset
                    

                else:
                    from miacag.dataloader.Classification.dataset_mil import MILDataset2D as MILDataset
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

        self.features = self.get_input_features(self.df)
        self.set_data_path(self.features)
        #df["DcmPathFlatten"] = df["DcmPathFlatten"].apply(lambda x: [x])
        self.df = reorder_rows(self.df)

        if self.config['debugging'] == True:
            self.df= replace_paths(self.df )
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.df = self.df.dropna(subset=config["labels_names"], how='all')
        self.df = self.df.fillna(value=np.nan)
#        self.df = self.df.loc[self.df.index.repeat(500000)].reset_index(drop=True)

        # create a column with unique values for each row based on index

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
            event = ['event']
        else:
            event = []
            
        if self.config['labels_names'][0].startswith('koronar') or self.config['labels_names'][0].startswith('duration') or self.config['labels_names'][0].startswith('treatment'):

            self.df, labels_to_weight = generate_weights_for_single_label(self.df, self.config)

            self.weights = _get_weights_classification(self.df, labels_to_weight , config)

          #  self.weights_cat = self._compute_weights(self.df, config)


        self.data = self.df[
            self.features + config['labels_names'] +
            ['rowid', "SOPInstanceUID", 'SeriesInstanceUID',
             "StudyInstanceUID", "PatientID", 'labels_predictions'] + event+ w_label_names + ['duration_transformed']]
     #   self.data = self.data.explode('DcmPathFlatten')
        self.data = self.data.to_dict('records')

    def get_transforms_val(self):
        if self.config['task_type'] == 'mil_classification':
            
             val_transforms = [
                    LoadImaged(keys=self.features, prune_meta_pattern="(^0008|^6000|^0010|^5004|^5006|^5|^0)", image_only=True),
                    EnsureChannelFirstD(keys=self.features),
                    self.resampleORresize(),
                    self.getCopy1to3Channels(),
                    self.getClipChannels(),
                  #  self.getMaybePad(),
                    CenterSpatialCropd(
                        keys=self.features,
                        roi_size=[
                            self.config['loaders']['Crop_height'],
                            self.config['loaders']['Crop_width'],
                            self.config['loaders']['Crop_depth']*4]),
                    monai.transforms.GridPatchd(keys=self.features,
                                                patch_size=(
                                                    self.config['loaders']['Crop_height'],
                                                    self.config['loaders']['Crop_width'],
                                                    self.config['loaders']['Crop_depth']),
                                                pad_mode="constant",
                                                ),

                    EnsureTyped(keys=self.features, data_type='tensor'),
                    ]
             

        else:

            val_transforms = [
                    LoadImaged(keys=self.features, prune_meta_pattern="(^0008|^6000|^0010|^5004|^5006|^5|^0)"),
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
                    self.CropTemporal(),
                    ConcatItemsd(keys=self.features, name='inputs'),                    DeleteItemsd(keys=self.features),
                    ]
        return val_transforms

    def __call__(self):


        val_transforms = self.get_transforms_val()
        val_transforms = Compose(val_transforms)
        val_transforms.set_random_state(seed=0)
        #if self.config['use_DDP'] == 'True':
        self.data = monai.data.partition_dataset(
            data=self.data,
            num_partitions=dist.get_world_size(),
            shuffle=False,
            even_divisible=True if self.config['loaders']['mode'] not in ['testing', 'prediction'] else False,
        )[dist.get_rank()]
        rowids = [i["rowid"] for i in self.data]
        if self.config['loaders']['mode'] not in ['prediction', 'testing']:
            if self.config['cache_num'] not in ['standard', 'None']:
                # from miacag.dataloader.Classification.dataset_mil_2024 import CacheDataset, SmartCacheDataset
                # #from miacag.dataloader.Classification.dataset_mil_2024_concat import CacheDataset, SmartCacheDataset

                # val_ds = CacheDataset(
                #     config=self.config,
                #     features=self.features,
                #     data=self.data,
                #     transform=val_transforms,
                #     copy_cache=False,
                #     cache_num=self.config['loaders']['val_method']['cache_num'],
                #     num_workers=int(self.config['num_workers']/2))
                
                from miacag.dataloader.Classification.dataset_mil import MyDataset
                val_ds = MyDataset(
                        config=self.config,
                        features=self.features,
                        data=self.data, transform=val_transforms,
                    )

            elif self.config['cache_num'] == 'standard':
                if self.config['mil_old']:
                    val_ds = MILDataset_old(
                                config=self.config,
                                features=self.features,
                                data=self.data, transform=val_transforms,
                                phase='val'
                            )
                else:
                    if self.config['model']['dimension'] == "2D+T":
                        from miacag.dataloader.Classification.dataset_mil import MILDataset3D as MILDataset
                    else:
                        from miacag.dataloader.Classification.dataset_mil import MILDataset2D as MILDataset
                    val_ds = MILDataset(
                                config=self.config,
                                features=self.features,
                                data=self.data, transform=val_transforms,
                                phase='val'
                            )
            else:
                from miacag.dataloader.Classification.dataset_mil import CacheDataset
                #from miacag.dataloader.Classification.dataset_mil_2024 import Dataset
                val_ds = CacheDataset(
                    config=self.config,
                    features=self.features,
                    data=self.data,
                    transform=val_transforms,
                    copy_cache=True,
                    cache_num=self.config['cache_num'],
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
                from miacag.dataloader.Classification.dataset_mil import MyDataset
                val_ds = MyDataset(
                        config=self.config,
                        features=self.features,
                        data=self.data, transform=val_transforms,
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

