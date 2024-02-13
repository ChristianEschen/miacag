from torch.utils.data import DataLoader
import monai
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
    ToDeviced,
    RandLambdad,
    CopyItemsd,
    LoadImaged,
    EnsureTyped,
    RandSpatialCropSamplesd,
    LoadImage,
    MapTransform,
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
# import matplotlib
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# matplotlib.use( 'tkagg' )
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

class train_monai_classification_loader(base_monai_loader):
    def __init__(self, df, config):
        super(base_monai_loader, self).__init__(
            df,
            config)
        if config['weighted_sampler'] == 'True':
            self.getSampler()
        self.features = self.get_input_features(self.df)
        self.set_data_path(self.features)
        w_label_names = []
        for label_name in config['labels_names']:
            
            self.weights = self._prepare_weights(reweight="inverse", target_name=label_name, max_target=100, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
            w_label_names.append('weights_' + label_name)
            self.df['weights_' + label_name] = self.weights
        self.data = self.df[self.features + config['labels_names'] + ['rowid'] + ['event'] + w_label_names]
        # make histogram of self.df['weights_' + label_name] to see if it is working
       
        self.data = self.data.to_dict('records')

    def transformations(self):
        self.transforms = [
            LoadImaged(keys=self.features, prune_meta_pattern="(^0008|^6000|^0010|^5004|^5006|^5|^0)"),
            EnsureChannelFirstD(keys=self.features),
            self.resampleORresize(),
            DeleteItemsd(
                keys=self.features[0]+"_meta_dict.[0-9]\\|[0-9]", use_re=True),
            self.getMaybePad(),
            self.getCopy1to3Channels(),
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
        if self.config['cache_num'] != 'None':
            train_ds = monai.data.SmartCacheDataset(
                data=self.data_par_train,
                transform=self.transformations(),
                copy_cache=True,
                cache_num=self.config['cache_num'],
                num_init_workers=int(self.config['num_workers']/2),
                replace_rate=self.config['replace_rate'],
                num_replace_workers=int(self.config['num_workers']/2))
        else:
            train_ds = monai.data.CacheDataset(
                data=self.data_par_train,
                transform=self.transformations(),
                copy_cache=True,
                num_workers=self.config['num_workers'])
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
     #   self.data = self.df[self.features + config['labels_names'] + ['rowid']]
        w_label_names = []

        for label_name in config['labels_names']:
            
            self.weights = self._prepare_weights(reweight="inverse", target_name=label_name, max_target=100, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
            w_label_names.append('weights_' + label_name)
            self.df['weights_' + label_name] = self.weights
        self.data = self.df[self.features + config['labels_names'] + ['rowid'] + ['event'] + w_label_names]
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
                self.maybeDeleteMeta(),
                self.getMaybePad(),
                self.getCopy1to3Channels(),
                ScaleIntensityd(keys=self.features),
                self.maybeNormalize(),
                EnsureTyped(keys=self.features, data_type='tensor'),
                self.maybeToGpu(self.features),
                self.maybeCenterCrop(self.features),
                ConcatItemsd(keys=self.features, name='inputs'),
                self.maybeDeleteFeatures(),
                ]

        self.transforms = Compose(self.transforms, log_stats=True)
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
            val_ds = monai.data.CacheDataset(
                data=self.data_par_val,
                transform=self.tansformations(),
                copy_cache=True,
                num_workers=self.config['num_workers'],
                cache_num=self.config['cache_num_val'])
        else:
            if self.config['cache_test'] == "True":
                val_ds = monai.data.CacheDataset(
                        data=self.data_par_val,
                        transform=self.tansformations(),
                        copy_cache=True,
                        num_workers=self.config['num_workers'])
            elif self.config['cache_test'] == "False":
                val_ds = monai.data.Dataset(
                        data=self.data_par_val,
                transform=self.tansformations())
            elif self.config['cache_test'] == "persistant":
                cachDir = os.path.join(
                    self.config['model']['pretrain_model'],
                    'persistent_cache')
                val_ds = monai.data.PersistentDataset(
                        data=self.data_par_val, transform=self.tansformations(),
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
                ScaleIntensityd(keys=self.features),
                NormalizeIntensityd(keys=self.features,
                                    channel_wise=True),
                EnsureTyped(keys=self.features, data_type='tensor'),

                ConcatItemsd(keys=self.features, name='inputs'),
                ]
        val_transforms = Compose(val_transforms, log_stats=True)
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