import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
from typing import Union, Callable, Sequence
import threading
import time
import warnings
from collections.abc import Callable, Sequence
from copy import copy, deepcopy
from multiprocessing.managers import ListProxy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch.multiprocessing import Manager
from torch.serialization import DEFAULT_PROTOCOL
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset
import monai
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import SUPPORTED_PICKLE_MOD, convert_tables_to_dicts, pickle_hashing
from monai.transforms import (
    Compose,
    Randomizable,
    RandomizableTrait,
    Transform,
    apply_transform,
    convert_to_contiguous,
    reset_ops_id,
)
from sklearn.utils import resample

from monai.utils import MAX_SEED, convert_to_tensor, get_seed, look_up_option, min_version, optional_import
from monai.utils.misc import first
import monai
import random
from monai.transforms import Randomizable
from monai.transforms.compose import MapTransform
import matplotlib.pyplot as plt
import matplotlib


if TYPE_CHECKING:
    from tqdm import tqdm

    has_tqdm = True
else:
    tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")

cp, _ = optional_import("cupy")
lmdb, _ = optional_import("lmdb")
pd, _ = optional_import("pandas")
kvikio_numpy, _ = optional_import("kvikio.numpy")
def replicate_list(lst, m):
    if len(lst) >= m:
        return lst
    else:
        return (lst * ((m // len(lst)) + 1))[:m]

def div_diff_tuple(diff):
    val1 = diff%2
    if val1 == 0:
        return int(diff/2), int(diff/2)
    else:
        return int(diff/2)+1, int(diff/2)

first_rand_spatial_crop_encountered = False

def display_frames(tensor):
    # Assuming tensor is of shape (h, w, depth)
    depth = tensor.shape[3]
    
    # Normalize each frame
    for i in range(depth):
        frame = torch.squeeze(tensor[0, :, :, i])
        # Normalize the frame to 0-255
        norm_frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255
        norm_frame = norm_frame.type(torch.uint8)  # Convert to unsigned int for display
        
        # Plot the frame
        plt.figure(figsize=(5, 5))
        plt.imshow(norm_frame.numpy(), cmap='gray')
        plt.title(f'Frame {i+1}')
        plt.axis('off')
        plt.show()
def is_randomizable_or_not_transform(t):
    global first_rand_spatial_crop_encountered
    # Check if the transform is an instance of RandSpatialCropd
    if isinstance(t, monai.transforms.croppad.dictionary.RandSpatialCropd):
        if not first_rand_spatial_crop_encountered:
            # If this is the first RandSpatialCropd encountered, set the flag and ignore it
            first_rand_spatial_crop_encountered = True
            return False
        else:
            # If this is not the first RandSpatialCropd, consider it in the check
            return True
    # For other transforms, check if they are randomizable or not a MapTransform
    return isinstance(t, Randomizable) or not isinstance(t, MapTransform)


class Dataset(_TorchDataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, config, features, data: Sequence, transform: Callable) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.

        """
        self.config = config
        self.features = features
        self.data = data
        self.transform: Any = transform
        self.normalize = monai.transforms.Compose([
            monai.transforms.ScaleIntensityd(keys=self.features[0]),
            monai.transforms.NormalizeIntensityd(
                keys=self.features[0],
                subtrahend=(0.485, 0.456, 0.406),
                divisor=(0.229, 0.224, 0.225),
                channel_wise=True)])

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)
    
    def _remove_zero_slices(self, data):
        # Initialize a list to hold filtered patches
        filtered_patches = []
        
        # Iterate over each patch in the data
        for patch in data:
            # Compute a mask for non-zero slices across the ch, h, w dimensions
            mask = patch.sum(dim=(0, 1, 2)) != 0
            
            # Use the mask to filter out zero slices
            filtered_patch = patch[:, :, :, mask]
            
            # Append the filtered patch to the list
            filtered_patches.append(filtered_patch)
        
        # Return the list of filtered patches
        return torch.concat(filtered_patches, -1)

class CacheDataset(Dataset):
    """
    Dataset with cache mechanism that can load data and cache deterministic transforms' result during training.

    By caching the results of non-random preprocessing transforms, it accelerates the training data pipeline.
    If the requested data is not in the cache, all transforms will run normally
    (see also :py:class:`monai.data.dataset.Dataset`).

    Users can set the cache rate or number of items to cache.
    It is recommended to experiment with different `cache_num` or `cache_rate` to identify the best training speed.

    The transforms which are supposed to be cached must implement the `monai.transforms.Transform`
    interface and should not be `Randomizable`. This dataset will cache the outcomes before the first
    `Randomizable` `Transform` within a `Compose` instance.
    So to improve the caching efficiency, please always put as many as possible non-random transforms
    before the randomized ones when composing the chain of transforms.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, if the transform is a `Compose` of::

        transforms = Compose([
            LoadImaged(),
            EnsureChannelFirstd(),
            Spacingd(),
            Orientationd(),
            ScaleIntensityRanged(),
            RandCropByPosNegLabeld(),
            ToTensord()
        ])

    when `transforms` is used in a multi-epoch training pipeline, before the first training epoch,
    this dataset will cache the results up to ``ScaleIntensityRanged``, as
    all non-random transforms `LoadImaged`, `EnsureChannelFirstd`, `Spacingd`, `Orientationd`, `ScaleIntensityRanged`
    can be cached. During training, the dataset will load the cached results and run
    ``RandCropByPosNegLabeld`` and ``ToTensord``, as ``RandCropByPosNegLabeld`` is a randomized transform
    and the outcome not cached.

    During training call `set_data()` to update input data and recompute cache content, note that it requires
    `persistent_workers=False` in the PyTorch DataLoader.

    Note:
        `CacheDataset` executes non-random transforms and prepares cache content in the main process before
        the first epoch, then all the subprocesses of DataLoader will read the same cache content in the main process
        during training. it may take a long time to prepare cache content according to the size of expected cache data.
        So to debug or verify the program before real training, users can set `cache_rate=0.0` or `cache_num=0` to
        temporarily skip caching.

    Lazy Resampling:
        If you make use of the lazy resampling feature of `monai.transforms.Compose`, please refer to
        its documentation to familiarize yourself with the interaction between `CacheDataset` and
        lazy resampling.

    """

    def __init__(
        self,
        config: dict,
        features: list,
        data: Sequence,
        transform: Union[Sequence[Callable], Callable, None] = None,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: Union[int, None] = 1,
        progress: bool = True,
        copy_cache: bool = True,
        as_contiguous: bool = True,
        hash_as_key: bool = False,
        hash_func: Callable[..., bytes] = pickle_hashing,
        runtime_cache: Union[bool, str, list, ListProxy] = False,
    ) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: transforms to execute operations on input data.
            cache_num: number of items to be cached. Default is `sys.maxsize`.
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            cache_rate: percentage of cached data in total, default is 1.0 (cache all).
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            num_workers: the number of worker threads if computing cache in the initialization.
                If num_workers is None then the number returned by os.cpu_count() is used.
                If a value less than 1 is specified, 1 will be used instead.
            progress: whether to display a progress bar.
            copy_cache: whether to `deepcopy` the cache content before applying the random transforms,
                default to `True`. if the random transforms don't modify the cached content
                (for example, randomly crop from the cached image and deepcopy the crop region)
                or if every cache item is only used once in a `multi-processing` environment,
                may set `copy=False` for better performance.
            as_contiguous: whether to convert the cached NumPy array or PyTorch tensor to be contiguous.
                it may help improve the performance of following logic.
            hash_as_key: whether to compute hash value of input data as the key to save cache,
                if key exists, avoid saving duplicated content. it can help save memory when
                the dataset has duplicated items or augmented dataset.
            hash_func: if `hash_as_key`, a callable to compute hash from data items to be cached.
                defaults to `monai.data.utils.pickle_hashing`.
            runtime_cache: mode of cache at the runtime. Default to `False` to prepare
                the cache content for the entire ``data`` during initialization, this potentially largely increase the
                time required between the constructor called and first mini-batch generated.
                Three options are provided to compute the cache on the fly after the dataset initialization:

                1. ``"threads"`` or ``True``: use a regular ``list`` to store the cache items.
                2. ``"processes"``: use a ListProxy to store the cache items, it can be shared among processes.
                3. A list-like object: a users-provided container to be used to store the cache items.

                For `thread-based` caching (typically for caching cuda tensors), option 1 is recommended.
                For single process workflows with multiprocessing data loading, option 2 is recommended.
                For multiprocessing workflows (typically for distributed training),
                where this class is initialized in subprocesses, option 3 is recommended,
                and the list-like object should be prepared in the main process and passed to all subprocesses.
                Not following these recommendations may lead to runtime errors or duplicated cache across processes.

        """
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        super().__init__(data=data, config=config, features=features, transform=transform)
        self.config = config
        self.features = features
        self.set_num = cache_num  # tracking the user-provided `cache_num` option
        self.set_rate = cache_rate  # tracking the user-provided `cache_rate` option
        self.progress = progress
        self.copy_cache = copy_cache
        self.as_contiguous = as_contiguous
        self.hash_as_key = hash_as_key
        self.hash_func = hash_func
        self.num_workers = num_workers
        if self.num_workers is not None:
            self.num_workers = max(int(self.num_workers), 1)
        self.runtime_cache = runtime_cache
        self.cache_num = 0
        self._cache: Union[list, ListProxy] = []
        self._hash_keys: list = []
        self.set_data(data)
        self.phase = 'train'

    def set_data(self, data: Sequence) -> None:
        """
        Set the input data and run deterministic transforms to generate cache content.

        Note: should call this func after an entire epoch and must set `persistent_workers=False`
        in PyTorch DataLoader, because it needs to create new worker processes based on new
        generated cache content.

        """
        self.data = data

        def _compute_cache_num(data_len: int):
            self.cache_num = min(int(self.set_num), int(data_len * self.set_rate), data_len)

        if self.hash_as_key:
            # only compute cache for the unique items of dataset, and record the last index for duplicated items
            mapping = {self.hash_func(v): i for i, v in enumerate(self.data)}
            _compute_cache_num(len(mapping))
            self._hash_keys = list(mapping)[: self.cache_num]
            indices = list(mapping.values())[: self.cache_num]
        else:
            _compute_cache_num(len(self.data))
            indices = list(range(self.cache_num))

        if self.runtime_cache in (False, None):  # prepare cache content immediately
            self._cache = self._fill_cache(indices)
            return
        if isinstance(self.runtime_cache, str) and "process" in self.runtime_cache:
            # this must be in the main process, not in dataloader's workers
            self._cache = Manager().list([None] * self.cache_num)
            return
        if (self.runtime_cache is True) or (isinstance(self.runtime_cache, str) and "thread" in self.runtime_cache):
            self._cache = [None] * self.cache_num
            return
        self._cache = self.runtime_cache  # type: ignore
        return

    def _fill_cache(self, indices=None) -> list:
        """
        Compute and fill the cache content from data source.

        Args:
            indices: target indices in the `self.data` source to compute cache.
                if None, use the first `cache_num` items.

        """
        if self.cache_num <= 0:
            return []
        if indices is None:
            indices = list(range(self.cache_num))
        if self.progress and not has_tqdm:
            warnings.warn("tqdm is not installed, will not show the caching progress bar.")
        with ThreadPool(self.num_workers) as p:
            if self.progress and has_tqdm:
                return list(tqdm(p.imap(self._load_cache_item, indices), total=len(indices), desc="Loading dataset"))
            return list(p.imap(self._load_cache_item, indices))


    def _nr_slices_with_zeros(self, data):
        sum_vector = torch.sum(data, dim=(0, 1, 2))
        # get number of slices with zeros
        nr_slices_with_zeros = torch.sum(sum_vector == 0)
        return nr_slices_with_zeros

    def _load_cache_item_i(self, item, first_random):
        item[self.features[0]] = replicate_list(item[self.features[0]], self.config['loaders']['nr_patches'])

        data_i_list = []
        counter = 0
        iter_i = 0
        for data_i_i in item[self.features[0]]:
            iter_i = iter_i + 1
            data_i_i = {
                self.features[0]: data_i_i}
            data_i_i = self.transform(data_i_i, end=first_random, threading=True)
            for n in self.config['labels_names']:
                data_i_i[n] = item[n]
        
            for w in self.config['labels_names']:
                data_i_i["weights_" +w] = item["weights_" + w]
            
            data_i_i["rowid"] = item["rowid"]
            if self.config['loss']['name'][0] == 'NNL':
                data_i_i["event"] = item["event"]
            
            data_i_i["SeriesInstanceUID"] = item["SeriesInstanceUID"][0]
            data_i_i["StudyInstanceUID"] = item["StudyInstanceUID"][0]
            data_i_i["PatientID"] = item["PatientID"]
            data_i_i["duration_transformed"] = item["duration_transformed"]
            data_i_i['labels_predictions'] = item['labels_predictions']
            if self.as_contiguous:
                data_i_i = convert_to_contiguous(data_i_i, memory_format=torch.contiguous_format)
            depth = data_i_i["DcmPathFlatten"].shape[-1]
            
            potential_patches = int(depth/self.config['loaders']['Crop_depth'])
            if self.config['model']['dimension'] == '2D':
                potential_patches = potential_patches + 1
                frames_with_zeros = (potential_patches) * self.config['loaders']['Crop_depth'] \
                    - depth
                potential_frames = potential_patches * self.config['loaders']['Crop_depth']
                potential_frames = potential_frames - frames_with_zeros
                counter = counter + potential_frames            
            else:
                counter = counter +  potential_patches

            data_i_list.append(data_i_i)
            if counter >= self.config['loaders']['nr_patches']:
                return data_i_list
        return data_i_list

    
    def _final_combine_3d(self, data: list):
        data_comb = data[0]
        try:
            stacked_data = torch.concat([i[self.features[0]] for i in data], dim=0)
        except RuntimeError as e:
            print("Caught a RuntimeError:")
            print(e)
            print("\nContents of the data that caused the error:")
            for idx, item in enumerate(data):
                print(f"Data item {idx}: feature '{self.features[0]}' with shape {item[self.features[0]].shape}")
      #  stacked_data = stacked_data.permute(1,0,2,3,4)
        data_comb[self.features[0]] = stacked_data
        
        return data_comb
        
    def replicate_tensor_3d(self, data):
        if data["DcmPathFlatten"].shape[0] < self.config['loaders']['nr_patches']:
            
            # pad the first dimension with zeros
            diff = self.config['loaders']['nr_patches'] - data["DcmPathFlatten"].shape[0]
            if diff < data["DcmPathFlatten"].shape[0]:
                data["DcmPathFlatten"] = torch.concatenate((data["DcmPathFlatten"], data["DcmPathFlatten"][0:diff,:,:,:,:]), dim=0)
            
            else:
                # concatenate the tensor with itself to length nr_patches
                replicate = int(self.config['loaders']['nr_patches']/data["DcmPathFlatten"].shape[0])+1
                data["DcmPathFlatten"] = data["DcmPathFlatten"].repeat(replicate,1,1,1,1)
            # slice the tensor
            data["DcmPathFlatten"]=data["DcmPathFlatten"].as_tensor()
            data["DcmPathFlatten"] = data["DcmPathFlatten"][0:self.config['loaders']['nr_patches'], :, :,:,:]
            return data
        else:
            data["DcmPathFlatten"]=data["DcmPathFlatten"].as_tensor()

            data["DcmPathFlatten"] = data["DcmPathFlatten"][0:self.config['loaders']['nr_patches'], :, :,:,:]
            return data
        
    def replicate_tensor_2d(self, data):
        if data["DcmPathFlatten"].shape[0] < self.config['loaders']['nr_patches']:
            # pad the first dimension with zeros
            diff = self.config['loaders']['nr_patches'] - data["DcmPathFlatten"].shape[0]
            if diff < data["DcmPathFlatten"].shape[0]:
                data["DcmPathFlatten"] = torch.concatenate((data["DcmPathFlatten"], data["DcmPathFlatten"][0:diff,:,:,:]), dim=0)
            else:
                # concatenate the tensor with itself to length nr_patches
                replicate = int(self.config['loaders']['nr_patches']/data["DcmPathFlatten"].shape[0])+1
                data["DcmPathFlatten"] = data["DcmPathFlatten"].repeat(replicate,1,1,1)
            # slice the tensor
            data["DcmPathFlatten"]=data["DcmPathFlatten"].as_tensor()
            data["DcmPathFlatten"] = data["DcmPathFlatten"][0:self.config['loaders']['nr_patches'], :, :,:]
            return data
        else:
            data["DcmPathFlatten"]=data["DcmPathFlatten"].as_tensor()

            data["DcmPathFlatten"] = data["DcmPathFlatten"][0:self.config['loaders']['nr_patches'], :, :,:]
            return data
        
    

    def get_first_random(self, transform):
        count = 0
        for i in transform.transforms:
            count += 1
            if isinstance(i, RandomizableTrait) or not isinstance(i, Transform):
                return count
        return None
            
    def _load_cache_item(self, idx: int, first_random: int = None):
        """
        Args:
            idx: the index of the input data sequence.
        """
        item = self.data[idx]

        first_random = self.transform.get_index_of_first(
            lambda t: isinstance(t, RandomizableTrait) or not isinstance(t, Transform)
        )

        item = self._load_cache_item_i(item, first_random)
        if self.as_contiguous:
            item = convert_to_contiguous(item, memory_format=torch.contiguous_format)
        return item
    
###

    def _final_combine_2d(self, data: list):
        data_comb = data[0]
        try:
            data_list = []
            for i in range(0, len(data)):
                data_list.append(self._remove_zero_slices(data[i][self.features[0]]))

            stacked_data = torch.concat(data_list, dim=-1)
                
        except RuntimeError as e:
            print("Caught a RuntimeError:")
            print(e)
            print("\nContents of the data that caused the error:")
            for idx, item in enumerate(data):
                print(f"Data item {idx}: feature '{self.features[0]}' with shape {item[self.features[0]].shape}")
        stacked_data = stacked_data.permute(3,0, 1,2)
        data_comb[self.features[0]] = stacked_data
        return data_comb
    
    
    def _final_combine(self, data: list):
        if self.config['model']['dimension'] == '2D+T':
            data = self._final_combine_3d(data) # so this is a problem,
            data = self.replicate_tensor_3d(data)
            return data
        elif self.config['model']['dimension'] == '2D':
            data = self._final_combine_2d(data)
            data = self.replicate_tensor_2d(data)
     
            return data
        else:
            raise ValueError('Dimension not supported')
        

        
    def my_get_last(self, transforms):
        return None
    
    def my_get_first_random(self, transforms):
        first_instance = False
        count = 0
        for i in range(len(transforms.transforms)):
            if isinstance(transforms.transforms[i], RandomizableTrait) or not isinstance(
                transforms.transforms[i], Transform):
                if isinstance(transforms.transforms[i], monai.transforms.croppad.dictionary.RandSpatialCropd):
                    if not first_instance:
                        first_instance = True
                        count += 1
                    else:
                        break

            else:
                count += 1
                
        return count

    def shuffle_cache(self):
        """
        Shuffle the cached data items.
        """
        for c in range(0, len(self._cache)):
            random.shuffle(self._cache[c])
            
        random.shuffle(self._cache)
    def _transform_cache(self, index: int):
        cache_index = None
        if self.hash_as_key:
            key = self.hash_func(self.data[index])
            if key in self._hash_keys:
                # if existing in cache, try to get the index in cache
                cache_index = self._hash_keys.index(key)
        elif index % len(self) < self.cache_num:  # support negative index
            cache_index = index

        if cache_index is None:
            # no cache for this index, execute all the transforms directly
            
            return self._load_cache_item(index, first_random=None)

        if self._cache is None:
            raise RuntimeError("cache buffer is not initialized, please call `set_data()` first.")
        data = self._cache[cache_index]
        # runtime cache computation
        if data is None:
            data = self._cache[cache_index] = self._load_cache_item(cache_index)

        # load data from cache and execute from the first random transform
        if not isinstance(self.transform, Compose):
            raise ValueError("transform must be an instance of monai.transforms.Compose.")
        #
       # first_random = self.my_get_first_random(self.transform)
        #first_random = self.my_get_last(self.transform)
        
        # this did not help
        first_random = self.transform.get_index_of_first(
            lambda t: isinstance(t, RandomizableTrait) or not isinstance(t, Transform)
        )
        if first_random is not None:
            data = deepcopy(data) if self.copy_cache is True else data
            data = self.transform(data, start=first_random)
        
        return data
        

    
    def _transform_standard(self, index: int):
        data = self._load_cache_item(index, first_random=None)
        #normalize each tensor
        data = self._final_combine(data)
        temp_data = []
        for i in range(0, data[self.features[0]].shape[0]):
            dict2= {"DcmPathFlatten": data[self.features[0]][i]}
            temp_data.append(self.normalize(dict2)[self.features[0]])
        data[self.features[0]] = torch.stack(temp_data, dim=0)
            


        return data
    def _transform(self, index: int):
        if self.config['loaders']['mode'] not in ['testing', 'predicting']:
           # data = self._final_combine(data)
            return self._transform_cache(index)
        else:
            return self._transform_standard(index)

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        data = self._transform(index)
        data_out = deepcopy(data)
        
        data_out = self._final_combine(data_out)

        temp_data = []
        for i in range(0, data[self.features[0]].shape[0]):
            dict2= {"DcmPathFlatten": data[self.features[0]][i]}
            temp_data.append(self.normalize(dict2)[self.features[0]])
        data[self.features[0]] = torch.stack(temp_data, dim=0)
            

    def __len__(self):
        """
        The dataset length is given by cache_num instead of len(data).

        """
        return self.cache_num
    

class SmartCacheDataset(Randomizable, CacheDataset):
    """
    Re-implementation of the SmartCache mechanism in NVIDIA Clara-train SDK.
    At any time, the cache pool only keeps a subset of the whole dataset. In each epoch, only the items
    in the cache are used for training. This ensures that data needed for training is readily available,
    keeping GPU resources busy. Note that cached items may still have to go through a non-deterministic
    transform sequence before being fed to GPU. At the same time, another thread is preparing replacement
    items by applying the transform sequence to items not in cache. Once one epoch is completed, Smart
    Cache replaces the same number of items with replacement items.
    Smart Cache uses a simple `running window` algorithm to determine the cache content and replacement items.
    Let N be the configured number of objects in cache; and R be the number of replacement objects (R = ceil(N * r),
    where r is the configured replace rate).
    For more details, please refer to:
    https://docs.nvidia.com/clara/clara-train-archive/3.1/nvmidl/additional_features/smart_cache.html
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, if we have 5 images: `[image1, image2, image3, image4, image5]`, and `cache_num=4`, `replace_rate=0.25`.
    so the actual training images cached and replaced for every epoch are as below::

        epoch 1: [image1, image2, image3, image4]
        epoch 2: [image2, image3, image4, image5]
        epoch 3: [image3, image4, image5, image1]
        epoch 3: [image4, image5, image1, image2]
        epoch N: [image[N % 5] ...]

    The usage of `SmartCacheDataset` contains 4 steps:

        1. Initialize `SmartCacheDataset` object and cache for the first epoch.
        2. Call `start()` to run replacement thread in background.
        3. Call `update_cache()` before every epoch to replace training items.
        4. Call `shutdown()` when training ends.

    During training call `set_data()` to update input data and recompute cache content, note to call
    `shutdown()` to stop first, then update data and call `start()` to restart.

    Note:
        This replacement will not work for below cases:
        1. Set the `multiprocessing_context` of DataLoader to `spawn`.
        2. Launch distributed data parallel with `torch.multiprocessing.spawn`.
        3. Run on windows(the default multiprocessing method is `spawn`) with `num_workers` greater than 0.
        4. Set the `persistent_workers` of DataLoader to `True` with `num_workers` greater than 0.

        If using MONAI workflows, please add `SmartCacheHandler` to the handler list of trainer,
        otherwise, please make sure to call `start()`, `update_cache()`, `shutdown()` during training.

    Args:
        data: input data to load and transform to generate dataset for model.
        transform: transforms to execute operations on input data.
        replace_rate: percentage of the cached items to be replaced in every epoch (default to 0.1).
        cache_num: number of items to be cached. Default is `sys.maxsize`.
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        cache_rate: percentage of cached data in total, default is 1.0 (cache all).
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        num_init_workers: the number of worker threads to initialize the cache for first epoch.
            If num_init_workers is None then the number returned by os.cpu_count() is used.
            If a value less than 1 is specified, 1 will be used instead.
        num_replace_workers: the number of worker threads to prepare the replacement cache for every epoch.
            If num_replace_workers is None then the number returned by os.cpu_count() is used.
            If a value less than 1 is specified, 1 will be used instead.
        progress: whether to display a progress bar when caching for the first epoch.
        shuffle: whether to shuffle the whole data list before preparing the cache content for first epoch.
            it will not modify the original input data sequence in-place.
        seed: random seed if shuffle is `True`, default to `0`.
        copy_cache: whether to `deepcopy` the cache content before applying the random transforms,
            default to `True`. if the random transforms don't modify the cache content
            or every cache item is only used once in a `multi-processing` environment,
            may set `copy=False` for better performance.
        as_contiguous: whether to convert the cached NumPy array or PyTorch tensor to be contiguous.
            it may help improve the performance of following logic.
        runtime_cache: Default to `False`, other options are not implemented yet.
    """

    def __init__(
        self,
        config: dict,
        features: list,
        data: Sequence,
        transform: Union[Sequence[Callable], Callable, None] = None,
        replace_rate: float = 0.1,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_init_workers: Union[int, None] = 1,
        num_replace_workers: Union[int, None] = 1,
        progress: bool = True,
        shuffle: bool = True,
        seed: int = 0,
        copy_cache: bool = True,
        as_contiguous: bool = True,
        runtime_cache=False,
    ) -> None:
        if shuffle:
            self.set_random_state(seed=seed)
        self.shuffle = shuffle
        self.phase ='train'
        self._start_pos: int = 0
        self._update_lock: threading.Lock = threading.Lock()
        self._round: int = 1
        self._replace_done: bool = False
        self._replace_mgr: Union[threading.Thread, None] = None
        if runtime_cache is not False:
            raise NotImplementedError("Options other than `runtime_cache=False` is not implemented yet.")

        super().__init__(
            config = config,
            features=features,
            data=data,
            transform=transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_init_workers,
            progress=progress,
            copy_cache=copy_cache,
            as_contiguous=as_contiguous,
            runtime_cache=False,
        )
        if self._cache is None:
            self._cache = self._fill_cache()
        if self.cache_num >= len(data):
            warnings.warn(
                "cache_num is greater or equal than dataset length, fall back to regular monai.data.CacheDataset."
            )
      #  if replace_rate <= 0:
       #     raise ValueError("replace_rate must be greater than 0, otherwise, please use monai.data.CacheDataset.")

        self.num_replace_workers: int | None = num_replace_workers
        if self.num_replace_workers is not None:
            self.num_replace_workers = max(int(self.num_replace_workers), 1)

        self._total_num: int = len(data)
        self._replace_num: int = min(math.ceil(self.cache_num * replace_rate), len(data) - self.cache_num)
        self._replacements: list[Any] = [None for _ in range(self._replace_num)]
        self._replace_data_idx: list[int] = list(range(self._replace_num))
        self._compute_data_idx()
        self.phase = 'train'

    def set_data(self, data: Sequence):
        """
        Set the input data and run deterministic transforms to generate cache content.

        Note: should call `shutdown()` before calling this func.

        """
        if self.is_started():
            warnings.warn("SmartCacheDataset is not shutdown yet, shutdown it directly.")
            self.shutdown()

        if self.shuffle:
            data = copy(data)
            self.randomize(data)
        super().set_data(data)

    def randomize(self, data: Sequence) -> None:
        try:
            self.R.shuffle(data)
            
        except TypeError as e:
            warnings.warn(f"input data can't be shuffled in SmartCacheDataset with numpy.random.shuffle(): {e}.")
            

    def _compute_data_idx(self) -> None:
        """
        Update the replacement data position in the total data.

        """

        for i in range(self._replace_num):
            pos: int = self._start_pos + self.cache_num + i
            if pos >= self._total_num:
                pos -= self._total_num
            self._replace_data_idx[i] = pos

    def is_started(self):
        """
        Check whether the replacement thread is already started.

        """
        return False if self._replace_mgr is None else self._replace_mgr.is_alive()

    def start(self):
        """
        Start the background thread to replace training items for every epoch.

        """
        if not self.is_started():
            self._restart()

    def _restart(self):
        """
        Restart background thread if killed for some reason.

        """
        self._round = 1
        self._replace_mgr = threading.Thread(target=self.manage_replacement, daemon=True)
        self._replace_mgr.start()

    def _try_update_cache(self):
        """
        Update the cache items with new replacement for current epoch.

        """
        with self._update_lock:
            if not self._replace_done:
                return False

            del self._cache[: self._replace_num]
            self._cache.extend(self._replacements)

            self._start_pos += self._replace_num
            if self._start_pos >= self._total_num:
                self._start_pos -= self._total_num

            self._compute_data_idx()

            # ready for next round
            self._round += 1
            self._replace_done = False
            return True

    def update_cache(self):
        """
        Update cache items for current epoch, need to call this function before every epoch.
        If the cache has been shutdown before, need to restart the `_replace_mgr` thread.

        """
        self.start()

        # make sure update is done
        while not self._try_update_cache():
            time.sleep(0.01)

    def _try_shutdown(self):
        """
        Wait for thread lock to shut down the background thread.

        """
        with self._update_lock:
            if self._replace_done:
                self._round = 0
                self._start_pos = 0
                self._compute_data_idx()
                self._replace_done = False
                return True
            return False

    def shutdown(self):
        """
        Shut down the background thread for replacement.

        """
        if not self.is_started():
            return

        # wait until replace mgr is done the current round
        while not self._try_shutdown():
            time.sleep(0.01)
        if self._replace_mgr is not None:
            self._replace_mgr.join(300)

    def _replace_cache_thread(self, index: int):
        """
        Execute deterministic transforms on the new data for replacement.

        """
        pos: int = self._replace_data_idx[index]
        self._replacements[index] = self._load_cache_item(pos)

    def _compute_replacements(self):
        """
        Compute expected items for the replacement of next epoch, execute deterministic transforms.
        It can support multi-threads to accelerate the computation progress.

        """
        with ThreadPool(self.num_replace_workers) as p:
            p.map(self._replace_cache_thread, list(range(self._replace_num)))

        self._replace_done = True

    def _try_manage_replacement(self, check_round):
        """
        Wait thread lock and replace training items in the background thread.

        """
        with self._update_lock:
            if self._round <= 0:
                # shutdown replacement
                self._replace_done = True
                return True, -1

            if self._round != check_round:
                self._compute_replacements()
            return False, self._round

    def manage_replacement(self) -> None:
        """
        Background thread for replacement.

        """
        check_round: int = -1
        done = False
        while not done:
            done, check_round = self._try_manage_replacement(check_round)
            time.sleep(0.01)

    def __len__(self):
        """
        The dataset length is given by cache_num instead of len(data).

        """
        return self.cache_num
    
    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        data = self._transform(index)
        data_out = deepcopy(data)
        data_out = self._final_combine(data_out)
        
        # shuffle each element 
        idx = torch.randperm(data_out[self.features[0]].shape[0])
        data_out[self.features[0]] = data_out[self.features[0]][idx]
        
        
        temp_data_out = []
        for i in range(0, data_out[self.features[0]].shape[0]):
            dict2= {"DcmPathFlatten": data_out[self.features[0]][i]}
            temp_data_out.append(self.normalize(dict2)[self.features[0]])
        data_out[self.features[0]] = torch.stack(temp_data_out, dim=0)
            

        return data_out
    
class MyDataset(CacheDataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, config, features, data: Sequence, transform: Callable) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.

        """
        self.phase = 'test'
        self.config = config
        self.features = features
        self.data = data
        self.transform: Any = transform
        self.normalize = monai.transforms.Compose([
            monai.transforms.ScaleIntensityd(keys=self.features[0]),
            monai.transforms.NormalizeIntensityd(
                keys=self.features[0],
                subtrahend=(0.485, 0.456, 0.406),
                divisor=(0.229, 0.224, 0.225),
                channel_wise=True)])


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        self.as_contiguous = True
        data = self._transform_standard(index)
        return data
    
    

# graveyard
# def _remove_zero_slices(self, data):
#     # Initialize a list to hold filtered patches
#     filtered_patches = []
    
#     # Iterate over each patch in the data
#     for patch in data:
#         # Compute a mask for non-zero slices across the ch, h, w dimensions
#         mask = patch.sum(dim=(0, 1, 2)) != 0
        
#         # Use the mask to filter out zero slices
#         filtered_patch = patch[:, :, :, mask]
        
#         # Append the filtered patch to the list
#         filtered_patches.append(filtered_patch)
    
#     # Return the list of filtered patches
#     return torch.concat(filtered_patches, -1)
