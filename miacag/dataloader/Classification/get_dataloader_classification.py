from torch.utils.data import DataLoader
import torch
from monai.data import (
    list_data_collate, pad_list_data_collate,
    ThreadDataLoader)
from torchvision import datasets
import psycopg2
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Sequence
import os
from monai.data import DistributedWeightedRandomSampler, DistributedSampler
from miacag.utils.sql_utils import getDataFromDatabase
import numpy as np
from torch.utils.data.dataloader import default_collate
import collections.abc

from torch.utils.data import DistributedSampler as _TorchDistributedSampler

__all__ = ["DistributedSampler", "DistributedWeightedRandomSampler"]

def replicate_until_length(lst, target_length):
    """
    Replicate elements in a list until it reaches the target length.

    Args:
    lst (list): The original list.
    target_length (int): The desired length of the list.

    Returns:
    list: The list with elements replicated until it reaches the target length.
    """
    if not lst:
        raise ValueError("List is empty")

    # Repeat the list until it reaches or exceeds the target length
    repeated_list = lst * (target_length // len(lst) + 1)

    # Slice the list to the exact target length
    return repeated_list[:target_length]

def patches_list_data_collate_fn(batch: collections.abc.Sequence, nr_patches=1, nr_cine_loops=1, batch_size=1):
    '''
        Combine instances from a list of dicts into a single dict, by stacking them along first dim
        [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
        followed by the default collate which will form a batch BxNx3xHxW
    '''
    batch_data =batch

    # for i, item in enumerate(batch):
    #     data = item[0]
    #     data['inputs'] = torch.concatenate([it["inputs"] for it in item], dim=-1)
    #  #   data["inputs"] = torch.stack([ix["inputs"] for ix in item], dim=0)
    #     # zero pad if nr_patches is not reached
    #     if data["inputs"].shape[-1] < nr_patches:
    #         # zero pad
    #         diff = nr_patches - data["inputs"].shape[-1]
    #         data["inputs"] = torch.nn.functional.pad(data["inputs"], (diff, 0, 0, 0),  mode='constant', value=0)
    #     # trim to nr_patches
    #     data["inputs"] = data["inputs"][:, :, :, 0:nr_patches]
    #     # drop SOPInstanceUID from dict
    #     data.pop('SOPInstanceUID', None)
    #     data["DcmPathFlatten"] = data["DcmPathFlatten"][0]
    #     data["SeriesInstanceUID"] = data["SeriesInstanceUID"][0]
    #     batch[i] = data
    batch = default_collate(batch)
   # batch['inputs'] = batch['inputs'].permute(0, 4, 1, 2, 3)
    return batch

def patches_list_data_collate(batch: collections.abc.Sequence, nr_patches=1, nr_cine_loops=1, batch_size=1):
    '''
        Combine instances from a list of dicts into a single dict, by stacking them along first dim
        [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
        followed by the default collate which will form a batch BxNx3xHxW
    '''
    batch_data =batch

    for i, item in enumerate(batch):
        data = item[0]
        data['inputs'] = torch.concatenate([it["inputs"] for it in item], dim=-1)
     #   data["inputs"] = torch.stack([ix["inputs"] for ix in item], dim=0)
        # zero pad if nr_patches is not reached
        if data["inputs"].shape[-1] < nr_patches:
            # zero pad
            diff = nr_patches - data["inputs"].shape[-1]
            data["inputs"] = torch.nn.functional.pad(data["inputs"], (diff, 0, 0, 0),  mode='constant', value=0)
        # trim to nr_patches
        data["inputs"] = data["inputs"][:, :, :, 0:nr_patches]
        # drop SOPInstanceUID from dict
        data.pop('SOPInstanceUID', None)
        data["DcmPathFlatten"] = data["DcmPathFlatten"][0]
        data["SeriesInstanceUID"] = data["SeriesInstanceUID"][0]
        batch[i] = data
    batch = default_collate(batch)
    batch['inputs'] = batch['inputs'].permute(0, 4, 1, 2, 3)
    return batch

def patches_list_data_collate_read_patches_individual(batch: collections.abc.Sequence, nr_cine_loops=1, nr_patches=1, batch_size=1):
    '''
        Combine instances from a list of dicts into a single dict, by stacking them along first dim
        [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
        followed by the default collate which will form a batch BxNx3xHxW
    '''
    grouped = {}

    for item in batch:
        key = (item['PatientID'], item['StudyInstanceUID'])
        if key not in grouped:
            grouped[key] = {}
            # but the inputs should be a list
            grouped[key]['inputs'] = []
        grouped[key]['inputs'].append(item['inputs'])  #
        item.pop('inputs', None)
        grouped[key].update(item)
        grouped[key].pop('SOPInstanceUID', None)
        grouped[key]["SeriesInstanceUID"] = grouped[key]["SeriesInstanceUID"][0]
        
        
    # Stack images for each group along the second dimension
    for key in grouped:
        grouped[key]['inputs'] = torch.concatenate(grouped[key]['inputs'], dim=-1)  # Stacking along last dimension

    for key in grouped:
        # zero pad if nr_patches is not reached
        if grouped[key]['inputs'].shape[-1]  < nr_patches:
            # zero pad
            diff = nr_patches - grouped[key]['inputs'].shape[-1]
            grouped[key]['inputs'] = torch.nn.functional.pad(grouped[key]['inputs'], (diff, 0, 0, 0,),  mode='constant', value=0)
        else:
            # trim
            grouped[key]['inputs'] = grouped[key]['inputs'][:, :, :, 0:nr_patches]
        
    # Prepare the final batch
    batch = list(grouped.values())
    # trim or pad batch
    if len(batch) >= batch_size:
        batch = batch[:batch_size]

    # If the list is shorter than the target length, pad it.
    elif len(batch) < batch_size:
        diff = batch_size - len(batch)
        batch = replicate_until_length(batch, batch_size)
    else:
        print('not implemented in this costum collate fn')
    

    # Apply the default collate if needed for other operations
    batch = default_collate(batch)

    # If there's any additional manipulation needed, do it here
    batch['inputs'] = batch['inputs'].permute(0, 4, 1, 2, 3)

    return batch


class DistributedSampler(_TorchDistributedSampler):
    """
    Enhance PyTorch DistributedSampler to support non-evenly divisible sampling.

    Args:
        dataset: Dataset used for sampling.
        even_divisible: if False, different ranks can have different data length.
            for example, input data: [1, 2, 3, 4, 5], rank 0: [1, 3, 5], rank 1: [2, 4].
        num_replicas: number of processes participating in distributed training.
            by default, `world_size` is retrieved from the current distributed group.
        rank: rank of the current process within `num_replicas`. by default,
            `rank` is retrieved from the current distributed group.
        shuffle: if `True`, sampler will shuffle the indices, default to True.
        kwargs: additional arguments for `DistributedSampler` super class, can be `seed` and `drop_last`.

    More information about DistributedSampler, please check:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler.

    """

    def __init__(
        self,
        dataset: Dataset,
        even_divisible: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, **kwargs)

        if not even_divisible:
            data_len = len(dataset)  # type: ignore
            if data_len < self.num_replicas:
                raise ValueError("the dataset length is less than the number of participating ranks.")
            extra_size = self.total_size - data_len
            if self.rank + extra_size >= self.num_replicas:
                self.num_samples -= 1
            self.total_size = data_len

class DistributedWeightedRandomSampler(DistributedSampler):
    """
    Extend the `DistributedSampler` to support weighted sampling.
    Refer to `torch.utils.data.WeightedRandomSampler`, for more details please check:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler.

    Args:
        dataset: Dataset used for sampling.
        weights: a sequence of weights, not necessary summing up to one, length should exactly
            match the full dataset.
        num_samples_per_rank: number of samples to draw for every rank, sample from
            the distributed subset of dataset.
            if None, default to the length of dataset split by DistributedSampler.
        generator: PyTorch Generator used in sampling.
        even_divisible: if False, different ranks can have different data length.
            for example, input data: [1, 2, 3, 4, 5], rank 0: [1, 3, 5], rank 1: [2, 4].'
        num_replicas: number of processes participating in distributed training.
            by default, `world_size` is retrieved from the current distributed group.
        rank: rank of the current process within `num_replicas`. by default,
            `rank` is retrieved from the current distributed group.
        shuffle: if `True`, sampler will shuffle the indices, default to True.
        kwargs: additional arguments for `DistributedSampler` super class, can be `seed` and `drop_last`.

    """

    def __init__(
        self,
        dataset: Dataset,
        weights: Sequence[float],
        num_samples_per_rank: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        even_divisible: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        config: dict = None,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            even_divisible=even_divisible,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            **kwargs,
        )
        self.weights = weights
        self.num_samples_per_rank = num_samples_per_rank if num_samples_per_rank is not None else self.num_samples
        self.generator = generator
        self.config = config

    def __iter__(self):
        indices = list(super().__iter__())
        weights = torch.as_tensor([self.weights[i] for i in indices], dtype=torch.float32)
        if self.config["cpu"] == "False":
            device = "cuda:{}".format(os.environ['LOCAL_RANK'])
        else:
            device = 'cpu'
        device = torch.device(device)
        weights = weights.to(device)
            # sample based on the provided weights
        rand_tensor = torch.multinomial(weights, self.num_samples_per_rank, True, generator=self.generator)

        for i in rand_tensor:
            yield indices[i]

    def __len__(self):
        return self.num_samples_per_rank



class DistributedBalancedRandomSampler(DistributedSampler):

    def __init__(
        self,
        dataset: Dataset,
        weights: Sequence[float],
        num_samples_per_rank: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        even_divisible: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            even_divisible=even_divisible,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            **kwargs,
        )
        self.labels = [i['event'] for i in self.dataset.data]
        self.num_samples_per_rank = num_samples_per_rank if num_samples_per_rank is not None else self.num_samples
        self.generator = generator

    def __iter__(self):
        indices = list(super().__iter__())
        labels = torch.tensor(self.labels)

        # Find the unique labels
        unique_labels = labels.unique()

        # Calculate the number of unique labels
        num_unique = len(unique_labels)

        # Calculate how many times we need to repeat the pattern to reach the desired length
        num_repeats = len(labels) // num_unique + (len(labels) % num_unique > 0)

        # Create an empty tensor to hold the final sequence
        balanced_sequence = torch.empty((num_repeats * num_unique,), dtype=torch.long)

        # Fill the balanced_sequence tensor with randomly shuffled unique labels
        for i in range(num_repeats):
            # Shuffle the unique labels
            shuffled = unique_labels[torch.randperm(num_unique)]
            # Place the shuffled labels into the balanced_sequence tensor
            start_index = i * num_unique
            end_index = start_index + num_unique
            balanced_sequence[start_index:end_index] = shuffled

        # Trim the balanced_sequence to match the length of the original labels list
        balanced_sequence = balanced_sequence[:len(labels)]

        # Convert back to a list if necessary
        balanced_sequence_list = balanced_sequence.tolist()
        rand_tensor = torch.tensor(balanced_sequence_list)
        for i in rand_tensor:
            yield indices[i]

    def __len__(self):
        return self.num_samples_per_rank


# def list_data_collate_mil(batch: collections.abc.Sequence):
#     '''
#         Combine instances from a list of dicts into a single dict, by stacking them along first dim
#         [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
#         followed by the default collate which will form a batch BxNx3xHxW
#     '''

#     for i, item in enumerate(batch):
#         data = item
#         data["inputs"] = torch.stack([ix["inputs"] for ix in batch], dim=0)
#         batch[i] = data
#     return default_collate(batch)


class ClassificationLoader():
    def __init__(self, config) -> None:
        self.config = config
        self.df, self.connection = getDataFromDatabase(self.config)
        
        
        if self.config['loaders']['mode'] != 'prediction':
            # set iloc of 0 to nan
            
            self.df = self.df.dropna(subset=config["labels_names"], how='all')
        if self.config['loaders']['mode'] == 'prediction':
            for col in self.config['labels_names']:
                #try:
                self.df[col].values[:] = 0
                # except:
                #     print('column not found')
                #     pass
        print('please changes these')
        if self.config['loaders']['mode'] in ['prediction', 'testing']:
            if self.config['debugging'] == True:
                self.val_df = self.df[self.df['phase'].isin(['train', 'arcade_train'])]
            else:
                self.val_df = self.df
        else:
           # self.train_df = self.df
           # self.val_df = self.df
            if self.config['debugging'] == True:

                self.train_df = self.df[self.df['phase'].isin(['train', 'arcade_train'])]
                self.val_df = self.df[self.df['phase'].isin(['train', 'arcade_train'])]
            else:
                self.train_df = self.df[self.df['phase'].isin(['train', 'arcade_train'])]
                self.val_df = self.df[self.df['phase'].isin(['val','arcade_val'])]
                
 
    def get_classification_loader_train(self, config):
        if config['loaders']['format'] in ['dicom']:
            if config['task_type'] in ['classification', 'regression']:
                from miacag.dataloader.Classification._3D.dataloader_monai_classification_3D import \
                    train_monai_classification_loader, val_monai_classification_loader
            elif config['task_type'] in ["mil_classification"]:
                if config['model']['dimension'] in ['2D+T']:
                    from miacag.dataloader.Classification.dataloader_monai_classification_mil import \
                        train_monai_classification_loader
                    from miacag.dataloader.Classification.dataloader_monai_classification_mil import \
                        val_monai_classification_loader
                elif config['model']['dimension'] in ['2D']:
                    from miacag.dataloader.Classification.dataloader_monai_classification_mil import \
                        train_monai_classification_loader, val_monai_classification_loader
                elif config['model']['dimension'] in ['1D']:
                    from miacag.dataloader.Classification._1D.Feature_vector_dataset import \
                        train_monai_classification_loader, val_monai_classification_loader
                else:
                    raise ValueError('model dimension is not implemented')
        elif config['loaders']['format'] in ['db']:
            from \
                miacag.dataloader.Classification.tabular.dataloader_monai_classification_tabular import \
                train_monai_classification_loader, val_monai_classification_loader
        elif config['loaders']['format'] in ['dicom_db']:
            print('NOT implemented')
        else:
            raise ValueError("Invalid validation moode %s" % repr(
                    config['loaders']['val_method']['type']))
       # if config['model']['dimension'] != '1D':
        train_ds = train_monai_classification_loader(
            self.train_df,
            config)

        if config['weighted_sampler'] == 'True':
            weights = train_ds.weights
            train_ds = train_ds()
            if self.config['loss']['name'][0] == 'CE':
                sampler = DistributedWeightedRandomSampler(
                    dataset=train_ds,
                    weights=weights,
                    even_divisible=True,
                    shuffle=True)
            elif self.config['loss']['name'][0] == 'NNL':
                # weights = [train_ds.data[i]['weights_duration_transformed'] for i in range(0, len(train_ds.data))]
                # sampler = DistributedBalancedRandomSampler(
                #     weights=weights,
                #     dataset=train_ds,
                #     even_divisible=True,
                #     shuffle=True)
                
                
                # sampler = DistributedSampler(
                #     dataset=train_ds,
                #     even_divisible=True,
                #     shuffle=True)
                weights = [train_ds.data[i]['weights'] for i in range(0, len(train_ds.data))]
                sampler = DistributedWeightedRandomSampler(
                    dataset=train_ds,
                    weights=weights,
                    even_divisible=True,
                    shuffle=True,
                    config=config)
            else:
                raise ValueError('sampler not implemented')
        else:
            train_ds = train_ds()
            sampler = DistributedSampler(
                dataset=train_ds,
                even_divisible=True,
                shuffle=True)
            
        val_ds = val_monai_classification_loader(
                self.val_df,
                config)
        val_ds = val_ds()

        if isinstance(config['cache_num'], int):

            train_loader = ThreadDataLoader(
                train_ds,
                sampler=sampler,
                batch_size=config['loaders']['batchSize'],
                shuffle=False,
                #  persistent_workers=True,
                num_workers=0, #config['num_workers'],
                collate_fn=list_data_collate, #patches_list_data_collate_read_patches_individual, #pad_list_data_collate,
                pin_memory=False) #True if config['cpu'] == "False" else False,)
            with torch.no_grad():
                val_loader = ThreadDataLoader(
                    val_ds,
                    batch_size=config['loaders']['batchSize'],
                    shuffle=False,
                    num_workers=0,
                    collate_fn=list_data_collate, #patches_list_data_collate_read_patches_individual, #pad_list_data_collate, #pad_list_data_collate if config['loaders']['val_method']['type'] == 'sliding_window' else list_data_collate,
                    pin_memory=False)
        else:
            train_loader = DataLoader(
                train_ds,
                sampler=sampler,
                batch_size=config['loaders']['batchSize'],
                shuffle=False,
                num_workers=config['num_workers'],
                collate_fn=list_data_collate, #patches_pad_list_data_collate_read_patches_individual, #pad_list_data_collate,
                pin_memory=True) #True if config['cpu'] == "False" else False,)
            with torch.no_grad():
                val_loader = DataLoader(
                    val_ds,
                    batch_size=config['loaders']['batchSize'],
                    shuffle=False,
                    num_workers=config['num_workers'],
                    collate_fn=list_data_collate, #patches_list_data_collate_read_patches_individual, #pad_list_data_collate, #pad_list_data_collate if config['loaders']['val_method']['type'] == 'sliding_window' else list_data_collate,
                    pin_memory=False) 
        return train_loader, val_loader, train_ds, val_ds

    def get_classificationloader_patch_lvl_test(self, config):
        if config['loaders']['format'] == 'dicom':
            if config['task_type'] in ['classification', "regression"]:
                from miacag.dataloader.Classification._3D. \
                    dataloader_monai_classification_3D import \
                    val_monai_classification_loader
                self.val_ds = val_monai_classification_loader(
                    self.val_df,
                    config)
            elif config['task_type'] in ["mil_classification"]:
                if config['model']['dimension'] in ['2D']:
                    from miacag.dataloader.Classification.dataloader_monai_classification_mil import \
                        val_monai_classification_loader
                elif config['model']['dimension'] in ['2D+T']:
                    from miacag.dataloader.Classification.dataloader_monai_classification_mil import \
                        val_monai_classification_loader
                else:
                    raise ValueError('this dimension is not implemented')
                self.val_ds = val_monai_classification_loader(
                    self.val_df,
                    config)
            else:
                raise ValueError("not implemented")

        elif config['loaders']['format'] == 'db':
            if config['loaders']['val_method']['type'] == 'full':
                from miacag.dataloader.Classification.tabular. \
                    dataloader_monai_classification_tabular import \
                    val_monai_classification_loader
                self.val_ds = val_monai_classification_loader(
                    self.val_df,
                    config)
            else:

                raise ValueError("Invalid validation moode %s" % repr(
                    config['loaders']['val_method']['type']))
        else:
            raise ValueError("Invalid data format %s" % repr(
                    config['loaders']['format']))
        self.val_ds = self.val_ds()
        with torch.no_grad():
            self.val_loader = DataLoader(
                self.val_ds,
                batch_size=config['loaders']['batchSize'],
                shuffle=False,
                num_workers=4, #0
                collate_fn=list_data_collate if
                        config['loaders']['val_method']['type'] != 'sliding_window' else pad_list_data_collate,
               # pin_memory=False if config['cpu'] == "False" else True,)
                pin_memory=False,
                drop_last=True if self.config['loss']['name'][0] == 'NNL' else False)
