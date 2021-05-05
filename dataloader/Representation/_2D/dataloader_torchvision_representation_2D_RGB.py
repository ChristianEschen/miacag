import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import monai
from monai.apps import DecathlonDataset
from monai.data import list_data_collate
from torchvision import transforms
from monai.data import PILReader
from PIL import Image
from skimage import io
from monai.transforms import (
    Activations,
    ToPILd,
    AsChannelFirstd,
    RepeatChanneld,
    AddChanneld,
    AsDiscrete,
    CenterSpatialCropd,
    ScaleIntensityd,
    Compose,
    RandZoomd,
    ToNumpyd,
    CopyItemsd,
    RandTorchVisiond,
    LoadImaged,
    LoadImage,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    Spacingd,
    RandRotate90d,
    ToTensord,
    ConcatItemsd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    RandAffined,
    CropForegroundd,
    Resized,
    GaussianSmoothd,
    Lambdad,
    RandSpatialCropSamplesd)
from dataloader.dataloader_base_monai import base_monai_loader
from dataloader.dataloader_base import DataloaderTrain, TwoCropsTransform


def show_img(array):
    array = array.numpy()
    #array = check_data['inputs'][0,:,0,:, :].numpy()
    array = np.transpose(array, (1, 2, 0))
    img = (array - np.min(array)) / \
        (np.amax(array) - np.amin(array) + 1e-8)  # + 1e-8
    img = img * 255.0
    img = np.uint8(img)
    plt.figure(figsize=(2,2))
    plt.imshow(img)
    plt.show()


class base_torchvision_representation_loader(DataloaderTrain):
    def __init__(self, image_path, csv_path_file, config,
                 use_complete_data=False):
        super(base_torchvision_representation_loader, self).__init__(
            image_path,
            csv_path_file,
            config,
            use_complete_data)
        self.features = self.get_input_features(self.csv)
        self.data = self.csv.to_dict('records')

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.image_path,
                                self.csv[self.features].iloc[idx][0])
        image = Image.open(img_name)
        if self.use_complete_data is True:
            labels = torch.tensor(np.int(self.labels.iloc[idx]))
            sample = {'inputs': image, 'labels': labels}
        else:
            sample = {'inputs': image}
        if self.transform:
            sample['inputs'] = self.transform(sample['inputs'])
        return sample

    def get_input_features(self, csv):
        features = [col for col in
                    self.csv.columns.tolist() if col.startswith('inputs')]
        return features

class train_torchvision_representation_loader(base_torchvision_representation_loader):
    def __init__(self, image_path, csv_path_file, config,
                 use_complete_data):
        super(train_torchvision_representation_loader, self).__init__(image_path,
                                                                csv_path_file,
                                                                config,
                                                                use_complete_data)

    def __call__(self):
        # define transforms for image
        train_transforms = [
                transforms.RandomResizedCrop((
                    self.config['loaders']['Resize_height'],
                    self.config['loaders']['Resize_width']),
                    scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))]
        train_transforms = Compose(train_transforms)
        train_transforms = TwoCropsTransform(train_transforms, concat=True)
        self.transform = train_transforms
        return self


class val_torchvision_representation_loader(base_torchvision_representation_loader):
    def __init__(self, image_path, csv_path_file, config, use_complete_data):
        super(val_torchvision_representation_loader, self).__init__(
            image_path,
            csv_path_file,
            config,
            use_complete_data)

    def __call__(self):
        # torchvision
        val_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010))])
        self.transform = val_transforms
        return self

### GRAVEYARD####

    # train_transforms = [
    #             # transforms on images
    #             LoadImaged(keys='inputs'),
    #             AsChannelFirstd(keys='inputs'),
    #             RandSpatialCropSamplesd(
    #                 keys='inputs',
    #                 roi_size=[
    #                     self.config['loaders']['Resize_height'],
    #                     self.config['loaders']['Resize_width']],
    #                 random_size=True,
    #                 num_samples=2),
    #             Resized(
    #                 keys='inputs',
    #                 spatial_size=(self.config['loaders']['Resize_height'],
    #                               self.config['loaders']['Resize_width'])
    #                               ),
    #             ScaleIntensityd(keys='inputs'),
    #             ToTensord(keys='inputs'),
    #             RandTorchVisiond(keys='inputs',
    #                              name="RandomResizedCrop",
    #                              size=(self.config['loaders']['Resize_height'],
    #                                    self.config['loaders']['Resize_width']),
    #                              scale=(0.2, 1.)),
    #             RandTorchVisiond(keys='inputs',
    #                              name="RandomHorizontalFlip"),
    #             # slow
    #             RandTorchVisiond(
    #                 keys='inputs', name='RandomApply',
    #                 transforms=[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
    #                 p=0.8),
    #             # RandTorchVisiond(
    #             #     keys='inputs', name='ColorJitter',
    #             #     brightness=0.4, contrast=0.4,
    #             #     saturation=0.4, hue=0.1),
    #             RandTorchVisiond(keys='inputs',
    #                              name="RandomGrayscale",
    #                              p=0.2),

    #             RandTorchVisiond(keys='inputs',
    #                              name="Normalize",
    #                              mean=(0.4914, 0.4822, 0.4465),
    #                              std=(0.2023, 0.1994, 0.2010))
    #                               ]