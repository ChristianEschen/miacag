from typing import Any
from dinov2.models.vision_transformer import \
    vit_large, vit_small
import torch
from miacag.utils.sql_utils import getDataFromDatabase
from miacag.configs.config import load_config
import os
import numpy as np
from monai.transforms import LoadImage
from monai.transforms import (
    AsChannelFirstd,
    RepeatChanneld,
    AddChanneld,
    EnsureChannelFirstD,
    RandAffined,
    DeleteItemsd,
    AsDiscrete,
    EnsureChannelFirstd,
    CenterSpatialCropd,
    ScaleIntensityd,
    RandTorchVisiond,
    RandRotated,
    RandZoomd,
    Compose,
    Rotate90d,
    NormalizeIntensity,
    SqueezeDimd,
    ToDeviced,
    RandLambdad,
    CopyItemsd,
    LoadImaged,
    EnsureTyped,
    EnsureChannelFirst,
    RandSpatialCropSamplesd,
    LoadImage,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    EnsureType,
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
    Resize,
    RepeatChannel,
    Lambdad,
    ScaleIntensity,
    RandSpatialCropSamplesd)


class FeatureForwarder():
    def __init__(self, config, output_directory) -> None:
        self.config = config
        self.output_directory = output_directory
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        df, _ = getDataFromDatabase(self.config)
        self.df = df
        self.features = self.get_input_features(df)
        self.df['raw_folder'] = self.df['DcmPathFlatten']
        self.set_data_path(self.features)
        self.df = self.df[self.features +
                          ['rowid'] + ['raw_folder']]
        self.data = self.df.to_dict('records')
        self.load_model()
        self.model = self.model.to(self.device)
        self.__transforms__()

    def get_input_features(self, df, features='DcmPathFlatten'):
        if features == 'DcmPathFlatten':
            features = [col for col in
                        df.columns.tolist() if col.startswith(features)]
        else:
            features = features
        return features

    def set_data_path(self, features):
        for feature in features:
            self.df[feature] = self.df[feature].apply(
                        lambda x: os.path.join(self.config['DataSetPath'], x))

    def get_model_from_name(self, name):
        if name == 'dinov2_vits14':
            self.model = vit_small(
                patch_size=14,
                img_size=self.config['loaders']['Resize_height'],
                init_values=1.0,
                block_chunks=0)
        elif name == 'dinov2_vitl16':
            raise NotImplementedError
        # self.model = vit_large(
        #     patch_size=14,
        #     img_size=526,
        #     init_values=1.0,
        #     block_chunks=0)
        #embed_dim
        else:
            raise NotImplementedError
    
    def load_model(self):
        self.get_model_from_name(self.config['model']['backbone'])
        path = os.path.dirname(
            __file__)[0:-len(os.path.basename(os.path.dirname(__file__)))]
        path = os.path.join(path, 'torchhub', '2D',
                            self.config['model']['backbone'],
                            "model.pt")
        self.model.load_state_dict(torch.load(path))

    
    # def normalize(self, x):
# # Use timm's names
# IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# def make_normalize_transform(
#     mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
#     std: Sequence[float] = IMAGENET_DEFAULT_STD,
# ) -> transforms.Normalize:
#     return transforms.Normalize(mean=mean, std=std)

    def __transforms__(self):
        self.transforms = [
              #  EnsureChannelFirst(),
                Resize((self.config['loaders']['Resize_height'],
                        self.config['loaders']['Resize_width'])),
                RepeatChannel(repeats=3),
                ScaleIntensity(),
                EnsureType(data_type='tensor'),
                NormalizeIntensity(
                subtrahend=(0.485, 0.456, 0.406),#(0.43216, 0.394666, 0.37645),
                divisor=(0.229, 0.224, 0.225),#(0.22803, 0.22145, 0.216989),
                channel_wise=True)
                ]

        self.transforms = Compose(self.transforms, log_stats=True)
        #self.transforms.set_random_state(seed=0)
        return self.transforms
    def get_video(self, sample_i):
        transform_video_dcm = Compose([
            LoadImaged(keys=['DcmPathFlatten']),
            EnsureChannelFirstd(keys=['DcmPathFlatten'])])
        
        img = transform_video_dcm(sample_i)
        return img['DcmPathFlatten']
    
    def apply_transforms(self, frame):
        frame = self.transforms(frame)
        return frame

    def save_feature_vector(self, feature_vector, sample_i_dict, frame_idx):
        # extract filename without extension
        filename = os.path.splitext(sample_i_dict['raw_folder'])[0]

        # get name of feature vector folder
        feature_path = os.path.join(
                self.output_directory,
                'feature_vectors',
                filename)
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        # save torch feature vector
        torch.save(feature_vector,
                   os.path.join(feature_path,
                                str(frame_idx) + '.pt'))
    def __call__(self):
        self.model.eval()
        for i in range(len(self.data)):

            sample_i = self.data[i]
            video = self.get_video(sample_i)
            for frame_idx in range(video.shape[-1]):
                frame = video[:,:,frame_idx]
              #  frame = np.expand_dims(frame, 0)
                frame = self.transforms(frame)
                frame = frame.to('cuda')
                frame = frame.unsqueeze(0)
                feature_vector = self.model(frame)
                self.save_feature_vector(feature_vector, sample_i, frame_idx)





if __name__ == '__main__':
    path = "/home/alatar/miacag/my_configs/pretrain_downstream/config.yaml"
    config = load_config(path)
    forwarder = FeatureForwarder(config,
                                 output_directory='/home/alatar/miacag/feature_vectors')
    forwarder()
    print('done forwading')