import os
import numpy as np
import pandas as pd
from dataloader.dataloader_base import DataloaderTrain


class base_monai_loader(DataloaderTrain):
    def __init__(self, image_path, csv_path_file,
                 config, use_complete_data):
        super(base_monai_loader, self).__init__(image_path,
                                                csv_path_file,
                                                config,
                                                use_complete_data)

    def get_input_flow(self, csv):
        features = [col for col in
                    csv.columns.tolist() if col.startswith('flow')]
        return features

    def set_flow_path(self, csv, features, image_path):
        feature_paths = features
        for feature in feature_paths:
            csv[feature] = csv[feature].apply(
                    lambda x: os.path.join(image_path, x))
        return csv

    def get_input_features(self, csv):
        features = [col for col in
                    csv.columns.tolist() if col.startswith('image')]
        return features

    def set_feature_path(self, csv, features, image_path):
        feature_paths = features
        for feature in feature_paths:
            csv[feature] = csv[feature].apply(
                    lambda x: os.path.join(image_path, x))
        return csv


