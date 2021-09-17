from torch.utils.data import DataLoader
import torch
from monai.data import list_data_collate, pad_list_data_collate
from torchvision import datasets
import sqlite3
import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit


class ClassificationLoader():
    def __init__(self, DataBasePath, DataSetPath, query, labels_dict) -> None:
        self.DataBasePath = DataBasePath
        self.DataSetPath = DataSetPath
        self.query = query
        self.labels_dict = labels_dict
        self.df = self.getDataFromDatabase()
        self.df = self.df[self.df['labels'].notna()]
        self.df = self.df.replace({'labels': labels_dict})
        self.train_df, self.val_df = self.groupEntriesPrPatient()

    def getDataFromDatabase(self):
        self.connection = sqlite3.connect(self.DataBasePath)
        df = pd.read_sql_query(self.query, self.connection)
        if len(df) == 0:
            print('The requested query does not have any data!')
        df['DcmPathFlatten'] = df['DcmPathFlatten'].apply(
                    lambda x: os.path.join(self.DataSetPath, x))
        return df


    def groupEntriesPrPatient(self):
        '''Grouping entries pr patients'''
        X = self.df.drop('labels', 1)
        y = self.df['labels']
        gs = GroupShuffleSplit(n_splits=2, test_size=.2, random_state=0)
        train_ix, val_ix = next(gs.split(X, y, groups=self.df['PatientID']))
        df_train = self.df.iloc[train_ix]
        df_val = self.df.iloc[val_ix]
        return df_train, df_val

    def get_classification_loader_train(self, config):
        if config['loaders']['format'] == 'avi':
            from dataloader.dataloader_base_video import \
                getVideoTrainTransforms, getVideoTestTransforms
            from dataloader.dataloader_avi_video import VideoDataloaderAVITrain

            transforms_train = getVideoTrainTransforms(
                nr_frames=config['loaders']['Crop_depth'],
                crop_size=(config['loaders']['Crop_height'],
                           config['loaders']['Crop_width']))
            transforms_val = getVideoTestTransforms(
                nr_frames=config['loaders']['Crop_depth'],
                crop_size=(config['loaders']['Crop_height'],
                           config['loaders']['Crop_width']))

            train_loader = VideoDataloaderAVITrain(
                config['loaders']['dataset_path'],
                config['loaders']['DatabasePath'],
                transforms_train)

            train_loader = DataLoader(
                train_loader,
                batch_size=config['loaders']['batchSize'],
                num_workers=config['num_workers'],
                sampler=train_loader.sampler)

            val_loader = VideoDataloaderAVITrain(
                config['loaders']['dataset_path'],
                config['loaders']['DatabasePath'],
                transforms_val)

            with torch.no_grad():
                val_loader = DataLoader(
                    val_loader,
                    batch_size=config['loaders']['batchSize'],
                    num_workers=config['num_workers'],
                    shuffle=False)
            return train_loader, val_loader

        elif config['loaders']['format'] in ['dicom']:
            from dataloader.Classification._3D.dataloader_monai_classification_3D import \
                train_monai_classification_loader
            from dataloader.Classification._3D.dataloader_monai_classification_3D import \
                val_monai_classification_loader

            train_loader = train_monai_classification_loader(
                self.train_df,
                config)

            if config['loaders']['val_method']['type'] == 'patches':
                from dataloader.Classification._3D.dataloader_monai_classification_3D import \
                    val_monai_classification_loader

            elif config['loaders']['val_method']['type'] == 'sliding_window':
                from dataloader.Classification._3D.dataloader_monai_classification_3D import \
                    val_monai_classification_loader_SW as val_monai_classification_loader

            else:
                raise ValueError("Invalid validation moode %s" % repr(
                    config['loaders']['val_method']['type']))
            val_loader = val_monai_classification_loader(
                    self.val_df,
                    config)
            train_loader = DataLoader(
                train_loader(),
                batch_size=config['loaders']['batchSize'],
                shuffle=True,
                num_workers=config['num_workers'],
                collate_fn=list_data_collate,
                pin_memory=True if config['cpu'] == "False" else False,)
            with torch.no_grad():
                val_loader = DataLoader(
                    val_loader(),
                    batch_size=config['loaders']['batchSize'],
                    shuffle=False,
                    num_workers=config['num_workers'],
                    collate_fn=pad_list_data_collate if config['loaders']['val_method']['type'] == 'sliding_window' else list_data_collate,
                    pin_memory=True if config['cpu'] == "False" else False,)
            return train_loader, val_loader
        elif config['loaders']['format'] == 'rgb':
            from dataloader.Representation._2D. \
                dataloader_monai_representation_2D_RGB \
                import val_monai_representation_loader \
                as val_loader
            train_loader = val_loader(
                config['dataset_path'],
                config['DatabasePath'],
                config,
                use_complete_data=False)

            val_loader = val_loader(
                config['ValdataRoot'],
                config['ValdataCSV'],
                config,
                use_complete_data=False)

            if config['loaders']['store_memory'] is True:
                train_loader = datasets.CIFAR10(root=config['dataset_path'],
                                                train=True,
                                                download=True,
                                                transform=train_loader().transform)
                val_loader = datasets.CIFAR10(root=config['ValdataRoot'],
                                        train=False,
                                        download=True,
                                        transform=val_loader().transform)
            train_loader = DataLoader(
                train_loader() if config['loaders']['store_memory'] is False else train_loader,
                drop_last=False,
                batch_size=config['loaders']['batchSize'],
                shuffle=True,
                num_workers=config['num_workers'],
                collate_fn=list_data_collate,
                pin_memory=torch.cuda.is_available(),)

            val_loader = DataLoader(
                val_loader() if config['loaders']['store_memory'] is False else val_loader,
                drop_last=False,
                batch_size=config['loaders']['batchSize'],
                shuffle=False,
                num_workers=config['num_workers'],
                collate_fn=list_data_collate,
                pin_memory=torch.cuda.is_available(),)
            return train_loader, val_loader
        raise ValueError("Data type is not implemented")

    def get_test_type(self, config):
        if config['loaders']['val_method']['type'] == 'patch_lvl':
            test_loader = self.get_classificationloader_patch_lvl_test(config)
            return test_loader
        elif config['loaders']['val_method']['type'] in \
                ['image_lvl+saliency_maps', 'image_lvl']:
            test_loader = self.get_classificationloader_image_lvl_test(config)
            return test_loader
        else:
            raise ValueError(
                "Data test loader type is not implemented %s" % repr(
                    config['loaders']['type']))

    def get_classificationloader_patch_lvl_test(self, config):
        if config['loaders']['format'] == 'dicom':
            if config['loaders']['val_method']['type'] == 'patches':
                from dataloader.Classification._3D. \
                    dataloader_monai_classification_3D import \
                    val_monai_classification_loader
                self.val_loader = val_monai_classification_loader(
                    self.val_df,
                    config)
            else:

                raise ValueError("Invalid validation moode %s" % repr(
                    config['loaders']['val_method']['type']))
            with torch.no_grad():
                self.val_loader = DataLoader(
                    self.val_loader(),
                    batch_size=config['loaders']['batchSize'],
                    shuffle=False,
                    num_workers=config['num_workers'],
                    collate_fn=list_data_collate if
                            config['loaders']['val_method']['type'] != 'sliding_window' else pad_list_data_collate,
                    pin_memory=True if config['cpu'] == "False" else False,)
        else:
            raise ValueError("Invalid data format %s" % repr(
                    config['loaders']['format']))


    def get_classificationloader_image_lvl_test(self, config):
        if config['loaders']['format'] == 'avi':
            from dataloader.dataloader_base_video import \
                getVideoTestTransforms
            from dataloader.dataloader_avi_video import \
                VideoDataloaderAVITest
            transforms_test = getVideoTestTransforms(
                nr_frames=config['loaders']['Crop_depth'],
                crop_size=(config['loaders']['Crop_height'],
                           config['loaders']['Crop_width']))
            test_loader = VideoDataloaderAVITest(
                config['loaders']['TestdataRoot'],
                config['loaders']['TestdataCSV'],
                config['loaders']['Crop_depth'],
                transforms_test)

            test_loader = DataLoader(test_loader,
                                     batch_size=1,
                                     num_workers=config['num_workers'],
                                     shuffle=False)

            return test_loader
        elif config['loaders']['format'] == 'nifty':
            from dataloader.dataloader_monai_classification_3D \
                import test_monai_classification_loader

            test_loader = test_monai_classification_loader(
                config['loaders']['TestdataRoot'],
                config['loaders']['TestdataCSV'],
                config)
            test_loader = DataLoader(
                test_loader(),
                batch_size=config['loaders']['batchSize'],
                shuffle=False,
                num_workers=config['num_workers'],
                collate_fn=list_data_collate,
                pin_memory=torch.cuda.is_available(),)

            return test_loader
