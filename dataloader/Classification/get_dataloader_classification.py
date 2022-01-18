from torch.utils.data import DataLoader
import torch
from monai.data import (
    list_data_collate, pad_list_data_collate,
    ThreadDataLoader)
from torchvision import datasets
import psycopg2
import pandas as pd
import os
from monai.data import DistributedWeightedRandomSampler


class ClassificationLoader():
    def __init__(self, config) -> None:
        self.config = config
        self.getDataFromDatabase()
        self.df = self.df[self.df['labels'].notna()]
        if self.config['loaders']['mode'] == 'testing':
            self.val_df = self.df
        else:
            self.train_df = self.df[self.df['phase'] == 'train']
            self.val_df = self.df[self.df['phase'] == 'val']

    def getDataFromDatabase(self):
        self.connection = psycopg2.connect(
            host=self.config['host'],
            database=self.config['database'],
            user=self.config['username'],
            password=self.config['password'])
        sql = self.config['query'].replace("?table_name",
                                           "\"" + self.config['table_name'] + "\"")
        self.df = pd.read_sql_query(sql, self.connection)
        if len(self.df) == 0:
            print('The requested query does not have any data!')

        #self.writeLabelsTrainDB()
        self.df['image_path1'] = self.df['DcmPathFlatten'].apply(
                    lambda x: os.path.join(self.config['DataSetPath'], x))

    def update(self, records, page_size=2):
        cur = self.connection.cursor()
        values = []
        for record in records:
            value = (record['predictions'],
                     record['confidences'],
                     record['rowid'])
            values.append(value)
        values = tuple(values)
        update_query = """
        UPDATE "{}" AS t
        SET predictions = e.predictions,
            confidences = e.confidences
        FROM (VALUES %s) AS e(predictions, confidences, rowid)
        WHERE e.rowid = t.rowid;""".format(self.config['table_name'])

        psycopg2.extras.execute_values(
            cur, update_query, values, template=None, page_size=100
        )
        self.connection.commit()
        cur.close()
        self.connection.close()

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

            train_ds = train_monai_classification_loader(
                self.train_df,
                config)

            weights = train_ds.weights
            train_ds = train_ds()
            sampler = DistributedWeightedRandomSampler(
                dataset=train_ds,
                weights=weights,
                even_divisible=True,
                shuffle=True)


            if config['loaders']['val_method']['type'] == 'patches':
                from dataloader.Classification._3D.dataloader_monai_classification_3D import \
                    val_monai_classification_loader

            elif config['loaders']['val_method']['type'] == 'sliding_window':
                from dataloader.Classification._3D.dataloader_monai_classification_3D import \
                    val_monai_classification_loader_SW as val_monai_classification_loader

            else:
                raise ValueError("Invalid validation moode %s" % repr(
                    config['loaders']['val_method']['type']))
            val_ds = val_monai_classification_loader(
                    self.val_df,
                    config)
            val_ds = val_ds()

            train_loader = ThreadDataLoader(
                train_ds,
                sampler=sampler,
                batch_size=config['loaders']['batchSize'],
                shuffle=False,
                num_workers=0, #config['num_workers'],
                collate_fn=pad_list_data_collate,
                pin_memory=False,) #True if config['cpu'] == "False" else False,)
            with torch.no_grad():
                val_loader = ThreadDataLoader(
                    val_ds,
                    batch_size=config['loaders']['batchSize'],
                    shuffle=False,
                    num_workers=config['num_workers'],
                    collate_fn=pad_list_data_collate,#pad_list_data_collate if config['loaders']['val_method']['type'] == 'sliding_window' else list_data_collate,
                    pin_memory=False,)
            return train_loader, val_loader, train_ds, val_ds
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
            elif config['loaders']['val_method']['type'] == 'sliding_window':
                
                from dataloader.Classification._3D. \
                    dataloader_monai_classification_3D import \
                    val_monai_classification_loader_SW
                self.val_loader = val_monai_classification_loader_SW(
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
