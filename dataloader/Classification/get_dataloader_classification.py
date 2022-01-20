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
        if config['loaders']['format'] in ['dicom']:
            from dataloader.Classification._3D.dataloader_monai_classification_3D import \
                train_monai_classification_loader
            if config['loaders']['val_method']['type'] == 'patches':
                from dataloader.Classification._3D.dataloader_monai_classification_3D import \
                    val_monai_classification_loader
            elif config['loaders']['val_method']['type'] == 'sliding_window':
                from dataloader.Classification._3D.dataloader_monai_classification_3D import \
                    val_monai_classification_loader_SW as val_monai_classification_loader
            else:
                raise ValueError("Invalid validation moode %s" % repr(
                    config['loaders']['val_method']['type']))
        elif config['loaders']['format'] in ['db']:
            from \
                dataloader.Classification.tabular.dataloader_monai_classification_tabular import \
                train_monai_classification_loader, val_monai_classification_loader
        elif config['loaders']['format'] in ['dicom_db']:
            print('NOT implemented')
        else:
            raise ValueError("Invalid validation moode %s" % repr(
                    config['loaders']['val_method']['type']))
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

