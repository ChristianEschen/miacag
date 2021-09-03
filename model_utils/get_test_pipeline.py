from model_utils.eval_utils import val_one_epoch
from model_utils.eval_utils import eval_one_step
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from preprocessing.pre_process import appendDataframes


class TestPipeline():
    def get_test_pipeline(self, model, criterion, config, test_loader,
                          device, init_metrics, increment_metrics,
                          normalize_metrics,
                          running_metric_test, running_loss_test):

        if config['task_type'] in ["classification"]:
            self.get_test_classification_pipeline(model, criterion,
                                                  config, test_loader,
                                                  device, init_metrics,
                                                  increment_metrics,
                                                  normalize_metrics,
                                                  running_metric_test,
                                                  running_loss_test)
        elif config['loaders']['task_type'] == "image2image":
            self.get_test_segmentation_pipeline(model, criterion,
                                                config, test_loader,
                                                device, init_metrics,
                                                increment_metrics,
                                                normalize_metrics,
                                                running_metric_test,
                                                running_loss_test)
        else:
            raise ValueError("Task type is not implemented")

    def get_test_classification_pipeline(self, model, criterion,
                                         config, test_loader,
                                         device, init_metrics,
                                         increment_metrics,
                                         normalize_metrics,
                                         running_metric_test,
                                         running_loss_test):

        if config['loaders']['val_method']['type'] in ['patches',
                                                       'sliding_window']:
            metrics, confidences = val_one_epoch(
                model, criterion, config,
                test_loader, device,
                running_metric_val=running_metric_test,
                running_loss_val=running_loss_test,
                saliency_maps=False)

        elif config['loaders']['val_method']['type'] in \
                ['image_lvl', 'image_lvl+saliency_maps']:
            if config['loaders']['backend'] == 'monai':
                from models.image2scalar_utils.utils_3D.test_utils_monai \
                    import _test_one_epoch
                if config['loaders']['val_method']['type'] == \
                        'image_lvl+saliency_maps':
                    saliency_maps = True
                else:
                    saliency_maps = False
                metrics = _test_one_epoch(
                    model, criterion, config,
                    test_loader, device,
                    eval_one_step,
                    init_metrics,
                    increment_metrics,
                    normalize_metrics,
                    saliency_maps,
                    running_metric_test=running_metric_test,
                    running_loss_test=running_loss_test)
            elif config['loaders']['backend'] == 'costum':
                from models.image2scalar_utils.utils_3D.test_utils import \
                    _test_one_epoch
                if config['loaders']['val_method']['type'] == \
                        'image_lvl+saliency_maps':
                    saliency_maps = True
                else:
                    saliency_maps = False
                metrics = _test_one_epoch(
                    model, criterion, config,
                    test_loader, device,
                    eval_one_step,
                    init_metrics,
                    increment_metrics,
                    normalize_metrics,
                    saliency_maps,
                    running_metric_test=running_metric_test,
                    running_loss_test=running_loss_test)
            else:
                raise ValueError(
                    "backend methodd not implemented %s" % repr(
                        config['loaders']['backend']))
        else:
            raise ValueError(
                "test pipeline is not implemented %s" % repr(
                    config['loaders']['val_method']['type']))
        
        df_test = self.read_validationCSV(config)
        df_test = self.buildCsvResults(df_test, confidences)
        df_test = df_test.astype('str')

        self.save_pre_val_csv(config['PreValCSV'], df_test)

        df_test = df_test.loc[:, ~df_test.columns.duplicated()]
        df_test.to_csv(
            os.path.join(config['model']['pretrain_model'], 'results.csv'),
            index=False)
        
        acc = {'accuracy correct': accuracy_score(df_test['labels'].astype('float').astype('int'), df_test['predictions'].astype('float').astype('int'))}

        print('accuracy_correct', acc)
        print('metrics (mean of all preds)', metrics)
        metrics.update(acc)
        with open(os.path.join(config['model']['pretrain_model'],
                               'test_log.txt'), 'w') as file:
            file.write(json.dumps({**metrics, **config},
                                  sort_keys=True, indent=4,
                                  separators=(',', ': ')))
        
    def get_test_segmentation_pipeline(self, model, criterion,
                                       config, test_loader,
                                       device, init_metrics, increment_metrics,
                                       normalize_metrics,
                                       running_metric_test, running_loss_test):
        from models.image2image_utils.utils_3D.test_utils_img2img_monai \
                import slidingWindowTest
        testModule = slidingWindowTest(model, criterion, config,
                                       test_loader, device,
                                       running_metric_val=running_metric_test,
                                       running_loss_val=running_loss_test,
                                       saliency_maps=False)
        testModule()

    def reorder_columns(self, df):
        cols = df.columns[0:5].to_list()
       # cols_end = df.columns[5:].to_list()
        # if 'label' in cols:
        #     cols.remove('label')
        # if 'labels' in cols_end:
        #     cols_end.remove('labels')
        start_cols = cols + ['labels_ori', 'labels', 'predictions', 'confidences']
        not_matches = self.returnNotMatches(start_cols, df.columns.to_list())
        new_cols = start_cols + not_matches[1]
        df = df[new_cols]
        return df

    def read_validationCSV(self, config):
        df_test = pd.read_csv(config['ValdataCSV'])
        if 'predictions' in df_test.columns:
            df_test = df_test.drop(columns=['predictions'])
        if 'confidences' in df_test.columns:
            df_test = df_test.drop(columns=['confidences'])
        df_test = df_test[df_test['labels'].notna()]
        df_test = pd.concat([df_test]*config['loaders']['val_method']['samples'], ignore_index=True)
        return df_test

    def buildCsvResults(self, df_test, confidences):
        df_pred = pd.DataFrame(
            {'confidences': confidences.numpy().tolist()}, columns=['confidences'])

        df_test = pd.concat([df_test, df_pred], axis=1)
        df_test['confidences'] = df_test['confidences'].apply(pd.to_numeric)
        df_test_conf = pd.DataFrame()
        df_test_conf['confidences'] = df_test.groupby('RecursiveFilePath')['confidences'].apply(np.mean)
        df_test_conf['predictions'] = df_test_conf['confidences'].apply(np.argmax)
        df_test = df_test.merge(
            df_test_conf,
            left_on='RecursiveFilePath',
            right_on='RecursiveFilePath',
            how='inner').drop_duplicates('RecursiveFilePath')
        df_test['confidences'] = df_test['confidences_y']
        df_test = df_test.drop(columns=['confidences_y', 'confidences_x'])
        df_test = self.reorder_columns(df_test)
        return df_test


    def save_pre_val_csv(self, list_pre_val_csv, df_test):
        list_pre_val = self.load_csv_files(list_pre_val_csv)
        idx = 0
        df_test = df_test[['predictions', 'confidences', 'bth_pid', 'TimeStamp']]
        for df in list_pre_val:
            if 'predictions' in df.columns:
                df = df.drop(columns=['predictions'])
            if 'confidences' in df.columns:
                df = df.drop(columns=['confidences'])
            col_order = list(df.columns)
            idx_col = list(df.columns).index('labels_ori')
            col_order[idx_col+1:1] = ['predictions', 'confidences']
            df = df_test.merge(
                df, left_on=['bth_pid', 'TimeStamp'],
                right_on=['bth_pid', 'TimeStamp'],
                how='inner')
            df = df[col_order]
            df.to_csv(list_pre_val_csv[idx], index=False)
            idx += 1

        return None


    def load_csv_files(self, list_pre_val_list):
        li = []
        for filename in list_pre_val_list:
            df = pd.read_csv(
                filename, index_col=None,
                header=0, dtype=str)
            li.append(df)
        return li

    def returnNotMatches(self, a, b):
        return [[x for x in a if x not in b], [x for x in b if x not in a]]
