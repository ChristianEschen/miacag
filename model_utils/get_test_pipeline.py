from model_utils.eval_utils import val_one_epoch
from model_utils.eval_utils import eval_one_step
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from preprocessing.pre_process import appendDataframes
import os
from preprocessing.pre_process import mkFolder


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
                test_loader.val_loader, device,
                running_metric_val=running_metric_test,
                running_loss_val=running_loss_test,
                saliency_maps=False)
        else:
            raise ValueError(
                "test pipeline is not implemented %s" % repr(
                    config['loaders']['val_method']['type']))

        test_loader.val_df = self.buildPandasResults(
            test_loader.val_df,
            config['loaders']['val_method']['samples'],
            confidences)

        self.resetDataPaths(test_loader, config)
        self.insert_data_to_db(test_loader)

        acc = {'accuracy ensemble': accuracy_score(
            test_loader.val_df['labels'].astype('float').astype('int'),
            test_loader.val_df['predictions'].astype('float').astype('int'))}

        print('accuracy_correct', acc)
        print('metrics (mean of all preds)', metrics)
        metrics.update(acc)
        log_name = os.path.basename(config["DataBasePath"]) + '_log.txt'
        with open(os.path.join(config['model']['pretrain_model'],
                               log_name), 'w') as file:
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

    def buildPandasResults(self, val_df, samples, confidences):
        val_df = val_df.append(
            [val_df] * (samples - 1),
            ignore_index=True)
        df_pred = pd.DataFrame(
            {'confidences': confidences.numpy().tolist()},
            columns=['confidences'],
            index=val_df.index)

        if 'predictions' in val_df.columns:
            val_df = val_df.drop(columns=['predictions'])
        if 'confidences' in val_df.columns:
            val_df = val_df.drop(columns=['confidences'])
        val_df = pd.concat([val_df, df_pred], axis=1)
        val_df['confidences'] = \
            val_df['confidences'].apply(pd.to_numeric)
        val_df_conf = pd.DataFrame()
        val_df_conf['confidences'] = val_df.groupby(
            'DcmPathFlatten')['confidences'].apply(np.mean)
        val_df_conf['predictions'] = \
            val_df_conf['confidences'].apply(np.argmax)
        val_df = val_df.merge(
            val_df_conf,
            left_on='DcmPathFlatten',
            right_on='DcmPathFlatten',
            how='inner').drop_duplicates('DcmPathFlatten')
        val_df['confidences'] = val_df['confidences_y']
        val_df = val_df.drop(
            columns=['confidences_y', 'confidences_x'])
        return val_df

    def insert_data_to_db(self, test_loader):
        paths = test_loader.val_df['DcmPathFlatten'].to_list()
        preds = test_loader.val_df['predictions'].to_list()
        confidences = test_loader.val_df['confidences'].to_list()

        confidences = self.array_to_tuple(confidences)
        test_loader.connection.executemany(
            'UPDATE DICOM_TABLE SET predictions=? WHERE DcmPathFlatten=?',
            zip(preds, paths))
        test_loader.connection.commit()
        test_loader.connection.executemany(
            'UPDATE DICOM_TABLE SET confidences=? WHERE DcmPathFlatten=?',
            zip(confidences, paths))
        test_loader.connection.commit()

    def resetDataPaths(self, test_loader, config):
        test_loader.val_df['DcmPathFlatten'] = \
            test_loader.val_df['DcmPathFlatten'].apply(
                lambda x:  "".join(
                    x.rsplit(config['DataSetPath'])).strip(os.sep))
        return test_loader

    def array_to_tuple(self, confidences):
        conf_list_tuple = []
        for conf in confidences:
            li = [np.format_float_positional(i, precision=4)
                  for i in tuple(conf)]
            li = [str(i) + ":" + li[i] for i in range(len(li))]
            li = tuple(li)
            li = self.tuple2key(li)
            conf_list_tuple.append(li)
        return conf_list_tuple

    def tuple2key(self, t, delimiter=u';'):
        return delimiter.join(t)
