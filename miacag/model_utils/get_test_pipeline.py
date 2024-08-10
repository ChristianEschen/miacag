from miacag.model_utils.eval_utils import val_one_epoch
from miacag.model_utils.eval_utils import eval_one_step
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from miacag.preprocessing.pre_process import appendDataframes
import os
from miacag.preprocessing.pre_process import mkFolder
import psycopg2
from psycopg2.extras import execute_batch
from miacag.utils.sql_utils import update_cols
import torch
import shutil
import time
import yaml
import random
from miacag.utils.feature_imortance import wrap_feature_names
def shuffle_feature(data, feature):
    dcm_paths = [d[feature] for d in data]

    # Shuffle the DcmPathFlatten values
    random.shuffle(dcm_paths)

    # Reassign the shuffled values back to the dictionaries
    for i, d in enumerate(data):
        d[feature] = dcm_paths[i]
    
    return data
class TestPipeline():
    def get_test_pipeline(self, model, criterion, config, test_loader,
                          device, init_metrics,
                          normalize_metrics,
                          running_metric_test, running_loss_test, dataframe=None, output=None):

        if config['task_type'] in ["classification", "regression", "mil_classification"]:
            if config['feature_importance']:
                self.get_feature_importance_pipeline(model, criterion,
                                                    config, test_loader,
                                                    device, init_metrics,
                                                    normalize_metrics,
                                                    running_metric_test,
                                                    running_loss_test,
                                                    dataframe=None,
                                                    output=None)
            else:
                
                self.get_test_classification_pipeline(model, criterion,
                                                    config, test_loader,
                                                    device, init_metrics,
                                                    normalize_metrics,
                                                    running_metric_test,
                                                    running_loss_test)
        elif config['loaders']['task_type'] == "image2image":
            self.get_test_segmentation_pipeline(model, criterion,
                                                config, test_loader,
                                                device, init_metrics,
                                                normalize_metrics,
                                                running_metric_test,
                                                running_loss_test)
        else:
            raise ValueError("Task type is not implemented")
    

                                            
    def get_test_classification_pipeline(self, model, criterion,
                                         config, test_loader,
                                         device, init_metrics,
                                         normalize_metrics,
                                         running_metric_test,
                                         running_loss_test):
        start = time.time()
        print('starting inference:')
        _, df_results = val_one_epoch(
            model, criterion, config,
            test_loader.val_loader, device,
            running_metric_val=running_metric_test,
            running_loss_val=running_loss_test,
            saliency_maps=False)
        stop = time.time()
        print('time for testing:', stop-start)
        for count, label in enumerate(config['labels_names']):
            csv_files = self.saveCsvFiles(label, df_results, config, count)
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            from miacag.utils.sql_upt_utils_temp_table import update_cols_based_on_temp_table
            update_cols_based_on_temp_table(csv_files, config)
            # for count, label in enumerate(config['labels_names']):
                # test_loader.val_df = self.buildPandasResults(
                #     label,
                #     test_loader.val_df,
                #     csv_files,
                #     count,
                #     config)
            #     self.insert_data_to_db(test_loader, label, config)
            if config['loss']['name'][0] == 'NNL':
                if config['is_already_trained']:
                    pass
                else:
                    config['cuts'] = str(config['cuts'])
                
            log_name = config["table_name"] +  '_log.txt'
            try:
                del config[False]
            except:
                pass
            with open(
                os.path.join(config['output_directory'], log_name),
                    'w') as file:
                file.write(json.dumps({**config},
                        sort_keys=True, indent=4,
                        separators=(',', ': ')))
            shutil.rmtree(csv_files)
            cachDir = os.path.join(
                            config['model']['pretrain_model'],
                            'persistent_cache')
            if os.path.exists(cachDir):
                shutil.rmtree(cachDir)

    def get_feature_importance_pipeline(self, model, criterion,
                                         config, test_loader,
                                         device, init_metrics,
                                         normalize_metrics,
                                         running_metric_test,
                                         running_loss_test,
                                         dataframe = None,
                                         output = None):
        

                # combine image features and tabular features
        features = ["DcmPathFlatten"] + config['loaders']['tabular_data_names']
        #test_loader
        with torch.no_grad():
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        baseline = self.get_feature_importance(model, criterion,
                                            config, test_loader,
                                            device, init_metrics,
                                            normalize_metrics,
                                            running_metric_test,
                                            running_loss_test)
        scores = []
        with torch.no_grad():
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        for feature in features:
            test_loader.val_loader.dataset.data = shuffle_feature(test_loader.val_loader.dataset.data, feature)
            score = self.get_feature_importance(model, criterion,
                                            config, test_loader,
                                            device, init_metrics,
                                            normalize_metrics,
                                            running_metric_test,
                                            running_loss_test,
                                            )
            with torch.no_grad():
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            scores.append(baseline - score)
        # create dict with feature names and scores
        feature_scores = dict(zip(features, scores))
        # save dict to file
        # Save dict to CSV file
        df = pd.DataFrame(list(feature_scores.items()), columns=['Feature', 'Importance'])
        csv_path = os.path.join(config['output_directory'], '_feature_importance.csv')
        df.to_csv(csv_path, index=False)
        # make bar plot
        import matplotlib.pyplot as plt
        # use Agg
        plt.switch_backend('Agg')
        path = os.path.join(config['output_directory'], 'feature_importance')
        mkFolder(path)
        plt.figure(figsize=(10, 8))
        plt.barh(features, scores)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(path, '_fi.png'))
        plt.show()
        
        plt.figure(figsize=(10, 8))
        plt.barh(features, scores)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(path, '_fi.pdf'))
        plt.show()
     #   del model
    #    del test_loader
        with torch.no_grad():
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()        
        
    def get_feature_importance_pipeline_lh(self, model, criterion,
                                         config, test_loader,
                                         device, init_metrics,
                                         normalize_metrics,
                                         running_metric_test,
                                         running_loss_test,
                                         dataframe = None,
                                         output = None):
        
        if dataframe is not None:
            # dataframe["DcmPathFlatten"] = dataframe["DcmPathFlatten"].apply(
            #             lambda x: os.path.join(config['DataSetPath'], x))

            from miacag.dataloader.Classification._3D.dataloader_monai_classification_3D import val_monai_classification_loader
            dataframe = dataframe.dropna(subset=['phase'])

            ds = val_monai_classification_loader(dataframe, config)
            test_loader.val_loader.dataset.data = ds.data
        else:
            output = "global"
                # combine image features and tabular features
        features = ["DcmPathFlatten"] + config['loaders']['tabular_data_names']
        #test_loader
        torch.cuda.empty_cache()
        baseline = self.get_feature_importance(model, criterion,
                                            config, test_loader,
                                            device, init_metrics,
                                            normalize_metrics,
                                            running_metric_test,
                                            running_loss_test)
        scores = []
        for feature in features:
            test_loader.val_loader.dataset.data = shuffle_feature(test_loader.val_loader.dataset.data, feature)
            score = self.get_feature_importance(model, criterion,
                                            config, test_loader,
                                            device, init_metrics,
                                            normalize_metrics,
                                            running_metric_test,
                                            running_loss_test,
                                            )
            scores.append(baseline - score)
        # create dict with feature names and scores
        feature_scores = dict(zip(features, scores))
        # save dict to file
        # Save dict to CSV file
        df = pd.DataFrame(list(feature_scores.items()), columns=['Feature', 'Importance'])
        csv_path = os.path.join(config['output_directory'], output + '_feature_importance.csv')
        df.to_csv(csv_path, index=False)
        # make bar plot
        import matplotlib.pyplot as plt
        # use Agg
        plt.switch_backend('Agg')
        path = os.path.join(config['output_directory'], 'feature_importance')
        mkFolder(path)
        plt.figure(figsize=(10, 8))
        plt.barh(features, scores)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(path, output + '_fi.png'))
        plt.show()
        
        plt.figure(figsize=(10, 8))
        plt.barh(features, scores)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(path, output + '_fi.pdf'))
        plt.show()
     #   del model
    #    del test_loader
        torch.cuda.empty_cache()


    def get_feature_importance(self, model, criterion,
                                         config, test_loader,
                                         device, init_metrics,
                                         normalize_metrics,
                                         running_metric_test,
                                         running_loss_test):
        start = time.time()
        torch.cuda.empty_cache()
        print('straing ')
        _, df_results = val_one_epoch(
            model, criterion, config,
            test_loader.val_loader, device,
            running_metric_val=running_metric_test,
            running_loss_val=running_loss_test,
            saliency_maps=False)
        stop = time.time()
        print('time for testing:', stop-start)
        #
        from miacag.metrics.survival_metrics import compute_brier_discrete, predict_surv
        for count, label in enumerate(config['labels_names']):
            csv_files = self.saveCsvFiles(label, df_results, config, count)
        torch.distributed.barrier()
       # if torch.distributed.get_rank() == 0:
        from miacag.utils.sql_upt_utils_temp_table import update_cols_based_on_temp_table, combine_dfs
        final_df = combine_dfs(csv_files, config)
        # broadcast final_df to all ranks
     #   final_df = torch.distributed.broadcast(final_df, src=0)
        from miacag.plots.plotter import convertConfFloats
        # merge final_df and df_results on rowid
        data_df = pd.DataFrame.from_dict(test_loader.val_loader.dataset.data)

        final_df = final_df.merge(
            data_df, left_on='rowid', right_on='rowid', how='inner')
        conf = convertConfFloats(final_df['duration_transformed_confidences'], 'NNL', config)
            
        surv = predict_surv(conf, config['cuts'])
        ibs, _ = compute_brier_discrete(surv, final_df["duration_transformed"].to_numpy(), final_df['event'].to_numpy(), config)
        torch.cuda.empty_cache()

        return ibs
        
    def get_test_segmentation_pipeline(self, model, criterion,
                                       config, test_loader,
                                       device, init_metrics,
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

    def get_output_pr_class(self, outputs, class_idx):
        outs = []
        for row in outputs[0]:
            for idx, value in enumerate(row):
                if idx == class_idx:
                    outs.append(value.item())
        return outs

    def buildPandasResults(self, label_name, val_df, csv_files, count, config):
        filename = os.path.join(csv_files, label_name)
        df_pred = self.appendDataframes(filename)
        df_pred['rowid'] = df_pred['rowid'].astype(float).astype(int)
        col_names = [
            i for i in df_pred.columns.to_list() if i.startswith(
                label_name + '_confidence')]
        df_pred[label_name+'_confidences'] = df_pred[col_names].values.tolist()

        if label_name + '_predictions' in val_df.columns:
            val_df = val_df.drop(columns=[label_name + '_predictions'])
        if label_name + '_confidences' in val_df.columns:
            val_df = val_df.drop(columns=[label_name + '_confidences'])
        #val_df = pd.concat([val_df, df_pred], axis=1)
        val_df = val_df.merge(
            df_pred, left_on='rowid', right_on='rowid', how='right')
        val_df[label_name + '_confidences'] = \
            val_df[label_name + '_confidences'].apply(pd.to_numeric)
        val_df_conf = pd.DataFrame()
        val_df_conf[label_name + '_confidences'] = val_df.groupby(
            'rowid')[label_name + '_confidences'].apply(np.mean)
        if config['loss']['name'][count].startswith('CE'):
            val_df_conf[label_name + '_predictions'] = \
                val_df_conf[label_name + '_confidences'].apply(np.argmax)
        elif config['loss']['name'][count] in ['MSE', '_L1', 'L1smooth', 'NNL', 'wfocall1']:
            val_df_conf[label_name + '_predictions'] = \
                val_df_conf[label_name + '_confidences'].astype(float)
        elif config['loss']['name'][count] == 'BCE_multilabel':
            val_df_conf[label_name + '_predictions'] = \
                val_df_conf[label_name + '_confidences'].apply(
                    np.round).astype(int)
        else:
            raise ValueError(
                'not implemented loss: ', config['loss']['name'][count])
        val_df = val_df.merge(
            val_df_conf,
            left_on='rowid',
            right_on='rowid',
            how='inner').drop_duplicates('rowid')
        val_df[label_name + '_confidences'] = \
            val_df[label_name + '_confidences_y']
        val_df = val_df.drop(
            columns=[label_name + '_confidences_y',
                     label_name + '_confidences_x'])
        return val_df

    def insert_data_to_db(self, test_loader, label_name, config):
        confidences = self.array_to_tuple(
            test_loader.val_df[label_name + '_confidences'].to_list())
        test_loader.val_df[label_name + '_confidences'] = confidences
        records = test_loader.val_df.to_dict('records')
        update_cols(
                    records, config,
                    [label_name + '_predictions', label_name + '_confidences'])

    def update_db_using_temp_csv(self, test_loader, label_name, config):
        confidences = self.array_to_tuple(
            test_loader.val_df[label_name + '_confidences'].to_list())
        test_loader.val_df[label_name + '_confidences'] = confidences
        records = test_loader.val_df.to_dict('records')
        update_cols(
                    records, config,
                    [label_name + '_predictions', label_name + '_confidences'])

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
            conf_list_tuple.append([li])
        return conf_list_tuple

    def tuple2key(self, t, delimiter=u';'):
        return delimiter.join(t)

    def saveCsvFiles(self, label_name, df, config, count):
        if config['loss']['name'][count].startswith(tuple(['CE', 'NNL'])):
            confidences = label_name + '_confidence'
           # confidences = confidences[count]
            confidence_col = [
                label_name + '_confidence_' +
                str(i) for i in range(0, config['model']['num_classes'][count])]
        else:
            confidences = label_name + '_confidence'
         #   df[confidences] = np.expand_dims(df[confidences], 1)
            confidence_col = [
                label_name + '_confidence_' +
                str(i) for i in range(0, 1)]
        csv_files = os.path.join(config['output_directory'], 'csv_files_pred')
        mkFolder(csv_files)
        label_name_csv_files = os.path.join(csv_files, label_name)
        mkFolder(label_name_csv_files)
        array = np.concatenate(
            (np.expand_dims(df[confidences].to_numpy(), 1),
             np.expand_dims(df["rowid"].to_numpy(), 1)), axis=1)
        print('Warning this is not correct for CE loss, maybe ok for regression')
        cols = confidence_col + ['rowid']
        if config['loss']['name'][count].startswith(tuple(['CE', 'NNL'])):
            # convert array with rows with liusts to liste
            
            liste = []
            for i in range(0, config['model']['num_classes'][count]):
                liste.append(array[i, 0] + [array[i,1]])
            l = []
            for i in array:
                l.append(i[0])
            l = np.vstack(l)
            df_new = pd.DataFrame(l, columns=confidence_col)
            df_new["rowid"] = df["rowid"]
            df = df_new
            #array = liste
        else:
            df = pd.DataFrame(
                array,
                columns=cols)
        df.to_csv(
            os.path.join(label_name_csv_files, str(torch.distributed.get_rank()))+'.csv')
        return csv_files
      #  df.to_csv()
    
    def appendDataframes(self, csv_files_dir):
        paths = os.listdir(csv_files_dir)
        paths = [os.path.join(csv_files_dir, p) for p in paths]
        li = []
        for filename in paths:
            df = pd.read_csv(filename, index_col=None,
                             header=0, dtype=str)
            li.append(df)
        return pd.concat(li, axis=0, ignore_index=True)