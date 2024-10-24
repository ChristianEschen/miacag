import uuid
import os
import socket
from datetime import datetime, timedelta
import torch.distributed
import yaml
from miacag.preprocessing.split_train_val import splitter
from miacag.plots.plotter import getNormConfMat
from miacag.utils.sql_utils import copy_table, add_columns, \
    copyCol, changeDtypes
import copy
import numpy as np
import pandas as pd
from miacag.postprocessing.append_results import appendDataFrame
import torch
from miacag.trainer import train
from miacag.tester import test
from miacag.configs.config import load_config, maybe_create_tensorboard_logdir
from miacag.configs.options import TrainOptions
import argparse
from miacag.preprocessing.labels_map import labelsMap
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from miacag.preprocessing.utils.check_experiments import checkExpExists, \
    checkCsvExists
from miacag.plots.plotter import plot_results, plotRegression
import pandas as pd
from miacag.preprocessing.transform_thresholds import transformThresholdRegression
from miacag.preprocessing.transform_missing_floats import transformMissingFloats
from miacag.utils.script_utils import create_empty_csv, mkFolder, maybe_remove, write_file, test_for_file
from miacag.postprocessing.aggregate_pr_group import Aggregator
from miacag.postprocessing.count_stenosis_pr_group \
    import CountSignificantStenoses
from miacag.utils.sql_utils import getDataFromDatabase
from miacag.plots.plot_predict_coronary_pathology import run_plotter_ruc_multi_class
from miacag.model_utils.predict_utils import compute_baseline_hazards#, predict_surv_df
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use('Agg')
import shutil
import timeit
import torch.distributed as dist
import tracemalloc
import psutil
import threading
import linecache
import time
import sys
import traceback
from miacag.scripts.script_utils import ConfigManipulator
from sklearn.metrics import f1_score, matthews_corrcoef, \
     accuracy_score, confusion_matrix#, plot_confusion_matrix
import json
from miacag.plots.plotter import compute_aggregation
from miacag.trainer import get_device
from miacag.models.BuildModel import ModelBuilder

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = "1"

parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--cpu', type=str,
    help="if cpu 'True' else 'False'")
parser.add_argument(
    "--local_rank", type=int,
    help="Local rank: torch.distributed.launch.")
parser.add_argument(
            "--local-rank", type=int,
            help="Local rank: torch.distributed.launch.")
parser.add_argument(
    "--num_workers", type=int,
    help="Number of cpu workers for training")
parser.add_argument(
    '--config_path', type=str,
    help="path to config file for downstream tasks")
parser.add_argument(
    '--table_name_input', type=str, default=None,
    help="path to config file for downstream tasks")
parser.add_argument(
    '--config_path_pretraining', type=str,
    help="path to config file for pretraining")
parser.add_argument("--debugging", action="store_true", default=False, help="do debugging")
parser.add_argument("--output_table_name", type=str, help="table name output table")


def get_exp_name(config, rank, config_path):
    tensorboard_comment = os.path.basename(config_path)[:-5]
    temp_file = os.path.join(config['output'], 'temp.txt')
    torch.distributed.barrier()
    if rank == 0:
        maybe_remove(temp_file)
        experiment_name = tensorboard_comment + '_' + \
            "SEP_" + \
            datetime.now().strftime('%b%d_%H-%M-%S')
        write_file(temp_file, experiment_name)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
    experiment_name = test_for_file(temp_file)[0]
    return experiment_name

def pretraining_downstreams(cpu, num_workers, config_path, table_name_input, debugging):
    print('loading config:', config_path)
    

    # with open(config_path_pretraining) as file:
    #     config_pretraining = yaml.load(file, Loader=yaml.FullLoader)
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        
    # for key, value in config_pretraining.items():
    #     if key not in config:
    #         config[key] = value
    mkFolder(config['output'])
    config['master_port'] = os.environ['MASTER_PORT']
    config['num_workers'] = num_workers
    config['cpu'] = cpu
    config['cpu'] = str(config['cpu'])

    config['debugging'] = debugging
    rank = int(os.environ['RANK'])

    experiment_name = get_exp_name(config, rank, config_path)
    if table_name_input is None:
        output_table_name = \
            experiment_name + "_" + config['table_name']
    else:
        output_table_name = table_name_input
    output_directory = os.path.join(
                        config['output'],
                        experiment_name)
    output_config = os.path.join(output_directory,
                                 os.path.basename(config_path))
    # begin pipeline
    # 1. copy table
    if rank == 0:
        os.system("mkdir -p {output_dir}".format(
            output_dir=output_directory))
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
  #  torch.distributed.barrier()

    if rank == 0:
        if table_name_input is not None:
            print('not copying table as we have input table name')
        else:
            copy_table(sql_config={
                'database': config['database'],
                'username': config['username'],
                'password': config['password'],
                'host': config['host'],
                'schema_name': config['schema_name'],
                'table_name_input': config['table_name'],
                'table_name_output': output_table_name})

        # # 2. copy config
        os.system(
            "cp {config_path} {config_file_temp}".format(
                config_path=config_path,
                config_file_temp=output_config))
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()



    # if config['model']['ssl_pretraining']:
    #     print('ssl pretraining')
    #     from ijepa.main_distributed import launch_in_pipeline
    #     use_cpu = False if cpu == "False" else True
    #     config['logging']['folder'] = output_directory
    #     config_pretrain = copy.deepcopy(config)
    #     torch.distributed.barrier()
    #     launch_in_pipeline(config_pretrain,
    #                        num_workers=num_workers, cpu=use_cpu,
    #                        init_ddp=False)
    #     torch.distributed.barrier()

    # else:        # copy model
    #     if config['model']['pretrain_model'] != 'None':
    #         print('need to adapt config')
    #         if rank == 0:
    #             shutil.copyfile(
    #                 os.path.join(config['model']['pretrain_model'], 'model.pt'),
    #                 os.path.join(output_directory, 'model.pt'))
    #             torch.distributed.barrier()
    #         config['model']['pretrain_model']  = output_directory
    #         config['model']['pretrained'] = "True"
    #     else:
    #         pass

    # loop through all indicator tasks
    unique_index = list(dict.fromkeys(config['task_indicator']))
    for task_index in unique_index:
        print('running task idx', task_index)
        config_new = copy.deepcopy(config)
        if task_index == 2:
            config_new['num_workers'] = 4
        elif task_index == 3:
            config_new['num_workers'] = 4
        elif task_index == 5:
            config_new['num_workers'] = 4
        else:
            config_new['num_workers'] = num_workers
      #  print('before run task barrier')
        torch.distributed.barrier()
        run_task(config_new, task_index, output_directory, output_table_name,
                cpu, train_test_indicator=True)
        
    for task_index in unique_index:
        config_new = copy.deepcopy(config)
        torch.distributed.barrier()
        run_task(config_new, task_index, output_directory, output_table_name,
                cpu, train_test_indicator=False)
    return None

#################################### this is new ##############################################
def plot_pretraining_downstreams(cpu, num_workers, config_path, debugging, output_table_name):
    print('loading config:', config_path)
    
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        
    
    config['num_workers'] = num_workers
    config['cpu'] = cpu
    config['cpu'] = str(config['cpu'])

  #  config['debugging'] = debugging

    #experiment_name = get_exp_name(config, rank, config_path)
    output_directory = config['output']



    config['model']['pretrain_model']  = output_directory
    config['model']['pretrained'] = "True"

    unique_index = list(dict.fromkeys(config['task_indicator']))
    for task_index in unique_index:
        print('running task idx', task_index)
        config_new = copy.deepcopy(config)
        run_task(config_new, task_index, output_directory, output_table_name,
                cpu, train_test_indicator=False)
    print('pipeline done')
    return None

def plot_task_not_ddp(config_task, output_table_name, conf, loss_names):
    # create plot folders
    output_plots = os.path.join(config_task['output_directory'], 'plots')
    mkFolder(output_plots)

    output_plots_train = os.path.join(output_plots, 'train')
    output_plots_val = os.path.join(output_plots, 'val')
    output_plots_test = os.path.join(output_plots, 'test')
    output_plots_test_large = os.path.join(output_plots, 'test_large')

    mkFolder(output_plots_train)
    mkFolder(output_plots_test)
    mkFolder(output_plots_test_large)
    mkFolder(output_plots_val)

    # plot results
    if loss_names[0] in ['L1smooth', 'MSE', 'wfocall1']:
        plot_regression_tasks(config_task, output_table_name, output_plots_train,
                            output_plots_val, output_plots_test, output_plots_test_large, conf)
    elif loss_names[0] in ['CE']:
        plot_classification_tasks(config_task, output_table_name,
                                output_plots_train, output_plots_val, output_plots_test, output_plots_test_large)
        
    elif loss_names[0] in ['NNL']:
        df_low_risk, df_high_risk = plot_time_to_event_tasks(config_task, output_table_name,
                                output_plots_train, output_plots_val, output_plots_test, output_plots_test_large)
        
    else:
        raise ValueError("Loss not supported")
    return
####################################################################################
def train_and_test(config_task):
   # torch.distributed.barrier()
    if not config_task['is_already_trained']:
        print('init training')
        train(config_task)
        # config_task_test = copy.deepcopy(config_task)        
        config_task['model']['pretrain_model'] = config_task['output_directory']
        config_task['model']['pretrained'] = "None"
    else:
        # crawl one level up in the direcotry fir config_task['output_directory]
        if config_task['labels_names'][0].startswith('sten' or 'ffr'):
            config_task['model']['pretrain_model'] = os.path.join(config_task['base_model'], config_task['artery_type'])
        else:
            config_task['model']['pretrain_model'] = config_task['base_model']

        print('model already trained')
    
 #   config_task['loaders']['nr_patches'] = config_task['loaders']['val_method']['nr_patches'] #100
 #   config_task['loaders']['batchSize'] = config_task['loaders']['val_method']['batchSize'] #100
    torch.distributed.barrier()
    # clear gpu memory
    torch.cuda.empty_cache()
    # 5 eval model
    if not config_task['is_already_tested']:
        # test if feature_importance key is in config_task
        if 'feature_importance' not in config_task.keys():
            config_task['feature_importance'] = False
        feature_importance_copy = copy.deepcopy(config_task['feature_importance'])
        config_task['feature_importance'] = False
        print('init testing')
        test({**config_task, 'query': config_task["query_test"], 'TestSize': 1})
        torch.distributed.barrier()

        config_task['feature_importance'] = True
        if feature_importance_copy:
            if len(config_task['loaders']['tabular_data_names'])>0:
                if config_task['labels_names'][0].startswith('duration'):
                    # global
                    print('init feature importance global')
                    ori_path = copy.deepcopy(config_task['output_directory'])
                    config_task['output_directory'] = os.path.join(ori_path, 'global_fi')
                    if dist.get_rank() == 0:
                        mkFolder(config_task['output_directory'])
                    test({**config_task, 'query': config_task["query_test"], 'TestSize': 1, config_task['feature_importance']: True})
                    torch.distributed.barrier()
                    # high risk
                    print('init feature importance high risk')
                    config_task['output_directory'] = os.path.join(ori_path, 'high_risk_fi')
                    if dist.get_rank() == 0:
                        mkFolder(config_task['output_directory'])
                    test({**config_task, 'query': config_task["query_high_risk"], 'TestSize': 1, config_task['feature_importance']: True})
                    torch.distributed.barrier()
                    config_task['output_directory'] = os.path.join(ori_path, 'low_risk_fi')
                    if dist.get_rank() == 0:
                        mkFolder(config_task['output_directory'])
                    # low risk
                    print('init feature importance low risk')
                    test({**config_task, 'query': config_task["query_low_risk"], 'TestSize': 1, config_task['feature_importance']: True})
                    config_task['output_directory'] = ori_path


                
    
            torch.distributed.barrier()





  #  config_task['loaders']['nr_patches'] = config_task['loaders']['val_method']['nr_patches'] #100
  #  config_task['loaders']['batchSize'] = config_task['loaders']['val_method']['batchSize'] #100

    print('kill gpu processes')
    torch.distributed.barrier()
    # clear gpu memory
    torch.cuda.empty_cache()
    return

def plot_task(config_task, output_table_name, conf, loss_names):
    # create plot folders
    output_plots = os.path.join(config_task['output_directory'], 'plots')
    mkFolder(output_plots)

    output_plots_train = os.path.join(output_plots, 'train')
    output_plots_val = os.path.join(output_plots, 'val')
    output_plots_test = os.path.join(output_plots, 'test')
    output_plots_test_large = os.path.join(output_plots, 'test_large')

    mkFolder(output_plots_train)
    mkFolder(output_plots_test)
    mkFolder(output_plots_test_large)
    mkFolder(output_plots_val)
    torch.distributed.barrier()
    config_task['loaders']['mode'] = 'testing'
        # plot results
    if loss_names[0] in ['L1smooth', 'MSE', 'wfocall1', 'BCE_multilabel']:
        if torch.distributed.get_rank() == 0:

            plot_regression_tasks(config_task, output_table_name, output_plots_train,
                                output_plots_val, output_plots_test, output_plots_test_large, conf)
        torch.distributed.barrier()

    elif loss_names[0] in ['CE']:
        if torch.distributed.get_rank() == 0:

            plot_classification_tasks(config_task, output_table_name,
                                    output_plots_train, output_plots_val, output_plots_test, output_plots_test_large)
        torch.distributed.barrier()

            
    elif loss_names[0] in ['NNL']:
        #if torch.distributed.get_rank() == 0:

        df_low_risk, df_high_risk = plot_time_to_event_tasks(config_task, output_table_name,
                                    output_plots_train, output_plots_val, output_plots_test, output_plots_test_large)
        torch.distributed.barrier()

        
    else:
        raise ValueError("Loss not supported")
    
    # if config_task['feature_importance']:
    #     if config_task['labels_names'][0].startswith('duration'):
    #         from miacag.metrics.survival_metrics import compute_feature_importance
    #         if not config_task['is_already_trained']:
    #             config_task['model']['pretrain_model'] = config_task['output_directory']
    #         compute_feature_importance(config_task, df_low_risk, df_high_risk)
            
    #     else:
    #         raise ValueError('feature importance not supported for this task')
    return



def run_task(config, task_index, output_directory, output_table_name, cpu, train_test_indicator):
    
    task_names = [
        name for i, name in zip(config['task_indicator'],
                                config['labels_names']) if i == task_index]
    loss_names = [
        name for i, name in zip(config['task_indicator'],
                                config['loss']['name']) if i == task_index]
    eval_names_train = [
        name for i, name in zip(
            config['task_indicator'],
            config['eval_metric_train']['name']) if i == task_index]
    num_classes = [
        name for i, name in zip(
            config['task_indicator'],
            config['model']['num_classes']) if i == task_index]
    
    eval_names_val = [
        name for i, name in zip(
            config['task_indicator'],
            config['eval_metric_val']['name']) if i == task_index]
    eval_names_val = [
        name for i, name in zip(
            config['task_indicator'],
            config['eval_metric_val']['name']) if i == task_index]
    # declare updated config
    config_task = config.copy()
    config_task['labels_names'] = task_names
    config_task['loss']['name'] = loss_names
    config_task['eval_metric_train']['name'] = eval_names_train
    config_task['eval_metric_val']['name'] = eval_names_val
    config['model']['num_classes'] = num_classes
    if not config['is_already_trained']:
        config_task['output'] = output_directory
        config_task['output_directory'] = os.path.join(output_directory, task_names[0])
    else:
        config_task['base_model'] = config['output']
        config_task['output'] = output_directory
        config_task['output_directory'] = output_directory
    mkFolder(config_task['output_directory'])
    config_task['table_name'] = output_table_name
    config_task['use_DDP'] = 'True'
    config_task['datasetFingerprintFile'] = None
        # rename labels and add columns;
    trans_label = [i + '_transformed' for i in config_task['labels_names']]
    labels_names_original = config_task['labels_names']
    config_task['labels_names'] = trans_label
    # add placeholder for confidences
    conf = [i + '_confidences' for i in config_task['labels_names']]
    # add placeholder for predictions
    pred = [i + '_predictions' for i in config_task['labels_names']]
    
    
#    # torch.distributed.barrier()
#     if loss_names[0] in ['CE']:
#         config_task['weighted_sampler'] = "True"
#     elif loss_names[0] in ['NNL']:
#         config_task['weighted_sampler'] = "True"
#     else:
#         raise ValueError('this loss is not implementeed:', loss_names)
#         #config_task['weighted_sampler'] = "False"
        
    # train(config_task)
    # TODO manipulate config_task
    manipulator = ConfigManipulator(config_task, task_index)
    config_task_list = manipulator()
    # # 5 eval model
    if train_test_indicator:
        count = 0
        print('train and test')
        if len(config_task_list)>1:
            for config_task_i in config_task_list:
                print('rca or lca', count)
                train_and_test(config_task_i)
                conf_i = [i + '_confidences' for i in config_task_i['labels_names']]
                loss_names_i = config_task_i['loss']['name']
                if config_task_i['artery_type'] == 'rca':
                    config_task_i['train_plot'] = config_task_i['train_plot_rca']
                    config_task_i['val_plot'] = config_task_i['val_plot_rca']
                    config_task_i['test_plot'] = config_task_i['test_plot_rca']
                elif config_task_i['artery_type'] == 'lca':
                    config_task_i['train_plot'] = config_task_i['train_plot_lca']
                    config_task_i['val_plot'] = config_task_i['val_plot_lca']
                    config_task_i['test_plot'] = config_task_i['test_plot_lca']
                else:
                    raise ValueError('artery type not supported')
                    
            # for config_task_i in config_task_list:
            #     print('plotting now')
            #     print('rca or lca', count)
            #     conf_i = [i + '_confidences' for i in config_task_i['labels_names']]
            #     loss_names_i = config_task_i['loss']['name']
            #     if config_task_i['artery_type'] == 'rca':
            #         config_task_i['train_plot'] = config_task_i['train_plot_rca']
            #         config_task_i['val_plot'] = config_task_i['val_plot_rca']
            #         config_task_i['test_plot'] = config_task_i['test_plot_rca']
            #     elif config_task_i['artery_type'] == 'lca':
            #         config_task_i['train_plot'] = config_task_i['train_plot_lca']
            #         config_task_i['val_plot'] = config_task_i['val_plot_lca']
            #         config_task_i['test_plot'] = config_task_i['test_plot_lca']
            #     else:
            #         raise ValueError('artery type not supported')

                # if dist.is_initialized():
                #     plot_task(config_task_i, output_table_name, conf_i, loss_names_i)
                # else:
                #     plot_task_not_ddp(config_task_i, output_table_name, conf, loss_names_i)
        else:
            config_task_list[0]['artery_type'] = 'both'
            config_task['artery_type'] = 'both'
            train_and_test(config_task_list[0])
            conf_i = [i + '_confidences' for i in config_task_list[0]['labels_names']]
            loss_names_i = config_task_list[0]['loss']['name']
            torch.cuda.empty_cache() # here empty

            # if dist.is_initialized():
            #     plot_task(config_task_list[0], output_table_name, conf_i, loss_names_i)
            # else:
            #     plot_task_not_ddp(config_task_list[0], output_table_name, conf, loss_names_i)
        
            count+=1


    else:
        config_task['artery_type'] = 'both'
        torch.cuda.empty_cache() # here empty

        if dist.is_initialized():
            plot_task(config_task, output_table_name, conf, loss_names)
        else:
            plot_task_not_ddp(config_task, output_table_name, conf, loss_names)
    return None

def change_psql_col_to_dates(config, output_table_name, col):
    sql = """
    UPDATE {s}.{t}
    SET {col} = my_timestamp_column::date;
    """.format(s=config['schema_name'], t=output_table_name, col=col)
    return None


def change_dtype_add_cols_ints(config, output_table_name, trans_label, labels_names_original, conf, pred, type):
    add_columns({
        'database': config['database'],
        'username': config['username'],
        'password': config['password'],
        'host': config['host'],
        'schema_name': config['schema_name'],
        'table_name': output_table_name,
        'table_name_output': output_table_name},
                trans_label,
                ['VARCHAR'] * len(trans_label))
    # copy content of labels
    copyCol(
        {'database': config["database"],
            'username': config["username"],
            'password': config['password'],
            'host': config['host'],
            'schema_name': config['schema_name'],
            'table_name': output_table_name,
            'query': config['query_transform']},
        labels_names_original,
        trans_label)
    if config['labels_names'][0].startswith('timi'):
        dict_map = config['timi_flow_dict']
    elif config['labels_names'][0].startswith('treatment'):
        dict_map = config['treatment_dict']

    else:
        dict_map = config['labels_dict']
    for lab_name in labels_names_original:
        lab_name = [lab_name]
        mapper_obj = labelsMap(
                    {
                        'labels_names': lab_name,
                        'database': config['database'],
                        'username': config['username'],
                        'password': config['password'],
                        'host': config['host'],
                        'schema_name': config['schema_name'],
                        'table_name': output_table_name,
                        'query': config['query_test'],
                        'TestSize': 1},
                    dict_map)
        mapper_obj()
    changeDtypes(
        {'database': config["database"],
            'username': config["username"],
            'password': config['password'],
            'host': config['host'],
            'schema_name': config['schema_name'],
            'table_name': output_table_name,
            'query': config['query_transform']},
        trans_label,
        [type] * len(trans_label))

    add_columns({
        'database': config['database'],
        'username': config['username'],
        'password': config['password'],
        'host': config['host'],
        'schema_name': config['schema_name'],
        'table_name': output_table_name,
        'table_name_output': output_table_name},
                conf,
                ["VARCHAR"] * len(conf))
    add_columns({
        'database': config['database'],
        'username': config['username'],
        'password': config['password'],
        'host': config['host'],
        'schema_name': config['schema_name'],
        'table_name': output_table_name,
        'table_name_output': output_table_name},
                pred,
                [type] * len(pred))


def transform_regression_data(config, output_table_name, trans_label):
    trans = transformMissingFloats({
        'labels_names': config['labels_names'],
        'database': config['database'],
        'username': config['username'],
        'password': config['password'],
        'host': config['host'],
        'schema_name': config['schema_name'],
        'table_name': output_table_name,
        'query': config['query_transform'],
        'TestSize': config['TestSize']})
    trans()

    trans_thres = transformThresholdRegression({
        'labels_names': config['labels_names'],
        'database': config['database'],
        'username': config['username'],
        'password': config['password'],
        'host': config['host'],
        'schema_name': config['schema_name'],
        'table_name': output_table_name,
        'query': config['query_transform'],
        'TestSize': config['TestSize']},
        config)
    trans_thres()

    # change dtypes for label

def plot_time_to_event_tasks(config_task, output_table_name, output_plots_train,
                              output_plots_val, output_plots_test, output_plots_test_large):
    
    df, conn = getDataFromDatabase({
                'database': config_task['database'],
                'username': config_task['username'],
                'password': config_task['password'],
                'host': config_task['host'],
                'labels_names': config_task['labels_names'],
                'schema_name': config_task['schema_name'],
                'table_name': output_table_name,
                'query': config_task['query_transform']})
    conn.close()
    
    df_target = df.dropna(subset=[config_task['labels_names'][0]+'_predictions'], how='any')
    df_target['event'] = df_target.apply(lambda row: 0 if row['duration_transformed'] > config_task['loss']['censur_date'] else row['event'], axis=1)
#    base_haz, bch = compute_baseline_hazards(df_target, max_duration=None, config=config_task)
    if config_task['debugging']:
        phases = [output_plots_train]
        phases_q = ['train']

    else:
        phases = [output_plots_test] # + output_plots_test_large]
        phases_q = ['test']#, 'test_large']
    for idx in range(0, len(phases)):
        phase_plot = phases[idx]
        phase_q = phases_q[idx]
        df, conn = getDataFromDatabase({
                'database': config_task['database'],
                'username': config_task['username'],
                'password': config_task['password'],
                'host': config_task['host'],
                'labels_names': config_task['labels_names'],
                'schema_name': config_task['schema_name'],
                'table_name': output_table_name,
                'query': config_task[phase_q +'_plot']})
        
        df_target = df.dropna(subset=[config_task['labels_names'][0]+'_predictions'], how='any')
        from miacag.plots.plotter import convertConfFloats
        preds = convertConfFloats(df_target[config_task['labels_names'][0]+'_confidences'], config_task['loss']['name'][0], config_task)
        from miacag.metrics.survival_metrics import convert_cuts_np


        # interpolate preds
        if not config_task['is_already_trained']:
            text_file_name = os.path.join(config_task['output_directory'], config_task['table_name']+'_log.txt')
            cuts = convert_cuts_np(json.load(open(text_file_name))["cuts"])
        else:
            with open(os.path.join(config_task['base_model'], 'config.yaml'), 'r') as stream:
                data_loaded = yaml.safe_load(stream)
            cuts = data_loaded['cuts']
        from miacag.metrics.survival_metrics import surv_plot, get_high_low_risk_from_df, plot_low_risk_high_risk, get_high_low_risk_from_df_plots
        #if config_task['feature_importance']:
      ##  low_risk_df, high_risk_df = get_high_low_risk_from_df(cuts, df_target, preds)
        get_high_low_risk_from_df_plots(cuts, df_target, preds, phase_plot, config_task)
           # return low_risk_df, high_risk_df

        surv_plot(config_task, cuts, df_target, preds, phase_plot, agg=False)
        

        
        phase_plot_agg = os.path.join(phase_plot, 'group_agg')
        if torch.distributed.get_rank() == 0:
            mkFolder(phase_plot_agg)
        for i in range(0, preds.shape[1]):
            df_target['preds_'+str(i)]= preds[:,i]
        if torch.distributed.get_rank() == 0:
            aggregated_cols_list = ["preds_"+str(i) for i in range(0, preds.shape[1])]
            df_agg = compute_aggregation(df_target, aggregated_cols_list, agg_type="max")
            df_agg =df_agg.sort_values('TimeStamp').drop_duplicates(['PatientID','StudyInstanceUID'], keep='first')
            get_high_low_risk_from_df_plots(cuts, df_agg, df_agg[aggregated_cols_list].to_numpy(), phase_plot_agg, config_task)
            surv_plot(config_task, cuts, df_agg, np.array(df_agg[aggregated_cols_list]), phase_plot_agg, agg=True)

        # if config_task['feature_importance']:
        #     return low_risk_df, high_risk_df
        # else:
        return None, None

def plot_regression_tasks(config_task, output_table_name, output_plots_train,
                          output_plots_val, output_plots_test, output_plots_test_large, conf):
    
    # 6 plot results:
    if config_task['debugging']:
        queries = [config_task['train_plot']]
        plots = [output_plots_train]
    else:
        queries = [config_task['test_plot']]
                # config_task['val_plot'],
                # config_task['test_plot']
              #              ]
        plots = [output_plots_train, output_plots_val,
                 output_plots_test,]
                 #output_plots_test_large]
        plots = [output_plots_test]
               # config_task['query_test_large_plot']]
    for idx, query in enumerate(queries):
        if conf[0].startswith(('sten', 'ffr')):
            plot_results({
                        'database': config_task['database'],
                        'username': config_task['username'],
                        'password': config_task['password'],
                        'host': config_task['host'],
                        'labels_names': config_task['labels_names'],
                        'schema_name': config_task['schema_name'],
                        'table_name': output_table_name,
                        'query': query},
                        config_task['labels_names'],
                        [i + "_predictions" for i in
                            config_task['labels_names']],
                        plots[idx],
                        config_task['model']['num_classes'],
                        config_task,
                        [i + "_confidences" for i in
                            config_task['labels_names']]
                        )

            # plotRegression({
            #             'database': config_task['database'],
            #             'username': config_task['username'],
            #             'password': config_task['password'],
            #             'host': config_task['host'],
            #             'labels_names': config_task['labels_names'],
            #             'schema_name': config_task['schema_name'],
            #             'table_name': output_table_name,
            #             'query': query,
            #             'loss_name': config_task['loss']['name'],
            #             'task_type': config_task['task_type']
            #             },
            #             config_task['labels_names'],
            #             conf,
            #             plots[idx],
            #             config_task,
            #             group_aggregated=False)
        
        # also group aggregated
            plot_i = plots[idx] + '_group_aggregated'
            mkFolder(plot_i)
            plot_results({
                        'database': config_task['database'],
                        'username': config_task['username'],
                        'password': config_task['password'],
                        'host': config_task['host'],
                        'labels_names': config_task['labels_names'],
                        'schema_name': config_task['schema_name'],
                        'table_name': output_table_name,
                        'query': query},
                        config_task['labels_names'],
                        [i + "_predictions" for i in
                            config_task['labels_names']],
                        plot_i,
                        config_task['model']['num_classes'],
                        config_task,
                        [i + "_confidences" for i in
                            config_task['labels_names']],
                        group_aggregated=True
                        )

            # plotRegression({
            #             'database': config_task['database'],
            #             'username': config_task['username'],
            #             'password': config_task['password'],
            #             'host': config_task['host'],
            #             'labels_names': config_task['labels_names'],
            #             'schema_name': config_task['schema_name'],
            #             'table_name': output_table_name,
            #             'query': query,
            #             'loss_name': config_task['loss']['name'],
            #             'task_type': config_task['task_type']
            #             },
            #             config_task['labels_names'],
            #             conf,
            #             plot_i,
            #             config_task,
            #             group_aggregated=True)
        


def plot_classification_tasks(config,
                             output_table_name,
                             output_plots_train, output_plots_val, output_plots_test, output_plots_test_large):
    if config['debugging']:
        phases = [output_plots_train]
        phases_q = ['train']
    else:
        phases = [output_plots_train] + [output_plots_val] + [output_plots_test] # + output_plots_test_large]
        phases_q = ['train', 'val', 'test'] #, 'test_large']
     #   phase_q = ['train']
    for idx in range(0, len(phases)):
        phase_plot = phases[idx]
        phase_q = phases_q[idx]
        df, conn = getDataFromDatabase({
                                'database': config['database'],
                                'username': config['username'],
                                'password': config['password'],
                                'host': config['host'],
                                'labels_names': config['labels_names'],
                                'schema_name': config['schema_name'],
                                'table_name': output_table_name,
                                'query': config[phase_q  +'_plot']})
        # test if _confidences exists
        if config['labels_names'][0] +"_confidences" in df.columns:
            labels_names = config['labels_names'][0] +"_confidences"
        elif config['labels_names'][0] +"_confid" in df.columns:
            labels_names = config['labels_names'][0] +"_confid"
        else:
            raise ValueError("No confidence column found in database")
            
        df = df.dropna(subset=[labels_names], how="any")
        if config['labels_names'][0].startswith('koronarpatologi'):
            col = 'koronarpatologi_transformed_confidences'
            target_name = "Corornay pathology"
            save_name = "roc_curve_coronar"
        else:
            col = "treatment_transformed_confidences"
            target_name = "Treatment"
            save_name = "roc_curve_treatment"
        y_pred = df[config['labels_names'][0] + '_predictions']
        support = len(y_pred)
        
        f1_transformed = f1_score(
            df[config['labels_names'][0]],
            df[config['labels_names'][0] + '_predictions'],
            average='macro')
        mcc = matthews_corrcoef(df[config['labels_names'][0]],
            df[config['labels_names'][0] + '_predictions'])
        #df[target_name] = df[[config['labels_names'][0]]
        getNormConfMat(
            df,
            config['labels_names'][0],
            config['labels_names'][0] + '_predictions',
            target_name,
            f1_transformed,
            phase_plot,
            config['model']['num_classes'],
            support,
            0,
            mcc=mcc)
    

    return None


def convert_string_to_numpy(df, column='koronarpatologi_nativekar_udfyldesforallept__transformed_confid'):
    list_values = []

    for row in df[column]:
        # Transforming the string to a dictionary
        row_dict = {int(k):float(v) for k,v in  (item.split(':') for item in row.strip('{}').split(';'))}
        # Adding the values to a list
        list_values.append(list(row_dict.values()))

    # Transforming the list of lists to a numpy array
    np_array = np.array(list_values)
    
    return np_array
if __name__ == '__main__':
    import torch
    import os   
    start_time = timeit.default_timer()


    args = parser.parse_args()
    torch.distributed.init_process_group(backend='nccl' if args.cpu == 'False' else "Gloo",
                                         init_method='env://',
                                         world_size=int(os.environ['WORLD_SIZE']),
                                         timeout=timedelta(seconds=18000000),)
    local_rank = int(os.environ['LOCAL_RANK'])
    if args.cpu == 'False':
        torch.cuda.set_device(local_rank)



    pretraining_downstreams(args.cpu, args.num_workers, args.config_path,
                        args.table_name_input, args.debugging)

    elapsed = timeit.default_timer() - start_time
    print('cpu', args.cpu)
    print(f"Execution time: {elapsed} seconds")
