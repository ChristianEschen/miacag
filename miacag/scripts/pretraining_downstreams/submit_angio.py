import uuid
import os
import socket
from datetime import datetime, timedelta
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
from miacag.utils.survival_utils import create_cols_survival
from miacag.model_utils.predict_utils import compute_baseline_hazards#, predict_surv_df
from miacag.metrics.survival_metrics import confidences_upper_lower_survival, confidences_upper_lower_survival_discrete
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
from miacag.metrics.survival_metrics import EvalSurv
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
        plot_time_to_event_tasks(config_task, output_table_name,
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
        config_task['model']['pretrain_model'] = os.path.join(config_task['base_model'], config_task['artery_type'])
        print('model already trained')
    
 #   config_task['loaders']['nr_patches'] = config_task['loaders']['val_method']['nr_patches'] #100
 #   config_task['loaders']['batchSize'] = config_task['loaders']['val_method']['batchSize'] #100
    torch.distributed.barrier()
    # clear gpu memory
    torch.cuda.empty_cache()
    # 5 eval model
    if not config_task['is_already_tested']:
        test({**config_task, 'query': config_task["query_test"], 'TestSize': 1})
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
    if torch.distributed.get_rank() == 0:
        # plot results
        if loss_names[0] in ['L1smooth', 'MSE', 'wfocall1']:
            plot_regression_tasks(config_task, output_table_name, output_plots_train,
                                output_plots_val, output_plots_test, output_plots_test_large, conf)
        elif loss_names[0] in ['CE']:
            plot_classification_tasks(config_task, output_table_name,
                                    output_plots_train, output_plots_val, output_plots_test, output_plots_test_large)
            
        elif loss_names[0] in ['NNL']:
            plot_time_to_event_tasks(config_task, output_table_name,
                                    output_plots_train, output_plots_val, output_plots_test, output_plots_test_large)
            
        else:
            raise ValueError("Loss not supported")
    torch.distributed.barrier()
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
    
    
   # torch.distributed.barrier()
    if loss_names[0] in ['CE']:
        config_task['weighted_sampler'] = "True"
    elif loss_names[0] in ['NNL']:
        config_task['weighted_sampler'] = "True"
    else:
        config_task['weighted_sampler'] = "False"
        
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
                    
            #   if not config_task_i['debugging']:
            for config_task_i in config_task_list:
                print('plotting now')
                print('rca or lca', count)
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

        else:
            config_task_list[0]['artery_type'] = 'both'
            train_and_test(config_task_list[0])

            if dist.is_initialized():
                plot_task(config_task_i, output_table_name, conf_i, loss_names_i)
            else:
                plot_task_not_ddp(config_task_i, output_table_name, conf, loss_names_i)
        
            count+=1


    else:
        config_task_list[0]['artery_type'] = 'both'
        if len(config_task_list)>1:
            config_task_list[0]["labels_names"] = config_task_list[0]["labels_names"] + config_task_list[1]["labels_names"]
            config_task_list[0]['loss']['name'] = config_task_list[0]['loss']['name'] + config_task_list[1]['loss']['name']
            config_task_list[0]['model']['num_classes'] = config_task_list[0]['model']['num_classes'] + config_task_list[1]['model']['num_classes']
        
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
    from miacag.plots.plotter import add_misssing_rows
    for label_name in config_task['labels_names']:
        df = add_misssing_rows(df, label_name)
    df_target = df.dropna(subset=[config_task['labels_names'][0]+'_predictions'], how='any')
    base_haz, bch = compute_baseline_hazards(df_target, max_duration=None, config=config_task)
    if config_task['debugging']:
        phases = [output_plots_train]
        phases_q = ['train']

    else:
        phases = [output_plots_train] + [output_plots_val] + [output_plots_test] # + output_plots_test_large]
        phases_q = ['train', 'val', 'test']#, 'test_large']
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
        # interpolate preds
        text_file_name = os.path.join(config_task['output_directory'], config_task['table_name']+'_log.txt')
        cuts = convert_cuts_np(json.load(open(text_file_name))["cuts"])
        
      #  surv =  pd.DataFrame(preds.transpose(), cuts)

        surv = predict_surv(preds, cuts)
        
        ev = EvalSurv(surv,
                      np.array(df_target[config_task['labels_names'][0]]),
                      np.array(df_target['event']),
                      censor_surv='km')


        plot_x_individuals(surv, phase_plot,x_individuals=5)
      #  out_dict = confidences_upper_lower_survival(df_target, base_haz, bch, config_task)
        out_dict = confidences_upper_lower_survival_discrete(surv,
                                                             np.array(df_target[config_task['labels_names'][0]]),
                                                             np.array(df_target['event']),
                                                             config_task)

        plot_scores(out_dict, phase_plot)
        # if config_task['debugging']:
        #     thresholds = [6000, 7000]
        # else:
        #     thresholds = [365, 365*5]
        # auc_1_year_dict = get_roc_auc_ytest_1_year_surv(df_target, base_haz, bch, config_task, threshold=thresholds[0])
        # auc_5_year_dict = get_roc_auc_ytest_1_year_surv(df_target, base_haz, bch, config_task, threshold=thresholds[1])
        # from miacag.metrics.survival_metrics import plot_auc_surv
        # plot_auc_surv(auc_1_year_dict, auc_5_year_dict, phase_plot)
        
        print('done')

def predict_surv(logits, duration_index):
    logits = torch.tensor(logits)
    hazard = torch.nn.Sigmoid()(logits)
    surv = (1 - hazard).add(1e-7).log().cumsum(1).exp()
    surv_np = surv.numpy()
    surv = pd.DataFrame(surv_np.transpose(), duration_index)
    new_index = np.linspace(surv.index.min(), surv.index.max(), len(duration_index)*10)

    # Interpolate DataFrame to new index
    surv = surv.reindex(surv.index.union(new_index)).interpolate('index').loc[new_index]

    return surv


def convert_cuts_np(cuts):
    string_list = cuts.strip('[]').split()

    # Konverterer listen af strings til floats og derefter til et numpy array
    numpy_array = np.array([float(i) for i in string_list])
    return numpy_array


def plot_x_individuals(surv, phase_plot,x_individuals=5):
    surv.iloc[:, :x_individuals].plot(drawstyle='steps-post')
    plt.ylabel('S(t | x)')
    _ = plt.xlabel('Time')
    plt.show()
    plt.savefig(os.path.join(phase_plot, "survival_curves.png"))
    plt.close()


def get_roc_auc_ytest_1_year_surv(df_target, base_haz, bch, config_task, threshold=365):
    from miacag.model_utils.predict_utils import predict_surv_df
    survival_estimates = predict_surv_df(df_target, base_haz, bch, config_task)
    survival_estimates = survival_estimates.reset_index()
    # merge two pandas dataframes
    survival_estimates = pd.merge(survival_estimates, df_target, on=config_task['labels_names'][0], how='inner')
    surv_preds_observed = pd.DataFrame({i: survival_estimates.loc[i, i] for i in survival_estimates.index}, index=[0])
    survival_ests = survival_estimates.set_index(config_task['labels_names'][0])
    

    # Get the first index less than the threshold
    selected_index = survival_ests.index[survival_ests.index < threshold][-1]
    # probability at threshold 6000
    yprobs = survival_ests.loc[selected_index][0:len(surv_preds_observed.columns)]
    ytest = (survival_estimates[config_task['labels_names'][0]] >threshold).astype(int)
    from miacag.plots.plot_utils import compute_bootstrapped_scores, compute_mean_lower_upper
    bootstrapped_auc = compute_bootstrapped_scores(yprobs, ytest, 'roc_auc_score')
    mean_auc, upper_auc, lower_auc = compute_mean_lower_upper(bootstrapped_auc)
    variable_dict = {
        k: v for k, v in locals().items() if k in [
            "mean_auc", "upper_auc", "lower_auc",
            "yprobs", "ytest"]}
    return variable_dict

def plot_scores(out_dict, ouput_path):

    mean_brier = out_dict["mean_brier"]
    uper_brier = out_dict["upper_brier"]
    ower_brier = out_dict["lower_brier"]
    mean_conc = out_dict["mean_conc"]
    uper_conc = out_dict["upper_conc"]
    ower_conc = out_dict["lower_conc"]
    plt.figure()
    plt.plot(out_dict['brier_scores'].index, 
                out_dict['brier_scores'].values, 
            label=f"Integregated brier score={mean_brier:.3f} ({ower_brier:.3f}-{uper_brier:.3f})\nC-index={mean_conc:.3f} ({ower_conc:.3f}-{uper_conc:.3f})")
    # add x label
    plt.xlabel('Time (days)')
    # add y label
    plt.ylabel('Brier score')
    # add legend
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(os.path.join(ouput_path, "brier_conc_scores.png"))
    plt.close()
    
    plt.figure()
    plt.plot(out_dict['brier_scores'].index, 
                out_dict['brier_scores'].values, 
            label=f"Integregated brier score={mean_brier:.3f} ({ower_brier:.3f}-{uper_brier:.3f})")
    # add x label
    plt.xlabel('Time (days)')
    # add y label
    plt.ylabel('Brier score')
    # add legend
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(os.path.join(ouput_path, "brier_scores.png"))
    plt.close()

def plot_regression_tasks(config_task, output_table_name, output_plots_train,
                          output_plots_val, output_plots_test, output_plots_test_large, conf):
    
    # 6 plot results:
    if config_task['debugging']:
        queries = [config_task['train_plot']]
        plots = [output_plots_train]
    else:
        queries = [config_task['train_plot']]
           #     config_task['val_plot'],
            #    config_task['test_plot'],
              #              ]
        plots = [output_plots_train, output_plots_val,
                 output_plots_test,]
                 #output_plots_test_large]

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

            plotRegression({
                        'database': config_task['database'],
                        'username': config_task['username'],
                        'password': config_task['password'],
                        'host': config_task['host'],
                        'labels_names': config_task['labels_names'],
                        'schema_name': config_task['schema_name'],
                        'table_name': output_table_name,
                        'query': query,
                        'loss_name': config_task['loss']['name'],
                        'task_type': config_task['task_type']
                        },
                        config_task['labels_names'],
                        conf,
                        plots[idx],
                        config_task,
                        group_aggregated=False)
        
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

            plotRegression({
                        'database': config_task['database'],
                        'username': config_task['username'],
                        'password': config_task['password'],
                        'host': config_task['host'],
                        'labels_names': config_task['labels_names'],
                        'schema_name': config_task['schema_name'],
                        'table_name': output_table_name,
                        'query': query,
                        'loss_name': config_task['loss']['name'],
                        'task_type': config_task['task_type']
                        },
                        config_task['labels_names'],
                        conf,
                        plot_i,
                        config_task,
                        group_aggregated=True)
        


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
