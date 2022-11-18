import uuid
import os
import socket
from datetime import datetime, timedelta
import yaml
from miacag.preprocessing.split_train_val import splitter
from miacag.utils.sql_utils import copy_table, add_columns, \
    copyCol, changeDtypes
import pandas as pd
from miacag.postprocessing.append_results import appendDataFrame
import torch
from miacag.trainer import train
from miacag.tester import test
from miacag.configs.config import load_config, maybe_create_tensorboard_logdir
from miacag.configs.options import TrainOptions
import argparse
from miacag.preprocessing.labels_map import labelsMap
from miacag.preprocessing.utils.check_experiments import checkExpExists, \
    checkCsvExists
from miacag.plots.plotter import plot_results
import pandas as pd
from miacag.preprocessing.transform_thresholds import transformThreshold
from miacag.preprocessing.transform_missing_floats import transformMissingFloats
from miacag.scripts.angiography_classifier.submit_angio import angio_classifier
#from miacag.scripts.stenosis_identifier.submit_angio import stenosis_identifier
#from miacag.scripts.stenosis_regression.submit_angio import stenosis_identifier
from miacag.utils.script_utils import create_empty_csv, mkFolder, maybe_remove, write_file
from miacag.postprocessing.count_stenosis_pr_group \
    import CountSignificantStenoses

parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--cpu', type=str,
    help="if cpu 'True' else 'False'")
parser.add_argument(
            "--local_rank", type=int,
            help="Local rank: torch.distributed.launch.")    
parser.add_argument(
            "--num_workers", type=int,
            help="Number of cpu workers for training")    
parser.add_argument(
    '--config_path_stenosis_rca_r_dom', type=str,
    help="path to folder with config files")
parser.add_argument(
    '--config_path_stenosis_rca_l_dom', type=str,
    help="path to folder with config files")
parser.add_argument(
    '--config_path_stenosis_rca_co_dom', type=str,
    help="path to folder with config files")
parser.add_argument(
    '--config_path_stenosis_lca_r_dom', type=str,
    help="path to folder with config files")
parser.add_argument(
    '--config_path_stenosis_lca_l_dom', type=str,
    help="path to folder with config files")
parser.add_argument(
    '--config_path_stenosis_lca_co_dom', type=str,
    help="path to folder with config files")
parser.add_argument(
    '--model_type', type=str,
    help="regression | identification")


def combine_results(config_paths_list, table_name):
    configs = []
    for i in config_paths_list:
        with open(os.path.join(i, os.listdir(i)[0])) as file:
            configs.append(yaml.load(file, Loader=yaml.FullLoader))
    config = configs[0]
    # compute the list of label_names

    label_names = []
    for i in configs:
        label_names += i['labels_names']
    label_names = [i + "_transformed" for i in label_names]
    label_names = [x for x in label_names if "ffr" not in x]

    sten_path_train = os.path.join(
        config['output'], 'cag_stenosis_count_train')
    mkFolder(sten_path_train)
    count = CountSignificantStenoses({
                        'labels_names': label_names,
                        'database': config['database'],
                        'username': config['username'],
                        'password': config['password'],
                        'host': config['host'],
                        'schema_name': config['schema_name'],
                        'table_name': table_name,
                        'query':
                        config['query_train_plot']},
                        [i + "_confidences_aggregated" for i in
                        label_names],
                        sten_path_train,
                        config)
    count()

    sten_path_val = os.path.join(
        config['output'], 'cag_stenosis_count_val')
    mkFolder(sten_path_val)
    count = CountSignificantStenoses({
                        'labels_names': label_names,
                        'database': config['database'],
                        'username': config['username'],
                        'password': config['password'],
                        'host': config['host'],
                        'schema_name': config['schema_name'],
                        'table_name': table_name,
                        'query':
                        config['query_val_plot']},
                        [i + "_confidences_aggregated" for i in
                        label_names],
                        sten_path_val)
    count()

    sten_path_test = os.path.join(
    config['output'], 'cag_stenosis_count_test')
    mkFolder(sten_path_test)
    count = CountSignificantStenoses({
                        'labels_names': label_names,
                        'database': config['database'],
                        'username': config['username'],
                        'password': config['password'],
                        'host': config['host'],
                        'schema_name': config['schema_name'],
                        'table_name': table_name,
                        'query':
                        config['query_test_plot']},
                        [i + "_confidences_aggregated" for i in
                         label_names],
                        sten_path_test)
    count()


if __name__ == '__main__':
    args = parser.parse_args()
    if args.model_type == 'classification':
        from miacag.scripts.stenosis_identifier.submit_angio import stenosis_identifier
    elif args.model_type == 'regression':
        from miacag.scripts.stenosis_regression.submit_angio import stenosis_identifier
    else:
        raise ValueError('this type is not implemented')
    table_name = stenosis_identifier(
        args.cpu, args.num_workers,
        args.config_path_stenosis_rca_r_dom)
    stenosis_identifier(
        args.cpu, args.num_workers,
        args.config_path_stenosis_rca_l_dom, table_name)

    stenosis_identifier(
        args.cpu, args.num_workers,
        args.config_path_stenosis_rca_co_dom, table_name)

    stenosis_identifier(
        args.cpu, args.num_workers,
        args.config_path_stenosis_lca_r_dom, table_name)

    stenosis_identifier(
        args.cpu, args.num_workers,
        args.config_path_stenosis_lca_l_dom, table_name)

    stenosis_identifier(
        args.cpu, args.num_workers,
        args.config_path_stenosis_lca_co_dom, table_name)

    config_path_list = [
        args.config_path_stenosis_rca_r_dom,
        args.config_path_stenosis_rca_l_dom,
        args.config_path_stenosis_rca_co_dom,
        args.config_path_stenosis_lca_r_dom,
        args.config_path_stenosis_lca_l_dom,
        args.config_path_stenosis_lca_co_dom]

    config_path_list = [
        args.config_path_stenosis_rca_r_dom,
        args.config_path_stenosis_rca_l_dom,
        args.config_path_stenosis_rca_co_dom,
        args.config_path_stenosis_lca_r_dom,
        args.config_path_stenosis_lca_l_dom,
        args.config_path_stenosis_lca_co_dom
    ]

    if torch.distributed.get_rank() == 0:
        combine_results(config_path_list, table_name)
