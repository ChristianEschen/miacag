import uuid
import os
import socket
import time
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
from miacag.plots.plotter import plot_results, plotRegression
import pandas as pd
from miacag.preprocessing.transform_thresholds import transformThresholdRegression
from miacag.preprocessing.transform_missing_floats import transformMissingFloats
from miacag.utils.script_utils import create_empty_csv, mkFolder, maybe_remove, write_file, test_for_file
from miacag.postprocessing.aggregate_pr_group import Aggregator
from miacag.postprocessing.count_stenosis_pr_group \
    import CountSignificantStenoses
import timeit

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
    help="path to folder with config files")
parser.add_argument('--table_name_input', type=str, default=None)

def stenosis_identifier(cpu, num_workers, config_path, table_name_input=None):



    print('loading config:', config_path)
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    mkFolder(config['output'])


    config['master_port'] = os.environ['MASTER_PORT']
    config['num_workers'] = num_workers
    config['cpu'] = cpu
    tensorboard_comment = os.path.basename(config_path)[:-5]
    temp_file = os.path.join(config['output'], 'temp.txt')
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        maybe_remove(temp_file)
        experiment_name = tensorboard_comment + '_' + \
            "SEP_" + \
            datetime.now().strftime('%b%d_%H-%M-%S')
        write_file(temp_file, experiment_name)
    torch.distributed.barrier()
    experiment_name = test_for_file(temp_file)[0]
    output_directory = os.path.join(
                config['output'],
                experiment_name)
    mkFolder(output_directory)
    output_config = os.path.join(output_directory,
                                    os.path.basename(config_path))
    if table_name_input is None:
        output_table_name = \
            experiment_name + "_" + config['table_name']
    else:
        output_table_name = table_name_input

    output_plots = os.path.join(output_directory, 'plots')
    mkFolder(output_plots)

    output_plots_train = os.path.join(output_plots, 'train')
    output_plots_val = os.path.join(output_plots, 'val')
    output_plots_test = os.path.join(output_plots, 'test')

    mkFolder(output_plots_train)
    mkFolder(output_plots_test)
    mkFolder(output_plots_val)

    # begin pipeline
    # 1. copy table
    os.system("mkdir -p {output_dir}".format(
        output_dir=output_directory))
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        if table_name_input is None:
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
    # rename labels and add columns;
    trans_label = [i + '_transformed' for i in config['labels_names']]
    labels_names_original = config['labels_names']
    config['labels_names'] = trans_label
    # add placeholder for confidences
    conf = [i + '_confidences' for i in config['labels_names']]
    conf_agg = [i + '_confidences_aggregated' for i in config['labels_names']]
    conf_agg_t = [i + '_confidences_aggregated_thres' for i in config['labels_names']]

    # add placeholder for predictions
    pred = [i + '_predictions' for i in config['labels_names']]

    if not config['is_already_trained']:
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            start = time.time()
            add_columns({
                'database': config['database'],
                'username': config['username'],
                'password': config['password'],
                'host': config['host'],
                'schema_name': config['schema_name'],
                'table_name': output_table_name,
                'table_name_output': output_table_name},
                        trans_label,
                        ["float8"] * len(trans_label))
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
                        ["float8"] * len(pred))
            add_columns({
                'database': config['database'],
                'username': config['username'],
                'password': config['password'],
                'host': config['host'],
                'schema_name': config['schema_name'],
                'table_name': output_table_name,
                'table_name_output': output_table_name},
                        conf_agg,
                        ["float8"] * len(conf))
            add_columns({
                'database': config['database'],
                'username': config['username'],
                'password': config['password'],
                'host': config['host'],
                'schema_name': config['schema_name'],
                'table_name': output_table_name,
                'table_name_output': output_table_name},
                        conf_agg_t,
                        ["float8"] * len(conf))
            
            add_columns({
                'database': config['database'],
                'username': config['username'],
                'password': config['password'],
                'host': config['host'],
                'schema_name': config['schema_name'],
                'table_name': output_table_name,
                'table_name_output': output_table_name},
                        ["antalsignifikantestenoser_pred"],
                        ["float8"])
            print(
                'done add columns, time elapsed: {}'.format(time.time() - start))
            # 3. split train and validation , and map labels
            start = time.time()
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
            print('done transform missing floats, time elapsed: {}'.format(time.time() - start))
            start = time.time()
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
            print('done transform threshold regression, time elapsed: {}'.format(time.time() - start))
            # change dtypes for label
            start = time.time()
            changeDtypes(
                {'database': config["database"],
                    'username': config["username"],
                    'password': config['password'],
                    'host': config['host'],
                    'schema_name': config['schema_name'],
                    'table_name': output_table_name,
                    'query': config['query_transform']},
                trans_label,
                ["float8"] * len(trans_label))
            print('done change dtypes, time elapsed: {}'.format(time.time() - start))
            start = time.time()
            splitter_obj = splitter(
                {
                    'labels_names': config['labels_names'],
                    'database': config['database'],
                    'username': config['username'],
                    'password': config['password'],
                    'host': config['host'],
                    'schema_name': config['schema_name'],
                    'table_name': output_table_name,
                    'query': config['query_split'],
                    'TestSize': config['TestSize']})
            splitter_obj()
            print('done split, time elapsed: {}'.format(time.time() - start))
            # ...and map data['labels'] test
    # 4. Train model
    config['output_directory'] = output_directory
    config['table_name'] = output_table_name
    config['use_DDP'] = 'True'
    config['datasetFingerprintFile'] = None
    if not config['is_already_trained']:

        torch.distributed.barrier()
        config['output'] = output_directory

        train(config)
        config['model']['pretrain_model'] = output_directory
    #  else:
        #config['model']['pretrain_model'] = output_directory

    # 5 eval model
    if not config['is_already_tested']:
        test({**config, 'query': config["query_test"], 'TestSize': 1})

    torch.distributed.barrier()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()


    # plotting results
    #torch.distributed.barrier()
    #if torch.distributed.get_rank() == 0:
    # 6 plot results:
    # train
    plot_results({
                'database': config['database'],
                'username': config['username'],
                'password': config['password'],
                'host': config['host'],
                'labels_names': config['labels_names'],
                'schema_name': config['schema_name'],
                'table_name': output_table_name,
                'query': config['query_train_plot']},
                config['labels_names'],
                [i + "_predictions" for i in
                    config['labels_names']],
                output_plots_train,
                config['model']['num_classes'],
                config,
                [i + "_confidences" for i in
                    config['labels_names']]
                )

    plotRegression({
                'database': config['database'],
                'username': config['username'],
                'password': config['password'],
                'host': config['host'],
                'labels_names': config['labels_names'],
                'schema_name': config['schema_name'],
                'table_name': output_table_name,
                'query': config['query_train_plot'],
                'loss_name': config['loss']['name'],
                'task_type': config['task_type']
                },
                config['labels_names'],
                conf,
                output_plots_train,
                group_aggregated=False)
        
    # val
    plot_results({
                'database': config['database'],
                'username': config['username'],
                'password': config['password'],
                'host': config['host'],
                'labels_names': config['labels_names'],
                'schema_name': config['schema_name'],
                'table_name': output_table_name,
                'query': config['query_val_plot']},
                config['labels_names'],
                [i + "_predictions" for i in
                    config['labels_names']],
                output_plots_val,
                config['model']['num_classes'],
                config,
                [i + "_confidences" for i in
                    config['labels_names']]
                )

    plotRegression({
                'database': config['database'],
                'username': config['username'],
                'password': config['password'],
                'host': config['host'],
                'labels_names': config['labels_names'],
                'schema_name': config['schema_name'],
                'table_name': output_table_name,
                'query': config['query_val_plot'],
                'loss_name': config['loss']['name'],
                'task_type': config['task_type']
                },
                config['labels_names'],
                conf,
                output_plots_val,
                group_aggregated=False)

    # test
    plot_results({
                'database': config['database'],
                'username': config['username'],
                'password': config['password'],
                'host': config['host'],
                'labels_names': config['labels_names'],
                'schema_name': config['schema_name'],
                'table_name': output_table_name,
                'query': config['query_test_plot']},
                config['labels_names'],
                [i + "_predictions" for i in
                    config['labels_names']],
                output_plots_test,
                config['model']['num_classes'],
                config,
                [i + "_confidences" for i in
                    config['labels_names']]
                )

    plotRegression({
                'database': config['database'],
                'username': config['username'],
                'password': config['password'],
                'host': config['host'],
                'labels_names': config['labels_names'],
                'schema_name': config['schema_name'],
                'table_name': output_table_name,
                'query': config['query_test_plot'],
                'loss_name': config['loss']['name'],
                'task_type': config['task_type']
                },
                config['labels_names'],
                conf,
                output_plots_test,
                group_aggregated=False)

   

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
    torch.cuda.set_device(local_rank)



    stenosis_identifier(args.cpu, args.num_workers, args.config_path, args.table_name_input)

    elapsed = timeit.default_timer() - start_time
    print('cpu', args.cpu)
    print(f"Execution time: {elapsed} seconds")

    args = parser.parse_args()
    stenosis_identifier(args.cpu, args.num_workers, args.config_path, args.table_name_input)
