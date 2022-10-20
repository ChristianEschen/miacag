import uuid
import os
import socket
from datetime import datetime, timedelta
import yaml
from miacag.preprocessing.split_train_val import splitter
from miacag.utils.sql_utils import copy_table, add_columns
import pandas as pd
from miacag.postprocessing.append_results import appendDataFrame
import torch
from miacag.predictor import pred
from miacag.configs.config import load_config, maybe_create_tensorboard_logdir
from miacag.configs.options import TrainOptions
import argparse
from miacag.preprocessing.labels_map import labelsMap
from miacag.preprocessing.utils.check_experiments import checkExpExists, \
    checkCsvExists
from miacag.plots.plotter import plot_results
import pandas as pd
from miacag.utils.script_utils import create_empty_csv, mkFolder, maybe_remove, write_file, test_for_file


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
    '--config_path', type=str,
    help="path to folder with config files")
parser.add_argument(
    "--model_path", type=str,
    help="model path (not the final model.pt), (only used for predictions")


def angio_predict(cpu, num_workers, config_path, model_path):
    torch.distributed.init_process_group(
            backend="nccl" if cpu == "False" else "Gloo",
            init_method="env://",
            timeout=timedelta(seconds=180000)
            )
    # config_path = [
    #     os.path.join(config_path, i) for i in os.listdir(config_path)]

    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config['master_port'] = os.environ['MASTER_PORT']
    config['num_workers'] = num_workers
    config['cpu'] = cpu
    temp_file = os.path.join(config['output'], 'temp.txt')
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        maybe_remove(temp_file)
        experiment_name = datetime.now().strftime('%b%d_%H-%M-%S')
        write_file(temp_file, experiment_name)
    torch.distributed.barrier()
    experiment_name = test_for_file(temp_file)[0]
    output_directory = os.path.join(
                config['output'],
                experiment_name)
    output_table_name = config['table_name'] + '_' + experiment_name

    # begin pipeline
    # 1. copy table
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:

        copy_table(sql_config={
            'database': config['database'],
            'username': config['username'],
            'password': config['password'],
            'host': config['host'],
            'schema_name': config['schema_name'],
            'table_name_input': config['table_name'],
            'table_name_output': output_table_name})

     

    # add placeholder for confidences
    confs = [i + '_confidences' for i in config['labels_names']]

    # add placeholder for predictions
    preds = [i + '_predictions' for i in config['labels_names']]
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        add_columns({
            'database': config['database'],
            'username': config['username'],
            'password': config['password'],
            'host': config['host'],
            'schema_name': config['schema_name'],
            'table_name': output_table_name,
            'table_name_output': output_table_name},
                    confs,
                    ["VARCHAR"] * len(confs))

        add_columns({
            'database': config['database'],
            'username': config['username'],
            'password': config['password'],
            'host': config['host'],
            'schema_name': config['schema_name'],
            'table_name': output_table_name,
            'table_name_output': output_table_name},
                    preds,
                    ["float8"] * len(preds)
                    )
    torch.distributed.barrier()
    config['output'] = output_directory
    config['output_directory'] = output_directory
    config['table_name'] = output_table_name
    config['use_DDP'] = 'True'
    config['datasetFingerprintFile'] = None
    # 2. forward model (predict)

    config['model']['pretrain_model'] = output_directory
    pred({**config, 'query': config["query_pred"], 'TestSize': 1}, model_path)

if __name__ == '__main__':
    args = parser.parse_args()

    angio_predict(args.cpu, args.num_workers,
                  args.config_path, args.model_path)
