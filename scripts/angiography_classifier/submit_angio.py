import uuid
import os
import socket
from datetime import datetime
from h11 import Data
import yaml
from mia.preprocessing.split_train_val import splitter
from mia.preprocessing.utils.sql_utils import copy_table
import pandas as pd
from mia.postprocessing.append_results import appendDataFrame
import torch
from mia.trainer import train
from mia.tester import test
from mia.configs.config import load_config, maybe_create_tensorboard_logdir
from mia.configs.options import TrainOptions
import argparse
from mia.preprocessing.labels_map import labelsMap
from mia.preprocessing.utils.check_experiments import checkExpExists
from mia.plots.plotter import plot_results


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


def mkFolder(dir):
    os.makedirs(dir, exist_ok=True)

def get_open_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("",0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def create_empty_csv(output_csv_test):
    df = {'Experiment name': [],
          'Test F1 score on data labels transformed': [],
          'Test F1 score on three class labels': [],
          'Test acc on three class labels': []
          }
    return df
   # df.to_csv(output_csv_test, index=True)



#if torch.distributed.get_rank() == 0:
# input config
if __name__ == '__main__':
    args = parser.parse_args()
    torch.distributed.init_process_group(
            backend="nccl" if args.cpu == "False" else "Gloo",
            init_method="env://"
            )
    config_path = "/home/sauroman/mia/configs/classification/_3D/lca_rca"
    config_path = [os.path.join(config_path, i) for i in os.listdir(config_path)]
    master_addr = os.environ['MASTER_ADDR']

    num_workers = args.num_workers
    
    # define outputs
    for i in range(0, len(config_path)):
      #  if torch.distributed.get_rank() == 0:
        print('loading config:', config_path[i])
        with open(config_path[i]) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        
        if checkExpExists(config_path[i], config['output']) is False:
            # start process group

            if i == 0:
                output_csv_test = os.path.join(config['output'], 'results.csv')

                df_results = create_empty_csv(output_csv_test)
            config['master_port'] = os.environ['MASTER_PORT']
            config['num_workers'] = num_workers
            master_addr = os.environ['MASTER_ADDR']
            tensorboard_comment = os.path.basename(config_path[i])[:-5]
            torch.distributed.barrier()
            #if torch.distributed.get_rank() == 0:
            experiment_name = tensorboard_comment + '_' + \
                datetime.now().strftime('%b%d_%H-%M-%S') \
                + '_' + socket.gethostname() # runs this on
            torch.distributed.barrier()
            output_directory = os.path.join(
                        config['output'],
                        experiment_name)
            mkFolder(output_directory)
            create_empty_csv(output_csv_test)
            output_model = os.path.join(output_directory, "model.pt")

            output_table_name = experiment_name + "_" + config['table_name']

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
            os.system("mkdir -p {output_dir}".format(output_dir=output_directory))
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:

                copy_table(sql_config={
                    'database': config['database'],
                    'username': config['username'],
                    'password': config['password'],
                    'host': config['host'],
                    'table_name_input': config['table_name'],
                    'table_name_output': output_table_name})

                # # 2. copy config
                # os.system(
                #     "cp {config_path} {config_file_temp}".format(
                #         config_path=config_path[i],
                #         config_file_temp=config_file_temp))
                # map labels
                mapper_obj = labelsMap(
                    {
                     'labels_names': config['labels_names'],
                     'database': config['database'],
                     'username': config['username'],
                     'password': config['password'],
                     'host': config['host'],
                     'table_name': output_table_name,
                     'query': config['query_test'],
                     'TestSize': 1},
                    config['labels_dict']
                )
                trans_labels = mapper_obj()
                config['labels_names'] = trans_labels
                # 3. split train and validation , and map labels
                splitter_obj = splitter(
                    {
                     'labels_names': config['labels_names'],
                     'database': config['database'],
                     'username': config['username'],
                     'password': config['password'],
                     'host': config['host'],
                     'table_name': output_table_name,
                     'query': config['query'],
                     'TestSize': config['TestSize']},
                    config['labels_dict'])
                splitter_obj()
                # ...and map data['labels'] test
            # 4. Train model
            config['output'] = output_directory
            config['output_directory'] = output_directory
            config['table_name'] = output_table_name
            config['use_DDP'] = 'True'
        # config['create_tensorboard_timestamp'] = True
            config['datasetFingerprintFile'] = None
        # config = maybe_create_tensorboard_logdir(config)
            train(config)

            # 5 eval model

        #  config['TestSize'] = 1
        #   config['qeury'] = config['query_']
            #{ **config, 'query': config["query_test"]}
            config['model']['pretrain_model'] = output_directory
            test({**config, 'query': config["query_test"], 'TestSize': 1})

            # plotting results
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                # 6 plot results:
                # train
                plot_results({
                            'database': config['database'],
                            'username': config['username'],
                            'password': config['password'],
                            'host': config['host'],
                            'labels_names': config['labels_names'],
                            'table_name': output_table_name,
                            'query': config['query_train_plot']},
                            output_plots_train
                            )
                # val
                plot_results({
                            'database': config['database'],
                            'username': config['username'],
                            'password': config['password'],
                            'host': config['host'],
                            'labels_names': config['labels_names'],
                            'table_name': output_table_name,
                            'query': config['query_val_plot']},
                            output_plots_val
                            )
                # test
                plot_results({
                            'database': config['database'],
                            'username': config['username'],
                            'password': config['password'],
                            'host': config['host'],
                            'labels_names': config['labels_names'],
                            'table_name': output_table_name,
                            'query': config['query_test_plot']},
                            output_plots_test
                            )

                csv_results = appendDataFrame(sql_config={
                                    'labels_names': config['labels_names'],
                                    'database': config['database'],
                                    'username': config['username'],
                                    'password': config['password'],
                                    'host': config['host'],
                                    'table_name': output_table_name,
                                    'query': config['query_test_plot']},
                                df_results=df_results,
                                experiment_name=experiment_name)
                print('config files processed', str(i+1))
                print('config files to process in toal:', len(config_path))

    if torch.distributed.get_rank() == 0:
        csv_results = pd.DataFrame(csv_results)
        csv_results.to_csv(output_csv_test, index=False, header=True)


    #  os.system("""Rscript plots/plotter.r \
    #                 {output_plots_train} \
    #                 {output_table_name} \
    #                 "{query}" \
    #                 {username} \
    #                 {password} \
    #                 {host} \
    #                 {database}""".format(
    #                     output_table_name=output_table_name,
    #                     query=config['query_train_plot'],
    #                     username=config['username'],
    #                     password=config['password'],
    #                     database=config['database'],
    #                     host=config['host'],
    #                     output_plots_train=output_plots_train
    #                     ))

    #             # val
    #             os.system("""Rscript plots/plotter.r \
    #                 {output_plots_val} \
    #                 {output_table_name} \
    #                 "{query}" \
    #                 {username} \
    #                 {password} \
    #                 {host} \
    #                 {database}""".format(
    #                     output_table_name=output_table_name,
    #                     query=config['query_val_plot'],
    #                     username=config['username'],
    #                     password=config['password'],
    #                     database=config['database'],
    #                     host=config['host'],
    #                     output_plots_val=output_plots_val
    #                     ))
    #             # test
    #             os.system("""Rscript plots/plotter.r \
    #                 {output_plots_test} \
    #                 {output_table_name} \
    #                 "{query}" \
    #                 {username} \
    #                 {password} \
    #                 {host} \
    #                 {database}""".format(
    #                     output_table_name=output_table_name,
    #                     query=config['query_test_plot'],
    #                     username=config['username'],
    #                     password=config['password'],
    #                     database=config['database'],
    #                     host=config['host'],
    #                     output_plots_test=output_plots_test
    #                     ))