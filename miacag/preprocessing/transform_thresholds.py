import argparse
import psycopg2
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from psycopg2.extras import execute_batch
from miacag.configs.config import load_config
from miacag.utils.sql_utils import update_cols, getDataFromDatabase
import numpy as np
from miacag.preprocessing.process_labels.process_total_occ \
    import ProcessLabelsOCC

parser = argparse.ArgumentParser(
    description='Define inputs.')
parser.add_argument(
            '--query', type=str,
            help='query for retrieving data',
            required=True)
parser.add_argument(
            '--config', type=str,
            help='Path to the YAML config file',
            required=True)
parser.add_argument(
    '--database', type=str,
    help="database name")
parser.add_argument(
    '--username', type=str,
    help="username for database")
parser.add_argument(
    '--password', type=str,
    help="password for database")
parser.add_argument(
    '--host', type=str,
    help="host for database")
parser.add_argument(
    '--port', type=str,
    help="port for database")
parser.add_argument(
    '--table_name', type=str,
    help="table_name in database")


class transformThreshold():
    def __init__(self, sql_config, config):
        self.sql_config = sql_config
        self.df, self.connection = getDataFromDatabase(sql_config=sql_config)
        self.config = config

    def __call__(self):
        for label_name in self.sql_config['labels_names']:
            if 'ffr' in label_name:
                thres = self.config['loaders']['val_method']['threshold_ffr']
                self.df[label_name][self.df[label_name] <= thres] = 1
                self.df[label_name][self.df[label_name] > thres] = 0
            elif 'sten' in label_name:

                thres = self.config['loaders']['val_method']['threshold_sten']
                if '5_lm' in label_name:
                    thres = 0.5
                self.df[label_name][self.df[label_name] >= thres] = 1
                self.df[label_name][self.df[label_name] < thres] = 0
            else:
                raise ValueError('Not implemented')
        update_cols(self.connection,
                    self.df.to_dict('records'),
                    self.sql_config,
                    self.sql_config['labels_names'],)


class transformThresholdRegression():
    def __init__(self, sql_config, config):
        self.sql_config = sql_config
        self.df, self.connection = getDataFromDatabase(sql_config=sql_config)
        self.config = config

    def __call__(self):
        for label_name in self.sql_config['labels_names']:
            if label_name.startswith('sten'):
                self.df[label_name][self.df[label_name] < 0] = 0
                self.df[label_name] = \
                    np.where((
                        (self.df[label_name] >= 100) &
                        (~np.isnan(self.df[label_name]))),
                        100, self.df[label_name])
                self.df[label_name] = \
                    np.where((
                        (~np.isnan(self.df[label_name]))),
                        self.df[label_name]/100, self.df[label_name])
            elif label_name.startswith('ffr'):
                self.df[label_name][self.df[label_name] < 0] = 0
                self.df[label_name] = \
                    np.where((
                        (self.df[label_name] >= 1) &
                        (~np.isnan(self.df[label_name]))),
                        1, self.df[label_name])
            else:
                raise ValueError(
                    'this type of label to process is not implemented')

        if self.config['process_labels'] == 'True':
            proccoessor = ProcessLabelsOCC(self.df,
                                           self.sql_config['labels_names'],
                                           self.config)
            self.df = proccoessor()
        update_cols(self.connection,
                    self.df.to_dict('records'),
                    self.sql_config,
                    self.sql_config['labels_names'],)
if __name__ == '__main__':
    args = parser.parse_args()

    config = load_config(args.config)
    labels_config = config['labels_dict']
    sql_config = {'database':
                  args.database,
                  'username':
                  args.username,
                  'password':
                  args.password,
                  'host':
                  args.host,
                  'port':
                  args.port,
                  'table_name':
                  args.table_name,
                  'query':
                  args.query,
                  'TestSize':
                  args.TestSize
                  }

    mapper = labelsMap(sql_config, labels_config)
    mapper()
