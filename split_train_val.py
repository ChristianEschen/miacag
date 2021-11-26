import argparse
import psycopg2
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from psycopg2.extras import execute_batch
from configs.config import load_config


parser = argparse.ArgumentParser(
    description='Define inputs for building database.')
parser.add_argument(
            '--query', type=str,
            help='query for retrieving data',
            required=True)
parser.add_argument(
            '--config', type=str,
            help='Path to the YAML config file',
            required=True)
parser.add_argument(
    '--TestSize', type=float,
    default=0.2,
    help='Proportion of dataset used for testing')
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


class splitter():
    def __init__(self, sql_config, labels_config):
        self.sql_config = sql_config
        self.getDataFromDatabase()

        self.df = self.df[self.df['labels'].notna()]
        self.df['labels_transformed'] = self.df['labels']
        self.df = self.df.replace({'labels_transformed': labels_config})


    def groupEntriesPrPatient(self):
        '''Grouping entries pr patients'''
        X = self.df.drop('labels', 1)
        y = self.df['labels']
        if self.sql_config['TestSize'] == 1:
            return None, self.df
        else:
            gs = GroupShuffleSplit(
                n_splits=2,
                test_size=self.sql_config['TestSize'],
                random_state=0)
            train_ix, val_ix = next(
                gs.split(X, y, groups=self.df['PatientID']))
            df_train = self.df.iloc[train_ix]
            df_val = self.df.iloc[val_ix]
            self.addPhase(df_train, df_val)
            return df_train, df_val

    def addPhase(self, train_df, val_df):
        train_df['phase'] = "train"
        val_df['phase'] = "val"
        val_df = val_df[['phase', 'rowid']]
        train_df = train_df[['phase', 'rowid']]

        self.update(self.connection,
                    val_df.to_dict('records'),
                    'phase',
                    self.sql_config)
        self.update(self.connection,
                    train_df.to_dict('records'),
                    'phase',
                    self.sql_config)

    def update(self, con, records, column, page_size=2):
        cur = con.cursor()
        values = []
        for record in records:
            value = (record['phase'], record['rowid'])
            values.append(value)
        values = tuple(values)
        update_query = """
        UPDATE {} AS t
        SET phase = e.phase
        FROM (VALUES %s) AS e(phase, rowid)
        WHERE e.rowid = t.rowid;""".format(self.sql_config['table_name'])

        psycopg2.extras.execute_values(
            cur, update_query, values, template=None, page_size=100
        )
        con.commit()

    def getDataFromDatabase(self):
        self.connection = psycopg2.connect(
            host=self.sql_config['host'],
            database=self.sql_config['database'],
            user=self.sql_config['username'],
            password=self.sql_config['password'])
        self.sql = self.sql_config['query'].replace(
            "?table_name", "\"" + self.sql_config['table_name'] + "\"")
        self.df = pd.read_sql_query(self.sql, self.connection)
        if len(self.df) == 0:
            print('The requested query does not have any data!')

    def __call__(self):
        self.train_df, self.val_df = self.groupEntriesPrPatient()


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

    spl = splitter(sql_config, labels_config)
    spl()
