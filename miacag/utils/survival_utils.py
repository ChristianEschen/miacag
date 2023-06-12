from miacag.utils.sql_utils import add_columns
from miacag.utils.sql_utils import getDataFromDatabase
import yaml
from miacag.utils.sql_utils import update_cols

def create_cols_survival(config, output_table_name, trans_label, event):
    for i in range(0, len(trans_label)):
        event_i = event[i]
        trans_label_i = trans_label[i]

        add_columns({
            'database': config['database'],
            'username': config['username'],
            'password': config['password'],
            'host': config['host'],
            'schema_name': config['schema_name'],
            'table_name': output_table_name,
            'table_name_output': output_table_name},
                    [event_i],
                    ['int8'] * len(trans_label_i))
        
        
        create_event_col({'database': config['database'],
            'username': config['username'],
            'password': config['password'],
            'host': config['host'],
            'schema_name': config['schema_name'],
            'table_name': output_table_name,
            'table_name_output': output_table_name,
            'query': config['query_transform']},
                        event_i)
        add_columns({
            'database': config['database'],
            'username': config['username'],
            'password': config['password'],
            'host': config['host'],
            'schema_name': config['schema_name'],
            'table_name': output_table_name,
            'table_name_output': output_table_name},
                    [trans_label_i],
                    ['float8'] * len(trans_label_i))
    
        add_columns({
            'database': config['database'],
            'username': config['username'],
            'password': config['password'],
            'host': config['host'],
            'schema_name': config['schema_name'],
            'table_name': output_table_name,
            'table_name_output': output_table_name},
                    [trans_label_i +'_confidences'],
                    ['VARCHAR'] * len(trans_label_i))
        
        add_columns({
            'database': config['database'],
            'username': config['username'],
            'password': config['password'],
            'host': config['host'],
            'schema_name': config['schema_name'],
            'table_name': output_table_name,
            'table_name_output': output_table_name},
                    [trans_label_i +'_predictions'],
                    ['float8'] * len(trans_label_i))
        
        create_duration_col({'database': config['database'],
            'username': config['username'],
            'password': config['password'],
            'host': config['host'],
            'schema_name': config['schema_name'],
            'table_name': output_table_name,
            'table_name_output': output_table_name,
            'query': config['query_transform']},
                            trans_label_i)
    return None

def create_duration_col(config, duration):
    
    
    df, conn = getDataFromDatabase(config)
    
    df['individual_end_dates'] = df[['status_date','end_of_data_date']].min(axis=1)

    df[duration] = (df['individual_end_dates'] - df['TimeStamp']).dt.days
    
    update_cols(
                df.to_dict('records'),
                config,
                [duration],)
    return None

def create_event_col(config, event):
    df, conn = getDataFromDatabase(config)
    conn.close()
    df[event] = df['status']
    df[event] = df[[event]].applymap(lambda x: 1 if x == 90 else 0)
    update_cols(
                df.to_dict('records'),
                config,
                [event],)
    return None

if __name__ == '__main__':
    config = "/home/alatar/miacag/my_configs/pretrain_downstream/config.yaml"
    output_table_name = "dicom_table2x_v2"
    trans_label =  'duration'
    event = 'event'
    with open(config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config['table_name'] = output_table_name
    create_cols_survival(config, output_table_name, trans_label, event)