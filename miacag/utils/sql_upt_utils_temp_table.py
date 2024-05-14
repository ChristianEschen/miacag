import pandas as pd
import os
import psycopg2
from tqdm import tqdm
import time
import numpy as np
def update_using_temp_table(temp_csv_filename, final_df, config):

    
    try:
        os.remove(temp_csv_filename)
    except OSError as e:
        print('file does not exists')
    final_df.to_csv(temp_csv_filename, index=False)
    print('final df', final_df.columns)
    # Step 2: Create a temporary table in PostgreSQL with a structure matching the columns of final_df
   # columns_definition = ", ".join([f"{col} VARCHAR" for col in final_df.columns if col != "rowid"])
    columns_definition = ", ".join([f"{col} float8" if col.endswith('_predictions') else f"{col} VARCHAR" for col in final_df.columns if col != "rowid"])

    columns_names = [col for col in final_df.columns if col != "rowid"]

    print('column definition', columns_definition)
    conn = psycopg2.connect(
        dbname=config["database"],
        user=config["username"],
        password=config["password"],
        host=config["host"])
    cursor = conn.cursor()
    sql = f"""
        CREATE TEMP TABLE tmp_update_table (rowid INT, {columns_definition});
    """
    print('sql', sql)

    cursor.execute(sql)
    # temp table
    for col_def in columns_names:
        sql_create_idx = """
        CREATE INDEX IF NOT EXISTS idx_{columns_definition} on
        tmp_update_table ({columns_definition});""".format(columns_definition=col_def)
        print('sql_create_idx', sql_create_idx)

        cursor.execute(sql_create_idx)

    # rowid
    sql_create_idx = "CREATE INDEX IF NOT EXISTS idx_rowid on tmp_update_table (rowid);"
    print('sql_create_idx', sql_create_idx)
    cursor.execute(sql_create_idx)
            
    # table to update:
    for col_def in columns_names:

        sql_create_idx = """
        CREATE INDEX IF NOT EXISTS idx_{columns_definition} on
        {schema_name}.\"{table_name}\" ({columns_definition});""".format(
            columns_definition=col_def, schema_name=config["schema_name"], table_name=config["table_name"])
        print('sql_create_idx', sql_create_idx)

        cursor.execute(sql_create_idx)
    # rowid
    sql_create_idx = "CREATE INDEX IF NOT EXISTS idx_rowid on {schema_name}.\"{table_name}\" (rowid);".format(
        schema_name=config["schema_name"], table_name=config["table_name"])
    cursor.execute(sql_create_idx)


    # Step 3: Use the COPY command to bulk load the CSV data into the temporary table
    with open(temp_csv_filename, 'r') as f:
        cursor.copy_expert(f"COPY tmp_update_table FROM STDIN WITH CSV HEADER", f)


    # Step 4: Perform an UPDATE join between the main table and the temporary table
    set_clauses = ", ".join(f"{col} = tmp.{col}" for col in columns_names if col != "rowid")
    print('set clause', set_clauses)
    cursor.execute(f"""
        UPDATE {config['schema_name']}.\"{config['table_name']}\" main
        SET {set_clauses}
        FROM tmp_update_table tmp
        WHERE main.rowid = tmp.rowid;
    """)

    # Commit changes
    conn.commit()

    # Step 5: The temporary table will be automatically deleted when the session ends.
    # If you've used a non-temp table, you'd drop it here.

    cursor.close()
    conn.close()

    # Cleanup: Remove the temporary CSV file
    os.remove(temp_csv_filename)
    return None

def add_prediction_names(df, parent_folder_basename, config):
    if not config['loss']['name'][0].startswith(tuple(['CE', 'NNL'])):
        df[parent_folder_basename + '_predictions'] = df[parent_folder_basename +'_confidences'].str.extract(r':([\-0-9.]+)')[0].astype(float)

    if config['loss']['name'][0].startswith('CE'):
        df[parent_folder_basename + '_predictions'] = \
            df[parent_folder_basename + '_predictions'].apply(
                np.round).astype(int)
        # df[parent_folder_basename + '_predictions'] = \
        #     df[parent_folder_basename + '_predictions'].apply(np.argmax)
    elif config['loss']['name'][0] in ['MSE', '_L1', 'L1smooth', 'NNL', 'wfocall1']:
        df[parent_folder_basename + '_predictions'] = \
            df[parent_folder_basename + '_predictions'].astype(float)
    elif config['loss']['name'][0] == 'BCE_multilabel':
        df[parent_folder_basename + '_predictions'] = \
            df[parent_folder_basename + '_predictions'].apply(
                np.round).astype(int)
    else:
        raise ValueError(
            'not implemented loss: ', config['loss']['name'][0])
    return df

def handle_CE_NLL_loss(df, parent_folder_basename, config, update_column_name):
    # 
    col_names = [config['labels_names'][0] + '_confidence_' + str(i) for i in range(0, config['model']['num_classes'][0])]
    df_red = df[col_names]
    
    df_to_array =np.array(df_red)
    idx = np.argmax(df_to_array, 1)
    max_vals = np.max(df_to_array, 1)
    print('handles this for nnl loss!')
    max_vals = []
    for c, row in enumerate(df_to_array):
        max_v = "{"
        for c_v, v in enumerate(row):
            #v = np.around(v, decimals=5)
            max_v = max_v + str(c_v) + ":" + str(v) + ";"
        max_v = max_v[:-1] + "}"
        max_vals.append(max_v)
  #  max_vals = ["{"+";".join(map(str, row)) +"}"for row in df_to_array]
    
        
    
    df_res = pd.DataFrame([list(idx), list(max_vals), list(df["rowid"])], [update_column_name[:-11]+ 'predictions', update_column_name, "rowid"]).transpose()

    return df_res
def handle_reg_loss(df, parent_folder_basename, config, update_column_name):
    col1 = 0
    col_to_updt = parent_folder_basename +"_confidence_" + str(col1)
    total_string = "{0:" + df[col_to_updt].astype(str) +"}" 
    df[update_column_name] = total_string
    return df
def comnbine_csv_files(df_list, config):
    dfs = []

    update_column_names = []
    for df in df_list:
        if config["loss"]["name"][0] == 'CE':
            label_name = config['labels_names'][0] + '_confidence_0'
        else:
            mask  =[i.startswith('sten') for i in list(df.columns)]
            label_name = np.array(list(df.columns))[np.array(mask)][0]
        parent_folder_basename = label_name[0:-13]
        confidence_name = label_name[0:-2] + 's'
        prediction_name = label_name[0:-12] + 'predictions'
        # Read the CSV into a DataFrame
      #  df = pd.read_csv(csv_file)
        
        # get name of parent folder
        # parent_folder = os.path.basename(os.path.dirname(csv_file))
        # # get basename of parent folder
        # parent_folder_basename = os.path.basename(parent_folder)

        # update_column_name = parent_folder_basename + "_confidences"
        update_column_names.append(
            confidence_name)
        update_column_names.append(
            prediction_name)
        if config['loss']['name'][0].startswith(tuple(['NNL', 'CE'])):
            df = handle_CE_NLL_loss(df, parent_folder_basename, config,confidence_name)
            #df[update_column_name] = "{0:" + df[update_column_name].astype(str) +"}"
        else:
            df = handle_reg_loss(df, parent_folder_basename, config,confidence_name)


        df = add_prediction_names(df, parent_folder_basename, config)

        dfs.append(df)

    # Concatenate DataFrames horizontally
    final_df = pd.concat(dfs, axis=1)
    final_df = final_df.loc[:,~final_df.columns.duplicated()].copy()
    final_df["rowid"] = final_df["rowid"].astype(int)
    final_df = final_df[['rowid'] + update_column_names]
    return final_df

def merge_dfs_pr_label_names(labels_names_dirs):
    csv_files = os.listdir(labels_names_dirs)
    df_list = []
    for csv_file in csv_files:
        file_name = os.path.join(labels_names_dirs, csv_file)
        df = pd.read_csv(file_name)
        df_list.append(df)
    final_df = pd.concat(df_list, axis=0)
    return final_df

def get_dataframes_pr_label_names(parent_dir):
    labels_names_dirs = os.listdir(parent_dir)
    df_list = []
    for labels_names_dir in labels_names_dirs:
        csv_file = os.path.join(parent_dir, labels_names_dir)
        df = merge_dfs_pr_label_names(csv_file)

        df_list.append(df)
    return df_list    
    

def update_cols_based_on_temp_table(parent_dir, config):
    temp_csv_filename = os.path.join(os.path.dirname(parent_dir), 'temp_csv_file.csv')
   # csv_files = [os.path.join(parent_dir, folder, "0.csv") for folder in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, folder))]
    df_list  =get_dataframes_pr_label_names(parent_dir)
    # merge files with same upper folder name
    
    final_df = comnbine_csv_files(df_list, config)
    update_using_temp_table(temp_csv_filename, final_df, config)
    return None
if __name__ == "__main__":
    parent_dir = "/home/alatar/miacag/output/outputs_stenosis_reg/classification_config_angio_SEP_Oct18_19-57-47/csv_files_pred/"
    temp_csv_filename = os.path.join(os.path.dirname(parent_dir), 'temp_csv_file.csv')
    csv_files = [os.path.join(parent_dir, folder, "0.csv") for folder in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, folder))]
    
    config = {"schema_name": "cag",
              "table_name": "dicom_table2x",
                "username": 'alatar',
                "password": '123qweasd',
                "database": 'mydb',
                "host": "localhost",
            'loss': {
                'name': ['wfocall1']
            }}
   # config['loss']['name'] = 'wfocall1'
    final_df = comnbine_csv_files(csv_files, config)
    update_using_temp_table(temp_csv_filename, final_df, config)
    
