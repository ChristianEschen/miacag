import psycopg2
import pandas as pd
import psycopg2.extras
import numpy as np
from tqdm import tqdm


def getDataFromDatabase(sql_config):
    connection = psycopg2.connect(
        host=sql_config['host'],
        database=sql_config['database'],
        user=sql_config['username'],
        password=sql_config['password'])
    sql = sql_config['query'].replace(
        "?table_name", "\"" + sql_config['table_name'] + "\"")
    sql = sql.replace(
        "?schema_name", "\"" + sql_config['schema_name'] + "\"")
    sql = sql.replace(
        "??", "\"")
    df = pd.read_sql_query(sql, connection)
    if len(df) == 0:
        print('The requested query does not have any data!')
    connection.close()
    return df, connection


def cols_to_set(cols):
    if len(cols) == 1:
        base = "?? = e.??"
        string = base.replace("??", cols[0])
    else:
        string = []
        base = "?? = e.??, "
        for i in cols:
            string.append(base.replace("??", i))
        string = "".join(string)
        string = string[:-2]
    return string


# def update_cols(con, records, sql_config, cols, page_size=2):
#     cur = con.cursor()
#     values = []
#     for record in records:
#         value = tuple([record[i] for i in cols+['rowid']])
#         values.append(value)
#     values = tuple(values)
#     string = cols_to_set(cols)
#     update_query = """
#     UPDATE "{schema_name}"."{table_name}" AS t
#     SET {cols_to_set}
#     FROM (VALUES %s) AS e({cols})
#     WHERE e.rowid = t.rowid;""".format(
#         schema_name=sql_config['schema_name'],
#         table_name=sql_config['table_name'],
#         cols=', '.join(cols+['rowid']),
#         cols_to_set=string)

#     psycopg2.extras.execute_values(
#         cur, update_query, values, template=None, page_size=100
#     )
#     con.commit()

def update_cols(records, sql_config, cols, page_size=2):
    conn = psycopg2.connect(
        host=sql_config['host'],
        database=sql_config['database'],
        user=sql_config['username'],
        password=sql_config['password'])
    values = []
    for record in records:
        value = tuple([record[i] for i in cols+['rowid']])
        values.append(value)
    values = tuple(values)
    string = cols_to_set(cols)
    update_query = """
    UPDATE "{schema_name}"."{table_name}" AS t
    SET {cols_to_set}
    FROM (VALUES %s) AS e({cols})
    WHERE e.rowid = t.rowid;""".format(
        schema_name=sql_config['schema_name'],
        table_name=sql_config['table_name'],
        cols=', '.join(cols+['rowid']),
        cols_to_set=string)
    cur = conn.cursor()
    n = 10000   
    print('len(values): ', len(values))
    with tqdm(total=len(values)) as pbar:
        for i in range(0, len(values), n):
            psycopg2.extras.execute_values(
                cur, update_query, values[i:i + n], template=None, page_size=n
                )
            conn.commit()
            pbar.update(cur.rowcount)
    cur.close()
    conn.close()

def copy_table(sql_config):
    sql = """
        CREATE TABLE {schema_name}."{table_name_in}" as
        (SELECT * FROM {schema_name}."{table_name_out}");
        """.format(
            schema_name=sql_config['schema_name'],
            table_name_in=sql_config['table_name_output'],
            table_name_out=sql_config['table_name_input'])

    connection = psycopg2.connect(
            host=sql_config['host'],
            database=sql_config['database'],
            user=sql_config['username'],
            password=sql_config['password'])
    cursor = connection.cursor()
    cursor.execute(sql)
    cursor.execute("COMMIT;")
    cursor.close()
    connection.close()


def add_columns(sql_config, column_names, data_types):

    for count, column_name in enumerate(column_names):
        data_type = data_types[count]
        sql = """
        ALTER TABLE "{schema_name}"."{table_name}"
        ADD COLUMN IF NOT EXISTS "{col_name}" {d_type};
        """.format(schema_name=sql_config['schema_name'],
                   table_name=sql_config['table_name'],
                   col_name=column_name,
                   d_type=data_type)

        connection = psycopg2.connect(
                host=sql_config['host'],
                database=sql_config['database'],
                user=sql_config['username'],
                password=sql_config['password'])
        cursor = connection.cursor()

        cursor.execute(sql)
        cursor.execute("COMMIT;")
        cursor.close()
        connection.close()

    return None


def changeDtypes(sql_config, columnm_names, data_types):

    for count, columnm_name in enumerate(columnm_names):
        connection = psycopg2.connect(
            host=sql_config['host'],
            database=sql_config['database'],
            user=sql_config['username'],
            password=sql_config['password'])
        data_type = data_types[count]
        sql = """
        ALTER TABLE "{schema_name}"."{table_name}"
        ALTER COLUMN "{col_name}" TYPE {dtype};""".format(
            schema_name=sql_config['schema_name'],
            table_name=sql_config["table_name"],
            col_name=columnm_name,
            dtype=data_type
            )
        cursor = connection.cursor()

        cursor.execute(sql)
        cursor.execute("COMMIT;")
        cursor.close()
        connection.close()
    return None


def copyCol(sql_config,
            source_column,
            destination_column):
    connection = psycopg2.connect(
            host=sql_config['host'],
            database=sql_config['database'],
            user=sql_config['username'],
            password=sql_config['password'])
    cursor = connection.cursor()

    combined = [destination_column[i] +" = " +source_column[i] + ", " for i in range(0, len(destination_column))]
    updt_part = "".join(combined)
    updt_part = updt_part[:-2]
    sql = """
    UPDATE "{schema_name}"."{table_name}"
    SET {updt_part};""".format(
        schema_name=sql_config['schema_name'],
        table_name=sql_config['table_name'],
        updt_part=updt_part)
    print('sql', sql)
    cursor.execute(sql)
    cursor.execute("COMMIT;")

    cursor.close()
    connection.close()
    return None
