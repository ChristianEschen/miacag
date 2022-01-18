import psycopg2
import pandas as pd


def getDataFromDatabase(sql_config):
    connection = psycopg2.connect(
        host=sql_config['host'],
        database=sql_config['database'],
        user=sql_config['username'],
        password=sql_config['password'])
    sql = sql_config['query'].replace(
        "?table_name", "\"" + sql_config['table_name'] + "\"")
    df = pd.read_sql_query(sql, connection)
    if len(df) == 0:
        print('The requested query does not have any data!')
    return df, connection


def update_cols(con, records, sql_config, cols, page_size=2):
    cur = con.cursor()
    values = []
    for record in records:
        value = tuple([record[i] for i in cols+['rowid']])
        values.append(value)
    values = tuple(values)
    update_query = """
    UPDATE "{}" AS t
    SET phase = e.phase
    FROM (VALUES %s) AS e(phase, labels, rowid)
    WHERE e.rowid = t.rowid;""".format(sql_config['table_name'])

    psycopg2.extras.execute_values(
        cur, update_query, values, template=None, page_size=100
    )
    con.commit()

def copy_table(sql_config):
    sql = """
        CREATE TABLE "{}" as
        (SELECT * FROM "{}");
        """.format(sql_config['table_name_output'],
                   sql_config['table_name_input'])

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
