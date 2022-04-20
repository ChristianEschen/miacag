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
    sql = sql.replace(
        "??", "\"")
    df = pd.read_sql_query(sql, connection)
    if len(df) == 0:
        print('The requested query does not have any data!')

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


def update_cols(con, records, sql_config, cols, page_size=2):
    cur = con.cursor()
    values = []
    for record in records:
        value = tuple([record[i] for i in cols+['rowid']])
        values.append(value)
    values = tuple(values)
    string = cols_to_set(cols)
    update_query = """
    UPDATE "{table_name}" AS t
    SET {cols_to_set}
    FROM (VALUES %s) AS e({cols})
    WHERE e.rowid = t.rowid;""".format(
        table_name=sql_config['table_name'],
        cols=', '.join(cols+['rowid']),
        cols_to_set=string)

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


def add_columns(sql_config, column_names, data_types):

    for count, column_name in enumerate(column_names):
        data_type = data_types[count]
        sql = """
        ALTER TABLE "{}"
        ADD COLUMN "{}" {};
        """.format(sql_config['table_name'],
                   column_name,
                   data_type)

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
