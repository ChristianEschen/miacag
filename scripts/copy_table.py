import psycopg2
import argparse

parser = argparse.ArgumentParser(
    description='Define inputs for copy table')
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
    '--table_name_input', type=str,
    help="table name in database")
parser.add_argument(
    '--table_name_output', type=str,
    help="table name in database")


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


if __name__ == "__main__":
    args = parser.parse_args()
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
                  'table_name_input':
                  args.table_name_input,
                  'table_name_output':
                  args.table_name_output
                  }
    copy_table(sql_config)
