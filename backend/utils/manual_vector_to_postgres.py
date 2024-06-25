# backend/utils/manual_vector_to_postgres.py

import pandas as pd
import os
import glob
from dotenv import load_dotenv
import psycopg2
import ast
import time
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

is_docker = os.getenv("IS_DOCKER", "false").lower() == "true"

try:
    if is_docker:
        host = os.environ["MANUAL_DB_INTERNAL_HOST"]
        port = int(os.environ["MANUAL_DB_INTERNAL_PORT"])
    else:
        host = os.environ["MANUAL_DB_EXTERNAL_HOST"]
        port = int(os.environ["MANUAL_DB_EXTERNAL_PORT"])

    conn = psycopg2.connect(
        dbname=os.environ["MANUAL_DB_NAME"],
        user=os.environ["MANUAL_DB_USER"],
        password=os.environ["MANUAL_DB_PASSWORD"],
        host=host,
        port=port
    )
    logger.info(f"Connected to database: {host}:{port}")
except KeyError as e:
    logger.error(f"Environment variable not set: {e}")
    raise
except psycopg2.Error as e:
    logger.error(f"Unable to connect to the database: {e}")
    raise

cursor = conn.cursor()

create_table_query = """
CREATE TABLE IF NOT EXISTS manual_table (
    id SERIAL PRIMARY KEY,
    file_name TEXT,
    sheet_name TEXT,
    manual TEXT,
    manual_vector vector(3072)
);
"""
cursor.execute(create_table_query)
conn.commit()

input_directory = '../data/csv/xlsx/'
csv_files = glob.glob(os.path.join(input_directory, '*_vector_normalized.csv'))
print(f"Found CSV files: {csv_files}")

# 全CSVファイルに対してベクトル化の処理を実行
for input_file_path in csv_files:
    df = pd.read_csv(input_file_path)

    # ベクトルデータをPostgreSQLに挿入
    for index, row in df.iterrows():
        try:
            print(f"Inserting row: {row}")

            # ベクトルをリスト形式に変換
            manual_vector = ast.literal_eval(row['manual_vector'])

            # リストの各要素をfloatにキャスト
            manual_vector = [float(x) for x in manual_vector]

            insert_query = """
            INSERT INTO manual_table (file_name, sheet_name, manual, manual_vector)
            VALUES (%s, %s, %s, %s);
            """
            cursor.execute(insert_query, (row['file_name'], row['sheet_name'], row['manual'], manual_vector))
            print("Row inserted")
        except Exception as e:
            print(f"Error inserting row: {e}")

    conn.commit()

cursor.close()
conn.close()
