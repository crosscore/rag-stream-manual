# backend/manual_vector_to_postgres.py

import pandas as pd
import os
import glob
from dotenv import load_dotenv
import psycopg2
import ast
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
    file_type TEXT,
    location TEXT,
    manual TEXT,
    manual_vector vector(3072)
);
"""
cursor.execute(create_table_query)
conn.commit()

def process_csv_files(directory, file_type):
    csv_files = glob.glob(os.path.join(directory, '*_vector_normalized.csv'))
    logger.info(f"Found {file_type} CSV files: {csv_files}")

    for input_file_path in csv_files:
        df = pd.read_csv(input_file_path)

        for index, row in df.iterrows():
            try:
                manual_vector = ast.literal_eval(row['manual_vector'])
                manual_vector = [float(x) for x in manual_vector]

                location = row['page'] if file_type == 'PDF' else row['sheet_name'] if file_type == 'XLSX' else 'N/A'

                insert_query = """
                INSERT INTO manual_table (file_name, file_type, location, manual, manual_vector)
                VALUES (%s, %s, %s, %s, %s);
                """
                cursor.execute(insert_query, (
                    row['file_name'],
                    file_type,
                    location,
                    row['manual'],
                    manual_vector
                ))
                logger.info(f"Inserted row for {file_type}: {row['file_name']}")
            except Exception as e:
                logger.error(f"Error inserting row: {e}")

        conn.commit()


process_csv_files('../data/csv/pdf/', 'PDF')
process_csv_files('../data/csv/xlsx/', 'XLSX')
process_csv_files('../data/csv/docx/', 'DOCX')

cursor.close()
conn.close()

logger.info("All CSV files have been processed and inserted into the database.")
