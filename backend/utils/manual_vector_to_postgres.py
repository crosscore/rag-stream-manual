# backend/manual_vector_to_postgres.py

import pandas as pd
import os
from dotenv import load_dotenv
import psycopg2
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

is_docker = os.getenv("IS_DOCKER", "false").lower() == "true"
CSV_OUTPUT_DIR = os.getenv("CSV_OUTPUT_DIR", "../data/csv/all")

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

def process_csv_file(file_path):
    logger.info(f"Processing CSV file: {file_path}")
    df = pd.read_csv(file_path)

    for index, row in df.iterrows():
        try:
            manual_vector = row['manual_vector']
            # Convert string representation of list to actual list
            if isinstance(manual_vector, str):
                manual_vector = eval(manual_vector)

            # Ensure the vector has the correct dimension
            if len(manual_vector) != 3072:
                logger.warning(f"Incorrect vector dimension for row {index}. Expected 3072, got {len(manual_vector)}. Skipping.")
                continue

            insert_query = """
            INSERT INTO manual_table (file_name, file_type, location, manual, manual_vector)
            VALUES (%s, %s, %s, %s, %s);
            """
            cursor.execute(insert_query, (
                row['file_name'],
                row['file_type'],
                row['location'],
                row['manual'],
                manual_vector
            ))
            logger.info(f"Inserted row {index} for file: {row['file_name']}")
        except Exception as e:
            logger.error(f"Error inserting row {index}: {e}")

    conn.commit()

def main():
    csv_file = os.path.join(CSV_OUTPUT_DIR, "all_documents_vector_normalized.csv")
    if os.path.exists(csv_file):
        process_csv_file(csv_file)
        logger.info("CSV file has been processed and inserted into the database.")
    else:
        logger.error(f"CSV file not found: {csv_file}")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
