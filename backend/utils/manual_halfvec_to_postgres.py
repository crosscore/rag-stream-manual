# backend/manual_vector_to_postgres.py

import pandas as pd
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_batch
import logging
from contextlib import contextmanager
import numpy as np

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

is_docker = os.getenv("IS_DOCKER", "false").lower() == "true"
CSV_OUTPUT_DIR = os.getenv("CSV_OUTPUT_DIR", "../data/csv/all")
BATCH_SIZE = 1000  # Number of rows to insert in a single batch

@contextmanager
def get_db_connection():
    conn = None
    try:
        host = os.environ["MANUAL_DB_INTERNAL_HOST" if is_docker else "MANUAL_DB_EXTERNAL_HOST"]
        port = int(os.environ["MANUAL_DB_INTERNAL_PORT" if is_docker else "MANUAL_DB_EXTERNAL_PORT"])
        conn = psycopg2.connect(
            dbname=os.environ["MANUAL_DB_NAME"],
            user=os.environ["MANUAL_DB_USER"],
            password=os.environ["MANUAL_DB_PASSWORD"],
            host=host,
            port=port
        )
        logger.info(f"Connected to database: {host}:{port}")
        yield conn
    except (KeyError, psycopg2.Error) as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn is not None:
            conn.close()
            logger.info("Database connection closed")

def create_table_and_index(cursor):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS manual_table (
        id SERIAL PRIMARY KEY,
        file_name TEXT,
        file_type TEXT,
        location TEXT,
        manual TEXT,
        manual_vector halfvec(3072)
    );
    """
    cursor.execute(create_table_query)

    create_index_query = """
    CREATE INDEX IF NOT EXISTS hnsw_manual_vector_idx ON manual_table
    USING hnsw (manual_vector halfvec_ip_ops)
    WITH (m = 16, ef_construction = 256);
    """
    cursor.execute(create_index_query)

    logger.info("Table and HNSW index created successfully")

def process_csv_file(file_path, conn):
    logger.info(f"Processing CSV file: {file_path}")
    df = pd.read_csv(file_path)

    with conn.cursor() as cursor:
        create_table_and_index(cursor)

        insert_query = """
        INSERT INTO manual_table (file_name, file_type, location, manual, manual_vector)
        VALUES (%s, %s, %s, %s, %s::halfvec);
        """

        data = []
        for _, row in df.iterrows():
            manual_vector = row['manual_vector']
            if isinstance(manual_vector, str):
                manual_vector = eval(manual_vector)
            if len(manual_vector) != 3072:
                logger.warning(f"Incorrect vector dimension for row. Expected 3072, got {len(manual_vector)}. Skipping.")
                continue

            # Convert to halfvec (float16)
            halfvec = np.array(manual_vector, dtype=np.float16).tobytes()

            data.append((row['file_name'], row['file_type'], row['location'], row['manual'], halfvec))

        try:
            execute_batch(cursor, insert_query, data, page_size=BATCH_SIZE)
            conn.commit()
            logger.info(f"Inserted {len(data)} rows into the database")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting batch: {e}")

def main():
    csv_file = os.path.join(CSV_OUTPUT_DIR, "all_documents_vector_normalized.csv")
    if not os.path.exists(csv_file):
        logger.error(f"CSV file not found: {csv_file}")
        return

    try:
        with get_db_connection() as conn:
            process_csv_file(csv_file, conn)
        logger.info("CSV file has been processed and inserted into the database with HNSW index and halfvec.")
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
