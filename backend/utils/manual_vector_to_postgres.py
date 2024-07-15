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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

is_docker = os.getenv("IS_DOCKER", "false").lower() == "true"
CSV_OUTPUT_DIR = os.getenv("CSV_OUTPUT_DIR", "../data/csv/all")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))
INDEX_TYPE = os.getenv("INDEX_TYPE", "ivfflat").lower()  # "hnsw", "ivfflat", or "none"
HNSW_M = int(os.getenv("HNSW_M", "16"))
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "256"))
IVFFLAT_LISTS = int(os.getenv("IVFFLAT_LISTS", "100"))
IVFFLAT_PROBES = int(os.getenv("IVFFLAT_PROBES", "10"))
VECTOR_DIMENSIONS = 3072

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
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS manual_table (
        id SERIAL PRIMARY KEY,
        file_name TEXT,
        file_type TEXT,
        location TEXT,
        manual TEXT,
        manual_vector vector({VECTOR_DIMENSIONS})
    );
    """
    cursor.execute(create_table_query)
    logger.info("Table created successfully")

    if INDEX_TYPE == "hnsw":
        create_index_query = f"""
        CREATE INDEX IF NOT EXISTS hnsw_manual_vector_idx ON manual_table
        USING hnsw ((manual_vector::halfvec({VECTOR_DIMENSIONS})) halfvec_ip_ops)
        WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});
        """
        cursor.execute(create_index_query)
        logger.info("HNSW index created successfully")
    elif INDEX_TYPE == "ivfflat":
        create_index_query = f"""
        CREATE INDEX IF NOT EXISTS ivfflat_manual_vector_idx ON manual_table
        USING ivfflat ((manual_vector::halfvec({VECTOR_DIMENSIONS})) halfvec_ip_ops)
        WITH (lists = {IVFFLAT_LISTS});
        """
        cursor.execute(create_index_query)
        # IVFFlat の probes 設定
        cursor.execute(f"SET ivfflat.probes = {IVFFLAT_PROBES};")
        logger.info("IVFFlat index created successfully")
    elif INDEX_TYPE == "none":
        logger.info("No index created as per configuration")
    else:
        raise ValueError(f"Unsupported index type: {INDEX_TYPE}")

def process_csv_file(file_path, conn):
    logger.info(f"Processing CSV file: {file_path}")
    df = pd.read_csv(file_path)

    with conn.cursor() as cursor:
        create_table_and_index(cursor)

        insert_query = f"""
        INSERT INTO manual_table (file_name, file_type, location, manual, manual_vector)
        VALUES (%s, %s, %s, %s, %s::vector({VECTOR_DIMENSIONS}));
        """

        data = []
        for _, row in df.iterrows():
            manual_vector = row['manual_vector']
            if isinstance(manual_vector, str):
                manual_vector = eval(manual_vector)
            if len(manual_vector) != VECTOR_DIMENSIONS:
                logger.warning(f"Incorrect vector dimension for row. Expected {VECTOR_DIMENSIONS}, got {len(manual_vector)}. Skipping.")
                continue

            data.append((row['file_name'], row['file_type'], row['location'], row['manual'], manual_vector))

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
        if INDEX_TYPE == "none":
            logger.info("CSV file has been processed and inserted into the database without creating any index.")
        else:
            logger.info(f"CSV file has been processed and inserted into the database with {INDEX_TYPE.upper()} index.")
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
