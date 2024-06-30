# rag-stream-manual/backend/utils/reading_psql_manual.py

import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, inspect
import ast
import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def get_db_url():
    is_docker = os.getenv("IS_DOCKER", "false").lower() == "true"
    host_key = "MANUAL_DB_INTERNAL_HOST" if is_docker else "MANUAL_DB_EXTERNAL_HOST"
    port_key = "MANUAL_DB_INTERNAL_PORT" if is_docker else "MANUAL_DB_EXTERNAL_PORT"

    return "postgresql://{user}:{password}@{host}:{port}/{dbname}".format(
        user=os.getenv("MANUAL_DB_USER"),
        password=os.getenv("MANUAL_DB_PASSWORD"),
        host=os.getenv(host_key, "localhost"),
        port=os.getenv(port_key),
        dbname=os.getenv("MANUAL_DB_NAME")
    )

def get_table_structure(engine):
    inspector = inspect(engine)
    columns = inspector.get_columns('manual_table')
    return columns

def read_manual_data():
    try:
        engine = create_engine(get_db_url())
        query = "SELECT * FROM manual_table"
        df = pd.read_sql(query, engine)
        df['manual_vector'] = df['manual_vector'].apply(ast.literal_eval)
        return engine, df
    except Exception as e:
        logger.error(f"データの読み込み中にエラーが発生しました: {e}")
        raise

def print_table_info(df):
    print("\n------ Table Info ------")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumn Info:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")

def print_sample_data(df):
    print("\n------ Sample Data (1st one) ------")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(f"\n{df.head(1).to_string()}")

def main():
    try:
        engine, df = read_manual_data()

        table_structure = get_table_structure(engine)
        print("------ Table Structure ------")
        for column in table_structure:
            print(f"  - {column['name']}: {column['type']}")

        print_table_info(df)
        print_sample_data(df)

        print("\n------ manual_vector length ------")
        for i in range(min(10, len(df))):
            print(f"len(df['manual_vector'][{i}]): {len(df['manual_vector'][i])}")

    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
