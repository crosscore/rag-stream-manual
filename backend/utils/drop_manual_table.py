# rag-stream-manual/backend/utils/drop_manual_table.py

import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
import logging

load_dotenv()

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_url():
    is_docker = os.getenv("IS_DOCKER", "false").lower() == "true"
    host_key = "MANUAL_DB_INTERNAL_HOST" if is_docker else "MANUAL_DB_EXTERNAL_HOST"
    port_key = "MANUAL_DB_INTERNAL_PORT" if is_docker else "MANUAL_DB_EXTERNAL_PORT"

    return f"postgresql://{os.getenv('MANUAL_DB_USER')}:{os.getenv('MANUAL_DB_PASSWORD')}@" \
            f"{os.getenv(host_key, 'localhost')}:{os.getenv(port_key)}/{os.getenv('MANUAL_DB_NAME')}"

def get_db_connection():
    try:
        return psycopg2.connect(get_db_url())
    except psycopg2.Error as e:
        logger.error(f"Unable to connect to the database: {e}")
        raise

def print_table_info(df):
    print("\n------ Table Info ------")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumn Info:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")

def get_table_data():
    try:
        engine = create_engine(get_db_url())
        return pd.read_sql("SELECT * FROM manual_table", engine)
    except Exception as e:
        logger.error(f"Error reading table data: {e}")
        return None

def drop_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    drop_table_query = sql.SQL("DROP TABLE {} CASCADE").format(sql.Identifier('manual_table'))

    try:
        cursor.execute(drop_table_query)
        conn.commit()
        print("テーブルが完全に削除されました。")
    except psycopg2.errors.UndefinedTable:
        print("エラー: テーブルは存在しません。")
    except psycopg2.Error as e:
        logger.error(f"エラーが発生しました: {e}")
    finally:
        cursor.close()
        conn.close()

def main():
    df = get_table_data()
    if df is not None and not df.empty:
        print_table_info(df)
        drop_table()
    else:
        print("テーブルが存在しないか、データの取得に失敗しました。")

if __name__ == "__main__":
    main()
