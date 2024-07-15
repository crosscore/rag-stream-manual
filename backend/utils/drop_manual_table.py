# rag-stream-manual/backend/utils/drop_manual_table.py

import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
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
        logger.error(f"データベースへの接続に失敗しました: {e}")
        raise

def print_table_info(df):
    logger.info("\n------ テーブル情報 ------")
    logger.info(f"行数: {len(df)}")
    logger.info(f"列数: {len(df.columns)}")
    logger.info("\n列の情報:")
    for col in df.columns:
        logger.info(f"  - {col}: {df[col].dtype}")

def get_table_data():
    try:
        engine = create_engine(get_db_url())
        df = pd.read_sql("SELECT * FROM manual_table", engine)
        logger.info(f"テーブルからデータを正常に読み込みました。行数: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"テーブルデータの読み込み中にエラーが発生しました: {e}")
        return None

def drop_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    drop_table_query = sql.SQL("DROP TABLE {} CASCADE").format(sql.Identifier('manual_table'))

    try:
        cursor.execute(drop_table_query)
        conn.commit()
        logger.info("テーブルが正常に削除されました。")
    except psycopg2.errors.UndefinedTable:
        logger.warning("指定されたテーブルは存在しません。")
    except psycopg2.Error as e:
        logger.error(f"テーブル削除中にエラーが発生しました: {e}")
    finally:
        cursor.close()
        conn.close()

def main():
    logger.info("テーブル削除プロセスを開始します。")
    df = get_table_data()
    if df is not None and not df.empty:
        print_table_info(df)
        drop_table()
    else:
        logger.warning("テーブルが存在しないか、データの取得に失敗しました。")
    logger.info("テーブル削除プロセスが完了しました。")

if __name__ == "__main__":
    main()
