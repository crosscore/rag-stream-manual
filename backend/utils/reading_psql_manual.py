# rag-stream-manual/backend/utils/reading_psql_manual.py

import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, inspect
import ast
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        logger.info(f"データベースから {len(df)} 行のデータを正常に読み込みました。")
        return engine, df
    except Exception as e:
        logger.error(f"データの読み込み中にエラーが発生しました: {e}")
        raise

def log_table_info(df):
    logger.info("\n------ テーブル情報 ------")
    logger.info(f"行数: {len(df)}")
    logger.info(f"列数: {len(df.columns)}")
    logger.info("列の情報:")
    for col in df.columns:
        logger.info(f"  - {col}: {df[col].dtype}")

def log_sample_data(df):
    logger.info("\n------ サンプルデータ (最初の1行) ------")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    logger.info(f"\n{df.head(1).to_string()}")

def main():
    try:
        logger.info("データベースからの読み取りを開始します。")
        engine, df = read_manual_data()

        table_structure = get_table_structure(engine)
        logger.info("------ テーブル構造 ------")
        for column in table_structure:
            logger.info(f"  - {column['name']}: {column['type']}")

        log_table_info(df)
        log_sample_data(df)

        logger.info("\n------ manual_vector の長さ ------")
        for i in range(min(10, len(df))):
            logger.info(f"len(df['manual_vector'][{i}]): {len(df['manual_vector'][i])}")

        logger.info("データベースの読み取りが完了しました。")
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
