# rag-stream-manual/backend/utils/reading_psql_manual.py

import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect, text
import ast
import logging
import struct

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

def get_index_info(engine):
    with engine.connect() as connection:
        query = text("""
        SELECT
            i.relname AS index_name,
            a.attname AS column_name,
            ix.indisprimary AS is_primary,
            ix.indisunique AS is_unique,
            am.amname AS index_type,
            pg_get_indexdef(i.oid) AS index_definition
        FROM
            pg_index ix
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_am am ON i.relam = am.oid
            LEFT JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
        WHERE
            t.relname = 'manual_table'
        ORDER BY
            i.relname, a.attnum;
        """)
        result = connection.execute(query)
        return result.fetchall()

def get_hnsw_index_settings(engine):
    with engine.connect() as connection:
        query = text("""
        SELECT reloptions
        FROM pg_class
        WHERE relname = 'hnsw_manual_vector_idx';
        """)
        result = connection.execute(query)
        return result.fetchone()

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

def verify_halfvec(engine):
    with engine.connect() as connection:
        query = text("""
        SELECT manual_vector::text
        FROM manual_table
        LIMIT 1;
        """)
        result = connection.execute(query)
        db_vector = result.fetchone()[0]

    logger.info("\n------ データベース内のベクトル形式 ------")
    logger.info(f"データベースから取得したベクトル（最初の10要素）: {db_vector[:100]}...")

def compare_float_representations(df):
    logger.info("\n------ 浮動小数点数の表現比較 ------")
    vector = df['manual_vector'][0]

    # float32 (単精度浮動小数点数)
    vector_f32 = np.array(vector, dtype=np.float32)

    # float16 (半精度浮動小数点数)
    vector_f16 = np.array(vector, dtype=np.float16)

    logger.info("最初の10要素の比較:")
    logger.info(f"元のベクトル:           {vector[:10]}")
    logger.info(f"float32として解釈:      {vector_f32[:10]}")
    logger.info(f"float16として解釈:      {vector_f16[:10]}")

def check_binary_representation(df):
    logger.info("\n------ バイナリ表現の確認 ------")
    vector = df['manual_vector'][0]

    # float32のバイナリ表現
    f32_binary = struct.pack('f', vector[0])
    f32_hex = f32_binary.hex()

    # float16のバイナリ表現
    f16_val = np.float16(vector[0])
    f16_binary = struct.pack('e', f16_val)
    f16_hex = f16_binary.hex()

    logger.info(f"最初の要素の値: {vector[0]}")
    logger.info(f"float32のバイナリ表現（16進数）: {f32_hex}")
    logger.info(f"float16のバイナリ表現（16進数）: {f16_hex}")

def main():
    try:
        logger.info("データベースからの読み取りを開始します。")
        engine, df = read_manual_data()

        table_structure = get_table_structure(engine)
        logger.info("------ テーブル構造 ------")
        for column in table_structure:
            logger.info(f"  - {column['name']}: {column['type']}")

        logger.info("\n------ インデックス情報 ------")
        index_info = get_index_info(engine)
        for index in index_info:
            logger.info(f"インデックス名: {index.index_name}")
            logger.info(f"  カラム: {index.column_name}")
            logger.info(f"  タイプ: {index.index_type}")
            logger.info(f"  定義: {index.index_definition}")
            logger.info("  ---")

        hnsw_settings = get_hnsw_index_settings(engine)
        if hnsw_settings:
            logger.info("\n------ HNSWインデックス設定 ------")
            logger.info(f"設定: {hnsw_settings[0]}")

        log_table_info(df)
        log_sample_data(df)

        logger.info("\n------ manual_vector の長さ ------")
        for i in range(min(10, len(df))):
            logger.info(f"len(df['manual_vector'][{i}]): {len(df['manual_vector'][i])}")

        # 新しい検証関数の呼び出し
        verify_halfvec(engine)
        compare_float_representations(df)
        check_binary_representation(df)

        logger.info("データベースの読み取りが完了しました。")
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
