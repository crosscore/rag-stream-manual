import os
import psycopg2
from psycopg2 import sql, Error
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

is_docker = os.getenv("IS_DOCKER", "false").lower() == "true"

try:
    if is_docker:
        host = os.environ["TOC_DB_INTERNAL_HOST"]
        port = int(os.environ["TOC_DB_INTERNAL_PORT"])
    else:
        host = os.environ["TOC_DB_EXTERNAL_HOST"]
        port = int(os.environ["TOC_DB_EXTERNAL_PORT"])

    conn = psycopg2.connect(
        dbname=os.environ["TOC_DB_NAME"],
        user=os.environ["TOC_DB_USER"],
        password=os.environ["TOC_DB_PASSWORD"],
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

# テーブルを完全に削除
drop_table_query = sql.SQL("DROP TABLE {} CASCADE").format(sql.Identifier('toc_table'))

try:
    cursor.execute(drop_table_query)
    conn.commit()
    print("テーブルのデータが完全に削除されました。")
except Error as e:
    if e.pgcode == '42P01':  # Undefined table
        print("エラー: テーブルは存在しません。")
    else:
        print(f"エラーが発生しました: {e}")
finally:
    cursor.close()
    conn.close()
