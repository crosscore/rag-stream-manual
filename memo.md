# rag-streaming

ports:
    - "<Host Port>:<Container Port>"

docker compose up --build -d

docker compose ps

docker compose exec pgvector_db bash

# pgvector環境のみを構築する場合
docker search pgvector
docker pull ankane/pgvector:latest

# もしくは手動でイメージをビルド
git clone --branch v0.7.2 https://github.com/pgvector/pgvector.git
cd pgvector
docker build --pull --build-arg PG_MAJOR=16 -t myuser/pgvector .

# pgvector拡張機能を有効化
CREATE EXTENSION IF NOT EXISTS vector;

psql -U user -d tocdb

# コンテナ外部から接続
psql -h localhost -U user -d toc_db -p 5432

# コンテナ内部から接続
psql -h pgvector_db -U user -d toc_db -p 5432

# .env
OPENAI_API_KEY=api_key

5432（左側）: ホストマシン側のポート番号。ホストマシンは、このポートを通じて外部からの接続を受け付けます。この場合、ホストマシンの5432ポートでPostgreSQLデータベースにアクセスできるように設定されます。

5432（右側）: コンテナ内のポート番号。コンテナ内で実行されているPostgreSQLデータベースは、このポートを使用して接続を受け付けます。PostgreSQLのデフォルトポート番号は5432であり、この設定ではデフォルトポートを使用します。

# テーブルの中身を削除
TRUNCATE TABLE toc_table;

# テーブルの削除
DROP TABLE toc_table CASCADE;

# バージョン確認
SELECT version();
\dx

CREATE TABLE IF NOT EXISTS toc_table (
    id SERIAL PRIMARY KEY,
    file_name TEXT,
    toc TEXT,
    page INTEGER,
    toc_halfvec halfvec(3072)
);

CREATE INDEX ON toc_table USING hnsw (toc_halfvec halfvec_ip_ops);

\d+ documents


# .env

CHUNK_SIZE=300
CHUNK_OVERLAP=0
SEPARATOR="\n\n"

# S3 Database
S3_DB_EXTERNAL_URL=http://localhost:9001
S3_DB_INTERNAL_HOST=s3_db
S3_DB_INTERNAL_PORT=9000
S3_DB_EXTERNAL_HOST=localhost
S3_DB_EXTERNAL_PORT=9001

# TOC Database
TOC_DB_NAME=tocdb
TOC_DB_USER=user
TOC_DB_PASSWORD=password
TOC_DB_INTERNAL_HOST=pgvector_toc
TOC_DB_INTERNAL_PORT=5432
TOC_DB_EXTERNAL_HOST=localhost
TOC_DB_EXTERNAL_PORT=5433

# Manual Database
MANUAL_DB_NAME=manualdb
MANUAL_DB_USER=user
MANUAL_DB_PASSWORD=password
MANUAL_DB_INTERNAL_HOST=pgvector_manual
MANUAL_DB_INTERNAL_PORT=5432
MANUAL_DB_EXTERNAL_HOST=localhost
MANUAL_DB_EXTERNAL_PORT=5434
