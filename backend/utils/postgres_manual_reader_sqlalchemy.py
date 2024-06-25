# backend/utils/postgres_manual_reader_sqlalchemy.py

import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import ast

load_dotenv()

def get_env_variable(var_name, default=None):
    value = os.getenv(var_name, default)
    if value is None:
        raise ValueError(f"Environment variable {var_name} is not set")
    return value

MANUAL_DB_NAME = get_env_variable("MANUAL_DB_NAME")
MANUAL_DB_USER = get_env_variable("MANUAL_DB_USER")
MANUAL_DB_PASSWORD = get_env_variable("MANUAL_DB_PASSWORD")

is_docker = os.getenv("IS_DOCKER", "false").lower() == "true"
if is_docker:
    MANUAL_DB_HOST = get_env_variable("MANUAL_DB_INTERNAL_HOST")
    MANUAL_DB_PORT = get_env_variable("MANUAL_DB_INTERNAL_PORT")
else:
    MANUAL_DB_HOST = get_env_variable("MANUAL_DB_EXTERNAL_HOST", "localhost")
    MANUAL_DB_PORT = get_env_variable("MANUAL_DB_EXTERNAL_PORT")

# SQLAlchemyエンジンの作成
POSTGRES_MANUAL_URL = f"postgresql://{MANUAL_DB_USER}:{MANUAL_DB_PASSWORD}@{MANUAL_DB_HOST}:{MANUAL_DB_PORT}/{MANUAL_DB_NAME}"

try:
    engine = create_engine(POSTGRES_MANUAL_URL)

    query = "SELECT * FROM manual_table"
    df = pd.read_sql(query, engine)

    # manual_vectorカラムをリストに変換
    df['manual_vector'] = df['manual_vector'].apply(ast.literal_eval)
    print(df)
    print("---------")

    for i in range(min(10, len(df))):
        print(f"Length of manual_vector at index {i}: {len(df['manual_vector'][i])}")

except Exception as e:
    print(f"An error occurred: {e}")
    raise
