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

TOC_DB_NAME = get_env_variable("TOC_DB_NAME")
TOC_DB_USER = get_env_variable("TOC_DB_USER")
TOC_DB_PASSWORD = get_env_variable("TOC_DB_PASSWORD")

is_docker = os.getenv("IS_DOCKER", "false").lower() == "true"
if is_docker:
    TOC_DB_HOST = get_env_variable("TOC_DB_INTERNAL_HOST")
    TOC_DB_PORT = get_env_variable("TOC_DB_INTERNAL_PORT")
else:
    TOC_DB_HOST = get_env_variable("TOC_DB_EXTERNAL_HOST", "localhost")
    TOC_DB_PORT = get_env_variable("TOC_DB_EXTERNAL_PORT")

# SQLAlchemyエンジンの作成
POSTGRES_TOC_URL = f"postgresql://{TOC_DB_USER}:{TOC_DB_PASSWORD}@{TOC_DB_HOST}:{TOC_DB_PORT}/{TOC_DB_NAME}"

try:
    engine = create_engine(POSTGRES_TOC_URL)

    query = "SELECT * FROM toc_table"
    df = pd.read_sql(query, engine)

    # toc_vectorカラムをリストに変換
    df['toc_vector'] = df['toc_vector'].apply(ast.literal_eval)
    print(df)
    print("---------")

    for i in range(min(10, len(df))):
        print(f"Length of toc_vector at index {i}: {len(df['toc_vector'][i])}")

except Exception as e:
    print(f"An error occurred: {e}")
    raise
