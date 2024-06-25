# backend/utils/search_toc_rank.py

import psycopg2
import numpy as np
from dotenv import load_dotenv
import os
import pandas as pd
import glob

load_dotenv()

def get_env_variable(var_name, default=None):
    value = os.getenv(var_name, default)
    if value is None:
        raise ValueError(f"Environment variable {var_name} is not set")
    return value

def get_vector_from_csv(csv_file_path, row_number):
    df = pd.read_csv(csv_file_path)
    if row_number < 0 or row_number >= len(df):
        raise ValueError("Invalid row number")
    search_vector = df.iloc[row_number]['search_vector']
    print(f"検索文字列：{df.iloc[row_number]['search_text']}")
    return np.array(eval(search_vector))

def display_results(results):
    if results:
        for rank, result in enumerate(results, start=1):
            file_name, toc, page, toc_vector, distance = result
            print(f"Rank {rank}: file_name: {file_name}, toc: {toc}, page: {page}, distance: {distance}")
    else:
        print("No results found")


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

SEARCH_INDEX = 0 #検索用のインデックスを指定

try:
    csv_files = glob.glob('../data/csv/search_vector/*.csv')
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the specified directory")

    print(f"CSV file path: {csv_files[0]}")
    normalize_search_vector = get_vector_from_csv(csv_files[0], SEARCH_INDEX)
    print(f"Normalized Search Vector: {normalize_search_vector}")

    conn = psycopg2.connect(
        dbname=TOC_DB_NAME,
        user=TOC_DB_USER,
        password=TOC_DB_PASSWORD,
        host=TOC_DB_HOST,
        port=TOC_DB_PORT
    )
    cursor = conn.cursor()

    similarity_search_query = """
    SELECT file_name, toc, page, toc_vector, (toc_vector <#> %s::vector) AS distance
    FROM toc_table
    ORDER BY distance ASC
    LIMIT 11;
    """
    cursor.execute(similarity_search_query, (normalize_search_vector.tolist(),))
    results = cursor.fetchall()
    display_results(results)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
