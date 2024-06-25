# backend/utils/read_xlsx_and_vectorizer.py

import os
import requests
import pandas as pd
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import numpy as np
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
SEPARATOR = os.getenv("SEPARATOR", "\n")
XLSX_CSV_OUTPUT_DIR = os.getenv("XLSX_CSV_OUTPUT_DIR", "../data/csv/xlsx")

is_docker = os.getenv("IS_DOCKER", "false").lower() == "true"
if is_docker:
    S3_DB_URL = os.getenv("S3_DB_INTERNAL_URL", "http://s3_db:9000")
else:
    S3_DB_URL = os.getenv("S3_DB_EXTERNAL_URL", "http://localhost:9001")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY,
)

def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

def get_xlsx_files_from_s3():
    response = requests.get(f"{S3_DB_URL}/data/xlsx")
    if response.status_code == 200:
        print(f"response:{response}")
        return [file for file in response.json() if file.endswith('.xlsx')]
    else:
        print(f"Error fetching XLSX files: {response.status_code}")
        return []

def process_and_vectorize_xlsx_file(file_name):
    response = requests.get(f"{S3_DB_URL}/data/xlsx/{file_name}", stream=True)
    if response.status_code != 200:
        print(f"Error fetching file {file_name}: {response.status_code}")
        return pd.DataFrame()

    xlsx = pd.ExcelFile(BytesIO(response.content))

    processed_data = []
    for sheet_name in xlsx.sheet_names:
        df = pd.read_excel(xlsx, sheet_name=sheet_name)
        sheet_content = df.astype(str).apply(lambda x: ' '.join(x), axis=1).str.cat(sep='\n')

        text_splitter = CharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separator=SEPARATOR
        )
        chunks = text_splitter.split_text(sheet_content)

        for chunk in chunks:
            print("excute: embed_query(chunk)")
            vector = embeddings.embed_query(chunk)
            normalized_vector = normalize_vector(vector).tolist()
            processed_data.append({
                'file_name': file_name,
                'sheet_name': sheet_name,
                'manual': chunk,
                'manual_vector': normalized_vector
            })

    return pd.DataFrame(processed_data)

def main():
    print(f"Using S3_DB_URL: {S3_DB_URL}")
    xlsx_files = get_xlsx_files_from_s3()

    for file_name in xlsx_files:
        print(f"Processing file: {file_name}")
        processed_data = process_and_vectorize_xlsx_file(file_name)
        if not processed_data.empty:
            output_file = f"{os.path.splitext(file_name)[0]}_vector_normalized.csv"
            os.makedirs(XLSX_CSV_OUTPUT_DIR, exist_ok=True)
            processed_data.to_csv(os.path.join(XLSX_CSV_OUTPUT_DIR, output_file), index=False)
            print(f"Processed, vectorized, normalized, and saved data from {file_name} to {output_file}")
        else:
            print(f"No data processed for {file_name}")

if __name__ == "__main__":
    main()
