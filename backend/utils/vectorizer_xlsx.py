# backend/utils/read_vectorize_xlsx.py

import os
import requests
import pandas as pd
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
import numpy as np
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT=os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
XLSX_CHUNK_SIZE = 100
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 0))
SEPARATOR = "\n"
XLSX_CSV_OUTPUT_DIR = os.getenv("XLSX_CSV_OUTPUT_DIR", "../data/csv/xlsx")

is_docker = os.getenv("IS_DOCKER", "false").lower() == "true"
if is_docker:
    S3_DB_URL = os.getenv("S3_DB_INTERNAL_URL", "http://s3_db:9000")
else:
    S3_DB_URL = os.getenv("S3_DB_EXTERNAL_URL", "http://localhost:9001")

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
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

def process_and_vectorize_xlsx_file(file_name, max_chunk_size=100):
    try:
        response = requests.get(f"{S3_DB_URL}/data/xlsx/{file_name}", stream=True)
        if response.status_code != 200:
            print(f"Error fetching file {file_name}: {response.status_code}")
            return pd.DataFrame()

        xlsx = pd.ExcelFile(BytesIO(response.content))
        print(f"Successfully loaded file: {file_name}")

        processed_data = []
        for sheet_name in xlsx.sheet_names:
            print(f"Processing sheet: {sheet_name}")
            df = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)
            print(f"Sheet {sheet_name} shape: {df.shape}")

            # 完全に空の行を削除
            df = df.dropna(how='all')

            print(f"Sheet shape after cleaning: {df.shape}")

            if df.empty:
                print(f"Sheet {sheet_name} is empty after cleaning")
                continue

            current_chunk = []
            current_chunk_size = 0

            for _, row in df.iterrows():
                row_content = " | ".join([str(value).strip() for value in row if pd.notna(value) and str(value).strip()])

                # 空の行をスキップ
                if not row_content:
                    continue

                row_length = len(row_content)

                if max_chunk_size == 0 or current_chunk_size + row_length > max_chunk_size:
                    if current_chunk:
                        chunk_text = "\n".join(current_chunk)
                        print(f"Processing chunk: {chunk_text[:50]}...")
                        vector = embeddings.embed_query(chunk_text)
                        normalized_vector = normalize_vector(vector).tolist()
                        processed_data.append({
                            'file_name': file_name,
                            'sheet_name': sheet_name,
                            'manual': chunk_text,
                            'manual_vector': normalized_vector
                        })
                        current_chunk = []
                        current_chunk_size = 0

                current_chunk.append(row_content)
                current_chunk_size += row_length

            # 残りのチャンクを処理
            if current_chunk:
                chunk_text = "\n".join(current_chunk)
                print(f"Processing final chunk: {chunk_text[:50]}...")
                vector = embeddings.embed_query(chunk_text)
                normalized_vector = normalize_vector(vector).tolist()
                processed_data.append({
                    'file_name': file_name,
                    'sheet_name': sheet_name,
                    'manual': chunk_text,
                    'manual_vector': normalized_vector
                })

        print(f"Processed {len(processed_data)} chunks from file {file_name}")
        return pd.DataFrame(processed_data)

    except Exception as e:
        print(f"Error processing file {file_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

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
            print(f"Processed data shape: {processed_data.shape}")
            print(f"Columns: {processed_data.columns.tolist()}")
            print(f"Sample data:\n{processed_data.head()}")
        else:
            print(f"No data processed for {file_name}")

if __name__ == "__main__":
    main()
