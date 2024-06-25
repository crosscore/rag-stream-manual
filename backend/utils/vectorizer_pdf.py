# backend/utils/read_vectorize_pdf.py

import os
import requests
import pandas as pd
import numpy as np
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 0))
CHUNK_SIZE = 300
SEPARATOR = "\n\n"
PDF_CSV_OUTPUT_DIR = os.getenv("PDF_CSV_OUTPUT_DIR", "../data/csv/pdf")

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

def get_pdf_files_from_s3():
    try:
        response = requests.get(f"{S3_DB_URL}/data/pdf", timeout=10)
        response.raise_for_status()
        print(f"Successfully fetched PDF file list. Status code: {response.status_code}")
        pdf_files = [file for file in response.json() if file.endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files: {pdf_files}")
        return pdf_files
    except requests.exceptions.RequestException as e:
        print(f"Error fetching PDF files: {e}")
        return []

def process_and_vectorize_pdf_file(file_name):
    try:
        response = requests.get(f"{S3_DB_URL}/data/pdf/{file_name}", stream=True, timeout=10)
        response.raise_for_status()
        print(f"Successfully fetched file {file_name}. Status code: {response.status_code}")

        pdf_reader = PdfReader(BytesIO(response.content))
        processed_data = []

        text_splitter = CharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separator=SEPARATOR
        )

        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()
            chunks = text_splitter.split_text(text)

            for chunk in chunks:
                print(f"Embedding chunk from page {page_num}")
                vector = embeddings.embed_query(chunk)
                normalized_vector = normalize_vector(vector).tolist()
                processed_data.append({
                    'file_name': file_name,
                    'page': page_num,
                    'manual': chunk,
                    'manual_vector': normalized_vector
                })

        return pd.DataFrame(processed_data)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching or processing file {file_name}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return pd.DataFrame()

def main():
    print(f"Using S3_DB_URL: {S3_DB_URL}")
    pdf_files = get_pdf_files_from_s3()

    if not pdf_files:
        print("No PDF files found or error occurred while fetching file list.")
        return

    for file_name in pdf_files:
        print(f"Processing file: {file_name}")
        processed_data = process_and_vectorize_pdf_file(file_name)
        if not processed_data.empty:
            output_file = f"{os.path.splitext(file_name)[0]}_vector_normalized.csv"
            os.makedirs(PDF_CSV_OUTPUT_DIR, exist_ok=True)
            processed_data.to_csv(os.path.join(PDF_CSV_OUTPUT_DIR, output_file), index=False)
            print(f"Processed, vectorized, normalized, and saved data from {file_name} to {output_file}")
        else:
            print(f"No data processed for {file_name}")

if __name__ == "__main__":
    main()
