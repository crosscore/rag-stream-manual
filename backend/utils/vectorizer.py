# backend/utils/vectorizer.py

import os
import requests
import pandas as pd
import numpy as np
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
CHUNK_SIZE = 300
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 0))
SEPARATOR = "\n\n"
CSV_OUTPUT_DIR = os.getenv("CSV_OUTPUT_DIR", "../data/csv")

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

def get_files_from_s3(file_type):
    response = requests.get(f"{S3_DB_URL}/data/{file_type}")
    if response.status_code == 200:
        return [file for file in response.json() if file.endswith(f'.{file_type}')]
    else:
        print(f"Error fetching {file_type.upper()} files: {response.status_code}")
        return []

def process_pdf(file_name):
    response = requests.get(f"{S3_DB_URL}/data/pdf/{file_name}", stream=True)
    if response.status_code != 200:
        print(f"Error fetching file {file_name}: {response.status_code}")
        return pd.DataFrame()

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
            vector = embeddings.embed_query(chunk)
            normalized_vector = normalize_vector(vector).tolist()
            processed_data.append({
                'file_name': file_name,
                'file_type': 'PDF',
                'location': str(page_num),
                'manual': chunk,
                'manual_vector': normalized_vector
            })

    return pd.DataFrame(processed_data)

def process_xlsx(file_name):
    response = requests.get(f"{S3_DB_URL}/data/xlsx/{file_name}", stream=True)
    if response.status_code != 200:
        print(f"Error fetching file {file_name}: {response.status_code}")
        return pd.DataFrame()

    xlsx = pd.ExcelFile(BytesIO(response.content))
    processed_data = []

    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator=SEPARATOR
    )

    for sheet_name in xlsx.sheet_names:
        df = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)
        df = df.dropna(how='all')

        if df.empty:
            continue

        text = df.to_string(index=False, header=False)
        chunks = text_splitter.split_text(text)

        for chunk in chunks:
            vector = embeddings.embed_query(chunk)
            normalized_vector = normalize_vector(vector).tolist()
            processed_data.append({
                'file_name': file_name,
                'file_type': 'XLSX',
                'location': sheet_name,
                'manual': chunk,
                'manual_vector': normalized_vector
            })

    return pd.DataFrame(processed_data)

def process_docx(file_name):
    response = requests.get(f"{S3_DB_URL}/data/docx/{file_name}", stream=True)
    if response.status_code != 200:
        print(f"Error fetching file {file_name}: {response.status_code}")
        return pd.DataFrame()

    doc = Document(BytesIO(response.content))
    processed_data = []

    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator=SEPARATOR
    )

    full_text = "\n".join([para.text for para in doc.paragraphs])
    chunks = text_splitter.split_text(full_text)

    for chunk_num, chunk in enumerate(chunks, start=1):
        vector = embeddings.embed_query(chunk)
        normalized_vector = normalize_vector(vector).tolist()
        processed_data.append({
            'file_name': file_name,
            'file_type': 'DOCX',
            'location': str(chunk_num),
            'manual': chunk,
            'manual_vector': normalized_vector
        })

    return pd.DataFrame(processed_data)

def main():
    file_types = ['pdf', 'xlsx', 'docx']
    all_processed_data = pd.DataFrame()

    for file_type in file_types:
        files = get_files_from_s3(file_type)
        for file_name in files:
            print(f"Processing {file_type.upper()} file: {file_name}")
            if file_type == 'pdf':
                processed_data = process_pdf(file_name)
            elif file_type == 'xlsx':
                processed_data = process_xlsx(file_name)
            elif file_type == 'docx':
                processed_data = process_docx(file_name)

            all_processed_data = pd.concat([all_processed_data, processed_data], ignore_index=True)

    if not all_processed_data.empty:
        os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(CSV_OUTPUT_DIR, "all_documents_vector_normalized.csv")
        all_processed_data.to_csv(output_file, index=False)
        print(f"All processed data saved to {output_file}")
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()
