import os
import requests
import pandas as pd
import numpy as np
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document
from openpyxl import load_workbook
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHUNK_SIZE = 300
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 0))
SEPARATOR = "\n\n"
CSV_OUTPUT_DIR = os.getenv("CSV_OUTPUT_DIR", "../data/csv/all")

is_docker = os.getenv("IS_DOCKER", "false").lower() == "true"
if is_docker:
    S3_DB_URL = os.getenv("S3_DB_INTERNAL_URL", "http://s3_db:9000")
else:
    S3_DB_URL = os.getenv("S3_DB_EXTERNAL_URL", "http://localhost:9001")

logger.info(f"Using S3_DB_URL: {S3_DB_URL}")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

def get_files_from_s3(file_type):
    try:
        response = requests.get(f"{S3_DB_URL}/data/{file_type}", timeout=10)
        response.raise_for_status()
        logger.info(f"Successfully fetched {file_type.upper()} file list. Status code: {response.status_code}")
        files = response.json()
        if not isinstance(files, list):
            logger.warning(f"Unexpected response format for {file_type.upper()} files. Expected list, got: {type(files)}")
            return []
        valid_files = [file for file in files if file.endswith(f'.{file_type}')]
        logger.info(f"Found {len(valid_files)} {file_type.upper()} files: {valid_files}")
        return valid_files
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching {file_type.upper()} files: {str(e)}")
        logger.error(f"Response content: {e.response.content if e.response else 'No response'}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching {file_type.upper()} files: {str(e)}")
        return []

def process_file(file_type, file_name):
    try:
        response = requests.get(f"{S3_DB_URL}/data/{file_type}/{file_name}", stream=True, timeout=10)
        response.raise_for_status()
        logger.info(f"Successfully fetched {file_type.upper()} file: {file_name}")

        if file_type == 'pdf':
            return process_pdf_content(file_name, response.content)
        elif file_type == 'xlsx':
            return process_xlsx_content(file_name, response.content)
        elif file_type == 'docx':
            return process_docx_content(file_name, response.content)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error processing {file_type.upper()} file {file_name}: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error processing {file_type.upper()} file {file_name}: {str(e)}")
        return pd.DataFrame()

def process_pdf_content(file_name, content):
    pdf_reader = PdfReader(BytesIO(content))
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

def process_xlsx_content(file_name, content):
    workbook = load_workbook(filename=BytesIO(content), read_only=True)
    processed_data = []

    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator=SEPARATOR
    )

    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        text = "\n".join([" ".join([str(cell.value) for cell in row if cell.value is not None]) for row in sheet.iter_rows()])
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

def process_docx_content(file_name, content):
    doc = Document(BytesIO(content))
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
        if not files:
            logger.info(f"No {file_type.upper()} files found or error occurred while fetching file list.")
            continue
        for file_name in files:
            logger.info(f"Processing {file_type.upper()} file: {file_name}")
            processed_data = process_file(file_type, file_name)
            all_processed_data = pd.concat([all_processed_data, processed_data], ignore_index=True)

    if not all_processed_data.empty:
        os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(CSV_OUTPUT_DIR, "all_documents_vector_normalized.csv")
        all_processed_data.to_csv(output_file, index=False)
        logger.info(f"All processed data saved to {output_file}")
    else:
        logger.warning("No data processed.")

if __name__ == "__main__":
    main()
