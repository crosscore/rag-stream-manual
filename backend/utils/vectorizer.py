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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 30))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 0))
SEPARATOR = os.getenv("SEPARATOR", "\n\n")
CSV_OUTPUT_DIR = os.getenv("CSV_OUTPUT_DIR", "../data/csv/all")

is_docker = os.getenv("IS_DOCKER", "false").lower() == "true"
S3_DB_URL = os.getenv("S3_DB_INTERNAL_URL" if is_docker else "S3_DB_EXTERNAL_URL", "http://localhost:9001")

logger.info(f"Using S3_DB_URL: {S3_DB_URL}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

def get_files_from_s3(file_type):
    try:
        response = requests.get(f"{S3_DB_URL}/data/{file_type}", timeout=10)
        response.raise_for_status()
        files = response.json()
        if not isinstance(files, list):
            logger.warning(f"Unexpected response format for {file_type.upper()} files. Expected list, got: {type(files)}")
            return []
        valid_files = [file for file in files if file.endswith(f'.{file_type}')]
        logger.info(f"Found {len(valid_files)} {file_type.upper()} files: {valid_files}")
        return valid_files
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching {file_type.upper()} files: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching {file_type.upper()} files: {str(e)}")
        return []

def fetch_file_content(file_type, file_name):
    try:
        response = requests.get(f"{S3_DB_URL}/data/{file_type}/{file_name}", stream=True, timeout=10)
        response.raise_for_status()
        logger.info(f"Successfully fetched {file_type.upper()} file: {file_name}")
        return response.content
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching {file_type.upper()} file {file_name}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {file_type.upper()} file {file_name}: {str(e)}")
        return None

def extract_text_pdf(content):
    pdf_reader = PdfReader(BytesIO(content))
    return [page.extract_text() for page in pdf_reader.pages]

def extract_text_xlsx(content):
    workbook = load_workbook(filename=BytesIO(content), read_only=True)
    return ["\n".join([" ".join([str(cell.value) for cell in row if cell.value is not None]) for row in sheet.iter_rows()]) for sheet in workbook]

def extract_text_docx(content):
    doc = Document(BytesIO(content))
    return ["\n".join([para.text for para in doc.paragraphs])]

def process_file(file_type, file_name):
    content = fetch_file_content(file_type, file_name)
    if content is None:
        return pd.DataFrame()

    text_extractor = {
        'pdf': extract_text_pdf,
        'xlsx': extract_text_xlsx,
        'docx': extract_text_docx
    }[file_type]

    texts = text_extractor(content)

    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator=SEPARATOR
    )

    processed_data = []
    for i, text in enumerate(texts, start=1):
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            vector = embeddings.embed_query(chunk)
            normalized_vector = normalize_vector(vector).tolist()
            processed_data.append({
                'file_name': file_name,
                'file_type': file_type.upper(),
                'location': str(i),
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
