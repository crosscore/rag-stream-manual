# rag-streaming/backend/main.py

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
import numpy as np
from dotenv import load_dotenv
import os
from langchain_openai import AzureOpenAIEmbeddings
from starlette.websockets import WebSocketDisconnect
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
MANUAL_DB_NAME = os.getenv("MANUAL_DB_NAME")
MANUAL_DB_USER = os.getenv("MANUAL_DB_USER")
MANUAL_DB_PASSWORD = os.getenv("MANUAL_DB_PASSWORD")
MANUAL_DB_HOST = os.getenv("MANUAL_DB_INTERNAL_HOST") if os.getenv("IS_DOCKER", "false").lower() == "true" else os.getenv("MANUAL_DB_EXTERNAL_HOST")
MANUAL_DB_PORT = os.getenv("MANUAL_DB_INTERNAL_PORT") if os.getenv("IS_DOCKER", "false").lower() == "true" else os.getenv("MANUAL_DB_EXTERNAL_PORT")
S3_DB_EXTERNAL_PORT = os.getenv("S3_DB_EXTERNAL_PORT", "9001")

logger.info(f"Database connection details: {MANUAL_DB_NAME}@{MANUAL_DB_HOST}:{MANUAL_DB_PORT}")

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
)

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    try:
        while True:
            try:
                data = await websocket.receive_json()
                logger.debug(f"Received data: {data}")
                question = data["question"]
                top_n = data.get("top_n", 3)

                logger.info(f"Processing question: '{question}' with top_n={top_n}")

                question_vector = normalize_vector(embeddings.embed_query(question))
                logger.debug("Question vector generated")

                conn = psycopg2.connect(
                    dbname=MANUAL_DB_NAME,
                    user=MANUAL_DB_USER,
                    password=MANUAL_DB_PASSWORD,
                    host=MANUAL_DB_HOST,
                    port=MANUAL_DB_PORT
                )
                logger.info("Database connection established")

                cursor = conn.cursor()

                similarity_search_query = """
                SELECT file_name, file_type, location, manual, manual_vector, (manual_vector <#> %s::vector) AS distance
                FROM manual_table
                ORDER BY distance ASC
                LIMIT %s;
                """
                logger.debug(f"Executing similarity search query with top_n={top_n}")
                cursor.execute(similarity_search_query, (question_vector.tolist(), top_n))
                results = cursor.fetchall()
                logger.info(f"Query returned {len(results)} results")

                formatted_results = []
                for result in results:
                    file_name, file_type, location, manual, _, distance = result
                    if file_type == 'PDF':
                        link = f"http://localhost:{S3_DB_EXTERNAL_PORT}/data/pdf/{file_name}?page={location}"
                        link_text = f"{file_name}, p.{location}"
                        content = None
                    else:  # XLSX or DOCX
                        link = None
                        link_text = f"{file_name}, {location}" if file_type == 'XLSX' else file_name
                        content = manual

                    formatted_results.append({
                        "file_name": file_name,
                        "file_type": file_type,
                        "location": location,
                        "manual": manual,
                        "distance": float(distance),
                        "link_text": link_text,
                        "link": link,
                        "content": content
                    })

                cursor.close()
                conn.close()
                logger.info("Database connection closed")

                logger.debug(f"Sending formatted results: {formatted_results}")
                await websocket.send_json({"results": formatted_results})

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}", exc_info=True)
                await websocket.send_json({"error": str(e)})
    finally:
        logger.info("WebSocket connection closed")

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the application")
    uvicorn.run(app, host="0.0.0.0", port=8001)
