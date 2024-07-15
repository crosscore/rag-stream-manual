# rag-streaming/backend/main.py
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import numpy as np
from dotenv import load_dotenv
import os
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from starlette.websockets import WebSocketDisconnect
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MANUAL_DB_NAME = os.getenv("MANUAL_DB_NAME")
MANUAL_DB_USER = os.getenv("MANUAL_DB_USER")
MANUAL_DB_PASSWORD = os.getenv("MANUAL_DB_PASSWORD")
MANUAL_DB_HOST = os.getenv("MANUAL_DB_INTERNAL_HOST") if os.getenv("IS_DOCKER", "false").lower() == "true" else os.getenv("MANUAL_DB_EXTERNAL_HOST")
MANUAL_DB_PORT = os.getenv("MANUAL_DB_INTERNAL_PORT") if os.getenv("IS_DOCKER", "false").lower() == "true" else os.getenv("MANUAL_DB_EXTERNAL_PORT")
S3_DB_EXTERNAL_PORT = os.getenv("S3_DB_EXTERNAL_PORT", "9001")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 1.0))
INDEX_TYPE = os.getenv("INDEX_TYPE", "ivfflat").lower()
VECTOR_DIMENSIONS = 3072

logger.info(f"Application initialized with SCORE_THRESHOLD: {SCORE_THRESHOLD} and INDEX_TYPE: {INDEX_TYPE}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector

def get_search_query(index_type):
    if index_type == "hnsw":
        return f"""
        SELECT file_name, file_type, location, manual,
                (manual_vector::halfvec({VECTOR_DIMENSIONS}) <#> %s::halfvec({VECTOR_DIMENSIONS})) AS distance
        FROM manual_table
        ORDER BY manual_vector::halfvec({VECTOR_DIMENSIONS}) <#> %s::halfvec({VECTOR_DIMENSIONS})
        LIMIT %s;
        """
    elif index_type == "ivfflat":
        return f"""
        SELECT file_name, file_type, location, manual,
                (manual_vector::halfvec({VECTOR_DIMENSIONS}) <#> %s::halfvec({VECTOR_DIMENSIONS})) AS distance
        FROM manual_table
        ORDER BY manual_vector::halfvec({VECTOR_DIMENSIONS}) <#> %s::halfvec({VECTOR_DIMENSIONS})
        LIMIT %s;
        """
    else:  # "none" or any other value
        return f"""
        SELECT file_name, file_type, location, manual,
                (manual_vector <#> %s::vector({VECTOR_DIMENSIONS})) AS distance
        FROM manual_table
        WHERE (manual_vector <#> %s::vector({VECTOR_DIMENSIONS})) <= %s
        ORDER BY distance
        LIMIT %s;
        """

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("New WebSocket connection established")
    try:
        while True:
            try:
                data = await websocket.receive_json()
                question = data["question"]
                top_n = data.get("top_n", 3)
                score_threshold = data.get("score_threshold", SCORE_THRESHOLD)

                logger.info(f"Processing question with top_n={top_n} and score_threshold={score_threshold}")

                question_vector = normalize_vector(embeddings.embed_query(question))

                with psycopg2.connect(
                    dbname=MANUAL_DB_NAME,
                    user=MANUAL_DB_USER,
                    password=MANUAL_DB_PASSWORD,
                    host=MANUAL_DB_HOST,
                    port=MANUAL_DB_PORT
                ) as conn:
                    with conn.cursor() as cursor:
                        search_query = get_search_query(INDEX_TYPE)
                        if INDEX_TYPE in ["hnsw", "ivfflat"]:
                            cursor.execute(search_query, (question_vector.tolist(), question_vector.tolist(), top_n))
                        else:  # "none" or any other value
                            cursor.execute(search_query, (question_vector.tolist(), question_vector.tolist(), score_threshold, top_n))
                        results = cursor.fetchall()

                logger.info(f"Query returned {len(results)} results")

                formatted_results = []
                for result in results:
                    file_name, file_type, location, manual, distance = result
                    formatted_result = {
                        "file_name": file_name,
                        "file_type": file_type,
                        "location": location,
                        "manual": manual,
                        "distance": float(distance),
                        "link_text": f"{file_name}, p.{location}" if file_type == 'PDF' else f"{file_name}, {location}" if file_type == 'XLSX' else file_name,
                        "link": f"http://localhost:{S3_DB_EXTERNAL_PORT}/data/pdf/{file_name}?page={location}" if file_type == 'PDF' else None,
                        "content": None if file_type == 'PDF' else manual
                    }
                    formatted_results.append(formatted_result)

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
