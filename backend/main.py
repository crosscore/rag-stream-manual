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

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
MANUAL_DB_NAME = os.getenv("MANUAL_DB_NAME")
MANUAL_DB_USER = os.getenv("MANUAL_DB_USER")
MANUAL_DB_PASSWORD = os.getenv("MANUAL_DB_PASSWORD")

is_docker = os.getenv("IS_DOCKER", "false").lower() == "true"

if is_docker:
    MANUAL_DB_HOST = os.getenv("MANUAL_DB_INTERNAL_HOST")
    MANUAL_DB_PORT = os.getenv("MANUAL_DB_INTERNAL_PORT")
    S3_DB_URL = os.getenv("S3_DB_INTERNAL_URL")
else:
    MANUAL_DB_HOST = os.getenv("MANUAL_DB_EXTERNAL_HOST", "localhost")
    MANUAL_DB_PORT = os.getenv("MANUAL_DB_EXTERNAL_PORT")
    S3_DB_URL = os.getenv("S3_DB_EXTERNAL_URL")

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
    try:
        while True:
            try:
                data = await websocket.receive_json()
                question = data["question"]
                top_n = data.get("top_n", 3)

                question_vector = normalize_vector(embeddings.embed_query(question))

                conn = psycopg2.connect(
                    dbname=MANUAL_DB_NAME,
                    user=MANUAL_DB_USER,
                    password=MANUAL_DB_PASSWORD,
                    host=MANUAL_DB_HOST,
                    port=MANUAL_DB_PORT
                )
                cursor = conn.cursor()

                similarity_search_query = """
                SELECT file_name, page, manual, manual_vector, (manual_vector <#> %s::vector) AS distance
                FROM manual_table
                ORDER BY distance ASC
                LIMIT %s;
                """
                cursor.execute(similarity_search_query, (question_vector.tolist(), top_n))
                results = cursor.fetchall()

                formatted_results = [
                    {
                        "file_name": result[0],
                        "page": result[1],
                        "manual": result[2],
                        "distance": float(result[4]),
                        "link_text": f"{result[0]}, p.{result[1]}",
                        "link": f"http://localhost:{os.getenv('S3_DB_EXTERNAL_PORT', '9001')}/data/pdf/{result[0]}?page={result[1]}"
                    }
                    for result in results
                ]

                cursor.close()
                conn.close()

                print(f"Sending data: {formatted_results}")  # DEBUG
                await websocket.send_json({"results": formatted_results})

            except WebSocketDisconnect:
                print("WebSocket disconnected")
                break
            except Exception as e:
                print(f"Error processing message: {str(e)}")
                await websocket.send_json({"error": str(e)})
    finally:
        print("WebSocket connection closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
