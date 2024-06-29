# rag-streaming/s3_db/main.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from pypdf import PdfReader, PdfWriter
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/data/{file_type}")
async def list_files(file_type: str):
    directory = f"/app/data/{file_type}"
    if not os.path.exists(directory):
        return []  # Return an empty list if the directory doesn't exist
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return files

@app.get("/data/{file_type}/{filename:path}")
async def serve_file(file_type: str, filename: str, page: int = Query(None)):
    file_path = f"/app/data/{file_type}/{filename}"
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    if file_type.lower() == 'pdf' and page is not None:
        return stream_pdf_page(file_path, page)

    return FileResponse(file_path)

def stream_pdf_page(file_path: str, page: int):
    try:
        reader = PdfReader(file_path)
        writer = PdfWriter()

        if 0 <= page - 1 < len(reader.pages):
            writer.add_page(reader.pages[page - 1])
        else:
            raise HTTPException(status_code=404, detail="Page not found")

        output = BytesIO()
        writer.write(output)
        output.seek(0)

        return StreamingResponse(output, media_type='application/pdf')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
