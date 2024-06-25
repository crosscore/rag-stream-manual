# rag-streaming/s3_db/main.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/data/pdf/{filename:path}")
async def serve_pdf(filename: str, page: int = Query(None)):
    file_path = os.path.join('/app/data/pdf', filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    if page is None:
        return FileResponse(file_path, media_type='application/pdf')

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
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
