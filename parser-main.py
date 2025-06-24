import os
import io
from fastapi import FastAPI, HTTPException
from minio import Minio
from minio.error import S3Error

app = FastAPI()

# MinIO Client Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio-service:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET_NAME = "my-bucket"

# Initialize MinIO Client
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
except Exception as e:
    print(f"Error initializing MinIO client: {e}")
    minio_client = None

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Splits the text into overlapping chunks.
    """
    if not isinstance(text, str):
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

@app.get("/")
def read_root():
    return {"message": "Welcome to the Document Parsing Service"}

@app.post("/parse/{file_name}")
async def parse_file(file_name: str):
    if not minio_client:
        raise HTTPException(status_code=500, detail="MinIO client not initialized")

    try:
        # Download file from MinIO
        response = minio_client.get_object(BUCKET_NAME, file_name)
        file_content = response.read().decode('utf-8')
        response.close()
        response.release_conn()

        # Chunk the text content
        chunks = chunk_text(file_content)
        
        return {
            "file_name": file_name,
            "chunks": chunks
        }

    except S3Error as exc:
        if exc.code == "NoSuchKey":                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
            raise HTTPException(status_code=404, detail=f"File '{file_name}' not found in bucket '{BUCKET_NAME}'")
        else:
            raise HTTPException(status_code=500, detail=f"MinIO S3 Error: {exc}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 