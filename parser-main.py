import os
import io
from fastapi import FastAPI, HTTPException
from minio import Minio
from minio.error import S3Error
from transformers import AutoTokenizer

app = FastAPI()

# --- Configuration ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio-service:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET_NAME = "my-bucket"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_MAX_LENGTH = 512

# --- Global Clients ---
minio_client = None
tokenizer = None

@app.on_event("startup")
def startup_event():
    """
    Initialize connections and models on startup.
    """
    global minio_client, tokenizer
    # Initialize MinIO Client
    try:
        minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        print("Successfully initialized MinIO client.")
    except Exception as e:
        print(f"Error initializing MinIO client: {e}")
        # We don't exit here, but endpoints will fail if minio_client is None

    # Initialize Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        print(f"Successfully loaded tokenizer for '{EMBEDDING_MODEL_NAME}'.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        # We don't exit, but parsing will fail if tokenizer is None


def chunk_text_by_tokens(text: str, chunk_size: int = MODEL_MAX_LENGTH, chunk_overlap: int = 50):
    """
    Splits the text into overlapping chunks based on token count.
    """
    if not isinstance(text, str) or not tokenizer:
        return []

    # First, split the text into tokens
    tokens = tokenizer.tokenize(text)
    
    chunks = []
    start_token = 0
    while start_token < len(tokens):
        end_token = start_token + chunk_size
        # Get the sub-list of tokens for the current chunk
        chunk_tokens = tokens[start_token:end_token]
        # Convert the chunk tokens back to a string
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move the start token for the next chunk
        if end_token >= len(tokens):
            break
        start_token += chunk_size - chunk_overlap
        
    return chunks


@app.get("/")
def read_root():
    return {"message": "Welcome to the Document Parsing Service"}

@app.post("/parse/{file_name}")
async def parse_file(file_name: str):
    if not minio_client:
        raise HTTPException(status_code=500, detail="MinIO client not initialized")
    if not tokenizer:
        raise HTTPException(status_code=500, detail="Tokenizer not initialized")

    try:
        # Download file from MinIO
        response = minio_client.get_object(BUCKET_NAME, file_name)
        file_content = response.read().decode('utf-8')
        response.close()
        response.release_conn()

        # Chunk the text content using the tokenizer
        chunks = chunk_text_by_tokens(file_content)
        
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