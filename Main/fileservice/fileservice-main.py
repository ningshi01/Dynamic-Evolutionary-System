import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from minio import Minio
from minio.error import S3Error
import io

app = FastAPI()

# MinIO Client Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio-service:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET_NAME = "my-bucket"

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False  # Set to True if using https
)

@app.on_event("startup")
def startup_event():
    # Check if the bucket exists. If not, create it.
    try:
        found = minio_client.bucket_exists(BUCKET_NAME)
        if not found:
            minio_client.make_bucket(BUCKET_NAME)
            print(f"Bucket '{BUCKET_NAME}' created.")
        else:
            print(f"Bucket '{BUCKET_NAME}' already exists.")
    except S3Error as exc:
        print("Error during bucket check/creation:", exc)
        raise

@app.get("/")
def read_root():
    return {"message": "Welcome to the File Storage Service"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        file_size = len(file_content)
        minio_client.put_object(
            BUCKET_NAME,
            file.filename,
            io.BytesIO(file_content),
            length=file_size,
            content_type=file.content_type
        )
        return {"filename": file.filename, "message": "File uploaded successfully"}
    except S3Error as exc:
        raise HTTPException(status_code=500, detail=f"Error uploading file to MinIO: {exc}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{file_name}")
async def download_file(file_name: str):
    try:
        response = minio_client.get_object(BUCKET_NAME, file_name)
        return StreamingResponse(response.stream(32*1024), media_type=response.headers["Content-Type"])
    except S3Error as exc:
        if exc.code == "NoSuchKey":
            raise HTTPException(status_code=404, detail="File not found")
        else:
            raise HTTPException(status_code=500, detail=f"Error downloading file from MinIO: {exc}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete/{file_name}")
async def delete_file(file_name: str):
    try:
        minio_client.remove_object(BUCKET_NAME, file_name)
        return {"filename": file_name, "message": "File deleted successfully"}
    except S3Error as exc:
        raise HTTPException(status_code=500, detail=f"Error deleting file from MinIO: {exc}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 