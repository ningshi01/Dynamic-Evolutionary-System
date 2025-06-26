import os
import httpx
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Query, Form
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI()

# Service URLs from environment variables, with defaults for Kubernetes service names
FILE_SERVICE_URL = os.getenv("FILE_SERVICE_URL", "http://fileservice-service")
PARSER_SERVICE_URL = os.getenv("PARSER_SERVICE_URL", "http://parser-service")
RETRIEVER_SERVICE_URL = os.getenv("RETRIEVER_SERVICE_URL", "http://retriever-service")

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG API Gateway"}

# 1. File management endpoints
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), parse: bool = Query(False, description="Set to true to automatically parse the file after upload.")):
    """
    Uploads a file and optionally triggers the parsing service.
    """
    # Forward file to fileservice
    async with httpx.AsyncClient() as client:
        files = {'file': (file.filename, await file.read(), file.content_type)}
        try:
            upload_response = await client.post(f"{FILE_SERVICE_URL}/upload", files=files)
            upload_response.raise_for_status()
            upload_result = upload_response.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Error calling file service: {exc}")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.json())

    # If parse=true, call parser service
    if parse:
        try:
            async with httpx.AsyncClient() as client:
                parse_response = await client.post(f"{PARSER_SERVICE_URL}/parse/{file.filename}")
                parse_response.raise_for_status()
                parse_result = parse_response.json()
            
            # Return combined result
            return {
                "upload_details": upload_result,
                "parsing_details": parse_result
            }
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"File uploaded, but failed to call parser service: {exc}")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.json())
    
    return upload_result

@app.get("/download/{file_name}")
async def download_file(file_name: str):
    """
    Downloads a file by streaming from the file service.
    """
    async with httpx.AsyncClient() as client:
        try:
            response_stream = await client.stream("GET", f"{FILE_SERVICE_URL}/download/{file_name}")
            response_stream.raise_for_status()
            return StreamingResponse(response_stream.aiter_bytes(), media_type=response_stream.headers.get("content-type"))
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Error calling file service: {exc}")
        except httpx.HTTPStatusError as exc:
            # Manually handle streaming response for error details
            error_details = await exc.response.aread()
            raise HTTPException(status_code=exc.response.status_code, detail=error_details.decode())


@app.delete("/delete/{file_name}")
async def delete_file(file_name: str):
    """
    Deletes a file by calling the file service.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(f"{FILE_SERVICE_URL}/delete/{file_name}")
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Error calling file service: {exc}")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.json())


# 2. File parsing endpoint
@app.post("/parse/{file_name}")
async def parse_file(file_name: str):
    """
    Triggers file parsing by calling the parser service.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{PARSER_SERVICE_URL}/parse/{file_name}")
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Error calling parser service: {exc}")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.json())

# 3. End-to-end processing endpoint
@app.post("/add_file_to_collection")
async def add_file_to_collection(
    collection_name: str = Form(...),
    create_collection_if_not_exists: bool = Form(True),
    file: UploadFile = File(...)
):
    """
    Automated pipeline: Upload -> Parse -> Store in Vector DB.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Step 0 (Optional): Create collection if it does not exist
        if create_collection_if_not_exists:
            try:
                await client.post(
                    f"{RETRIEVER_SERVICE_URL}/create_collection",
                    json={"collection_name": collection_name}
                )
            except httpx.HTTPStatusError as exc:
                # 409 Conflict means the collection already exists, which is fine.
                if exc.response.status_code != 409:
                    raise HTTPException(
                        status_code=exc.response.status_code,
                        detail=f"Error creating collection: {exc.response.text}"
                    )

        # Step 1: Upload file to MinIO via fileservice
        try:
            upload_files = {'file': (file.filename, await file.read(), file.content_type)}
            upload_response = await client.post(f"{FILE_SERVICE_URL}/upload", files=upload_files)
            upload_response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"Error calling file service: {exc}")

        # Step 2: Parse the file into chunks via parser-service
        try:
            parse_response = await client.post(f"{PARSER_SERVICE_URL}/parse/{file.filename}")
            parse_response.raise_for_status()
            chunks = parse_response.json().get("chunks", [])
            if not chunks:
                return {"message": "File uploaded but no content was parsed.", "file_name": file.filename}
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"File uploaded, but error calling parser service: {exc}")

        # Step 3: Add parsed chunks to Milvus via retriever-service
        try:
            add_docs_response = await client.post(
                f"{RETRIEVER_SERVICE_URL}/add_documents",
                json={"collection_name": collection_name, "documents": chunks}
            )
            add_docs_response.raise_for_status()
            add_docs_result = add_docs_response.json()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"File parsed, but error calling retriever service: {exc}")

    return {
        "message": "File processed and added to collection successfully.",
        "file_name": file.filename,
        "chunks_generated": len(chunks),
        "retriever_response": add_docs_result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 