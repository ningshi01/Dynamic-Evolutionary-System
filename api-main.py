import os
import httpx
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI()

# Service URLs from environment variables, with defaults for Kubernetes service names
FILE_SERVICE_URL = os.getenv("FILE_SERVICE_URL", "http://fileservice-service")
PARSER_SERVICE_URL = os.getenv("PARSER_SERVICE_URL", "http://parser-service")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 