import os
import httpx
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict

# --- Configuration ---
# Service URLs from environment variables, with defaults for Kubernetes service names
FILE_SERVICE_URL = os.getenv("FILE_SERVICE_URL", "http://fileservice-service")
PARSER_SERVICE_URL = os.getenv("PARSER_SERVICE_URL", "http://parser-service")
RETRIEVER_SERVICE_URL = os.getenv("RETRIEVER_SERVICE_URL", "http://retriever-service")

app = FastAPI(title="RAG API Gateway")

# Use a single, reusable httpx.AsyncClient for performance
async_client = httpx.AsyncClient(timeout=300.0)

# --- Pydantic Models ---

class CollectionRequest(BaseModel):
    collection_name: str

class ChatCompletionRequest(BaseModel):
    model: str  # We'll repurpose this to mean "collection_name"
    messages: List[Dict[str, str]]
    stream: bool = False
    
    @property
    def query(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

# --- API Endpoints ---

@app.post("/collections", status_code=201, summary="Create Knowledge Base")
async def create_collection(request: CollectionRequest):
    try:
        response = await async_client.post(
            f"{RETRIEVER_SERVICE_URL}/create_collection",
            json={"collection_name": request.collection_name}
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.delete("/collections/{collection_name}", status_code=200, summary="Delete Knowledge Base")
async def delete_collection(collection_name: str):
    try:
        response = await async_client.delete(f"{RETRIEVER_SERVICE_URL}/collections/{collection_name}")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.post("/add_file_to_collection", summary="Add File to Knowledge Base")
async def add_file_to_collection(
    file: UploadFile = File(...), 
    collection_name: str = Form(...),
    create_collection_if_not_exists: bool = Form(True)
):
    """
    Automated pipeline: Upload -> Parse -> Store in Vector DB.
    """
    # Step 0 (Optional): Create collection if it does not exist
    if create_collection_if_not_exists:
        try:
            create_req = await async_client.post(
                f"{RETRIEVER_SERVICE_URL}/create_collection",
                json={"collection_name": collection_name}
            )
            if create_req.status_code not in [201, 409]:
                create_req.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 409:
                raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An internal error occurred while checking collection: {str(e)}")

    # Step 1: Upload file to MinIO via fileservice
    try:
        upload_files = {'file': (file.filename, file.file, file.content_type)}
        upload_response = await async_client.post(f"{FILE_SERVICE_URL}/upload", files=upload_files)
        upload_response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file to file service: {str(e)}")

    # Step 2: Parse the file into chunks via parser-service
    try:
        parse_response = await async_client.post(f"{PARSER_SERVICE_URL}/parse/{file.filename}")
        parse_response.raise_for_status()
        chunks = parse_response.json().get("chunks", [])
        if not chunks:
            return {"message": "File uploaded but no content was parsed.", "file_name": file.filename}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"File uploaded, but error calling parser service: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred while calling parser service: {str(e)}")

    # Step 3: Add parsed chunks to Milvus via retriever-service
    try:
        add_docs_payload = {"collection_name": collection_name, "documents": chunks}
        add_docs_response = await async_client.post(
            f"{RETRIEVER_SERVICE_URL}/add_documents", json=add_docs_payload
        )
        add_docs_response.raise_for_status()
        add_docs_result = add_docs_response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"File parsed, but error calling retriever service: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred while adding documents to retriever: {str(e)}")

    return {
        "message": "File processed and added to collection successfully.",
        "file_name": file.filename,
        "chunks_generated": len(chunks),
        "retriever_response": add_docs_result
    }

@app.post("/chat/completions", summary="Ask a Question (RAG)")
async def chat_completions(request: ChatCompletionRequest):
    """
    Provides RAG-based chat completions, compatible with OpenAI's format.
    The 'model' field is repurposed to specify the collection_name.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="No user message found in request.")

    rag_payload = {"collection_name": request.model, "query": request.query}
    
    async def stream_generator():
        try:
            async with async_client.stream("POST", f"{RETRIEVER_SERVICE_URL}/rag_generate", json=rag_payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    if not request.stream:
                        yield chunk
                        continue
                    
                    data = {"choices": [{"delta": {"content": chunk}}]}
                    yield f"data: {json.dumps(data)}\n\n"
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            yield f"data: {json.dumps({'error': error_detail})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        if request.stream:
            yield "data: [DONE]\n\n"

    if request.stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        full_response = "".join([chunk async for chunk in stream_generator()])
        return {"choices": [{"message": {"role": "assistant", "content": full_response}}]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 