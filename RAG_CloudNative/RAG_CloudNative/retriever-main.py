import os
import httpx
import logging
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, AsyncGenerator

# Langchain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Milvus
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough,RunnableLambda

# Pymilvus imports
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

# --- Configuration ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL")
RERANKER_SERVICE_URL = os.getenv("RERANKER_SERVICE_URL")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
RERANKER_MODEL = os.getenv("RERANKER_MODEL_NAME")
LLM_MODEL = os.getenv("LLM_MODEL_NAME")

EMBEDDING_DIM = 1024 
COLLECTION_NAME_PREFIX = "rag_collection_"

app = FastAPI(title="RAG Retriever Service")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class CreateCollectionRequest(BaseModel):
    collection_name: str = Field(..., description="Name of the collection to create. Will be prefixed.")

class AddDocumentsRequest(BaseModel):
    collection_name: str
    documents: List[str]
    metadatas: List[dict] | None = None

class RAGRequest(BaseModel):
    collection_name: str
    query: str
    top_k: int = Field(10, description="Documents to retrieve from vector store.")
    rerank_top_n: int = Field(3, description="Documents to keep after reranking.")

# --- Startup and Shutdown Events ---
@app.on_event("startup")
def startup_event():
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    except Exception as e:
        # This is a critical failure, we can raise an exception to stop startup
        raise RuntimeError(f"Could not connect to Milvus: {e}")

@app.on_event("shutdown")
def shutdown_event():
    connections.disconnect("default")

# --- LangChain and Service Components ---

# Configure a custom httpx client with a longer timeout for embedding calls
# This is crucial for handling larger documents that result in many chunks.
embedding_client = httpx.AsyncClient(timeout=180.0) # 3-minute timeout

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_base=f"{EMBEDDING_SERVICE_URL}/v1",
    openai_api_key="dummy-key",
    # Set a long timeout for embedding many chunks from a large file.
    request_timeout=180.0,
)

llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_base=f"{LLM_SERVICE_URL}/v1",
    openai_api_key="dummy-key",
    streaming=True,
    temperature=0.1,
)


# --- start for evolution strategy ---
# --- Query Classification Function ---
async def classify_query(query: str) -> str:
    """
    使用LLM判断查询是否需要检索知识库
    返回: "RETRIEVE" 或 "ANSWER_DIRECTLY"
    """
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能路由助手。请根据以下规则对用户的问题进行分类：
**需要检索知识库的问题（RETRIEVE）**：问题涉及特定的、非公开的、专业的信息。例如：
- 关于公司内部政策、流程、项目的问题
- 关于产品的技术细节、API使用方法
- 关于某份特定文档内容的问题

**无需检索知识库的问题（ANSWER_DIRECTLY）**：问题属于通用知识、闲聊或问候。例如：
- 通用科学、历史、文化知识
- 编程语言语法等公共技术问题
- 简单的寒暄、问候、笑话

请只输出两种分类结果之一：`RETRIEVE` 或 `ANSWER_DIRECTLY`。
不要输出其他任何内容。"""),
        ("human", "{query}")
    ])
    
    classification_chain = classification_prompt | llm | StrOutputParser()
    
    try:
        result = await classification_chain.ainvoke({"query": query})
        # 清理输出，确保只包含预期的分类结果
        result = result.strip().upper()
        if "RETRIEVE" in result:
            return "RETRIEVE"
        elif "ANSWER_DIRECTLY" in result:
            return "ANSWER_DIRECTLY"
        else:
            # 如果LLM没有按预期输出，默认使用检索路径
            logger.info(f"LLM返回了意外的分类结果: '{result}'，默认使用RETRIEVE")
            return "RETRIEVE"
    except Exception as e:
        logger.info(f"查询分类失败: {e}，默认使用RETRIEVE")
        return "RETRIEVE"
# --- end for evolution strategy ---


async def rerank(query: str, documents: List[Document], top_n: int) -> List[Document]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        api_url = f"{RERANKER_SERVICE_URL}/v1/rerank"
        doc_texts = [doc.page_content for doc in documents]
        try:
            response = await client.post(api_url, json={
                "model": RERANKER_MODEL,
                "query": query,
                "documents": doc_texts,
            })
            response.raise_for_status()
            results = response.json()["results"]
            
            # Sort results by relevance score and keep top_n
            reranked_results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:top_n]
            
            # Map back to original documents
            reranked_docs = [documents[res["index"]] for res in reranked_results]
            return reranked_docs
        except httpx.HTTPError as e:
            print(f"Error calling reranker service: {e}")
            # Fallback to original documents if reranker fails
            return documents[:top_n]

# --- API Endpoints ---
@app.post("/create_collection", status_code=201)
def create_collection(request: CreateCollectionRequest):
    full_collection_name = f"{COLLECTION_NAME_PREFIX}{request.collection_name}"
    if utility.has_collection(full_collection_name):
        raise HTTPException(status_code=409, detail=f"Collection '{full_collection_name}' already exists.")
    
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    ]
    # COSINE for similarity
    schema = CollectionSchema(fields, "Document collection for RAG",metric_type="COSINE")
    Collection(name=full_collection_name, schema=schema)
    return {"message": f"Collection '{full_collection_name}' created successfully."}

@app.delete("/collections/{collection_name}", status_code=200)
def delete_collection(collection_name: str):
    """
    Deletes a collection from Milvus.
    """
    full_collection_name = f"{COLLECTION_NAME_PREFIX}{collection_name}"
    if not utility.has_collection(full_collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{full_collection_name}' not found.")
    
    try:
        utility.drop_collection(full_collection_name)
        logger.info(f"Collection '{full_collection_name}' deleted successfully.")
        return {"message": f"Collection '{full_collection_name}' deleted successfully."}
    except Exception as e:
        logger.error(f"Failed to delete collection {full_collection_name}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.post("/add_documents")
async def add_documents(request: AddDocumentsRequest):
    full_collection_name = f"{COLLECTION_NAME_PREFIX}{request.collection_name}"
    if not utility.has_collection(full_collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{full_collection_name}' not found.")
    
    try:
        vector_store = Milvus(
            embedding_function=embeddings,
            collection_name=full_collection_name,
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
            auto_id=True,
            text_field="text",
            vector_field="vector"
        )
        ids = await vector_store.aadd_texts(texts=request.documents, metadatas=request.metadatas)
        return {"message": f"Added {len(ids)} documents to '{full_collection_name}'.", "ids": ids}
    except Exception as e:
        logger.error(f"Failed to add documents to {full_collection_name}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.post("/rag_generate")
async def rag_generate(request: RAGRequest) -> StreamingResponse:
    full_collection_name = f"{COLLECTION_NAME_PREFIX}{request.collection_name}"
    if not utility.has_collection(full_collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{full_collection_name}' not found.")

    query_class = await classify_query(request.query)
    logger.info(f"查询分类结果: '{request.query}' -> {query_class}")
    # if need retrieve
    # maybe istio can show the path's modify
    if query_class == "ANSWER_DIRECTLY":
        logger.info(f"问题无需检索，直接由LLM回答: {request.query}")
        
        no_context_prompt_template = """Please answer the following question based on your knowledge.
Question:
{question}
"""
        no_context_prompt = ChatPromptTemplate.from_template(no_context_prompt_template)
        
        async def generate_direct_response() -> AsyncGenerator[str, None]:
            no_context_chain = (
                {"question": RunnablePassthrough()}
                | no_context_prompt
                | llm
                | StrOutputParser()
            )
            async for chunk in no_context_chain.astream(request.query):
                yield chunk
        
        return StreamingResponse(generate_direct_response(), media_type="text/plain")

    logger.info(f"问题需要进行检索，响应路径修改-rag")
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=full_collection_name,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        auto_id=True,
        text_field="text",
        vector_field="vector"
    )
    
    # get the retrievel-results
    raw_docs_with_scores = vector_store.similarity_search_with_score(
        request.query, k=request.top_k
    )
    
    docs = [doc for doc, score in raw_docs_with_scores]
    raw_scores = [score for doc, score in raw_docs_with_scores]
    
    scores = [1- score for score in raw_scores]
    scores = [(sim + 1) / 2 for sim in scores]
    scores = [max(0.0, min(1.0, sim)) for sim in scores]

    # def limits for selecting the path
    HIGH_SIMILARITY_THRESHOLD = 0.8
    LOW_SIMILARITY_THRESHOLD = 0.3
    
    async def determine_response_strategy():
        logger.info(f"相似度最高: {max(scores)}")
        # higher
        if scores and scores[0] >= HIGH_SIMILARITY_THRESHOLD:
            logger.info(f"使用top1直接响应，相似度: {scores[0]}")
            return docs[0:1]
        
        # lower
        elif scores and max(scores) < LOW_SIMILARITY_THRESHOLD:
            logger.info(f"所有文档相似度低，最高分: {max(scores)}，直接由LLM响应")
            return []
        
        # normal
        else:
            logger.info(f"正常处理")
            reranked_docs = await rerank(request.query, docs, request.rerank_top_n)
            return reranked_docs

    prompt_template = """Based on the following context, please answer the question.
Context:
{context}

Question:
{question}
"""
    
    no_context_prompt_template = """Please answer the following question based on your knowledge.
Question:
{question}
"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    no_context_prompt = ChatPromptTemplate.from_template(no_context_prompt_template)

    async def generate_response() -> AsyncGenerator[str, None]:
        selected_docs = await determine_response_strategy()
        
        if not selected_docs:
            no_context_chain = (
                {"question": RunnablePassthrough()}
                | no_context_prompt
                | llm
                | StrOutputParser()
            )
            async for chunk in no_context_chain.astream(request.query):
                yield chunk
                
        else:
            context = "\n\n---\n\n".join(doc.page_content for doc in selected_docs)
            rag_chain = (
                {"context": RunnableLambda(lambda x: context), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            async for chunk in rag_chain.astream(request.query):
                yield chunk

    return StreamingResponse(generate_response(), media_type="text/plain")


# @app.post("/rag_generate")
# async def rag_generate(request: RAGRequest) -> StreamingResponse:
#     logger.info("this is a test!")
#     full_collection_name = f"{COLLECTION_NAME_PREFIX}{request.collection_name}"
#     if not utility.has_collection(full_collection_name):
#         raise HTTPException(status_code=404, detail=f"Collection '{full_collection_name}' not found.")

#     vector_store = Milvus(
#         embedding_function=embeddings,
#         collection_name=full_collection_name,
#         connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
#         auto_id=True,
#         text_field="text",
#         vector_field="vector"
#     ).as_retriever(search_kwargs={"k": request.top_k})

#     prompt_template = """Based on the following context, please answer the question.
# Context:
# {context}

# Question:
# {question}
# """
#     prompt = ChatPromptTemplate.from_template(prompt_template)

#     async def format_docs_and_rerank(docs: List[Document]) -> str:
#         reranked_docs = await rerank(request.query, docs, request.rerank_top_n)
#         return "\n\n---\n\n".join(doc.page_content for doc in reranked_docs)

#     rag_chain = (
#         {"context": vector_store | format_docs_and_rerank, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
    
#     async def stream_response() -> AsyncGenerator[str, None]:
#         async for chunk in rag_chain.astream(request.query):
#             yield chunk

#     return StreamingResponse(stream_response(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)