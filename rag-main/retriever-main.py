import os
import httpx
import logging
import time
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, AsyncGenerator, Optional
import asyncio
import json
import re

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
LLM_MODEL_BIG = os.getenv("LLM_MODEL_BIG_NAME")

EMBEDDING_DIM = 1024 
COLLECTION_NAME_PREFIX = "rag_collection_"

CACHE_COLLECTION_NAME = "semantic_cache"  # 语义缓存集合名称
CACHE_SIMILARITY_THRESHOLD = 0.90  # 语义相似度阈值
MAX_CACHE_ENTRIES = 10000  # 最大缓存条目数

# ---- LLM 服务 ----
LLM_SERVICE_URLS = os.getenv("LLM_SERVICE_URLS")
LLM_MODEL_NAMES = os.getenv("LLM_MODEL_NAMES")
LLM_SERVICE_CAPACITIES = os.getenv("LLM_SERVICE_CAPACITIES")

# Fallback to single-service env vars for backwards compatibility
PRIMARY_LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL")
PRIMARY_LLM_MODEL = os.getenv("LLM_MODEL_NAME")

# -------------------------------------------------

app = FastAPI(title="RAG Retriever Service")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for Milvus vector_store instances to avoid re-creating per request
vector_store_cache: Dict[str, Milvus] = {}

def get_vector_store(collection_name: str) -> Milvus:
    """Return a cached Milvus vector store instance for the given collection."""
    if collection_name in vector_store_cache:
        return vector_store_cache[collection_name]
    vs = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        auto_id=True,
        text_field="text",
        vector_field="vector"
    )
    vector_store_cache[collection_name] = vs
    return vs


def normalize_milvus_score(score: float, metric: str = "COSINE") -> float:
    """Normalize Milvus search score to a 0..1 similarity value.

    For COSINE metric, Milvus often returns distance-like scores depending on index/params.
    This helper tries common conversions:
    - If score seems in [-1,1], treat it as cosine similarity and clip.
    - If score > 1 (distance), convert with 1 - score if it's in [0,1], or a sigmoid-like mapping.
    - Fallback: clamp to [0,1].
    """
    try:
        s = float(score)
    except Exception:
        return 0.0

    metric_up = (metric or "").upper()
    if metric_up == "COSINE":
        # Common case: higher is more similar and within [-1,1]
        if -1.0 <= s <= 1.0:
            return max(0.0, min(1.0, s))
        if s >= 0.0:
            return max(0.0, min(1.0, 1.0 - s))

    if metric_up in ("L2", "EUCLIDEAN"):
        # L2 is a distance: lower is better. Use 1/(1+dist) mapping to (0,1]
        if s <= 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 / (1.0 + s)))

    if metric_up in ("IP", "INNER_PRODUCT"):
        # Inner product: higher is better. Normalize via simple ratio if large, else clamp
        try:
            if 0.0 <= s <= 1.0:
                return s
            return max(0.0, min(1.0, s / (1.0 + abs(s))))
        except Exception:
            return max(0.0, min(1.0, 1.0 / (1.0 + abs(s))))
    if s < 0:
        s = abs(s)
    return max(0.0, min(1.0, 1.0 / (1.0 + s)))


async def similarity_search_collection(collection_name: str, query: str, top_k: int = 10, metric: str = "COSINE") -> List[tuple]:
    col = None
    try:
        col = Collection(collection_name)
        try:
            col.load()
        except Exception:
            pass
        query_emb = await embeddings.aembed_query(query)

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        try:
            results = col.search(
                data=[query_emb],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["text"]
            )
        except Exception as search_e:
            msg = str(search_e)
            logger.warning(f"Collection search failed, will attempt metric-aware retries: {msg}")
            tried_metrics = [metric]
            if "expected=L2" in msg or "expected: L2" in msg:
                alt_metrics = ["L2", "IP", "COSINE"]
            else:
                alt_metrics = ["COSINE", "L2", "IP"]

            results = None
            for alt in alt_metrics:
                if alt in tried_metrics:
                    continue
                tried_metrics.append(alt)
                try:
                    search_params_alt = {"metric_type": alt, "params": {"nprobe": 10}}
                    results = col.search(
                        data=[query_emb],
                        anns_field="vector",
                        param=search_params_alt,
                        limit=top_k,
                        output_fields=["text"]
                    )
                    metric = alt
                    break
                except Exception:
                    continue

        hits = []
        if results and len(results) > 0:
            for hit in results[0]:
                try:
                    text = None
                    if hasattr(hit, 'entity') and isinstance(hit.entity, dict):
                        text = hit.entity.get('text')
                    else:
                        try:
                            text = hit.entity.get('text')
                        except Exception:
                            text = None

                    page_content = text if text is not None else ""
                    doc = Document(page_content=page_content)
                    sim = normalize_milvus_score(hit.score, metric=metric)
                    hits.append((doc, sim))
                except Exception:
                    continue

        if not hits:
            try:
                vs = get_vector_store(collection_name)
                raw = await asyncio.to_thread(lambda: vs.similarity_search_with_score(query, k=top_k))
                fallback_hits = []
                for doc, raw_score in raw:
                    try:
                        sim = normalize_milvus_score(raw_score, metric=metric)
                        fallback_hits.append((doc, sim))
                    except Exception:
                        continue
                if fallback_hits:
                    return fallback_hits
            except Exception:
                logger.exception("fallback similarity_search_with_score failed")

        return hits

    except Exception as e:
        logger.exception(f"similarity_search_collection failed: {e}")
        return []

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

# --- Semantic Cache Functions ---
def init_semantic_cache():
    """初始化语义缓存集合"""
    # Robust init: handle transient Milvus unavailability and different pymilvus versions
    retry_attempts = 3
    for attempt in range(1, retry_attempts + 1):
        try:
            if not utility.has_collection(CACHE_COLLECTION_NAME):
                logger.info("Creating semantic cache collection...")
                fields = [
                    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="query", dtype=DataType.VARCHAR, max_length=4096),
                    FieldSchema(name="response", dtype=DataType.VARCHAR, max_length=8192),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                    FieldSchema(name="timestamp", dtype=DataType.DOUBLE),
                    FieldSchema(name="access_count", dtype=DataType.INT64),
                    FieldSchema(name="last_accessed", dtype=DataType.DOUBLE)
                ]
                schema = CollectionSchema(fields, "Semantic cache for RAG")
                cache_col = Collection(name=CACHE_COLLECTION_NAME, schema=schema)

                # Create index for the embedding vector field to avoid 'index not found' on load
                try:
                    index_params = {
                        "index_type": "HNSW",
                        "metric_type": "COSINE",
                        "params": {"M": 16, "efConstruction": 200}
                    }
                    if hasattr(cache_col, "create_index"):
                        cache_col.create_index(field_name="embedding", index_params=index_params)
                    else:
                        utility.create_index(CACHE_COLLECTION_NAME, "embedding", index_params)

                    cache_col.flush()
                    cache_col.load()
                    logger.info("Semantic cache collection created and index built successfully.")
                except Exception:
                    logger.exception("Failed to create index for semantic cache collection; collection created without index.")
            else:
                logger.info("Semantic cache collection already exists.")
                try:
                    existing_col = Collection(CACHE_COLLECTION_NAME)
                    existing_col.load()
                    logger.info("Semantic cache collection loaded into memory.")
                except Exception:
                    logger.exception("Failed to load existing semantic cache collection at init.")

            # success
            return

        except Exception as e:
            logger.warning(f"init_semantic_cache attempt {attempt} failed: {e}")
            if attempt < retry_attempts:
                time.sleep(1.0)
                continue
            else:
                logger.exception("init_semantic_cache failed after retries; giving up for now.")
                return

async def search_semantic_cache(query: str) -> Optional[dict]:
    """在语义缓存中搜索相似查询"""
    try:
        # Ensure cache exists (will try to init if missing)
        if not utility.has_collection(CACHE_COLLECTION_NAME):
            logger.info("Semantic cache missing; attempting to initialize before search.")
            init_semantic_cache()

        cache_collection = Collection(CACHE_COLLECTION_NAME)
        
        query_embedding = await embeddings.aembed_query(query)
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = cache_collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=1,
            output_fields=["pk", "query", "response", "timestamp", "access_count"]
        )
        
        if results and results[0]:
            hit = results[0][0]
            # hit.score 的含义依 metric_type 而定；这里保守使用阈值判断
            if hit.score >= CACHE_SIMILARITY_THRESHOLD:
                current_time = time.time()
                # 使用主键删除旧条目（pk 为 INT64 auto_id 产生）
                pk_val = hit.entity.get("pk")
                if pk_val is not None:
                    cache_collection.delete(expr=f"pk in [{pk_val}]")
                updated_entry = {
                    "query": hit.entity.get("query"),
                    "response": hit.entity.get("response"),
                    "embedding": query_embedding,
                    "timestamp": hit.entity.get("timestamp") or current_time,
                    "access_count": (hit.entity.get("access_count") or 0) + 1,
                    "last_accessed": current_time
                }
                # 注意：当 pk 为 auto_id=True 时，插入时不要包含 pk 字段
                cache_collection.insert([updated_entry])
                cache_collection.flush()
                
                logger.info(f"Cache hit with similarity score: {hit.score:.4f}")
                return {
                    "query": hit.entity.get("query"),
                    "response": hit.entity.get("response"),
                    "timestamp": hit.entity.get("timestamp")
                }
        return None
    except Exception as e:
        logger.exception(f"Error searching semantic cache: {e}")
        return None

async def add_to_semantic_cache(query: str, response: str):
    """添加新的查询-响应对到语义缓存"""
    try:
        # Ensure cache exists before inserting
        if not utility.has_collection(CACHE_COLLECTION_NAME):
            logger.info("Semantic cache missing; attempting to initialize before insert.")
            init_semantic_cache()

        cache_collection = Collection(CACHE_COLLECTION_NAME)
        
        query_embedding = await embeddings.aembed_query(query)
        current_time = time.time()
        
        # 当 pk 为 auto_id=True 时，不要传 pk 字段
        cache_entry = {
            "query": query,
            "response": response,
            "embedding": query_embedding,
            "timestamp": current_time,
            "access_count": 1,
            "last_accessed": current_time
        }
        
        cache_collection.insert([cache_entry])
        cache_collection.flush()
        
        logger.info(f"Added new entry to semantic cache: {query[:50]}...")
    except Exception as e:
        logger.exception(f"Error adding to semantic cache: {e}")

@app.delete("/cache/clear")
async def clear_semantic_cache():
    """清空语义缓存"""
    try:
        if utility.has_collection(CACHE_COLLECTION_NAME):
            utility.drop_collection(CACHE_COLLECTION_NAME)
            logger.info("Semantic cache cleared successfully.")
            return {"message": "Semantic cache cleared successfully."}
        else:
            raise HTTPException(status_code=404, detail="Semantic cache collection not found.")
    except Exception as e:
        logger.error(f"Failed to clear semantic cache: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.get("/cache/stats")
async def get_cache_stats():
    """获取语义缓存统计信息"""
    try:
        if not utility.has_collection(CACHE_COLLECTION_NAME):
            return {"message": "Semantic cache collection does not exist", "count": 0}
        
        cache_collection = Collection(CACHE_COLLECTION_NAME)
        cache_collection.load()
        count = cache_collection.num_entities
        
        return {
            "cache_collection": CACHE_COLLECTION_NAME,
            "total_entries": count,
            "max_entries": MAX_CACHE_ENTRIES,
            "similarity_threshold": CACHE_SIMILARITY_THRESHOLD
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# --- Startup and Shutdown Events ---
@app.on_event("startup")
def startup_event():
    # Attempt to connect to Milvus; allow retries to handle startup ordering in containers
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
            logger.info(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
            break
        except Exception as conn_e:
            logger.warning(f"Attempt {attempt} failed to connect to Milvus: {conn_e}")
            if attempt == max_retries:
                logger.exception("Could not establish connection to Milvus after retries; continuing without cache initialization.")
            else:
                time.sleep(1.0)

    # Try to initialize semantic cache but do not crash the whole app if Milvus is unavailable
    try:
        init_semantic_cache()
    except Exception as e:
        logger.exception(f"init_semantic_cache failed during startup: {e}")

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
    default_headers={"X-Use-Cache": "true"},
)

llm_big = ChatOpenAI(
    model=LLM_MODEL_BIG,
    openai_api_base=f"{LLM_SERVICE_URL}/v1",
    openai_api_key="dummy-key",
    streaming=True,
    temperature=0.1,
)

# 添加查询分析类
class QueryComplexityAnalyzer:
    def __init__(self):
        # 复杂查询关键词
        self.complex_keywords = [
            '解释', '分析', '比较', '为什么', '如何实现', '原理', '机制',
            '优化', '架构', '设计模式', '算法', '数据结构', '分布式', '性能'
        ]
        # 技术领域关键词  
        self.technical_keywords = [
            '代码', '编程', 'API', '部署', '配置', 'Docker', 'Kubernetes',
            '数据库', '网络', '安全', '容器', '集群', '微服务'
        ]
    
    def analyze_query(self, query: str) -> dict:
        """分析查询复杂度并返回路由头信息"""
        query_lower = query.lower()
        
        # 计算复杂度分数
        complexity_score = 0
        
        # 检查复杂关键词
        if any(keyword in query_lower for keyword in self.complex_keywords):
            complexity_score += 2
            
        # 检查技术关键词
        if any(keyword in query_lower for keyword in self.technical_keywords):
            complexity_score += 1
            
        # 查询长度因素
        if len(query) > 50:
            complexity_score += 1
            
        # 问题类型判断
        if '?' in query and '吗' not in query:  # 非简单疑问句
            complexity_score += 1
        
        # 确定复杂度等级
        if complexity_score >= 3:
            return {
                "x-query-complexity": "high",
                "x-query-type": "technical"
            }
        elif complexity_score >= 2:
            return {
                "x-query-complexity": "medium",
                "x-query-type": "technical" 
            }
        else:
            return {
                "x-query-complexity": "low", 
                "x-query-type": "general"
            }

# 初始化分析器
query_analyzer = QueryComplexityAnalyzer()



# --- start for evolution strategy ---
# --- Query Classification Function ---
async def classify_query(query: str) -> str:
    """
    使用LLM判断查询是否需要检索知识库
    返回: "RETRIEVE" 或 "ANSWER_DIRECTLY"
    """
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个只返回两类结果的分类器：RETRIEVE 或 ANSWER_DIRECTLY。依据下面规则：
- 如果问题涉及具体项目/公司内部信息、代码片段、API 用法、部署/配置、文档特定内容或需要查阅外部/专有知识 => 返回 RETRIEVE。
- 如果问题是通用知识、闲聊、问候、笑话、常识性事实（不依赖项目/内部文档）=> 返回 ANSWER_DIRECTLY。

示例：
Q: \"能给我讲个笑话吗？\" -> ANSWER_DIRECTLY
Q: \"我们项目的 payment-service 报错 500，如何排查？\" -> RETRIEVE
Q: \"Python 列表推导是什么？\" -> ANSWER_DIRECTLY
Q: \"如何在我们 k8s 集群里给 ingress 加注解？\" -> RETRIEVE
Q: \"你好\" -> ANSWER_DIRECTLY
Q: \"今天天气真不错。\" -> ANSWER_DIRECTLY
Q: \"基于已有的知识回答我：k8s相关知识\" -> RETRIEVE

只输出单个词：RETRIEVE 或 ANSWER_DIRECTLY，不要任何多余说明。"""),
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

    # 语义缓存检查（默认开启）
    cache_hit = None
    cache_hit = await search_semantic_cache(request.query)
    if cache_hit:
        logger.info(f"语义缓存命中，跳过RAG流程: {request.query[:50]}...")
        async def stream_cached_response() -> AsyncGenerator[str, None]:
            yield cache_hit["response"]
        return StreamingResponse(stream_cached_response(), media_type="text/plain")

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
            # 收集完整响应用于缓存
            full_response = ""
            async for chunk in no_context_chain.astream(request.query):
                full_response += chunk
                yield chunk
            # 缓存结果（如果启用并且有响应）
            if full_response:
                await add_to_semantic_cache(request.query, full_response)
        
        return StreamingResponse(generate_direct_response(), media_type="text/plain")

    logger.info(f"问题需要进行检索，响应路径修改-rag")
    # Use explicit similarity search via Milvus Collection.search to obtain normalized scores
    # This avoids ambiguity from vector_store implementations and provides stable 0..1 scores
    raw_docs_with_scores = await similarity_search_collection(full_collection_name, request.query, top_k=request.top_k, metric="COSINE")

    docs = [doc for doc, score in raw_docs_with_scores]
    scores = [score for doc, score in raw_docs_with_scores]

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

        # 收集完整响应用于缓存
        full_response = ""
        logger.info("test2--------------------------------------")
        if not selected_docs:
            no_context_chain = (
                {"question": RunnablePassthrough()}
                | no_context_prompt
                | llm
                | StrOutputParser()
            )
            async for chunk in no_context_chain.astream(request.query):
                full_response += chunk
                yield chunk
            # 缓存结果
            if full_response:
                await add_to_semantic_cache(request.query, full_response)

        else:
            logger.info("test3--------------------------------------")
            context = "\n\n---\n\n".join(doc.page_content for doc in selected_docs)
            rag_chain = (
                {"context": RunnableLambda(lambda x: context), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            logger.info("test4--------------------------------------")
            async for chunk in rag_chain.astream(request.query):
                # 累加响应以便后续缓存
                full_response += chunk
                yield chunk

            logger.info("test5--------------------------------------")
            # 缓存RAG结果（仅当之前没有命中缓存且有完整响应）
            if full_response and not cache_hit:
                await add_to_semantic_cache(request.query, full_response)

    return StreamingResponse(generate_response(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)