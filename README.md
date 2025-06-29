# RAG System on Kubernetes

This project provides a complete, multi-service RAG (Retrieval-Augmented Generation) system designed to run on Kubernetes. It includes services for file storage (MinIO), document parsing, text embedding, reranking, language model serving (LLM), and a unified API Gateway.

## High-Level Architecture

1.  **API Gateway (`api-service`)**: The single entry point for all external requests.
2.  **File Service (`fileservice`)**: Handles file uploads and downloads to and from MinIO.
3.  **Parser Service (`parser-service`)**: Splits documents into text chunks.
4.  **Retriever Service (`retriever-service`)**: The core orchestration layer. It communicates with the embedding, reranker, and LLM models, and manages vector storage in Milvus.
5.  **Model Services (`embedding`, `reranker`, `llm`)**: Dedicated, GPU-powered services running models with vLLM for high performance.
6.  **Vector Database (`milvus`)**: Stores and indexes document vectors for fast retrieval.
7.  **Object Storage (`minio`)**: Stores the original source documents.

## Deployment

Please refer to the individual `*.yaml` files for deployment instructions for each component. Ensure you build and push the Docker images for `api-service`, `fileservice`, `parser-service`, and `retriever-service` to a registry accessible by your Kubernetes cluster before applying the manifests.

## System API Endpoints (via API Gateway)

All interactions with the RAG system should go through the **API Gateway**. It provides a clean, high-level interface for managing knowledge bases and performing queries.

First, find the NodePort for the API Gateway:
```bash
kubectl get svc api-service -n rag
```
Use the returned IP and port for all the following requests (`http://<your-node-ip>:<api-gateway-node-port>`).

---

### 1. Knowledge Base Management

#### Create a Knowledge Base

Creates a new, empty knowledge base (a "collection" in Milvus).

- **Endpoint**: `POST /collections`
- **Body**:
  ```json
  {
    "collection_name": "my_new_kb"
  }
  ```
- **Example:**
  ```bash
  curl -X POST http://<api-gateway-url>/collections \
    -H "Content-Type: application/json" \
    -d '{"collection_name": "my_new_kb"}'
  ```

#### Delete a Knowledge Base

Permanently deletes a knowledge base and all its content.

- **Endpoint**: `DELETE /collections/{collection_name}`
- **Example:**
  ```bash
  curl -X DELETE http://<api-gateway-url>/collections/my_new_kb
  ```

---

### 2. Adding Content

#### Add a File to a Knowledge Base

This is the main pipeline for adding new information. It automatically handles file upload, text splitting, embedding, and storage.

- **Endpoint**: `POST /add_file_to_collection`
- **Request Type**: `multipart/form-data`
- **Form Fields**:
  - `file`: The document to be added (e.g., `.txt`, `.md`).
  - `collection_name`: The target knowledge base.
- **Example:**
  ```bash
  curl -X POST http://<api-gateway-url>/add_file_to_collection \
    -F "file=@/path/to/your/document.txt" \
    -F "collection_name=my_new_kb"
  ```

---

### 3. Asking Questions (RAG)

#### Get a Chat Completion

Ask a question against a specific knowledge base and get a RAG-powered answer. This endpoint is designed to be compatible with the OpenAI Chat Completions API format.

- **Endpoint**: `POST /chat/completions`
- **Body**:
  - `model`: (Required) The name of the knowledge base to query (e.g., `"my_new_kb"`).
  - `messages`: A list of messages, following the OpenAI format. The content of the last "user" role message is used as the query.
  - `stream`: (Optional) Set to `true` for a streaming response. Defaults to `false`.
- **Example (Non-Streaming):**
  ```bash
  curl -X POST http://<api-gateway-url>/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "my_new_kb",
      "messages": [
        {"role": "user", "content": "What is the main topic of the document?"}
      ],
      "stream": false
    }'
  ```

- **Example (Streaming):**
  ```bash
  curl -N -X POST http://<api-gateway-url>/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "my_new_kb",
      "messages": [
        {"role": "user", "content": "Summarize the document in three bullet points."}
      ],
      "stream": true
    }'
  ```

## Project Structure

```
.
├── Dockerfile
├── fileservice-deployment.yaml
├── main.py
├── minio-deployment.yaml
├── parser-deployment.yaml
├── parser-main.py
├── parser.Dockerfile
├── parser-requirements.txt
└── requirements.txt
```

## Deployment Steps

Follow these steps to deploy the entire stack to your Kubernetes cluster.

### 1. Deploy MinIO

First, deploy the MinIO server. This command will also create the `rag` namespace for all our services.

```bash
kubectl apply -f minio-deployment.yaml
```

After applying, you can check the status of the MinIO pod in the `rag` namespace:
```bash
kubectl get pods -l app=minio -n rag
```

### 2. Build and Push the File Service Docker Image

The file service needs to be containerized before deploying.

**Important:** You need to have Docker installed and be logged into a container registry (like Docker Hub) that your Kubernetes cluster can pull from.

1.  **Build the image:**
    Navigate to the project directory and run the build command. Replace `your-dockerhub-username/minio-fileservice:latest` with your own image name.

    ```bash
    docker build -t crater-harbor.act.buaa.edu.cn/user-liujh24/minio-fileservice:latest .
    ```

2.  **Push the image:**
    Push the image to your container registry.

    ```bash
    docker push crater-harbor.act.buaa.edu.cn/user-liujh24/minio-fileservice:latest
    ```

### 3. Build, Push, and Deploy the File Service

1.  **Build the fileservice image:**
    Use the `fileservice.Dockerfile` to build the image. Replace the image name with your own.

    ```bash
    docker build -t crater-harbor.act.buaa.edu.cn/user-liujh24/minio-fileservice:latest -f fileservice.Dockerfile .
    ```

2.  **Push the fileservice image:**
    ```bash
    docker push crater-harbor.act.buaa.edu.cn/user-liujh24/minio-fileservice:latest
    ```

3.  **Deploy the service:**
    Apply the manifest to deploy the file service into the `rag` namespace.

    ```bash
    kubectl apply -f fileservice-deployment.yaml
    ```

    Check the status of the file service pod:
    ```bash
    kubectl get pods -l app=fileservice -n rag
    ```

### 4. Build, Push, and Deploy the Document Parsing Service

This service fetches files from MinIO and splits them into chunks.

1.  **Build the parser image:**
    Use the `parser.Dockerfile` to build the image. Replace the image name with your own.

    ```bash
    docker build -t crater-harbor.act.buaa.edu.cn/user-liujh24/minio-parser:latest -f parser.Dockerfile .
    ```

2.  **Push the parser image:**
    ```bash
    docker push crater-harbor.act.buaa.edu.cn/user-liujh24/minio-parser:latest
    ```

3.  **Deploy the parser service:**
    The `parser-deployment.yaml` is already configured with the correct image name. Apply the manifest:

    ```bash
    kubectl apply -f parser-deployment.yaml
    ```
    Check the status:
    ```bash
    kubectl get pods -l app=parser -n rag
    ```

### 5. Build, Push, and Deploy the API Gateway Service

This service acts as the main entry point for the entire RAG system.

1.  **Build the API gateway image:**
    ```bash
    docker build -t crater-harbor.act.buaa.edu.cn/user-liujh24/rag-api-gateway:latest -f api.Dockerfile .
    ```

2.  **Push the API gateway image:**
    ```bash
    docker push crater-harbor.act.buaa.edu.cn/user-liujh24/rag-api-gateway:latest
    ```

3.  **Deploy the API gateway service:**
    ```bash
    kubectl apply -f api-deployment.yaml
    ```
    Check the status:
    ```bash
    kubectl get pods -l app=api-gateway -n rag
    ```

### 6. Deploy the Embedding Model Service

This service runs a dedicated embedding model using the high-performance vLLM server.

**IMPORTANT:** Before you begin, ensure your Kubernetes cluster has nodes with available NVIDIA GPUs and the NVIDIA device plugin is installed.

1.  **Create a Hugging Face Token Secret:**
    The vLLM server needs a Hugging Face token to download the model. Replace `YOUR_HF_TOKEN` with your actual token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

    ```bash
    kubectl create secret generic hf-token-secret --from-literal=token='xxx' -n rag
    ```

2.  **Build and Push the Embedding Service Image:**
    Unlike other services, the embedding service is built using a custom `embedding.Dockerfile` for maximum hardware compatibility.

    ```bash
    docker build -t crater-harbor.act.buaa.edu.cn/user-liujh24/rag-embedding-service:latest -f embedding.Dockerfile .
    ```
    Push the newly built image to your registry:
    ```bash
    docker push crater-harbor.act.buaa.edu.cn/user-liujh24/rag-embedding-service:latest
    ```

3.  **Deploy the Embedding Service:**
    This command will create a Persistent Volume Claim for the model cache, the Deployment for the vLLM server, and a Service to expose it.

    ```bash
    kubectl apply -f embedding-deployment.yaml
    ```

4.  **Check the Status:**
    It may take several minutes for the pod to become ready as it needs to download the model and initialize the server.

    ```bash
    # Check pod status
    kubectl get pods -l app=embedding-service -n rag

    # View logs to see download progress and server status
    kubectl logs -f -l app=embedding-service -n rag
    ```

### 7. 部署 Reranker 模型服务

本服务用于提供重排序（rerank）能力，基于 BAAI/bge-reranker-base 模型，依赖 GPU 资源。

1.  **构建并推送 Reranker 服务镜像：**
    
    （如需自定义镜像，可参考 embedding 服务的构建方法，默认可直接使用预设镜像）

2.  **部署 Reranker 服务：**
    
    ```bash
    kubectl apply -f reranker-deployment.yaml
    ```

3.  **检查服务状态：**
    
    ```bash
    kubectl get pods -l app=reranker-service -n rag
    kubectl logs -f -l app=reranker-service -n rag
    ```

#### reranker 服务测试方法

1. 查询 NodePort：

```bash
kubectl get svc reranker-service -n rag
```

2. 发送 rerank 请求（OpenAI API 格式）：

```bash
curl -X POST http://<NodeIP>:<NodePort>/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-reranker-base",
    "query": "中国的首都是哪里？",
    "documents": [
      "北京是中国的首都。",
      "上海是中国最大的城市。",
      "广州是中国南方的重要城市。"
    ]
  }'
```

返回结果会包含每个 document 的分数和排序。

### 8. 部署 LLM 模型服务

本服务基于 Qwen/Qwen3-0.6B 模型，提供核心的语言模型能力。

1.  **部署 LLM 服务：**
    
    ```bash
    kubectl apply -f llm-deployment.yaml
    ```

2.  **检查服务状态：**
    
    ```bash
    kubectl get pods -l app=llm-service -n rag
    kubectl logs -f -l app=llm-service -n rag
    ```

#### LLM 服务测试方法

1.  **查询 NodePort：**
    
    ```bash
    kubectl get svc llm-service -n rag
    ```

2.  **发送聊天补全请求（OpenAI API 格式）：**
    
    ```bash
    curl -X POST http://<NodeIP>:<NodePort>/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
          {"role": "user", "content": "你好，请介绍一下你自己。"}
        ]
      }'
    ```

### 9. 部署 Milvus 服务

Milvus 是向量数据库，用于存储和检索向量数据。本部署会自动连接已存在的 minio 服务（minio-service:9000），无需重复部署 MinIO。

1. **部署 Milvus 及其依赖（etcd）：**

```bash
kubectl apply -f milvus-deployment.yaml
```

2. **检查服务状态：**

```bash
kubectl get pods -l app=milvus-etcd -n rag
kubectl get pods -l app=milvus-standalone -n rag
kubectl logs -f -l app=milvus-standalone -n rag
```

3. **服务端口说明：**
- gRPC 端口：19530（用于 SDK/客户端连接）
- HTTP 端口：9091（用于健康检查和监控）

4. **MinIO 连接说明：**
Milvus 会自动通过环境变量连接 rag 命名空间下的 minio-service，访问密钥为 minioadmin/minioadmin。

如需外部访问 Milvus，可通过 NodePort 方式连接上述端口。

#### Milvus 服务测试方法

你可以通过 `pymilvus` SDK 来测试 Milvus 服务。以下提供两种测试方法：

**方法一：在 Kubernetes 集群内部测试（推荐）**

这是最直接可靠的方法，它会在 `rag` 命名空间下启动一个临时的 Python Pod，安装 `pymilvus` 并尝试连接 Milvus。

执行以下命令：

```bash
kubectl run -it --rm --image=python:3.9-slim --restart=Never milvus-test -n rag -- /bin/bash -c "pip install pymilvus && python -c \"
import pymilvus
try:
    pymilvus.connections.connect(alias='default', host='milvus-service', port='19530')
    print('Successfully connected to Milvus!')
    print(f'Existing collections: {pymilvus.utility.list_collections()}')
except Exception as e:
    print(f'Failed to connect: {e}')
finally:
    if 'default' in pymilvus.connections.list_connections():
        pymilvus.connections.disconnect('default')
\""
```
如果看到 `Successfully connected to Milvus!` 和一个集合列表（可能为空），说明 Milvus 服务正常。

**方法二：在本地机器上测试（通过 NodePort）**

1.  **安装 `pymilvus` 客户端：**
    
    ```bash
    pip install pymilvus
    ```

2.  **获取连接信息：**
    
    *   获取任一 Kubernetes节点的 IP 地址（`EXTERNAL-IP` 或 `INTERNAL-IP`）。
        ```bash
        kubectl get nodes -o wide
        ```
    *   获取 Milvus gRPC 服务的 NodePort。
        ```bash
        kubectl get svc milvus-service -n rag
        ```
        记下 `19530` 对应的端口号（例如 `3xxxx`）。

3.  **创建并运行测试脚本：**
    
    创建一个名为 `test_milvus.py` 的文件，内容如下，并将 `<NodeIP>` 和 `<NodePort>` 替换为上一步获取的值。
    
    ```python
    import pymilvus
    
    MILVUS_HOST = "<NodeIP>"       # 替换为你的节点 IP
    MILVUS_PORT = "<NodePort>"   # 替换为 Milvus 的 NodePort
    
    try:
        print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
        pymilvus.connections.connect(alias='default', host=MILVUS_HOST, port=str(MILVUS_PORT))
        print("Successfully connected to Milvus!")
        print(f"Existing collections: {pymilvus.utility.list_collections()}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'default' in pymilvus.connections.list_connections():
            pymilvus.connections.disconnect('default')
            print("Disconnected from Milvus.")
    ```
    
    运行脚本：
    ```bash
    python test_milvus.py
    ```

### 10. Accessing the System via the API Gateway

The **API Gateway** is the primary entry point to the system, exposed via `NodePort`. The other services (`fileservice`, `parser-service`) are now considered internal components.

To find the port for the **API Gateway**, run:
```bash
kubectl get svc api-service -n rag
```
The main API Base URL for the entire system is `http://<your-node-ip>:<api-gateway-node-port>`.

### Architecture Note (Security Improvement)

For easier debugging, `fileservice-service` and `parser-service` are currently set to `NodePort`. In a production environment, you should change their type back to `ClusterIP` in their respective `.yaml` files, so they are not directly exposed to the internet. The API Gateway would be the only externally accessible component.

## API Endpoints (via API Gateway)

All requests should now go through the API Gateway.

- **`POST /add_file_to_collection`** (Automated Workflow)
  - This is the primary endpoint for adding knowledge to the system. It automates the upload -> parse -> store pipeline.
  - **Form Data:**
    - `file`: The document file to process.
    - `collection_name`: The name of the knowledge base to add the document to.
    - `create_collection_if_not_exists`: `true` (default) or `false`.
  - **Example using curl:**
    ```bash
    curl -X POST http://<your-node-ip>:<api-gateway-node-port>/add_file_to_collection \
      -F "file=@/path/to/your/document.txt" \
      -F "collection_name=my_knowledge_base"
    ```

- **`POST /upload`** (Manual Step)
  - Upload a file. To automatically trigger parsing, add the `?parse=true` query parameter.
  - **Example (Upload only):**
    ```bash
    curl -X POST -F "file=@/path/to/your/file.txt" http://<your-node-ip>:<api-gateway-node-port>/upload
    ```
  - **Example (Upload and Parse):**
    ```bash
    curl -X POST -F "file=@/path/to/your/file.txt" "http://<your-node-ip>:<api-gateway-node-port>/upload?parse=true"
    ```

- **`GET /download/{file_name}`**
  - Download a file.
  - **Example:**
    ```bash
    curl http://<your-node-ip>:<api-gateway-node-port>/download/file.txt -o downloaded_file.txt
    ```

- **`DELETE /delete/{file_name}`**
  - Delete a file.
  - **Example:**
    ```bash
    curl -X DELETE http://<your-node-ip>:<api-gateway-node-port>/delete/file.txt
    ```

- **`POST /parse/{file_name}`**
  - Manually trigger parsing for a file that already exists in MinIO.
  - **Example:**
    ```bash
    curl -X POST http://<your-node-ip>:<api-gateway-node-port>/parse/test.txt
    ```

## Internal Service Testing
## Service Testing (Debug Mode)

For easier debugging, some internal services are temporarily exposed via `NodePort`.

**Test the Embedding Service:**
1.  Find the port assigned to the embedding service:
    ```bash
    kubectl get svc embedding-service -n rag
    ```
2.  In a new terminal, send a request to the `/v1/embeddings` endpoint. Replace `<your-node-ip>` and `<embedding-node-port>` with the correct values.
    ```bash
    curl http://<your-node-ip>:<embedding-node-port>/v1/embeddings \
      -H "Content-Type: application/json" \
      -d '{
        "input": "This is a test sentence.",
        "model": "BAAI/bge-small-en-v1.5"
      }'
    ``` 

### 11. 部署 RAG 检索服务

这个服务是整个 RAG 系统的核心，它基于 LangChain 构建，负责编排 embedding, reranker, llm 和 milvus 服务，实现完整的检索增强生成流程。

1.  **构建并推送检索服务镜像：**
    
    ```bash
    docker build -t crater-harbor.act.buaa.edu.cn/user-liujh24/rag-retriever-service:latest -f retriever.Dockerfile .
    docker push crater-harbor.act.buaa.edu.cn/user-liujh24/rag-retriever-service:latest
    ```

2.  **部署服务：**
    
    ```bash
    kubectl apply -f retriever-deployment.yaml
    ```

3.  **检查服务状态：**
    
    ```bash
    kubectl get pods -l app=retriever-service -n rag
    kubectl logs -f -l app=retriever-service -n rag
    ```

#### 检索服务测试方法

你可以通过 `curl` 调用其 API 来测试整个 RAG 流程。

1.  **获取服务 NodePort：**
    
    ```bash
    kubectl get svc retriever-service -n rag
    ```

2.  **第一步：创建知识库（Collection）**
    
    ```bash
    curl -X POST http://<NodeIP>:<NodePort>/create_collection \
      -H "Content-Type: application/json" \
      -d '{
        "collection_name": "my_knowledge_base"
      }'
    ```

3.  **第二步：向知识库添加文档**
    
    ```bash
    curl -X POST http://<NodeIP>:<NodePort>/add_documents \
      -H "Content-Type: application/json" \
      -d '{
        "collection_name": "my_knowledge_base",
        "documents": [
          "The Eiffel Tower is located in Paris, France.",
          "The capital of Japan is Tokyo.",
          "The Great Wall of China is one of the seven wonders of the world."
        ]
      }'
    ```

4.  **第三步：执行 RAG 生成**
    
    ```bash
    curl -N -X POST http://<NodeIP>:<NodePort>/rag_generate \
      -H "Content-Type: application/json" \
      -d '{
        "collection_name": "my_knowledge_base",
        "query": "Where is the Eiffel Tower?"
      }'
    ```
    
    `-N` 参数用于接收流式响应。

### 12. Deploy Horizontal Pod Autoscalers (Optional)

To enable automatic scaling based on load, you can deploy Horizontal Pod Autoscalers (HPA) for the stateless services.

**Prerequisite:** Ensure the `resources.requests` are defined in the `api-deployment.yaml`, `fileservice-deployment.yaml`, `parser-deployment.yaml`, and `retriever-deployment.yaml` files, as HPA relies on these values to calculate CPU utilization.

1.  **Deploy the HPAs:**
    The following command applies the HPA configurations for all four services. The YAML files are configured with a low CPU threshold (e.g., 10%) for easy testing.

    ```bash
    kubectl apply -f api-hpa.yaml -f fileservice-hpa.yaml -f parser-hpa.yaml -f retriever-hpa.yaml
    ```

2.  **Monitor the HPA Status:**
    In a new terminal, run this command to watch the HPA status in real-time.

    ```bash
    kubectl get hpa -n rag -w
    ```
    You will see the current CPU utilization against the target (e.g., `8% / 10%`) and the number of replicas for each service.

3.  **Generate Load for Testing:**
    To trigger autoscaling, you need to send traffic to the API Gateway. Here is a PowerShell example that continuously calls the chat endpoint in a loop.

    ```powershell
    # Make sure to replace <api-gateway-url> with the correct URL
    while ($true) {
        curl -N -X POST http://<api-gateway-url>/chat/completions `
            -H "Content-Type: application/json" `
            -d '{"model": "my_knowledge_base", "messages": [{"role": "user", "content": "hello"}]}' `
            -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 100
    }
    ```
    As you run this script, you should see the CPU usage in the monitor window increase past the target, and Kubernetes will automatically start new pods, increasing the `REPLICAS` count.

### 12. 访问系统入口（API Gateway）

The **API Gateway** is the primary entry point to the system, exposed via `NodePort`. The other services (`fileservice`, `parser-service`) are now considered internal components.

To find the port for the **API Gateway**, run:
```bash
kubectl get svc api-service -n rag
```
The main API Base URL for the entire system is `http://<your-node-ip>:<api-gateway-node-port>`.

### Architecture Note (Security Improvement)

For easier debugging, `fileservice-service` and `parser-service` are currently set to `NodePort`. In a production environment, you should change their type back to `ClusterIP` in their respective `.yaml` files, so they are not directly exposed to the internet. The API Gateway would be the only externally accessible component.

## API Endpoints (via API Gateway)

All requests should now go through the API Gateway.

- **`POST /add_file_to_collection`** (Automated Workflow)
  - This is the primary endpoint for adding knowledge to the system. It automates the upload -> parse -> store pipeline.
  - **Form Data:**
    - `file`: The document file to process.
    - `collection_name`: The name of the knowledge base to add the document to.
    - `create_collection_if_not_exists`: `true` (default) or `false`.
  - **Example using curl:**
    ```bash
    curl -X POST http://<your-node-ip>:<api-gateway-node-port>/add_file_to_collection \
      -F "file=@/path/to/your/document.txt" \
      -F "collection_name=my_knowledge_base"
    ```

- **`POST /upload`** (Manual Step)
  - Upload a file. To automatically trigger parsing, add the `?parse=true` query parameter.
  - **Example (Upload only):**
    ```bash
    curl -X POST -F "file=@/path/to/your/file.txt" http://<your-node-ip>:<api-gateway-node-port>/upload
    ```
  - **Example (Upload and Parse):**
    ```bash
    curl -X POST -F "file=@/path/to/your/file.txt" "http://<your-node-ip>:<api-gateway-node-port>/upload?parse=true"
    ```

- **`GET /download/{file_name}`**
  - Download a file.
  - **Example:**
    ```bash
    curl http://<your-node-ip>:<api-gateway-node-port>/download/file.txt -o downloaded_file.txt
    ```

- **`DELETE /delete/{file_name}`**
  - Delete a file.
  - **Example:**
    ```bash
    curl -X DELETE http://<your-node-ip>:<api-gateway-node-port>/delete/file.txt
    ```

- **`POST /parse/{file_name}`**
  - Manually trigger parsing for a file that already exists in MinIO.
  - **Example:**
    ```bash
    curl -X POST http://<your-node-ip>:<api-gateway-node-port>/parse/test.txt
    ```

## Internal Service Testing
## Service Testing (Debug Mode)

For easier debugging, some internal services are temporarily exposed via `NodePort`.

**Test the Embedding Service:**
1.  Find the port assigned to the embedding service:
    ```bash
    kubectl get svc embedding-service -n rag
    ```
2.  In a new terminal, send a request to the `/v1/embeddings` endpoint. Replace `<your-node-ip>` and `<embedding-node-port>` with the correct values.
    ```bash
    curl http://<your-node-ip>:<embedding-node-port>/v1/embeddings \
      -H "Content-Type: application/json" \
      -d '{
        "input": "This is a test sentence.",
        "model": "BAAI/bge-small-en-v1.5"
      }'
    ``` 
