# Dynamic-Evolutionary-System

构建一个云原生软件系统动态演化框架，目前具体表现形式可能是一个RAG应用或Agent应用（待定），当前形式为自适应RAG对话系统，基于Kubernetes，要求支持微服务副本调整、服务拓扑调整、数据流优化等≥3 种系统演化能力。该系统将各RAG组件微服务化，实现基础设施资源的智能调度与分配，支持服务弹性伸缩，并通过完整的监控体系展示系统动态演化过程。
本项目当前内嵌服务为一个完整的多服务检索增强生成 (RAG) 系统，旨在 Kubernetes 上运行。它包括文件存储 (MinIO)、文档解析、文本嵌入、重排序、语言模型服务 (LLM) 和统一 API 网关等服务。

## 高层架构

* **API 网关 (api-service)**：所有外部请求的单一入口点。
* **文件服务 (fileservice)**：处理文件到 MinIO 的上传和下载。
* **解析服务 (parser-service)**：将文档分割成文本块。
* **检索器服务 (retriever-service)**：核心编排层。它与嵌入、重排序和 LLM 模型通信，并管理 Milvus 中的向量存储。
* **模型服务 (embedding, reranker, llm)**：专用、GPU 驱动的服务，使用 vLLM 运行模型以实现高性能。
* **向量数据库 (milvus)**：存储和索引文档向量，用于快速检索。
* **对象存储 (minio)**：存储原始源文档。



## 部署步骤

按照以下步骤将整个堆栈部署到您的 Kubernetes 集群。

### 1. 部署 MinIO

首先，部署 MinIO 服务器。此命令还将为我们所有服务创建 `rag` 命名空间。

```bash
kubectl apply -f minio-deployment.yaml
```

应用后，您可以在 `rag` 命名空间中检查 MinIO Pod 的状态：

```bash
kubectl get pods -l app=minio -n rag
```

### 2. 构建并推送文件服务 Docker 镜像

文件服务需要在部署前进行容器化。
**重要提示**：您需要安装 Docker 并登录到您的 Kubernetes 集群可以从中拉取镜像的容器注册表（例如 Docker Hub）。

**构建镜像**：
导航到项目目录并运行构建命令。将 `crater-harbor.act.buaa.edu.cn/user-liujh24/minio-fileservice:latest` 替换为您自己的镜像名称。

```bash
docker build -t crater-harbor.act.buaa.edu.cn/user-liujh24/minio-fileservice:latest .
```

**推送镜像**：
将镜像推送到您的容器注册表。

```bash
docker push crater-harbor.act.buaa.edu.cn/user-liujh24/minio-fileservice:latest
```

### 3. 构建、推送和部署文件服务

**构建 fileservice 镜像**：
使用 `fileservice.Dockerfile` 构建镜像。替换为您自己的镜像名称。

```bash
docker build -t crater-harbor.act.buaa.edu.cn/user-liujh24/minio-fileservice:latest -f fileservice.Dockerfile .
```

**推送 fileservice 镜像**：

```bash
docker push crater-harbor.act.buaa.edu.cn/user-liujh24/minio-fileservice:latest
```

**部署服务**：
应用清单以将文件服务部署到 `rag` 命名空间。

```bash
kubectl apply -f fileservice-deployment.yaml
```

检查文件服务 Pod 的状态：

```bash
kubectl get pods -l app=fileservice -n rag
```

### 4. 构建、推送和部署文档解析服务

此服务从 MinIO 获取文件并将其分割成块。

**构建 parser 镜像**：
使用 `parser.Dockerfile` 构建镜像。替换为您自己的镜像名称。

```bash
docker build -t crater-harbor.act.buaa.edu.cn/user-liujh24/minio-parser:latest -f parser.Dockerfile .
```

**推送 parser 镜像**：

```bash
docker push crater-harbor.act.buaa.edu.cn/user-liujh24/minio-parser:latest
```

**部署解析服务**：
`parser-deployment.yaml` 已配置正确的镜像名称。应用清单：

```bash
kubectl apply -f parser-deployment.yaml
```

检查状态：

```bash
kubectl get pods -l app=parser -n rag
```

### 5. 构建、推送和部署 API 网关服务

此服务充当整个 RAG 系统的主要入口点。

**构建 API 网关镜像**：

```bash
docker build -t crater-harbor.act.buaa.edu.cn/user-liujh24/rag-api-gateway:latest -f api.Dockerfile .
```

**推送 API 网关镜像**：

```bash
docker push crater-harbor.act.buaa.edu.cn/user-liujh24/rag-api-gateway:latest
```

**部署 API 网关服务**：

```bash
kubectl apply -f api-deployment.yaml
```

检查状态：

```bash
kubectl get pods -l app=api-gateway -n rag
```

### 6. 部署嵌入模型服务

此服务使用高性能 vLLM 服务器运行专用嵌入模型。
**重要提示**：在开始之前，请确保您的 Kubernetes 集群具有带可用 NVIDIA GPU 的节点，并且已安装 NVIDIA 设备插件。

**创建 Hugging Face Token Secret**：
vLLM 服务器需要一个 Hugging Face token 来下载模型。将 `YOUR_HF_TOKEN` 替换为您从 `https://huggingface.co/settings/tokens` 获取的实际 token。

```bash
kubectl create secret generic hf-token-secret --from-literal=token='xxx' -n rag
```

**构建并推送嵌入服务镜像**：
与其他服务不同，嵌入服务使用自定义的 `embedding.Dockerfile` 构建，以实现最大的硬件兼容性。

````bash
docker build -t crater-harbor.act.buaa.edu.cn/user-liujh24/rag-embedding-service:latest -f embedding.Dockerfile .```

将新构建的镜像推送到您的注册表：

```bash
docker push crater-harbor.act.buaa.edu.cn/user-liujh24/rag-embedding-service:latest
````

**部署嵌入服务**：
此命令将创建一个模型缓存的持久卷声明 (Persistent Volume Claim)、vLLM 服务器的部署 (Deployment) 和暴露它的服务 (Service)。

```bash
kubectl apply -f embedding-deployment.yaml
```

**检查状态**：
Pod 可能需要几分钟才能准备就绪，因为它需要下载模型并初始化服务器。

```bash
# 检查 Pod 状态
kubectl get pods -l app=embedding-service -n rag
# 查看日志以了解下载进度和服务器状态
kubectl logs -f -l app=embedding-service -n rag
```

### 7. 部署 Reranker 模型服务

本服务用于提供重排序（rerank）能力，基于 `BAAI/bge-reranker-base` 模型，依赖 GPU 资源。

**构建并推送 Reranker 服务镜像**：
（如需自定义镜像，可参考 embedding 服务的构建方法，默认可直接使用预设镜像）

```bash
# docker build -t crater-harbor.act.buaa.edu.cn/user-liujh24/rag-reranker-service:latest -f reranker.Dockerfile .
# docker push crater-harbor.act.buaa.edu.cn/user-liujh24/rag-reranker-service:latest
```

**部署 Reranker 服务**：

```bash
kubectl apply -f reranker-deployment.yaml
```

**检查服务状态**：

```bash
kubectl get pods -l app=reranker-service -n rag
kubectl logs -f -l app=reranker-service -n rag
```

**Reranker 服务测试方法**

**查询 NodePort**：

```bash
kubectl get svc reranker-service -n rag
```

**发送 rerank 请求 (OpenAI API 格式)**：

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

本服务基于 `Qwen/Qwen3-0.6B` 模型，提供核心的语言模型能力。

**部署 LLM 服务**：

```bash
kubectl apply -f llm-deployment.yaml
```

**检查服务状态**：

```bash
kubectl get pods -l app=llm-service -n rag
kubectl logs -f -l app=llm-service -n rag
```

**LLM 服务测试方法**

**查询 NodePort**：

```bash
kubectl get svc llm-service -n rag
```

**发送聊天补全请求 (OpenAI API 格式)**：

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

Milvus 是向量数据库，用于存储和检索向量数据。本部署会自动连接已存在的 minio 服务 (`minio-service:9000`)，无需重复部署 MinIO。

**部署 Milvus 及其依赖 (etcd)**：

```bash
kubectl apply -f milvus-deployment.yaml
```

**检查服务状态**：

```bash
kubectl get pods -l app=milvus-etcd -n rag
kubectl get pods -l app=milvus-standalone -n rag
kubectl logs -f -l app=milvus-standalone -n rag
```

**服务端口说明**：

* gRPC 端口：19530（用于 SDK/客户端连接）
* HTTP 端口：9091（用于健康检查和监控）

**MinIO 连接说明**：
Milvus 会自动通过环境变量连接 `rag` 命名空间下的 `minio-service`，访问密钥为 `minioadmin/minioadmin`。

如需外部访问 Milvus，可通过 NodePort 方式连接上述端口。

**Milvus 服务测试方法**
可以在电脑上装一个 `attu` 来测试 Milvus 服务。

### 10. 部署 RAG 检索服务

这个服务是整个 RAG 系统的核心，它基于 LangChain 构建，负责编排 embedding, reranker, llm 和 milvus 服务，实现完整的检索增强生成流程。

**构建并推送检索服务镜像**：

```bash
docker build -t crater-harbor.act.buaa.edu.cn/user-liujh24/rag-retriever-service:latest -f retriever.Dockerfile .
docker push crater-harbor.act.buaa.edu.cn/user-liujh24/rag-retriever-service:latest
```

**部署服务**：

```bash
kubectl apply -f retriever-deployment.yaml
```

**检查服务状态**：

```bash
kubectl get pods -l app=retriever-service -n rag
kubectl logs -f -l app=retriever-service -n rag
```

**检索服务测试方法**
你可以通过 curl 调用其 API 来测试整个 RAG 流程。

**获取服务 NodePort**：

```bash
kubectl get svc retriever-service -n rag
```

**第一步：创建知识库 (Collection)**

```bash
curl -X POST http://<NodeIP>:<NodePort>/create_collection \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_knowledge_base"
  }'
```

**第二步：向知识库添加文档**

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

**第三步：执行 RAG 生成**

```bash
curl -N -X POST http://<NodeIP>:<NodePort>/rag_generate \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_knowledge_base",
    "query": "Where is the Eiffel Tower?"
  }'
```

-N 参数用于接收流式响应。

### 11. 部署水平 Pod 自动伸缩 (可选)

为了实现基于负载的自动伸缩，您可以为无状态服务部署水平 Pod 自动伸缩器 (HPA)。
**先决条件**：确保 `api-deployment.yaml`、`fileservice-deployment.yaml`、`parser-deployment.yaml` 和 `retriever-deployment.yaml` 文件中定义了 `resources.requests`，因为 HPA 依赖这些值来计算 CPU 利用率。

**部署 HPA**：
以下命令应用所有四个服务的 HPA 配置。YAML 文件配置了较低的 CPU 阈值（例如 10%）以便于测试。

```bash
kubectl apply -f api-hpa.yaml -f fileservice-hpa.yaml -f parser-hpa.yaml -f retriever-hpa.yaml
```

**监控 HPA 状态**：
在一个新的终端中，运行此命令以实时观察 HPA 状态。

````bash
kubectl get hpa -n rag -w```
您将看到每个服务的当前 CPU 利用率与目标（例如 8% / 10%）以及副本数量。

**生成负载进行测试**：
要触发自动伸缩，您需要向 API 网关发送流量。这是一个 PowerShell 示例，它在循环中持续调用聊天端点。

```powershell
# 确保替换 <api-gateway-url> 为正确的 URL
while ($true) {
    curl -N -X POST http://<api-gateway-url>/chat/completions -H "Content-Type: application/json" `
        -d '{"model": "my_knowledge_base", "messages": [{"role": "user", "content": "hello"}]}' -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 100
}
````

当您运行此脚本时，您应该会看到监控窗口中的 CPU 使用率超过目标，并且 Kubernetes 将自动启动新的 Pod，增加 `REPLICAS` 计数。

---

请参考各个 `*.yaml` 文件以获取每个组件的部署说明。在应用清单之前，请确保您已为 `api-service`、`fileservice`、`parser-service` 和 `retriever-service` 构建并将 Docker 镜像推送到您的 Kubernetes 集群可访问的注册表。

## 系统 API 端点 (通过 API 网关)

所有与 RAG 系统的交互都应通过 API 网关进行。它提供了一个清晰、高级的接口来管理知识库和执行查询。

首先，找到 API 网关的 NodePort：

```bash
kubectl get svc api-service -n rag
```

使用返回的 IP 和端口进行所有后续请求 (`http://<your-node-ip>:<api-gateway-node-port>`)。

### 1. 知识库管理

**创建知识库**
创建一个新的空知识库（Milvus 中的“集合”）。

* **端点**：`POST /collections`
* **请求体**：

  ```json
  {
    "collection_name": "my_new_kb"
  }
  ```
* **示例**：

  ```bash
  curl -X POST http://<api-gateway-url>/collections \
    -H "Content-Type: application/json" \
    -d '{"collection_name": "my_new_kb"}'
  ```

**删除知识库**
永久删除知识库及其所有内容。

* **端点**：`DELETE /collections/{collection_name}`
* **示例**：

  ```bash
  curl -X DELETE http://<api-gateway-url>/collections/my_new_kb
  ```

### 2. 添加内容

**向知识库添加文件**
这是添加新信息的主要管道。它自动处理文件上传、文本分割、嵌入和存储。

* **端点**：`POST /add_file_to_collection`
* **请求类型**：`multipart/form-data`
* **表单字段**：

  * `file`：要添加的文档（例如，`.txt`, `.md`）。
  * `collection_name`：目标知识库。
* **示例**：

  ```bash
  curl -X POST http://<api-gateway-url>/add_file_to_collection \
    -F "file=@/path/to/your/document.txt" \
    -F "collection_name=my_new_kb"
  ```

### 3. 提问 (RAG)

**获取聊天补全**
针对特定知识库提问并获取 RAG 驱动的答案。此端点设计为与 OpenAI Chat Completions API 格式兼容。

* **端点**：`POST /chat/completions`

* **请求体**：

  * `model`：(必需) 要查询的知识库名称（例如，“my\_new\_kb”）。
  * `messages`：消息列表，遵循 OpenAI 格式。最后一个“user”角色消息的内容用作查询。
  * `stream`：(可选) 设置为 `true` 以获取流式响应。默认为 `false`。

* **示例 (非流式)**：

  ```bash
  curl -X POST http://<api-gateway-url>/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "my_new_kb",
      "messages": [
        {"role": "user", "content": "文档的主要主题是什么？"}
      ],
      "stream": false
    }'
  ```

* **示例 (流式)**：

  ```bash
  curl -N -X POST http://<api-gateway-url>/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "my_new_kb",
      "messages": [
        {"role": "user", "content": "用三点总结文档。"}
      ],
      "stream": true
    }'
  ```

### 其他 API 网关暴露的 API 端点

所有请求现在都应通过 API 网关进行。

* `POST /add_file_to_collection` (自动化工作流)

  * 这是向系统添加知识的主要端点。它自动化了上传 -> 解析 -> 存储的管道。
  * **表单数据**：

    * `file`：要处理的文档文件。
    * `collection_name`：要添加文档的知识库名称。
    * `create_collection_if_not_exists`：`true` (默认) 或 `false`。
  * **示例 (使用 curl)**：

    ```bash
    curl -X POST http://<your-node-ip>:<api-gateway-node-port>/add_file_to_collection \
      -F "file=@/path/to/your/document.txt" \
      -F "collection_name=my_knowledge_base"
    ```

* `POST /upload` (手动步骤)

  * 上传文件。要自动触发解析，请添加 `?parse=true` 查询参数。
  * **示例 (仅上传)**：

    ```bash
    curl -X POST -F "file=@/path/to/your/file.txt" http://<your-node-ip>:<api-gateway-node-port>/upload
    ```
  * **示例 (上传并解析)**：

    ```bash
    curl -X POST -F "file=@/path/to/your/file.txt" "http://<your-node-ip>:<api-gateway-node-port>/upload?parse=true"
    ```

* `GET /download/{file_name}`

  * 下载文件。
  * **示例**：

    ```bash
    curl http://<your-node-ip>:<api-gateway-node-port>/download/file.txt -o downloaded_file.txt
    ```

* `DELETE /delete/{file_name}`

  * 删除文件。
  * **示例**：

    ```bash
    curl -X DELETE http://<your-node-ip>:<api-gateway-node-port>/delete/file.txt
    ```

* `POST /parse/{file_name}`

  * 手动触发 MinIO 中已存在文件的解析。
  * **示例**：

    ```bash
    curl -X POST http://<your-node-ip>:<api-gateway-node-port>/parse/test.txt
    ```

