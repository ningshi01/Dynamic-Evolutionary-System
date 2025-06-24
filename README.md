# MinIO-based File Storage Service for Kubernetes

This project provides a complete setup for deploying a MinIO object storage server on Kubernetes, along with a Python-based file service to handle file uploads, downloads, and deletions.

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
    docker build -t crater-harbor.act.buaa.edu.cn/user-liujh24/minio-api-gateway:latest -f api.Dockerfile .
    ```

2.  **Push the API gateway image:**
    ```bash
    docker push crater-harbor.act.buaa.edu.cn/user-liujh24/minio-api-gateway:latest
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
    kubectl create secret generic hf-token-secret --from-literal=token='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' -n rag
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

### 7. Accessing the System via the API Gateway

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

- **`POST /upload`**
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
        "model": "BAAI/bge-large-en-v1.5"
      }'
    ``` 