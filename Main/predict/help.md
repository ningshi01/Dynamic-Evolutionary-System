
## 查询命令
```


# 1. QPS (按目标服务聚合，总请求速率) --- predict
sum(rate(istio_requests_total{destination_service_namespace="act-test"}[1h])) by (destination_service)


# 2. 请求延迟 (按目标服务聚合，P95延迟)
histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket[1h])) by (le, destination_service))

# 3. 错误率 (HTTP 5xx 错误率)
sum(rate(istio_requests_total{response_code=~"5.."}[1h])) by (destination_service)
/ sum(rate(istio_requests_total[1h])) by (destination_service)

# 4. 容器内存使用率
container_memory_usage_bytes{container!="", namespace="act-test"}

# 5. 容器CPU使用率 --- predict
rate(container_cpu_usage_seconds_total{container!="", namespace="act-test"}[1h])

# 6. 网络I/O (接收字节速率)
rate(container_network_receive_bytes_total{namespace="act-test"}[1h])

# 7. 网络I/O (发送字节速率)
rate(container_network_transmit_bytes_total{namespace="act-test"}[1h])

# 8. 模型平均推理延迟
rate(model_inference_latency_seconds_sum[10m]) / rate(model_inference_latency_seconds_count[10m])

# 9. GPU显存使用量
gpu_memory_usage_bytes

# 10. Milvus 查询 QPS
rate(milvus_query_duration_seconds_count[10m])

```