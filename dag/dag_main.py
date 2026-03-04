from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import logging
import requests
import re
import threading
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from urllib.parse import urlencode
from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

# 全局日志使用 Asia/Shanghai 时区
_CST = timezone(timedelta(hours=8))


class _CSTFormatter(logging.Formatter):
    """Formatter that always uses CST (UTC+8) timestamps."""
    converter = None  # disable default converter

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=_CST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime('%Y-%m-%d %H:%M:%S') + f',{int(record.msecs):03d}'


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 演化策略专用日志 - 使用特殊前缀便于在 K8s 中过滤
evo_logger = logging.getLogger("evolution_strategy")
evo_logger.setLevel(logging.INFO)
if not evo_logger.handlers:
    _evo_handler = logging.StreamHandler()
    _evo_handler.setFormatter(_CSTFormatter('%(asctime)s [EVOLUTION] %(message)s'))
    evo_logger.addHandler(_evo_handler)
    evo_logger.propagate = False

# Prometheus server URL - can be configured via environment variable
PROMETHEUS_URL = os.environ.get('PROMETHEUS_URL')

# Critical path analysis interval (seconds)
CRITICAL_PATH_INTERVAL = int(os.environ.get('CRITICAL_PATH_INTERVAL', '300'))

# Resource adjustment ratio for critical path nodes
CRITICAL_PATH_RESOURCE_BOOST = float(os.environ.get('CRITICAL_PATH_RESOURCE_BOOST', '1.2'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/logical-topology')
def get_logical_topology():
    """
    Returns the logical topology of the RAG system.
    This represents the conceptual data flow, not the actual network traffic.
    Currently mocked as a standard RAG pipeline.
    """
    # Mock RAG logical topology
    # In the future, this could be user-defined or read from a config file
    
    nodes = [
        # Ingestion Pipeline
        {"data": {"id": "user-input", "label": "User Input", "type": "entry", "description": "User uploads documents"}},
        {"data": {"id": "fileservice", "label": "File Service", "type": "service", "description": "Stores and manages files"}},
        {"data": {"id": "parser", "label": "Parser", "type": "processor", "description": "Parses documents into chunks"}},
        {"data": {"id": "embedding-ingest", "label": "Embedding", "type": "processor", "description": "Converts text to vectors"}},
        {"data": {"id": "vectordb", "label": "Vector DB (Milvus)", "type": "storage", "description": "Stores document embeddings"}},
        
        # Query Pipeline
        {"data": {"id": "user-query", "label": "User Query", "type": "entry", "description": "User asks a question"}},
        {"data": {"id": "retriever", "label": "Retriever", "type": "processor", "description": "Searches relevant documents"}},
        {"data": {"id": "embedding-query", "label": "Query Embedding", "type": "processor", "description": "Embeds user query"}},
        {"data": {"id": "reranker", "label": "Reranker", "type": "processor", "description": "Reranks retrieved results"}},
        {"data": {"id": "llm", "label": "LLM", "type": "processor", "description": "Generates response"}},
        {"data": {"id": "response", "label": "Response", "type": "output", "description": "Final answer to user"}}
    ]
    
    edges = [
        # Ingestion Pipeline (Document Processing)
        {"data": {"id": "e1", "source": "user-input", "target": "fileservice", "label": "upload", "pipeline": "ingestion"}},
        {"data": {"id": "e2", "source": "fileservice", "target": "parser", "label": "fetch", "pipeline": "ingestion"}},
        {"data": {"id": "e3", "source": "parser", "target": "embedding-ingest", "label": "chunks", "pipeline": "ingestion"}},
        {"data": {"id": "e4", "source": "embedding-ingest", "target": "vectordb", "label": "store", "pipeline": "ingestion"}},
        
        # Query Pipeline (Retrieval & Generation)
        {"data": {"id": "e5", "source": "user-query", "target": "retriever", "label": "query", "pipeline": "query"}},
        {"data": {"id": "e6", "source": "retriever", "target": "embedding-query", "label": "embed", "pipeline": "query"}},
        {"data": {"id": "e7", "source": "embedding-query", "target": "vectordb", "label": "search", "pipeline": "query"}},
        {"data": {"id": "e8", "source": "vectordb", "target": "retriever", "label": "results", "pipeline": "query"}},
        {"data": {"id": "e9", "source": "retriever", "target": "reranker", "label": "candidates", "pipeline": "query"}},
        {"data": {"id": "e10", "source": "reranker", "target": "llm", "label": "top-k", "pipeline": "query"}},
        {"data": {"id": "e11", "source": "llm", "target": "response", "label": "generate", "pipeline": "query"}}
    ]
    
    return jsonify({
        "nodes": nodes,
        "edges": edges,
        "pipelines": [
            {"id": "ingestion", "name": "Document Ingestion", "color": "#3e8635"},
            {"id": "query", "name": "Query & Retrieval", "color": "#06c"}
        ]
    })


# ============================================================
# Prometheus Proxy API
# ============================================================

@app.route('/api/prometheus/query')
def prometheus_query():
    """
    Proxy for Prometheus instant query API.
    Accepts query parameter and forwards to Prometheus.
    """
    query = request.args.get('query', '')
    if not query:
        return jsonify({'status': 'error', 'error': 'Missing query parameter'}), 400
    
    try:
        params = {'query': query}
        response = requests.get(f'{PROMETHEUS_URL}/api/v1/query', params=params, timeout=30)
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        logger.error(f'Prometheus query error: {e}')
        return jsonify({'status': 'error', 'error': str(e), 'data': {'result': []}}), 500


@app.route('/api/prometheus/query_range')
def prometheus_query_range():
    """
    Proxy for Prometheus range query API.
    Accepts query, start, end, step parameters and forwards to Prometheus.
    """
    query = request.args.get('query', '')
    start = request.args.get('start', '')
    end = request.args.get('end', '')
    step = request.args.get('step', '15')
    
    if not query:
        return jsonify({'status': 'error', 'error': 'Missing query parameter'}), 400
    
    try:
        params = {
            'query': query,
            'start': start,
            'end': end,
            'step': step
        }
        response = requests.get(f'{PROMETHEUS_URL}/api/v1/query_range', params=params, timeout=30)
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        logger.error(f'Prometheus query_range error: {e}')
        return jsonify({'status': 'error', 'error': str(e), 'data': {'result': []}}), 500


@app.route('/api/prometheus/label/<label_name>/values')
def prometheus_label_values(label_name):
    """
    Proxy for Prometheus label values API.
    Returns all values for a given label name.
    """
    try:
        response = requests.get(f'{PROMETHEUS_URL}/api/v1/label/{label_name}/values', timeout=30)
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        logger.error(f'Prometheus label values error: {e}')
        return jsonify({'status': 'error', 'error': str(e), 'data': []}), 500


# ============================================================
# Kubernetes Pod Logs API - Evolution Strategy Logs
# ============================================================

_k8s_ready = False

def _init_k8s():
    global _k8s_ready
    if _k8s_ready:
        return True
    try:
        config.load_incluster_config()
        _k8s_ready = True
        return True
    except ConfigException:
        pass
    try:
        config.load_kube_config()
        _k8s_ready = True
        return True
    except ConfigException:
        logger.warning("K8s config not available for pod logs API")
        return False


@app.route('/api/logs/evolution')
def get_evolution_logs():
    """
    Fetch pod logs from all pods in the namespace and filter for Evolution Strategy lines.
    Query params:
      - namespace (default: value of NAMESPACE env or 'act-test')
      - tail_lines (default: 1000)
      - strategy (optional): filter by strategy number, e.g. '1', '2', '3'
    """
    if not _init_k8s():
        return jsonify({'status': 'error', 'error': 'Kubernetes not configured'}), 500

    namespace = request.args.get('namespace', os.environ.get('NAMESPACE', 'act-test'))
    tail_lines = int(request.args.get('tail_lines', '1000'))
    strategy_filter = request.args.get('strategy', '')

    v1 = client.CoreV1Api()
    evolution_logs = []

    try:
        pods = v1.list_namespaced_pod(namespace=namespace)
        for pod in pods.items:
            pod_name = pod.metadata.name
            # Get all containers in the pod
            containers = []
            if pod.spec.containers:
                containers = [c.name for c in pod.spec.containers]

            for container_name in containers:
                try:
                    log_text = v1.read_namespaced_pod_log(
                        name=pod_name,
                        namespace=namespace,
                        container=container_name,
                        tail_lines=tail_lines,
                    )
                    if not log_text:
                        continue

                    for line in log_text.split('\n'):
                        if 'Evolution Strategy' not in line:
                            continue
                        if strategy_filter and f'Evolution Strategy - {strategy_filter}' not in line:
                            continue

                        # Parse the log line
                        entry = {
                            'pod': pod_name,
                            'container': container_name,
                            'message': line.strip(),
                            'timestamp': '',
                            'strategy': '',
                            'description': '',
                        }

                        # Try to extract timestamp from log line
                        ts_match = re.match(r'^(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}[.,]?\d*)\s', line)
                        if ts_match:
                            entry['timestamp'] = ts_match.group(1)

                        # Extract strategy number
                        strategy_match = re.search(r'Evolution Strategy - (\d+) - (.+)', line)
                        if strategy_match:
                            entry['strategy'] = strategy_match.group(1)
                            entry['description'] = strategy_match.group(2)

                        evolution_logs.append(entry)

                except Exception as e:
                    logger.debug(f'Could not read logs for {pod_name}/{container_name}: {e}')
                    continue

    except Exception as e:
        logger.error(f'Failed to list pods: {e}')
        return jsonify({'status': 'error', 'error': str(e)}), 500

    # Sort by timestamp ascending (oldest first, newest at bottom)
    evolution_logs.sort(key=lambda x: x.get('timestamp', ''))

    return jsonify({
        'status': 'success',
        'namespace': namespace,
        'total': len(evolution_logs),
        'logs': evolution_logs
    })


# ============================================================
# Evolution Strategy 2 - 服务拓扑调整: 关键路径检测与资源倾斜
# ============================================================

# RAG 系统 DAG 拓扑定义 (logical-topology API)
RAG_DAG = {
    'nodes': {
        'user-input':       {'label': 'User Input',       'type': 'entry',     'service': None},
        'fileservice':      {'label': 'File Service',     'type': 'service',   'service': 'fileservice'},
        'parser':           {'label': 'Parser',           'type': 'processor', 'service': 'parser'},
        'embedding-ingest': {'label': 'Embedding',        'type': 'processor', 'service': 'embedding'},
        'vectordb':         {'label': 'Vector DB',        'type': 'storage',   'service': 'milvus'},
        'user-query':       {'label': 'User Query',       'type': 'entry',     'service': None},
        'retriever':        {'label': 'Retriever',        'type': 'processor', 'service': 'retriever'},
        'embedding-query':  {'label': 'Query Embedding',  'type': 'processor', 'service': 'embedding'},
        'reranker':         {'label': 'Reranker',         'type': 'processor', 'service': 'reranker'},
        'llm':              {'label': 'LLM',              'type': 'processor', 'service': 'llm'},
        'response':         {'label': 'Response',         'type': 'output',    'service': None},
    },
    'edges': [
        # Ingestion pipeline
        {'source': 'user-input',       'target': 'fileservice',      'pipeline': 'ingestion'},
        {'source': 'fileservice',      'target': 'parser',           'pipeline': 'ingestion'},
        {'source': 'parser',           'target': 'embedding-ingest', 'pipeline': 'ingestion'},
        {'source': 'embedding-ingest', 'target': 'vectordb',         'pipeline': 'ingestion'},
        # Query pipeline
        {'source': 'user-query',       'target': 'retriever',        'pipeline': 'query'},
        {'source': 'retriever',        'target': 'embedding-query',  'pipeline': 'query'},
        {'source': 'embedding-query',  'target': 'vectordb',         'pipeline': 'query'},
        {'source': 'vectordb',         'target': 'retriever',        'pipeline': 'query'},
        {'source': 'retriever',        'target': 'reranker',         'pipeline': 'query'},
        {'source': 'reranker',         'target': 'llm',              'pipeline': 'query'},
        {'source': 'llm',              'target': 'response',         'pipeline': 'query'},
    ],
    'pipelines': {
        'ingestion': ['user-input', 'fileservice', 'parser', 'embedding-ingest', 'vectordb'],
        'query':     ['user-query', 'retriever', 'embedding-query', 'vectordb', 'retriever', 'reranker', 'llm', 'response'],
    }
}

# Node-to-deployment mapping (for K8s resource adjustment)
NODE_DEPLOYMENT_MAP = {
    'fileservice':      'fileservice-deployment',
    'parser':           'parser-deployment',
    'embedding-ingest': 'embedding-deployment',
    'embedding-query':  'embedding-deployment',
    'retriever':        'retriever-deployment',
    'reranker':         'reranker-deployment',
    'llm':              'llm-deployment',
}

# DAG 节点 → K8s Service FQDN（用于 Istio DestinationRule/VirtualService）
NODE_SERVICE_HOST_MAP = {
    'fileservice':      'fileservice-service.act-test.svc.cluster.local',
    'parser':           'parser-service.act-test.svc.cluster.local',
    'embedding-ingest': 'embedding-service.act-test.svc.cluster.local',
    'embedding-query':  'embedding-service.act-test.svc.cluster.local',
    'retriever':        'retriever-service.act-test.svc.cluster.local',
    'reranker':         'reranker-service.act-test.svc.cluster.local',
    'llm':              'llm-service.act-test.svc.cluster.local',
    'vectordb':         'milvus-service.act-test.svc.cluster.local',
}


# ============================================================
# Istio 断路器 / 超时 模板 (Solution 4)
# ============================================================

def _build_destination_rule(service_host, name_prefix):
    """构建 Istio DestinationRule 断路器 CRD 字典。"""
    return {
        'apiVersion': 'networking.istio.io/v1beta1',
        'kind': 'DestinationRule',
        'metadata': {
            'name': f'{name_prefix}-circuit-breaker',
            'namespace': os.environ.get('NAMESPACE', 'act-test'),
        },
        'spec': {
            'host': service_host,
            'trafficPolicy': {
                'connectionPool': {
                    'tcp': {
                        'maxConnections': 80,
                        'connectTimeout': '10s',
                    },
                    'http': {
                        'http1MaxPendingRequests': 50,
                        'http2MaxRequests': 80,
                        'maxRequestsPerConnection': 8,
                        'maxRetries': 2,
                        'idleTimeout': '60s',
                    },
                },
                'outlierDetection': {
                    'consecutive5xxErrors': 3,
                    'interval': '10s',
                    'baseEjectionTime': '30s',
                    'maxEjectionPercent': 50,
                    'minHealthPercent': 30,
                },
            },
        },
    }


def _build_virtual_service(service_host, name_prefix, timeout='30s',
                           per_try_timeout='12s', port=80):
    """构建 Istio VirtualService 超时/重试 CRD 字典。"""
    return {
        'apiVersion': 'networking.istio.io/v1beta1',
        'kind': 'VirtualService',
        'metadata': {
            'name': f'{name_prefix}-virtualservice',
            'namespace': os.environ.get('NAMESPACE', 'act-test'),
        },
        'spec': {
            'hosts': [service_host],
            'http': [{
                'name': f'{name_prefix}-route',
                'timeout': timeout,
                'retries': {
                    'attempts': 2,
                    'perTryTimeout': per_try_timeout,
                    'retryOn': '5xx,reset,connect-failure',
                },
                'route': [{
                    'destination': {
                        'host': service_host,
                        'port': {'number': port},
                    },
                }],
            }],
        },
    }


_ISTIO_CRD_PLURAL_MAP = {
    'DestinationRule': 'destinationrules',
    'VirtualService': 'virtualservices',
    'EnvoyFilter': 'envoyfilters',
}

# 动态 CRD 标识标签 — 只有带此标签的 CRD 才会被自动清理
# LLM / Gateway 等手动部署的 CRD 不携带此标签, 因此不会被误删
_DYNAMIC_CRD_LABEL = {'rag-evolution/managed-by': 'dag-strategy-2'}
_DYNAMIC_CRD_LABEL_SELECTOR = 'rag-evolution/managed-by=dag-strategy-2'


def _apply_istio_crd(crd_body, kind):
    """
    通过 K8s CustomObject API 创建或更新 Istio CRD。
    kind: 'DestinationRule' | 'VirtualService' | 'EnvoyFilter'
    自动注入 _DYNAMIC_CRD_LABEL 以标识动态创建的资源。
    """
    # 注入动态标签, 确保清理时可按标签选择
    metadata = crd_body.setdefault('metadata', {})
    labels = metadata.setdefault('labels', {})
    labels.update(_DYNAMIC_CRD_LABEL)

    namespace = crd_body['metadata']['namespace']
    name = crd_body['metadata']['name']
    group = 'networking.istio.io'
    version = 'v1alpha3' if kind == 'EnvoyFilter' else 'v1beta1'
    plural = _ISTIO_CRD_PLURAL_MAP.get(kind, kind.lower() + 's')

    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()

    api = client.CustomObjectsApi()
    try:
        # 尝试获取已有资源
        api.get_namespaced_custom_object(group, version, namespace, plural, name)
        # 存在 → 替换
        api.replace_namespaced_custom_object(
            group, version, namespace, plural, name, crd_body
        )
        evo_logger.info(f'Evolution Strategy - 2 - 已更新Istio {kind} "{name}" (namespace={namespace})')
    except client.exceptions.ApiException as e:
        if e.status == 404:
            # 不存在 → 创建
            api.create_namespaced_custom_object(
                group, version, namespace, plural, crd_body
            )
            evo_logger.info(f'Evolution Strategy - 2 - 已创建Istio {kind} "{name}" (namespace={namespace})')
        else:
            raise


def _delete_istio_crd(name, kind, namespace):
    """
    通过 K8s CustomObject API 删除单个 Istio CRD。
    """
    group = 'networking.istio.io'
    version = 'v1alpha3' if kind == 'EnvoyFilter' else 'v1beta1'
    plural = _ISTIO_CRD_PLURAL_MAP.get(kind, kind.lower() + 's')

    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()

    api = client.CustomObjectsApi()
    try:
        api.delete_namespaced_custom_object(group, version, namespace, plural, name)
        evo_logger.info(
            f'Evolution Strategy - 2 - 已清理Istio {kind} "{name}" (namespace={namespace})'
        )
        return True
    except client.exceptions.ApiException as e:
        if e.status == 404:
            # 已不存在, 无需删除
            return False
        else:
            evo_logger.warning(
                f'Evolution Strategy - 2 - 清理Istio {kind} "{name}" 失败: {e.reason}'
            )
            raise


def cleanup_dynamic_crds(namespace):
    """
    清理所有由本程序动态创建的 Istio CRD (带 _DYNAMIC_CRD_LABEL 标签)。
    手动部署的 CRD (如 llm-destination, llm-virtualservice, gateway 等) 不携带
    该标签, 不会被清理。
    """
    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()

    api = client.CustomObjectsApi()
    cleaned = []

    for kind, plural in _ISTIO_CRD_PLURAL_MAP.items():
        group = 'networking.istio.io'
        version = 'v1alpha3' if kind == 'EnvoyFilter' else 'v1beta1'

        try:
            result = api.list_namespaced_custom_object(
                group, version, namespace, plural,
                label_selector=_DYNAMIC_CRD_LABEL_SELECTOR,
            )
            items = result.get('items', [])
            for item in items:
                item_name = item['metadata']['name']
                try:
                    api.delete_namespaced_custom_object(
                        group, version, namespace, plural, item_name
                    )
                    cleaned.append(f'{kind}/{item_name}')
                except client.exceptions.ApiException as e:
                    if e.status != 404:
                        evo_logger.warning(
                            f'Evolution Strategy - 2 - 清理 {kind}/{item_name} 失败: {e.reason}'
                        )
        except Exception as e:
            evo_logger.warning(
                f'Evolution Strategy - 2 - 列举 {plural} 失败: {e}'
            )

    if cleaned:
        evo_logger.info(
            f'Evolution Strategy - 2 - 动态CRD清理完成, '
            f'共清理{len(cleaned)}个资源: | ' + ' | '.join(cleaned)
        )
    else:
        evo_logger.info('Evolution Strategy - 2 - 动态CRD清理: 无需清理 (无动态创建的资源)')

    return cleaned


def apply_circuit_breakers_for_cycle(cycle_nodes):
    """
    针对一组环路节点, 自动创建/更新 Istio 断路器 DestinationRule 与 VirtualService。
    cycle_nodes: 环路中涉及的 DAG 节点 ID 列表
    """
    namespace = os.environ.get('NAMESPACE', 'act-test')

    # 去重: 同一个 service host 只配置一次
    applied_hosts = set()
    applied_names = []

    for node_id in cycle_nodes:
        host = NODE_SERVICE_HOST_MAP.get(node_id)
        if not host or host in applied_hosts:
            continue
        applied_hosts.add(host)

        # 从 host 中取简短名
        name_prefix = host.split('.')[0].replace('-service', '')

        # DestinationRule (断路器)
        dr = _build_destination_rule(host, name_prefix)
        _apply_istio_crd(dr, 'DestinationRule')

        # VirtualService (超时/重试)
        vs = _build_virtual_service(host, name_prefix)
        _apply_istio_crd(vs, 'VirtualService')

        applied_names.append(name_prefix)

    return applied_names


def _query_prometheus_instant(query):
    """Helper to query Prometheus instant API."""
    if not PROMETHEUS_URL:
        return None
    try:
        resp = requests.get(
            f'{PROMETHEUS_URL}/api/v1/query',
            params={'query': query},
            timeout=15,
        )
        data = resp.json()
        if data.get('status') == 'success':
            return data.get('data', {}).get('result', [])
    except Exception as e:
        logger.debug(f'Prometheus query failed: {e}')
    return None


def _get_edge_latency(source_svc, target_svc, namespace):
    """Get P95 latency (ms) for traffic from source to target service."""
    query = (
        f'histogram_quantile(0.95, sum(rate('
        f'istio_request_duration_milliseconds_bucket{{'
        f'source_app=~".*{source_svc}.*",'
        f'destination_app=~".*{target_svc}.*",'
        f'reporter="source",'
        f'destination_service_namespace="{namespace}"'
        f'}}[5m])) by (le))'
    )
    result = _query_prometheus_instant(query)
    if result and len(result) > 0:
        val = float(result[0].get('value', [0, 0])[1])
        return val if not (val != val) else 0.0  # NaN check
    return 0.0


def _get_node_qps(service_name, namespace):
    """Get QPS for a given service."""
    query = (
        f'sum(rate(istio_requests_total{{'
        f'destination_app=~".*{service_name}.*",'
        f'destination_service_namespace="{namespace}"'
        f'}}[5m]))'
    )
    result = _query_prometheus_instant(query)
    if result and len(result) > 0:
        val = float(result[0].get('value', [0, 0])[1])
        return val if not (val != val) else 0.0
    return 0.0


def _get_node_cpu_usage(service_name, namespace):
    """Get CPU usage rate for pods matching a service name."""
    query = (
        f'sum(rate(container_cpu_usage_seconds_total{{'
        f'namespace="{namespace}",'
        f'container!="",'
        f'pod=~".*{service_name}.*"'
        f'}}[5m]))'
    )
    result = _query_prometheus_instant(query)
    if result and len(result) > 0:
        val = float(result[0].get('value', [0, 0])[1])
        return val if not (val != val) else 0.0
    return 0.0


def _get_node_error_rate(service_name, namespace):
    """Get 5xx error rate for a given service."""
    total_query = (
        f'sum(rate(istio_requests_total{{'
        f'destination_app=~".*{service_name}.*",'
        f'destination_service_namespace="{namespace}"'
        f'}}[5m]))'
    )
    error_query = (
        f'sum(rate(istio_requests_total{{'
        f'destination_app=~".*{service_name}.*",'
        f'destination_service_namespace="{namespace}",'
        f'response_code=~"5.."'
        f'}}[5m]))'
    )
    total_result = _query_prometheus_instant(total_query)
    error_result = _query_prometheus_instant(error_query)

    total = 0.0
    errors = 0.0
    if total_result and len(total_result) > 0:
        total = float(total_result[0].get('value', [0, 0])[1])
    if error_result and len(error_result) > 0:
        errors = float(error_result[0].get('value', [0, 0])[1])

    if total > 0:
        return errors / total
    return 0.0


def compute_critical_paths(namespace):
    """
    Compute critical paths in the RAG DAG by weighting edges with latency
    and nodes with QPS/CPU metrics from Prometheus.
    Returns a dict with critical path info and per-node scores.
    """
    evo_logger.info(f'Evolution Strategy - 2 - 开始关键路径分析, namespace={namespace}')

    # Step 1: Collect per-node metrics
    node_metrics = {}
    for node_id, node_info in RAG_DAG['nodes'].items():
        svc = node_info.get('service')
        if not svc:
            node_metrics[node_id] = {'qps': 0, 'cpu': 0, 'error_rate': 0, 'score': 0}
            continue
        qps = _get_node_qps(svc, namespace)
        cpu = _get_node_cpu_usage(svc, namespace)
        err = _get_node_error_rate(svc, namespace)
        # Composite load score: weighted combination
        score = qps * 0.4 + cpu * 100 * 0.4 + err * 100 * 0.2
        node_metrics[node_id] = {
            'qps': round(qps, 4),
            'cpu': round(cpu, 4),
            'error_rate': round(err, 4),
            'score': round(score, 4),
        }

    # 汇总输出节点指标
    active_nodes = [
        f'{RAG_DAG["nodes"][nid]["label"]}({RAG_DAG["nodes"][nid]["service"]}): '
        f'QPS={nm["qps"]}, CPU={nm["cpu"]}, Err={nm["error_rate"]}, Score={nm["score"]}'
        for nid, nm in node_metrics.items()
        if RAG_DAG['nodes'][nid].get('service') and (nm['qps'] > 0 or nm['cpu'] > 0)
    ]
    if active_nodes:
        evo_logger.info(
            f'Evolution Strategy - 2 - 节点指标采集完成({len(active_nodes)}个活跃节点): | '
            + ' | '.join(active_nodes)
        )

    # Step 2: Collect per-edge latency
    edge_weights = []
    for edge in RAG_DAG['edges']:
        src_svc = RAG_DAG['nodes'][edge['source']].get('service')
        tgt_svc = RAG_DAG['nodes'][edge['target']].get('service')
        latency = 0.0
        if src_svc and tgt_svc:
            latency = _get_edge_latency(src_svc, tgt_svc, namespace)
        edge_weights.append({
            'source': edge['source'],
            'target': edge['target'],
            'pipeline': edge['pipeline'],
            'latency_ms': round(latency, 2),
        })

    # Step 3: Compute critical path per pipeline (longest latency path)
    critical_paths = {}
    for pipeline_name, pipeline_nodes in RAG_DAG['pipelines'].items():
        # Deduplicate while preserving order
        seen = set()
        ordered_nodes = []
        for n in pipeline_nodes:
            if n not in seen:
                seen.add(n)
                ordered_nodes.append(n)

        # Sum node scores + edge latencies along the pipeline
        path_total_latency = 0.0
        path_total_score = 0.0
        path_details = []

        for i, node_id in enumerate(ordered_nodes):
            n_score = node_metrics.get(node_id, {}).get('score', 0)
            path_total_score += n_score

            # Find edge to next node
            edge_lat = 0.0
            if i < len(ordered_nodes) - 1:
                next_node = ordered_nodes[i + 1]
                for ew in edge_weights:
                    if ew['source'] == node_id and ew['target'] == next_node:
                        edge_lat = ew['latency_ms']
                        break
            path_total_latency += edge_lat

            path_details.append({
                'node': node_id,
                'label': RAG_DAG['nodes'][node_id]['label'],
                'node_score': n_score,
                'edge_latency_to_next': edge_lat,
            })

        critical_paths[pipeline_name] = {
            'nodes': ordered_nodes,
            'total_latency_ms': round(path_total_latency, 2),
            'total_load_score': round(path_total_score, 4),
            'details': path_details,
        }

    # 汇总输出 pipeline 指标
    pipeline_summary = ' | '.join(
        f'{pn}: 延迟={cp["total_latency_ms"]}ms, 负载={cp["total_load_score"]}'
        for pn, cp in critical_paths.items()
    )
    evo_logger.info(f'Evolution Strategy - 2 - Pipeline分析完成: | {pipeline_summary}')

    # Step 4: Identify bottleneck nodes (top load score nodes across all pipelines)
    all_svc_nodes = [
        (nid, nm) for nid, nm in node_metrics.items()
        if RAG_DAG['nodes'][nid].get('service') and nm['score'] > 0
    ]
    all_svc_nodes.sort(key=lambda x: x[1]['score'], reverse=True)

    # Top nodes that account for >60% of total score or top 3
    total_score = sum(nm['score'] for _, nm in all_svc_nodes) or 1
    bottleneck_nodes = []
    cumulative = 0
    for nid, nm in all_svc_nodes:
        bottleneck_nodes.append(nid)
        cumulative += nm['score']
        if cumulative / total_score >= 0.6 or len(bottleneck_nodes) >= 3:
            break

    if bottleneck_nodes:
        names = ', '.join(
            f'{RAG_DAG["nodes"][n]["label"]}(score={node_metrics[n]["score"]})'
            for n in bottleneck_nodes
        )
        evo_logger.info(f'Evolution Strategy - 2 - 检测到高负载瓶颈节点: | {names}')
    else:
        evo_logger.info('Evolution Strategy - 2 - 当前无明显高负载瓶颈节点')

    # Step 5: Determine which pipeline is the critical (heaviest) pipeline
    heaviest_pipeline = max(
        critical_paths.items(),
        key=lambda x: x[1]['total_latency_ms'] + x[1]['total_load_score'],
        default=(None, None)
    )
    if heaviest_pipeline[0]:
        evo_logger.info(
            f'Evolution Strategy - 2 - 关键路径: [{heaviest_pipeline[0]}] pipeline, '
            f'延迟={heaviest_pipeline[1]["total_latency_ms"]}ms, '
            f'负载={heaviest_pipeline[1]["total_load_score"]}'
        )

    return {
        'node_metrics': node_metrics,
        'edge_weights': edge_weights,
        'critical_paths': critical_paths,
        'bottleneck_nodes': bottleneck_nodes,
        'heaviest_pipeline': heaviest_pipeline[0] if heaviest_pipeline[0] else None,
    }


def apply_resource_tilt(bottleneck_nodes, namespace):
    """
    Apply resource tilt (increase CPU requests) for bottleneck nodes on the
    critical path by patching their K8s deployments.
    """
    if not _init_k8s():
        evo_logger.info('Evolution Strategy - 2 - 跳过资源倾斜: Kubernetes 配置不可用')
        return []

    apps_v1 = client.AppsV1Api()
    adjustments = []

    for node_id in bottleneck_nodes:
        deployment_name = NODE_DEPLOYMENT_MAP.get(node_id)
        if not deployment_name:
            continue

        try:
            deploy = apps_v1.read_namespaced_deployment(deployment_name, namespace)
        except Exception as e:
            logger.debug(f'Cannot read deployment {deployment_name}: {e}')
            continue

        containers = deploy.spec.template.spec.containers or []
        patched = False
        for container in containers:
            resources = container.resources
            if not resources or not resources.requests:
                continue

            cpu_req = resources.requests.get('cpu', '')
            if not cpu_req:
                continue

            # Parse current CPU request
            try:
                from kubernetes.utils.quantity import parse_quantity
                current_cores = float(parse_quantity(cpu_req))
            except Exception:
                continue

            new_cores = round(current_cores * CRITICAL_PATH_RESOURCE_BOOST, 3)
            new_cpu_str = f'{int(new_cores * 1000)}m'

            evo_logger.info(
                f'Evolution Strategy - 2 - 资源倾斜: {deployment_name} '
                f'CPU request {cpu_req} -> {new_cpu_str} (boost={CRITICAL_PATH_RESOURCE_BOOST}x)'
            )

            # Patch deployment
            container.resources.requests['cpu'] = new_cpu_str
            # Also adjust limits if set
            if resources.limits and resources.limits.get('cpu'):
                try:
                    current_limit = float(parse_quantity(resources.limits['cpu']))
                    new_limit = round(current_limit * CRITICAL_PATH_RESOURCE_BOOST, 3)
                    resources.limits['cpu'] = f'{int(new_limit * 1000)}m'
                except Exception:
                    pass
            patched = True

        if patched:
            try:
                apps_v1.patch_namespaced_deployment(deployment_name, namespace, deploy)
                adjustments.append({
                    'deployment': deployment_name,
                    'node': node_id,
                    'action': 'cpu_boost',
                    'boost_ratio': CRITICAL_PATH_RESOURCE_BOOST,
                })
                evo_logger.info(
                    f'Evolution Strategy - 2 - 已成功应用资源倾斜: {deployment_name}'
                )
            except Exception as e:
                evo_logger.info(
                    f'Evolution Strategy - 2 - 资源倾斜应用失败 {deployment_name}: {e}'
                )

    return adjustments


@app.route('/api/critical-path/analyze')
def api_critical_path_analyze():
    """
    API endpoint to trigger critical path analysis.
    Query params:
      - namespace (default: env NAMESPACE or 'act-test')
      - apply (default: false) - whether to apply resource adjustment
    """
    namespace = request.args.get('namespace', os.environ.get('NAMESPACE', 'act-test'))
    apply_adjustment = request.args.get('apply', 'false').lower() in ('true', '1', 'yes')

    try:
        result = compute_critical_paths(namespace)

        adjustments = []
        if apply_adjustment and result['bottleneck_nodes']:
            adjustments = apply_resource_tilt(result['bottleneck_nodes'], namespace)

        evo_logger.info(
            f'Evolution Strategy - 2 - 关键路径分析完成, '
            f'瓶颈节点数={len(result["bottleneck_nodes"])}, '
            f'资源调整数={len(adjustments)}'
        )

        return jsonify({
            'status': 'success',
            'namespace': namespace,
            'node_metrics': result['node_metrics'],
            'edge_weights': result['edge_weights'],
            'critical_paths': result['critical_paths'],
            'bottleneck_nodes': result['bottleneck_nodes'],
            'heaviest_pipeline': result['heaviest_pipeline'],
            'adjustments': adjustments,
        })
    except Exception as e:
        logger.error(f'Critical path analysis failed: {e}')
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ============================================================
# Evolution Strategy 2 - 服务拓扑调整: 过长链路检测与聚合
# ============================================================

# 链路长度阈值 (超过此跳数视为"过长")
CHAIN_LENGTH_THRESHOLD = int(os.environ.get('CHAIN_LENGTH_THRESHOLD', '5'))


def detect_long_chains():
    """
    检测 RAG_DAG 各 pipeline 中的链路长度。
    返回每条 pipeline 的跳数、各段延迟、以及超阈值的链路列表。
    """
    chains = []
    for pipeline_name, pipeline_nodes in RAG_DAG['pipelines'].items():
        # 计算实际跳数 (service 节点之间)
        svc_nodes = [n for n in pipeline_nodes if RAG_DAG['nodes'][n].get('service')]
        hop_count = len(pipeline_nodes) - 1  # 总边数

        # 构建跳数明细
        hops = []
        for i in range(len(pipeline_nodes) - 1):
            src = pipeline_nodes[i]
            tgt = pipeline_nodes[i + 1]
            hops.append({
                'from': src,
                'to': tgt,
                'from_label': RAG_DAG['nodes'][src]['label'],
                'to_label': RAG_DAG['nodes'][tgt]['label'],
            })

        chain_info = {
            'pipeline': pipeline_name,
            'total_hops': hop_count,
            'service_nodes': len(svc_nodes),
            'nodes': pipeline_nodes,
            'hops': hops,
            'is_long': hop_count > CHAIN_LENGTH_THRESHOLD,
        }
        chains.append(chain_info)

    return chains


def _identify_mergeable_segments(chain_info, namespace):
    """
    在一条过长链路中识别可聚合/可合并的相邻服务段。
    基于以下条件判定:
      1. 相邻服务之间延迟极低 (< 5ms) → 说明耦合紧密, 适合合并部署
      2. 相邻服务属于同一功能类别 (如两个 processor) → 适合请求聚合
      3. 存在 A→B→C 且 B 仅被 A 和 C 调用 → B 是纯中间节点, 适合被吸收
    """
    hops = chain_info['hops']
    segments = []

    for i, hop in enumerate(hops):
        src_node = hop['from']
        tgt_node = hop['to']
        src_info = RAG_DAG['nodes'][src_node]
        tgt_info = RAG_DAG['nodes'][tgt_node]

        # 条件 1: 同类 processor 相邻
        if src_info.get('type') == tgt_info.get('type') == 'processor':
            segments.append({
                'type': 'same_type_adjacent',
                'nodes': [src_node, tgt_node],
                'labels': [src_info['label'], tgt_info['label']],
                'reason': f'{src_info["label"]} 与 {tgt_info["label"]} 均为 processor 类型, 可考虑请求聚合',
            })

        # 条件 2: 中间节点仅有单入单出
        if i > 0 and i < len(hops) - 1:
            # 检查 tgt_node 的入度和出度
            in_degree = sum(1 for e in RAG_DAG['edges'] if e['target'] == tgt_node)
            out_degree = sum(1 for e in RAG_DAG['edges'] if e['source'] == tgt_node)
            if in_degree == 1 and out_degree == 1 and tgt_info.get('service'):
                segments.append({
                    'type': 'pass_through_node',
                    'nodes': [src_node, tgt_node, hops[i + 1]['to'] if i + 1 < len(hops) else tgt_node],
                    'labels': [src_info['label'], tgt_info['label'],
                               RAG_DAG['nodes'][hops[i + 1]['to']]['label'] if i + 1 < len(hops) else ''],
                    'reason': f'{tgt_info["label"]} 为单入单出中间节点, 可被上下游吸收以减少一跳',
                })

    return segments


def _get_chain_latencies(chain_info, namespace):
    """获取链路中各段边的延迟(从 Prometheus)。"""
    edge_latencies = []
    for hop in chain_info['hops']:
        src_svc = RAG_DAG['nodes'][hop['from']].get('service')
        tgt_svc = RAG_DAG['nodes'][hop['to']].get('service')
        latency = 0.0
        if src_svc and tgt_svc:
            latency = _get_edge_latency(src_svc, tgt_svc, namespace)
        edge_latencies.append({
            'from': hop['from_label'],
            'to': hop['to_label'],
            'latency_ms': round(latency, 2),
        })
    return edge_latencies


def _build_envoy_filter_call_depth(namespace, max_depth):
    """
    构建 EnvoyFilter CRD: 在网格内所有 sidecar 注入 Lua 脚本,
    通过 x-call-depth header 追踪调用深度。
    当深度超过 max_depth 时返回 HTTP 508 Loop Detected。
    """
    lua_code = f'''
function envoy_on_request(request_handle)
  local depth_str = request_handle:headers():get("x-call-depth") or "0"
  local depth = tonumber(depth_str) or 0
  depth = depth + 1
  request_handle:headers():replace("x-call-depth", tostring(depth))

  if depth > {max_depth} then
    request_handle:logWarn("[ChainGuard] call depth " .. tostring(depth) .. " exceeds max {max_depth}, short-circuiting")
    request_handle:respond(
      {{[":status"] = "508"}},
      "Call chain too deep (depth=" .. tostring(depth) .. ", max={max_depth}). Request short-circuited by EnvoyFilter."
    )
  end
end

function envoy_on_response(response_handle)
  local depth_str = response_handle:headers():get("x-call-depth")
  if depth_str then
    response_handle:headers():replace("x-chain-depth-trace", depth_str)
  end
end
'''
    return {
        'apiVersion': 'networking.istio.io/v1alpha3',
        'kind': 'EnvoyFilter',
        'metadata': {
            'name': 'chain-depth-guard',
            'namespace': namespace,
        },
        'spec': {
            'configPatches': [{
                'applyTo': 'HTTP_FILTER',
                'match': {
                    'context': 'SIDECAR_INBOUND',
                    'listener': {
                        'filterChain': {
                            'filter': {
                                'name': 'envoy.filters.network.http_connection_manager',
                                'subFilter': {
                                    'name': 'envoy.filters.http.router',
                                },
                            },
                        },
                    },
                },
                'patch': {
                    'operation': 'INSERT_BEFORE',
                    'value': {
                        'name': 'envoy.lua.chain_depth_guard',
                        'typed_config': {
                            '@type': 'type.googleapis.com/envoy.extensions.filters.http.lua.v3.Lua',
                            'inlineCode': lua_code.strip(),
                        },
                    },
                },
            }],
        },
    }


def _build_chain_aggregation_virtual_service(namespace):
    """
    构建请求聚合 VirtualService: 当请求携带 x-call-depth >= 阈值 时,
    将后续调用路由到 retriever-service 的聚合端点 /aggregate,
    由 Retriever 一站式完成 embedding + vectordb + rerank 的聚合调用。
    """
    threshold = CHAIN_LENGTH_THRESHOLD
    return {
        'apiVersion': 'networking.istio.io/v1beta1',
        'kind': 'VirtualService',
        'metadata': {
            'name': 'chain-aggregation-routing',
            'namespace': namespace,
        },
        'spec': {
            'hosts': [
                'embedding-service.act-test.svc.cluster.local',
                'milvus-service.act-test.svc.cluster.local',
                'reranker-service.act-test.svc.cluster.local',
            ],
            'http': [
                # 当调用深度 >= 阈值时, 将请求重定向到 retriever 聚合端点
                {
                    'name': 'deep-chain-redirect',
                    'match': [{
                        'headers': {
                            'x-call-depth': {
                                'regex': f'[{threshold}-9]|[1-9][0-9]+',
                            },
                        },
                    }],
                    'timeout': '45s',
                    'retries': {
                        'attempts': 1,
                        'perTryTimeout': '40s',
                        'retryOn': '5xx,reset',
                    },
                    'rewrite': {
                        'uri': '/aggregate',
                    },
                    'route': [{
                        'destination': {
                            'host': 'retriever-service.act-test.svc.cluster.local',
                            'port': {'number': 80},
                        },
                    }],
                },
                # 默认路由: 正常转发
                {
                    'name': 'default',
                    'route': [{
                        'destination': {
                            'host': 'embedding-service.act-test.svc.cluster.local',
                            'port': {'number': 80},
                        },
                    }],
                },
            ],
        },
    }


def analyze_long_chains():
    """
    检测 DAG 中的过长链路, 识别可聚合段, 输出演化日志并返回分析结果。
    """
    namespace = os.environ.get('NAMESPACE', 'act-test')
    evo_logger.info(
        f'Evolution Strategy - 2 - 开始过长链路检测, 阈值={CHAIN_LENGTH_THRESHOLD}跳'
    )

    chains = detect_long_chains()
    long_chains = [c for c in chains if c['is_long']]

    if not long_chains:
        evo_logger.info(
            'Evolution Strategy - 2 - 链路检测完成: '
            + ' | '.join(
                f'{c["pipeline"]}: {c["total_hops"]}跳(未超阈值)'
                for c in chains
            )
        )
        return {
            'has_long_chains': False,
            'chains': chains,
            'long_chains': [],
            'suggestions': [],
            'envoy_filter_applied': False,
            'aggregation_routing_applied': False,
        }

    # 获取各段延迟
    for chain in long_chains:
        chain['edge_latencies'] = _get_chain_latencies(chain, namespace)
        chain['mergeable_segments'] = _identify_mergeable_segments(chain, namespace)

    # 汇总日志
    chain_summaries = []
    for c in long_chains:
        path_labels = [RAG_DAG['nodes'][n]['label'] for n in c['nodes']]
        seg_count = len(c.get('mergeable_segments', []))
        chain_summaries.append(
            f'{c["pipeline"]}: {c["total_hops"]}跳 '
            f'({" -> ".join(path_labels)}), '
            f'可聚合段={seg_count}'
        )

    evo_logger.info(
        f'Evolution Strategy - 2 - 检测到{len(long_chains)}条过长链路: | '
        + ' | '.join(chain_summaries)
    )

    # 可聚合段详情
    all_segments = []
    for c in long_chains:
        for seg in c.get('mergeable_segments', []):
            all_segments.append(seg)

    if all_segments:
        seg_descs = [s['reason'] for s in all_segments]
        evo_logger.info(
            f'Evolution Strategy - 2 - 可聚合段分析({len(all_segments)}个): | '
            + ' | '.join(seg_descs)
        )

    # 执行链路聚合方案: EnvoyFilter 调用深度守卫 + VirtualService 聚合路由
    envoy_filter_applied = False
    aggregation_routing_applied = False

    try:
        # 1. 部署 EnvoyFilter (调用深度追踪 + 短路)
        ef = _build_envoy_filter_call_depth(namespace, CHAIN_LENGTH_THRESHOLD)
        _apply_istio_crd(ef, 'EnvoyFilter')
        envoy_filter_applied = True
        evo_logger.info(
            f'Evolution Strategy - 2 - 已部署EnvoyFilter调用深度守卫, '
            f'最大深度={CHAIN_LENGTH_THRESHOLD}, 超限请求将返回HTTP 508短路'
        )
    except Exception as e:
        evo_logger.warning(
            f'Evolution Strategy - 2 - EnvoyFilter部署失败: {e}'
        )

    try:
        # 2. 部署聚合路由 VirtualService
        vs = _build_chain_aggregation_virtual_service(namespace)
        _apply_istio_crd(vs, 'VirtualService')
        aggregation_routing_applied = True
        evo_logger.info(
            f'Evolution Strategy - 2 - 已部署链路聚合VirtualService路由, '
            f'深度>={CHAIN_LENGTH_THRESHOLD}的请求将重定向至retriever聚合端点'
        )
    except Exception as e:
        evo_logger.warning(
            f'Evolution Strategy - 2 - 聚合路由部署失败: {e}'
        )

    return {
        'has_long_chains': True,
        'chains': chains,
        'long_chains': [
            {
                'pipeline': c['pipeline'],
                'total_hops': c['total_hops'],
                'service_nodes': c['service_nodes'],
                'path': ' -> '.join(RAG_DAG['nodes'][n]['label'] for n in c['nodes']),
                'edge_latencies': c.get('edge_latencies', []),
                'mergeable_segments': c.get('mergeable_segments', []),
            }
            for c in long_chains
        ],
        'envoy_filter_applied': envoy_filter_applied,
        'aggregation_routing_applied': aggregation_routing_applied,
    }


@app.route('/api/chain-analysis/detect')
def api_detect_long_chains():
    """API endpoint: 过长链路检测与聚合分析。"""
    try:
        result = analyze_long_chains()
        return jsonify({'status': 'success', **result})
    except Exception as e:
        evo_logger.warning(f'Evolution Strategy - 2 - 链路检测失败: {e}')
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/chain-analysis/apply-aggregation', methods=['POST'])
def api_apply_chain_aggregation():
    """手动触发: 部署 EnvoyFilter + 聚合路由。"""
    namespace = os.environ.get('NAMESPACE', 'act-test')
    results = {'envoy_filter': False, 'aggregation_routing': False}

    try:
        ef = _build_envoy_filter_call_depth(namespace, CHAIN_LENGTH_THRESHOLD)
        _apply_istio_crd(ef, 'EnvoyFilter')
        results['envoy_filter'] = True
    except Exception as e:
        evo_logger.warning(f'Evolution Strategy - 2 - EnvoyFilter部署失败: {e}')

    try:
        vs = _build_chain_aggregation_virtual_service(namespace)
        _apply_istio_crd(vs, 'VirtualService')
        results['aggregation_routing'] = True
    except Exception as e:
        evo_logger.warning(f'Evolution Strategy - 2 - 聚合路由部署失败: {e}')

    if results['envoy_filter'] or results['aggregation_routing']:
        evo_logger.info(
            f'Evolution Strategy - 2 - 手动触发链路聚合完成: '
            f'EnvoyFilter={"已部署" if results["envoy_filter"] else "失败"}, '
            f'聚合路由={"已部署" if results["aggregation_routing"] else "失败"}'
        )

    return jsonify({'status': 'success', **results})


# ============================================================
# Evolution Strategy 2 - 服务拓扑调整: DAG 环路检测
# ============================================================

# 环路解决方案: Istio (DestinationRule + VirtualService)


def detect_dag_cycles():
    """
    使用 DFS 检测 RAG_DAG 中的所有环路。
    返回检测到的环路列表, 每个环路为节点 ID 序列。
    """
    # 构建邻接表
    adj = defaultdict(list)
    for edge in RAG_DAG['edges']:
        adj[edge['source']].append(edge['target'])

    all_nodes = list(RAG_DAG['nodes'].keys())
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in all_nodes}
    parent = {n: None for n in all_nodes}
    cycles = []

    def dfs(u, path):
        color[u] = GRAY
        path.append(u)
        for v in adj.get(u, []):
            if color[v] == GRAY:
                # 发现回边 -> 提取环
                cycle_start = path.index(v)
                cycle = path[cycle_start:] + [v]
                cycles.append(cycle)
            elif color[v] == WHITE:
                parent[v] = u
                dfs(v, path)
        path.pop()
        color[u] = BLACK

    for node in all_nodes:
        if color[node] == WHITE:
            dfs(node, [])

    return cycles


def analyze_dag_cycles():
    """
    执行 DAG 环路检测, 输出演化日志并返回结果。
    """
    evo_logger.info('Evolution Strategy - 2 - 开始DAG环路检测')

    cycles = detect_dag_cycles()

    if not cycles:
        evo_logger.info('Evolution Strategy - 2 - DAG环路检测完成: 未检测到环路, 拓扑结构无环')
        return {
            'has_cycles': False,
            'cycles': [],
            'risk_level': 'safe',
            'suggestions': [],
        }

    # 去重（同一个环可能从不同起点被检测到）
    unique_cycles = []
    seen_sets = []
    for cycle in cycles:
        edge_set = frozenset(
            (cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)
        )
        if edge_set not in seen_sets:
            seen_sets.append(edge_set)
            unique_cycles.append(cycle)

    # 风险等级判定
    if len(unique_cycles) >= 3:
        risk_level = 'high'
    elif len(unique_cycles) >= 2:
        risk_level = 'medium'
    else:
        risk_level = 'low'

    # 汇总输出环路信息
    cycle_descriptions = []
    for i, cycle in enumerate(unique_cycles, 1):
        labels = [RAG_DAG['nodes'][n]['label'] for n in cycle]
        path_str = ' -> '.join(labels)
        cycle_descriptions.append(f'环路{i}: {path_str}')

    evo_logger.info(
        f'Evolution Strategy - 2 - DAG环路检测完成: 检测到{len(unique_cycles)}个环路, '
        f'风险等级={risk_level}: | ' + ' | '.join(cycle_descriptions)
    )

    # 执行环路解决方案: 为环路服务配置断路器 (DestinationRule + VirtualService)
    circuit_breaker_applied = []
    try:
        # 收集所有环路涉及的节点
        all_cycle_nodes = set()
        for cycle in unique_cycles:
            all_cycle_nodes.update(cycle)
        # 移除 entry/output 类型节点（无实际 service）
        svc_nodes = [n for n in all_cycle_nodes if RAG_DAG['nodes'][n].get('service')]
        applied = apply_circuit_breakers_for_cycle(svc_nodes)
        circuit_breaker_applied = applied
        if applied:
            evo_logger.info(
                f'Evolution Strategy - 2 - 已为环路服务配置Istio断路器(连接池+熔断+超时重试): '
                + ', '.join(applied)
            )
    except Exception as e:
        evo_logger.warning(
            f'Evolution Strategy - 2 - 断路器配置失败: {e}'
        )

    return {
        'has_cycles': True,
        'cycle_count': len(unique_cycles),
        'cycles': [
            {
                'nodes': cycle,
                'labels': [RAG_DAG['nodes'][n]['label'] for n in cycle],
                'path': ' -> '.join(RAG_DAG['nodes'][n]['label'] for n in cycle),
            }
            for cycle in unique_cycles
        ],
        'risk_level': risk_level,
        'circuit_breaker_applied': circuit_breaker_applied,
    }


@app.route('/api/dag-cycles/detect')
def api_detect_dag_cycles():
    """
    API endpoint to detect cycles in the RAG DAG topology.
    """
    try:
        result = analyze_dag_cycles()
        return jsonify({'status': 'success', **result})
    except Exception as e:
        evo_logger.warning(f'Evolution Strategy - 2 - DAG环路检测失败: {e}')
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/dag-cycles/apply-circuit-breaker', methods=['POST'])
def api_apply_circuit_breaker():
    """
    手动触发: 为所有检测到的环路服务配置 Istio 断路器。
    """
    try:
        cycles = detect_dag_cycles()
        if not cycles:
            return jsonify({'status': 'success', 'message': '未检测到环路, 无需配置断路器', 'applied': []})

        all_cycle_nodes = set()
        for cycle in cycles:
            all_cycle_nodes.update(cycle)
        svc_nodes = [n for n in all_cycle_nodes if RAG_DAG['nodes'][n].get('service')]

        applied = apply_circuit_breakers_for_cycle(svc_nodes)
        evo_logger.info(
            f'Evolution Strategy - 2 - 手动触发断路器配置完成: ' + ', '.join(applied)
        )
        return jsonify({'status': 'success', 'applied': applied})
    except Exception as e:
        evo_logger.warning(f'Evolution Strategy - 2 - 手动触发断路器配置失败: {e}')
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ============================================================
# Background scheduler for periodic critical path analysis
# ============================================================

_critical_path_timer = None


def _periodic_critical_path():
    """Run critical path analysis periodically in background."""
    namespace = os.environ.get('NAMESPACE', 'act-test')

    # ── 每轮检测前, 先清理上一轮动态创建的 Istio CRD ──
    # 只清理带 _DYNAMIC_CRD_LABEL 标签的资源, LLM/Gateway 等保持不变
    try:
        cleanup_dynamic_crds(namespace)
    except Exception as e:
        evo_logger.warning(f'Evolution Strategy - 2 - 动态CRD清理失败(继续执行检测): {e}')

    try:
        evo_logger.info('Evolution Strategy - 2 - 定时关键路径分析任务启动')
        result = compute_critical_paths(namespace)
        # Only auto-apply if explicitly enabled via env var
        if os.environ.get('CRITICAL_PATH_AUTO_APPLY', 'false').lower() in ('true', '1'):
            if result['bottleneck_nodes']:
                apply_resource_tilt(result['bottleneck_nodes'], namespace)
    except Exception as e:
        evo_logger.warning(f'Evolution Strategy - 2 - 定时关键路径分析失败: {e}')

    # 环路检测
    try:
        analyze_dag_cycles()
    except Exception as e:
        evo_logger.warning(f'Evolution Strategy - 2 - 定时环路检测失败: {e}')

    # 过长链路检测与聚合
    try:
        analyze_long_chains()
    except Exception as e:
        evo_logger.warning(f'Evolution Strategy - 2 - 定时链路检测失败: {e}')

    # Re-schedule
    global _critical_path_timer
    _critical_path_timer = threading.Timer(CRITICAL_PATH_INTERVAL, _periodic_critical_path)
    _critical_path_timer.daemon = True
    _critical_path_timer.start()


def start_critical_path_scheduler():
    """Start the background critical path analysis scheduler."""
    global _critical_path_timer
    if _critical_path_timer:
        return
    evo_logger.info(
        f'Evolution Strategy - 2 - 关键路径定时分析已启动, 间隔={CRITICAL_PATH_INTERVAL}s'
    )
    # Initial delay of 30s to let the app start up
    _critical_path_timer = threading.Timer(30, _periodic_critical_path)
    _critical_path_timer.daemon = True
    _critical_path_timer.start()


if __name__ == '__main__':
    # Start background critical path analysis
    start_critical_path_scheduler()
    app.run(host='0.0.0.0', port=8080)
