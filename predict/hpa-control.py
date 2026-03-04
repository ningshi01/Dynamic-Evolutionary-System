import requests
import logging
import math
from datetime import datetime, timedelta
import pandas as pd
from prophet import Prophet
import time
import schedule
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.config.config_exception import ConfigException
from kubernetes.utils.quantity import parse_quantity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_K8S_CONFIG_READY = False
NAMESPACE = "act-test"
PROM_URL = "http://prometheus.istio-system.svc.cluster.local:9090"

def load_kubernetes_config():
    global _K8S_CONFIG_READY
    if _K8S_CONFIG_READY:
        return True
    try:
        config.load_incluster_config()
        _K8S_CONFIG_READY = True
        logger.info("已加载集群内 Kubernetes 配置")
        return True
    except ConfigException:
        pass

    try:
        config.load_kube_config()
        _K8S_CONFIG_READY = True
        logger.info("已加载本地 kubeconfig 配置")
        return True
    except ConfigException as exc:
        logger.error(f"加载 Kubernetes 配置失败: {exc}")
        return False

class ResourceConfig:
    def __init__(self, name, k8s_resource_name, prom_query_fmt):
        self.name = name
        self.k8s_resource_name = k8s_resource_name
        self.prom_query_fmt = prom_query_fmt

# 定义支持的资源类型及其配置
RESOURCE_CONFIGS = [
    ResourceConfig(
        name='cpu',
        k8s_resource_name='cpu',
        # CPU查询: rate计算速率
        prom_query_fmt='sum by (deployment) (label_replace(rate(container_cpu_usage_seconds_total{{container!="istio-proxy", container!="", namespace="{namespace}"}}[{time_range}]), "deployment", "$1", "pod", "(.*)-[a-z0-9]+-[a-z0-9]+"))'
    ),
    ResourceConfig(
        name='gpu',
        k8s_resource_name='nvidia.com/gpu',
        # GPU查询: 直接获取利用率/数量
        prom_query_fmt='sum by (deployment) (label_replace(DCGM_FI_DEV_GPU_UTIL{{namespace="{namespace}", container!="istio-proxy", container!=""}}, "deployment", "$1", "pod", "(.*)-[a-z0-9]+-[a-z0-9]+"))'
    )
]

def parse_quantity_to_float(quantity):
    """解析K8s资源数量字符串为浮点数"""
    if not quantity:
        return None
    try:
        return float(parse_quantity(quantity))
    except (ValueError, TypeError):
        # 尝试直接转换 (针对纯数字字符串)
        try:
            return float(quantity)
        except (ValueError, TypeError) as exc:
            logger.warning(f"解析资源数量 {quantity} 失败: {exc}")
            return None

def extract_resource_request(deployment, resource_name):
    """从Deployment中提取指定资源的总Request值"""
    containers = deployment.spec.template.spec.containers or []
    total_requested = 0.0
    found = False
    for container in containers:
        resources = container.resources
        if not resources or not resources.requests:
            continue
        req = resources.requests.get(resource_name)
        val = parse_quantity_to_float(req)
        if val:
            total_requested += val
            found = True
    return total_requested if found else None

def get_hpa_map(namespace):
    """获取命名空间下所有 Deployment 类型的 HPA，返回 {deployment_name: hpa_object} 映射"""
    if not load_kubernetes_config():
        return {}
    
    autoscaling_v2 = client.AutoscalingV2Api()
    try:
        hpas = autoscaling_v2.list_namespaced_horizontal_pod_autoscaler(namespace=namespace)
        mapping = {}
        for hpa in hpas.items:
            scale_target = hpa.spec.scale_target_ref
            if scale_target.kind == 'Deployment':
                mapping[scale_target.name] = hpa
        return mapping
    except ApiException as exc:
        logger.error(f"获取 HPA 列表失败: {exc}")
        return {}

def fetch_prometheus_data(resource_config, namespace, time_range="1h"):
    """从Prometheus获取数据"""
    query = resource_config.prom_query_fmt.format(namespace=namespace, time_range=time_range)
    
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    step = "1m"

    try:
        response = requests.get(
            f"{PROM_URL}/api/v1/query_range",
            params={
                'query': query,
                'start': start_time.timestamp(),
                'end': end_time.timestamp(),
                'step': step
            },
            timeout=10
        )
        
        if response.status_code != 200:
            logger.error(f"Prometheus 查询失败: {response.status_code}")
            return {}
            
        data = response.json()
        if not data.get('data', {}).get('result'):
            logger.info(f"未查询到 {resource_config.name} 数据")
            return {}

        deployments_data = {}
        for result in data['data']['result']:
            metric = result['metric']
            deployment_name = metric.get('deployment', 'unknown')
            if deployment_name == 'unknown':
                continue

            if deployment_name not in deployments_data:
                deployments_data[deployment_name] = []

            for value_pair in result['values']:
                deployments_data[deployment_name].append({
                    'timestamp': float(value_pair[0]),
                    'datetime': datetime.fromtimestamp(float(value_pair[0])).strftime('%Y-%m-%d %H:%M:%S'),
                    'value': float(value_pair[1])
                })
        
        return deployments_data

    except Exception as e:
        logger.error(f"获取 Prometheus 数据异常: {e}")
        return {}

def generate_forecasts(deployments_data):
    """使用Prophet生成预测，返回 {deployment_name: predicted_value}"""
    scores = {}
    
    for deployment_name, data_points in deployments_data.items():
        try:
            if not data_points:
                continue

            df = pd.DataFrame(data_points)
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df['datetime']),
                'y': df['value']
            })

            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=False,
                weekly_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.add_seasonality(name='hourly', period=1/24, fourier_order=5)
            
            # 抑制Prophet日志
            logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
            
            model.fit(prophet_df)

            future = model.make_future_dataframe(periods=60, freq='1min')
            forecast = model.predict(future)
            
            # 取未来60分钟的预测均值
            future_forecast = forecast.tail(60)
            predicted_values = future_forecast['yhat'].clip(lower=0)
            mean_prediction = predicted_values.mean()
            
            scores[deployment_name] = mean_prediction
            # logger.info(f"Deployment {deployment_name} 预测值: {mean_prediction:.4f}")

        except Exception as e:
            logger.error(f"Deployment {deployment_name} 预测失败: {e}")
            
    return scores

def apply_scaling(deployment_scores, namespace, resource_config):
    """根据预测结果应用扩缩容"""
    if not load_kubernetes_config():
        return

    apps_v1 = client.AppsV1Api()
    autoscaling_v2 = client.AutoscalingV2Api()
    
    # 自动发现当前命名空间下的所有HPA
    hpa_map = get_hpa_map(namespace)
    
    for deployment_name, predicted_usage in deployment_scores.items():
        if deployment_name not in hpa_map:
            continue
            
        hpa = hpa_map[deployment_name]
        hpa_name = hpa.metadata.name
        
        # 检查HPA是否配置了当前资源类型的指标
        average_util = None
        if hpa.spec.metrics:
            for metric in hpa.spec.metrics:
                if (metric.type == 'Resource' and 
                    metric.resource and 
                    metric.resource.name == resource_config.k8s_resource_name):
                    
                    target = metric.resource.target
                    if target and target.average_utilization is not None:
                        average_util = target.average_utilization
                        break
        
        if average_util is None:
            # 该HPA不包含当前处理资源的指标配置，跳过
            continue

        try:
            deployment = apps_v1.read_namespaced_deployment(name=deployment_name, namespace=namespace)
        except ApiException as exc:
            logger.warning(f"读取 Deployment {deployment_name} 失败: {exc}")
            continue

        # 提取Deployment的资源Request
        resource_request = extract_resource_request(deployment, resource_config.k8s_resource_name)
        if resource_request is None or resource_request <= 0:
            logger.warning(f"Deployment {deployment_name} 未配置 {resource_config.name} requests，跳过")
            continue

        # 计算目标值
        target_ratio = average_util / 100.0
        target_resource_per_pod = resource_request * target_ratio
        
        if target_resource_per_pod <= 0:
            continue

        # 计算推荐副本数
        min_replicas = hpa.spec.min_replicas or 1
        max_replicas = hpa.spec.max_replicas or min_replicas

        if predicted_usage <= 0:
            recommended = min_replicas
        else:
            required = math.ceil(predicted_usage / target_resource_per_pod)
            recommended = max(required, 1)

        # 边界检查
        if recommended < min_replicas:
            logger.info(f"[{resource_config.name}] {deployment_name}: 预测需求 {recommended} < HPA下限 {min_replicas}，需缩容")
        elif recommended > max_replicas:
            logger.info(f"[{resource_config.name}] {deployment_name}: 预测需求 {recommended} > HPA上限 {max_replicas}，按上限处理")
            recommended = max_replicas
        else:
            logger.info(f"[{resource_config.name}] {deployment_name}: 预测需求 {recommended} (当前范围 {min_replicas}-{max_replicas})")

        # 执行更新
        if recommended != min_replicas:
            try:
                autoscaling_v2.patch_namespaced_horizontal_pod_autoscaler(
                    name=hpa_name,
                    namespace=namespace,
                    body={'spec': {'minReplicas': recommended}},
                )
                logger.info(f"更新 HPA {hpa_name} minReplicas -> {recommended}")
            except ApiException as exc:
                logger.error(f"更新 HPA {hpa_name} 失败: {exc}")

        current_replicas = deployment.spec.replicas or min_replicas
        if recommended != current_replicas:
            try:
                apps_v1.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body={'spec': {'replicas': recommended}},
                )
                logger.info(f"更新 Deployment {deployment_name} replicas -> {recommended}")
            except ApiException as exc:
                logger.error(f"更新 Deployment {deployment_name} 失败: {exc}")

def process_resource_pipeline(resource_config):
    """单个资源类型的处理流水线"""
    logger.info(f"=== 开始处理资源: {resource_config.name} ===")
    
    # 1. 获取数据
    data = fetch_prometheus_data(resource_config, NAMESPACE)
    if not data:
        return

    # 2. 生成预测
    scores = generate_forecasts(data)
    if not scores:
        logger.info(f"未生成 {resource_config.name} 预测结果")
        return
        
    # 3. 应用调整
    apply_scaling(scores, NAMESPACE, resource_config)

def job():
    """定时任务入口"""
    logger.info(f"开始执行预测与调整任务 - {datetime.now()}")
    
    for config in RESOURCE_CONFIGS:
        try:
            process_resource_pipeline(config)
        except Exception as e:
            logger.error(f"处理资源 {config.name} 流程发生未捕获异常: {e}")
            
    logger.info("任务执行结束")

def run_scheduler():
    """调度器"""
    # 立即执行一次
    job()
    
    schedule.every(1).hours.do(job)
    logger.info("调度器已启动，将每小时执行一次...")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    run_scheduler()
