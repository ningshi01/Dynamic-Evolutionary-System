import requests
import json
import csv
import logging
import math
from datetime import datetime, timedelta
import pandas as pd
from prophet import Prophet
import os
import schedule
import time
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.config.config_exception import ConfigException
from kubernetes.utils.quantity import parse_quantity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_DEPLOYMENTS = {
    'api-deployment': 'api-hpa',
    'fileservice-deployment': 'fileservice-hpa',
    'parser-deployment': 'parser-hpa',
    'retriever-deployment': 'retriever-hpa',
}

_K8S_CONFIG_READY = False


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


def parse_cpu_quantity_to_cores(cpu_quantity):
    if not cpu_quantity:
        return None
    try:
        return float(parse_quantity(cpu_quantity))
    except (ValueError, TypeError) as exc:
        logger.warning(f"解析 CPU request {cpu_quantity} 时出错: {exc}")
        return None


def extract_cpu_request_from_deployment(deployment):
    containers = deployment.spec.template.spec.containers or []
    total_requested = 0.0
    for container in containers:
        resources = container.resources
        if not resources or not resources.requests:
            continue
        cpu_request = resources.requests.get('cpu')
        cores = parse_cpu_quantity_to_cores(cpu_request)
        if cores:
            total_requested += cores
    return total_requested if total_requested > 0 else None

def parse_gpu_quantity_to_count(gpu_quantity):
    if not gpu_quantity:
        return None
    try:
        # GPU 数量通常是整数，如 "1", "2" 等
        return float(gpu_quantity)
    except (ValueError, TypeError) as exc:
        logger.warning(f"解析 GPU request {gpu_quantity} 时出错: {exc}")
        return None

def extract_gpu_request_from_deployment(deployment):
    containers = deployment.spec.template.spec.containers or []
    total_requested = 0.0
    for container in containers:
        resources = container.resources
        if not resources or not resources.requests:
            continue
        gpu_request = resources.requests.get('nvidia.com/gpu')
        gpu_count = parse_gpu_quantity_to_count(gpu_request)
        if gpu_count:
            total_requested += gpu_count
    return total_requested if total_requested > 0 else None

def allocate_pods_based_on_forecast(deployment_scores, namespace, type):
    if not load_kubernetes_config():
        logger.warning("跳过自动副本调整：Kubernetes 配置不可用")
        return

    apps_v1 = client.AppsV1Api()
    autoscaling_v2 = client.AutoscalingV2Api()

    # 定义资源配置
    resource_configs = {
        1: {'name': 'cpu', 'display': 'CPU', 'request_extractor': extract_cpu_request_from_deployment},
        2: {'name': 'nvidia.com/gpu', 'display': 'GPU', 'request_extractor': extract_gpu_request_from_deployment},
    }

    if type not in resource_configs:
        logger.error(f"不支持的type参数: {type}")
        return

    config = resource_configs[type]
    resource_name = config['name']
    resource_display = config['display']
    request_extractor = config['request_extractor']

    for deployment_name, hpa_name in TARGET_DEPLOYMENTS.items():
        predicted_usage = deployment_scores.get(deployment_name)
        if predicted_usage is None:
            continue

        try:
            hpa = autoscaling_v2.read_namespaced_horizontal_pod_autoscaler(
                name=hpa_name,
                namespace=namespace,
            )
        except ApiException as exc:
            logger.warning(f"读取 {hpa_name} 失败: {exc}")
            continue

        average_util = None
        if hpa.spec.metrics:
            for metric in hpa.spec.metrics:
                if metric.type == 'Resource' and metric.resource and metric.resource.name == resource_name:
                    target = metric.resource.target
                    if target and target.average_utilization is not None:
                        average_util = target.average_utilization
                        break

        min_replicas = hpa.spec.min_replicas or 1
        max_replicas = hpa.spec.max_replicas or min_replicas

        if average_util is None:
            logger.warning(f"{hpa_name} 缺少 {resource_display} averageUtilization 配置，跳过")
            continue

        try:
            deployment = apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
            )
        except ApiException as exc:
            logger.warning(f"读取 Deployment {deployment_name} 失败: {exc}")
            continue

        resource_request = request_extractor(deployment)
        if resource_request is None or resource_request <= 0:
            logger.warning(f"Deployment {deployment_name} 未配置 {resource_display} requests，跳过自动副本调整")
            continue

        target_ratio = average_util / 100.0
        if target_ratio <= 0:
            logger.warning(f"{hpa_name} 的 averageUtilization = {average_util} 无效，跳过")
            continue

        target_resource_per_pod = resource_request * target_ratio
        if target_resource_per_pod <= 0:
            logger.warning(f"Deployment {deployment_name} 计算得到的单 Pod 目标 {resource_display} 为 0，跳过")
            continue

        if predicted_usage <= 0:
            recommended = min_replicas
        else:
            required = math.ceil(predicted_usage / target_resource_per_pod)
            recommended = max(required, 1)

        if recommended < min_replicas:
            logger.info(
                f"Deployment {deployment_name} 预测所需副本 {recommended} 低于 HPA 下限 {min_replicas}，需要缩容"
            )
        elif recommended > max_replicas:
            logger.info(
                f"Deployment {deployment_name} 预测所需副本 {recommended} 超出 HPA 上限 {max_replicas}，按上限处理"
            )
            recommended = max_replicas

        if recommended != min_replicas:
            try:
                autoscaling_v2.patch_namespaced_horizontal_pod_autoscaler(
                    name=hpa_name,
                    namespace=namespace,
                    body={'spec': {'minReplicas': recommended}},
                )
                logger.info(f"已将 {hpa_name} 的 minReplicas 更新为 {recommended}")
            except ApiException as exc:
                logger.error(f"更新 {hpa_name} minReplicas 失败: {exc}")

        current_replicas = deployment.spec.replicas or min_replicas
        if recommended != current_replicas:
            try:
                apps_v1.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body={'spec': {'replicas': recommended}},
                )
                logger.info(f"已将 Deployment {deployment_name} 副本数调整为 {recommended}")
            except ApiException as exc:
                logger.warning(f"更新 Deployment {deployment_name} 副本数失败: {exc}")

# def allocate_pods_based_on_forecast(deployment_scores, namespace):
#     if not load_kubernetes_config():
#         logger.warning("跳过自动副本调整：Kubernetes 配置不可用")
#         return

#     apps_v1 = client.AppsV1Api()
#     autoscaling_v2 = client.AutoscalingV2Api()

#     for deployment_name, hpa_name in TARGET_DEPLOYMENTS.items():
#         predicted_cpu = deployment_scores.get(deployment_name)
#         if predicted_cpu is None:
#             continue

#         try:
#             hpa = autoscaling_v2.read_namespaced_horizontal_pod_autoscaler(
#                 name=hpa_name,
#                 namespace=namespace,
#             )
#         except ApiException as exc:
#             logger.warning(f"读取 {hpa_name} 失败: {exc}")
#             continue

#         average_util = None
#         if hpa.spec.metrics:
#             for metric in hpa.spec.metrics:
#                 if metric.type == 'Resource' and metric.resource and metric.resource.name == 'cpu':
#                     target = metric.resource.target
#                     if target and target.average_utilization is not None:
#                         average_util = target.average_utilization
#                         break

#         min_replicas = hpa.spec.min_replicas or 1
#         max_replicas = hpa.spec.max_replicas or min_replicas

#         if average_util is None:
#             logger.warning(f"{hpa_name} 缺少 CPU averageUtilization 配置，跳过")
#             continue

#         try:
#             deployment = apps_v1.read_namespaced_deployment(
#                 name=deployment_name,
#                 namespace=namespace,
#             )
#         except ApiException as exc:
#             logger.warning(f"读取 Deployment {deployment_name} 失败: {exc}")
#             continue

#         cpu_request = extract_cpu_request_from_deployment(deployment)
#         if cpu_request is None or cpu_request <= 0:
#             logger.warning(f"Deployment {deployment_name} 未配置 CPU requests，跳过自动副本调整")
#             continue

#         target_ratio = average_util / 100.0
#         if target_ratio <= 0:
#             logger.warning(f"{hpa_name} 的 averageUtilization = {average_util} 无效，跳过")
#             continue

#         target_cpu_per_pod = cpu_request * target_ratio
#         if target_cpu_per_pod <= 0:
#             logger.warning(f"Deployment {deployment_name} 计算得到的单 Pod 目标 CPU 为 0，跳过")
#             continue

#         if predicted_cpu <= 0:
#             recommended = min_replicas
#         else:
#             required = math.ceil(predicted_cpu / target_cpu_per_pod)
#             recommended = max(required, 1)

#         if recommended < min_replicas:
#             logger.info(
#                 f"Deployment {deployment_name} 预测所需副本 {recommended} 低于 HPA 下限 {min_replicas}，需要缩容"
#             )
#         elif recommended > max_replicas:
#             logger.warning(
#                 f"Deployment {deployment_name} 预测所需副本 {recommended} 超出 HPA 上限 {max_replicas}，按上限处理"
#             )
#             recommended = max_replicas

#         if recommended != min_replicas:
#             try:
#                 autoscaling_v2.patch_namespaced_horizontal_pod_autoscaler(
#                     name=hpa_name,
#                     namespace=namespace,
#                     body={'spec': {'minReplicas': recommended}},
#                 )
#                 logger.info(f"已将 {hpa_name} 的 minReplicas 更新为 {recommended}")
#             except ApiException as exc:
#                 logger.warning(f"更新 {hpa_name} minReplicas 失败: {exc}")

#         current_replicas = deployment.spec.replicas or min_replicas
#         if recommended != current_replicas:
#             try:
#                 apps_v1.patch_namespaced_deployment(
#                     name=deployment_name,
#                     namespace=namespace,
#                     body={'spec': {'replicas': recommended}},
#                 )
#                 logger.info(f"已将 Deployment {deployment_name} 副本数调整为 {recommended}")
#             except ApiException as exc:
#                 logger.warning(f"更新 Deployment {deployment_name} 副本数失败: {exc}")

#         logger.info(
#             f"Deployment {deployment_name}: 预测平均 CPU {predicted_cpu:.4f} cores，单 Pod 目标 {target_cpu_per_pod:.4f} cores，计划副本数 {recommended}"
#         )

def export_prometheus_CPU_data():
    prom_url = "http://prometheus.istio-system.svc.cluster.local:9090"
    namespace = "act-test"
    time_range = "1h"

    # 构建查询 - 排除istio-proxy容器-基于deployment
    query = f'''
    sum by (deployment) (
        label_replace(
            rate(container_cpu_usage_seconds_total{{container!="istio-proxy", container!="", namespace="{namespace}"}}[{time_range}]),
            "deployment", "$1", "pod", "(.*)-[a-z0-9]+-[a-z0-9]+"
        )
    )
    '''

    # 使用query_range获取时间序列数据
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    step = "1m"  # 1分钟间隔

    response = requests.get(
        f"{prom_url}/api/v1/query_range",
        params={
            'query': query,
            'start': start_time.timestamp(),
            'end': end_time.timestamp(),
            'step': step
        }
    )

    if response.status_code == 200:
        data = response.json()

        if data['data']['result']:
            deployments_cpu_data = {}

            # 按deployment分组数据
            for result in data['data']['result']:
                metric = result['metric']
                deployment_name = metric.get('deployment', 'unknown')

                if deployment_name not in deployments_cpu_data:
                    deployments_cpu_data[deployment_name] = {
                        'deployment': deployment_name,
                        'data': []
                    }

                # 收集该deployment的所有时间序列数据
                for value_pair in result['values']:
                    timestamp = value_pair[0]
                    value = float(value_pair[1])
                    readable_time = datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S')

                    deployments_cpu_data[deployment_name]['data'].append({
                        'timestamp': timestamp,
                        'datetime': readable_time,
                        'value': value
                    })

            # 为每个deployment创建CSV文件 - 已注释掉
            deployment_files = {}
            for deployment_name, deployment_info in deployments_cpu_data.items():
                # 跳过unknown deployment
                if deployment_name == 'unknown':
                    continue

                filename = f'data_{namespace}_{time_range}_CPU_deployment_{deployment_name}.csv'

                # 注释掉CSV文件写入代码
                # with open(filename, 'w', newline='') as f:
                #     writer = csv.writer(f)
                #     writer.writerow(['timestamp', 'datetime', 'deployment', 'cpu_cores'])
                #
                #     for data_point in deployment_info['data']:
                #         writer.writerow([
                #             data_point['timestamp'],
                #             data_point['datetime'],
                #             deployment_name,
                #             f"{data_point['value']:.6f}"  # 保留6位小数
                #         ])

                # 仍然保留文件名映射用于后续处理
                deployment_files[deployment_name] = filename
                # logger.info(f"Deployment {deployment_name} 的CPU数据处理完成")

            logger.info(f"共处理 {len([d for d in deployments_cpu_data.keys() if d != 'unknown'])} 个deployment的时间序列数据")

            # 为每个deployment生成预测
            generate_prophet_forecasts(deployments_cpu_data, namespace, 1)
        else:
            logger.info("查询成功，但未找到CPU使用率数据")
    else:
        logger.error(f"CPU查询失败: {response.status_code}")

def export_prometheus_GPU_data():
    prom_url = "http://prometheus.istio-system.svc.cluster.local:9090"
    namespace = "act-test"
    time_range = "1h"

    # 构建查询 - 排除istio-proxy容器-基于deployment
    query = f'''
    sum by (deployment) (
        label_replace(
            DCGM_FI_DEV_GPU_UTIL{{namespace="{namespace}", container!="istio-proxy", container!=""}},
            "deployment", "$1", "pod", "(.*)-[a-z0-9]+-[a-z0-9]+"
        )
    )
    '''

    # 使用query_range获取时间序列数据
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    step = "1m"  # 1分钟间隔

    response = requests.get(
        f"{prom_url}/api/v1/query_range",
        params={
            'query': query,
            'start': start_time.timestamp(),
            'end': end_time.timestamp(),
            'step': step
        }
    )

    if response.status_code == 200:
        data = response.json()

        if data['data']['result']:
            deployments_gpu_data = {}

            # 按deployment分组数据
            for result in data['data']['result']:
                metric = result['metric']
                deployment_name = metric.get('deployment', 'unknown')

                if deployment_name not in deployments_gpu_data:
                    deployments_gpu_data[deployment_name] = {
                        'deployment': deployment_name,
                        'data': []
                    }

                # 收集该deployment的所有时间序列数据
                for value_pair in result['values']:
                    timestamp = value_pair[0]
                    value = float(value_pair[1])
                    readable_time = datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S')

                    deployments_gpu_data[deployment_name]['data'].append({
                        'timestamp': timestamp,
                        'datetime': readable_time,
                        'value': value
                    })

            deployment_files = {}
            for deployment_name, deployment_info in deployments_gpu_data.items():
                # 跳过unknown deployment
                if deployment_name == 'unknown':
                    continue

                filename = f'data_{namespace}_{time_range}_GPU_deployment_{deployment_name}.csv'
                deployment_files[deployment_name] = filename
                logger.info(f"Deployment {deployment_name} 的GPU数据处理完成")

            logger.info(f"共处理 {len([d for d in deployments_gpu_data.keys() if d != 'unknown'])} 个deployment的时间序列数据")

            # 为每个deployment生成预测
            generate_prophet_forecasts(deployments_gpu_data, namespace, 2)
        else:
            logger.info("查询成功，但未找到GPU使用率数据")
    else:
        logger.info(f"GPU查询失败: {response.status_code}")

def generate_prophet_forecasts(deployments_data, namespace, type):
    """
    为每个deployment的数据使用Prophet进行预测，并计算预测得分的均值
    type=1: CPU使用率预测
    type=2: GPU使用率预测  
    type=3: 
    """
    deployment_scores = {}

    # 定义指标配置
    metric_configs = {
        1: {'column': 'cpu_cores', 'name': 'CPU', 'display': 'CPU使用率'},
        2: {'column': 'gpu_usage', 'name': 'GPU', 'display': 'GPU使用率'},
        # 3: {'column': 'memory_usage', 'name': '内存', 'display': '内存使用率'}
    }

    # 检查type是否支持
    if type not in metric_configs:
        logger.error(f"不支持的type参数: {type}，支持的类型: {list(metric_configs.keys())}")
        return

    config = metric_configs[type]
    metric_column = config['column']
    metric_name = config['name']
    metric_display = config['display']

    for deployment_name, deployment_info in deployments_data.items():
        # 跳过unknown deployment
        if deployment_name == 'unknown':
            continue

        try:
            # 直接从内存中的数据创建DataFrame，而不是从CSV文件读取
            data_points = deployment_info['data']

            # 准备DataFrame
            df_data = {
                'timestamp': [point['timestamp'] for point in data_points],
                'datetime': [point['datetime'] for point in data_points],
                metric_column: [point['value'] for point in data_points]
            }
            df = pd.DataFrame(df_data)

            # 准备Prophet需要的数据格式
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df['datetime']),
                'y': df[metric_column]
            })

            # 创建并训练Prophet模型
            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=False,
                weekly_seasonality=False,
                changepoint_prior_scale=0.05
            )

            # 添加小时级别的季节性
            model.add_seasonality(name='hourly', period=1 / 24, fourier_order=5)

            # 训练模型
            model.fit(prophet_df)

            # 创建未来1小时的预测数据框（保持1分钟间隔）
            future = model.make_future_dataframe(periods=60, freq='1min')

            # 进行预测
            forecast = model.predict(future)

            # 只保留未来的预测部分（最后60个数据点）
            future_forecast = forecast.tail(60).copy()

            # 添加deployment列
            future_forecast['deployment'] = deployment_name

            # 根据type选择列名
            if type == 1:
                forecast_result = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'deployment']]
                forecast_result.columns = ['datetime', 'predicted_cpu_cores', 'predicted_lower', 'predicted_upper',
                                           'deployment']
            elif type == 2:
                forecast_result = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'deployment']]
                forecast_result.columns = ['datetime', 'predicted_gpu_usage', 'predicted_lower', 'predicted_upper',
                                           'deployment']
            elif type == 3:
                forecast_result = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'deployment']]
                forecast_result.columns = ['datetime', 'predicted_memory_usage', 'predicted_lower', 'predicted_upper',
                                           'deployment']

            # 格式化datetime
            forecast_result['datetime'] = forecast_result['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # 计算预测得分的均值（使用yhat列）
            predicted_values = forecast_result.iloc[:, 1]  # 第二列是预测值
            predicted_values_non_negative = predicted_values.clip(lower=0)  # 将所有小于0的值设为0
            mean_prediction = predicted_values_non_negative.mean()
            deployment_scores[deployment_name] = mean_prediction

            # logger.info(f"Deployment {deployment_name} 的{metric_name}预测完成")
            # logger.info(f"预测数据量: {len(forecast_result)} 条记录")

        except Exception as e:
            logger.error(f"为Deployment {deployment_name} 生成{metric_name}预测时出错: {str(e)}")

    # 打印每个deployment的预测得分均值
    logger.info(f"\n=== 各Deployment预测得分均值 ({metric_display}) ===")
    for deployment_name, mean_score in deployment_scores.items():
        logger.info(f"Deployment {deployment_name}: 平均预测{metric_display} = {mean_score * 100:.2f}%")

    if deployment_scores:
        allocate_pods_based_on_forecast(deployment_scores, namespace, type)
    else:
        logger.info(f"未生成任何{metric_display}预测结果，跳过自动副本调整")

# def generate_prophet_forecasts(deployments_data, namespace,type):
#     """
#     为每个deployment的数据使用Prophet进行预测，并计算预测得分的均值
#     """
#     deployment_scores = {}

#     for deployment_name, deployment_info in deployments_data.items():
#         # 跳过unknown deployment
#         if deployment_name == 'unknown':
#             continue

#         try:
#             # 直接从内存中的数据创建DataFrame，而不是从CSV文件读取
#             data_points = deployment_info['data']

#             # 准备DataFrame
#             df_data = {
#                 'timestamp': [point['timestamp'] for point in data_points],
#                 'datetime': [point['datetime'] for point in data_points],
#                 'cpu_cores': [point['value'] for point in data_points]
#             }
#             df = pd.DataFrame(df_data)

#             # 准备Prophet需要的数据格式
#             prophet_df = pd.DataFrame({
#                 'ds': pd.to_datetime(df['datetime']),
#                 'y': df['cpu_cores']
#             })

#             # 创建并训练Prophet模型
#             model = Prophet(
#                 daily_seasonality=True,
#                 yearly_seasonality=False,
#                 weekly_seasonality=False,
#                 changepoint_prior_scale=0.05
#             )

#             # 添加小时级别的季节性
#             model.add_seasonality(name='hourly', period=1 / 24, fourier_order=5)

#             # 训练模型
#             model.fit(prophet_df)

#             # 创建未来1小时的预测数据框（保持1分钟间隔）
#             future = model.make_future_dataframe(periods=60, freq='1min')

#             # 进行预测
#             forecast = model.predict(future)

#             # 只保留未来的预测部分（最后60个数据点）
#             future_forecast = forecast.tail(60).copy()

#             # 添加deployment列
#             future_forecast['deployment'] = deployment_name

#             # 选择需要的列并重命名
#             forecast_result = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'deployment']]
#             forecast_result.columns = ['datetime', 'predicted_cpu_cores', 'predicted_lower', 'predicted_upper',
#                                        'deployment']

#             # 格式化datetime
#             forecast_result['datetime'] = forecast_result['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

#             # 计算预测得分的均值（使用yhat列）
#             predicted_cores = forecast_result['predicted_cpu_cores']
#             predicted_cores_non_negative = predicted_cores.clip(lower=0)  # 将所有小于0的值设为0
#             mean_prediction = predicted_cores_non_negative.mean()
#             deployment_scores[deployment_name] = mean_prediction

#             # 保存预测结果到CSV - 已注释掉
#             # forecast_filename = f'forecast_{namespace}_1h_CPU_deployment_{deployment_name}.csv'
#             # forecast_result.to_csv(forecast_filename, index=False)

#             logger.info(f"Deployment {deployment_name} 的CPU预测完成")
#             logger.info(f"预测数据量: {len(forecast_result)} 条记录")

#         except Exception as e:
#             logger.info(f"为Deployment {deployment_name} 生成预测时出错: {str(e)}")

#     # 打印每个deployment的预测得分均值
#     logger.info("\n=== 各Deployment预测得分均值 ===")
#     for deployment_name, mean_score in deployment_scores.items():
#         logger.info(f"Deployment {deployment_name}: 平均预测CPU使用率 = {mean_score * 100:.2f}%")

#     if deployment_scores:
#         allocate_pods_based_on_forecast(deployment_scores, namespace)
#     else:
#         logger.info("未生成任何预测结果，跳过自动副本调整")


def job():
    """每小时执行的任务（带重试机制）"""
    logger.info(f"开始执行数据导出任务 - {datetime.now()}")
    
    max_retries = 3  # 最大重试次数
    retry_delay = 5  # 重试延迟（秒）
    
    for attempt in range(max_retries):
        try:
            export_prometheus_CPU_data()
            export_prometheus_GPU_data()
            logger.info("任务执行成功")
            return  # 成功则退出函数
        except Exception as e:
            logger.error(f"任务执行失败 (第{attempt + 1}次尝试): {e}")
            
            if attempt < max_retries - 1:  # 不是最后一次尝试
                logger.info(f"{retry_delay}秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                logger.error(f"任务重试{max_retries}次后仍失败，放弃执行")

def run_scheduler():
    """调度器"""
    schedule.every(1).hours.do(job)
    job()
    logger.info("调度器已启动，将每小时执行一次...")
    while True:
        schedule.run_pending()
        time.sleep(60)  # 每分钟检查一次

if __name__ == "__main__":
    run_scheduler()