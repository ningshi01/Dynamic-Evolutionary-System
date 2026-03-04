# Parser VPA 配置说明

## 概述
此 VPA (Vertical Pod Autoscaler) 配置用于自动调整 parser-deployment 的资源请求（CPU 和内存）。

## 文件说明
- `parser-vpa.yaml`: VPA 配置文件

## VPA 配置详解

### UpdateMode 选项
- **Auto**: 自动更新资源请求并在必要时重启 Pod（推荐用于生产环境）
- **Recreate**: 在 Pod 重建时应用推荐值
- **Initial**: 仅在 Pod 创建时设置资源请求，之后不再修改
- **Off**: 仅生成推荐值，不自动应用

### 资源限制
- **minAllowed**: 最小资源请求
  - CPU: 50m (0.05 核)
  - Memory: 128Mi
- **maxAllowed**: 最大资源请求
  - CPU: 2000m (2 核)
  - Memory: 2Gi

## 前置要求

### 1. 安装 VPA
VPA 需要单独安装到 Kubernetes 集群中：

```bash
# 克隆 VPA 仓库
git clone https://github.com/kubernetes/autoscaler.git
cd autoscaler/vertical-pod-autoscaler

# 安装 VPA
./hack/vpa-up.sh
```

或使用 Helm 安装：
```bash
helm repo add fairwinds-stable https://charts.fairwinds.com/stable
helm install vpa fairwinds-stable/vpa --namespace vpa --create-namespace
```

### 2. 验证 VPA 安装
```bash
kubectl get pods -n kube-system | grep vpa
# 应该看到以下三个组件：
# - vpa-admission-controller
# - vpa-recommender
# - vpa-updater
```

## 部署步骤

### 1. 应用 VPA 配置
```bash
kubectl apply -f parser-vpa.yaml
```

### 2. 验证 VPA 状态
```bash
# 查看 VPA 对象
kubectl get vpa -n act-test

# 查看 VPA 详细信息
kubectl describe vpa parser-vpa -n act-test
```

### 3. 查看资源推荐值
```bash
kubectl get vpa parser-vpa -n act-test -o yaml
```

输出示例：
```yaml
status:
  recommendation:
    containerRecommendations:
    - containerName: parser
      lowerBound:
        cpu: 100m
        memory: 256Mi
      target:
        cpu: 200m
        memory: 512Mi
      upperBound:
        cpu: 500m
        memory: 1Gi
```

## VPA 与 HPA 共存

**重要提示**: VPA 和 HPA 可以共存，但需要注意：
- HPA 基于 CPU/内存利用率进行**水平扩展**（增减 Pod 数量）
- VPA 进行**垂直扩展**（调整单个 Pod 的资源请求）
- **不要让 HPA 和 VPA 同时基于相同的指标**（如都基于 CPU），否则可能产生冲突

### 推荐配置
1. **HPA 基于 CPU 利用率**进行水平扩展（已配置在 parser-hpa.yaml）
2. **VPA 调整内存资源请求**，或设置为 `updateMode: "Off"` 仅提供推荐值

### 修改 VPA 为推荐模式（避免与 HPA 冲突）
如果希望 VPA 仅提供推荐而不自动应用，可以修改：
```yaml
spec:
  updatePolicy:
    updateMode: "Off"  # 仅推荐，不自动应用
```

### 或者让 VPA 仅管理内存
```yaml
spec:
  resourcePolicy:
    containerPolicies:
    - containerName: parser
      controlledResources: ["memory"]  # 仅控制内存
      mode: Auto
```

## 监控和调试

### 查看 VPA 推荐历史
```bash
kubectl get vpa parser-vpa -n act-test -o jsonpath='{.status.recommendation}' | jq
```

### 查看 VPA 事件
```bash
kubectl get events -n act-test --field-selector involvedObject.name=parser-vpa
```

### 查看 Pod 资源使用情况
```bash
kubectl top pods -n act-test -l app=parser
```

## 性能调优建议

### 1. 初始阶段
- 使用 `updateMode: "Off"` 观察推荐值
- 收集至少 24 小时的数据以获得准确推荐

### 2. 稳定阶段
- 根据推荐值手动调整 Deployment 的资源请求
- 或启用 `updateMode: "Auto"` 让 VPA 自动管理

### 3. 生产环境
- 设置合理的 `minAllowed` 和 `maxAllowed` 范围
- 配合 PodDisruptionBudget 确保高可用性
- 监控 VPA 导致的 Pod 重启频率

## 故障排查

### VPA 不生效
1. 检查 VPA 组件是否正常运行
2. 确认 VPA 对象状态：`kubectl describe vpa parser-vpa -n act-test`
3. 查看 vpa-recommender 日志：`kubectl logs -n kube-system <vpa-recommender-pod>`

### Pod 频繁重启
- VPA 在 `Auto` 模式下调整资源会重启 Pod
- 考虑使用 `updateMode: "Recreate"` 或 `"Initial"`
- 或仅启用内存调整，让 HPA 处理 CPU

