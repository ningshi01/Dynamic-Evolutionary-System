# 知识库导入脚本
# API 端点
API_URL="http://192.168.5.65:32719/add_documents"

# 构建 JSON 数据
# 使用 cat 和 heredoc 生成结构，避免复杂的转义
# 注意：此处假设目标API接受JSON，且字段名为"collection_name"和"documents"

# 临时文件（可选，避免命令行过长）
TMP_JSON=$(mktemp)

# 写入 JSON 开头
cat > "$TMP_JSON" << 'EOF'
{
  "collection_name": "act_test_kb",
  "documents": [
EOF

# 生成100条知识语句
# 使用循环追加到临时文件，每条后用逗号分隔，最后一条不带逗号
for i in {1..100}; do
  # 根据索引生成不同的知识语句
  case $i in
    1) statement="Kubernetes 是一个开源的容器编排平台，用于自动部署、扩展和管理容器化应用程序。" ;;
    2) statement="云原生技术有利于各组织在公有云、私有云和混合云等新型动态环境中构建和运行可弹性扩展的应用。" ;;
    3) statement="AI infra 指的是为人工智能工作负载设计的硬件和软件基础设施，包括 GPU、TPU 以及分布式训练框架。" ;;
    4) statement="Kubelet 是 Kubernetes 节点上的主要代理，负责确保容器运行在 Pod 中并保持健康。" ;;
    5) statement="etcd 是 Kubernetes 用于存储所有集群数据的分布式键值存储系统，具有高可用性和一致性。" ;;
    6) statement="容器技术（如 Docker）通过将代码和依赖项打包在一起，实现了应用程序的环境一致性和可移植性。" ;;
    7) statement="Kubectl 是 Kubernetes 的命令行工具，用于对集群运行命令、部署应用、查看日志和资源管理。" ;;
    8) statement="微服务架构将单一应用程序划分为一组小型的、独立的服务，每个服务都可以独立部署和扩展。" ;;
    9) statement="服务网格（如 Istio）为微服务提供了流量管理、安全通信和可观测性，无需修改应用程序代码。" ;;
    10) statement="Prometheus 是云原生计算基金会（CNCF）的毕业项目，是一个强大的监控和报警工具。" ;;
    11) statement="Helm 是 Kubernetes 的包管理器，通过 Chart 简化应用程序的定义、安装和升级。" ;;
    12) statement="无服务器计算（Serverless）使开发者能够专注于编写代码而无需管理底层服务器。" ;;
    13) statement="Kubeflow 致力于使 Kubernetes 上的机器学习工作流部署变得简单、可移植和可扩展。" ;;
    14) statement="Kubernetes Pod 是集群中可创建和管理的最小部署单元，包含一个或多个容器。" ;;
    15) statement="云原生存储通常通过 CSI（容器存储接口）与 Kubernetes 集成，支持动态配置和快照。" ;;
    16) statement="GPU 虚拟化技术允许多个任务共享一块 GPU，提高 AI 训练和推理的硬件利用率。" ;;
    17) statement="Dockerfile 是一个文本文件，包含了构建 Docker 镜像所需的所有指令。" ;;
    18) statement="Kubernetes 命名空间提供了在单个集群中划分资源组的方法，适用于多环境或多租户场景。" ;;
    19) statement="Horizontal Pod Autoscaler（HPA）根据 CPU 利用率或其他指标自动调整 Pod 副本数量。" ;;
    20) statement="CNCF（云原生计算基金会）托管着 Kubernetes、Prometheus、Envoy 等关键云原生项目。" ;;
    21) statement="Envoy 是一个为服务网格设计的高性能 C++ 代理，处理服务间的通信和可观测性。" ;;
    22) statement="Jaeger 是一个开源的端到端分布式追踪系统，用于监控和故障排除微服务架构。" ;;
    23) statement="Fluentd 是一个开源数据收集器，统一日志层，使数据消费更加简单和灵活。" ;;
    24) statement="CoreDNS 是 Kubernetes 默认的 DNS 服务器，用于服务发现和名称解析。" ;;
    25) statement="Calico 是一个流行的 Kubernetes 网络插件，支持网络策略和 BGP 路由。" ;;
    26) statement="YAML 是 Kubernetes 资源配置的常用格式，定义了所需状态的声明式描述。" ;;
    27) statement="Kubernetes Operator 是一种封装、部署和管理 Kubernetes 应用程序的方法，扩展了 API。" ;;
    28) statement="RBAC（基于角色的访问控制）在 Kubernetes 中用于授权用户和服务账户的权限。" ;;
    29) statement="DevOps 文化强调开发与运维之间的协作，而云原生技术是实现 DevOps 实践的有力工具。" ;;
    30) statement="Terraform 是一种基础设施即代码工具，可用于在 AWS、GCP 等平台上创建和管理 Kubernetes 集群。" ;;
    31) statement="Kubernetes Ingress 管理集群内服务的外部访问，通常通过 HTTP 路由规则实现。" ;;
    32) statement="容器镜像仓库（如 Harbor、Docker Hub）用于存储和分发容器镜像。" ;;
    33) statement="Kubernetes 中的 ConfigMap 用于将配置数据与容器镜像分离，提高应用的可移植性。" ;;
    34) statement="Kubernetes Secret 用于存储敏感信息，如密码、OAuth 令牌和 ssh key。" ;;
    35) statement="持久卷（PV）和持久卷声明（PVC）是 Kubernetes 中管理存储资源的抽象。" ;;
    36) statement="StatefulSet 专用于管理有状态应用程序，提供稳定的网络标识和持久存储。" ;;
    37) statement="DaemonSet 确保所有（或某些）节点运行一个 Pod 的副本，常用于日志收集和监控代理。" ;;
    38) statement="Kubernetes 中的 Job 负责处理一次性或批处理任务，保证任务成功完成。" ;;
    39) statement="CronJob 基于时间调度 Kubernetes Job，类似于 Linux 的 crontab。" ;;
    40) statement="Kube-proxy 是 Kubernetes 每个节点上运行的网络代理，实现 Service 的部分概念。" ;;
    41) statement="容器网络接口（CNI）是配置 Linux 容器网络的插件规范，Kubernetes 广泛使用。" ;;
    42) statement="CRI（容器运行时接口）使 kubelet 能够与不同的容器运行时（如 containerd、CRI-O）交互。" ;;
    43) statement="OCI（开放容器倡议）制定了容器镜像格式和运行时规范，确保不同工具的兼容性。" ;;
    44) statement="Kubeadm 是一个工具，用于创建符合最佳实践的 Kubernetes 集群。" ;;
    45) statement="Minikube 是一个轻量级 Kubernetes 实现，可在本地虚拟机中创建单节点集群用于开发和测试。" ;;
    46) statement="Kind（Kubernetes in Docker）使用 Docker 容器作为节点，快速创建本地集群。" ;;
    47) statement="K3s 是 Rancher 开发的轻量级 Kubernetes 发行版，专为边缘计算和物联网设计。" ;;
    48) statement="微服务架构中的断路器模式可以防止故障级联，Hystrix 是一个经典实现。" ;;
    49) statement="分布式追踪帮助理解请求在微服务间的传播路径，OpenTelemetry 是统一的可观测性标准。" ;;
    50) statement="云原生安全涉及容器安全、镜像安全、运行时安全和供应链安全等多个层面。" ;;
    51) statement="Falco 是云原生运行时安全项目，作为 CNCF 的孵化项目，检测异常行为。" ;;
    52) statement="OPA（开放策略代理）是一种策略引擎，可用于在 Kubernetes 中实施细粒度的准入控制。" ;;
    53) statement="Gatekeeper 是 OPA 的 Kubernetes 专用版本，提供可配置的准入控制策略。" ;;
    54) statement="Kubernetes 审计日志记录了对集群 API 的每个请求，有助于安全分析和合规性。" ;;
    55) statement="镜像扫描工具（如 Trivy、Clair）可以检测容器镜像中的漏洞。" ;;
    56) statement="Sigstore 是一个用于签名和验证软件工具体，以提高软件供应链安全性。" ;;
    57) statement="Tekton 是一个强大的 CI/CD 框架，允许开发者在 Kubernetes 上构建、测试和部署。" ;;
    58) statement="Argo CD 是声明式的 GitOps 持续交付工具，遵循 Git 作为应用定义的真实来源。" ;;
    59) statement="Flux 是另一个流行的 GitOps 工具，与 Argo CD 一样，确保集群状态与 Git 仓库同步。" ;;
    60) statement="Jenkins X 是一个基于 Kubernetes 的 CI/CD 平台，自动执行环境管理和版本发布。" ;;
    61) statement="Knative 扩展了 Kubernetes，提供构建、部署和管理无服务器工作负载的组件。" ;;
    62) statement="Dapr（分布式应用程序运行时）通过 sidecar 模式为微服务提供构建块，如状态管理和服务调用。" ;;
    63) statement="KEDA（基于 Kubernetes 的事件驱动自动伸缩）可以根据事件源（如 Kafka 消息）伸缩容器。" ;;
    64) statement="Volcano 是一个基于 Kubernetes 的批处理系统，专为高性能工作负载（如 AI/ML）设计。" ;;
    65) statement="Kubeflow Pipelines 是一个用于构建和部署可移植的、可扩展的机器学习工作流的平台。" ;;
    66) statement="PyTorch 和 TensorFlow 是两种最流行的深度学习框架，常在云原生环境中运行。" ;;
    67) statement="Jupyter Notebook 在 Kubeflow 中可作为中央组件，用于交互式数据科学和模型开发。" ;;
    68) statement="模型训练可以通过 Kubernetes 的 Job 或 Volcano 进行分布式调度，提高训练效率。" ;;
    69) statement="模型推理通常使用专门的服务器（如 TensorFlow Serving、TorchServe）部署在 Kubernetes 上。" ;;
    70) statement="NVIDIA GPU Operator 自动管理 Kubernetes 集群中 GPU 节点的资源调配和监控。" ;;
    71) statement="Habana Gaudi 是专为深度学习训练设计的 AI 处理器，其 Operator 可集成到 Kubernetes。" ;;
    72) statement="AWS Trainium 和 Inferentia 是 AWS 自研的机器学习芯片，通过 Neuron SDK 与 Kubernetes 集成。" ;;
    73) statement="Kubernetes 中的设备插件框架允许 Pod 访问 GPU、FPGA 等特殊硬件资源。" ;;
    74) statement="拓扑感知调度在 AI 训练中很重要，确保 GPU 间高速通信（如 NVLink）的利用。" ;;
    75) statement="RDMA（远程直接内存访问）和 InfiniBand 在高性能计算和分布式训练中用于低延迟通信。" ;;
    76) statement="分布式训练框架（如 Horovod）支持跨多个 GPU 和节点并行训练模型。" ;;
    77) statement="AllReduce 是一种常见的分布式训练通信模式，用于聚合梯度。" ;;
    78) statement="Kubectl exec 命令允许在运行的容器中执行命令，用于调试和故障排查。" ;;
    79) statement="kubectl port-forward 可以将本地端口转发到 Pod，方便访问内部服务。" ;;
    80) statement="Kubernetes 中的服务质量（QoS）类包括 Guaranteed、Burstable 和 BestEffort，用于资源回收优先级。" ;;
    81) statement="cgroup（控制组）是 Linux 内核特性，用于限制和隔离进程的资源使用，容器运行时依赖它。" ;;
    82) statement="Namespace（Linux 命名空间）是容器隔离的基础，提供 PID、网络、挂载等隔离视图。" ;;
    83) statement="OverlayFS 是一种联合文件系统，常被 Docker 用作存储驱动，实现镜像分层。" ;;
    84) statement="Kata Containers 结合了虚拟机的安全性和容器的速度，使用轻量级虚拟机作为容器边界。" ;;
    85) statement="gVisor 是 Google 开发的容器运行时，为用户空间内核拦截系统调用，增强隔离性。" ;;
    86) statement="Firecracker 是 AWS 开发的微虚拟机管理程序，用于 Lambda 和 Fargate 等无服务器服务。" ;;
    87) statement="Kubernetes 社区版本每三个月发布一次小版本，遵循语义化版本控制。" ;;
    88) statement="Kubernetes Enhancement Proposals（KEPs）是引入主要功能的设计文档和流程。" ;;
    89) statement="SIG（特别兴趣小组）是 Kubernetes 社区的组织单元，负责不同组件和领域。" ;;
    90) statement="KubeCon + CloudNativeCon 是云原生社区的主要会议，汇集了开发者和用户。" ;;
    91) statement="云原生定义由 CNCF 维护，强调容器化、微服务、动态管理和可观测性。" ;;
    92) statement="FinOps 在云原生环境中变得重要，涉及成本分配、优化和治理。" ;;
    93) statement="Kubecost 是一个用于监控 Kubernetes 工作负载成本和健康度的开源工具。" ;;
    94) statement="混沌工程（如 Chaos Mesh）通过注入故障测试系统的弹性，在云原生中广泛实践。" ;;
    95) statement="Litmus 是一个云原生混沌工程工具集，提供实验框架和混沌中心。" ;;
    96) statement="Kubernetes 多集群管理（如 Karmada、Federation v2）允许跨不同云或数据中心统一管理。" ;;
    97) statement="Cluster API 是一个 Kubernetes 子项目，提供声明式 API 来创建、配置和管理集群。" ;;
    98) statement="Cilium 利用 eBPF 技术提供高效的网络、可观测性和安全策略，支持服务网格。" ;;
    99) statement="eBPF 是一项革命性的 Linux 内核技术，允许在不修改内核的情况下运行沙盒程序。" ;;
    100) statement="WebAssembly（WASM）被视为下一代云原生运行时，可能替代部分容器工作负载。" ;;
  esac

  # 将语句追加到临时文件，需要转义内部引号（如果语句中包含双引号，这里已确保没有）
  # 但为了安全，使用 printf 处理可能的特殊字符，并将结果用双引号括起来
  # 由于语句来自 case，我们假设它们没有双引号，可直接追加
  if [ $i -eq 100 ]; then
    # 最后一条不加逗号
    printf '    "%s"\n' "$statement" >> "$TMP_JSON"
  else
    printf '    "%s",\n' "$statement" >> "$TMP_JSON"
  fi
done

# 写入 JSON 结尾
cat >> "$TMP_JSON" << 'EOF'
  ]
}
EOF

# 使用 curl 发送 POST 请求
echo "正在导入知识语句到 $API_URL ..."
curl -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d @"$TMP_JSON"

# 检查 curl 执行结果
if [ $? -eq 0 ]; then
  echo -e "\n导入完成。"
else
  echo -e "\n导入失败，请检查网络或API服务状态。"
fi

# 清理临时文件
rm -f "$TMP_JSON"