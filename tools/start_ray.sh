#!/bin/bash

HOSTFILE="/etc/mpi/hostfile"

# 解析 Head 节点 IP（取第一个IP）
HEAD_IP=$(head -n 1 "$HOSTFILE" | awk '{print $1}')

# 解析 Worker 节点 IP（排除第一行后取所有IP）
mapfile -t WORKER_IPS < <(tail -n +2 "$HOSTFILE" | awk '{print $1}')

# 打印解析结果
echo "Head 节点: $HEAD_IP"
echo "Worker 节点: ${WORKER_IPS[*]}"

# 启动 Head 节点
echo "正在启动 Head 节点 ($HEAD_IP)..."
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_MIN_NCHANNELS=16
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
ray stop
ray start --head --port=6379 --dashboard-port=8265
HEAD_ADDRESS="$HEAD_IP:6379"


for WORKER_IP in "${WORKER_IPS[@]}"; do
  ssh -o BatchMode=yes "$WORKER_IP" bash -l <<EOF &
conda activate archer
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_MIN_NCHANNELS=16
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
ray stop
ray start --address="$HEAD_ADDRESS"
EOF

done
wait

echo "Ray 集群启动完成！"
echo "Dashboard 地址: http://$HEAD_IP:8265"