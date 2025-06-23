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
ray stop && ray start --head --port=6379 --dashboard-port=8265
HEAD_ADDRESS="$HEAD_IP:6379"


for WORKER_IP in "${WORKER_IPS[@]}"; do
  (ssh -o ConnectTimeout=10 "$WORKER_IP" "\
  ray stop && ray start --address=$HEAD_ADDRESS") &
done
wait

echo "Ray 集群启动完成！"
echo "Dashboard 地址: http://$HEAD_IP:8265"