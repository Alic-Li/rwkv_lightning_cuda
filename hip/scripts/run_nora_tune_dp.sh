#!/usr/bin/env bash
set -euo pipefail

trainer="./build-hip/rwkv_nora_tune_dp"
model="/home/rwkv/alic-li/rwkv_weights/rwkv7-g1l-7.2b-20261031-ctx34304.pth"
data="/home/rwkv/alic-li/train_sample.jsonl"
vocab="./assets/rwkv_vocab_v20230424.txt"
output="./out/nora-dp-smoke"


for file in "$model" "$data" "$vocab"; do
    [[ -f "$file" && -r "$file" ]] || { echo "无法读取文件 $file" >&2; exit 1; }
done

# 沿用 train_amd.sh 的 ROCm 环境。DEVICES 使用清除可见性过滤后的设备编号。
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/opt/rocm/lib"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export HSA_XNACK=0 HSA_NO_SCRATCH_RECLAIM=1 HSA_FORCE_FINE_GRAIN_PCIE=1
export HIP_FORCE_DEV_KERNARG=1 GPU_MAX_HW_QUEUES=2
devices="${DEVICES:-0,1,2,3,4,5,6,7}"
IFS=, read -ra gpu_ids <<< "$devices"
export WORLD_SIZE="${#gpu_ids[@]}"
RWKV_RENDEZVOUS=$(mktemp -d)
export RWKV_RENDEZVOUS
trap 'rmdir -- "$RWKV_RENDEZVOUS"' EXIT

# _dp 父进程按 --devices 创建 worker 并管理各 rank。
"$trainer" \
    --model "$model" --vocab "$vocab" --data "$data" --output "$output" \
    --devices "$devices" --batch-size "32" --micro-bsz 2 \
    --ctx 20480 --chunk 20480  \
    --targets all --rank 16 --alpha 16 --lr 0.00002 --lr-final 0.000002 --gemm-mode auto \
    "$@"
