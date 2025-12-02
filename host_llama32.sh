#!/bin/bash

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate dspy-env

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${1:-0}"

MODEL_DIR="/scratch/alpine/wli19@xsede.org/lms/unsloth-Llama-3.2-3B-Instruct"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype auto \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --swap-space 24
