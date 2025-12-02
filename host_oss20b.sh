#!/bin/bash

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate dspy-env

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${1:-0}"

MODEL_DIR="/scratch/alpine/$USER/lms/openai-gpt-oss-20b"
PORT=8000

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --host 127.0.0.1 \
  --port $PORT \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype auto \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --swap-space 24 \
  > vllm_server.log 2>&1 &

VLLM_PID=$!

echo "Started vLLM server hosting openai/gpt-oss-20b with PID $VLLM_PID on port $PORT"
