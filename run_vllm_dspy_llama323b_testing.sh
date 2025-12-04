#!/bin/bash

set -euo pipefail

# --- Resolve repo root & python script path ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
PY_SCRIPT="$REPO_ROOT/py_scripts/dspy_test.py"

# Sanity print
echo "SCRIPT_DIR = $SCRIPT_DIR"
echo "PY_SCRIPT  = $PY_SCRIPT"

# --- ENV / MODEL CONFIG (only place you change per model) ---

eval "$(conda shell.bash hook)"
conda activate dspy-env

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# specify pathing to model on disk 
MODEL_DIR="/scratch/alpine/$USER/lms/unsloth-Llama-3.2-3B-Instruct"
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory does not exist: $MODEL_DIR"
    exit 1
fi
SERVED_MODEL_NAME="unsloth/Llama-3.2-3B-Instruct"

PORT="${PORT:-8000}"

# Expose config for Python script
export TOOL_CACHE_ROOT="/scratch/alpine/$USER/.tool_cache"
export MODEL_ID="$SERVED_MODEL_NAME"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY="local"
export LM_MAX_TOKENS="${LM_MAX_TOKENS:-1024}"
export LM_SEED="${LM_SEED:-42}"

export PRISM_DATA_PATH="/projects/$USER/data/PRISM_processed/overlap_set/"
if [ ! -d "$PRISM_DATA_PATH" ]; then
    echo "ERROR: PRISM data path does not exist: $PRISM_DATA_PATH"
    exit 1
fi

export MLFLOW_TRACKING_URI="http://your-mlflow:5000"
export MLFLOW_EXPERIMENT_NAME="my_memless_experiment"
export N_REPLICATES=10
export MASTER_SEED=42

echo "Starting vLLM with:"
echo "  MODEL_DIR         = $MODEL_DIR"
echo "  SERVED_MODEL_NAME = $SERVED_MODEL_NAME"
echo "  MODEL_ID (for DSPy) = $MODEL_ID"
echo "  OPENAI_BASE_URL   = $OPENAI_BASE_URL"

# --- START vLLM IN BACKGROUND ---

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype auto \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --swap-space 24 \
  > "$REPO_ROOT/vllm_server.log" 2>&1 &

VLLM_PID=$!
echo "Started vLLM server (PID $VLLM_PID) on port $PORT"

# --- WAIT FOR SERVER READINESS ---

echo "Waiting for vLLM server to be ready..."
for i in {1..180}; do
  if curl -s "${OPENAI_BASE_URL}/models" >/dev/null 2>&1; then
    echo "vLLM is up!"
    break
  fi
  sleep 5
done

if ! curl -s "${OPENAI_BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: vLLM server never became ready. Exiting."
  kill "$VLLM_PID" || true
  exit 1
fi

# --- RUN YOUR DSPy SCRIPT (NO HARD-CODED CONFIG INSIDE) ---

python "$PY_SCRIPT"

# --- CLEANUP ---

echo "Shutting down vLLM (PID $VLLM_PID)"
kill "$VLLM_PID" || true
wait "$VLLM_PID" 2>/dev/null || true

echo "Done."
