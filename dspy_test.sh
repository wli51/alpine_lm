#!/bin/bash

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate dspy-env

echo "Waiting for vLLM server to be ready..."
for i in {1..30}; do
  if curl -s "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1; then
    echo "vLLM is up!"
    break
  fi
  sleep 5
done

python dspy_test.py
