#!/bin/bash

export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="dummy"

curl "$OPENAI_BASE_URL/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Medium-Llama-3.2-3B-Instruct",
    "prompt": "Say hello in one sentence.",
    "max_tokens": 64,
    "temperature": 0.7,
    "stream": false
  }' | jq -r '.choices[0].text'
