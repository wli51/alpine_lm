#!/bin/bash

export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="dummy"

curl "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"unsloth/Llama-3.2-3B-Instruct",
    "messages":[{"role":"user","content":"Say hello in one sentence."}],
    "stream": false
  }' | jq -r '.choices[0].message.content'
