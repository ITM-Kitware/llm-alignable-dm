#!/bin/bash

YOUR_API_HOST=localhost
YOUR_API_PORT=8080
YOUR_CODE_EXEC_ENGINE_HOST=localhost
YOUR_CODE_EXEC_ENGINE_PORT=8081

# Make sure you started model server (vLLM or llama.cpp) and code execution engine before running this!
set -x
python3 scripts/codeact_demo.py \
    --model-name CodeActAgent-Mistral-7b-v0.1 \
    --openai-base-url "http://$YOUR_API_HOST:$YOUR_API_PORT/v1" \
    --jupyter-kernel-url "http://$YOUR_CODE_EXEC_ENGINE_HOST:$YOUR_CODE_EXEC_ENGINE_PORT/execute"
