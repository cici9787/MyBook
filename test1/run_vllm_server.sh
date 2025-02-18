#!/bin/bash

python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/models/qwen2-1.5b \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --api-key cici-beauty