#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
vllm serve "$MODEL" \
  --port 8000 \
  --max-model-len "${MAX_LEN:-16384}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --seed 0
