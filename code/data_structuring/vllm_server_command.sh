#!/bin/bash

# vLLM Server Command for Multimodal Inference
# This script starts a vLLM server with multimodal support for the Qwen3-VL model

# Set CUDA devices to use (GPUs 2 and 3)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Start the vLLM server with multimodal support
vllm serve QuantTrio/Qwen3-VL-235B-A22B-Thinking-FP8 \
  --tensor-parallel-size 4 \
  --max-model-len 250000 \
  --limit-mm-per-prompt '{"image":50}' \
  --allowed-local-media-path /data/mguru/01_DataFiltering \
  --reasoning-parser deepseek_r1 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95


# # Previous Gemma configuration (deprecated):
# # vllm serve google/gemma-3-27b-it \
# #   --tensor-parallel-size 2 \
# #   --dtype bfloat16 \
# #   --max-model-len 120000 \
# #   --limit-mm-per-prompt '{"image":50}' \
# #   --allowed-local-media-path /data/mguru/01_DataFiltering \
# #   --trust-remote-code \
# #   --port 8000

# Notes:
# --limit-mm-per-prompt '{"image":50}': Allows up to 50 images per prompt
# --allowed-local-media-path: Directory where images are stored 
# --tensor-parallel-size 4: Uses 4 GPUs for model parallelism
# --trust-remote-code: Required for some models with custom code
# --reasoning-parser deepseek_r1: Enables chain-of-thought reasoning extraction

# To run:
# 1. Make executable: chmod +x vllm_server_command.sh
# 2. Execute: ./vllm_server_command.sh

# Before running main.py, make sure the vLLM server is running!
