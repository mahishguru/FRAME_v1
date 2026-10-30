#!/bin/bash
# Script to sequentially run 50% and then 80% Qwen3-VL finetuning with DeepSpeed

set -e

# Activate your Python environment if needed
# source /data/1guru/04_Finetuning/finetune_qwen3/bin/activate

# Move to the script directory
cd "$(dirname "$0")"

# Run 50% finetune

# echo "[INFO] Starting 50% finetune..."
# deepspeed fine_tune_Qwen3_VL_lora_50.py \
#   --train_prompts ../train_prompts_set2.jsonl \
#   --train_images ../train_images_set2.json \
#   --model_name Qwen/Qwen3-VL-8B-Instruct

# After 50% is done, run 80%
echo "[INFO] 50% finetune complete. Starting 80% finetune..."
echo "[INFO] Resuming from checkpoint: Qwen3-VL-8B-Instruct_p80/checkpoint-250"
deepspeed fine_tune_Qwen3_VL_lora_80.py \
  --train_prompts ../train_prompts_set2.jsonl \
  --train_images ../train_images_set2.json \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --resume_from_checkpoint Qwen3-VL-8B-Instruct_p80/checkpoint-250

echo "[INFO] Both finetunes complete."
