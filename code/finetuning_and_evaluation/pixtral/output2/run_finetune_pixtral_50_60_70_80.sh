#!/bin/bash
# Sequentially run Pixtral finetuning for 50%, 60%, 70%, and 80% subsets

set -e

cd "$(dirname "$0")"

run_step() {
  local fraction_script="$1"
  local label="$2"

  echo "\n==============================="
  echo "Starting Pixtral finetune ($label)"
  echo "===============================\n"

  deepspeed "$fraction_script" \
    --prompts_jsonl ../train_prompts_set2.jsonl \
    --images_json ../train_images_set2.json \
    --model_name mistral-community/pixtral-12b

  echo "\n==============================="
  echo "Completed Pixtral finetune ($label)"
  echo "===============================\n"
}

run_step fine_tune_Pixtral_lora_50.py "50%"
run_step fine_tune_Pixtral_lora_60.py "60%"
run_step fine_tune_Pixtral_lora_70.py "70%"
run_step fine_tune_Pixtral_lora_80.py "80%"

echo "All Pixtral finetune runs complete."
