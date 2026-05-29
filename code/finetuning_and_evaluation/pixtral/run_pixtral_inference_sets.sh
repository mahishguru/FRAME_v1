#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="/data/mguru/04_Finetuning/finetune/bin/activate"
PIXTRAL_ROOT="/data/mguru/04_Finetuning/frame-finetuning-evaluation/pixtral"

if [[ -f "${ENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_PATH}"
else
  echo "Warning: Python environment not found at ${ENV_PATH}. Proceeding without activating a venv." >&2
fi

echo "=== Running Pixtral inference in output1 (fine-tuned + base) ==="
cd "${PIXTRAL_ROOT}/output1"

CUDA_VISIBLE_DEVICES=1 python infer_pixtral_lora.py \
  --model_path "${PIXTRAL_ROOT}/output1/pixtral-12b-set1/checkpoint-final" \
  --test_prompts_jsonl "${PIXTRAL_ROOT}/test_prompts_set1.jsonl" \
  --test_images_json "${PIXTRAL_ROOT}/test_images_set1.json" \
  --output_json "pixtral_lora_set1_results.json"

CUDA_VISIBLE_DEVICES=1 python infer_pixtral_base.py \
  --test_prompts_jsonl "${PIXTRAL_ROOT}/test_prompts_set1.jsonl" \
  --test_images_json "${PIXTRAL_ROOT}/test_images_set1.json" \
  --output_json "pixtral_base_set1_results.json"

echo "=== Running Pixtral inference in output2 (fine-tuned + base) ==="
cd "${PIXTRAL_ROOT}/output2"

CUDA_VISIBLE_DEVICES=1 python infer_pixtral_lora.py \
  --model_path "${PIXTRAL_ROOT}/output2/pixtral-12b-set2/checkpoint-final" \
  --test_prompts_jsonl "${PIXTRAL_ROOT}/test_prompts_set2.jsonl" \
  --test_images_json "${PIXTRAL_ROOT}/test_images_set2.json" \
  --output_json "pixtral_lora_set2_results.json"

CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_pixtral_base.py \
  --test_prompts_jsonl "${PIXTRAL_ROOT}/test_prompts_set2.jsonl" \
  --test_images_json "${PIXTRAL_ROOT}/test_images_set2.json" \
  --output_json "pixtral_base_set2_results.json"

echo "All Pixtral inference jobs finished. Outputs saved in each output directory."
