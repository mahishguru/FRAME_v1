#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="/data/mguru/04_Finetuning/finetune/bin/activate"
LLAVA_ROOT="/data/mguru/04_Finetuning/frame-finetuning-evaluation/llava_ov"

if [[ -f "${ENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_PATH}"
else
  echo "Warning: Python environment not found at ${ENV_PATH}. Proceeding without activating a venv." >&2
fi

echo "=== Running LLaVA-OV inference in output1 (fine-tuned + base) ==="
cd "${LLAVA_ROOT}/output1"

CUDA_VISIBLE_DEVICES=0 python infer_llava_lora.py \
  --model_path "${LLAVA_ROOT}/output1/llava_ov_finetuned_set1/checkpoint-final" \
  --test_prompts_jsonl "${LLAVA_ROOT}/test_prompts_set1.jsonl" \
  --test_images_json "${LLAVA_ROOT}/test_images_set1.json" \
  --output_json "llava_ov_lora_set1_results.json"

CUDA_VISIBLE_DEVICES=0 python infer_llava_base.py \
  --test_prompts_jsonl "${LLAVA_ROOT}/test_prompts_set1.jsonl" \
  --test_images_json "${LLAVA_ROOT}/test_images_set1.json" \
  --output_json "llava_ov_base_set1_results.json"

echo "=== Running LLaVA-OV inference in output2 (fine-tuned + base) ==="
cd "${LLAVA_ROOT}/output2"

CUDA_VISIBLE_DEVICES=0 python infer_llava_lora.py \
  --model_path "${LLAVA_ROOT}/output2/llava_ov_finetuned_set2/checkpoint-final" \
  --test_prompts_jsonl "${LLAVA_ROOT}/test_prompts_set2.jsonl" \
  --test_images_json "${LLAVA_ROOT}/test_images_set2.json" \
  --output_json "llava_ov_lora_set2_results.json"

CUDA_VISIBLE_DEVICES=0 python infer_llava_base.py \
  --test_prompts_jsonl "${LLAVA_ROOT}/test_prompts_set2.jsonl" \
  --test_images_json "${LLAVA_ROOT}/test_images_set2.json" \
  --output_json "llava_ov_base_set2_results.json"

echo "All inference jobs finished. Outputs saved in each output directory."
