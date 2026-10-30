#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="/data/mguru/04_Finetuning/finetune/bin/activate"
QWEN_ROOT="/data/mguru/04_Finetuning/frame-finetuning-evaluation/qwen-2.5"

if [[ -f "${ENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_PATH}"
else
  echo "Warning: Python environment not found at ${ENV_PATH}. Proceeding without activating a venv." >&2
fi

echo "=== Running Qwen3-VL inference in output1 (fine-tuned + pretrained) ==="
cd "${QWEN_ROOT}/output1"

CUDA_VISIBLE_DEVICES=2 python infer_qwen3_vl_lora.py \
  --model_path "${QWEN_ROOT}/output1/Qwen3VL_set1/checkpoint-final" \
  --test_prompts_jsonl "${QWEN_ROOT}/test_prompts_set1.jsonl" \
  --test_images_json "${QWEN_ROOT}/test_images_set1.json" \
  --output_json "qwen3_vl_lora_set1_results.json"

CUDA_VISIBLE_DEVICES=2 python infer_qwen3_vl_pretrained.py \
  --test_prompts_jsonl "${QWEN_ROOT}/test_prompts_set1.jsonl" \
  --test_images_json "${QWEN_ROOT}/test_images_set1.json" \
  --output_json "qwen3_vl_base_set1_results.json"

echo "=== Running Qwen3-VL inference in output2 (fine-tuned + pretrained) ==="
cd "${QWEN_ROOT}/output2"

CUDA_VISIBLE_DEVICES=2 python infer_qwen3_vl_lora.py \
  --model_path "${QWEN_ROOT}/output2/Qwen3VL_set2/checkpoint-final" \
  --test_prompts_jsonl "${QWEN_ROOT}/test_prompts_set2.jsonl" \
  --test_images_json "${QWEN_ROOT}/test_images_set2.json" \
  --output_json "qwen3_vl_lora_set2_results.json"

CUDA_VISIBLE_DEVICES=2 python infer_qwen3_vl_pretrained.py \
  --test_prompts_jsonl "${QWEN_ROOT}/test_prompts_set2.jsonl" \
  --test_images_json "${QWEN_ROOT}/test_images_set2.json" \
  --output_json "qwen3_vl_base_set2_results.json"

echo "All Qwen3-VL inference jobs finished. Outputs saved in each output directory."
