#!/bin/bash
# Run LoRA inference for all 4 Pixtral checkpoints (p50 / p60 / p70 / p80)
# GPUs available: 0, 2, 3
# Strategy: run p50/p60/p70 in parallel (one per GPU), then p80 on GPU 0
#            once p50 finishes.

set -e
cd "$(dirname "$0")"

PYTHON=/data/mguru/04_Finetuning/finetune/bin/python
SCRIPT=infer_pixtral_lora.py
TEST_PROMPTS=../test_prompts_set2.jsonl
TEST_IMAGES=../test_images_set2.json
BASE_MODEL=mistral-community/pixtral-12b

mkdir -p logs

echo "=========================================="
echo "Starting Pixtral LoRA inference"
echo "  p50  →  GPU 0 (background)"
echo "  p60  →  GPU 2 (background)"
echo "  p70  →  GPU 3 (background)"
echo "  p80  →  GPU 0 (after p50 finishes)"
echo "=========================================="

# ── p50 on GPU 0 ──────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
    --model_path pixtral-12b_p50/checkpoint-final \
    --base_model $BASE_MODEL \
    --test_prompts_jsonl $TEST_PROMPTS \
    --test_images_json $TEST_IMAGES \
    --output_json pixtral_lora_p50_results.json \
    --device cuda \
    > logs/infer_p50.log 2>&1 &
PID_P50=$!
echo "[$(date '+%H:%M:%S')] p50 started (PID $PID_P50) on GPU 0"

# ── p60 on GPU 2 ──────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT \
    --model_path pixtral-12b_p60/checkpoint-final \
    --base_model $BASE_MODEL \
    --test_prompts_jsonl $TEST_PROMPTS \
    --test_images_json $TEST_IMAGES \
    --output_json pixtral_lora_p60_results.json \
    --device cuda \
    > logs/infer_p60.log 2>&1 &
PID_P60=$!
echo "[$(date '+%H:%M:%S')] p60 started (PID $PID_P60) on GPU 2"

# ── p70 on GPU 3 ──────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=3 $PYTHON $SCRIPT \
    --model_path pixtral-12b_p70/checkpoint-final \
    --base_model $BASE_MODEL \
    --test_prompts_jsonl $TEST_PROMPTS \
    --test_images_json $TEST_IMAGES \
    --output_json pixtral_lora_p70_results.json \
    --device cuda \
    > logs/infer_p70.log 2>&1 &
PID_P70=$!
echo "[$(date '+%H:%M:%S')] p70 started (PID $PID_P70) on GPU 3"

# ── wait for p50, then run p80 on freed GPU 0 ─────────────────────────────────
echo "[$(date '+%H:%M:%S')] Waiting for p50 (GPU 0) to finish before starting p80..."
wait $PID_P50
echo "[$(date '+%H:%M:%S')] p50 done. Starting p80 on GPU 0"

CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
    --model_path pixtral-12b_p80/checkpoint-final \
    --base_model $BASE_MODEL \
    --test_prompts_jsonl $TEST_PROMPTS \
    --test_images_json $TEST_IMAGES \
    --output_json pixtral_lora_p80_results.json \
    --device cuda \
    > logs/infer_p80.log 2>&1 &
PID_P80=$!
echo "[$(date '+%H:%M:%S')] p80 started (PID $PID_P80) on GPU 0"

# ── wait for all remaining jobs ───────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Waiting for p60, p70, p80 to finish..."
wait $PID_P60 && echo "[$(date '+%H:%M:%S')] p60 done."
wait $PID_P70 && echo "[$(date '+%H:%M:%S')] p70 done."
wait $PID_P80 && echo "[$(date '+%H:%M:%S')] p80 done."

echo ""
echo "=========================================="
echo "All 4 inference runs complete."
echo "Results:"
echo "  pixtral_lora_p50_results.json"
echo "  pixtral_lora_p60_results.json"
echo "  pixtral_lora_p70_results.json"
echo "  pixtral_lora_p80_results.json"
echo "Logs: logs/infer_p{50,60,70,80}.log"
echo "=========================================="
