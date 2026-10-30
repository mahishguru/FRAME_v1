#!/bin/bash
# Run LoRA inference for all 4 Qwen3-VL-8B checkpoints (p50 / p60 / p70 / p80)
# GPU available: 1 only
# Strategy: run all 4 jobs sequentially on GPU 1

set -e
cd "$(dirname "$0")"

PYTHON=/data/mguru/04_Finetuning/finetune/bin/python
SCRIPT=infer_qwen3_vl_lora.py
TEST_PROMPTS=../test_prompts_set2.jsonl
TEST_IMAGES=../test_images_set2.json
BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct

mkdir -p logs

echo "=========================================="
echo "Starting Qwen3-VL-8B LoRA inference"
echo "  All jobs run sequentially on GPU 1"
echo "=========================================="

run_job() {
    local pct=$1
    echo "[$(date '+%H:%M:%S')] Starting p${pct} on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT \
        --model_path Qwen3-VL-8B-Instruct_p${pct}/checkpoint-final \
        --base_model $BASE_MODEL \
        --test_prompts_jsonl $TEST_PROMPTS \
        --test_images_json $TEST_IMAGES \
        --output_json inference_results_p${pct}.json \
        --device cuda \
        --max_new_tokens 2048 \
        --temperature 0.7 \
        --top_p 0.8 \
        --top_k 20 \
        > logs/infer_p${pct}.log 2>&1
    echo "[$(date '+%H:%M:%S')] p${pct} done."
}

run_job 50
run_job 60
run_job 70
run_job 80

echo ""
echo "=========================================="
echo "All 4 inference runs complete."
echo "Results:"
echo "  inference_results_p50.json"
echo "  inference_results_p60.json"
echo "  inference_results_p70.json"
echo "  inference_results_p80.json"
echo "Logs: logs/infer_p{50,60,70,80}.log"
echo "=========================================="
