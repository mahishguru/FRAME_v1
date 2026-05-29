#!/bin/bash
# Run Gemma-3-27B-IT inference via OpenRouter for both test sets

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

PROMPTS_SET1="$PROJECT_ROOT/test_prompts_set1.jsonl"
IMAGES_SET1="$PROJECT_ROOT/test_images_set1.json"
OUTPUT_SET1="$PROJECT_ROOT/gemma3_27b_results_set1.json"

PROMPTS_SET2="$PROJECT_ROOT/test_prompts_set2.jsonl"
IMAGES_SET2="$PROJECT_ROOT/test_images_set2.json"
OUTPUT_SET2="$PROJECT_ROOT/gemma3_27b_results_set2.json"

MODEL_NAME=${MODEL_NAME:-"google/gemma-3-27b-it"}
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-""}
MAX_TOKENS=${MAX_TOKENS:-3000}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
SITE_URL=${SITE_URL:-"https://github.com/kbali1297/fine_tune_llm_post_processing"}
SITE_NAME=${SITE_NAME:-"Fine-tune LLM Post Processing"}

run_inference() {
  local script="$1"
  local prompts="$2"
  local images="$3"
  local output="$4"
  local label="$5"

  echo "=============================================="
  echo "Running Gemma-3-27B-IT inference for ${label}"
  echo "=============================================="

  python3 "$SCRIPT_DIR/$script" \
    ${OPENROUTER_API_KEY:+--api_key "$OPENROUTER_API_KEY"} \
    --model "$MODEL_NAME" \
    --test_prompts_jsonl "$prompts" \
    --test_images_json "$images" \
    --output_json "$output" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --site_url "$SITE_URL" \
    --site_name "$SITE_NAME"
}

run_inference "infer_gemma3_27b_openrouter_set1.py" "$PROMPTS_SET1" "$IMAGES_SET1" "$OUTPUT_SET1" "Test Set 1"
run_inference "infer_gemma3_27b_openrouter_set2.py" "$PROMPTS_SET2" "$IMAGES_SET2" "$OUTPUT_SET2" "Test Set 2"

echo "\nAll Gemma-3-27B-IT inferences completed."
