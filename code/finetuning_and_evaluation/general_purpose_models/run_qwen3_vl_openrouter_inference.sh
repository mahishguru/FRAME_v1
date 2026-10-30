#!/bin/bash
# Run Qwen3-VL-235B inference via OpenRouter for both test sets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "Qwen3-VL-235B Inference via OpenRouter"
echo "=============================================="

# Run Set 1
echo ""
echo ">>> Running Set 1 inference..."
python3 "$SCRIPT_DIR/infer_qwen3_vl_openrouter_set1.py" \
    --test_prompts_jsonl="test_prompts_set1.jsonl" \
    --test_images_json="test_images_set1.json" \
    --output_json="qwen3_vl_openrouter_results_set1.json"

# Run Set 2
echo ""
echo ">>> Running Set 2 inference..."
python3 "$SCRIPT_DIR/infer_qwen3_vl_openrouter_set2.py" \
    --test_prompts_jsonl="test_prompts_set2.jsonl" \
    --test_images_json="test_images_set2.json" \
    --output_json="qwen3_vl_openrouter_results_set2.json"

echo ""
echo "=============================================="
echo "✅ All inference complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - $SCRIPT_DIR/qwen3_vl_openrouter_results_set1.json"
echo "  - $SCRIPT_DIR/qwen3_vl_openrouter_results_set2.json"
