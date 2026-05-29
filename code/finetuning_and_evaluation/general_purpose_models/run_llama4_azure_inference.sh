#!/bin/bash
# Run Llama-4-Maverick inference via Azure AI Foundry for both test sets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AZURE_ENDPOINT="https://llmpost9832750527.services.ai.azure.com/models"
AZURE_API_VERSION="2024-05-01-preview"
AZURE_API_KEY="FP5JvpTzN3SRbnjv1TWDZYOvp4znMKS6hpyt2lDlrYpnkuqegyokJQQJ99BCACfhMk5XJ3w3AAAAACOG1xDn"

echo "=============================================="
echo "Llama-4-Maverick Azure AI Foundry Inference"
echo "=============================================="

# Run Set 1
echo ""
echo ">>> Running Set 1 inference..."
python3 "$SCRIPT_DIR/infer_llama4_azure_set1.py" \
	--endpoint "$AZURE_ENDPOINT" \
	--api_key "$AZURE_API_KEY" \
	--api_version "$AZURE_API_VERSION"

# Run Set 2
echo ""
echo ">>> Running Set 2 inference..."
python3 "$SCRIPT_DIR/infer_llama4_azure_set2.py" \
	--endpoint "$AZURE_ENDPOINT" \
	--api_key "$AZURE_API_KEY" \
	--api_version "$AZURE_API_VERSION"

echo ""
echo "=============================================="
echo "✅ All inference complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - $SCRIPT_DIR/llama4_maverick_results_set1.json"
echo "  - $SCRIPT_DIR/llama4_maverick_results_set2.json"
