#!/bin/bash
# Run Claude 4.5 Sonnet inference via Azure AI Foundry (Anthropic) for both test sets

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"

AZURE_ENDPOINT="https://llmpost9832750527.services.ai.azure.com/anthropic/"
AZURE_API_VERSION="2024-05-01-preview"
AZURE_API_KEY="FP5JvpTzN3SRbnjv1TWDZYOvp4znMKS6hpyt2lDlrYpnkuqegyokJQQJ99BCACfhMk5XJ3w3AAAAACOG1xDn"
DEPLOYMENT_NAME="claude-sonnet-4-5"

run_inference() {
  local label="$1"
  local script_name="$2"

  echo "=============================================="
  echo "Claude 4.5 Sonnet Azure Inference - ${label}"
  echo "=============================================="

  python3 "$SCRIPT_DIR/$script_name" \
    --endpoint "$AZURE_ENDPOINT" \
    --api_version "$AZURE_API_VERSION" \
    --api_key "$AZURE_API_KEY" \
    --deployment "$DEPLOYMENT_NAME"
}

run_inference "Test Set 1" "infer_claude_sonnet_azure_set1.py"
run_inference "Test Set 2" "infer_claude_sonnet_azure_set2.py"

echo "\n✅ Claude 4.5 Sonnet Azure inferences completed."
