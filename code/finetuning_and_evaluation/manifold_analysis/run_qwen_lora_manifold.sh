#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/mguru/04_Finetuning/fine_tune_llm_post_processing"
PYTHON_BIN="/data/mguru/04_Finetuning/finetune/bin/python"

cd "$REPO_ROOT"
"$PYTHON_BIN" evaluation/manifold_analysis/qwen_lora_manifold.py "$@"
