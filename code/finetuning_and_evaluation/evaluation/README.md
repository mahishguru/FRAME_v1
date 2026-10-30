# Metrics Evaluation Pipeline

This directory contains a Python-based parallel evaluation pipeline for computing text generation metrics.

## Overview

The evaluation pipeline consists of:
- **7 primary metric scripts** (Qwen-Embedding, Qwen-Chunk, SBERT-Chunk, ROUGE-L, METEOR, NLI-Factuality, LLM-as-Judge)
- **1 analysis script** for generating reports
- **1 config-driven pipeline** for parallel execution (`run_metrics_pipeline.py`)

## Directory Structure

```
evaluation/
├── compute_qwen_embedding.py # Qwen embedding similarity (sentence-level)
├── compute_qwen_chunk.py     # Qwen chunk-level recall/precision/F1
├── compute_sbert_chunk.py    # SBERT chunk-level recall/precision/F1
├── compute_rouge.py          # ROUGE-L score computation
├── compute_meteor.py         # METEOR score computation
├── compute_nli_factuality.py # NLI-based factuality metrics
├── compute_llm_judge.py      # LLM-as-Judge evaluation (Azure OpenAI)
├── analyze_metrics.py        # Analysis and reporting
├── run_metrics_pipeline.py   # Config-driven parallel pipeline
├── output1/
│   └── results/              # Results for output_1
│       ├── metrics_output_1.json
│       ├── report_output_1.json
│       └── plots/
└── output2/
    └── results/              # Results for output_2
        ├── metrics_output_2.json
        ├── report_output_2.json
        └── plots/
```

## Features

### Incremental Saving
- Each metric script saves results **incrementally** (every 10 samples)
- If a script is interrupted, it can resume from the last saved state
- All scripts write to the **same JSON file**, adding their metric to existing entries

### Null Handling
- Automatically skips cases where `generated_output` is `null` or empty
- Only processes valid inference results

### Ground Truth
- Reads ground truth from: `/data/mguru/04_Finetuning/frame-finetuning-evaluation/final_output.json`
- Supports both `output_1` and `output_2` comparisons

## Quick Start

Edit `metrics_config.json` to specify your inference files:

```
{
  "ground_truth": "/path/to/final_output.json",
  "max_parallel_jobs": 4,
  "inference_jobs": [
    {
      "name": "YourModel",
      "inference_results": "/path/to/inference_results.json",
      "output_key": "output_1",
      "results_dir": "evaluation/output1/results"
    }
  ],
  "metrics": [
    "qwen_embedding",
    "qwen_chunk",
    "sbert_chunk",
    "nli_factuality",
    "rouge",
    "meteor",
    "llm_judge"
  ]
}
```
```

Then run the pipeline:

```bash
# Dry run to verify
python run_metrics_pipeline.py --dry-run

# Execute
python run_metrics_pipeline.py
```

**Note:** GPUs are automatically assigned in round-robin fashion. No need to specify GPU IDs in config.

## Run Individual Metrics (Optional)

### 1. Qwen Embedding
```bash
python compute_qwen_embedding.py \
  --inference_results /path/to/inference_results.json \
  --output_key output_1 \
  --results_json evaluation/output1/results/metrics_output_1.json
```

### 2. Qwen Chunk
```bash
python compute_qwen_chunk.py \
  --inference_results /path/to/inference_results.json \
  --output_key output_1 \
  --results_json evaluation/output1/results/metrics_output_1.json \
  --chunk_size 1
```

### 3. SBERT Chunk
```bash
python compute_sbert_chunk.py \
  --inference_results /path/to/inference_results.json \
  --output_key output_1 \
  --results_json evaluation/output1/results/metrics_output_1.json \
  --chunk_size 1
```

### 4. ROUGE-L
```bash
python compute_rouge.py \
    --inference_results /path/to/inference_results.json \
    --output_key output_1 \
    --results_json evaluation/output1/results/metrics_output_1.json
```

### 5. METEOR
```bash
python compute_meteor.py \
  --inference_results /path/to/inference_results.json \
  --output_key output_1 \
  --results_json evaluation/output1/results/metrics_output_1.json
```

### 6. NLI Factuality
```bash
python compute_nli_factuality.py \
    --inference_results /path/to/inference_results.json \
    --output_key output_1 \
    --results_json evaluation/output1/results/metrics_output_1.json \
    --top_k 10
```

### 7. LLM-as-Judge
```bash
python compute_llm_judge.py \
    --inference_results /path/to/inference_results.json \
    --output_key output_1 \
    --results_json evaluation/output1/results/metrics_output_1.json \
    --azure_endpoint "https://your-resource.openai.azure.com/" \
    --azure_api_key "your-api-key" \
    --model "gpt-4o" \
    --max_retries 3
```

## Metrics Computed

### Qwen-Embedding Score
- **Description**: Cosine similarity between Qwen3-Embedding-8B representations of full outputs and references
- **Range**: 0.0 to 1.0 (higher is better)
- **Use case**: Sentence-level semantic alignment

### Qwen Chunk Metrics
- **Description**: Chunk-level hybrid F1 using Qwen embeddings with instruction prefix
- **Recall**: Hungarian matching across chunk pairs (hard, one-to-one)
- **Precision**: Max-sim thresholding per generated chunk (soft, one-to-many)
- **Outputs**: `Qwen_Recall`, `Qwen_Precision`, `Qwen_F1`, `Qwen_MatchedSimilarity`

### SBERT Chunk Metrics
- **Description**: Same hybrid strategy as Qwen Chunk but using SBERT embeddings (no instruction prompt)
- **Model**: `all-MiniLM-L6-v2`
- **Outputs**: `SBERT_Chunk_Recall`, `SBERT_Chunk_Precision`, `SBERT_Chunk_F1`, `SBERT_Chunk_MatchedSimilarity`

### ROUGE-L
- **Description**: Longest Common Subsequence-based F1 score
- **Range**: 0.0 to 1.0 (higher is better)
- **Implementation**: Uses stemming for better matching

### METEOR
- **Description**: Precision/recall harmonic mean with WordNet synonyms
- **Range**: 0.0 to 1.0 (higher is better)
- **Features**: Handles synonyms, stemming, and paraphrasing

### NLI Factuality
- **Description**: Entailment-based factuality using `cross-encoder/nli-deberta-v3-base`
- **Range**: 0.0 to 1.0 (higher is better)
- **Metrics**:
  - **NLI_Recall**: Coverage – "Does generated text prove all GT facts?"
  - **NLI_Precision**: Hallucination check – "Are all generated facts supported by GT?"
  - **NLI_F1**: Harmonic mean of recall and precision
- **Method**: Uses top-K semantic similarity pre-filtering, then computes entailment scores

### LLM-as-Judge
- **Description**: Uses Azure OpenAI (GPT-4o) to evaluate generated text against ground truth
- **Range**: 1-5 for each dimension (higher is better)
- **Metrics**:
  - **LLM_Judge_Faithfulness**: Hallucination check – numerical accuracy and factual consistency
  - **LLM_Judge_Completeness**: Recall – captures all key parameters and strategies
  - **LLM_Judge_Reasoning**: Physics understanding – cause-effect relationships
  - **LLM_Judge_Total**: Average of the three scores (normalized to 1-5 scale)
- **Features**: 
  - JSON schema validation with automatic retry on malformed responses
  - Engineering-specific rubric (strict on numbers, rewards valid reasoning)
  - Stores both scores and textual rationales

## Output Files

### Metrics JSON (`metrics_output_1.json` or `metrics_output_2.json`)
```json
{
  "input_case_id": {
    "input_case": "input_case_id",
    "output_key": "output_1",
    "Qwen_Recall": 0.82,
    "Qwen_Precision": 0.76,
    "Qwen_F1": 0.79,
    "SBERT_Chunk_F1": 0.74,
    "ROUGE_L": 0.61,
    "METEOR": 0.58,
    "NLI_F1": 0.81,
    "LLM_Judge_Faithfulness": 4,
    "LLM_Judge_Completeness": 5,
    "LLM_Judge_Reasoning": 4,
    "LLM_Judge_Total": 4.33
  }
}
```

### Report JSON (`report_output_1.json` or `report_output_2.json`)
```json
{
  "output_key": "output_1",
  "total_samples": 100,
  "metrics": {
    "SBERT_SCORE": {
      "mean": 0.8234,
      "median": 0.8456,
      "std": 0.0823,
      "min": 0.6234,
      "max": 0.9823,
      "count": 100
    }
  }
}
```

## Requirements

```bash
pip install sentence-transformers rouge-score moverscore-v2 numpy matplotlib seaborn tqdm
```
