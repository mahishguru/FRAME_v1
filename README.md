# FRAME: Finite Element Reasoning and AI-Agent Model Engine

**A Generative Physics Understanding Benchmark and Fine-Tuning Framework**

This repository contains the supplementary code, data, and evaluation results for the FRAME benchmark paper. FRAME is a curated multimodal benchmark and fine-tuning framework for physics-grounded interpretation of Finite Element Analysis (FEA) studies, containing 4,363 structured datapoints.

---

## Repository Structure

```
.
├── FRAME_input_keys.json / .csv      # Benchmark input keys (4,363 datapoints)
├── FRAME_output_keys.json / .csv     # Benchmark output keys (4,363 datapoints)
├── code/                             # All source code
│   ├── data_filtering/               # Multi-pass data filtering pipeline
│   ├── data_structuring/             # LLM-driven structured extraction pipeline
│   └── finetuning_and_evaluation/    # Fine-tuning, inference, and metrics
└── Evaluation_Results/               # Model outputs and evaluation results
    ├── output1/                      # Set 1 inference results (all models)
    ├── output2/                      # Set 2 inference results (all models)
    ├── human_eval/                   # Human expert evaluation data and analysis
    └── manifold_analysis/            # LoRA weight-space manifold analysis
```

---

## Benchmark Data

| File | Description |
|------|-------------|
| `FRAME_input_keys.json` / `.csv` | Structured simulation context for each datapoint: component definitions (key_1), boundary conditions (key_2), simulation objectives (key_3), and design constraints (key_4) |
| `FRAME_output_keys.json` / `.csv` | Reference outputs: physical behavior analysis (output_1) and optimization strategy generation (output_2) |

Each datapoint is keyed by a normalized DOI identifier (e.g., `10_1007_s43236_022_00530_x`).

---

## Code

### 1. Data Filtering (`code/data_filtering/`)

A multi-pass pipeline that selects papers containing genuine FE simulation contour plots from a larger corpus:

| Pass | Script | Purpose |
|------|--------|---------|
| First | `first_pass.py` | File-system check for papers with figures and full text |
| Second | `second_pass.py` | VLM-based classification: "Does this image contain an FE contour plot?" |
| Third | `third_pass.py`, `third_pass_imagefilter.py` | Refined filtering and manual review |

Supporting files: `delete_images.py`, `move.py` for dataset curation.

### 2. Data Structuring (`code/data_structuring/`)

Automated extraction pipeline using Qwen3-VL-235B to process filtered papers into structured input/output JSON pairs via multi-turn conversations.

| File | Purpose |
|------|---------|
| `main.py` | Orchestrator: iterates over papers, manages multi-turn extraction |
| `config/` | System prompts and key-specific extraction prompts |
| `services/` | vLLM API client for Qwen3-VL inference |
| `fallback_openrouter.py` | Fallback to OpenRouter API when local GPU unavailable |
| `join_jsons.py` | Merges per-paper JSONs into final benchmark files |
| `vllm_server_command.sh` | Command to launch the vLLM server (4x GPU, FP8) |

### 3. Fine-Tuning and Evaluation (`code/finetuning_and_evaluation/`)

Training and evaluation of vision-language models on FRAME.

#### Fine-Tuned Models

| Directory | Model | Notes |
|-----------|-------|-------|
| `pixtral/` | Pixtral-12B | LoRA fine-tuning data generation + inference |
| `qwen-2.5/` | Qwen3-VL-8B | LoRA fine-tuning data generation + inference |
| `llava_ov/` | LLaVA-OneVision | LoRA fine-tuning + inference |

Each contains: `generate_fine_tune_data.py`, training/test prompt JSONL files, image manifests, and inference output directories.

#### General-Purpose Model Evaluation (`general_purpose_models/`)

Zero-shot inference scripts for frontier models:

- GPT 5.1 (Azure)
- Claude Sonnet (Azure)
- Gemini 2.5 Pro (OpenRouter)
- Gemma3-27B (OpenRouter)
- Llama 4 Maverick (Azure)
- Qwen3-VL-Large (OpenRouter)

#### Evaluation Metrics (`evaluation/`)

A parallel metrics pipeline computing:

| Metric | Script | Description |
|--------|--------|-------------|
| Embedding cosine | `compute_emb_cosine.py` | Sentence-transformer similarity |
| Chunk OT | `compute_chunk_OT.py` | Optimal-transport chunk coverage |
| SCALE factuality | `compute_scale_factuality.py` | Factual precision/recall |
| ROUGE-L | `compute_rouge.py` | Lexical overlap |
| METEOR | `compute_meteor.py` | Alignment-based metric |
| LLM-as-Judge | `compute_llm_judge_vllm.py` | Structured quality scoring |

Run via: `run_metrics_pipeline.py` with `metrics_config.json`.

---

## Evaluation Results

### Model Inference Outputs (`Evaluation_Results/output1/`, `output2/`)

Raw inference results from all evaluated models (JSON format), organized by test set:

| Model | Set 1 | Set 2 |
|-------|-------|-------|
| GPT 5.1 | `gpt5_results_set1.json` | `gpt5_results_set2.json` |
| Claude Sonnet | `claude_sonnet_results_set1.json` | `claude_sonnet_results_set2.json` |
| Gemini 2.5 Pro | `gemini_results_set1.json` | `gemini_results_set2.json` |
| Gemma3-27B | `gemma3_27b_results_set1.json` | `gemma3_27b_results_set2.json` |
| Llama 4 Maverick | `llama4_maverick_results_set1.json` | `llama4_maverick_results_set2.json` |
| Qwen3-VL (Base) | `qwen3_vl_base_set1_results.json` | `qwen3_vl_base_set2_results.json` |
| Qwen3-VL (LoRA) | `qwen3_vl_lora_set1_results.json` | `qwen3_vl_lora_set2_results.json` |
| Pixtral (Base) | `pixtral_base_set1_results.json` | `pixtral_base_set2_results.json` |
| Pixtral (LoRA) | `pixtral_lora_set1_results.json` | `pixtral_lora_set2_results.json` |
| LLaVA-OV (Base) | `llava_ov_base_set1_results.json` | `llava_ov_base_set2_results.json` |
| LLaVA-OV (LoRA) | `llava_ov_lora_set1_results.json` | `llava_ov_lora_set2_results.json` |

Computed metrics are in the `results/` subdirectory of each output folder.

### Human Expert Evaluation (`Evaluation_Results/human_eval/`)

Three-expert audit of benchmark quality. Ratings on Correctness, Completeness, Coherence, and Faithfulness (1-5 scale).

- `Evaluation_clean.csv`, `Evaluation_KL_clean.csv`, `Evaluation_NBK_clean.csv` -- expert rating data
- `analyze_human_eval.py` -- generates summary statistics, inter-rater reliability (Gwet's AC2), and publication figures
- `results/` -- output figures (Likert distributions, reliability plots) and summary tables

### Manifold Analysis (`Evaluation_Results/manifold_analysis/`)

LoRA weight-space geometry analysis for fine-tuning trajectories of Qwen3-VL and Pixtral.

- `multi_run_manifold.py` -- computes Frobenius inner-product kernel over LoRA checkpoints
- `plot_manifold.py` -- renders PCA/t-SNE manifold visualizations
- `qwen_lora_manifold.py` -- Qwen-specific manifold computation
- Precomputed results in `multi_run_manifold_results/`, `pixtral_multi_run_manifold_results/`, `qwen3vl_manifold/`

---

## Key Results (from the paper)

- **90.0%** of human expert ratings are 4 or 5; mean Gwet's AC2 = 0.91
- **GPT 5.1** achieves strongest general-purpose performance: 4.732/5 (problem understanding), 4.320/5 (optimization strategy)
- **Domain adaptation** most strongly improves optimization generation: Pixtral-12B judge score increases from 1.986 to 3.799 after FRAME fine-tuning

---

## Requirements

See `code/data_structuring/requirements.txt` and `code/finetuning_and_evaluation/requirements_pip.txt` for Python dependencies.

Core dependencies: `torch`, `transformers`, `vllm`, `openai`, `peft`, `datasets`, `scikit-learn`, `nltk`, `rouge-score`, `sentence-transformers`.

---

## Citation

If you use FRAME in your research, please cite:

```
Guru, M.K., Gupta, V., Bali, K., Linka, K., Pyczak, F., Ben Khalifa, N., & Aydin, R.C.
FRAME (Finite element Reasoning and AI-Agent Model Engine):
A Generative Physics Understanding Benchmark and Fine-Tuning.
```

---

## Contact

Corresponding author: Mahish K. Guru (mahish.guru@hereon.de)

Institute of Material and Process Design, Helmholtz-Zentrum Hereon, Germany
