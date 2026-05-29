# FRAME: Finite element Reasoning and AI-Agent Model Engine

A multimodal benchmark and fine-tuning framework for evaluating and training Vision-Language Models (VLMs) on physics-grounded understanding of Finite Element Analysis (FEA) simulation results.

---

## Overview

FRAME addresses a critical gap in computational engineering: the lack of domain-specific benchmarks for evaluating whether AI models can *interpret* FEA results with the depth and accuracy of a senior engineer. While general-purpose multimodal benchmarks test textbook-style reasoning, FRAME targets the real-world task of reading contour plots, identifying physical phenomena, extracting critical values, and proposing actionable design modifications.

The project spans four stages:

1. **Data Mining & Ingestion** — Systematic collection of FEA studies from academic literature
2. **Data Filtering** — Multi-pass VLM-based filtering to retain only papers with genuine FE contour plots
3. **Data Structuring** — LLM-driven extraction of structured input/output JSON pairs from papers
4. **Fine-Tuning & Evaluation** — Training open-source VLMs and benchmarking against general-purpose models

---

## Repository Structure

```
frame-finetuning-evaluation/
├── final_input.json          # Benchmark inputs: problem descriptions (keys 1-4)
├── final_output.json         # Benchmark outputs: physics analysis + optimization (outputs 1-2)
├── format.txt                # Prompt template structure
├── run_all_finetunes.sh      # Master script to launch all fine-tuning jobs
│
├── pixtral/                  # Pixtral-12B fine-tuning data & inference
│   ├── generate_fine_tune_data.py
│   ├── train_prompts_set{1,2}.jsonl
│   ├── test_prompts_set{1,2}.jsonl
│   ├── train_images_set{1,2}.json
│   ├── test_images_set{1,2}.json
│   └── output{1,2}/         # Inference results per set
│
├── qwen-2.5/                 # Qwen3-VL-8B fine-tuning data & inference
│   └── (same structure as pixtral/)
│
├── llava_ov/                 # LLaVA-OneVision fine-tuning data & inference
│   └── (same structure as pixtral/)
│
├── general_purpose_models/   # Zero-shot evaluation of frontier models
│   ├── infer_gpt5_azure.py
│   ├── infer_claude_sonnet_azure_set{1,2}.py
│   ├── infer_gemini_openrouter.py
│   ├── infer_gemma3_27b_openrouter_set{1,2}.py
│   ├── infer_llama4_azure_set{1,2}.py
│   ├── infer_qwen3_vl_openrouter_set{1,2}.py
│   └── *_results_set{1,2}.json
│
└── evaluation/               # Metrics pipeline & analysis
    ├── run_metrics_pipeline.py      # Orchestrates parallel metric computation
    ├── compute_emb_cosine.py        # Sentence-transformer cosine similarity
    ├── compute_chunk_OT.py          # Optimal-transport chunk coverage
    ├── compute_scale_factuality.py  # SCALE factuality precision
    ├── compute_rouge.py             # ROUGE-L
    ├── compute_meteor.py            # METEOR
    ├── compute_llm_judge_vllm.py    # LLM-as-Judge (vLLM backend)
    ├── analyze_metrics.py           # Aggregation & statistical analysis
    ├── create_comparison_plots.py   # Publication-quality bar/radar charts
    ├── metrics_config.json          # Pipeline configuration
    ├── human_eval/                  # Human expert evaluation analysis
    ├── paper/                       # LaTeX manuscript & figures
    ├── output1/                     # Set 1 inference results & metrics
    └── output2/                     # Set 2 inference results & metrics
```

---

## Upstream Pipelines

### Data Filtering (`01_DataFiltering_02/`)

A three-pass filtering pipeline that selects papers containing genuine FE simulation contour plots:

| Pass | Method | Purpose |
|------|--------|---------|
| **First** | File-system check | Retain papers with both `infographic/` and `fulltext/` directories |
| **Second** | Qwen2-VL-7B classification | VLM asks: "Does this image contain a finite element contour plot?" |
| **Third** | Qwen2-VL-7B refined | Stricter prompt distinguishing FE plots from generic colored gradients |

### Data Structuring (`03_DataStructuring_Open/`)

Automated extraction of structured JSON input/output pairs using **Gemma-3-27B-IT**:

- **Input keys** (problem specification):
  - `key_1`: Component identity, material, application, geometry
  - `key_2`: Boundary conditions, loads, interactions, simulation type
  - `key_3`: Physics solved, target outputs/metrics, optimization goals
  - `key_4`: Design space, performance limits, process constraints

- **Output keys** (expert analysis):
  - `output_1`: Physical behavior — dominant fields, critical phenomena, failure modes, optimization strategies
  - `output_2`: Specific optimization recommendations — category, modification, location, rationale

---

## Benchmark Task

**Given:**
- FEA contour plot image(s) (e.g., von Mises stress, displacement, temperature)
- Structured problem metadata (material, BCs, objectives)

**Generate:**
- Expert-level physics analysis identifying critical phenomena
- Quantitative extraction of key values from the visualization
- Actionable, parameterized design improvement recommendations

---

## Models Evaluated

### General-Purpose (Zero-Shot)
| Model | Provider |
|-------|----------|
| GPT 5.1 | Azure OpenAI |
| Claude Sonnet 4.5 | Azure |
| Gemini 3 Pro | OpenRouter |
| Qwen3-VL-235B | OpenRouter |
| Llama4 Maverick | Azure |
| Gemma3-27B | OpenRouter |

### Fine-Tuned (LoRA)
| Model | Parameters | Method |
|-------|-----------|--------|
| Qwen3-VL-8B | 8B | LoRA |
| Pixtral-12B | 12B | LoRA + DeepSpeed |
| LLaVA-OneVision | 7B | LoRA |

---

## Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **ROUGE-L** | Longest common subsequence overlap |
| **METEOR** | Token-level alignment with synonyms/stems |
| **Cosine Similarity** | Semantic embedding similarity (sentence-transformers) |
| **OT Coverage** | Optimal-transport chunk-level coverage of ground truth |
| **SCALE Coverage** | Factuality precision via claim decomposition |
| **LLM-as-Judge** | GPT-based holistic scoring (0–5 scale) |

---

## Human Evaluation

Three domain experts independently rated benchmark datapoints on a 1–5 Likert scale across four criteria:
- **Correctness** — factual accuracy of extracted information
- **Completeness** — coverage of all relevant physics and parameters
- **Coherence** — logical consistency and structure
- **Faithfulness** — absence of hallucination vs. source paper

Results: Mean AC₂ = 0.91 (inter-expert agreement), 90% of scores ≥ 4.

---

## Quick Start

### Fine-Tuning

```bash
# Pixtral-12B with DeepSpeed (4 GPUs)
deepspeed --num_gpus=4 fine_tune_Pixtral_lora.py \
  --model_name "mistral-community/pixtral-12b" \
  --prompts_jsonl "../train_prompts_set1.jsonl" \
  --images_json "../train_images_set1.json" \
  --num_train_epochs 3 --batch_size 1 --accum 8 --lr 2e-4

# Qwen3-VL-8B
CUDA_VISIBLE_DEVICES=0,1,2,3 python fine_tune_Qwen_2.5_lora.py \
  --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
  --prompts_jsonl "../train_prompts_set2.jsonl" \
  --images_json "../train_images_set2.json" \
  --num_train_epochs 3 --batch_size 1 --accum 4 --lr 2e-4

# LLaVA-OneVision
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 fine_tune_LLava_OV_lora.py \
  --model_name "llava-hf/llava-onevision-qwen2-7b-ov-hf" \
  --prompts_jsonl "../train_prompts_set2.jsonl" \
  --images_json "../train_images_set2.json" \
  --num_train_epochs 3 --batch_size 1 --accum 4 --lr 2e-4
```

### Inference

```bash
# Fine-tuned Pixtral
CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_pixtral_lora.py \
  --model_path pixtral-12b/checkpoint-final \
  --test_prompts_jsonl ../test_prompts_set2.jsonl \
  --test_images_json ../test_images_set2.json \
  --output_json pixtral_inference_results.json \
  --max_new_tokens 2048 --temperature 0.7 --device auto

# General-purpose model (Gemini)
python infer_gemini_openrouter.py \
  --test_prompts_jsonl test_prompts_set1.jsonl \
  --test_images_json test_images_set1.json \
  --output_json gemini_results.json \
  --max_tokens 2048 --temperature 0.7
```

### Evaluation

```bash
cd evaluation/
python run_metrics_pipeline.py --config metrics_config.json
```

### Monitoring

```bash
tensorboard --logdir=./pixtral/output1/pixtral-12b/runs --port=6007
```

---

## Key Results

- Fine-tuned **Qwen3-VL-8B** achieves best overall ROUGE-L (0.374), METEOR (0.433), and Cosine Similarity (0.969)
- Fine-tuned **Pixtral-12B** achieves best SCALE Coverage (0.272) and LLM-as-Judge (3.792) among fine-tuned models
- **GPT 5.1** leads general-purpose models in LLM-as-Judge (4.320) and SCALE Coverage (0.256)
- Domain fine-tuning yields up to **+0.178 ROUGE-L** and **+1.797 LLM-as-Judge** improvements over base models

---

## Requirements

```
torch>=2.0
transformers>=4.40
deepspeed
sentence-transformers
scikit-learn
nltk
rouge-score
numpy
matplotlib
openai
anthropic
Pillow
tqdm
```

---

## Citation

```bibtex
@article{guru2025frame,
  title={FRAME (Finite element Reasoning and AI-Agent Model Engine): A Generative Physics Understanding Benchmark and Fine-Tuning Strategies},
  author={Guru, Mahish K. and Gupta, Vipul and Bali, Kartik and Linka, Kevin and Pyczak, Florian and Khalifa, Noomane Ben and Aydin, Roland C.},
  year={2025}
}
```

---

## License

Please refer to individual model licenses for fine-tuned weights. The benchmark dataset and evaluation code are released for academic research.
