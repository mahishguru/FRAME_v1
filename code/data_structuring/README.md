# Engineering Data Structuring Pipeline (Qwen3-VL)

## Overview

This project is an automated **engineering data extraction pipeline** that processes research papers (text + figures) into structured engineering knowledge using the **Qwen3-VL** large vision-language model served via **vLLM**.

Given a collection of filtered research papers (each consisting of full text and associated figures with captions), the pipeline extracts structured engineering data organized into two categories:

- **Input Keys (Simulation Setup):** Component definitions, boundary conditions, simulation objectives, and constraints.
- **Output Keys (Analysis Results):** System behavior/physics, quality metrics, failure modes, and optimization strategies.

The model processes each paper in a multi-turn conversational format, building context across keys to produce coherent, physics-grounded extractions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     main.py (Orchestrator)                    │
│  - Iterates over datapoint folders                           │
│  - Manages incremental saving & failure tracking             │
│  - Multi-turn conversation builder per paper                 │
└───────────────┬─────────────────────────────────┬───────────┘
                │                                 │
    ┌───────────▼───────────┐         ┌───────────▼───────────┐
    │  config/prompts.py     │         │  services/qwen_service│
    │  - System prompts      │         │  - vLLM API client    │
    │  - Input key prompts   │         │  - OpenAI-compatible  │
    │  - Output key prompts  │         │    chat completions   │
    └────────────────────────┘         └───────────┬───────────┘
                                                   │
                                       ┌───────────▼───────────┐
                                       │  vLLM Server           │
                                       │  Qwen3-VL-235B-A22B   │
                                       │  (FP8, 4x GPU TP)     │
                                       └────────────────────────┘
```

---

## Model

- **Primary:** `QuantTrio/Qwen3-VL-235B-A22B-Thinking-FP8` — a 235B parameter Mixture-of-Experts vision-language model (22B active) with chain-of-thought reasoning, quantized to FP8.
- **Fallback:** `qwen/qwen3-vl-235b-a22b-thinking` via OpenRouter API (see `fallback_openrouter.py`).
- **Local weights cached:** `huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Thinking/` (smaller variant for testing).

The model is served via vLLM's OpenAI-compatible API endpoint with DeepSeek-R1-style reasoning parsing enabled.

---

## Data Flow

1. **Input:** Filtered research paper directories from `../01_DataFiltering/` containing:
   - `fulltext/fulltext.txt` — extracted paper text
   - `figure/` — paper figures (PNG/JPG)
   - `caption/figure_captions.json` — figure-caption mappings

2. **Processing:** For each paper, two multi-turn conversations are constructed:
   - **Input conversation** (4 turns): Extracts simulation setup — component definition, boundary conditions, objectives, and constraints.
   - **Output conversation** (2 turns): Extracts analysis results — system behavior/physics and optimization strategies.

3. **Output:** Structured JSON files:
   - `output/input.json` — all input key extractions indexed by paper ID
   - `output/output.json` — all output key extractions indexed by paper ID
   - `output/failed.txt` — failed datapoints for retry
   - `output/api_log.txt` — timestamped API call log

---

## Prompt Design (Theory)

The system uses role-specific system prompts to constrain model behavior:

### Input System Prompt
The model acts as a "high-precision technical data extraction engine" with rules:
- No yapping (no introductory phrases)
- Strict factuality (only stated/visible information)
- No implementation details (physics over code)
- Dense bullet-point format

### Output System Prompt
The model acts as a "Senior Lead Engineer" summarizing results with rules:
- Physics over code (describe physical changes, not software steps)
- Explicit reasoning for every observation
- Structured headers and bullet points

### Key Prompts
Each key prompt is a structured template that guides extraction of specific information categories. The multi-turn format allows the model to build on prior context (e.g., boundary conditions reference the component already defined in key_1).

---

## Project Structure

```
├── main.py                    # Main orchestrator — processes all datapoints
├── fallback_openrouter.py     # Fallback script using OpenRouter API
├── join_jsons.py              # Utility to merge split JSON output files
├── config/
│   ├── config.py              # General config (MAX_TOKENS, paths)
│   └── prompts.py             # All system & key prompts
├── services/
│   └── qwen_service.py        # Qwen vLLM API client (QwenService class)
├── utils/
│   └── file_utils.py          # File I/O helpers (read papers, figures, captions)
├── vllm_server_command.sh     # vLLM server launch script
├── output/                    # Extraction results
│   ├── input.json             # Structured input extractions
│   ├── output.json            # Structured output extractions
│   ├── failed.txt             # Failed datapoint IDs
│   ├── api_log.txt            # API call log
│   └── human_eval/            # Human evaluation samples
├── test/                      # Test & evaluation scripts
├── data/                      # Local data (images, text samples)
└── requirements.txt           # Python dependencies
```

---

## Setup & Usage

### Prerequisites

- Python 3.11+
- CUDA-capable GPUs (4x recommended for 235B model with TP=4)
- ~160 GB GPU memory total (FP8 quantized model)

### Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 1: Start the vLLM Server

```bash
chmod +x vllm_server_command.sh
./vllm_server_command.sh
```

This launches vLLM serving `Qwen3-VL-235B-A22B-Thinking-FP8` with:
- 4-way tensor parallelism across GPUs
- 250K token context window
- Up to 50 images per prompt
- DeepSeek-R1 reasoning parser
- 95% GPU memory utilization

### Step 2: Run the Pipeline

```bash
python main.py
```

Optional: process only specific datapoints:
```bash
python main.py --datapoint-list path/to/datapoint_ids.txt
```

### Step 3 (Optional): Fallback via OpenRouter

If the local vLLM server is unavailable, use the cloud fallback:

```bash
# Set OPENROUTER_API_KEY in .env
python fallback_openrouter.py
```

### Step 4 (Optional): Merge Split Outputs

```bash
python join_jsons.py --output-dir output
```

---

## Configuration

| Variable | Location | Description |
|----------|----------|-------------|
| `VLLM_API_URL` | Environment / `qwen_service.py` | vLLM endpoint (default: `http://localhost:8000/v1/chat/completions`) |
| `VLLM_MODEL_ID` | Environment / `qwen_service.py` | Model identifier |
| `HUGGINGFACE_TOKEN` | Environment | HF token for model access |
| `GPU_DEVICE` | Environment | CUDA device override |
| `MAX_TOKENS` | `config/config.py` | Max generation tokens (default: 4096) |
| `DATAPOINT_ROOTS` | `main.py` | Source data directories |

---

## Resilience Features

- **Incremental saving:** Results saved after each datapoint — safe to interrupt and resume.
- **Skip completed:** Already-processed datapoints are detected and skipped automatically.
- **Failure logging:** Failed datapoints logged to `failed.txt` with error details for retry.
- **Empty response detection:** Empty model responses are caught and logged as failures.

---

## Requirements

- vLLM (with multimodal/vision support)
- requests, tqdm, Pillow, python-dotenv
- huggingface-hub, transformers
- bitsandbytes, accelerate
- OpenAI Python SDK (for fallback script)