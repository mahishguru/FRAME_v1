# Qwen3 Embedding Metrics

Two complementary scripts for evaluating generated text against ground-truth answers using
[Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) embeddings.
Both share the same rigorous four-layer transform stack that eliminates the ~0.85 cosine
similarity floor present in same-domain engineering text.

---

## Why a four-layer transform?

Raw cosine similarity between Qwen3 embeddings of same-domain engineering text clusters
around **0.85 regardless of factual accuracy**.  This makes the raw scores useless for
ranking models: a hallucinated answer and a correct one look identical.

Four transforms are applied in order to every similarity matrix before scoring:

```
raw cosine sim
      ↓
1. hard_token_gate    — zero cells where GT numbers are absent from gen
      ↓
2. double_center      — subtract row mean + col mean − grand mean, clip ≥ 0
      ↓
3. power_sharpen      — raise to power α (default 8)
      ↓
transformed sim used for scoring
```

| Transform | What it removes | Effect |
|---|---|---|
| Token gate | Factually-wrong matches (wrong numbers) | Hard zeros, not soft penalty |
| Double-center | Per-GT-chunk difficulty bias (row) + verbosity/filler bias (col) | Collapses ~0.85 floor to ~0 |
| Power sharpen | Residual low-similarity noise | 0.85 → 0.27, 0.50 → 0.004, 1.0 → 1.0 |

After the transforms only genuinely relevant (i,j) cell pairs are non-zero.

**Cross-case null calibration** then subtracts the expected score against K=20 random other
generated outputs from the same model (same-domain topic inflation), giving a calibrated
signal centred on zero for an uninformative response.

```
calibrated = (actual − null_mean) / (1 − null_mean)
```

---

## Instruction prefixes (asymmetric retrieval)

Both scripts use Qwen3-Embedding's asymmetric retrieval mode, matching the model's training:

| Side | Prefix |
|---|---|
| Ground truth chunks (document / passage) | `"Represent this text for retrieval:"` |
| Generated chunks (query side) | `"Represent this query for semantic matching:"` |

---

## Script 1 — `compute_qwen_OT.py`

### What it does

Measures **GT coverage** using Unbalanced Optimal Transport (UOT).

Instead of asking "what is the best single gen-chunk match for each GT chunk?", OT solves a
full transport plan $T \in \mathbb{R}^{n_{gt} \times n_{gen}}$ that distributes GT mass
across all gen chunks simultaneously.  This is dispersion-robust: even if a GT chunk's
information is split across three gen chunks, the transport plan aggregates all three
contributions.

The GT marginal constraint is fixed (penalty $= 10^9 \approx \infty$), meaning every GT
chunk *must* be covered.  The gen marginal is relaxed (penalty controlled by `--reg_m`),
allowing verbose responses to be penalised rather than rewarded.

### Scoring formula

$$
\text{Coverage} = \frac{\sum_i \left( \sum_j T_{ij} \cdot \tilde{S}_{ij} \right) \cdot w_i^{gt}}{\sum_i w_i^{gt}}
$$

where $\tilde{S}$ is the transformed similarity matrix and $w_i^{gt}$ is the word-count
weight of GT chunk $i$.

`CoveredFrac` counts the fraction of GT chunks whose transport-weighted similarity exceeds
`--coverage_threshold`.

### Usage

```bash
python compute_qwen_OT.py \
  --inference_results  path/to/inference.json \
  --ground_truth       path/to/ground_truth.json \
  --output_key         output_1 \
  --results_json       path/to/results.json
```

### Parameters

#### Required

| Parameter | Description |
|---|---|
| `--inference_results` | Path to inference results JSON |
| `--ground_truth` | Path to ground-truth JSON |
| `--output_key` | `output_1` or `output_2` — which GT field to score against |
| `--results_json` | Path to results JSON (created/updated in-place) |

#### Chunking

| Parameter | Default | Description |
|---|---|---|
| `--gt_window_size` | `1` | Sentences per sliding-window chunk for GT |
| `--gt_window_step` | `1` | Step size for GT sliding window |
| `--gen_window_size` | `2` | Sentences per sliding-window chunk for generated text |
| `--gen_window_step` | `1` | Step size for gen sliding window (1 = overlapping) |

#### Model

| Parameter | Default | Description |
|---|---|---|
| `--model_name` | `Qwen/Qwen3-Embedding-8B` | HuggingFace model name |
| `--batch_size` | `8` | Embedding batch size |
| `--gt_instruction_prefix` | `Represent this text for retrieval:` | Prefix for GT chunks |
| `--gen_instruction_prefix` | `Represent this query for semantic matching:` | Prefix for gen chunks |

#### Transform hyper-parameters

| Parameter | Default | Description |
|---|---|---|
| `--alpha` | `8.0` | Power-sharpening exponent. Higher = more contrast. |
| `--min_gt_numbers` | `2` | Min distinct numbers in a GT chunk to activate the token gate |
| `--no_token_gate` | off | Flag to disable the hard factual token gate |
| `--no_double_center` | off | Flag to disable double-centering |
| `--no_power_sharpen` | off | Flag to disable power sharpening |

#### OT solver

| Parameter | Default | Description |
|---|---|---|
| `--reg` | `0.01` | Sinkhorn entropic regularisation $\varepsilon$ (smoother ↑, sparser ↓) |
| `--reg_m` | `0.1` | Unbalanced gen-marginal penalty (lower = more relaxed gen marginal) |
| `--coverage_threshold` | `0.5` | Minimum transport-weighted sim for a GT chunk to count as "covered" |

#### Null calibration

| Parameter | Default | Description |
|---|---|---|
| `--null_k` | `20` | Number of random other cases to use as null pool per case |
| `--null_seed` | `42` | RNG seed for null sampling |

#### Other

| Parameter | Default | Description |
|---|---|---|
| `--overwrite-existing` | off | Re-compute even if result already exists in `--results_json` |

### Output fields

| Field | Range | Description |
|---|---|---|
| `Qwen_OT_Coverage` | [0, 1] | Raw transport-weighted GT coverage |
| `Qwen_OT_CoveredFrac` | [0, 1] | Fraction of GT chunks with coverage above threshold |
| `Qwen_OT_MinChunkCov` | [0, 1] | Minimum coverage across all GT chunks (weakest-link) |
| `Qwen_OT_Null` | [0, 1] | Mean coverage against K random null gen outputs |
| **`Qwen_OT_CalibratedCov`** | [−1, 1] | **PRIMARY METRIC** — coverage above the same-domain null baseline |
| `Qwen_OT_EMD` | ≥ 0 | Earth Mover's Distance of the transport plan (lower = more concentrated matches) |

> **Use `Qwen_OT_CalibratedCov` as the ranking metric.**  
> Positive = better than random same-domain text. Negative = worse (hallucination pattern).

---

## Script 2 — `compute_qwen_chunk.py`

### What it does

Measures **soft recall, precision, and F1** using max-sim (and avg-top-k for recall)
aggregation per chunk.

For each GT chunk $i$, instead of a single `max_j` (which ignores cases where a GT chunk's
content is dispersed across multiple gen chunks), the recall aggregation takes the **mean of
the top-k transformed similarities** along the GT row:

$$
\text{SoftRecall} = \frac{\sum_i w_i^{gt} \cdot \overline{\text{top-}k}_j\,\tilde{S}_{ij}}{\sum_i w_i^{gt}}
$$

For precision, `max` over GT chunks per gen chunk is used (verbose gen chunks that don't
match anything specific are penalised by the transform stack).

### Usage

```bash
python compute_qwen_chunk.py \
  --inference_results  path/to/inference.json \
  --ground_truth       path/to/ground_truth.json \
  --output_key         output_1 \
  --results_json       path/to/results.json
```

### Parameters

#### Required

| Parameter | Description |
|---|---|
| `--inference_results` | Path to inference results JSON |
| `--ground_truth` | Path to ground-truth JSON |
| `--output_key` | `output_1` or `output_2` |
| `--results_json` | Path to results JSON (created/updated in-place) |

#### Chunking

| Parameter | Default | Description |
|---|---|---|
| `--gt_window_size` | `1` | Sentences per sliding-window chunk for GT |
| `--gt_window_step` | `1` | Step size for GT sliding window |
| `--gen_window_size` | `2` | Sentences per sliding-window chunk for generated text |
| `--gen_window_step` | `1` | Step size for gen sliding window (1 = overlapping) |

#### Model

| Parameter | Default | Description |
|---|---|---|
| `--model_name` | `Qwen/Qwen3-Embedding-8B` | HuggingFace model name |
| `--batch_size` | `8` | Embedding batch size |
| `--gt_instruction_prefix` | `Represent this text for retrieval:` | Prefix for GT chunks |
| `--gen_instruction_prefix` | `Represent this query for semantic matching:` | Prefix for gen chunks |

#### Transform hyper-parameters

| Parameter | Default | Description |
|---|---|---|
| `--alpha` | `8.0` | Power-sharpening exponent |
| `--top_k` | `3` | Number of top gen-chunk similarities to average per GT chunk for recall |
| `--no_token_gate` | off | Disable hard factual token gate |
| `--no_double_center` | off | Disable double-centering |
| `--no_power_sharpen` | off | Disable power sharpening |

#### Null calibration

| Parameter | Default | Description |
|---|---|---|
| `--null_k` | `20` | Null pool size K per case (0 = disable) |
| `--null_seed` | `42` | RNG seed for null sampling |

#### Other

| Parameter | Default | Description |
|---|---|---|
| `--overwrite-existing` | off | Re-compute even if result already exists |

### Output fields

| Field | Range | Description |
|---|---|---|
| `Qwen_SoftRecall` | [0, 1] | Length-weighted avg-top-k transformed recall (GT → gen) |
| `Qwen_SoftPrecision` | [0, 1] | Length-weighted max-sim transformed precision (gen → GT) |
| `Qwen_SoftF1` | [0, 1] | Harmonic mean of SoftRecall and SoftPrecision |
| `Qwen_SoftAvgSim` | [0, 1] | Unweighted mean of top-k recall scores + gen max scores |
| `Qwen_Null_Recall` | [0, 1] | Mean recall against K random null gen outputs |
| **`Qwen_CalibratedRecall`** | [−1, 1] | **PRIMARY METRIC** — recall above the same-domain null baseline |

> **Use `Qwen_CalibratedRecall` as the ranking metric.**  
> It answers: _"How much better does this model recall GT information than a random
> same-domain response?"_

---

## Comparison: when to use which

| Question | Use |
|---|---|
| Does the answer **cover all ground-truth facts**? | `Qwen_OT_CalibratedCov` |
| Does the answer **recall specific GT content** without over-crediting verbosity? | `Qwen_CalibratedRecall` |
| Is verbosity / padding **penalised** (precision-side)? | `Qwen_SoftPrecision` |
| What is the **worst-covered GT chunk** (robustness check)? | `Qwen_OT_MinChunkCov` |

The OT script is stricter: it enforces that **every GT chunk must be covered** (fixed GT
marginal), making it sensitive to omissions.  The chunk script is more symmetric (recall +
precision), making it a better proxy for overall answer quality.

For final model rankings, combining both primaries is recommended:

```
combined = 0.5 * Qwen_OT_CalibratedCov + 0.5 * Qwen_CalibratedRecall
```

---

## Chunking strategy

Both scripts use the same hybrid chunker (`chunk_text_sliding_window`):

- **Structured text** (markdown headers `#`, bullets `- *`): splits on structural
  boundaries, groups sub-bullets with their parent, then applies a sliding window.
- **Prose**: sentence-tokenises with NLTK `punkt`, then applies a sliding window.

A **sliding window with overlap** (step < window_size) is used so that facts spanning two
adjacent sentences are captured in at least one chunk.

---

## Example: full pipeline call (via `run_metrics_pipeline.py`)

```bash
python run_metrics_pipeline.py \
  --metrics qwen_OT qwen_chunk \
  --overwrite-metrics \
  --max-workers 1
```

Or run a single script directly:

```bash
python compute_qwen_OT.py \
  --inference_results results/claude/inference.json \
  --ground_truth      ../final_output.json \
  --output_key        output_1 \
  --results_json      results/claude/metrics.json \
  --alpha 8.0 \
  --null_k 20 \
  --overwrite-existing
```
