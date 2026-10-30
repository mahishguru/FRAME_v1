#!/usr/bin/env python3
"""
Compute Qwen3 chunk-level soft recall/precision/F1 — rigorous discriminative recipe.

Four stacked improvements over the original max-sim approach:

1. Hard factual token gate
   Zero out sim[i,j] when GT chunk i has ≥2 distinct numbers with zero
   overlap in gen chunk j. Kill same-topic but numerically-wrong matches.
   Uses whole-word boundaries (\\b) to avoid 200/2000 collisions.

2. Double-centering of the similarity matrix
   sim_dc[i,j] = sim[i,j] - row_mean[i] - col_mean[j] + grand_mean, clipped to 0.
   Removes the per-GT-chunk difficulty offset (row) and the generic-filler
   verbosity offset (column), collapsing the ~0.85 embedding floor to near zero.

3. Power sharpening (α = 4 default for bge-m3)
   sim_sharp = sim_dc ** α. Nonlinear contrast: 0.50 → 0.0625, 0.90 → 0.656.
   α=8 was used with Qwen3-Embedding (DC residuals ~0.002 max, needed strong
   amplification). bge-m3 has a lower floor (~0.65–0.75 raw) → DC residuals are
   already ~0.1–0.3, so α=4 preserves discriminative signal without over-killing.
   Unlike linear tau-shift, this changes the shape of the distribution.

4. avg-top-k recall (k = 3 default) + cross-case null calibration
   Instead of max_j sim(i,j) per GT chunk (vulnerable to dispersion), take the
   mean of the top-k sim values along each GT row.  After transforms, only
   genuinely good matches are non-zero, so top-k captures dispersed evidence
   across multiple gen chunks.

   Cross-case null: sample K=20 random other gen outputs from the same model.
   Compute soft recall of GT_c against those mismatched gen texts.
   calibrated = (actual − null) / (1 − null)  ← primary output.

Output fields
-------------
  Chunk_SoftRecall        : transformed avg-top-k recall (replaces old max-sim value)
  Chunk_SoftPrecision     : transformed max-sim precision
  Chunk_SoftF1            : harmonic mean of above
  Chunk_NullRecall        : cross-case null baseline for recall
  Chunk_CalibratedRecall  : recall above null, in [-1, 1]  ← PRIMARY METRIC
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import nltk

# Compile regex once — \b ensures 50 and 5000 are distinct tokens
NUMBER_PATTERN = re.compile(r'\b\d+\.?\d*\b')


def normalize_inference_data(data: dict) -> dict:
    if 'inference_results' in data and isinstance(data['inference_results'], list):
        normalized = {}
        for item in data['inference_results']:
            key = item.get('input_case')
            if key:
                normalized[key] = item
        return normalized
    return data


NLTK_DATA_DIR = Path(os.environ.get("NLTK_DATA", Path.home() / ".cache" / "nltk_data"))
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
nltk.data.path = [str(NLTK_DATA_DIR)]

_NLTK_SENT_OK = False


def ensure_sentence_tokenizer():
    global _NLTK_SENT_OK
    if _NLTK_SENT_OK:
        return
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab/english')
    except LookupError:
        nltk.download('punkt', download_dir=str(NLTK_DATA_DIR), quiet=True)
        nltk.download('punkt_tab', download_dir=str(NLTK_DATA_DIR), quiet=True)
    _NLTK_SENT_OK = True


def chunk_text_sliding_window(text: str, window_size: int = 2, step: int = 1) -> list:
    """
    Chunk text using a hybrid approach that handles both prose and structured text.

    For structured text (markdown, bullets):
    - Splits on headers (# lines)
    - Splits on bullet points (- or *)
    - Groups sub-bullets with their parent

    For prose:
    - Falls back to sentence-based chunking

    Then applies sliding window to preserve context.
    """
    has_headers = bool(re.search(r'^\s*#+\s+', text, re.MULTILINE))
    has_bullets = bool(re.search(r'^\s*[-*]\s+', text, re.MULTILINE))

    if has_headers or has_bullets:
        units = []
        lines = text.split('\n')
        current_unit = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_unit:
                    units.append('\n'.join(current_unit))
                    current_unit = []
                continue
            if re.match(r'^\s*#+\s+', line):
                if current_unit:
                    units.append('\n'.join(current_unit))
                current_unit = [line]
            elif re.match(r'^[-*]\s+', stripped):
                if current_unit:
                    units.append('\n'.join(current_unit))
                current_unit = [line]
            else:
                current_unit.append(line)

        if current_unit:
            units.append('\n'.join(current_unit))

        units = [u.strip() for u in units if u.strip()]

        if not units:
            return []

        if len(units) <= window_size:
            return ["\n\n".join(units)]

        chunks = []
        for i in range(0, len(units) - window_size + 1, step):
            window = units[i: i + window_size]
            chunks.append("\n\n".join(window))
        return chunks

    else:
        ensure_sentence_tokenizer()
        try:
            sentences = nltk.tokenize.sent_tokenize(text)
        except Exception:
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        sentences = [s.strip() for s in sentences if len(s.strip()) >= 5]

        if not sentences:
            return [text.strip()] if text.strip() else []

        if len(sentences) <= window_size:
            return [" ".join(sentences)]

        chunks = []
        for i in range(0, len(sentences) - window_size + 1, step):
            window = sentences[i: i + window_size]
            chunks.append(" ".join(window))
        return chunks


# ---------------------------------------------------------------------------
# Similarity matrix transforms (same stack as compute_chunk_OT.py)
# ---------------------------------------------------------------------------

def hard_token_gate(sim: np.ndarray, gt_chunks: list, gen_chunks: list,
                    min_gt_numbers: int = 2) -> np.ndarray:
    """Zero out sim[i,j] when GT chunk i has ≥min_gt_numbers distinct numbers
    with zero overlap in gen chunk j.  Kills same-topic but factually-wrong
    matches.  Word-boundary regex avoids 200/2000 substring collisions."""
    sim = sim.copy()
    for i, gt_c in enumerate(gt_chunks):
        gt_nums = set(NUMBER_PATTERN.findall(gt_c))
        if len(gt_nums) < min_gt_numbers:
            continue
        for j, gen_c in enumerate(gen_chunks):
            gen_nums = set(NUMBER_PATTERN.findall(gen_c))
            if len(gt_nums & gen_nums) == 0:
                sim[i, j] = 0.0
    return sim


def double_center(sim: np.ndarray) -> np.ndarray:
    """Remove per-GT-chunk difficulty bias (row mean) and per-gen-chunk
    verbosity/genericity bias (col mean) relative to the grand mean.
    Collapses the ~0.85 embedding floor to near zero.  Values below
    zero clipped to 0 (no negative anti-similarity)."""
    row_mean = sim.mean(axis=1, keepdims=True)
    col_mean = sim.mean(axis=0, keepdims=True)
    grand_mean = sim.mean()
    centered = sim - row_mean - col_mean + grand_mean
    return np.clip(centered, 0.0, None)


def power_sharpen(sim: np.ndarray, alpha: float = 4.0) -> np.ndarray:
    """Nonlinear contrast amplifier: sim ** alpha.
    For bge-m3 (α=4): 0.50 → 0.0625, 0.80 → 0.410, 0.95 → 0.815, 1.00 → 1.00.
    α=4 is the default for bge-m3 whose DC residuals are already ~0.1–0.3.
    (α=8 was used for Qwen3-Embedding where DC residuals were ~0.002.)
    Unlike linear tau-shift, this actually changes the shape and raises SNR."""
    return np.power(sim, alpha)


def load_bge_model(model_name="BAAI/bge-m3"):
    """Load BGE-M3 via SentenceTransformers.  Returns (model, device)."""
    print(f"Loading BGE embedding model: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    print(f"Model loaded on device: {device}")
    return model, device


def embed_chunks(chunks, model, device, instruction_prefix="", batch_size=8):
    """Embed a list of text chunks using SentenceTransformer (bge-m3).

    instruction_prefix: prepended to each chunk when non-empty.  BGE-M3 does
    not require asymmetric prefixes, so both GT and gen sides use empty string
    by default.  A custom prefix can still be supplied if desired.
    """
    if not chunks:
        return torch.empty(0, 0, device=device)

    texts = [
        f"{instruction_prefix} {chunk}".strip() if instruction_prefix else chunk
        for chunk in chunks
    ]

    # SentenceTransformer.encode returns a numpy (N, D) array
    np_embs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return torch.tensor(np_embs, dtype=torch.float32, device=device)


def precompute_gt_embeddings(ground_truth_data, output_key, model, device,
                             instruction_prefix, window_size, window_step,
                             inference_data=None, batch_size=8):
    print(f"\nPre-computing ground truth embeddings for {output_key}...")
    gt_cache = {}

    if inference_data is not None:
        cases_to_process = set(ground_truth_data.keys()) & set(inference_data.keys())
        print(f"Filtering: {len(cases_to_process)} cases in inference data "
              f"(out of {len(ground_truth_data)} total GT entries)")
    else:
        cases_to_process = ground_truth_data.keys()

    for input_case in tqdm(cases_to_process, desc="Caching GT embeddings"):
        gt_data = ground_truth_data[input_case]
        ground_truth = gt_data.get(output_key, "")
        if not ground_truth:
            continue

        gt_chunks = chunk_text_sliding_window(ground_truth, window_size=window_size, step=window_step)
        if gt_chunks:
            gt_embeddings = embed_chunks(gt_chunks, model, device,
                                         instruction_prefix, batch_size)
            gt_cache[input_case] = {
                'chunks': gt_chunks,
                'embeddings': gt_embeddings
            }

    print(f"Cached embeddings for {len(gt_cache)} ground truth cases")
    return gt_cache


def precompute_gen_embeddings(inference_data: dict, model, device,
                               instruction_prefix: str,
                               window_size: int, window_step: int,
                               valid_cases: set = None, batch_size: int = 8) -> dict:
    """Cache all gen chunk embeddings upfront.  Required for cross-case null pool."""
    print("\nPre-computing gen embeddings for null calibration pool...")
    gen_cache = {}
    cases = [c for c in inference_data if (valid_cases is None or c in valid_cases)]
    for input_case in tqdm(cases, desc="Caching gen embeddings"):
        generated_output = inference_data[input_case].get("generated_output", "")
        if not generated_output:
            continue
        gen_chunks = chunk_text_sliding_window(generated_output, window_size=window_size,
                                               step=window_step)
        if gen_chunks:
            gen_embs = embed_chunks(gen_chunks, model, device,
                                    instruction_prefix, batch_size)
            gen_cache[input_case] = {'chunks': gen_chunks, 'embeddings': gen_embs}
    print(f"Cached gen embeddings for {len(gen_cache)} cases")
    return gen_cache


def compute_metrics(gt_chunks, gt_embs, gen_chunks, gen_embs, null_gen_pool,
                    gt_lens, gen_lens, alpha: float = 4.0, top_k: int = 3,
                    use_token_gate: bool = True, use_double_center: bool = True,
                    use_power_sharpen: bool = True):
    """
    Rigorous soft recall/precision with 4-layer transform stack + null calibration.

    Transform pipeline applied to raw cosine sim matrix:
        hard_token_gate  →  double_center  →  power_sharpen

    Recall aggregation: avg-top-k (default k=3) per GT chunk rather than max,
    so dispersed facts spread across multiple gen chunks still get credit.
    Precision direction uses max (verbosity penalised via the transform stack).

    Cross-case null calibration: null_gen_pool is a list of
    {'chunks': [...], 'embeddings': tensor} from K random other cases
    (same model, different input).  Recall against those is the baseline.

    Returns
    -------
    (soft_recall, soft_precision, soft_f1, avg_sim, null_recall, calibrated_recall)
    """
    if gt_embs is None or gen_embs is None or gt_embs.numel() == 0 or gen_embs.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if gt_lens.sum() == 0 or gen_lens.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def _score_recall(g_embs, c_embs, g_chunks, c_chunks, g_lens, scale_factor=None):
        """Compute transformed avg-top-k recall score.

        scale_factor: if provided, normalize DC output by this value instead of
        the matrix's own max.  Pass the actual matrix's scale_factor to all null
        calls so they share the same reference scale (same fix as compute_chunk_OT).
        """
        sim = torch.matmul(g_embs, c_embs.transpose(0, 1)).detach().cpu().numpy()
        if sim.size == 0:
            return 0.0, 0.0, np.zeros(len(g_chunks), dtype=np.float32), 1.0
        if use_token_gate:
            sim = hard_token_gate(sim, g_chunks, c_chunks)
        if use_double_center:
            sim = double_center(sim)
            if scale_factor is None:
                scale_factor = float(sim.max())
            if scale_factor > 1e-9:
                sim = np.clip(sim / scale_factor, 0.0, 1.0)
        else:
            if scale_factor is None:
                scale_factor = 1.0
        if use_power_sharpen:
            sim = power_sharpen(sim, alpha)
        k = min(top_k, sim.shape[1])
        gt_topk = np.sort(sim, axis=1)[:, -k:].mean(axis=1)  # [n_gt]
        recall = float(np.dot(gt_topk, g_lens) / g_lens.sum())
        return recall, sim, gt_topk, scale_factor

    def _score_precision(g_embs, c_embs, g_chunks, c_chunks, g_lens, c_lens, sim_or_none=None):
        """Compute transformed max-sim precision score (optionally reuse sim)."""
        if sim_or_none is None:
            sim = torch.matmul(g_embs, c_embs.transpose(0, 1)).detach().cpu().numpy()
            if sim.size == 0:
                return 0.0
            if use_token_gate:
                sim = hard_token_gate(sim, g_chunks, c_chunks)
            if use_double_center:
                sim = double_center(sim)
            if use_power_sharpen:
                sim = power_sharpen(sim, alpha)
        else:
            sim = sim_or_none
        gen_scores = np.max(sim, axis=0)  # [n_gen]
        precision = float(np.dot(gen_scores, c_lens) / c_lens.sum())
        return precision

    # --- Actual scores ---
    recall, sim_transformed, gt_topk, scale_factor = _score_recall(
        gt_embs, gen_embs, gt_chunks, gen_chunks, gt_lens)

    gen_lens_np = gen_lens if isinstance(gen_lens, np.ndarray) else np.array(gen_lens, dtype=np.float32)

    precision = _score_precision(
        gt_embs, gen_embs, gt_chunks, gen_chunks, gt_lens, gen_lens_np,
        sim_or_none=sim_transformed)

    f1 = 0.0
    if (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))

    gen_scores_diag = np.max(sim_transformed, axis=0) if sim_transformed.size > 0 else np.zeros(1)
    avg_sim = float(np.concatenate([gt_topk, gen_scores_diag]).mean())

    # --- Cross-case null calibration ---
    null_recall = 0.0
    calibrated_recall = recall
    if null_gen_pool:
        null_scores = []
        for null_entry in null_gen_pool:
            null_embs = null_entry['embeddings']
            null_chunks = null_entry['chunks']
            null_lens = np.array([len(c.split()) for c in null_chunks], dtype=np.float32)
            if null_embs.numel() == 0 or null_lens.sum() == 0:
                continue
            nr, _, _, _ = _score_recall(gt_embs, null_embs, gt_chunks, null_chunks, gt_lens,
                                         scale_factor=scale_factor)
            null_scores.append(nr)
        if null_scores:
            null_recall = float(np.mean(null_scores))
            denom = 1.0 - null_recall
            raw_cal = float((recall - null_recall) / denom) if abs(denom) > 1e-9 else 0.0
            # Clip to [-1, 1]: values outside this range are numerical artefacts
            # caused by repetitive models whose null pool accidentally scores very
            # high (null_recall → 1), making the denominator → 0.
            calibrated_recall = float(np.clip(raw_cal, -1.0, 1.0))

    return recall, precision, f1, avg_sim, null_recall, calibrated_recall



def atomic_write(data, path: Path):
    """Write data to JSON file atomically, converting numpy types to Python types."""
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    data = convert_to_native(data)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp_path, 'w', encoding='utf-8') as tmp_file:
        json.dump(data, tmp_file, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Qwen3 chunk-level soft recall/precision — rigorous 4-layer recipe'
    )
    parser.add_argument('--inference_results', type=str, required=True,
                        help='Path to inference results JSON file')
    parser.add_argument('--ground_truth', type=str,
                        default='/data/mguru/04_Finetuning/frame-finetuning-evaluation/final_output.json',
                        help='Path to ground truth JSON file')
    parser.add_argument('--output_key', type=str, required=True,
                        choices=['output_1', 'output_2'],
                        help='Which output to compare against (output_1 or output_2)')
    parser.add_argument('--results_json', type=str, required=True,
                        help='Path to results JSON file (will be created/updated)')
    parser.add_argument('--model_name', type=str,
                        default='BAAI/bge-m3',
                        help='SentenceTransformer embedding model name (default: BAAI/bge-m3)')
    parser.add_argument('--gt_window_size', type=int, default=1,
                        help='Sentences per sliding window for ground truth (default 1)')
    parser.add_argument('--gt_window_step', type=int, default=1,
                        help='Step size for GT sliding window (default 1)')
    parser.add_argument('--gen_window_size', type=int, default=2,
                        help='Sentences per sliding window for generated text (default 2)')
    parser.add_argument('--gen_window_step', type=int, default=1,
                        help='Step size for gen sliding window (default 1, overlapping)')
    parser.add_argument('--gt_instruction_prefix', type=str,
                        default='',
                        help='Instruction prefix for GT chunks (default: empty for bge-m3)')
    parser.add_argument('--gen_instruction_prefix', type=str,
                        default='',
                        help='Instruction prefix for generated chunks (default: empty for bge-m3)')
    # Transform hyper-parameters
    parser.add_argument('--alpha', type=float, default=4.0,
                        help='Power-sharpening exponent (default 4.0 for bge-m3; use 8.0 for Qwen3-Embedding)')
    parser.add_argument('--top_k', type=int, default=3,
                        help='avg-top-k recall: number of gen chunks to average per GT chunk (default 3)')
    parser.add_argument('--null_k', type=int, default=20,
                        help='Cross-case null pool size K (default 20, 0 = disable)')
    parser.add_argument('--null_seed', type=int, default=42,
                        help='RNG seed for null pool sampling (default 42)')
    # Toggle individual transforms
    parser.add_argument('--no_token_gate', action='store_true',
                        help='Disable hard factual token gate')
    parser.add_argument('--no_double_center', action='store_true',
                        help='Disable double-centering of sim matrix')
    parser.add_argument('--no_power_sharpen', action='store_true',
                        help='Disable power sharpening')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for embedding computation (default 8)')
    parser.add_argument('--overwrite-existing', action='store_true',
                        help='Recompute metrics even if values already exist')
    return parser.parse_args()


def main():
    args = parse_args()

    use_token_gate    = not args.no_token_gate
    use_double_center = not args.no_double_center
    use_power_sharpen = not args.no_power_sharpen

    print("=" * 80)
    print("BGE-M3 Chunk Metrics — Rigorous Recipe (token gate + double-center + power-sharpen + null)")
    print("=" * 80)
    print(f"Inference results:      {args.inference_results}")
    print(f"Ground truth:           {args.ground_truth}")
    print(f"Output key:             {args.output_key}")
    print(f"Results JSON:           {args.results_json}")
    print(f"Model:                  {args.model_name}")
    print(f"GT  Window:             size={args.gt_window_size}, step={args.gt_window_step}")
    print(f"Gen Window:             size={args.gen_window_size}, step={args.gen_window_step}")
    print(f"GT  instruction prefix: '{args.gt_instruction_prefix}'")
    print(f"Gen instruction prefix: '{args.gen_instruction_prefix}'")
    print(f"Alpha (power sharpen):  {args.alpha}")
    print(f"Top-k (recall avg):     {args.top_k}")
    print(f"Null pool K:            {args.null_k}")
    print(f"Token gate:             {'ON' if use_token_gate else 'OFF'}")
    print(f"Double-center:          {'ON' if use_double_center else 'OFF'}")
    print(f"Power sharpen:          {'ON' if use_power_sharpen else 'OFF'}")
    print(f"Batch size:             {args.batch_size}")
    print(f"Overwrite existing:     {args.overwrite_existing}")
    print("=" * 80)

    with open(args.inference_results, 'r', encoding='utf-8') as f:
        raw_inference = json.load(f)
    inference_data = normalize_inference_data(raw_inference)
    print(f"Loaded {len(inference_data)} inference results")

    with open(args.ground_truth, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    print(f"Loaded {len(ground_truth_data)} ground truth entries")

    results_path = Path(args.results_json)
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing result entries")
    else:
        results = {}
        results_path.parent.mkdir(parents=True, exist_ok=True)
        print("Creating new results file...")

    model, device = load_bge_model(args.model_name)

    # --- Pre-compute GT embeddings (document / retrieval side) ---
    gt_embedding_cache = precompute_gt_embeddings(
        ground_truth_data,
        args.output_key,
        model,
        device,
        args.gt_instruction_prefix,
        args.gt_window_size,
        args.gt_window_step,
        inference_data,
        args.batch_size
    )

    # --- Pre-compute gen embeddings for null calibration pool ---
    gen_embedding_cache = precompute_gen_embeddings(
        inference_data,
        model,
        device,
        args.gen_instruction_prefix,
        args.gen_window_size,
        args.gen_window_step,
        valid_cases=set(gt_embedding_cache.keys()),
        batch_size=args.batch_size
    )

    # All valid cases for null sampling
    all_valid_cases = sorted(gen_embedding_cache.keys())
    rng = np.random.default_rng(args.null_seed)

    processed = 0
    skipped = 0

    print(f"\nComputing chunk metrics for {args.output_key}...")
    for input_case, inference_value in tqdm(inference_data.items(), desc="Processing"):
        generated_output = inference_value.get("generated_output")
        if not generated_output:
            tqdm.write(f"Skipping {input_case}: generated_output is null/empty")
            skipped += 1
            continue
        if input_case not in ground_truth_data:
            tqdm.write(f"Warning: {input_case} not found in ground truth. Skipping.")
            skipped += 1
            continue
        if input_case not in gt_embedding_cache:
            tqdm.write(f"Warning: {input_case} has no cached GT embeddings. Skipping.")
            skipped += 1
            continue
        if input_case not in gen_embedding_cache:
            tqdm.write(f"Warning: {input_case} has no gen embeddings. Skipping.")
            skipped += 1
            continue

        existing_entry = results.get(input_case, {})
        if not args.overwrite_existing and "Chunk_CalibratedRecall" in existing_entry:
            processed += 1
            continue

        gt_chunks = gt_embedding_cache[input_case]['chunks']
        gt_embs   = gt_embedding_cache[input_case]['embeddings']
        gen_chunks = gen_embedding_cache[input_case]['chunks']
        gen_embs   = gen_embedding_cache[input_case]['embeddings']

        gt_lens  = np.array([len(c.split()) for c in gt_chunks],  dtype=np.float32)
        gen_lens = np.array([len(c.split()) for c in gen_chunks], dtype=np.float32)

        # --- Build null pool (K random other cases) ---
        null_gen_pool = []
        if args.null_k > 0:
            pool_candidates = [c for c in all_valid_cases if c != input_case]
            k_actual = min(args.null_k, len(pool_candidates))
            chosen = rng.choice(pool_candidates, size=k_actual, replace=False)
            null_gen_pool = [gen_embedding_cache[c] for c in chosen]

        try:
            recall, precision, f1, avg_sim, null_recall, calibrated_recall = compute_metrics(
                gt_chunks, gt_embs, gen_chunks, gen_embs, null_gen_pool,
                gt_lens, gen_lens,
                alpha=args.alpha,
                top_k=args.top_k,
                use_token_gate=use_token_gate,
                use_double_center=use_double_center,
                use_power_sharpen=use_power_sharpen
            )
        except Exception as exc:
            tqdm.write(f"Error computing metrics for {input_case}: {exc}")
            skipped += 1
            continue

        if input_case not in results:
            results[input_case] = {
                "input_case": input_case,
                "output_key": args.output_key
            }
        results[input_case]["Chunk_SoftRecall"]        = round(recall,             4)
        results[input_case]["Chunk_SoftPrecision"]     = round(precision,          4)
        results[input_case]["Chunk_SoftF1"]            = round(f1,                 4)
        results[input_case]["Chunk_AvgSim"]            = round(avg_sim,            4)
        results[input_case]["Chunk_NullRecall"]        = round(null_recall,        4)
        results[input_case]["Chunk_CalibratedRecall"]  = round(calibrated_recall,  4)
        processed += 1
        atomic_write(results, results_path)

    print(f"\nSaving final results to {results_path}...")
    atomic_write(results, results_path)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully processed: {processed}")
    print(f"Skipped:               {skipped}")
    print(f"Results saved to:      {results_path}")
    print("=" * 80)
    print("✓ Done!")


if __name__ == '__main__':
    main()
