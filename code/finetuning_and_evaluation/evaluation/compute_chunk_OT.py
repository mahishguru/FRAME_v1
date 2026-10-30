#!/usr/bin/env python3
"""
Qwen Unbalanced-OT Coverage — Rigorous Discriminative Recipe
=============================================================

Four stacked improvements over v1 to break the embedding-similarity floor:

1. Hard factual token gate
   Zero out sim[i,j] whenever the GT chunk contains ≥2 distinct numbers that
   have zero overlap with the gen chunk.  Prevents topically-similar but
   numerically-wrong text from receiving any coverage credit.

2. Double-centering of the similarity matrix
   sim_dc[i,j] = sim[i,j] - row_mean[i] - col_mean[j] + grand_mean
   Clip to zero.  This removes:
     - Row (GT chunk) difficulty offset: GT chunks that are simply harder to match
     - Col (gen chunk) generic-verbosity offset: fluffy gen chunks that are
       uniformly similar to everything get their column mean subtracted,
       collapsing their contribution to near-zero.
   This is the single most important fix for the similarity floor.

3. Power sharpening  (default α = 8)
   sim_sharp[i,j] = sim_dc[i,j] ** α
   A nonlinear amplifier: 0.20→0.0002, 0.50→0.004, 0.90→0.43, 1.00→1.00.
   Unlike a linear tau-shift (same SNR after rescaling), the power function
   changes the *shape* of the distribution — good matches pull far ahead of
   mediocre ones.

4. Cross-case null calibration  (default K = 20 samples)
   For each case c, sample K random OTHER cases c'_1…c'_K from the same
   inference file.  Compute the full OT coverage score using GT_c but gen_{c'_k}
   (wrong match — same-domain text, but for a different question).

   null(c)  =  mean_k  OT_v2(GT_c, gen_{c'_k})

   calibrated_cov(c) = (actual(c) - null(c)) / (1 − null(c))

   This gives the amount of coverage *above what any domain-relevant generation
   scores by chance*, regardless of the raw similarity floor.

   Because gen embeddings for all cases are precomputed, the null only costs
   K extra UOT solves per case — no extra model inference.

Outputs (in addition to all v1 metrics)
-----------------------------------------
  Chunk_OT_Coverage          : UOT coverage with transforms (no null subtraction)
  Chunk_OT_CoveredFrac       : fraction of GT chunks above threshold (transformed)
  Chunk_OT_MinChunkCov       : worst GT chunk coverage (transformed)
  Chunk_OT_Null              : mean cross-case null score (case difficulty proxy)
  Chunk_OT_CalibratedCov     : coverage above null, normalised to [0,1]  ← PRIMARY METRIC
  Chunk_OT_EMD               : transport cost (transformed cost matrix)
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import nltk
import ot

# ==============================
# Constants / Utilities
# ==============================

NUMBER_PATTERN = re.compile(r'\b\d+\.?\d*\b')

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
    Hybrid chunker: structured (markdown/bullets) → semantic units with sliding
    window; prose → sentence-based sliding window.
    """
    has_headers = bool(re.search(r'^\s*#+\s+', text, re.MULTILINE))
    has_bullets  = bool(re.search(r'^\s*[-*]\s+',  text, re.MULTILINE))

    if has_headers or has_bullets:
        units = []
        lines = text.split('\n')
        current_unit: list[str] = []
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
            return ['\n\n'.join(units)]
        return ['\n\n'.join(units[i: i + window_size])
                for i in range(0, len(units) - window_size + 1, step)]
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
            return [' '.join(sentences)]
        return [' '.join(sentences[i: i + window_size])
                for i in range(0, len(sentences) - window_size + 1, step)]


# ==============================
# Model Loading
# ==============================

def load_bge_model(model_name='BAAI/bge-m3'):
    print(f'Loading BGE embedding model: {model_name}...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer(model_name, device=str(device))
    print(f'Model loaded on device: {device}')
    return model, device


# ==============================
# Embedding
# ==============================

def embed_chunks(chunks, model, device, batch_size=8):
    if not chunks:
        return torch.empty(0, 0, device=device)
    np_embs = model.encode(
        chunks,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return torch.tensor(np_embs, dtype=torch.float32, device=device)


# ==============================
# GT Embedding Cache
# ==============================

def precompute_gt_embeddings(ground_truth_data, output_key, model, device,
                              window_size, window_step,
                              inference_data=None, batch_size=8):
    print(f'\nPre-computing GT embeddings for {output_key}...')
    gt_cache: dict = {}
    cases = (set(ground_truth_data.keys()) & set(inference_data.keys())
             if inference_data is not None else ground_truth_data.keys())
    if inference_data is not None:
        print(f'  Filtering to {len(cases)} cases present in inference data')
    for case_id in tqdm(cases, desc='GT embeddings'):
        gt_text = ground_truth_data[case_id].get(output_key, '')
        if not gt_text:
            continue
        chunks = chunk_text_sliding_window(gt_text, window_size, window_step)
        if chunks:
            embs = embed_chunks(chunks, model, device, batch_size)
            gt_cache[case_id] = {'chunks': chunks, 'embeddings': embs}
    print(f'  Cached {len(gt_cache)} GT cases')
    return gt_cache


# ==============================
# GEN Embedding Cache
# ==============================

def precompute_gen_embeddings(inference_data, model, device,
                               window_size, window_step,
                               batch_size=8):
    """
    Precompute gen embeddings for ALL cases in the inference file.
    Used both for the actual score and for building the cross-case null pool.
    """
    print('\nPre-computing gen embeddings for ALL cases (needed for null pool)...')
    gen_cache: dict = {}
    for case_id, inf_val in tqdm(inference_data.items(), desc='Gen embeddings'):
        text = inf_val.get('generated_output', '')
        if not text:
            continue
        chunks = chunk_text_sliding_window(text, window_size, window_step)
        if chunks:
            embs = embed_chunks(chunks, model, device, batch_size)
            gen_cache[case_id] = {'chunks': chunks, 'embeddings': embs}
    print(f'  Cached {len(gen_cache)} gen cases')
    return gen_cache


# ==============================
# Similarity Matrix Transforms
# ==============================

def hard_token_gate(sim_matrix: np.ndarray,
                    gt_chunks: list[str],
                    gen_chunks: list[str],
                    min_gt_numbers: int = 2) -> np.ndarray:
    """
    Zero out sim[i,j] when:
      - GT chunk i has ≥ min_gt_numbers distinct numbers, AND
      - none of those numbers appear anywhere in gen chunk j.

    Uses min_gt_numbers=2 to avoid penalising GT chunks where the only
    "number" is a year or footnote reference (which typically appears
    in every engineering text anyway).
    """
    sim = sim_matrix.copy()
    for i, gt_c in enumerate(gt_chunks):
        gt_nums = set(NUMBER_PATTERN.findall(gt_c))
        if len(gt_nums) < min_gt_numbers:
            continue  # not enough numeric evidence to gate on
        for j, gen_c in enumerate(gen_chunks):
            gen_nums = set(NUMBER_PATTERN.findall(gen_c))
            if not gt_nums & gen_nums:
                sim[i, j] = 0.0
    return sim


def double_center(sim_matrix: np.ndarray) -> np.ndarray:
    """
    Remove row means (GT-chunk difficulty) and column means (gen-chunk genericity).

      sim_dc[i,j] = sim[i,j] - mean(row_i) - mean(col_j) + grand_mean

    Clip negatives to 0.  Normalization (to make power_sharpen work with the
    narrow Qwen similarity range) is handled externally in transform_sim_matrix
    so that actual and null matrices share the same scale factor.
    """
    row_means   = sim_matrix.mean(axis=1, keepdims=True)
    col_means   = sim_matrix.mean(axis=0, keepdims=True)
    grand_mean  = sim_matrix.mean()
    dc = sim_matrix - row_means - col_means + grand_mean
    return np.maximum(dc, 0.0)


def power_sharpen(sim_matrix: np.ndarray, alpha: float = 8.0) -> np.ndarray:
    """
    Nonlinear amplifier: sim → sim^alpha.
    Unlike a linear shift, this changes the *shape* of the distribution —
    good matches (near 1.0) stay large, mediocre matches (0.3–0.6) collapse.
    """
    return np.power(sim_matrix, alpha)


def transform_sim_matrix(sim_matrix: np.ndarray,
                          gt_chunks: list[str],
                          gen_chunks: list[str],
                          alpha: float = 8.0,
                          use_token_gate: bool = True,
                          use_double_center: bool = True,
                          use_power_sharpen: bool = True,
                          min_gt_numbers: int = 2,
                          scale_factor: float = None) -> tuple[np.ndarray, float]:
    """
    Apply all enabled transforms in the canonical order.

    Returns (transformed_matrix, scale_factor_used) so callers can pass the
    same scale_factor to null matrices, keeping actual and null on the same
    absolute scale (critical for null calibration to be meaningful).

    scale_factor: if provided, normalize DC output by this value instead of
                  the matrix's own max — use the actual matrix's scale_factor
                  for all null matrices.
    """
    sim = np.clip(sim_matrix, 0.0, 1.0)
    if use_token_gate:
        sim = hard_token_gate(sim, gt_chunks, gen_chunks, min_gt_numbers)
    if use_double_center:
        sim = double_center(sim)
        # Normalize to [0,1] using a consistent scale so power_sharpen works
        # with Qwen's narrow similarity range (raw DC residuals ≈ [0, 0.05]).
        # If scale_factor is given (null pass), use it; otherwise compute from
        # this matrix (actual pass) and return it for reuse.
        if scale_factor is None:
            scale_factor = float(sim.max())
        if scale_factor > 1e-9:
            sim = np.clip(sim / scale_factor, 0.0, 1.0)
    else:
        if scale_factor is None:
            scale_factor = 1.0
    if use_power_sharpen:
        sim = power_sharpen(sim, alpha)
    return sim, scale_factor


# ==============================
# UOT Core
# ==============================

def compute_ot_coverage(sim_matrix: np.ndarray,
                        gt_lens: np.ndarray,
                        gen_lens: np.ndarray,
                        reg: float = 0.01,
                        reg_m: float = 0.1,
                        coverage_threshold: float = 0.5):
    """Unbalanced Sinkhorn OT on a (possibly transformed) sim matrix."""
    n_gt, n_gen = sim_matrix.shape
    if n_gt == 0 or n_gen == 0 or sim_matrix.size == 0:
        empty = np.array([])
        return 0.0, 0.0, 0.0, 0.0, 0.0, empty

    cost = 1.0 - np.clip(sim_matrix, 0.0, 1.0)
    w_gt  = gt_lens  / gt_lens.sum()
    w_gen = gen_lens / gen_lens.sum()

    T = ot.unbalanced.sinkhorn_unbalanced(
        w_gt, w_gen, cost,
        reg=reg,
        reg_m=(1e9, reg_m),
        numItermax=500,
        stopThr=1e-6,
    )

    transport_cost = float(np.sum(T * cost))

    per_chunk = np.zeros(n_gt)
    for i in range(n_gt):
        if w_gt[i] > 1e-12:
            per_chunk[i] = np.sum(T[i, :] * sim_matrix[i, :]) / w_gt[i]
    per_chunk = np.clip(per_chunk, 0.0, 1.0)

    coverage      = float(np.dot(w_gt, per_chunk))
    covered_frac  = float(np.mean(per_chunk >= coverage_threshold))
    min_chunk_cov = float(np.min(per_chunk))

    return coverage, covered_frac, min_chunk_cov, float(np.mean(per_chunk)), transport_cost, per_chunk


# ==============================
# Full Per-Case Metric (actual + null)
# ==============================

def compute_metrics(gt_chunks, gt_embeddings,
                       gen_chunks, gen_embeddings,
                       null_gen_pool,           # list of (chunks, embeddings) tuples
                       gt_lens, gen_lens,
                       alpha=8.0,
                       use_token_gate=True,
                       use_double_center=True,
                       use_power_sharpen=True,
                       min_gt_numbers=2,
                       reg=0.01, reg_m=0.1,
                       coverage_threshold=0.5):
    """
    Args:
        null_gen_pool : list[(chunks, embeddings)] from K randomly sampled
                        other cases (same model, different input).  Used to
                        compute the cross-case null baseline.
    Returns:
        (coverage, covered_frac, min_cov, null_mean, calibrated_cov, emd)
    """
    # ---------- actual score ----------
    sim_actual = torch.matmul(gt_embeddings,
                              gen_embeddings.t()).detach().cpu().numpy()
    sim_t, scale_factor = transform_sim_matrix(
        sim_actual, gt_chunks, gen_chunks,
        alpha, use_token_gate, use_double_center, use_power_sharpen, min_gt_numbers,
        scale_factor=None)   # compute scale from actual matrix
    cov, cov_frac, min_cov, _, emd, _ = compute_ot_coverage(
        sim_t, gt_lens, gen_lens, reg, reg_m, coverage_threshold)

    # ---------- null baseline ----------
    null_covs = []
    for (null_chunks, null_embs) in null_gen_pool:
        if null_embs.numel() == 0:
            continue
        null_lens = np.maximum(
            np.array([len(c.split()) for c in null_chunks], dtype=np.float64), 1.0)
        sim_null = torch.matmul(gt_embeddings,
                                null_embs.t()).detach().cpu().numpy()
        # use the SAME scale_factor as the actual pass so null is comparable
        sim_null_t, _ = transform_sim_matrix(
            sim_null, gt_chunks, null_chunks,
            alpha, use_token_gate, use_double_center, use_power_sharpen, min_gt_numbers,
            scale_factor=scale_factor)
        n_cov, _, _, _, _, _ = compute_ot_coverage(
            sim_null_t, gt_lens, null_lens, reg, reg_m, coverage_threshold)
        null_covs.append(n_cov)

    null_mean = float(np.mean(null_covs)) if null_covs else 0.0
    denom = max(1e-6, 1.0 - null_mean)
    calibrated = float(np.clip((cov - null_mean) / denom, -1.0, 1.0))

    return cov, cov_frac, min_cov, null_mean, calibrated, emd


# ==============================
# Argument Parsing
# ==============================

def parse_args():
    p = argparse.ArgumentParser(description='Qwen OT Coverage — rigorous discriminative recipe')
    p.add_argument('--inference_results', required=True)
    p.add_argument('--ground_truth', required=True)
    p.add_argument('--output_key', required=True, choices=['output_1', 'output_2'])
    p.add_argument('--results_json', required=True)
    p.add_argument('--model_name', default='BAAI/bge-m3')
    p.add_argument('--gt_window_size',  type=int, default=1)
    p.add_argument('--gt_window_step',  type=int, default=1)
    p.add_argument('--gen_window_size', type=int, default=2)
    p.add_argument('--gen_window_step', type=int, default=1)
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--alpha',       type=float, default=4.0,
                   help='Power-sharpening exponent (default 4.0 for bge-m3).')
    p.add_argument('--null_k', type=int, default=20,
                   help='Number of cross-case null samples per case (default 20).')
    p.add_argument('--null_seed', type=int, default=42,
                   help='Random seed for null case sampling (default 42).')
    p.add_argument('--coverage_threshold', type=float, default=0.5)
    p.add_argument('--reg',   type=float, default=0.01)
    p.add_argument('--reg_m', type=float, default=0.1)
    p.add_argument('--no_token_gate',    dest='use_token_gate',    action='store_false', default=True)
    p.add_argument('--no_double_center', dest='use_double_center', action='store_false', default=True)
    p.add_argument('--no_power_sharpen', dest='use_power_sharpen', action='store_false', default=True)
    p.add_argument('--min_gt_numbers', type=int, default=2,
                   help='Min distinct numbers in GT chunk to activate token gate (default 2).')
    p.add_argument('--overwrite-existing', action='store_true')
    return p.parse_args()


# ==============================
# Main
# ==============================

def main():
    args = parse_args()

    print('=' * 80)
    print('Qwen OT Coverage — double-centering + power sharpen + token gate + null calibration')
    print('=' * 80)
    for k, v in vars(args).items():
        print(f'  {k:<30} {v}')
    print('=' * 80)

    # ---- Load data ----
    with open(args.inference_results, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, dict) and 'inference_results' in raw:
        inference_data = {x['input_case']: x
                          for x in raw['inference_results']
                          if 'input_case' in x}
    else:
        inference_data = raw
    print(f'Loaded {len(inference_data)} inference results')

    with open(args.ground_truth, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    print(f'Loaded {len(ground_truth_data)} ground-truth entries')

    results_path = Path(args.results_json)
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f'Loaded {len(results)} existing result entries')
    else:
        results = {}
        results_path.parent.mkdir(parents=True, exist_ok=True)
        print('Creating new results file...')

    # ---- Load model ----
    model, device = load_bge_model(args.model_name)

    # ---- Precompute GT embeddings ----
    gt_cache = precompute_gt_embeddings(
        ground_truth_data, args.output_key,
        model, device,
        args.gt_window_size, args.gt_window_step,
        inference_data, args.batch_size,
    )

    # ---- Precompute ALL gen embeddings (needed for null pool) ----
    gen_cache = precompute_gen_embeddings(
        inference_data, model, device,
        args.gen_window_size, args.gen_window_step,
        args.batch_size,
    )

    # ---- RNG for reproducible null sampling ----
    rng = np.random.default_rng(args.null_seed)
    all_gen_case_ids = list(gen_cache.keys())   # pool for null sampling

    processed = skipped = errors = 0

    print(f'\nComputing OT v2 metrics for {args.output_key}...')
    for case_id, inf_val in tqdm(inference_data.items(), desc='Processing'):
        existing = results.get(case_id, {})
        if not args.overwrite_existing and 'Chunk_OT_CalibratedCov' in existing:
            processed += 1
            continue

        if not inf_val.get('generated_output'):
            skipped += 1
            continue
        if case_id not in gt_cache or case_id not in gen_cache:
            skipped += 1
            continue

        gt_chunks    = gt_cache[case_id]['chunks']
        gt_embeddings = gt_cache[case_id]['embeddings']
        gen_chunks   = gen_cache[case_id]['chunks']
        gen_embeddings = gen_cache[case_id]['embeddings']

        if gt_embeddings.numel() == 0 or gen_embeddings.numel() == 0:
            skipped += 1
            continue

        gt_lens  = np.maximum(np.array([len(c.split()) for c in gt_chunks],  dtype=np.float64), 1.0)
        gen_lens = np.maximum(np.array([len(c.split()) for c in gen_chunks], dtype=np.float64), 1.0)

        # ---- Build null pool: K random OTHER cases ----
        other_ids = [cid for cid in all_gen_case_ids if cid != case_id]
        if len(other_ids) == 0:
            null_pool = []
        else:
            sample_size = min(args.null_k, len(other_ids))
            sampled_ids = rng.choice(other_ids, size=sample_size, replace=False)
            null_pool = [(gen_cache[cid]['chunks'], gen_cache[cid]['embeddings'])
                         for cid in sampled_ids if cid in gen_cache]

        try:
            cov, cov_frac, min_cov, null_mean, calibrated, emd = compute_metrics(
                gt_chunks, gt_embeddings,
                gen_chunks, gen_embeddings,
                null_pool, gt_lens, gen_lens,
                alpha=args.alpha,
                use_token_gate=args.use_token_gate,
                use_double_center=args.use_double_center,
                use_power_sharpen=args.use_power_sharpen,
                min_gt_numbers=args.min_gt_numbers,
                reg=args.reg, reg_m=args.reg_m,
                coverage_threshold=args.coverage_threshold,
            )
        except Exception as exc:
            tqdm.write(f'Error on {case_id}: {exc}')
            errors += 1
            continue

        if case_id not in results:
            results[case_id] = {'input_case': case_id, 'output_key': args.output_key}

        results[case_id]['Chunk_OT_Coverage']      = round(cov,        4)
        results[case_id]['Chunk_OT_CoveredFrac']   = round(cov_frac,   4)
        results[case_id]['Chunk_OT_MinChunkCov']   = round(min_cov,    4)
        results[case_id]['Chunk_OT_Null']          = round(null_mean,  4)
        results[case_id]['Chunk_OT_CalibratedCov'] = round(calibrated, 4)
        results[case_id]['Chunk_OT_EMD']           = round(emd,        4)
        processed += 1

        if processed % 10 == 0:
            tmp = results_path.with_suffix(results_path.suffix + f'.{os.getpid()}.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            os.replace(tmp, results_path)

    # Final save
    tmp = results_path.with_suffix(results_path.suffix + f'.{os.getpid()}.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, results_path)

    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'Processed : {processed}')
    print(f'Skipped   : {skipped}')
    print(f'Errors    : {errors}')
    print(f'Saved to  : {results_path}')
    print('=' * 80)
    print('✓ Done.')


if __name__ == '__main__':
    main()
