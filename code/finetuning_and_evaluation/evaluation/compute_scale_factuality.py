#!/usr/bin/env python3
"""
Compute SCALE-based factuality metrics.

Uses SCALE with Flan-T5-XL for evaluating how well LLM-generated outputs
cover ground truth facts via contextual alignment scoring.

Approach:
  1. Decompose GT into sentences
  2. Decompose generated text into sentences
    3. Proper SCALE (paper-faithful):
         - Coverage/recall: premise = full generated document; hypotheses = GT sentences
         - Precision:       premise = full ground-truth document; hypotheses = generated sentences
         SCALE handles long-context chunking internally via chunk_size/window_size.
    4. Length-weighted averages = SCALE_Coverage / SCALE_Precision

Score interpretation:
  - SCALE_Coverage ∈ [0, 1]:  fraction of GT facts covered by generated text
  - Higher scores = better factuality (GT facts entailed/aligned)

Model: SCALE with Flan-T5-XL
  - Optimized for long contexts (handles 4k+ token GTs)
  - Contextual alignment scoring (better than SummaC for QA tasks)
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm

# SCALE imports
SCALEScorer = None
try:
    from scale_score.scorer import SCALEScorer as _SCALEScorer
    SCALEScorer = _SCALEScorer
except Exception:
    # Optional dependency — only required when backend=scale_flan
    SCALEScorer = None

# Sentence splitting
import nltk


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def select_cases(
    inference_data: dict,
    case_list_path: Optional[str] = None,
    max_cases: int = 0,
    sample_strategy: str = "random",
    sample_seed: int = 42,
) -> list:
    """Select a subset of case IDs to process.

    - If case_list_path is provided, uses it (one case ID per line).
    - Else uses all cases in inference_data.
    - If max_cases > 0, truncates/sample to that size.
    """
    cases = list(inference_data.keys())
    if case_list_path:
        p = Path(case_list_path)
        if not p.exists():
            raise FileNotFoundError(f"case_list not found: {case_list_path}")
        requested = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        requested_set = set(requested)
        cases = [c for c in cases if c in requested_set]

    cases = sorted(cases)

    if max_cases and max_cases > 0 and len(cases) > max_cases:
        if sample_strategy == "first":
            cases = cases[:max_cases]
        elif sample_strategy == "random":
            rng = np.random.default_rng(sample_seed)
            idx = rng.choice(len(cases), size=max_cases, replace=False)
            cases = [cases[i] for i in sorted(idx)]
        else:
            raise ValueError(f"Unknown sample_strategy: {sample_strategy}")

    return cases


# ---------------------------------------------------------------------------
# NLTK setup
# ---------------------------------------------------------------------------

NLTK_DATA_DIR = Path(os.environ.get("NLTK_DATA", Path.home() / ".cache" / "nltk_data"))
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
nltk.data.path = [str(NLTK_DATA_DIR)]

_NLTK_SENT_OK = False


def ensure_sentence_tokenizer():
    """Ensure NLTK sentence tokenizer is available."""
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


# ---------------------------------------------------------------------------
# Sentence utilities
# ---------------------------------------------------------------------------

def sent_tokenize(text: str) -> list:
    """
    Split text into sentences with smart handling of structured vs prose text.
    
    For structured text (markdown, bullets):
    - Treats each top-level bullet/header as a separate unit
    - Groups indented sub-bullets with their parent bullet/header
    
    For prose:
    - Uses NLTK sentence tokenization
    
    Returns list of sentence strings.
    """
    if not text or not text.strip():
        return []
    
    # Detect if text is structured (has markdown/bullets/numbered lists)
    has_headers = bool(re.search(r'^\s*#+\s+', text, re.MULTILINE))
    has_bullets = bool(re.search(r'^\s*[-*]\s+', text, re.MULTILINE))
    has_numbered = bool(re.search(r'^\s*\d+[\.)]\s+', text, re.MULTILINE))

    # Treat numbered lists as structured bullets too (common in Claude/Gemini)
    bullet_re = re.compile(r'^\s*([-*]|\d+[\.)])\s+')
    
    if has_headers or has_bullets or has_numbered:
        # Structured text: split on semantic units (headers/bullets)
        units = []
        lines = text.split('\n')
        current_unit = []

        # If a structured unit becomes extremely long (e.g., a whole "strategy"
        # section), SCALE recall can get penalized due to premise dilution /
        # truncation. Keep units reasonably sized.
        max_unit_words = 120
        
        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            
            # Empty line - might be unit boundary
            if not stripped:
                if current_unit:
                    units.append('\n'.join(current_unit))
                    current_unit = []
                continue
            
            # Header - new unit
            if re.match(r'^\s*#+\s+', line):
                if current_unit:
                    units.append('\n'.join(current_unit))
                current_unit = [line]
            # Bullet lines (including numbered lists)
            elif bullet_re.match(line):
                # Indented bullets are treated as sub-bullets and grouped
                # with the current parent unit.
                if indent >= 2:
                    if not current_unit:
                        current_unit = [line]
                    else:
                        current_unit.append(line)
                # Top-level bullet starts a new unit.
                else:
                    if current_unit:
                        units.append('\n'.join(current_unit))
                    current_unit = [line]
            # Sub-bullet or continuation - add to current unit
            else:
                current_unit.append(line)

            # Enforce a soft max length for structured units
            if current_unit:
                wc = len(' '.join(current_unit).split())
                if wc > max_unit_words:
                    units.append('\n'.join(current_unit))
                    current_unit = []
        
        # Don't forget the last unit
        if current_unit:
            units.append('\n'.join(current_unit))
        
        # Filter out empty units
        sents = [u.strip() for u in units if u.strip()]
        
    else:
        # Prose text: use sentence tokenization
        ensure_sentence_tokenizer()
        try:
            sents = nltk.sent_tokenize(text)
        except Exception as e:
            # Fallback: split on common sentence delimiters
            sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Keep short but meaningful technical fragments (e.g., "S1.", "A.")
        # to avoid dropping concise factual units.
        sents = [s.strip() for s in sents if s.strip()]
    
    return sents if sents else [text.strip()]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_scale_scorer(device: str = "cuda", size: str = "xl"):
    """
    Load SCALE scorer with Flan-T5.

    Args:
        device: 'cuda' or 'cpu'
        size: 'xl' (Flan-T5-XL, best) or 'large' (Flan-T5-Large, faster)

    Returns:
        SCALEScorer instance
    """
    print(f"\nLoading SCALE scorer: Flan-T5-{size.upper()}")
    print(f"Device: {device}")
    print("=" * 80)

    scorer = SCALEScorer(size=size, device=device)

    # Suppress misleading Transformers warnings like:
    # "Token indices sequence length is longer than ... (3028 > 512)"
    # The installed SCALE implementation chunks premises internally, but it
    # still tokenizes some long strings without truncation, which triggers the
    # warning because Flan-T5 reports model_max_length=512.
    # Bumping this value avoids noisy logs without changing chunking behavior
    # (we always pass an explicit chunk_size).
    try:
        if getattr(scorer, "tokenizer", None) is not None:
            scorer.tokenizer.model_max_length = int(1_000_000)
            if hasattr(scorer.tokenizer, "init_kwargs") and isinstance(scorer.tokenizer.init_kwargs, dict):
                scorer.tokenizer.init_kwargs["model_max_length"] = int(1_000_000)
    except Exception:
        pass

    print(f"✓ SCALE scorer loaded (Flan-T5-{size.upper()})")
    print("  Optimized for long contexts (4k+ tokens)")
    print("=" * 80)

    return scorer


# ---------------------------------------------------------------------------
# SCALE scoring
# ---------------------------------------------------------------------------

def compute_scale_coverage(
    ground_truth: str,
    generated: str,
    scorer: Any,
    chunk_size: int = 1000,
    window_size: float = 0.25,
) -> dict:
    """
    Compute proper SCALE coverage/precision/F1.

    Paper-faithful usage:
      - Coverage/recall: premise = full generated document; hypotheses = GT sentences
      - Precision:       premise = full ground-truth document; hypotheses = generated sentences

    SCALE internally chunks long premises using (chunk_size, window_size). We
    intentionally do NOT pre-chunk premises here.

    Args:
        ground_truth: Reference text
        generated: LLM output text
        scorer: SCALEScorer instance
        chunk_size: SCALE premise chunk size in tokens (internal to SCALE).
        window_size: SCALE premise window overlap fraction (internal to SCALE).

    Returns:
        dict with keys:
            SCALE_Coverage: float in [0,1] — fraction of GT facts covered
                n_gt_sents: number of GT sentences
                n_gen_sents: number of generated sentences
    """
    if not ground_truth or not generated:
        return {
            "SCALE_Coverage": 0.0,
            "SCALE_Precision": 0.0,
            "SCALE_F1": 0.0,
            "n_gt_sents": 0,
            "n_gen_sents": 0,
        }

    gt_sents = sent_tokenize(ground_truth)
    gen_sents = sent_tokenize(generated)

    if not gt_sents or not gen_sents:
        return {
            "SCALE_Coverage": 0.0,
            "SCALE_Precision": 0.0,
            "SCALE_F1": 0.0,
            "n_gt_sents": len(gt_sents),
            "n_gen_sents": len(gen_sents),
        }

    try:
        # ── Coverage/recall pass ─────────────────────────────────────────
        coverage_scores = scorer.score([generated], [gt_sents], chunk_size, window_size)
        best_coverage = np.array(coverage_scores, dtype=np.float64)  # (n_gt,)

        gt_lens = np.array([max(len(s.split()), 1) for s in gt_sents], dtype=np.float64)
        w_gt = gt_lens / gt_lens.sum()
        scale_coverage = float(np.sum(w_gt * best_coverage))

        # ── Precision pass ──────────────────────────────────────────────
        precision_scores = scorer.score([ground_truth], [gen_sents], chunk_size, window_size)
        best_precision = np.array(precision_scores, dtype=np.float64)  # (n_gen,)

        gen_lens = np.array([max(len(s.split()), 1) for s in gen_sents], dtype=np.float64)
        w_gen = gen_lens / gen_lens.sum()
        scale_precision = float(np.sum(w_gen * best_precision))

        # ── F1 ──────────────────────────────────────────────────────────
        if scale_coverage + scale_precision > 0:
            scale_f1 = 2 * scale_coverage * scale_precision / (scale_coverage + scale_precision)
        else:
            scale_f1 = 0.0

    except Exception as e:
        print(f"Error in SCALE scoring: {e}")
        return {
            "SCALE_Coverage": 0.0,
            "SCALE_Precision": 0.0,
            "SCALE_F1": 0.0,
            "n_gt_sents": len(gt_sents),
            "n_gen_sents": len(gen_sents),
            "error": str(e)
        }

    return {
        "SCALE_Coverage":   round(float(np.clip(scale_coverage, 0.0, 1.0)),   4),
        "SCALE_Precision":  round(float(np.clip(scale_precision, 0.0, 1.0)),  4),
        "SCALE_F1":         round(float(np.clip(scale_f1, 0.0, 1.0)),         4),
        "n_gt_sents":  len(gt_sents),
        "n_gen_sents": len(gen_sents),
    }


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def normalize_inference_data(data: dict) -> dict:
    """Normalize inference data from list to dict format."""
    if "inference_results" in data and isinstance(data["inference_results"], list):
        normalized = {}
        for item in data["inference_results"]:
            key = item.get("input_case")
            if key:
                normalized[key] = item
        return normalized
    return data


def atomic_write(data, path: Path):
    """Write data to JSON file atomically."""
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
    with open(tmp_path, "w", encoding="utf-8") as tmp_file:
        json.dump(data, tmp_file, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute SCALE-based factuality metrics (GT coverage)"
    )
    parser.add_argument(
        "--backend", type=str, default="scale_flan",
        choices=["scale_flan"],
        help="Scoring backend (proper SCALE / Flan-T5).",
    )
    parser.add_argument(
        "--inference_results", type=str, required=True,
        help="Path to inference results JSON file",
    )
    parser.add_argument(
        "--ground_truth", type=str,
        default="/data/mguru/04_Finetuning/frame-finetuning-evaluation/final_output.json",
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "--output_key", type=str, required=True,
        choices=["output_1", "output_2"],
        help="Which output to compare against (output_1 or output_2)",
    )
    parser.add_argument(
        "--results_json", type=str, required=True,
        help="Path to results JSON file (will be created/updated)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run model on (default: cuda)",
    )
    parser.add_argument(
        "--model_size", type=str, default="xl",
        choices=["xl", "large"],
        help="SCALE model size: 'xl' (Flan-T5-XL, best) or 'large' (faster)",
    )

    # Fast subset mode
    parser.add_argument(
        "--case_list", type=str, default=None,
        help="Optional path to a text file with case IDs (one per line) to process.",
    )
    parser.add_argument(
        "--max_cases", type=int, default=0,
        help="If >0, process only this many cases (after case_list filtering).",
    )
    parser.add_argument(
        "--sample_strategy", type=str, default="random",
        choices=["random", "first"],
        help="How to select max_cases from the candidate set (default: random).",
    )
    parser.add_argument(
        "--sample_seed", type=int, default=42,
        help="RNG seed for random sampling (default: 42).",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=1024,
        help="Chunk size for SCALE scoring (handles long texts, default: 1024)",
    )
    parser.add_argument(
        "--window_size", type=float, default=0.25,
        help="SCALE internal window overlap fraction (default: 0.25)",
    )
    parser.add_argument(
        "--overwrite-existing", action="store_true",
        help="Recompute SCALE metrics even if values already exist",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 80)
    print("SCALE Factuality Evaluation (Proper SCALE / Flan-T5)")
    print("=" * 80)
    print(f"Inference results: {args.inference_results}")
    print(f"Ground truth:      {args.ground_truth}")
    print(f"Output key:        {args.output_key}")
    print(f"Results JSON:      {args.results_json}")
    print(f"Backend:           {args.backend}")
    print(f"Device:            {args.device}")
    print(f"Model size:        {args.model_size}")
    print(f"Chunk size:        {args.chunk_size}")
    print(f"Window size:       {args.window_size}")
    print(f"Overwrite:         {args.overwrite_existing}")
    if args.case_list or args.max_cases:
        print(f"Subset:            case_list={args.case_list}, max_cases={args.max_cases}, strategy={args.sample_strategy}, seed={args.sample_seed}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    with open(args.inference_results, "r", encoding="utf-8") as f:
        raw_inference = json.load(f)
    inference_data = normalize_inference_data(raw_inference)
    print(f"  Inference results: {len(inference_data)} entries")

    with open(args.ground_truth, "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)
    print(f"  Ground truth:      {len(ground_truth_data)} entries")

    # Load or create results file
    results_path = Path(args.results_json)
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"  Existing results:  {len(results)} entries")
    else:
        results = {}
        results_path.parent.mkdir(parents=True, exist_ok=True)
        print("  Creating new results file...")

    # Device
    device = args.device if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")

    if SCALEScorer is None:
        raise ImportError(
            "scale_flan requires scale-score. Install with: pip install scale-score"
        )
    scorer = load_scale_scorer(device, size=args.model_size)

    processed = 0
    skipped = 0
    errors = 0
    coverage_scores = []

    print(f"\nComputing SCALE coverage for {args.output_key}...")
    print("Metric: For each GT sentence, find best alignment in generated text")
    print("        Length-weighted average = SCALE_Coverage (0-1)")
    print("=" * 80)

    selected_cases = select_cases(
        inference_data,
        case_list_path=args.case_list,
        max_cases=args.max_cases,
        sample_strategy=args.sample_strategy,
        sample_seed=args.sample_seed,
    )
    selected_set = set(selected_cases)
    print(f"\nSelected cases to process: {len(selected_cases)} / {len(inference_data)}")

    for input_case in tqdm(selected_cases, desc="SCALE Coverage"):
        inference_value = inference_data[input_case]
        # Skip if already computed and not overwriting
        existing_entry = results.get(input_case, {})
        existing_backend = existing_entry.get("SCALE_Backend")
        if not args.overwrite_existing:
            # Skip only when the existing entry is already from the *proper*
            # SCALE backend *and* it matches the current scoring parameters.
            # This allows reruns with different chunk/window settings to
            # automatically recompute without requiring --overwrite-existing.
            existing_chunk = existing_entry.get("SCALE_ChunkSize")
            existing_window = existing_entry.get("SCALE_WindowSize")
            same_params = (
                (existing_chunk is not None)
                and (int(existing_chunk) == int(args.chunk_size))
                and (existing_window is not None)
                and (abs(float(existing_window) - float(args.window_size)) < 1e-12)
            )
            if ("SCALE_F1" in existing_entry) and (existing_backend == "scale_flan_proper") and same_params:
                processed += 1
                continue

        generated_output = inference_value.get("generated_output")
        if not generated_output:
            skipped += 1
            continue

        if input_case not in ground_truth_data:
            skipped += 1
            continue

        ground_truth = ground_truth_data[input_case].get(args.output_key, "")
        if not ground_truth:
            skipped += 1
            continue

        # Compute metrics
        try:
            scores = compute_scale_coverage(
                ground_truth,
                generated_output,
                scorer,
                chunk_size=args.chunk_size,
                window_size=args.window_size,
            )
        except Exception as exc:
            tqdm.write(f"Error for {input_case}: {exc}")
            errors += 1
            continue

        # Initialize or update result entry
        if input_case not in results:
            results[input_case] = {
                "input_case": input_case,
                "output_key": args.output_key,
            }

        # Store metrics
        results[input_case]["SCALE_Coverage"]  = scores["SCALE_Coverage"]
        results[input_case]["SCALE_Precision"] = scores["SCALE_Precision"]
        results[input_case]["SCALE_F1"]        = scores["SCALE_F1"]
        results[input_case]["SCALE_Backend"]   = "scale_flan_proper"
        results[input_case]["SCALE_ChunkSize"] = int(args.chunk_size)
        results[input_case]["SCALE_WindowSize"] = float(args.window_size)

        coverage_scores.append(scores["SCALE_Coverage"])
        processed += 1

        # Save after each case for maximum progress preservation
        atomic_write(results, results_path)

    # Final save
    print(f"\nSaving final results to {results_path}...")
    atomic_write(results, results_path)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Processed:  {processed}")
    print(f"  Skipped:    {skipped}")
    print(f"  Errors:     {errors}")
    if coverage_scores:
        all_prec  = [results[c]["SCALE_Precision"] for c in results if "SCALE_Precision" in results[c]]
        all_f1    = [results[c]["SCALE_F1"]        for c in results if "SCALE_F1"        in results[c]]
        print(f"\n  SCALE_Coverage statistics:")
        print(f"    Mean:   {np.mean(coverage_scores):.4f}")
        print(f"    Median: {np.median(coverage_scores):.4f}")
        print(f"    Std:    {np.std(coverage_scores):.4f}")
        if all_prec:
            print(f"\n  SCALE_Precision statistics:")
            print(f"    Mean:   {np.mean(all_prec):.4f}")
            print(f"    Median: {np.median(all_prec):.4f}")
        if all_f1:
            print(f"\n  SCALE_F1 statistics:")
            print(f"    Mean:   {np.mean(all_f1):.4f}")
            print(f"    Median: {np.median(all_f1):.4f}")
    print(f"\n  Results saved to: {results_path}")
    print("=" * 80)
    print("\nMetric interpretation:")
    print("  SCALE_Coverage  ∈ [0,1]: fraction of GT facts covered by generation (recall)")
    print("  SCALE_Precision ∈ [0,1]: fraction of gen content grounded in GT (precision)")
    print("  SCALE_F1        ∈ [0,1]: harmonic mean — length-invariant quality score")
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
