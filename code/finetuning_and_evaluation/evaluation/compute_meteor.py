#!/usr/bin/env python3
"""Compute METEOR scores for inference results compared to ground truth."""

import argparse
import json
import os
from pathlib import Path

import nltk
from nltk.translate.meteor_score import meteor_score
from tqdm import tqdm


NLTK_DATA_DIR = Path(os.environ.get("NLTK_DATA", Path.home() / ".cache" / "nltk_data"))
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
nltk.data.path = [str(NLTK_DATA_DIR)]


def ensure_nltk_resources():
    """Ensure required NLTK datasets are available in our local data dir."""
    resources = {
        "wordnet": NLTK_DATA_DIR / "corpora" / "wordnet",
        "omw-1.4": NLTK_DATA_DIR / "corpora" / "omw-1.4",
    }
    for resource, target in resources.items():
        if not target.exists():
            nltk.download(resource, download_dir=str(NLTK_DATA_DIR), quiet=True)


def normalize_inference_data(data: dict) -> dict:
    """Normalize inference JSON into {input_case: {...}} format."""
    if "inference_results" in data and isinstance(data["inference_results"], list):
        normalized = {}
        for item in data["inference_results"]:
            key = item.get("input_case")
            if key:
                normalized[key] = item
        return normalized
    return data


def compute_meteor_score(reference: str, candidate: str) -> float:
    """Compute METEOR score between reference and candidate strings."""
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    if not ref_tokens or not cand_tokens:
        return 0.0
    return float(meteor_score([ref_tokens], cand_tokens))


def atomic_write(data, path: Path):
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp_path, 'w', encoding='utf-8') as tmp_file:
        json.dump(data, tmp_file, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute METEOR scores")
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
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("METEOR Score Computation")
    print("=" * 80)
    print(f"Inference results: {args.inference_results}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Output key: {args.output_key}")
    print(f"Results JSON: {args.results_json}")
    print("=" * 80)

    # Ensure NLTK data is present
    ensure_nltk_resources()

    # Load inference data
    print("\nLoading inference results...")
    with open(args.inference_results, 'r', encoding='utf-8') as f:
        raw_inference = json.load(f)
    inference_data = normalize_inference_data(raw_inference)
    print(f"Loaded {len(inference_data)} inference results")

    # Load ground truth
    print("Loading ground truth...")
    with open(args.ground_truth, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    print(f"Loaded {len(ground_truth_data)} ground truth entries")

    # Load or initialize results file
    results_path = Path(args.results_json)
    if results_path.exists():
        print(f"Loading existing results from {results_path}...")
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing result entries")
    else:
        print("Creating new results file...")
        results = {}
        results_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    print(f"\nComputing METEOR scores for {args.output_key}...")
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

        ground_truth = ground_truth_data[input_case].get(args.output_key, "")
        if not ground_truth:
            tqdm.write(f"Warning: {input_case} has empty {args.output_key}. Skipping.")
            skipped += 1
            continue

        # Skip if METEOR already present to avoid overwriting
        existing_entry = results.get(input_case, {})
        if "METEOR" in existing_entry:
            processed += 1
            continue

        meteor = compute_meteor_score(ground_truth, generated_output)

        if input_case not in results:
            results[input_case] = {
                "input_case": input_case,
                "output_key": args.output_key
            }

        results[input_case]["METEOR"] = round(meteor, 4)
        processed += 1
        atomic_write(results, results_path)

    # Final save
    print(f"\nSaving final results to {results_path}...")
    atomic_write(results, results_path)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Results saved to: {results_path}")
    print("=" * 80)
    print("✓ Done!")


if __name__ == "__main__":
    main()
