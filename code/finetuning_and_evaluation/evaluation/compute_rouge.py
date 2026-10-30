#!/usr/bin/env python3
"""
ROUGE-L Computation Script
Computes ROUGE-L scores between generated outputs and ground truth.
ROUGE-L measures longest common subsequence between texts.
"""

import json
import os
import argparse
from pathlib import Path
from rouge_score import rouge_scorer
from tqdm import tqdm


def normalize_inference_data(data: dict) -> dict:
    """
    Normalize inference data to a consistent dictionary format.
    
    Handles two formats:
    1. Dictionary format: {input_case: {generated_output: ...}, ...}
    2. List format: {inference_results: [{input_case: ..., generated_output: ...}, ...]}
    
    Returns:
        Dictionary keyed by input_case
    """
    # Check if it's the list format (has 'inference_results' key with a list)
    if 'inference_results' in data and isinstance(data['inference_results'], list):
        normalized = {}
        for item in data['inference_results']:
            input_case = item.get('input_case')
            if input_case:
                normalized[input_case] = item
        return normalized
    
    # Already in dictionary format
    return data


def atomic_write(data, path: Path):
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp_path, 'w', encoding='utf-8') as tmp_file:
        json.dump(data, tmp_file, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def compute_rouge_l(generated: str, ground_truth: str, scorer) -> float:
    """
    Compute ROUGE-L score.
    ROUGE-L measures longest common subsequence between texts.
    """
    scores = scorer.score(ground_truth, generated)
    # Return F1 score for ROUGE-L
    return scores['rougeL'].fmeasure


def main():
    parser = argparse.ArgumentParser(description='Compute ROUGE-L scores')
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
    args = parser.parse_args()
    
    print("="*80)
    print("ROUGE-L Score Computation")
    print("="*80)
    print(f"Inference results: {args.inference_results}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Output key: {args.output_key}")
    print(f"Results JSON: {args.results_json}")
    print("="*80)
    
    # Load inference results
    print("\nLoading inference results...")
    with open(args.inference_results, 'r', encoding='utf-8') as f:
        raw_inference_data = json.load(f)
    inference_data = normalize_inference_data(raw_inference_data)
    print(f"Loaded {len(inference_data)} inference results")
    
    # Load ground truth
    print("Loading ground truth...")
    with open(args.ground_truth, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    print(f"Loaded {len(ground_truth_data)} ground truth entries")
    
    # Load or initialize results JSON
    results_path = Path(args.results_json)
    if results_path.exists():
        print(f"Loading existing results from {results_path}...")
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing result entries")
    else:
        print("Creating new results file...")
        results = {}
        # Ensure directory exists
        results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize ROUGE scorer
    print("\nInitializing ROUGE scorer...")
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    print("ROUGE scorer ready!")
    
    # Compute ROUGE-L scores
    print(f"\nComputing ROUGE-L scores for {args.output_key}...")
    processed = 0
    skipped = 0
    
    for input_case, inference_value in tqdm(inference_data.items(), desc="Processing"):
        # Skip if generated_output is null
        generated_output = inference_value.get("generated_output")
        if generated_output is None or generated_output == "":
            tqdm.write(f"Skipping {input_case}: generated_output is null/empty")
            skipped += 1
            continue
        
        # Check if key exists in ground truth
        if input_case not in ground_truth_data:
            tqdm.write(f"Warning: {input_case} not found in ground truth. Skipping.")
            skipped += 1
            continue
        
        # Get ground truth
        ground_truth = ground_truth_data[input_case].get(args.output_key, "")
        if not ground_truth:
            tqdm.write(f"Warning: {input_case} has empty {args.output_key}. Skipping.")
            skipped += 1
            continue
        
        existing_entry = results.get(input_case, {})
        if "ROUGE_L" in existing_entry:
            processed += 1
            continue

        # Compute ROUGE-L score
        rouge_l_score = compute_rouge_l(generated_output, ground_truth, rouge_scorer_obj)
        
        # Initialize entry if not exists
        if input_case not in results:
            results[input_case] = {
                "input_case": input_case,
                "output_key": args.output_key
            }
        
        # Update with ROUGE-L score
        results[input_case]["ROUGE_L"] = round(rouge_l_score, 4)
        
        processed += 1
        atomic_write(results, results_path)
    
    # Final save
    print(f"\nSaving final results to {results_path}...")
    atomic_write(results, results_path)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Results saved to: {results_path}")
    print(f"{'='*80}")
    print("✓ Done!")


if __name__ == "__main__":
    main()
