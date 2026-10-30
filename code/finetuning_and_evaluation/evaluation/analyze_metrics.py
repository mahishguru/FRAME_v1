#!/usr/bin/env python3
"""
Metrics Analysis and Reporting Script
Reads metric results, computes statistics, and creates a report.

Supports 8 metrics:
- SBERT_SCORE (all-MiniLM-L6-v2 sentence embeddings)
- MoverScore (word-level with IDF weighting)
- ROUGE_L (longest common subsequence)
- METEOR (precision/recall harmonic using WordNet alignments)
- JACCARD (token overlap)
- Qwen_Embedding (Qwen3-Embedding-8B sentence embeddings)
- Qwen_Chunk_Recall (chunk-level MaxSim recall with Qwen embeddings)
- Qwen_MoverScore (MoverScore with Qwen embeddings)
"""

import json
import argparse
from pathlib import Path
import numpy as np


def load_results(results_json):
    """Load results from JSON file."""
    with open(results_json, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_statistics(results, metric_name):
    """Compute statistics for a given metric."""
    values = []
    for case_data in results.values():
        if metric_name in case_data and case_data[metric_name] is not None:
            # Skip string metrics (like reason fields)
            value = case_data[metric_name]
            if isinstance(value, (int, float)):
                values.append(value)
    
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0
        }
    
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "count": len(values)
    }


def detect_metrics(results):
    """Determine which metrics are present in the results file."""
    preferred_order = [
        "SBERT_SCORE",
        "MoverScore",
        "ROUGE_L",
        "METEOR",
        "JACCARD",
        "Emb_Cosine",
        "Qwen_Recall",
        "Qwen_Precision",
        "Qwen_F1",
        "Chunk_MatchedSimilarity",
        "Qwen_Chunk_Recall",
        "Qwen_MoverScore"
    ]
    base_fields = {"input_case", "output_key"}
    detected = set()
    for case in results.values():
        for key, value in case.items():
            if key in base_fields:
                continue
            if value is None:
                continue
            detected.add(key)

    ordered = [metric for metric in preferred_order if metric in detected]
    remaining = sorted(detected - set(ordered))
    return ordered + remaining


def generate_report(results, output_key, report_path):
    """Generate comprehensive report with statistics."""
    metrics = detect_metrics(results)
    
    report = {
        "output_key": output_key,
        "total_samples": len(results),
        "metrics": {}
    }
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS REPORT - {output_key}")
    print(f"{'='*80}")
    print(f"Total samples: {len(results)}")
    print(f"{'='*80}\n")
    
    if not metrics:
        print("No metrics found in results file. Report will contain counts only.")

    for metric in metrics:
        stats = compute_statistics(results, metric)
        report["metrics"][metric] = stats
        
        print(f"{metric}:")
        print(f"  Mean:   {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Std:    {stats['std']:.4f}")
        print(f"  Min:    {stats['min']:.4f}")
        print(f"  Max:    {stats['max']:.4f}")
        print(f"  Count:  {stats['count']}")
        print()
    
    # Save report
    print(f"Saving report to {report_path}...")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"{'='*80}")
    print("✓ Report generated successfully!")
    print(f"{'='*80}\n")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Analyze metrics and generate report')
    parser.add_argument('--results_json', type=str, required=True,
                        help='Path to results JSON file')
    parser.add_argument('--output_key', type=str, required=True,
                        choices=['output_1', 'output_2'],
                        help='Which output this analysis is for (output_1 or output_2)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save plots and report')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Name of the model for labeling plots')
    args = parser.parse_args()
    
    print("="*80)
    print("METRICS ANALYSIS AND REPORTING")
    print("="*80)
    print(f"Results JSON: {args.results_json}")
    print(f"Output key: {args.output_key}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model name: {args.model_name}")
    print("="*80)
    
    # Load results
    print("\nLoading results...")
    results = load_results(args.results_json)
    print(f"Loaded {len(results)} result entries")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate report with model name in filename
    model_name_clean = args.model_name.replace(" ", "_").replace("-", "_")
    report_path = output_dir / f"report_{args.output_key}_{model_name_clean}.json"
    report = generate_report(results, args.output_key, report_path)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Report saved to: {report_path}")
    print(f"{'='*80}")
    print("✓ Done!")


if __name__ == "__main__":
    main()
