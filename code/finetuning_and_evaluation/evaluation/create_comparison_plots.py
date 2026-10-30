#!/usr/bin/env python3
"""
Comparison Plotting Script
Creates bar chart comparisons for all models across each metric.
Generates separate plots for output_1 and output_2.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_report(report_path):
    """Load a report JSON file."""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_metric_comparison_plot(metric_data, metric_name, output_key, output_path):
    """
    Create a bar chart comparing all models for a single metric.
    
    Args:
        metric_data: Dictionary {model_name: mean_score}
        metric_name: Name of the metric
        output_key: 'output_1' or 'output_2'
        output_path: Path to save the plot
    """
    # Sort models by score (descending)
    sorted_items = sorted(metric_data.items(), key=lambda x: x[1], reverse=True)
    models = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar chart
    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, scores, color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1.5)
    
    # Add value labels on top of bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{metric_name} Score', fontsize=14, fontweight='bold')
    ax.set_title(f'{metric_name} - {output_key.replace("_", " ").title()} Comparison',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
    
    # Set y-axis limits with some padding
    ax.set_ylim(0, max(scores) * 1.15 if scores else 1.0)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description='Create comparison plots from report files')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing report JSON files (e.g., evaluation/output1/results)')
    parser.add_argument('--output_key', type=str, required=True,
                        choices=['output_1', 'output_2'],
                        help='Which output to create plots for')
    parser.add_argument('--plots_dir', type=str, default=None,
                        help='Directory to save plots (default: results_dir/plots)')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Set plots directory
    if args.plots_dir:
        plots_dir = Path(args.plots_dir)
    else:
        plots_dir = results_dir / "plots"
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPARISON PLOTS GENERATION")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Output key: {args.output_key}")
    print(f"Plots directory: {plots_dir}")
    print("="*80)
    
    # Find all report files for this output_key
    pattern = f"report_{args.output_key}_*.json"
    report_files = list(results_dir.glob(pattern))
    
    if not report_files:
        print(f"\n❌ No report files found matching pattern: {pattern}")
        print(f"   in directory: {results_dir}")
        return
    
    print(f"\nFound {len(report_files)} report files:")
    for rf in sorted(report_files):
        print(f"  - {rf.name}")
    
    # Load all reports
    print("\nLoading reports...")
    reports = {}
    for report_file in report_files:
        # Extract model name from filename: report_output_1_ModelName.json
        model_name = report_file.stem.replace(f"report_{args.output_key}_", "").replace("_", "-")
        
        try:
            report = load_report(report_file)
            reports[model_name] = report
            print(f"  ✓ Loaded: {model_name}")
        except Exception as e:
            print(f"  ✗ Failed to load {report_file.name}: {e}")
    
    if not reports:
        print("\n❌ No reports loaded successfully")
        return
    
    # Define metrics
    metrics = ["SBERT_SCORE", "MoverScore", "ROUGE_L", "JACCARD", "Emb_Cosine", "Qwen_MoverScore"]
    
    # Create a plot for each metric
    print(f"\nGenerating comparison plots for {args.output_key}...")
    for metric in metrics:
        # Collect mean scores for this metric across all models
        metric_data = {}
        for model_name, report in reports.items():
            if "metrics" in report and metric in report["metrics"]:
                mean_score = report["metrics"][metric].get("mean", 0.0)
                metric_data[model_name] = mean_score
        
        if not metric_data:
            print(f"  ⚠ Skipping {metric}: No data found")
            continue
        
        # Create plot
        plot_filename = f"{metric}_{args.output_key}_comparison.png"
        plot_path = plots_dir / plot_filename
        
        create_metric_comparison_plot(
            metric_data=metric_data,
            metric_name=metric,
            output_key=args.output_key,
            output_path=plot_path
        )
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Models analyzed: {len(reports)}")
    print(f"Plots created: {len(metrics)}")
    print(f"Plots saved to: {plots_dir}")
    print(f"{'='*80}")
    print("✓ Done!")


if __name__ == "__main__":
    main()
