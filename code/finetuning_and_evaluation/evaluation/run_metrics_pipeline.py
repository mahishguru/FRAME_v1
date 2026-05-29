#!/usr/bin/env python3
"""
Parallel Metrics Computation Pipeline

This script orchestrates parallel metric computation across multiple inference results.
It uses Python's ProcessPoolExecutor to run up to N jobs in parallel, where each job:
1. Runs all metric computation scripts sequentially for one inference file
2. Acquires a per-results-file lock to avoid write conflicts
3. Calls analyze_metrics.py to generate reports

Configuration is loaded from metrics_config.json.
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import time
from queue import Queue
import threading


# Metric script mapping
METRIC_SCRIPTS = {
    "emb_cosine":    "compute_emb_cosine.py",
    "chunk":         "compute_chunk_similarity.py",
    "chunk_simple":  "compute_chunk_simple.py",
    "chunk_v2":      "compute_qwen_chunk_v2.py",
    "chunk_OT":      "compute_chunk_OT.py",
    "sbert_chunk": "compute_sbert_chunk.py",
    "nli_factuality": "compute_nli_factuality.py",
    "scale_factuality": "compute_scale_factuality.py",
    "rouge": "compute_rouge.py",
    "meteor": "compute_meteor.py",
    "llm_judge": "compute_llm_judge_vllm.py"
}

METRIC_OVERWRITE_FLAGS = {
    "emb_cosine":   "--overwrite-existing",
    "chunk":        "--overwrite-existing",
    "chunk_simple": "--overwrite-existing",
    "chunk_v2":     "--overwrite-existing",
    "chunk_OT":     "--overwrite-existing",
    "nli_factuality": "--overwrite-existing",
    "scale_factuality": "--overwrite-existing",
    "llm_judge": "--overwrite-existing",
}


def acquire_file_lock(lock_file_path, timeout=300):
    """
    Acquire an exclusive lock on a file.
    Returns the file handle (must be kept open to maintain lock).
    """
    lock_file = open(lock_file_path, 'w')
    start_time = time.time()
    
    while True:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_file
        except IOError:
            if time.time() - start_time > timeout:
                lock_file.close()
                raise TimeoutError(f"Could not acquire lock on {lock_file_path} within {timeout}s")
            time.sleep(0.5)


def release_file_lock(lock_file):
    """Release the file lock."""
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()


def run_command(command, description, env=None):
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(command)}\n")
    
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with return code {result.returncode}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        return False
    
    print(f"\n✓ {description} completed successfully!")
    return True


def process_inference_job(job_config, ground_truth, metrics_to_run, script_dir, gpu_id=None, overwrite_metrics=False):
    """
    Process a single inference job: run all metrics, then analyze.
    
    Args:
        job_config: Dict with inference job configuration
        ground_truth: Path to ground truth JSON
        metrics_to_run: List of metric names to compute
        script_dir: Path to directory containing metric scripts
        gpu_id: GPU ID to assign to this job (optional)
        overwrite_metrics: Whether to force supporting scripts to overwrite existing entries
    
    Returns:
        Tuple of (job_name, success_status, results_json_path)
    """
    job_name = f"{job_config['name']}_{job_config['output_key']}"
    
    print(f"\n{'#'*80}")
    print(f"# Processing Job: {job_name}")
    print(f"# GPU: {gpu_id if gpu_id is not None else 'N/A'}")
    print(f"{'#'*80}\n")
    
    # Set GPU environment variable if specified
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Create results directory
    results_dir = Path(job_config['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine results JSON path (same naming as run_all_metrics.py)
    model_name_clean = job_config['name'].replace(" ", "_").replace("-", "_")
    results_json = results_dir / f"metrics_{job_config['output_key']}_{model_name_clean}.json"
    
    # Create lock file path
    lock_file_path = results_json.with_suffix('.lock')
    
    # Acquire lock for this results file
    print(f"Acquiring lock for {results_json}...")
    try:
        lock_file = acquire_file_lock(lock_file_path)
    except TimeoutError as e:
        print(f"❌ Failed to acquire lock: {e}")
        return (job_name, False, str(results_json))
    
    try:
        # Initialize results JSON if it doesn't exist
        if not results_json.exists():
            print(f"Creating new results file: {results_json}")
            with open(results_json, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2)
        
        # Run each metric computation
        all_success = True
        for metric_name in metrics_to_run:
            if metric_name not in METRIC_SCRIPTS:
                print(f"⚠️  Warning: Unknown metric '{metric_name}', skipping...")
                continue
            
            script_path = script_dir / METRIC_SCRIPTS[metric_name]
            
            command = [
                sys.executable,
                str(script_path),
                "--inference_results", job_config['inference_results'],
                "--ground_truth", ground_truth,
                "--output_key", job_config['output_key'],
                "--results_json", str(results_json)
            ]
            
            if overwrite_metrics and metric_name in METRIC_OVERWRITE_FLAGS:
                command.append(METRIC_OVERWRITE_FLAGS[metric_name])

            success = run_command(
                command,
                f"Computing {metric_name} for {job_name}",
                env=env
            )
            
            if not success:
                all_success = False
                print(f"⚠️  Warning: {metric_name} computation failed, continuing...")
        
        # Run analysis and generate report
        analyze_command = [
            sys.executable,
            str(script_dir / "analyze_metrics.py"),
            "--results_json", str(results_json),
            "--output_key", job_config['output_key'],
            "--output_dir", str(results_dir),
            "--model_name", job_config['name']
        ]
        
        success = run_command(
            analyze_command,
            f"Analysis and reporting for {job_name}",
            env=env
        )
        
        if not success:
            all_success = False
        
        print(f"\n{'='*80}")
        print(f"Job {job_name} completed {'successfully' if all_success else 'with errors'}")
        print(f"Results: {results_json}")
        print(f"{'='*80}\n")
        
        return (job_name, all_success, str(results_json))
    
    finally:
        # Always release the lock
        release_file_lock(lock_file)
        # Clean up lock file
        try:
            lock_file_path.unlink()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(
        description='Run metrics computation pipeline in parallel'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='metrics_config.json',
        help='Path to configuration JSON file (default: metrics_config.json)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        help='Maximum number of parallel workers (overrides config)'
    )
    parser.add_argument(
        '--processes-per-gpu',
        type=int,
        default=3,
        help='Number of processes to run per GPU (default: 3)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be done without executing'
    )
    parser.add_argument(
        '--overwrite-metrics',
        action='store_true',
        help='Force supported metric scripts to overwrite existing entries'
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        choices=list(METRIC_SCRIPTS.keys()),
        help='Override the metrics list and run only the specified metrics (e.g., --metrics sbert_chunk)'
    )
    parser.add_argument(
        '--gpus',
        nargs='+',
        type=int,
        help='GPU IDs to use (e.g., --gpus 0 1 3). Defaults to all detected GPUs.'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        return 1
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Get configuration
    ground_truth = config['ground_truth']
    inference_jobs = config['inference_jobs']
    metrics_to_run = args.metrics if args.metrics else config.get('metrics', list(METRIC_SCRIPTS.keys()))
    
    # Detect available GPUs early to calculate proper max_workers
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True)
        available_gpus = [int(idx.strip()) for idx in result.stdout.strip().split('\n') if idx.strip()]
    except:
        available_gpus = [0, 1, 2, 3]  # Fallback

    # Filter to user-specified GPUs if provided
    if args.gpus:
        invalid = [g for g in args.gpus if g not in available_gpus]
        if invalid:
            print(f"Warning: GPU IDs {invalid} not found in detected GPUs {available_gpus}, ignoring them")
        available_gpus = [g for g in args.gpus if g in available_gpus]
        if not available_gpus:
            print(f"Error: None of the requested GPUs {args.gpus} are available")
            return 1
    
    # Determine max workers: either explicit or (num_gpus * processes_per_gpu)
    processes_per_gpu = args.processes_per_gpu
    if args.max_workers:
        max_workers = args.max_workers
    else:
        max_workers = len(available_gpus) * processes_per_gpu
    
    print("="*80)
    print("PARALLEL METRICS COMPUTATION PIPELINE")
    print("="*80)
    print(f"Configuration: {config_path}")
    print(f"Ground truth: {ground_truth}")
    print(f"Number of jobs: {len(inference_jobs)}")
    print(f"Detected GPUs: {available_gpus}")
    print(f"Processes per GPU: {processes_per_gpu}")
    print(f"Max parallel workers: {max_workers} ({len(available_gpus)} GPUs × {processes_per_gpu} processes)")
    print(f"Metrics to compute: {', '.join(metrics_to_run)}")
    print("="*80)
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Jobs to be processed:\n")
        
        for i, job in enumerate(inference_jobs, 1):
            gpu_slot = (i-1) % (len(available_gpus) * processes_per_gpu)
            gpu_id = available_gpus[gpu_slot % len(available_gpus)] if available_gpus else None
            print(f"{i}. {job['name']} ({job['output_key']})")
            print(f"   Inference: {job['inference_results']}")
            print(f"   Results: {job['results_dir']}")
            print(f"   Auto-assigned GPU: {gpu_id if gpu_id is not None else 'N/A'}")
        print("\n✓ Dry run complete. Use without --dry-run to execute.")
        return 0
    
    # Process jobs in parallel
    print(f"\n🚀 Starting parallel processing with {max_workers} workers...\n")
    
    # Create GPU queue for dynamic allocation
    # Each GPU can run processes_per_gpu processes simultaneously
    gpu_queue = Queue()
    for gpu_id in available_gpus:
        for _ in range(processes_per_gpu):
            gpu_queue.put(gpu_id)
    
    print(f"GPU queue initialized: {processes_per_gpu} slots per GPU = {len(available_gpus) * processes_per_gpu} total slots\n")
    
    results = []
    results_lock = threading.Lock()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit jobs dynamically as GPUs become available
        future_to_job = {}
        future_to_gpu = {}
        job_queue = list(inference_jobs)  # Copy of jobs to process
        active_futures = set()
        
        # Submit initial batch of jobs (up to max_workers)
        while len(active_futures) < max_workers and job_queue:
            job = job_queue.pop(0)
            gpu_id = gpu_queue.get()  # Block until GPU available
            
            future = executor.submit(
                process_inference_job,
                job,
                ground_truth,
                metrics_to_run,
                script_dir,
                gpu_id,
                args.overwrite_metrics
            )
            future_to_job[future] = job
            future_to_gpu[future] = gpu_id
            active_futures.add(future)
        
        # Process completed jobs and submit new ones
        while active_futures:
            # Wait for at least one job to complete
            done_futures = set()
            for future in as_completed(active_futures):
                done_futures.add(future)
                break  # Process one at a time to maintain queue
            
            for future in done_futures:
                job = future_to_job[future]
                gpu_id = future_to_gpu[future]
                
                try:
                    job_name, success, results_json = future.result()
                    with results_lock:
                        results.append({
                            'job': job_name,
                            'success': success,
                            'results_json': results_json
                        })
                except Exception as e:
                    job_name = f"{job['name']}_{job['output_key']}"
                    print(f"\n❌ Exception in job {job_name}: {e}")
                    with results_lock:
                        results.append({
                            'job': job_name,
                            'success': False,
                            'error': str(e)
                        })
                
                # Return GPU to queue
                gpu_queue.put(gpu_id)
                active_futures.remove(future)
                
                # Submit next job if any remain
                if job_queue:
                    next_job = job_queue.pop(0)
                    next_gpu_id = gpu_queue.get()
                    
                    next_future = executor.submit(
                        process_inference_job,
                        next_job,
                        ground_truth,
                        metrics_to_run,
                        script_dir,
                        next_gpu_id,
                        args.overwrite_metrics
                    )
                    future_to_job[next_future] = next_job
                    future_to_gpu[next_future] = next_gpu_id
                    active_futures.add(next_future)
    
    # Print final summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n✓ Successful jobs: {len(successful)}/{len(results)}")
    for r in successful:
        print(f"  - {r['job']}: {r['results_json']}")
    
    if failed:
        print(f"\n❌ Failed jobs: {len(failed)}/{len(results)}")
        for r in failed:
            error_msg = r.get('error', 'See logs above')
            print(f"  - {r['job']}: {error_msg}")
    
    print("\n" + "="*80)
    print("Pipeline execution complete!")
    print("To generate plots, run the plotting script manually.")
    print("="*80)
    
    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
