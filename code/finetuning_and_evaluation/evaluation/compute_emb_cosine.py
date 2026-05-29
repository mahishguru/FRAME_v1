#!/usr/bin/env python3
"""
Qwen-Embedding Score Computation Script
Computes semantic similarity using Qwen3-Embedding-8B model.
Uses cosine similarity between embeddings from Qwen3-Embedding-8B.
Supports up to 8192 tokens without chunking.
"""

import json
import os
import argparse
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple


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


def load_qwen_model(model_name="Qwen/Qwen3-Embedding-8B"):
    """Load Qwen embedding model and tokenizer with optimizations."""
    print(f"Loading Qwen embedding model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load with optimizations
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Use BFloat16 to match Qwen training precision
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Enable inference optimizations
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("✓ Torch compilation enabled for faster inference")
        except Exception as e:
            print(f"⚠️  Torch compilation not available: {e}")
    
    # Print model configuration
    print(f"Model loaded on device: {device}")
    if hasattr(model.config, 'hidden_size'):
        print(f"📊 Embedding dimension: {model.config.hidden_size}")
    
    return model, tokenizer, device


def get_qwen_embedding_batch(texts: List[str], model, tokenizer, device, max_length=8192):
    """
    Get embeddings for a batch of texts (more efficient than one-by-one).
    
    Args:
        texts: List of input texts to embed
        model: Qwen model
        tokenizer: Qwen tokenizer
        device: torch device
        max_length: Maximum sequence length (default 8192 for Qwen3-Embedding-8B)
    
    Returns:
        Tensor of normalized embeddings (batch_size, embedding_dim)
    """
    # Clean inputs: strip whitespace and artifacts
    texts = [text.strip() for text in texts]
    
    # Tokenize batch with truncation
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length, 
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Use masked mean pooling (critical: ignore padding tokens)
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum valid tokens only
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        # Count valid tokens (clamp to avoid divide by zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Divide sum by count to get mean of valid tokens only
        embeddings = sum_embeddings / sum_mask
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings


def get_qwen_embedding(text, model, tokenizer, device, max_length=8192):
    """
    Get embedding from Qwen model (single text - uses batch version internally).
    
    Args:
        text: Input text to embed
        model: Qwen model
        tokenizer: Qwen tokenizer
        device: torch device
        max_length: Maximum sequence length (default 8192 for Qwen3-Embedding-8B)
    
    Note: Qwen3-Embedding-8B supports up to 8192 tokens, much longer than SBERT.
    """
    embeddings = get_qwen_embedding_batch([text], model, tokenizer, device, max_length)
    
    # Debug: Print embedding shape on first call (for verification)
    if not hasattr(get_qwen_embedding, '_shape_printed'):
        print(f"✓ Embedding output shape: {embeddings.shape} (batch_size, embedding_dim)")
        get_qwen_embedding._shape_printed = True
    
    return embeddings


def compute_qwen_similarity(generated: str, ground_truth: str, model, tokenizer, device, max_length=8192) -> float:
    """
    Compute similarity score using Qwen3-Embedding-8B.
    Uses cosine similarity between normalized embeddings.
    
    Args:
        max_length: Maximum tokens to process (default 8192 for Qwen3-Embedding-8B)
    
    Note: Qwen3-Embedding-8B supports up to 8192 tokens, which is sufficient for most texts.
    Texts longer than 8192 tokens will be automatically truncated.
    """
    # Get embeddings (strip inputs to remove whitespace artifacts)
    emb1 = get_qwen_embedding(generated.strip(), model, tokenizer, device, max_length)
    emb2 = get_qwen_embedding(ground_truth.strip(), model, tokenizer, device, max_length)
    
    # Compute cosine similarity
    cosine_score = F.cosine_similarity(emb1, emb2).item()
    
    return cosine_score


def atomic_write(data, path: Path):
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp_path, 'w', encoding='utf-8') as tmp_file:
        json.dump(data, tmp_file, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description='Compute Qwen-Embedding scores')
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
                        default='Qwen/Qwen3-Embedding-8B',
                        help='Qwen embedding model name')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for embedding computation (default: 8)')
    parser.add_argument('--save_frequency', type=int, default=50,
                        help='Save results every N samples (default: 50)')
    parser.add_argument('--overwrite-existing', action='store_true',
                        help='Overwrite existing Qwen_Embedding scores in results JSON')
    parser.add_argument('--gt_instruction_prefix', type=str,
                        default='Instruct: Retrieve scientific analysis passages that match the given reference passage\nQuery: ',
                        help='Instruction prefix for GT/query texts (default: Qwen3-Embedding asymmetric query prefix)')
    parser.add_argument('--gen_instruction_prefix', type=str,
                        default='',
                        help='Instruction prefix for generated/document texts (default: empty — doc side needs no prefix)')
    args = parser.parse_args()
    
    print("="*80)
    print("Qwen-Embedding Score Computation")
    print("="*80)
    print(f"Inference results: {args.inference_results}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Output key: {args.output_key}")
    print(f"Results JSON: {args.results_json}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save frequency: every {args.save_frequency} samples")
    print(f"GT  instruction prefix: '{args.gt_instruction_prefix[:60]}{'...' if len(args.gt_instruction_prefix) > 60 else ''}'")
    print(f"Gen instruction prefix: '{args.gen_instruction_prefix}'")
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
    
    # Initialize Qwen model
    model, tokenizer, device = load_qwen_model(args.model_name)
    max_length = 8192  # Qwen3-Embedding-8B supports up to 8192 tokens
    print(f"Model loaded successfully! Max sequence length: {max_length} tokens")
    print(f"⚠️  Note: Texts longer than {max_length} tokens will be automatically truncated")
    
    # Pre-compute ground truth embeddings (major optimization!)
    print(f"\nPre-computing ground truth embeddings...")
    ground_truth_embeddings = {}
    unique_ground_truths = {}
    
    # Collect unique ground truth texts to embed
    for input_case in tqdm(inference_data.keys(), desc="Collecting ground truths"):
        if input_case in ground_truth_data:
            gt_text = ground_truth_data[input_case].get(args.output_key, "")
            if gt_text:
                unique_ground_truths[input_case] = gt_text
    
    # Batch process ground truth embeddings
    gt_cases = list(unique_ground_truths.keys())
    gt_texts = list(unique_ground_truths.values())
    
    print(f"Computing embeddings for {len(gt_texts)} unique ground truth texts...")
    for i in tqdm(range(0, len(gt_texts), args.batch_size), desc="Ground truth embeddings"):
        batch_texts = gt_texts[i:i + args.batch_size]
        batch_cases = gt_cases[i:i + args.batch_size]
        # Apply asymmetric query-side prefix to GT texts
        prefixed_batch = [
            f"{args.gt_instruction_prefix}{t}".strip() if args.gt_instruction_prefix else t
            for t in batch_texts
        ]
        batch_embeddings = get_qwen_embedding_batch(prefixed_batch, model, tokenizer, device, max_length)
        
        for case, embedding in zip(batch_cases, batch_embeddings):
            ground_truth_embeddings[case] = embedding.cpu()  # Store on CPU to save GPU memory
    
    print(f"✓ Pre-computed {len(ground_truth_embeddings)} ground truth embeddings")
    
    # Compute Qwen-Embedding scores
    print(f"\nComputing Qwen-Embedding scores for {args.output_key}...")
    processed = 0
    skipped = 0
    truncated_count = 0
    
    # Prepare batches of items to process
    items_to_process = []
    for input_case, inference_value in inference_data.items():
        # Skip if already processed (unless overwrite flag is set)
        if not args.overwrite_existing:
            existing_entry = results.get(input_case, {})
            if "Emb_Cosine" in existing_entry:
                processed += 1
                continue
        
        # Skip if generated_output is null
        generated_output = inference_value.get("generated_output")
        if generated_output is None or generated_output == "":
            skipped += 1
            continue
        
        # Check if key exists in ground truth embeddings
        if input_case not in ground_truth_embeddings:
            skipped += 1
            continue
        
        items_to_process.append((input_case, generated_output))
    
    print(f"Processing {len(items_to_process)} new samples...")
    
    # Process in batches
    for batch_start in tqdm(range(0, len(items_to_process), args.batch_size), desc="Computing scores"):
        batch_items = items_to_process[batch_start:batch_start + args.batch_size]
        batch_cases = [item[0] for item in batch_items]
        batch_texts = [item[1] for item in batch_items]
        
        try:
            # Compute generated output embeddings in batch
            # Apply asymmetric doc-side prefix (empty by default for Qwen3-Embedding)
            prefixed_gen_texts = [
                f"{args.gen_instruction_prefix}{t}".strip() if args.gen_instruction_prefix else t
                for t in batch_texts
            ]
            gen_embeddings = get_qwen_embedding_batch(prefixed_gen_texts, model, tokenizer, device, max_length)
            
            # Compute similarities
            for i, input_case in enumerate(batch_cases):
                gen_emb = gen_embeddings[i]
                gt_emb = ground_truth_embeddings[input_case].to(device)
                
                # Compute cosine similarity
                cosine_score = F.cosine_similarity(gen_emb.unsqueeze(0), gt_emb.unsqueeze(0)).item()
                
                # Initialize entry if not exists
                if input_case not in results:
                    results[input_case] = {
                        "input_case": input_case,
                        "output_key": args.output_key
                    }
                
                # Update with embedding cosine similarity score
                results[input_case]["Emb_Cosine"] = round(cosine_score, 4)
                processed += 1
            
            # Periodic save
            if processed % args.save_frequency == 0:
                atomic_write(results, results_path)
                
        except Exception as e:
            tqdm.write(f"Error processing batch starting at {batch_cases[0]}: {e}")
            skipped += len(batch_items)
            continue
    
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
