#!/usr/bin/env python3
"""
Inference script for Llama-4-Maverick via Azure AI Foundry.
Uses OpenAI-compatible API for multimodal inference.

This script runs inference on test set 1.
"""

import os
import json
import argparse
import base64
from pathlib import Path
from io import BytesIO

from PIL import Image
from tqdm import tqdm
from openai import OpenAI


# ---------- CLI Arguments ----------
parser = argparse.ArgumentParser(description='Inference with Llama-4-Maverick via Azure AI Foundry')
parser.add_argument('--api_key', type=str,
                    default="",
                    help='Azure AI Foundry API key')
parser.add_argument('--endpoint', type=str,
                    default="",
                    help='Azure AI Foundry endpoint URL')
parser.add_argument('--deployment', type=str,
                    default="Llama-4-Maverick-17B-128E-Instruct-FP8",
                    help='Model deployment name')
parser.add_argument('--test_prompts_jsonl', type=str, 
                    default="test_prompts_set1.jsonl",
                    help='Path to test prompts JSONL file')
parser.add_argument('--test_images_json', type=str, 
                    default="test_images_set1.json",
                    help='Path to test images JSON file')
parser.add_argument('--output_json', type=str, 
                    default="llama4_maverick_results_set1.json",
                    help='Output JSON filename')
parser.add_argument('--max_tokens', type=int, default=3000,
                    help='Maximum number of tokens to generate')
parser.add_argument('--temperature', type=float, default=0.7,
                    help='Temperature for generation')
parser.add_argument('--top_p', type=float, default=0.9,
                    help='Top-p (nucleus sampling) parameter')
parser.add_argument('--api_version', type=str, default="2024-05-01-preview",
                    help='API version for the Azure endpoint')
args = parser.parse_args()


def load_test_data(prompts_path, images_path):
    """Load test prompts and image paths."""
    prompts = []
    with open(prompts_path, 'r') as f:
        for line in f:
            prompts.append(json.loads(line))
    
    with open(images_path, 'r') as f:
        images_data = json.load(f)
    
    print(f"Loaded {len(prompts)} test prompts")
    return prompts, images_data


def image_to_base64(image_path, resize=(500, 500)):
    """Convert image to base64 string for API."""
    img = Image.open(image_path).convert("RGB")
    
    # Resize to reduce payload size
    if resize:
        img = img.resize(resize, Image.BILINEAR)
    
    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Return data URL format
    return f"data:image/jpeg;base64,{img_base64}"


def prepare_multimodal_content(user_prompt, image_paths):
    """
    Prepare multimodal content array for OpenAI-compatible API.
    Formats user prompt with images.
    """
    # Build multimodal content array
    content = []
    
    # Add text first
    content.append({
        "type": "text",
        "text": user_prompt
    })
    
    # Add all images as base64 data URLs
    for img_path in image_paths:
        try:
            img_data_url = image_to_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": img_data_url
                }
            })
        except Exception as e:
            print(f"Warning: Failed to encode image {img_path}: {e}")
    
    return content


def initialize_client(api_key, endpoint, api_version):
    """Initialize OpenAI-compatible client for Azure AI Foundry."""
    base_url = endpoint.rstrip('/')
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers={"api-key": api_key},
        default_query={"api-version": api_version}
    )
    return client


def run_inference(args):
    """Main inference function."""
    script_dir = Path(__file__).parent
    prompts_path = script_dir / args.test_prompts_jsonl
    images_path = script_dir / args.test_images_json
    output_path = script_dir / args.output_json
    
    print("=" * 80)
    print("Llama-4-Maverick Inference via Azure AI Foundry")
    print("=" * 80)
    print(f"Endpoint: {args.endpoint}")
    print(f"Deployment: {args.deployment}")
    print(f"Test prompts: {prompts_path}")
    print(f"Test images: {images_path}")
    print(f"Output: {output_path}")
    print(f"Max tokens: {args.max_tokens}")
    print("=" * 80)
    
    # Step 1: Initialize client
    print("\n🔐 Initializing Azure AI Foundry client...")
    try:
        client = initialize_client(args.api_key, args.endpoint, args.api_version)
        print("✓ Client initialized successfully\n")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return
    
    # Step 2: Load test data
    test_prompts, test_images = load_test_data(prompts_path, images_path)
    
    # Step 3: Check for existing results (resume capability)
    if output_path.exists():
        print(f"📂 Found existing results, loading...")
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
            # Only consider cases as processed if they have non-empty generated_output
            processed = set()
            empty_output_cases = []
            for case_id, case_data in results.items():
                output = case_data.get("generated_output", "")
                if output and output.strip() and not output.startswith("ERROR"):
                    processed.add(case_id)
                else:
                    empty_output_cases.append(case_id)
            
            print(f"✓ Loaded {len(results)} existing results")
            print(f"  - {len(processed)} with valid outputs (will skip)")
            print(f"  - {len(empty_output_cases)} with empty/error outputs (will regenerate)\n")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️  Warning: Corrupted JSON file, starting fresh. Error: {e}\n")
            results = {}
            processed = set()
    else:
        results = {}
        processed = set()
    
    # Filter out already processed
    remaining = [p for p in test_prompts if p.get("input_case") not in processed]
    
    if not remaining:
        print("✅ All samples already processed!")
        return
    
    print(f"🚀 Processing {len(remaining)} samples...\n")
    
    # Step 4: Run inference
    for prompt_data in tqdm(remaining, desc="Generating", unit="sample"):
        input_case = prompt_data.get("input_case")
        system_prompt = prompt_data.get("system", "")
        user_prompt = prompt_data.get("user", "")
        
        if input_case not in test_images:
            tqdm.write(f"⚠️  {input_case}: No images found, skipping")
            continue
        
        # Get image paths
        image_paths_ = list(test_images[input_case]["paths"].values())
        image_paths = [pp for pp in image_paths_ if os.path.exists(pp)]
        
        if len(image_paths) == 0:
            tqdm.write(f"⚠️  {input_case}: No existing images found, skipping")
            continue
        
        tqdm.write(f"📸 {input_case}: {len(image_paths)} images")
        
        # Prepare multimodal content
        user_content = prepare_multimodal_content(user_prompt, image_paths)
        
        # Build messages array
        messages = []
        
        # Add system message if present
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add user message with text and images
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Call API
        try:
            completion = client.chat.completions.create(
                model=args.deployment,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            
            # Extract generated text
            generated_output = completion.choices[0].message.content
            finish_reason = completion.choices[0].finish_reason
            
            # Convert usage to dict
            usage_dict = {}
            if completion.usage:
                usage_dict = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }
            
            # Store result
            results[input_case] = {
                "input_case": input_case,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "generated_output": generated_output if generated_output else "",
                "num_images_used": len(image_paths),
                "model": args.deployment,
                "usage": usage_dict,
                "finish_reason": finish_reason,
                "status": "success"
            }
            
            tqdm.write(f"✅ {input_case}: {len(generated_output) if generated_output else 0} chars")
            
            # Save after each sample
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        except Exception as e:
            tqdm.write(f"❌ {input_case}: {type(e).__name__}: {str(e)[:100]}")
            results[input_case] = {
                "input_case": input_case,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "generated_output": f"ERROR: {str(e)}",
                "error": str(e),
                "error_type": type(e).__name__,
                "num_images_used": len(image_paths),
                "status": "failed"
            }
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✅ Complete! Results: {output_path}")
    print(f"Processed: {len(results)}/{len(test_prompts)}")
    print("=" * 80)


if __name__ == "__main__":
    run_inference(args)
