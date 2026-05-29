import os
import json
import argparse
import base64
from pathlib import Path
from io import BytesIO

from PIL import Image
from tqdm import tqdm
import requests


# Excluded input cases (hardcoded list) - will be skipped during inference
EXCLUDED_CASES = [
    "10_1016_j_compstruct_2023_116967",
    "10_1016_j_compstruct_2024_118781",
    "10_1016_j_jobe_2022_105346",
    "10_1007_s41104_016_0014_0",
    # Add more cases here as needed
]


# ---------- CLI Arguments ----------
parser = argparse.ArgumentParser(description='Inference with Gemini 3 Pro via OpenRouter')
parser.add_argument('--api_key', type=str, default="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",
                    help='OpenRouter API key')
parser.add_argument('--model', type=str, default="google/gemini-3-pro-preview",
                    help='Model name on OpenRouter (default: google/gemini-3-pro-preview)')
parser.add_argument('--test_prompts_jsonl', type=str, required=True,
                    help='Path to test prompts JSONL file')
parser.add_argument('--test_images_json', type=str, required=True,
                    help='Path to test images JSON file')
parser.add_argument('--output_json', type=str, required=True,
                    help='Output JSON filename')
parser.add_argument('--max_tokens', type=int, default=3000,
                    help='Maximum number of completion tokens (output only, reasoning tokens unlimited)')
parser.add_argument('--temperature', type=float, default=0.7,
                    help='Temperature for generation')
parser.add_argument('--top_p', type=float, default=0.9,
                    help='Top-p (nucleus sampling) parameter')
parser.add_argument('--site_url', type=str, default='https://github.com/kbali1297/fine_tune_llm_post_processing',
                    help='Site URL for OpenRouter rankings (optional)')
parser.add_argument('--site_name', type=str, default='Fine-tune LLM Post Processing',
                    help='Site name for OpenRouter rankings (optional)')
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
    """Convert image to base64 data URL."""
    img = Image.open(image_path).convert("RGB")
    
    # Resize to reduce payload size
    if resize:
        img = img.resize(resize, Image.BILINEAR)
    
    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"


def prepare_multimodal_prompt(user_prompt, image_paths):
    """
    Prepare multimodal prompt for OpenRouter.
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


def call_openrouter(api_key, model, messages, max_tokens, temperature, top_p, site_url, site_name):
    """Call OpenRouter API using official format."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Add optional headers for rankings
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name
    
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,  # Limits completion/output tokens only (reasoning unlimited for Gemini)
        "temperature": temperature,
        "top_p": top_p,
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    response.raise_for_status()
    
    return response.json()


def run_inference(args):
    """Main inference function."""
    output_path = Path.cwd() / args.output_json
    
    print("=" * 80)
    print("Gemini 2.5 Pro Inference via OpenRouter")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Output: {output_path}")
    print(f"API Key: {args.api_key[:20]}...")
    print("=" * 80)
    
    # Step 1: Load test data
    test_prompts, test_images = load_test_data(
        args.test_prompts_jsonl, 
        args.test_images_json
    )
    
    # Step 2: Check for existing results (resume capability)
    if output_path.exists():
        print(f"📂 Found existing results, loading...")
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
            processed = set(results.keys())
            print(f"✓ Loaded {len(results)} existing results\n")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️  Warning: Corrupted JSON file, starting fresh. Error: {e}\n")
            results = {}
            processed = set()
    else:
        results = {}
        processed = set()
    
    # Filter out already processed and excluded cases
    remaining = [p for p in test_prompts 
                 if p.get("input_case") not in processed 
                 and p.get("input_case") not in EXCLUDED_CASES]
    
    excluded_count = len([p for p in test_prompts if p.get("input_case") in EXCLUDED_CASES])
    if excluded_count > 0:
        print(f"🚫 Excluded {excluded_count} cases from inference")
    
    if not remaining:
        print("✅ All samples already processed!")
        return
    
    print(f"🚀 Processing {len(remaining)} samples...\n")
    
    # Step 3: Run inference
    for prompt_data in tqdm(remaining, desc="Generating", unit="sample"):
        input_case = prompt_data.get("input_case")
        system_prompt = prompt_data.get("system", "")
        user_prompt = prompt_data.get("user", "")
        
        # Fallback for old format (if using "text" field)
        if not user_prompt and "text" in prompt_data:
            user_prompt = prompt_data["text"]
        
        if input_case not in test_images:
            tqdm.write(f"⚠️  {input_case}: No images found, skipping")
            continue
        
        image_paths_ = list(test_images[input_case]["paths"].values())
        image_paths = [pp for pp in image_paths_ if os.path.exists(pp)]
        
        if len(image_paths) == 0:
            tqdm.write(f"⚠️  {input_case}: No existing images found, skipping")
            continue
        
        tqdm.write(f"📸 {input_case}: {len(image_paths)} images")
        
        try:
            # Prepare multimodal content
            content = prepare_multimodal_prompt(user_prompt, image_paths)
            
            # Build messages array with system prompt if provided
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                "role": "user",
                "content": content
            })
            
            # Call OpenRouter API
            response = call_openrouter(
                api_key=args.api_key,
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                site_url=args.site_url,
                site_name=args.site_name
            )
            
            # Extract generated text
            generated_text = response["choices"][0]["message"]["content"]
            
            # Store result
            results[input_case] = {
                "input_case": input_case,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "generated_output": generated_text,
                "num_images_used": len(image_paths),
                "model": args.model,
                "usage": response.get("usage", {}),
                "status": "success"
            }
            
            tqdm.write(f"✅ {input_case}: {len(generated_text)} chars")
            
            # Save after each sample (crash recovery)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        except Exception as e:
            tqdm.write(f"❌ {input_case}: {type(e).__name__}: {str(e)}")
            # Store error result with clear error flag
            results[input_case] = {
                "input_case": input_case,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "generated_output": None,  # Explicitly None for errors
                "error": str(e),
                "error_type": type(e).__name__,
                "num_images_used": len(image_paths) if 'image_paths' in locals() else 0,
                "status": "failed"
            }
            # Save error record immediately
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✅ Complete! Results: {output_path}")
    print(f"Processed: {len(results)}/{len(test_prompts)}")
    print("=" * 80)


if __name__ == "__main__":
    run_inference(args)
