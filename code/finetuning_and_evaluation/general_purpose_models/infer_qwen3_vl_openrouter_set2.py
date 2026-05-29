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
    # Add any problematic cases here if needed
]

# ---------- CLI Arguments ----------
parser = argparse.ArgumentParser(description='Inference with Qwen3-VL-235B via OpenRouter')
parser.add_argument('--api_key', type=str, default="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",
                    help='OpenRouter API key')
parser.add_argument('--model', type=str, default="qwen/qwen3-vl-235b-a22b-instruct",
                    help='Model name on OpenRouter (default: qwen/qwen3-vl-235b-a22b-instruct)')
parser.add_argument('--test_prompts_jsonl', type=str, required=True,
                    help='Path to test prompts JSONL file')
parser.add_argument('--test_images_json', type=str, required=True,
                    help='Path to test images JSON file')
parser.add_argument('--output_json', type=str, required=True,
                    help='Output JSON filename')
parser.add_argument('--max_tokens', type=int, default=3000,
                    help='Maximum number of completion tokens')
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
    if resize:
        img = img.resize(resize, Image.BILINEAR)
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def prepare_multimodal_prompt(user_prompt, image_paths):
    content = [
        {
            "type": "text",
            "text": user_prompt
        }
    ]
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
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    response.raise_for_status()
    return response.json()

def run_inference(args):
    output_path = Path.cwd() / args.output_json
    print("=" * 80)
    print("Qwen3-VL-235B Inference via OpenRouter")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Output: {output_path}")
    print(f"API Key: {args.api_key[:20]}...")
    print("=" * 80)
    test_prompts, test_images = load_test_data(
        args.test_prompts_jsonl, 
        args.test_images_json
    )
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
    for prompt_data in tqdm(remaining, desc="Generating", unit="sample"):
        input_case = prompt_data.get('input_case')
        system_prompt = prompt_data.get('system', '')
        user_prompt = prompt_data.get('user', '')
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
        user_content = prepare_multimodal_prompt(user_prompt, image_paths)
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        messages.append({
            "role": "user",
            "content": user_content
        })
        try:
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
            generated_output = response['choices'][0]['message']['content']
            results[input_case] = {
                "input_case": input_case,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "generated_output": generated_output,
                "num_images_used": len(image_paths),
                "model": args.model,
                "usage": response.get('usage', {}),
                "status": "success"
            }
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            tqdm.write(f"✓ {input_case}: {response.get('usage', {}).get('total_tokens', 'N/A')} tokens")
        except Exception as e:
            tqdm.write(f"✗ {input_case}: {type(e).__name__}: {str(e)}")
            results[input_case] = {
                "input_case": input_case,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "generated_output": None,
                "error": str(e),
                "error_type": type(e).__name__,
                "num_images_used": len(image_paths) if 'image_paths' in locals() else 0,
                "status": "failed"
            }
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
    print("\n" + "=" * 80)
    print(f"Inference complete! Results saved to: {output_path}")
    print(f"Total cases processed: {len(results)}")
    print("=" * 80)

if __name__ == '__main__':
    run_inference(args)
