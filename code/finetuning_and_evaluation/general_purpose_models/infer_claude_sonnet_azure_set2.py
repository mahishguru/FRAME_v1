#!/usr/bin/env python3
"""
Inference script for Claude 4.5 Sonnet via Azure AI Foundry (Anthropic endpoint).
Runs on test set 2 and mirrors the set1 script structure.
"""
import os
import json
import argparse
import base64
from pathlib import Path
from io import BytesIO

from PIL import Image
from tqdm import tqdm
from anthropic import AnthropicFoundry, APIError


def parse_optional_float(value: str):
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if value.lower() in {"none", "null", ""}:
        return None
    return float(value)

parser = argparse.ArgumentParser(description="Inference with Claude 4.5 Sonnet via Azure Foundry (Test Set 2)")
parser.add_argument('--api_key', type=str, default="FP5JvpTzN3SRbnjv1TWDZYOvp4znMKS6hpyt2lDlrYpnkuqegyokJQQJ99BCACfhMk5XJ3w3AAAAACOG1xDn",
                    help='Azure Foundry API key')
parser.add_argument('--endpoint', type=str, default="https://llmpost9832750527.services.ai.azure.com/anthropic/",
                    help='Azure Foundry Anthropic endpoint base URL')
parser.add_argument('--deployment', type=str, default="claude-sonnet-4-5",
                    help='Deployment name within Azure Foundry')
parser.add_argument('--test_prompts_jsonl', type=str, default="test_prompts_set2.jsonl",
                    help='Path to test prompts JSONL file')
parser.add_argument('--test_images_json', type=str, default="test_images_set2.json",
                    help='Path to test images JSON file')
parser.add_argument('--output_json', type=str, default="claude_sonnet_results_set2.json",
                    help='Output JSON filename')
parser.add_argument('--max_tokens', type=int, default=3000,
                    help='Maximum tokens to generate per sample')
parser.add_argument('--temperature', type=parse_optional_float, default="none",
                    help='Generation temperature (set to "none" to skip)')
parser.add_argument('--top_p', type=parse_optional_float, default="none",
                    help='Top-p sampling parameter (set to "none" to skip; only use if temperature is None)')
parser.add_argument('--resize_px', type=int, default=500,
                    help='Image resize dimension (square) before encoding')
parser.add_argument('--excluded_cases', type=str, nargs='*', default=[],
                    help='Specific input_case IDs to skip')
args = parser.parse_args()


def load_test_data(prompts_path: Path, images_path: Path):
    prompts = []
    with open(prompts_path, 'r', encoding='utf-8') as f:
        for line in f:
            prompts.append(json.loads(line))
    with open(images_path, 'r', encoding='utf-8') as f:
        images_data = json.load(f)
    print(f"Loaded {len(prompts)} test prompts")
    return prompts, images_data


def image_to_base64(image_path: str, resize_px: int = 500):
    img = Image.open(image_path).convert("RGB")
    if resize_px:
        img = img.resize((resize_px, resize_px), Image.BILINEAR)
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded, "image/jpeg"


def prepare_user_content(user_prompt: str, image_paths, resize_px: int):
    content = [{"type": "text", "text": user_prompt}]
    for img_path in image_paths:
        try:
            img_b64, media_type = image_to_base64(img_path, resize_px)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_b64,
                },
            })
        except Exception as e:
            print(f"Warning: Failed to encode image {img_path}: {e}")
    return content


def init_client(api_key: str, endpoint: str):
    base_url = endpoint if endpoint.endswith('/') else endpoint + '/'
    return AnthropicFoundry(
        api_key=api_key,
        base_url=base_url,
    )


def extract_text_blocks(response):
    parts = []
    for block in response.content:
        if getattr(block, 'type', None) == 'text':
            parts.append(block.text)
    return ''.join(parts)


def run_inference():
    script_dir = Path(__file__).parent
    prompts_path = script_dir / args.test_prompts_jsonl
    images_path = script_dir / args.test_images_json
    output_path = script_dir / args.output_json

    if args.temperature is not None and args.top_p is not None:
        raise ValueError("Only one of --temperature or --top_p can be provided. Set the other to None.")

    print("=" * 80)
    print("Claude 4.5 Sonnet Inference via Azure AI Foundry (Set 2)")
    print("=" * 80)
    print(f"Deployment: {args.deployment}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Output file: {output_path}")
    print("=" * 80)

    try:
        client = init_client(args.api_key, args.endpoint)
        print("Client initialized successfully.\n")
    except Exception as e:
        print(f"Failed to initialize Anthropic client: {e}")
        return

    test_prompts, test_images = load_test_data(prompts_path, images_path)

    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            processed = {k for k, v in results.items() if v.get('generated_output')}
            print(f"Resuming from {len(results)} stored results ({len(processed)} completed).\n")
        except (json.JSONDecodeError, ValueError):
            print("Warning: existing output file corrupt. Starting fresh.\n")
            results = {}
            processed = set()
    else:
        results = {}
        processed = set()

    skip_set = set(processed) | set(args.excluded_cases)
    remaining = [p for p in test_prompts if p.get('input_case') not in skip_set]
    if not remaining:
        print("All samples already processed.")
        return

    print(f"Processing {len(remaining)} samples...\n")

    for prompt_data in tqdm(remaining, desc="Generating", unit="sample"):
        input_case = prompt_data.get('input_case')
        system_prompt = prompt_data.get('system', '')
        user_prompt = prompt_data.get('user') or prompt_data.get('text', '')

        if not user_prompt:
            tqdm.write(f"⚠️  {input_case}: Missing user prompt, skipping")
            continue

        if input_case not in test_images:
            tqdm.write(f"⚠️  {input_case}: No images found, skipping")
            continue

        image_paths_all = list(test_images[input_case]['paths'].values())
        image_paths = [p for p in image_paths_all if os.path.exists(p)]
        if not image_paths:
            tqdm.write(f"⚠️  {input_case}: Image files missing, skipping")
            continue

        user_content = prepare_user_content(user_prompt, image_paths, args.resize_px)
        messages = [{"role": "user", "content": user_content}]

        try:
            response = client.messages.create(
                model=args.deployment,
                system=system_prompt or None,
                messages=messages,
                max_tokens=args.max_tokens,
                **({"temperature": args.temperature} if args.temperature is not None else {}),
                **({"top_p": args.top_p} if args.top_p is not None else {}),
            )
            generated_output = extract_text_blocks(response)
            usage = {}
            if getattr(response, 'usage', None):
                usage = {
                    "input_tokens": getattr(response.usage, 'input_tokens', None),
                    "output_tokens": getattr(response.usage, 'output_tokens', None),
                    "total_tokens": getattr(response.usage, 'total_tokens', None),
                }
            results[input_case] = {
                "input_case": input_case,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "generated_output": generated_output,
                "num_images_used": len(image_paths),
                "model": args.deployment,
                "usage": usage,
                "status": "success",
            }
            tqdm.write(f"✓ {input_case}: {len(generated_output)} chars")
        except APIError as api_err:
            tqdm.write(f"✗ {input_case}: APIError {api_err}")
            results[input_case] = {
                "input_case": input_case,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "generated_output": None,
                "error": str(api_err),
                "error_type": "APIError",
                "num_images_used": len(image_paths),
                "status": "failed",
            }
        except Exception as e:
            tqdm.write(f"✗ {input_case}: {type(e).__name__}: {e}")
            results[input_case] = {
                "input_case": input_case,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "generated_output": None,
                "error": str(e),
                "error_type": type(e).__name__,
                "num_images_used": len(image_paths),
                "status": "failed",
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Completed inference. Results stored in {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_inference()
