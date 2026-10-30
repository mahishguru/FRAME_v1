import os
import json
import argparse
import base64
from pathlib import Path
from io import BytesIO

from PIL import Image
from tqdm import tqdm
from openai import AzureOpenAI


# Excluded input cases (hardcoded list) - will be skipped during inference
EXCLUDED_CASES = [
    "10_1016_j_compstruct_2023_116967",
    # Add more cases here as needed
]


# ---------- CLI Arguments ----------
parser = argparse.ArgumentParser(description='Inference with GPT-5 via Azure OpenAI')
parser.add_argument('--api_key', type=str,
                    default="",
                    help='Azure OpenAI API key')
parser.add_argument('--endpoint', type=str,
                    default="",
                    help='Azure OpenAI endpoint URL')
parser.add_argument('--deployment', type=str,
                    default="gpt-5.1",
                    help='Azure OpenAI deployment name')
parser.add_argument('--api_version', type=str,
                    default="2025-01-01-preview",
                    help='Azure OpenAI API version')
parser.add_argument('--test_prompts_jsonl', type=str, required=True,
                    help='Path to test prompts JSONL file')
parser.add_argument('--test_images_json', type=str, required=True,
                    help='Path to test images JSON file')
parser.add_argument('--output_json', type=str, required=True,
                    help='Output JSON filename')
parser.add_argument('--max_tokens', type=int, default=8192,
                    help='Maximum number of completion tokens (includes reasoning tokens for GPT-5)')
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
    """Convert image to base64 string for Azure OpenAI."""
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
    Prepare multimodal content array for Azure OpenAI SDK.
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


def initialize_azure_client(api_key, endpoint, api_version):
    """Initialize Azure OpenAI client with API key authentication."""
    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )
    
    return client


def run_inference(args):
    """Main inference function."""
    output_path = Path.cwd() / args.output_json
    
    print("=" * 80)
    print("GPT-5 Inference via Azure OpenAI SDK")
    print("=" * 80)
    print(f"Endpoint: {args.endpoint}")
    print(f"Deployment: {args.deployment}")
    print(f"API Version: {args.api_version}")
    print(f"Output: {output_path}")
    print("=" * 80)
    
    # Step 1: Initialize Azure OpenAI client
    print("🔐 Initializing Azure OpenAI client with API key...")
    try:
        client = initialize_azure_client(args.api_key, args.endpoint, args.api_version)
        print("✓ Client initialized successfully\n")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return
    
    # Step 2: Load test data
    test_prompts, test_images = load_test_data(
        args.test_prompts_jsonl, 
        args.test_images_json
    )
    
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
                if output and output.strip():  # Non-empty output
                    processed.add(case_id)
                else:  # Empty output - needs to be regenerated
                    empty_output_cases.append(case_id)
            
            print(f"✓ Loaded {len(results)} existing results")
            print(f"  - {len(processed)} with valid outputs (will skip)")
            print(f"  - {len(empty_output_cases)} with empty outputs (will regenerate)\n")
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
    
    # Step 4: Run inference on each test sample
    for prompt_data in tqdm(remaining, desc="Generating", unit="sample"):
        input_case = prompt_data.get('input_case')
        
        # Get system and user prompts
        system_prompt = prompt_data.get('system', '')
        user_prompt = prompt_data.get('user', '')
        
        # Fallback for old format (if using "text" field)
        if not user_prompt and "text" in prompt_data:
            user_prompt = prompt_data["text"]
        
        if input_case not in test_images:
            tqdm.write(f"⚠️  {input_case}: No images found, skipping")
            continue
        
        # Extract image paths from nested dict structure
        image_paths_ = list(test_images[input_case]["paths"].values())
        image_paths = [pp for pp in image_paths_ if os.path.exists(pp)]
        
        if len(image_paths) == 0:
            tqdm.write(f"⚠️  {input_case}: No existing images found, skipping")
            continue
        
        tqdm.write(f"📸 {input_case}: {len(image_paths)} images")
        
        # Prepare multimodal content
        user_content = prepare_multimodal_content(user_prompt, image_paths)
        
        # Build messages array using Azure OpenAI SDK format
        messages = []
        
        # Add developer/system message if present (using "developer" role for GPT-5)
        if system_prompt:
            messages.append({
                "role": "developer",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            })
        
        # Add user message with text and images
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Call API using Azure OpenAI SDK
        try:
            completion = client.chat.completions.create(
                model=args.deployment,
                messages=messages,
                max_completion_tokens=args.max_tokens,
                stop=None,
                stream=False
            )
            
            # Extract generated text
            generated_output = completion.choices[0].message.content
            finish_reason = completion.choices[0].finish_reason
            
            # Convert usage to dict (GPT-5 specific: includes reasoning tokens)
            usage_dict = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }
            
            # Add reasoning tokens if available (GPT-5 feature)
            if hasattr(completion.usage, 'completion_tokens_details'):
                details = completion.usage.completion_tokens_details
                if hasattr(details, 'reasoning_tokens'):
                    usage_dict["reasoning_tokens"] = details.reasoning_tokens
            
            # Debug: Check if output is empty and log finish reason
            if not generated_output or generated_output.strip() == "":
                tqdm.write(f"⚠️  {input_case}: Empty response! Finish reason: {finish_reason}, Reasoning tokens: {usage_dict.get('reasoning_tokens', 0)}")
            
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
            
            # Save incrementally after each sample
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            tqdm.write(f"✓ {input_case}: {usage_dict.get('total_tokens', 'N/A')} tokens")
            
        except Exception as e:
            tqdm.write(f"✗ {input_case}: {type(e).__name__}: {str(e)[:100]}")
            # Store error result with clear error flag
            results[input_case] = {
                "input_case": input_case,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "generated_output": None,  # Explicitly None for errors
                "error": str(e),
                "error_type": type(e).__name__,
                "num_images_used": len(image_paths),
                "status": "failed"
            }
            # Save error record immediately
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"Inference complete! Results saved to: {output_path}")
    print(f"Total cases processed: {len(results)}")
    print("=" * 80)


if __name__ == '__main__':
    run_inference(args)
