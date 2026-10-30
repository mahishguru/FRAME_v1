import os
import json
import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


# ---------- CLI Arguments ----------
parser = argparse.ArgumentParser(description='Inference with LoRA fine-tuned LLaVA-OV model')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to fine-tuned model checkpoint (e.g., /path/to/checkpoint-final)')
parser.add_argument('--base_model', type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
                    help='Base model name (same as used in training)')
parser.add_argument('--test_prompts_jsonl', type=str, required=True,
                    help='Path to test prompts JSONL file')
parser.add_argument('--test_images_json', type=str, required=True,
                    help='Path to test images JSON file')
parser.add_argument('--output_json', type=str, required=True,
                    help='Output JSON filename')
parser.add_argument('--max_new_tokens', type=int, default=2048,
                    help='Maximum number of tokens to generate')
parser.add_argument('--max_seq_length', type=int, default=28000,
                    help='Maximum sequence length for filtering (same as training)')
parser.add_argument('--temperature', type=float, default=0.7,
                    help='Temperature for generation')
parser.add_argument('--top_p', type=float, default=0.9,
                    help='Top-p (nucleus sampling) parameter')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device: "cuda", "cpu", or "cuda:0"')
args = parser.parse_args()


def load_test_data(prompts_path, images_path):
    """Load test prompts and image paths - SAME as training script."""
    prompts = []
    with open(prompts_path, 'r') as f:
        for line in f:
            prompts.append(json.loads(line))
    
    with open(images_path, 'r') as f:
        images_data = json.load(f)
    
    print(f"Loaded {len(prompts)} test prompts")
    return prompts, images_data


def filter_prompts(prompts, images_data, processor, max_seq_length):
    """Filter prompts by token count - EXACT same logic as training script."""
    TOKENS_PER_IMAGE = 730  # Same as training
    filtered_prompts = []
    excluded_samples = []
    
    print("\n" + "=" * 80)
    print("Filtering samples based on token count...")
    print("=" * 80)
    
    for p in prompts:
        full_prompt = p["text"]
        input_case = p.get("input_case")
        
        # Extract ONLY the user prompt for token counting
        if "<|im_start|>assistant" in full_prompt:
            user_prompt_only = full_prompt.split("<|im_start|>assistant")[0].strip()
        else:
            user_prompt_only = full_prompt
        
        if not input_case or input_case not in images_data:
            excluded_samples.append({
                "input_case": input_case,
                "reason": "No images found",
                "total_tokens": 0
            })
            continue
        
        image_paths = list(images_data[input_case]["paths"].values())
        
        # EXACT token counting logic from training script (using user prompt only)
        input_ids = processor.tokenizer(user_prompt_only, add_special_tokens=True).input_ids
        num_text_tokens = len(input_ids)
        num_images = len(image_paths)
        num_image_tokens = num_images * TOKENS_PER_IMAGE
        total_tokens = num_text_tokens + num_image_tokens
        
        # Same threshold as training
        if total_tokens > max_seq_length:
            print(f"❌ EXCLUDED: {input_case} - {total_tokens} tokens (limit: {max_seq_length})")
            excluded_samples.append({
                "input_case": input_case,
                "reason": f"Exceeds max_seq_length ({max_seq_length})",
                "total_tokens": total_tokens
            })
        else:
            if total_tokens > max_seq_length * 0.9:
                print(f"⚠️  WARNING: {input_case} - {total_tokens} tokens (approaching limit)")
            filtered_prompts.append(p)
    
    print("\n" + "=" * 80)
    print(f"Total prompts: {len(prompts)}")
    print(f"Excluded: {len(excluded_samples)}")
    print(f"Remaining: {len(filtered_prompts)}")
    print("=" * 80 + "\n")
    
    return filtered_prompts, excluded_samples


def load_model_and_processor(model_path, base_model, device):
    """Load the fine-tuned LoRA model and processor."""
    print(f"Loading processor from: {base_model}")
    processor = AutoProcessor.from_pretrained(base_model)
    
    print(f"Loading fine-tuned LoRA model from: {model_path}")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device if device == 'auto' else None,
        trust_remote_code=True,
    )
    
    if device != 'auto':
        model = model.to(device)
    
    model.eval()
    
    return model, processor


def prepare_inputs(prompt_text, image_paths, processor):
    """
    Prepare inputs - SAME approach as training script's collator.
    """
    images = [Image.open(p).convert("RGB") for p in image_paths]
    
    # Use the SAME processor call as training
    inputs = processor(
        text=[prompt_text],  # List format, same as training
        images=[images],     # List of list of images, same as training
        padding="longest",   # Same as training
        return_tensors="pt",
    )
    
    return inputs


def generate_output(model, inputs, max_new_tokens, temperature, top_p, device, processor):
    """Generate output from the model."""
    # Move inputs to device
    inputs = {k: v.to(device if not isinstance(device, str) or device == 'cuda' else device) 
              for k, v in inputs.items()}
    
    input_length = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            pad_token_id=inputs['input_ids'][0][-1],  # Use last token as pad
            eos_token_id=processor.tokenizer.eos_token_id,  # Ensure proper stopping
            repetition_penalty=1.1,  # Prevent repetitive outputs
            no_repeat_ngram_size=3,  # Prevent repeating 3-grams
        )
    
    # Extract only generated tokens (skip input prompt)
    generated_ids = outputs[0][input_length:]
    
    # Decode using the processor's tokenizer
    generated_text = processor.tokenizer.decode(
        generated_ids, 
        skip_special_tokens=True
    )
    
    return generated_text.strip()


def run_inference(args):
    """Main inference function."""
    output_path = Path.cwd() / args.output_json
    excluded_path = Path.cwd() / f"excluded_{args.output_json}"
    
    print("=" * 80)
    print("LLaVA-OV LoRA Fine-tuned Model Inference")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Output: {output_path}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Step 1: Load test data
    test_prompts, test_images = load_test_data(
        args.test_prompts_jsonl, 
        args.test_images_json
    )
    
    # Step 2: Load processor for filtering (lightweight)
    print(f"\nLoading processor for token counting...")
    processor = AutoProcessor.from_pretrained(args.base_model)
    
    # Step 3: Filter prompts BEFORE loading model
    filtered_prompts, excluded_samples = filter_prompts(
        test_prompts, 
        test_images, 
        processor, 
        args.max_seq_length
    )
    
    # Save excluded samples
    if excluded_samples:
        with open(excluded_path, 'w') as f:
            json.dump(excluded_samples, f, indent=2)
        print(f"📁 Saved {len(excluded_samples)} excluded samples to: {excluded_path}\n")
    
    if not filtered_prompts:
        print("❌ No samples remaining after filtering!")
        return
    
    # Step 4: NOW load the heavy model
    print("Loading fine-tuned model...\n")
    model, processor = load_model_and_processor(
        args.model_path, 
        args.base_model, 
        args.device
    )
    
    # Step 5: Check for existing results (resume capability)
    if output_path.exists():
        print(f"📂 Found existing results, loading...")
        with open(output_path, 'r') as f:
            results = json.load(f)
        processed = set(results.keys())
        print(f"✓ Loaded {len(results)} existing results\n")
    else:
        results = {}
        processed = set()
    
    # Filter out already processed
    remaining = [p for p in filtered_prompts 
                 if p.get("input_case") not in processed]
    
    if not remaining:
        print("✅ All samples already processed!")
        return
    
    print(f"🚀 Processing {len(remaining)} samples...\n")
    
    # Step 6: Run inference
    for prompt_data in tqdm(remaining, desc="Generating", unit="sample"):
        input_case = prompt_data.get("input_case")
        full_prompt = prompt_data["text"]
        
        # Extract ONLY the user prompt (everything before <|im_start|>assistant)
        if "<|im_start|>assistant" in full_prompt:
            user_prompt = full_prompt.split("<|im_start|>assistant")[0].strip()
        else:
            # If no assistant tag found, skip this sample with error
            tqdm.write(f"⚠️  {input_case}: No '<|im_start|>assistant' tag found in prompt, skipping")
            results[input_case] = {
                "input_case": input_case,
                "user_prompt": full_prompt[:200] + "...",
                "generated_output": "ERROR: Missing <|im_start|>assistant tag in prompt",
                "error": "Missing assistant tag"
            }
            continue
        
        # CRITICAL: Add the assistant tag to trigger generation
        user_prompt = user_prompt + "\n<|im_start|>assistant\n"
        
        if input_case not in test_images:
            tqdm.write(f"⚠️  {input_case}: No images found, skipping")
            continue
        
        image_paths = list(test_images[input_case]["paths"].values())
        tqdm.write(f"📸 {input_case}: {len(image_paths)} images")
        
        try:
            # Prepare inputs with ONLY user prompt (same as training)
            inputs = prepare_inputs(user_prompt, image_paths, processor)
            
            # Generate
            generated_text = generate_output(
                model, inputs, 
                args.max_new_tokens, 
                args.temperature, 
                args.top_p, 
                args.device,
                processor
            )
            
            # Store result
            results[input_case] = {
                "input_case": input_case,
                "user_prompt": user_prompt,
                "generated_output": generated_text,
                "num_images_used": len(image_paths)
            }
            
            tqdm.write(f"✅ {input_case}: {len(generated_text)} chars")
            
            # Save after each sample (crash recovery)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        except Exception as e:
            tqdm.write(f"❌ {input_case}: {type(e).__name__}: {str(e)}")
            results[input_case] = {
                "input_case": input_case,
                "user_prompt": user_prompt if 'user_prompt' in locals() else full_prompt,
                "generated_output": f"ERROR: {str(e)}",
                "error": str(e)
            }
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✅ Complete! Results: {output_path}")
    print(f"Processed: {len(results)}/{len(filtered_prompts)}")
    print("=" * 80)


if __name__ == "__main__":
    run_inference(args)
