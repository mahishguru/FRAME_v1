import os
import json
import argparse
from pathlib import Path

# Set environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq


# ---------- CLI Arguments ----------
parser = argparse.ArgumentParser(description='Inference with base Pixtral model (no fine-tuning)')
parser.add_argument('--base_model', type=str, default="mistral-community/pixtral-12b",
                    help='Base model name to use for inference')
parser.add_argument('--test_prompts_jsonl', type=str, required=True,
                    help='Path to test prompts JSONL file')
parser.add_argument('--test_images_json', type=str, required=True,
                    help='Path to test images JSON file')
parser.add_argument('--output_json', type=str, required=True,
                    help='Output JSON filename')
parser.add_argument('--max_new_tokens', type=int, default=2048,
                    help='Maximum number of tokens to generate')
parser.add_argument('--max_seq_length', type=int, default=28000,
                    help='Maximum sequence length for filtering (same as training permitted_max_tokens)')
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
    filtered_prompts = []
    excluded_samples = []
    
    print("\n" + "=" * 80)
    print("Filtering samples based on token count...")
    print("=" * 80)
    
    for p in prompts:
        full_prompt = p["text"]
        input_case = p.get("input_case")
        
        # Extract ONLY the user prompt for token counting (everything before [/INST])
        if "[/INST]" in full_prompt:
            user_prompt_only = full_prompt.split("[/INST]")[0] + "[/INST]"
        else:
            # If no [/INST] found, use full prompt for counting
            user_prompt_only = full_prompt
        
        if not input_case or input_case not in images_data:
            excluded_samples.append({
                "input_case": input_case,
                "reason": "No images found",
                "total_tokens": 0
            })
            continue
        
        image_paths_ = list(images_data[input_case]["paths"].values())
        image_paths = [pp for pp in image_paths_ if os.path.exists(pp)]
        
        if len(image_paths) == 0:
            excluded_samples.append({
                "input_case": input_case,
                "reason": "No existing image files found",
                "total_tokens": 0
            })
            continue
        
        # EXACT token counting logic from training script (using user prompt only)
        # Resize images to 400x400 as in training
        imgs = [Image.open(p).convert("RGB").resize((400, 400), Image.BILINEAR) for p in image_paths]
        out_batch_encodings = processor(
            text=user_prompt_only,
            images=imgs,
            padding="longest",
            return_tensors="pt",
        )
        total_tokens = out_batch_encodings.input_ids.shape[1]
        
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


def load_model_and_processor(base_model, device):
    """Load the base model (no fine-tuning) and processor."""
    print(f"Loading processor from: {base_model}")
    processor = AutoProcessor.from_pretrained(base_model)
    
    # Set pad token (same as training)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
    print(f"Loading base model from: {base_model}")
    model = AutoModelForVision2Seq.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device if device == 'auto' else None,
    )
    
    if device != 'auto':
        model = model.to(device)
    
    model.eval()
    
    return model, processor


def prepare_inputs(prompt_text, image_paths, processor):
    """
    Prepare inputs - SAME approach as training script's collator.
    Resize images to 400x400 as in the collator.
    """
    # Resize to 400x400 as in training collator
    images = [Image.open(p).convert("RGB").resize((400, 400), Image.BILINEAR) for p in image_paths]
    
    # Use the SAME processor call as training
    inputs = processor(
        text=prompt_text,
        images=images,
        padding="longest",
        return_tensors="pt",
    )
    
    return inputs


def generate_output(model, inputs, max_new_tokens, temperature, top_p, device, processor):
    """Generate output from the model."""
    # Move inputs to device and convert to bfloat16 where needed
    if device == 'auto':
        # Model already on device with device_map='auto'
        inputs = {k: v.to(model.device).to(torch.bfloat16) if v.dtype == torch.float32 and k == 'pixel_values' else v.to(model.device) 
                  for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device).to(torch.bfloat16) if v.dtype == torch.float32 and k == 'pixel_values' else v.to(device) 
                  for k, v in inputs.items()}
    
    input_length = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
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
    print("Pixtral Base Model Inference (No Fine-tuning)")
    print("=" * 80)
    print(f"Model: {args.base_model}")
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
    
    # Set pad token
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
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
    print("Loading base model...\n")
    model, processor = load_model_and_processor(
        args.base_model, 
        args.device
    )
    
    # Step 5: Check for existing results (resume capability)
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
        
        # Extract ONLY the user prompt (everything before [/INST])
        if "[/INST]" in full_prompt:
            user_prompt = full_prompt.split("[/INST]")[0] + "[/INST]"
        else:
            # If no [/INST] tag found, skip this sample with error
            tqdm.write(f"⚠️  {input_case}: No '[/INST]' tag found in prompt, skipping")
            results[input_case] = {
                "input_case": input_case,
                "user_prompt": full_prompt[:200] + "...",
                "generated_output": "ERROR: Missing [/INST] tag in prompt",
                "error": "Missing [/INST] tag"
            }
            continue
        
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
