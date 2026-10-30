import os
import json
import re
import argparse
from pathlib import Path

# Disable unnecessary warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from peft import PeftModel


# ---------- CLI Arguments ----------
parser = argparse.ArgumentParser(description='Inference with LoRA fine-tuned Qwen3-VL model')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to fine-tuned model checkpoint')
parser.add_argument('--base_model', type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                    help='Base model name')
parser.add_argument('--test_prompts_jsonl', type=str, required=True,
                    help='Path to test prompts JSONL file')
parser.add_argument('--test_images_json', type=str, required=True,
                    help='Path to test images JSON file')
parser.add_argument('--output_json', type=str, required=True,
                    help='Output JSON filename')
parser.add_argument('--max_new_tokens', type=int, default=2048,
                    help='Maximum number of tokens to generate')
parser.add_argument('--temperature', type=float, default=0.7,
                    help='Temperature for generation')
parser.add_argument('--top_p', type=float, default=0.8,
                    help='Top-p (nucleus sampling) parameter')
parser.add_argument('--top_k', type=int, default=20,
                    help='Top-k sampling parameter')
parser.add_argument('--repetition_penalty', type=float, default=1.0,
                    help='Repetition penalty')
parser.add_argument('--max_seq_length', type=int, default=32000,
                    help='Maximum sequence length for filtering')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device: "cuda", "cpu", or "cuda:0"')
parser.add_argument('--use_flash_attention', action='store_true',
                    help='Enable flash attention 2')
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


def filter_prompts(prompts, images_data, processor, max_seq_length):
    """Filter prompts by token count."""
    filtered_prompts = []
    excluded_samples = []
    
    print("\n" + "=" * 80)
    print("Filtering samples based on token count...")
    print("=" * 80)
    
    for p in prompts:
        full_prompt = p["text"]
        input_case = p.get("input_case")
        
        # Extract ONLY the user prompt for token counting
        if "<|im_start|>assistant" in full_prompt or "assistant" in full_prompt:
            # Try to split at assistant marker
            if "<|im_start|>assistant" in full_prompt:
                user_prompt_only = full_prompt.split("<|im_start|>assistant")[0].strip()
            else:
                user_prompt_only = full_prompt.split("assistant")[0].strip()
        else:
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
                "reason": "No valid image files",
                "total_tokens": 0
            })
            continue

        # Prepare messages for token counting
        try:
            images = [Image.open(pth).convert("RGB").resize((448, 448), Image.BILINEAR) for pth in image_paths]
            
            # Clean up user prompt text
            clean_text = re.sub(r'<\|vision_start\|>.*?<\|vision_end\|>', '', user_prompt_only)
            clean_text = re.sub(r'<\|image_pad\|>', '', clean_text)
            clean_text = re.sub(r'<\|im_start\|>system\n.*?<\|im_end\|>\n?', '', clean_text, flags=re.DOTALL)
            clean_text = re.sub(r'<\|im_start\|>user\n?', '', clean_text)
            clean_text = re.sub(r'<\|im_end\|>', '', clean_text)
            clean_text = clean_text.strip()
            
            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images] + 
                          [{"type": "text", "text": clean_text}]
            }]
            
            # Get text first, then process
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            text_inputs = processor(
                text=[text],
                images=[images],
                return_tensors="pt",
                padding=True
            )
            total_tokens = text_inputs.input_ids.shape[1]
        except Exception as e:
            print(f"⚠️  ERROR processing {input_case}: {e}")
            excluded_samples.append({
                "input_case": input_case,
                "reason": f"Processing error: {str(e)}",
                "total_tokens": 0
            })
            continue
        
        if total_tokens > max_seq_length:
            print(f"⚠️  EXCLUDED: {input_case} - {total_tokens} tokens (exceeds {max_seq_length})")
            excluded_samples.append({
                "input_case": input_case,
                "reason": f"Token count too high: {total_tokens}",
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


def load_model_and_processor(model_path, base_model, device, use_flash_attention):
    """Load the fine-tuned LoRA model and processor."""
    print(f"Loading processor from: {base_model}")
    processor = AutoProcessor.from_pretrained(base_model)
    
    print(f"Loading base model: {base_model}")
    print(f"Flash Attention: {'Enabled' if use_flash_attention else 'Disabled'}")
    
    # First load the base model
    if use_flash_attention:
        base_model_loaded = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device if device == 'auto' else None,
            trust_remote_code=True,
        )
    else:
        base_model_loaded = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device if device == 'auto' else None,
            trust_remote_code=True,
        )
    
    # Then load the LoRA adapter on top
    print(f"Loading LoRA adapter from: {model_path}")
    model = PeftModel.from_pretrained(base_model_loaded, model_path)
    
    if device != 'auto':
        model = model.to(device)
    
    model.eval()
    
    return model, processor


def prepare_inputs(prompt_text, image_paths, processor, device):
    """Prepare inputs in Qwen3-VL format."""
    # Resize to 448x448
    images = [Image.open(p).convert("RGB").resize((448, 448), Image.BILINEAR) for p in image_paths]
    
    # Extract user prompt (remove assistant response if present)
    if "<|im_start|>assistant" in prompt_text:
        user_text = prompt_text.split("<|im_start|>assistant")[0].strip()
    elif "assistant" in prompt_text:
        user_text = prompt_text.split("assistant")[0].strip()
    else:
        user_text = prompt_text
    
    # Remove any existing image tokens from the text - we'll add fresh ones
    # Remove vision tokens if present
    user_text = re.sub(r'<\|vision_start\|>.*?<\|vision_end\|>', '', user_text)
    user_text = re.sub(r'<\|image_pad\|>', '', user_text)
    
    # Also clean up the chat template markers for a fresh start
    user_text = re.sub(r'<\|im_start\|>system\n.*?<\|im_end\|>\n?', '', user_text, flags=re.DOTALL)
    user_text = re.sub(r'<\|im_start\|>user\n?', '', user_text)
    user_text = re.sub(r'<\|im_end\|>', '', user_text)
    user_text = user_text.strip()
    
    # Create messages in Qwen3-VL chat format
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": img} for img in images] + 
                  [{"type": "text", "text": user_text}]
    }]
    
    # Apply chat template to get formatted text
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Process text and images together
    inputs = processor(
        text=[text],
        images=[images],
        return_tensors="pt",
        padding=True
    )
    
    return inputs


def generate_output(model, inputs, max_new_tokens, temperature, top_p, top_k, repetition_penalty, device, processor):
    """Generate output from the model using Qwen3-VL recommended parameters."""
    # Move inputs to device
    inputs = {k: v.to(device if not isinstance(device, str) or device == 'cuda' else device) 
              for k, v in inputs.items()}
    
    input_length = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True if temperature > 0 else False,
            repetition_penalty=repetition_penalty,
        )
    
    # Extract only generated tokens (skip input prompt)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
    ]
    
    # Decode
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0].strip()


def main():
    print("\n" + "=" * 80)
    print("Qwen3-VL-8B-Instruct Fine-Tuned Model Inference")
    print("=" * 80 + "\n")
    
    # Load test data
    prompts, images_data = load_test_data(args.test_prompts_jsonl, args.test_images_json)
    
    # Load model and processor
    model, processor = load_model_and_processor(
        args.model_path, 
        args.base_model, 
        args.device,
        args.use_flash_attention
    )
    
    # Filter prompts
    filtered_prompts, excluded_samples = filter_prompts(
        prompts, 
        images_data, 
        processor, 
        args.max_seq_length
    )
    
    # Run inference
    print("\n" + "=" * 80)
    print("Running inference on filtered prompts...")
    print("=" * 80 + "\n")
    
    results = []
    
    # Check if output file exists (for resume functionality)
    completed_cases = set()
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json, 'r') as f:
                existing_data = json.load(f)
                if "inference_results" in existing_data:
                    results = existing_data["inference_results"]
                    completed_cases = {r["input_case"] for r in results}
                    print(f"📂 Resuming from existing file: {len(completed_cases)} already completed")
        except Exception as e:
            print(f"⚠️  Could not load existing file, starting fresh: {e}")
    
    for i, prompt_data in enumerate(tqdm(filtered_prompts, desc="Generating")):
        input_case = prompt_data.get("input_case")
        prompt_text = prompt_data["text"]
        
        # Skip if already completed
        if input_case in completed_cases:
            continue
        
        if not input_case or input_case not in images_data:
            continue
        
        image_paths_ = list(images_data[input_case]["paths"].values())
        image_paths = [pp for pp in image_paths_ if os.path.exists(pp)]
        
        if len(image_paths) == 0:
            continue
        
        try:
            # Prepare inputs
            inputs = prepare_inputs(prompt_text, image_paths, processor, args.device)
            
            # Generate
            output = generate_output(
                model, inputs,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                args.top_k,
                args.repetition_penalty,
                args.device,
                processor
            )
            
            results.append({
                "input_case": input_case,
                "prompt": prompt_text,
                "generated_output": output,
                "num_images": len(image_paths)
            })
            
            # Save after each generation (iterative saving)
            output_data = {
                "inference_results": results,
                "excluded_samples": excluded_samples,
                "stats": {
                    "total_prompts": len(prompts),
                    "excluded": len(excluded_samples),
                    "successfully_generated": len(results),
                    "remaining": len(filtered_prompts) - len(results) - len(completed_cases)
                }
            }
            with open(args.output_json, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            # Print sample output
            if i < 3:
                print(f"\n{'='*60}")
                print(f"Sample {i+1}: {input_case}")
                print(f"{'='*60}")
                print(f"Output: {output[:200]}...")
                print(f"{'='*60}\n")
                
        except Exception as e:
            print(f"❌ Error processing {input_case}: {e}")
            continue
    
    # Final save
    output_data = {
        "inference_results": results,
        "excluded_samples": excluded_samples,
        "stats": {
            "total_prompts": len(prompts),
            "excluded": len(excluded_samples),
            "successfully_generated": len(results)
        }
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Inference Complete!")
    print(f"Results saved to: {args.output_json}")
    print(f"Total generated: {len(results)}")
    print(f"Excluded: {len(excluded_samples)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
