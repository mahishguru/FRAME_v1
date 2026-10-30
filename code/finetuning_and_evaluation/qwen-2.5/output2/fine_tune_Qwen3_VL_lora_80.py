import os
import json
import argparse
import random
from datetime import datetime

# Set environment variables before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image
from datasets import Dataset as HFDataset
from tqdm import tqdm
from transformers import AutoProcessor
from transformers import (
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# Import Qwen3VL model class directly
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

# Subset fraction
FRACTION = 0.8
RANDOM_SEED = 42

# ---------- CLI Arguments ----------
parser = argparse.ArgumentParser(description='PEFT fine-tuning for Qwen3-VL-8B-Instruct (80% data)')
parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-VL-8B-Instruct")
parser.add_argument('--train_prompts', type=str, default="./train_prompts_set2.jsonl")
parser.add_argument('--train_images', type=str, default="./train_images_set2.json")
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--max_tokens', type=int, default=2048)
parser.add_argument('--cur_dir_path', type=str, default=os.getcwd())
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--num_train_epochs', type=int, default=3)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--permitted_max_tokens', type=int, default=32000)
parser.add_argument('--deepspeed_config', type=str, default='deepspeed_config.json')
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed by DeepSpeed launcher')
parser.add_argument('--use_flash_attention', action='store_true', help='Enable flash attention 2')
parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
args = parser.parse_args()

if args.model_dir is None:
    args.model_dir = f"{args.model_name.split('/')[-1]}_p80"

# ---------- Load Model & Processor ----------
print("Loading Qwen3-VL-8B-Instruct model...")
print(f"Flash Attention: {'Enabled' if args.use_flash_attention else 'Disabled'}")

if args.use_flash_attention:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,  # Let DeepSpeed handle device placement
        trust_remote_code=True,
    )
else:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,  # Let DeepSpeed handle device placement
        trust_remote_code=True,
    )

processor = AutoProcessor.from_pretrained(args.model_name)

# ---------- LoRA Config ----------
# Qwen3-VL uses Qwen architecture, target attention and MLP projections
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()

# ---------- Load Dataset ----------
prompts = []
with open(args.train_prompts, "r") as f:
    for line in f:
        prompts.append(json.loads(line))

with open(args.train_images, "r") as f:
    images_data = json.load(f)

log_file_path = "prompt_token_counts.txt"

# Check if log file exists and extract already skipped prompts
skipped_keys = set()
if os.path.exists(log_file_path):
    print(f"Found existing {log_file_path}, using it to filter prompts (no token counting)...")
    with open(log_file_path, "r") as f:
        for line in f:
            if line.startswith("SKIPPED: Prompt key '"):
                key = line.split("'")[1]
                skipped_keys.add(key)
    print(f"Found {len(skipped_keys)} skipped prompts in existing log. Building dataset without token counting...")
    
    # Build dataset directly, excluding skipped keys
    dataset = []
    skipped_prompts = 0
    for i, p in tqdm(enumerate(prompts), total=len(prompts), desc="Building dataset"):
        prompt_text = p["text"]
        input_case_key = p.get("input_case")
        
        if not input_case_key or input_case_key not in images_data:
            skipped_prompts += 1
            continue
        
        if input_case_key in skipped_keys:
            skipped_prompts += 1
            continue
        
        image_paths_ = list(images_data[input_case_key]["paths"].values())
        image_paths = [pp for pp in image_paths_ if os.path.exists(pp)]
        if len(image_paths) == 0:
            skipped_prompts += 1
            continue
        
        dataset.append({"prompt": prompt_text, "image_path": image_paths})
    
    print(f"\n--------------------------------------------------")
    print(f"Dataset built from existing log.")
    print(f"Total prompts originally: {len(prompts)}")
    print(f"Prompts skipped: {skipped_prompts}")
    print(f"Prompts remaining for training: {len(dataset)}")
    print(f"--------------------------------------------------\n")
    
else:
    # No log file exists, do full token counting
    print("No existing log file found. Performing full token counting...")
    with open(log_file_path, "w") as log_file:
        log_file.write("Processing prompts and calculating total (text + image) token count...\n\n")
        
        dataset = []
        skipped_prompts = 0
        for i, p in tqdm(enumerate(prompts), total=len(prompts), desc="Token counting"):
            prompt_text = p["text"]
            input_case_key = p.get("input_case")
            
            if not input_case_key or input_case_key not in images_data:
                skipped_prompts += 1
                print(f"SKIPPED: Prompt at index {i} missing 'input_case' or not found in images data.")
                log_file.write(f"SKIPPED: Prompt at index {i} missing 'input_case' or not found in images data.\n")
                continue

            image_paths_ = list(images_data[input_case_key]["paths"].values())
            image_paths = [pp for pp in image_paths_ if os.path.exists(pp)]
            if len(image_paths) == 0:
                skipped_prompts += 1
                print(f"SKIPPED: Prompt key '{input_case_key}' - no existing image files found.")
                log_file.write(f"SKIPPED: Prompt key '{input_case_key}' - no existing image files found.\n")
                continue

            # Prepare messages in Qwen3-VL format for token counting
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.open(pth).convert("RGB").resize((448, 448), Image.BILINEAR)} 
                    for pth in image_paths
                ] + [{"type": "text", "text": prompt_text}]
            }]
            
            try:
                text_inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt"
                )
                total_tokens = text_inputs.input_ids.shape[1]
            except Exception as e:
                print(f"SKIPPED: Error processing '{input_case_key}': {e}")
                log_file.write(f"SKIPPED: Prompt key '{input_case_key}' - error: {e}\n")
                skipped_prompts += 1
                continue
            
            if total_tokens > args.permitted_max_tokens:
                skipped_prompts += 1
                print(f"SKIPPED: Prompt key '{input_case_key}' due to token count: {total_tokens}")
                log_file.write(f"SKIPPED: Prompt key '{input_case_key}' due to token count: {total_tokens}\n")
                continue

            dataset.append({"prompt": prompt_text, "image_path": image_paths})

        summary_lines = [
            "\n--------------------------------------------------",
            "Prompt processing complete.",
            f"Total prompts originally: {len(prompts)}",
            f"Prompts skipped: {skipped_prompts}",
            f"Prompts remaining for training: {len(dataset)}",
            "--------------------------------------------------\n"
        ]
        for line in summary_lines:
            print(line)
            log_file.write(line + "\n")

# Apply subset fraction
random.seed(RANDOM_SEED)
random.shuffle(dataset)
subset_size = max(1, int(len(dataset) * FRACTION))
dataset = dataset[:subset_size]
print(f"Using subset fraction {FRACTION*100:.0f}% => {subset_size} samples")

hf_dataset = HFDataset.from_list(dataset)

# ---------- Collator ----------
class Qwen3VLCollator:
    def __init__(self, processor, target_size=(448, 448)):
        self.processor = processor
        self.target_size = target_size
        self.step_counter = 0

    def __call__(self, batch):
        self.step_counter += 1
        texts = []
        all_images = []
        for ex in batch:
            texts.append(ex['prompt'])
            paths = ex['image_path']
            imgs = []
            for p in paths:
                try:
                    img = Image.open(p).convert("RGB").resize(self.target_size, Image.BILINEAR)
                    imgs.append(img)
                except Exception as e:
                    print(f"Warning: failed to open image {p}: {e}")
                    imgs.append(Image.new("RGB", self.target_size, (0, 0, 0)))
            if len(imgs) == 0:
                imgs = [Image.new("RGB", self.target_size, (0, 0, 0))]
            all_images.append(imgs)

        inputs = self.processor(
            text=texts,
            images=all_images,
            return_tensors="pt",
            padding=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids))
        pixel_values = inputs.get('pixel_values', None)
        image_grid_thw = inputs.get('image_grid_thw', None)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids("assistant")
        for i in range(len(texts)):
            try:
                input_ids_list = input_ids[i].tolist()
                if assistant_token_id in input_ids_list:
                    assistant_position = input_ids_list.index(assistant_token_id)
                    labels[i, :assistant_position + 1] = -100
                else:
                    print(f"Warning: assistant token not found in sample {i}, masking entire sequence")
                    labels[i, :] = -100
            except Exception as e:
                print(f"Warning: Could not mask prompt for sample {i}: {e}")
                labels[i, :] = -100

        total_input_tokens = attention_mask.sum().item()
        trainable_tokens = (labels != -100).sum().item()
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[Batch {self.step_counter}] Total tokens: {total_input_tokens} | Trainable: {trainable_tokens}")

        batch_out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_values is not None:
            batch_out["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            batch_out["image_grid_thw"] = image_grid_thw
        return batch_out

collator = Qwen3VLCollator(processor, target_size=(448, 448))

# ---------- Training Arguments ----------
deep_speed_path = os.path.join(args.cur_dir_path, "deepspeed_qwen3vl.json")
with open(deep_speed_path, "w") as f:
    json.dump({
      "train_batch_size": "auto",
      "train_micro_batch_size_per_gpu": "auto",
      "gradient_accumulation_steps": "auto",
      "optimizer": {
        "type": "AdamW",
        "params": { "lr": "auto", "betas": [0.9,0.999], "eps": "auto", "weight_decay": "auto" }
      },
      "bf16": {"enabled": True},
      "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "overlap_comm": True,
        "contiguous_gradients": True,
      },
      "gradient_clipping": 1.0,
    }, f, indent=2)

training_args = TrainingArguments(
    output_dir=args.model_dir,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_train_epochs,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=None,
    logging_steps=1,
    logging_first_step=True,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    bf16=True,
    report_to="tensorboard",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    deepspeed=deep_speed_path,
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    disable_tqdm=False,
)

# ---------- Trainer ----------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset,
    data_collator=collator,
)

# ---------- Train ----------
print("\n" + "="*80)
if args.resume_from_checkpoint:
    print(f"Resuming Qwen3-VL-8B-Instruct Fine-Tuning (80% data) from {args.resume_from_checkpoint}")
else:
    print("Starting Qwen3-VL-8B-Instruct Fine-Tuning (80% data)")
print("="*80 + "\n")

trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
trainer.save_model(f"{args.model_dir}/checkpoint-final")

print("\n" + "="*80)
print("Training Complete!")
print(f"Model saved to: {args.model_dir}/checkpoint-final")
print("="*80 + "\n")
