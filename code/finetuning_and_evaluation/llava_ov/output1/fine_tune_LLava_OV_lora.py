import os
import json
import argparse
from datetime import datetime
import torch
from PIL import Image
from datasets import Dataset as HFDataset
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ---------- CLI Arguments ----------
# This parser defines the command-line arguments you can use to run the script.
parser = argparse.ArgumentParser(description='Stable PEFT fine-tuning for LLaVA-OV')
parser.add_argument('--model_name', type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-hf", help="The name of the pretrained model to use.")
parser.add_argument('--max_tokens', type=int, default=2048, help="Placeholder for max tokens argument to match run command.")
parser.add_argument('--prompts_jsonl', type=str, required=True, help="Path to the JSONL file containing training prompts.")
parser.add_argument('--images_json', type=str, required=True, help="Path to the JSON file containing image paths.")
parser.add_argument('--model_dir', type=str, default="llava_ov_finetuned", help="Directory to save the fine-tuned model.")
parser.add_argument('--cur_dir_path', type=str, default=os.getcwd(), help="Current working directory path.")
parser.add_argument('--batch_size', type=int, default=1, help="Per-device batch size for training.")
parser.add_argument('--accum', type=int, default=8, help="Number of gradient accumulation steps.")
parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate for the optimizer.")
parser.add_argument('--num_train_epochs', type=int, default=3, help="Total number of training epochs.")
parser.add_argument('--weight_decay', type=float, default=0.0001, help="Weight decay for the optimizer.")
args = parser.parse_args()


# ---------- Time Stamp ----------
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if args.model_dir is None:
    args.model_dir = f"{args.model_name.split('/')[-1]}_{now}"


# ---------- Load Model & Processor ----------
print(f"Loading base model: {args.model_name}")
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16,
    device_map=None,
    # NOTE: Flash Attention 2 is disabled. This is a key step for stability,
    # as its optimized kernels can sometimes conflict with gradient checkpointing.
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(args.model_name)


# ---------- Prepare Model for LoRA Training ----------
print("Applying PEFT LoRA adapters...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(model, lora_config)

# This is a required step for ensuring stability with PEFT and gradient checkpointing.
model.enable_input_require_grads()
model.print_trainable_parameters()


# ---------- Load and Process Dataset ----------
TOKENS_PER_IMAGE = 730 # Estimated token cost per image
prompts = []
with open(args.prompts_jsonl, "r") as f:
    for line in f:
        prompts.append(json.loads(line))

with open(args.images_json, "r") as f:
    images_data = json.load(f)

log_file_path = "prompt_token_counts.txt"
with open(log_file_path, "w") as log_file:
    print("Processing prompts and calculating total (text + image) token count...")
    log_file.write("Processing prompts and calculating total (text + image) token count...\n\n")
    dataset = []
    skipped_prompts = 0
    for i, p in enumerate(prompts):
        prompt_text = p["text"]
        input_case_key = p.get("input_case")
        if not input_case_key or input_case_key not in images_data:
            skipped_prompts += 1
            continue

        image_paths = list(images_data[input_case_key]["paths"].values())
        input_ids = processor.tokenizer(prompt_text, add_special_tokens=True).input_ids
        num_text_tokens = len(input_ids)
        num_images = len(image_paths)
        num_image_tokens = num_images * TOKENS_PER_IMAGE
        total_tokens = num_text_tokens + num_image_tokens
        
        # This threshold is a safeguard against sequences that are too long.
        if total_tokens > 28000:
            skipped_prompts += 1
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

hf_dataset = HFDataset.from_list(dataset)


# ---------- Data Collator ----------
# This class prepares batches of data for the model. It is critical for ensuring
# that the inputs are deterministic, which is required for gradient checkpointing.
class MultimodalCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        texts = [ex['prompt'] for ex in batch]
        images = [[Image.open(p).convert("RGB") for p in ex['image_path']] for ex in batch]

        inputs = self.processor(
            text=texts,
            images=images,
            padding="longest",
            return_tensors="pt",
        )
        
        # Explicitly creating the attention mask is a best practice for preventing
        # non-deterministic behavior in the model's forward pass.
        attention_mask = (inputs.input_ids != self.processor.tokenizer.pad_token_id).long()
        inputs['attention_mask'] = attention_mask
        
        # Move floating-point tensors to the correct training dtype
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(torch.bfloat16)

        # Create labels for language model training - ONLY for assistant responses
        labels = inputs["input_ids"].clone()
        
        # === PROPER LABEL MASKING FOR INSTRUCTION FINE-TUNING ===
        # The model should ONLY learn to predict the assistant's response,
        # not the user's prompt. We mask all tokens before "<|im_start|>assistant"
        
        assistant_start_token = "<|im_start|>assistant"
        
        for i, text in enumerate(texts):
            # Find where assistant response starts
            if assistant_start_token in text:
                # Tokenize the user prompt part only (before assistant)
                user_prompt_part = text.split(assistant_start_token)[0] + assistant_start_token
                user_prompt_ids = self.processor.tokenizer(
                    user_prompt_part, 
                    add_special_tokens=False
                ).input_ids
                
                # Mask all user prompt tokens with -100 (don't compute loss on them)
                num_user_tokens = len(user_prompt_ids)
                labels[i, :num_user_tokens] = -100
            else:
                # If no assistant tag found, mask everything (safety fallback)
                labels[i, :] = -100
        
        # Mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        # Mask out image tokens in labels so the model doesn't learn to predict them
        image_token_str = self.processor.tokenizer.additional_special_tokens[0]
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(image_token_str)
        labels[labels == image_token_id] = -100
        
        inputs["labels"] = labels
        
        return inputs

collator = MultimodalCollator(processor)


# ---------- DeepSpeed Configuration ----------
deepspeed_config_path = os.path.join(args.cur_dir_path, "deepspeed_config.json")
with open(deepspeed_config_path, "w") as f:
    json.dump({
      "train_micro_batch_size_per_gpu": args.batch_size, 
      "gradient_accumulation_steps": args.accum,  
      "optimizer": {
        "type": "AdamW",
        "params": {
          "lr": args.lr,
          "betas": [0.9, 0.999],
          "eps": 1e-8,
          "weight_decay": args.weight_decay,
        }
      },
      "bf16": { "enabled": True },
      "zero_optimization": {
        "stage": 3,
        "offload_optimizer": { "device": "cpu", "pin_memory": True },
        "offload_param": { "device": "cpu", "pin_memory": True },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
      },
      "gradient_clipping": 1.0,
      "steps_per_print": 10,
    }, f, indent=2)


# ---------- Training Arguments ----------
training_args = TrainingArguments(
    output_dir=args.model_dir,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.accum,
    learning_rate=args.lr,
    weight_decay=args.weight_decay, 
    num_train_epochs=args.num_train_epochs,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=None,
    logging_steps=10,
    bf16=True,
    report_to="tensorboard",
    remove_unused_columns=False,
    deepspeed=deepspeed_config_path,
    
    # --- THE DEFINITIVE SOLUTION ---
    # We enable gradient checkpointing to save memory, which is essential for long sequences.
    gradient_checkpointing=True,
    # We force the use of the older, re-entrant implementation. The modern version
    # (`use_reentrant=False`) was causing a persistent metadata mismatch error,
    # indicating a deep incompatibility with this specific model architecture.
    # This older version is more robust and should solve the CheckpointError.
    gradient_checkpointing_kwargs={"use_reentrant": True},
)


# ---------- Trainer Initialization ----------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset,
    data_collator=collator,
)


# ---------- Train the Model ----------
print("Starting training with the stable reentrant checkpointing implementation...")
trainer.train()
print("Training finished successfully.")
trainer.save_model(f"{args.model_dir}/checkpoint-final")
print("Final model saved.")