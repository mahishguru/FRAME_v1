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
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

FRACTION = 0.5
RANDOM_SEED = 42

# ---------- CLI Arguments ----------
parser = argparse.ArgumentParser(description='PEFT fine-tuning for Pixtral (50% data)')
parser.add_argument('--model_name', type=str, default="mistral-community/pixtral-12b")
parser.add_argument('--prompts_jsonl', type=str, default="../train_prompts_set2.jsonl")
parser.add_argument('--images_json', type=str, default="../train_images_set2.json")
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--max_tokens', type=int, default=2048)
parser.add_argument('--cur_dir_path', type=str, default=os.getcwd())
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--accum', type=int, default=8)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--num_train_epochs', type=int, default=3)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--permitted_max_tokens', type=int, default=28000)
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed by DeepSpeed launcher')
args = parser.parse_args()

if args.model_dir is None:
    args.model_dir = f"{args.model_name.split('/')[-1]}_p50"

# ---------- Load Model & Processor ----------
print("Loading Pixtral model...")
model = AutoModelForVision2Seq.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(args.model_name)
processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

# ---------- LoRA Config ----------
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
script_dir = os.path.dirname(os.path.abspath(__file__))
prompts_path = f'{script_dir}/{args.prompts_jsonl}'
images_path = f'{script_dir}/{args.images_json}'

prompts = [json.loads(l) for l in open(prompts_path)]
images_data = json.load(open(images_path))

log_file_path = "prompt_token_counts.txt"
skipped_keys = set()
if os.path.exists(log_file_path):
    print(f"Found existing {log_file_path}, using it to filter prompts (no token counting)...")
    with open(log_file_path, "r") as f:
        for line in f:
            if line.startswith("SKIPPED: Prompt key '"):
                key = line.split("'")[1]
                skipped_keys.add(key)
    print(f"Found {len(skipped_keys)} skipped prompts in existing log. Building dataset without token counting...")
    dataset = []
    skipped = 0
    for i, p in tqdm(enumerate(prompts), total=len(prompts), desc="Building dataset"):
        text = p["text"]
        key = p.get("input_case")
        if not key or key not in images_data:
            skipped += 1
            continue
        if key in skipped_keys:
            skipped += 1
            continue
        image_paths = [pp for pp in images_data[key]["paths"].values() if os.path.exists(pp)]
        if len(image_paths) == 0:
            skipped += 1
            continue
        dataset.append({"prompt": text, "image_path": image_paths, "key": key})
    print(f"\n--------------------------------------------------")
    print(f"Dataset built from existing log.")
    print(f"Total prompts originally: {len(prompts)}")
    print(f"Prompts skipped: {skipped}")
    print(f"Prompts remaining for training: {len(dataset)}")
    print(f"--------------------------------------------------\n")
else:
    print("No existing log file found. Performing full token counting...")
    with open(log_file_path, "w") as log_file:
        log_file.write("Processing prompts and calculating total (text + image) token count...\n\n")
        dataset = []
        skipped = 0
        for i, p in tqdm(enumerate(prompts), total=len(prompts), desc="Token counting"):
            text = p["text"]
            key = p.get("input_case")
            if not key or key not in images_data:
                skipped += 1
                print(f"SKIPPED: Prompt at index {i} missing 'input_case' or not found in images data.")
                log_file.write(f"SKIPPED: Prompt at index {i} missing 'input_case' or not found in images data.\n")
                continue
            image_paths = [pp for pp in images_data[key]["paths"].values() if os.path.exists(pp)]
            if len(image_paths) == 0:
                skipped += 1
                print(f"SKIPPED: Prompt key '{key}' - no existing image files found.")
                log_file.write(f"SKIPPED: Prompt key '{key}' - no existing image files found.\n")
                continue
            imgs = [Image.open(pth).convert("RGB").resize((400, 400), Image.BILINEAR) for pth in image_paths]
            encoding = processor(text=text, images=imgs, padding="longest", return_tensors="pt")
            total_tokens = encoding.input_ids.shape[1]
            if total_tokens > args.permitted_max_tokens:
                skipped += 1
                print(f"SKIPPED: Prompt key '{key}' - total tokens {total_tokens} exceed permitted max {args.permitted_max_tokens}.")
                log_file.write(f"SKIPPED: Prompt key '{key}' due to token count: {total_tokens}\n")
                continue
            dataset.append({"prompt": text, "image_path": image_paths, "key": key})
        summary_lines = [
            "\n--------------------------------------------------",
            "Prompt processing complete.",
            f"Total prompts originally: {len(prompts)}",
            f"Prompts skipped: {skipped}",
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
class PixtralCollator:
    def __init__(self, processor, target_size=(400, 400)):
        self.processor = processor
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids("[IMG]")
        self.target_size = target_size
        self.step_counter = 0

    def __call__(self, batch):
        self.step_counter += 1
        texts = [ex['prompt'] for ex in batch]
        keys = [ex.get('key', 'unknown') for ex in batch]
        imgs_batch = []
        for ex in batch:
            imgs = []
            for p in ex['image_path']:
                try:
                    imgs.append(Image.open(p).convert("RGB").resize(self.target_size, Image.BILINEAR))
                except:
                    imgs.append(Image.new("RGB", self.target_size, (0, 0, 0)))
            imgs_batch.append(imgs)
        enc = self.processor(text=texts, images=imgs_batch, padding="longest", return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        if self.image_token_id is not None:
            labels[labels == self.image_token_id] = -100
        inst_token_id = self.processor.tokenizer.convert_tokens_to_ids("[/INST]")
        for i in range(len(texts)):
            try:
                input_ids_list = enc["input_ids"][i].tolist()
                if inst_token_id in input_ids_list:
                    inst_position = input_ids_list.index(inst_token_id)
                    labels[i, :inst_position + 1] = -100
                else:
                    print(f"Warning: [/INST] token not found in sample {i} with key {keys[i]}, masking entire sequence")
                    labels[i, :] = -100
            except Exception as e:
                print(f"Warning: Could not mask prompt for sample {i}: {e}")
                labels[i, :] = -100
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        total_input_tokens = attention_mask.sum().item()
        trainable_tokens = (labels != -100).sum().item()
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            key_str = keys[0] if len(keys) > 0 else 'unknown'
            print(f"[Batch {self.step_counter}] Key: {key_str} | Total tokens: {total_input_tokens} | Trainable: {trainable_tokens}")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": enc["pixel_values"],
            "labels": labels
        }

collator = PixtralCollator(processor)

# ---------- Training Arguments ----------
deep_speed_cfg = os.path.join(args.cur_dir_path, "deepspeed_pixtral.json")
with open(deep_speed_cfg, "w") as f:
    json.dump({
      "train_micro_batch_size_per_gpu": "auto",
      "gradient_accumulation_steps": "auto",
      "optimizer": {"type": "AdamW", "params": {"lr": "auto"}},
      "bf16": {"enabled": True},
      "zero_optimization": {
          "stage": 3,
          "offload_optimizer": {"device": "cpu", "pin_memory": True},
          "offload_param": {"device": "cpu", "pin_memory": True},
          "overlap_comm": True,
          "contiguous_gradients": True,
          "sub_group_size": 1e9,
          "reduce_bucket_size": 1e6,
          "stage3_prefetch_bucket_size": 1e6,
          "stage3_param_persistence_threshold": 1e5,
          "stage3_max_live_parameters": 1e8,
          "stage3_max_reuse_distance": 1e8,
          "stage3_gather_16bit_weights_on_model_save": True
      },
      "gradient_clipping": 1.0,
      "steps_per_print": 1
    }, f, indent=2)

training_args = TrainingArguments(
    output_dir=args.model_dir,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.accum,
    learning_rate=args.lr,
    num_train_epochs=args.num_train_epochs,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=None,
    logging_steps=1,
    logging_first_step=True,
    bf16=True,
    report_to="tensorboard",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    deepspeed=deep_speed_cfg,
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    disable_tqdm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset,
    data_collator=collator,
)

trainer.train()
trainer.save_model(f"{args.model_dir}/checkpoint-final")
