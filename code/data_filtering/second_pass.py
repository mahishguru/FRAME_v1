import os
import gc
import torch
from PIL import Image, UnidentifiedImageError
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Optionally set before import torch:
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"

def is_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False

def main():
    # Load model in bf16 + FlashAttention
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    base_dir = "First_Filtered_Data"
    out_file = "second_pass_names.txt"
    open(out_file, "w").close()

    for sub in os.listdir(base_dir):
        inf_dir = os.path.join(base_dir, sub, "infographic")
        if not os.path.isdir(inf_dir):
            continue

        found_true = False
        for fname in os.listdir(inf_dir):
            path = os.path.join(inf_dir, fname)
            if not is_image(path):
                continue

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": path},
                    {"type": "text", "text": "Does this image contain a finite element contour plot showing a variable as color gradients over a part’s surface? Reply only 'True' or 'False'."}
                ]
            }]

            try:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                img_in, vid_in = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=img_in,
                    videos=vid_in,
                    padding=True,
                    return_tensors="pt",
                ).to(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=128)
                out_ids = [g[len(i):] for i, g in zip(inputs.input_ids, gen)]
                resp = processor.batch_decode(
                    out_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].strip()

                if "True" in resp:
                    with open(out_file, "a") as f:
                        f.write(sub + "\n")
                    print(f"[+] {sub}: True → logged")
                    found_true = True
                    break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"[!] OOM on {path}, clearing cache and retrying on CPU…")
                    torch.cuda.empty_cache()
                    gc.collect()
                    # fall back to CPU
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    with torch.no_grad():
                        gen = model.generate(**inputs, max_new_tokens=128)
                    out_ids = [g[len(i):] for i, g in zip(inputs["input_ids"], gen)]
                    resp = processor.batch_decode(
                        out_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0].strip()
                    if "True" in resp:
                        with open(out_file, "a") as f:
                            f.write(sub + "\n")
                        print(f"[+] {sub}: True → logged (CPU fallback)")
                        found_true = True
                        break
                    else:
                        print(f"[!] CPU fallback did not yield True for {path}")
                else:
                    print(f"[!] Error on {path}: {e}")
            except Exception as e:
                print(f"[!] Error on {path}: {e}")
            finally:
                # always free GPU memory between iterations
                try:
                    del inputs, gen, out_ids
                except Exception:
                    pass
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
        if not found_true:
            print(f"[-] {sub}: No qualifying image found.")

if __name__ == "__main__":
    main()
