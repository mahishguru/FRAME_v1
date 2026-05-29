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

def process_image_with_llm(image_path, model, processor, prompt_text):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text}
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
            return True
        elif "False" in resp:
            return False
        else:
            print(f"[!] Unexpected LLM response for {image_path}: {resp}")
            return None # Indicate an unclear response
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"[!] OOM on {image_path}, clearing cache and retrying on CPU…")
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
                print(f"[+] {image_path}: True → logged (CPU fallback)")
                return True
            elif "False" in resp:
                print(f"[-] {image_path}: False → logged (CPU fallback)")
                return False
            else:
                print(f"[!] Unexpected LLM response for {image_path} (CPU fallback): {resp}")
                return None
        else:
            print(f"[!] Error on {image_path}: {e}")
            return None
    except Exception as e:
        print(f"[!] Error on {image_path}: {e}")
        return None
    finally:
        # always free GPU memory between iterations
        try:
            del inputs, gen, out_ids
        except Exception:
            pass
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

def main():
    # Load model in bf16 + FlashAttention
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # New logic for processing images based on names.txt
    images_to_delete_file = "delete.txt"
    open(images_to_delete_file, "w").close()  # Clear the file at the start

    with open("names.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Extract folder name (prompt key)
            try:
                # Assuming the format is 'SKIPPED: Prompt key 'folder_name' due to token count: XXXX'
                folder_name_start = line.find("'") + 1
                folder_name_end = line.rfind("'")
                folder_name = line[folder_name_start:folder_name_end]
            except IndexError:
                print(f"[-] Could not extract folder name from line: {line}. Skipping.")
                continue

            image_subdir = os.path.join("images", folder_name)
            if not os.path.isdir(image_subdir):
                print(f"[-] Image subdirectory not found: {image_subdir}. Skipping.")
                continue

            for fname in os.listdir(image_subdir):
                image_path = os.path.join(image_subdir, fname)
                if not is_image(image_path):
                    continue

                # Define the prompt for LLM classification.
                # You can change this prompt if you have a different classification in mind for these new images.
                prompt_text_for_deletion = "Does this image contain a finite element plot showing a finite element variable (from structural, dynamic, thermal, frequency/vibration, magnetic, electrical, fluid simulations) OR a CAD, Drawing, Sketch of a component? Reply only 'True' or 'False'. NOTE: It's not necessary that every colored gradient plot is a finite element plot, so be careful. Also, Keep in mind that any curve or any graph plot is not a finite element plot, so be careful."
                result = process_image_with_llm(image_path, model, processor, prompt_text_for_deletion)

                if result is False:
                    with open(images_to_delete_file, "a") as f:
                        f.write(image_path + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                    print(f"[x] Added to {images_to_delete_file}: {image_path}")
                elif result is True:
                    print(f"[\u2713] Image classified as True (not marked for deletion): {image_path}")
                else:
                    print(f"[!] LLM response unclear for {image_path}. Not marked for deletion.")

if __name__ == "__main__":
    main()
