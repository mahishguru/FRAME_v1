import os
import gc
import requests
import json
import base64

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:27b"

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def main():
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
            # Only process image files by extension
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
                continue

            prompt = (
                "Does this image contain a finite element contour plot showing a variable as color gradients over a part’s surface? Reply only 'True' or 'False'."
            )

            # Encode the image to base64
            try:
                img_b64 = encode_image_to_base64(path)
            except Exception as e:
                print(f"[!] Could not read image {path}: {e}")
                continue

            # Send prompt and image to Ollama
            payload = {
                "model": MODEL_NAME,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False
            }
            try:
                response = requests.post(OLLAMA_URL, json=payload)
                response.raise_for_status()
                result = response.json()
                resp = result.get("response", "").strip()

                if "True" in resp:
                    with open(out_file, "a") as f:
                        f.write(sub + "\n")
                    print(f"[+] {sub}: True → logged")
                    found_true = True
                    break
                elif "Unknown" in resp:
                    print(f"[!] {sub}: Model could not determine from image+prompt.")
            except Exception as e:
                print(f"[!] Error on {path}: {e}")
            finally:
                gc.collect()
        if not found_true:
            print(f"[-] {sub}: No qualifying image found.")

if __name__ == "__main__":
    main()
