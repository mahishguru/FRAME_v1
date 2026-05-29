import os
import json
import base64
import mimetypes
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

from config.prompts import (
    INPUT_SYSTEM_PROMPT, OUTPUT_SYSTEM_PROMPT,
    INPUT_KEY_PROMPTS, OUTPUT_KEY_PROMPTS
)
from utils.file_utils import (
    get_datapoint_folders,
    read_fulltext,
    get_image_caption_pairs,
    save_json
)

# Load environment variables
load_dotenv()

# Configuration
DATAPOINT_ROOT = "../01_DataFiltering/filtered_fourth_pass"
OUTPUT_DIR = "output"
FAILED_LOG = os.path.join(OUTPUT_DIR, "failed.txt")
OPENROUTER_MODEL = "qwen/qwen3-vl-235b-a22b-thinking"
MAX_TOKENS = 4096


def load_failed_ids() -> set:
    """Load the list of unique failed datapoint IDs from failed.txt (first column)"""
    if not os.path.exists(FAILED_LOG):
        return set()
    ids = set()
    with open(FAILED_LOG, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if parts and parts[0]:
                ids.add(parts[0])
    return ids


def load_existing_results(filename: str) -> Dict[str, Dict[str, str]]:
    """Load existing JSON results if they exist"""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def encode_image_to_base64(image_path: str) -> Tuple[str, str]:
    """Encode an image file to base64 and return (mime_type, base64_data)"""
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    return mime_type, image_data


def process_keys_openrouter(
    fulltext: str,
    image_caption_pairs: List[Tuple[str, str]],
    key_prompts: Dict[str, str],
    system_prompt: str,
    client: OpenAI,
    datapoint_id: str,
) -> Dict[str, str]:
    """
    Process each key using OpenRouter API with multimodal attachments.
    Builds conversation history incrementally like the original process_keys.
    """
    result = {}
    messages = []
    key_list = list(key_prompts.items())
    
    # Add system prompt
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt.strip()
        })
    
    # Build initial user message with fulltext
    initial_text = "Fulltext of the research paper: " + fulltext.strip()
    
    # Add image captions to text
    for img_path, caption in image_caption_pairs:
        initial_text += f"\n\nImage caption: {caption}"
    
    # Add first key prompt
    first_key, first_prompt = key_list[0]
    initial_text += f"\n\n{first_prompt}"
    
    messages.append({
        "role": "user",
        "content": initial_text
    })
    
    # Prepare attachments (all images as base64)
    attachments = []
    for img_path, _ in image_caption_pairs:
        mime_type, base64_data = encode_image_to_base64(img_path)
        attachments.append({
            "type": mime_type,
            "data": base64_data
        })
    
    # Get response for first key
    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.95,
            extra_body={"attachments": attachments} if attachments else {}
        )
        first_response = response.choices[0].message.content or ""
        result[first_key] = first_response
        messages.append({
            "role": "assistant",
            "content": first_response
        })
        print(f"  ✓ {first_key}")
    except Exception as exc:
        print(f"  ✗ {first_key}: {exc}")
        result[first_key] = ""
    
    # Process remaining keys (no attachments needed after first message)
    for key, prompt in key_list[1:]:
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=0.7,
                top_p=0.95,
            )
            key_response = response.choices[0].message.content or ""
            result[key] = key_response
            messages.append({
                "role": "assistant",
                "content": key_response
            })
            print(f"  ✓ {key}")
        except Exception as exc:
            print(f"  ✗ {key}: {exc}")
            result[key] = ""
    
    return result


def process_datapoint_openrouter(
    datapoint_dir: str,
    datapoint_id: str,
    client: OpenAI
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Process a single datapoint using OpenRouter"""
    print(f"Processing: {datapoint_id}")
    
    fulltext = read_fulltext(datapoint_dir)
    image_caption_pairs = get_image_caption_pairs(datapoint_dir)
    
    # Process input keys
    print("  Input keys:")
    input_keys = process_keys_openrouter(
        fulltext=fulltext,
        image_caption_pairs=image_caption_pairs,
        key_prompts=INPUT_KEY_PROMPTS,
        system_prompt=INPUT_SYSTEM_PROMPT,
        client=client,
        datapoint_id=datapoint_id,
    )
    
    # Process output keys
    print("  Output keys:")
    output_keys = process_keys_openrouter(
        fulltext=fulltext,
        image_caption_pairs=image_caption_pairs,
        key_prompts=OUTPUT_KEY_PROMPTS,
        system_prompt=OUTPUT_SYSTEM_PROMPT,
        client=client,
        datapoint_id=datapoint_id,
    )
    
    return input_keys, output_keys


def main():
    """Main entry point for OpenRouter fallback processing"""
    # Initialize OpenRouter client
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    # Load failed IDs
    failed_ids = load_failed_ids()
    if not failed_ids:
        print("No failed datapoints found in failed.txt")
        return
    
    print(f"Found {len(failed_ids)} failed datapoints to retry with OpenRouter")
    
    # Load existing results
    input_json = load_existing_results("input.json")
    output_json = load_existing_results("output.json")
    
    # Get all datapoint folders
    all_folders = get_datapoint_folders(DATAPOINT_ROOT)
    
    # Filter to only failed datapoints
    failed_folders = [
        folder for folder in all_folders
        if os.path.basename(folder) in failed_ids
    ]
    
    print(f"Processing {len(failed_folders)} datapoints with OpenRouter fallback\n")
    
    # Process each failed datapoint
    for datapoint_dir in tqdm(failed_folders, desc="OpenRouter fallback"):
        dp_name = os.path.basename(datapoint_dir)
        
        try:
            input_obj, output_obj = process_datapoint_openrouter(
                datapoint_dir, dp_name, client
            )
            
            # Store results
            input_json[dp_name] = input_obj
            output_json[dp_name] = output_obj
            
            # Save incrementally
            save_json(input_json, "input.json", OUTPUT_DIR)
            save_json(output_json, "output.json", OUTPUT_DIR)
            
        except Exception as exc:
            print(f"Failed {dp_name} even with OpenRouter: {exc}")
            continue


if __name__ == "__main__":
    main()
