import argparse
import json
import os
import mimetypes
from typing import List, Dict, Any, Tuple, Optional, Set
from tqdm import tqdm

from config.prompts import (
    INPUT_SYSTEM_PROMPT, OUTPUT_SYSTEM_PROMPT,
    INPUT_KEY_PROMPTS, OUTPUT_KEY_PROMPTS
)
from services.qwen_service import QwenService
from utils.file_utils import (
    get_datapoint_folders,
    read_fulltext,
    get_image_caption_pairs,
    save_json
)

# Set your datapoint root directories and output directory
DATAPOINT_ROOTS = [
    "../01_DataFiltering/Third_Filtered_Data",
    "../01_DataFiltering/filtered_fourth_pass",
]
OUTPUT_DIR = "output"
FAILED_LOG = os.path.join(OUTPUT_DIR, "failed.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process datapoints with vLLM service")
    parser.add_argument(
        "--datapoint-list",
        type=str,
        default=None,
        help="Path to a newline-separated list of datapoint identifiers to process. If omitted, all datapoints are considered",
    )
    return parser.parse_args()


def load_existing_results(filename: str) -> Dict[str, Dict[str, str]]:
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def has_all_keys(entry: Dict[str, str], required_keys) -> bool:
    if not entry:
        return False
    for key in required_keys:
        value = entry.get(key)
        if not value or not str(value).strip():
            return False
    return True


def load_failed_ids() -> set:
    if not os.path.exists(FAILED_LOG):
        return set()
    ids = set()
    with open(FAILED_LOG, "r", encoding="utf-8") as f:
        for line in f:
            identifier = line.strip().split("\t", 1)[0]
            if identifier:
                ids.add(identifier)
    return ids


def log_failure(identifier: str, detail: str = ""):
    """Append a datapoint identifier to failed.txt for later retries."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        if detail:
            f.write(f"{identifier}	{detail}\n")
        else:
            f.write(f"{identifier}\n")

def collect_datapoint_folders() -> List[str]:
    folders: List[str] = []
    for root in DATAPOINT_ROOTS:
        if not os.path.isdir(root):
            print(f"Warning: datapoint root not found: {root}")
            continue
        folders.extend(get_datapoint_folders(root))
    return folders


def get_mime_type(file_path: str) -> str:
    """Determine the MIME type of a file based on its extension."""
    mime_type, _ = mimetypes.guess_type(file_path)
    # Default to image/jpeg if can't determine
    return mime_type or "image/jpeg"

def process_keys(
    fulltext: str,
    image_caption_pairs: List[Tuple[str, str]],
    key_prompts: Dict[str, str],
    system_prompt: str,
    llm_service: QwenService,
    datapoint_id: str,
) -> Dict[str, str]:
    """
    For each key, append the new user prompt and assistant response to the growing chat history.
    Start with system and user (context + key1), then alternate user/assistant for each key.
    Only add the initial context+key1 once at the start.
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
    # Build the initial content blocks with fulltext and images
    initial_content = []
    initial_content.append({
        "type": "text",
        "text": "Fulltext of the research paper: " + fulltext.strip()
    })
    for img_path, caption in image_caption_pairs:
        initial_content.append({
            "type": "text",
            "text": f"Image caption: {caption}"
        })
        abs_path = os.path.abspath(img_path)
        initial_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"file://{abs_path}"
            }
        })
    # Add key1 prompt to the initial user message
    first_key, first_prompt = key_list[0]
    initial_content.append({
        "type": "text",
        "text": first_prompt
    })
    messages.append({
        "role": "user",
        "content": initial_content
    })
    # Get response for key1
    try:
        response = llm_service.process_with_history(
            messages=messages,
            log_context={"key": first_key, "datapoint": datapoint_id}
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed datapoint={datapoint_id} key={first_key}: {exc}"
        ) from exc
    if not response or not response.strip():
        detail = f"empty response key={first_key}"
        log_failure(datapoint_id, detail)
        raise RuntimeError(
            f"Failed datapoint={datapoint_id} key={first_key}: {detail}"
        )
    result[first_key] = response
    messages.append({
        "role": "assistant",
        "content": response
    })
    # For each subsequent key, append user/assistant turns
    for key, prompt in key_list[1:]:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        })
        try:
            response = llm_service.process_with_history(
                messages=messages,
                log_context={"key": key, "datapoint": datapoint_id}
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed datapoint={datapoint_id} key={key}: {exc}"
            ) from exc
        if not response or not response.strip():
            detail = f"empty response key={key}"
            log_failure(datapoint_id, detail)
            raise RuntimeError(
                f"Failed datapoint={datapoint_id} key={key}: {detail}"
            )
        result[key] = response
        messages.append({
            "role": "assistant",
            "content": response
        })
    return result


def process_datapoint(datapoint_dir: str, datapoint_id: str, llm_service: QwenService) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Process a single datapoint directory through the input and output prompts.
    
    Args:
        datapoint_dir: Path to the datapoint directory
        llm_service: The QwenService instance for LLM inference
        
    Returns:
        Tuple of (input_keys_results, output_keys_results) dictionaries
    """
    print(f"Processing: {datapoint_dir}")
    
    # Read content from the datapoint
    fulltext = read_fulltext(datapoint_dir)
    image_caption_pairs = get_image_caption_pairs(datapoint_dir)
    
    # Process input and output prompts
    input_keys = process_keys(
        fulltext=fulltext, 
        image_caption_pairs=image_caption_pairs,
        key_prompts=INPUT_KEY_PROMPTS, 
        system_prompt=INPUT_SYSTEM_PROMPT, 
        llm_service=llm_service,
        datapoint_id=datapoint_id,
    )
    
    output_keys = process_keys(
        fulltext=fulltext, 
        image_caption_pairs=image_caption_pairs,
        key_prompts=OUTPUT_KEY_PROMPTS, 
        system_prompt=OUTPUT_SYSTEM_PROMPT, 
        llm_service=llm_service,
        datapoint_id=datapoint_id,
    )
    
    return input_keys, output_keys


def load_datapoint_whitelist(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Datapoint list file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        ids = {line.strip() for line in f if line.strip()}
    return ids or None


def main(datapoint_whitelist: Optional[Set[str]] = None):
    """Main entry point to process all datapoints and save results."""
    # Configure environment
    os.environ["PYTORCH_SDP_ATTENTION"] = "0"  # disable fused SDPA
    
    # Initialize services
    llm_service = QwenService()
    
    # Load any previously saved results
    input_json = load_existing_results("input.json")
    output_json = load_existing_results("output.json")
    failed_ids = load_failed_ids()

    # Get all datapoint folders to process
    datapoint_folders = collect_datapoint_folders()
    if datapoint_whitelist is not None:
        datapoint_folders = [
            path for path in datapoint_folders
            if os.path.basename(path) in datapoint_whitelist
        ]
    
    # Process each datapoint with progress tracking
    for datapoint_dir in tqdm(datapoint_folders, desc="Processing datapoints"):
        dp_name = os.path.basename(datapoint_dir)
        if dp_name in failed_ids:
            print(f"Skipping {dp_name}: previously marked as failed")
            continue

        input_complete = has_all_keys(input_json.get(dp_name), INPUT_KEY_PROMPTS.keys())
        output_complete = has_all_keys(output_json.get(dp_name), OUTPUT_KEY_PROMPTS.keys())
        if input_complete and output_complete:
            print(f"Skipping {dp_name}: already complete")
            continue

        if dp_name in failed_ids and not (input_complete and output_complete):
            print(f"Retrying {dp_name}: previously marked as failed but incomplete entries found")

        try:
            input_obj, output_obj = process_datapoint(datapoint_dir, dp_name, llm_service)
        except Exception as exc:
            print(f"Failed datapoint {dp_name}: {exc}")
            log_failure(dp_name, str(exc))
            continue
        
        # Store results
        input_json[dp_name] = input_obj
        output_json[dp_name] = output_obj
        
        # Incrementally save results after each datapoint
        save_json(input_json, "input.json", OUTPUT_DIR)
        save_json(output_json, "output.json", OUTPUT_DIR)

if __name__ == "__main__":
    args = parse_args()
    whitelist = load_datapoint_whitelist(args.datapoint_list)
    main(datapoint_whitelist=whitelist) 