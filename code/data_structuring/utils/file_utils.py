import os
import json
import ast
from typing import List, Dict, Any, Tuple
from config.config import TEXT_DIR, IMAGE_DIR, OUTPUT_DIR

def read_text_file(file_path: str) -> str:
    """Read content from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_text_files() -> List[str]:
    """Get all text files from the text directory."""
    return [f for f in os.listdir(TEXT_DIR) if f.endswith('.txt')]

def get_image_files() -> List[str]:
    """Get all image files from the image directory."""
    return [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

def get_datapoint_folders(root_dir: str) -> List[str]:
    """Get all datapoint folders in the root directory (ignore hidden)."""
    return [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]

def read_fulltext(datapoint_dir: str) -> str:
    """Read the fulltext from fulltext/fulltext.txt in the datapoint folder."""
    fulltext_path = os.path.join(datapoint_dir, 'fulltext', 'fulltext.txt')
    with open(fulltext_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_infographic_images(datapoint_dir: str) -> List[str]:
    """Get all image file paths in infographic/ of the datapoint folder."""
    inf_dir = os.path.join(datapoint_dir, 'infographic')
    return [os.path.join(inf_dir, f) for f in os.listdir(inf_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def read_figure_captions(datapoint_dir: str) -> Dict[str, Dict[str, Any]]:
    """Read figure_captions.json and map image filename to its caption and tag.
    If no JSON file is found or it's invalid, assigns a default caption for each figure image present."""
    captions_path = os.path.join(datapoint_dir, 'caption', 'figure_captions.json')
    mapping: Dict[str, Dict[str, Any]] = {}

    def assign_default_captions() -> Dict[str, Dict[str, Any]]:
        figures_dir = os.path.join(datapoint_dir, 'figure')
        defaults: Dict[str, Dict[str, Any]] = {}
        if os.path.isdir(figures_dir):
            for fname in os.listdir(figures_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg')):
                    defaults[fname] = {
                        'caption': 'Caption in figure itself.',
                        'tag': ''
                    }
        return defaults

    # If no captions file, or file is empty/invalid JSON, use defaults
    if not os.path.exists(captions_path):
        return assign_default_captions()

    try:
        with open(captions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        # Invalid or empty JSON, fallback to defaults
        return assign_default_captions()

    # Process entries in valid JSON
    for entry in data:
        path = entry.get('path')
        if not path or not isinstance(path, str):
            continue  # skip entries with missing or invalid path
        filename = os.path.basename(path)
        desc = entry.get('description', '')

        # Normalize description field
        if isinstance(desc, str) and desc.strip().startswith('[') and desc.strip().endswith(']'):
            try:
                desc_list = ast.literal_eval(desc)
                description = ' '.join(desc_list)
            except Exception:
                description = desc
        elif isinstance(desc, list):
            description = ' '.join(desc)
        else:
            description = str(desc)

        mapping[filename] = {
            'caption': description,
            'tag': entry.get('tag', '')
        }

    # If mapping empty, possibly JSON had no usable entries: fallback
    if not mapping:
        return assign_default_captions()

    return mapping


def get_image_caption_pairs(datapoint_dir: str) -> List[Tuple[str, str]]:
    """Return list of (image_path, caption) for all images in infographic/ with captions."""
    images = get_infographic_images(datapoint_dir)
    captions = read_figure_captions(datapoint_dir)
    pairs = []
    for img_path in images:
        filename = os.path.basename(img_path)
        if filename in captions:
            pairs.append((img_path, captions[filename]['caption']))
    return pairs

def save_json(data: Dict[str, Any], filename: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_full_paths(directory: str, filenames: List[str]) -> List[str]:
    """Get full paths for files in a directory."""
    return [os.path.join(directory, filename) for filename in filenames] 