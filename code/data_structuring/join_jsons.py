import argparse
import json
import os
from typing import Dict, Any

DEFAULT_OUTPUT_DIR = "output"
INPUT_FILES = ["input_0.json", "input_1.json"]
OUTPUT_FILES = ["output_0.json", "output_1.json"]


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for data in dicts:
        merged.update(data)
    return merged


def write_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def join_files(files, output_name, base_dir):
    data_dicts = [load_json(os.path.join(base_dir, name)) for name in files]
    merged = merge_dicts(*data_dicts)
    write_json(merged, os.path.join(base_dir, output_name))
    print(f"Created {output_name} with {len(merged)} entries")


def main():
    parser = argparse.ArgumentParser(description="Join split JSON files into a single file.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing the split JSON files and where joined files will be written."
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=INPUT_FILES,
        help="Input JSON files for the input dataset."
    )
    parser.add_argument(
        "--input-joined",
        default="input_joined.json",
        help="Filename for merged input JSON."
    )
    parser.add_argument(
        "--output-files",
        nargs="+",
        default=OUTPUT_FILES,
        help="Input JSON files for the output dataset."
    )
    parser.add_argument(
        "--output-joined",
        default="output_joined.json",
        help="Filename for merged output JSON."
    )
    args = parser.parse_args()

    join_files(args.input_files, args.input_joined, args.output_dir)
    join_files(args.output_files, args.output_joined, args.output_dir)


if __name__ == "__main__":
    main()
