#!/usr/bin/env python3
"""
Compute LLM-as-Judge metrics using vLLM server.
IMPROVED VERSION: Includes Chain-of-Thought, Tolerance Calibration, and Robust Parsing.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

from openai import OpenAI
from tqdm import tqdm


def normalize_inference_data(data: dict) -> dict:
    """Normalize inference data to a dictionary keyed by input_case."""
    if 'inference_results' in data and isinstance(data['inference_results'], list):
        normalized = {}
        for item in data['inference_results']:
            key = item.get('input_case')
            if key:
                normalized[key] = item
        return normalized
    return data


def parse_json_from_text(text: str) -> Optional[dict]:
    """
    Robustly extract JSON object from text using regex.
    Handles Markdown code blocks and CoT preamble.
    """
    try:
        # 1. Try to find JSON inside markdown blocks
        if "```json" in text:
            # Extract content inside ```json ... ```
            pattern = r"```json(.*?)```"
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return json.loads(matches[-1].strip()) # Take the last one if multiple
        
        # 2. If no markdown, use Regex to find the first valid { ... } structure
        #    This finds the largest outer-most brackets
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
            
        # 3. Fallback: Try loading the whole text
        return json.loads(text)
    except Exception:
        return None


def validate_llm_response(response_data: dict) -> tuple:
    """
    Validate LLM judge response has correct schema.
    Returns: (is_valid, error_message)
    """
    required_fields = {
        "faithfulness_score": int,
        "faithfulness_reason": str,
        "completeness_score": int,
        "completeness_reason": str,
        "reasoning_score": int,
        "reasoning_reason": str
    }
    
    if not isinstance(response_data, dict):
        return False, "Response is not a dictionary"

    for field, expected_type in required_fields.items():
        if field not in response_data:
            return False, f"Missing required field: {field}"
        
        # Allow float for scores if model outputs 4.5, cast later
        if "score" in field and isinstance(response_data[field], float):
            continue
            
        if not isinstance(response_data[field], expected_type):
            return False, f"Field '{field}' must be {expected_type.__name__}, got {type(response_data[field]).__name__}"
    
    # Validate score ranges (1-5)
    score_fields = ["faithfulness_score", "completeness_score", "reasoning_score"]
    for field in score_fields:
        score = response_data[field]
        if not (1 <= score <= 5):
            return False, f"Score '{field}' must be between 1 and 5, got {score}"
    
    return True, None


def llm_judge_metric(ground_truth_text: str, generated_text: str, client: OpenAI,
                     model: str, temperature: float, max_tokens: int, 
                     max_retries: int = 3) -> Optional[Dict]:
    """
    Uses vLLM server to grade the generation against the Ground Truth.
    Uses Chain-of-Thought prompting to reduce pedantry in large models.
    """
    
    # ------------------------------------------------------------------
    # SYSTEM PROMPT: Defines Role, Tolerance, and Scoring Anchors
    # ------------------------------------------------------------------
    system_prompt = """You are a Principal Engineering Lead acting as an Evaluator. 
Your goal is to assess a "Candidate Response" against a "Ground Truth" (GT).

### CRITICAL ENGINEERING PRINCIPLES (CALIBRATION):
1. **Physics over Phrasing:** If the Candidate explains the correct mechanism using different words than the GT, it is CORRECT. (e.g., "Thermal breakdown" == "Heat degradation").
2. **Numerical Tolerance:** Allow for engineering approximations (±5%) unless exact precision is explicitly required. 
   - GT: "594°C" vs Candidate: "600°C" -> SCORE 5 (Acceptable).
   - GT: "594°C" vs Candidate: "800°C" -> SCORE 1 (Failure).
3. **Information Density:** Do not penalize conciseness. If the Candidate merges two GT steps into one valid summary step, give full marks.

### SCORING RUBRIC (1-5):
1. **Faithfulness (Hallucination Check):**
   - 5: Factually aligned. No invented constraints.
   - 3: Minor inaccuracies or "fluff" that doesn't harm the outcome.
   - 1: Major hallucination (wrong material, contradicts physics).

2. **Completeness (Recall):**
   - 5: Captures all Critical Process Parameters (CPP) mentioned in GT.
   - 3: Misses minor details but captures the main strategy.
   - 1: Misses the primary objective (e.g., optimizes Temp but ignores Pressure).

3. **Reasoning (Physics Logic):**
   - 5: Correct Cause->Effect chain. Shows understanding of *why*.
   - 3: Surface level. Lists steps without linking them to physics.
   - 1: Incoherent or "Word Salad".

### OUTPUT INSTRUCTIONS:
You must first output a **Step-by-Step Analysis** checking the parameters, and then output a **JSON** block."""

    # ------------------------------------------------------------------
    # USER PROMPT: Forces extraction and comparison
    # ------------------------------------------------------------------
    user_prompt = f"""### GROUND TRUTH (Reference):
{ground_truth_text}

### CANDIDATE (Generated):
{generated_text}

### EVALUATION TASK:
1. **Parameter Check:** List key numbers/materials from GT. Does Candidate match them (within tolerance)?
2. **Logic Check:** Does the Candidate's reasoning hold up physically?
3. **Score:** Assign scores based on the analysis.

Output the textual analysis first, followed immediately by the JSON object:
```json
{{
    "faithfulness_score": <int>,
    "faithfulness_reason": "<string>",
    "completeness_score": <int>,
    "completeness_reason": "<string>",
    "reasoning_score": <int>,
    "reasoning_reason": "<string>"
}}
```"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    for attempt in range(max_retries):
        try:
            print(f"  [Attempt {attempt + 1}/{max_retries}] Calling vLLM API...")
            sys.stdout.flush()
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95 if temperature > 0 else 1.0,
                timeout=180
            )
            
            if not response.choices:
                print("  Error: No choices in response")
                continue

            # Extract content (handle DeepSeek/Reasoning models if applicable)
            message = response.choices[0].message
            response_text = message.content or ""
            
            # If model uses separate reasoning field (some new models), append it for logging
            if hasattr(message, 'reasoning') and message.reasoning:
                print("  (Model used explicit reasoning field)")
            
            # Parse JSON using robust regex
            result = parse_json_from_text(response_text)
            
            if not result:
                print(f"  Error: Could not find valid JSON in response. Text length: {len(response_text)}")
                # Optional: Print snippet to debug
                # print(f"  Snippet: {response_text[-200:]}")
                if attempt < max_retries - 1:
                    continue
                return None
            
            # Validate schema
            is_valid, error_msg = validate_llm_response(result)
            
            if is_valid:
                # Calculate average
                total = (
                    result["faithfulness_score"] + 
                    result["completeness_score"] + 
                    result["reasoning_score"]
                ) / 3.0
                result["total_score"] = total
                return result
            else:
                print(f"  Validation error: {error_msg}")
                if attempt < max_retries - 1:
                    continue
                    
        except json.JSONDecodeError:
            print(f"  JSON Decode Error on attempt {attempt+1}")
        except Exception as e:
            print(f"  LLM Judge Error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return None
    
    return None


def atomic_write(data, path: Path):
    """Atomically write JSON data to a file."""
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp_path, 'w', encoding='utf-8') as tmp_file:
        json.dump(data, tmp_file, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def parse_args():
    parser = argparse.ArgumentParser(description='Compute Improved LLM-as-Judge metrics')
    parser.add_argument('--inference_results', type=str, required=True,
                        help='Path to inference results JSON file')
    parser.add_argument('--ground_truth', type=str,
                        default='/data/mguru/04_Finetuning/frame-finetuning-evaluation/final_output.json',
                        help='Path to ground truth JSON file')
    parser.add_argument('--output_key', type=str, required=True,
                        choices=['output_1', 'output_2'],
                        help='Which output to compare against')
    parser.add_argument('--results_json', type=str, required=True,
                        help='Path to results JSON file')
    parser.add_argument('--base_url', type=str, default='http://localhost:8000/v1',
                        help='vLLM server base URL')
    parser.add_argument('--model', type=str, default='QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ',
                        help='Model name')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Low temp (0.1) better than 0.0 for CoT reasoning')
    parser.add_argument('--max_tokens', type=int, default=4096,
                        help='Max tokens')
    parser.add_argument('--max_retries', type=int, default=3,
                        help='Max retries')
    parser.add_argument('--overwrite-existing', action='store_true',
                        help='Recompute even if exists')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("IMPROVED LLM-as-Judge (Physics-Aware, CoT Enabled)")
    print("=" * 80)
    print(f"Inference results: {args.inference_results}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Output key: {args.output_key}")
    print(f"Model: {args.model}")
    print("=" * 80)

    # Load inference results
    try:
        with open(args.inference_results, 'r', encoding='utf-8') as f:
            raw_inference = json.load(f)
        inference_data = normalize_inference_data(raw_inference)
        print(f"\nLoaded {len(inference_data)} inference results")
    except Exception as e:
        print(f"Error loading inference results: {e}")
        return

    # Load ground truth
    try:
        with open(args.ground_truth, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)
        print(f"Loaded {len(ground_truth_data)} ground truth entries")
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return

    # Load or create results file
    results_path = Path(args.results_json)
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing result entries")
    else:
        results = {}
        results_path.parent.mkdir(parents=True, exist_ok=True)
        print("Creating new results file...")

    # Initialize OpenAI client
    print("\nConnecting to vLLM server...")
    try:
        client = OpenAI(base_url=args.base_url, api_key="EMPTY")
        # Quick health check
        client.models.list()
        print(f"✓ Connected to {args.base_url}")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return

    processed = 0
    skipped = 0

    print(f"\nComputing metrics for {args.output_key}...")
    
    for input_case, inference_value in tqdm(inference_data.items(), desc="Processing"):
        generated_output = inference_value.get("generated_output")
        if not generated_output:
            skipped += 1
            continue
            
        if input_case not in ground_truth_data:
            skipped += 1
            continue
            
        ground_truth = ground_truth_data[input_case].get(args.output_key, "")
        if not ground_truth:
            skipped += 1
            continue

        # Check existing
        existing_entry = results.get(input_case, {})
        if not args.overwrite_existing and "LLM_Judge_Total" in existing_entry:
            skipped += 1
            continue

        # Compute Metrics
        tqdm.write(f"\n>>> Analyzing {input_case}...")
        judge_result = llm_judge_metric(
            ground_truth,
            generated_output,
            client,
            args.model,
            args.temperature,
            args.max_tokens,
            args.max_retries
        )
            
        if judge_result:
            if input_case not in results:
                results[input_case] = {"input_case": input_case, "output_key": args.output_key}
            
            # Map results to storage keys
            results[input_case].update({
                "LLM_Judge_Faithfulness": judge_result["faithfulness_score"],
                "LLM_Judge_Faithfulness_Reason": judge_result["faithfulness_reason"],
                "LLM_Judge_Completeness": judge_result["completeness_score"],
                "LLM_Judge_Completeness_Reason": judge_result["completeness_reason"],
                "LLM_Judge_Reasoning": judge_result["reasoning_score"],
                "LLM_Judge_Reasoning_Reason": judge_result["reasoning_reason"],
                "LLM_Judge_Total": judge_result["total_score"]
            })
            
            processed += 1
            tqdm.write(f"✓ Score: {judge_result['total_score']:.2f} (F:{judge_result['faithfulness_score']} C:{judge_result['completeness_score']} R:{judge_result['reasoning_score']})")
            atomic_write(results, results_path)
        else:
            tqdm.write(f"✗ Failed to score {input_case}")
            skipped += 1

    print(f"\n✓ Done! Processed: {processed}, Skipped: {skipped}")
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()