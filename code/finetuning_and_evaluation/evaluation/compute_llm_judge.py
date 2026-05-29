#!/usr/bin/env python3
"""Compute LLM-as-Judge metrics using Azure OpenAI."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

from openai import AzureOpenAI
from tqdm import tqdm


def normalize_inference_data(data: dict) -> dict:
    if 'inference_results' in data and isinstance(data['inference_results'], list):
        normalized = {}
        for item in data['inference_results']:
            key = item.get('input_case')
            if key:
                normalized[key] = item
        return normalized
    return data


def validate_llm_response(response_data: dict) -> tuple[bool, Optional[str]]:
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
    
    for field, expected_type in required_fields.items():
        if field not in response_data:
            return False, f"Missing required field: {field}"
        
        if not isinstance(response_data[field], expected_type):
            return False, f"Field '{field}' must be {expected_type.__name__}, got {type(response_data[field]).__name__}"
    
    # Validate score ranges (1-5)
    score_fields = ["faithfulness_score", "completeness_score", "reasoning_score"]
    for field in score_fields:
        score = response_data[field]
        if not (1 <= score <= 5):
            return False, f"Score '{field}' must be between 1 and 5, got {score}"
    
    return True, None


def llm_judge_metric(ground_truth_text: str, generated_text: str, client: AzureOpenAI, 
                     model: str, max_retries: int = 3) -> Optional[Dict]:
    """
    Uses Azure OpenAI to grade the generation against the Ground Truth.
    Returns: Dict with faithfulness, completeness, reasoning scores and total average.
    """
    
    system_prompt = """
You are a Senior Chief Engineer acting as an Evaluator. 
You will compare a "Generated Engineering Analysis" against a "Ground Truth Reference".

Your goal is to grade the Generation on three specific metrics.

CRITICAL INSTRUCTION FOR ENGINEERING DATA:
1. **Numbers Matter:** If the GT says "594°C" and Gen says "600°C", strictly penalize.
2. **Reasoning over Phrasing:** If the Gen explains the *physics* (e.g., "Time-Temp Equivalence") correctly but uses different words than the GT, **reward it**. Do not penalize valid detailed reasoning.
3. **Hallucination Check:** If the Gen adds optimization steps that physically contradict the GT, score Faithfulness low.
"""

    user_prompt = f"""
### GROUND TRUTH (Reference):
{ground_truth_text}

### GENERATED TEXT (Candidate):
{generated_text}

### TASK:
Evaluate the Candidate based on the following Rubric. 
Provide a score (1-5) and a short rationale for each.

1. **Faithfulness (Hallucination Check):**
   - 5: No hallucinations. All numbers/facts are supported by GT or valid inferences.
   - 3: Minor inaccuracies (e.g., slight number drift) or unverifiable fluff.
   - 1: Major hallucinations (invented physics, wrong material, contradicts GT).

2. **Completeness (Recall):**
   - 5: Captures ALL key optimization strategies, parameters, and locations mentioned in GT.
   - 3: Misses 1-2 minor parameters but gets the main concept.
   - 1: Misses critical parameters (e.g., optimizes Temp but forgets Pressure).

3. **Reasoning Depth (Physics Understanding):**
   - 5: Explicitly links Cause -> Effect (e.g., "Reduce Time -> To minimize thermal degradation"). Shows deep understanding.
   - 3: Surface level. Lists steps (e.g., "Reduce Time") without explaining the engineering physics.
   - 1: Keyword salad. Uses correct words but in incoherent sentences.

### OUTPUT FORMAT (JSON ONLY):
{{
    "faithfulness_score": <int 1-5>,
    "faithfulness_reason": "<string>",
    "completeness_score": <int 1-5>,
    "completeness_reason": "<string>",
    "reasoning_score": <int 1-5>,
    "reasoning_reason": "<string>"
}}
"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Deterministic grading
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate response
            is_valid, error_msg = validate_llm_response(result)
            
            if is_valid:
                # Calculate total as average of three scores
                total = (
                    result["faithfulness_score"] + 
                    result["completeness_score"] + 
                    result["reasoning_score"]
                ) / 3.0
                
                result["total_score"] = round(total, 2)
                return result
            else:
                print(f"Validation failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
                if attempt < max_retries - 1:
                    # Add validation error to prompt for retry
                    user_prompt += f"\n\n### PREVIOUS ATTEMPT FAILED:\n{error_msg}\nPlease provide a valid response."
                    continue
                else:
                    print(f"Max retries reached. Last error: {error_msg}")
                    return None
                    
        except json.JSONDecodeError as e:
            print(f"JSON decode error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                user_prompt += f"\n\n### PREVIOUS ATTEMPT FAILED:\nInvalid JSON format. Please provide valid JSON only."
                continue
            else:
                return None
                
        except Exception as e:
            print(f"LLM Judge Error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                continue
            else:
                return None
    
    return None


def atomic_write(data, path: Path):
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp_path, 'w', encoding='utf-8') as tmp_file:
        json.dump(data, tmp_file, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def parse_args():
    parser = argparse.ArgumentParser(description='Compute LLM-as-Judge metrics using Azure OpenAI')
    parser.add_argument('--inference_results', type=str, required=True,
                        help='Path to inference results JSON file')
    parser.add_argument('--ground_truth', type=str,
                        default='/data/mguru/04_Finetuning/frame-finetuning-evaluation/final_output.json',
                        help='Path to ground truth JSON file')
    parser.add_argument('--output_key', type=str, required=True,
                        choices=['output_1', 'output_2'],
                        help='Which output to compare against (output_1 or output_2)')
    parser.add_argument('--results_json', type=str, required=True,
                        help='Path to results JSON file (will be created/updated)')
    parser.add_argument('--azure_endpoint', type=str, default=os.environ.get('AZURE_OPENAI_ENDPOINT'),
                        help='Azure OpenAI endpoint URL (or set AZURE_OPENAI_ENDPOINT env var)')
    parser.add_argument('--azure_api_key', type=str, default=os.environ.get('AZURE_OPENAI_KEY') or os.environ.get('AZURE_OPENAI_API_KEY'),
                        help='Azure OpenAI API key (or set AZURE_OPENAI_KEY / AZURE_OPENAI_API_KEY env var)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='Azure OpenAI model deployment name (default: gpt-4o)')
    parser.add_argument('--api_version', type=str, default='2024-02-15-preview',
                        help='Azure OpenAI API version')
    parser.add_argument('--max_retries', type=int, default=3,
                        help='Maximum retries for validation failures (default: 3)')
    parser.add_argument('--overwrite-existing', action='store_true',
                        help='Recompute LLM judge metrics even if values already exist')
    return parser.parse_args()


def main():
    args = parse_args()

    azure_endpoint = args.azure_endpoint or os.environ.get('AZURE_OPENAI_ENDPOINT')
    azure_api_key = args.azure_api_key or os.environ.get('AZURE_OPENAI_KEY') or os.environ.get('AZURE_OPENAI_API_KEY')

    if not azure_endpoint:
        raise ValueError("Azure endpoint not provided. Pass --azure_endpoint or set AZURE_OPENAI_ENDPOINT env var.")
    if not azure_api_key:
        raise ValueError("Azure API key not provided. Pass --azure_api_key or set AZURE_OPENAI_KEY / AZURE_OPENAI_API_KEY env var.")

    print("=" * 80)
    print("LLM-as-Judge Metrics (Azure OpenAI)")
    print("=" * 80)
    print(f"Inference results: {args.inference_results}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Output key: {args.output_key}")
    print(f"Results JSON: {args.results_json}")
    print(f"Azure endpoint: {azure_endpoint}")
    print(f"Model: {args.model}")
    print(f"API version: {args.api_version}")
    print(f"Max retries: {args.max_retries}")
    print(f"Overwrite existing: {args.overwrite_existing}")
    print("=" * 80)

    with open(args.inference_results, 'r', encoding='utf-8') as f:
        raw_inference = json.load(f)
    inference_data = normalize_inference_data(raw_inference)
    print(f"Loaded {len(inference_data)} inference results")

    with open(args.ground_truth, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    print(f"Loaded {len(ground_truth_data)} ground truth entries")

    results_path = Path(args.results_json)
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing result entries")
    else:
        results = {}
        results_path.parent.mkdir(parents=True, exist_ok=True)
        print("Creating new results file...")

    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version=args.api_version
    )

    processed = 0
    skipped = 0

    print(f"\nComputing LLM-as-Judge metrics for {args.output_key}...")
    for input_case, inference_value in tqdm(inference_data.items(), desc="Processing"):
        generated_output = inference_value.get("generated_output")
        if not generated_output:
            tqdm.write(f"Skipping {input_case}: generated_output is null/empty")
            skipped += 1
            continue
        if input_case not in ground_truth_data:
            tqdm.write(f"Warning: {input_case} not found in ground truth. Skipping.")
            skipped += 1
            continue
        ground_truth = ground_truth_data[input_case].get(args.output_key, "")
        if not ground_truth:
            tqdm.write(f"Warning: {input_case} has empty {args.output_key}. Skipping.")
            skipped += 1
            continue

        existing_entry = results.get(input_case, {})
        if not args.overwrite_existing and "LLM_Judge_Total" in existing_entry:
            skipped += 1
            continue

        try:
            judge_result = llm_judge_metric(
                ground_truth,
                generated_output,
                client,
                args.model,
                args.max_retries
            )
            
            if judge_result is None:
                tqdm.write(f"Error: LLM judge failed for {input_case} after {args.max_retries} retries")
                skipped += 1
                continue
                
        except Exception as exc:
            tqdm.write(f"Error computing LLM judge for {input_case}: {exc}")
            skipped += 1
            continue

        if input_case not in results:
            results[input_case] = {
                "input_case": input_case,
                "output_key": args.output_key
            }
        
        # Store all scores and reasons
        results[input_case]["LLM_Judge_Faithfulness"] = judge_result["faithfulness_score"]
        results[input_case]["LLM_Judge_Faithfulness_Reason"] = judge_result["faithfulness_reason"]
        results[input_case]["LLM_Judge_Completeness"] = judge_result["completeness_score"]
        results[input_case]["LLM_Judge_Completeness_Reason"] = judge_result["completeness_reason"]
        results[input_case]["LLM_Judge_Reasoning"] = judge_result["reasoning_score"]
        results[input_case]["LLM_Judge_Reasoning_Reason"] = judge_result["reasoning_reason"]
        results[input_case]["LLM_Judge_Total"] = judge_result["total_score"]
        
        processed += 1
        atomic_write(results, results_path)

    print(f"\nSaving final results to {results_path}...")
    atomic_write(results, results_path)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Results saved to: {results_path}")
    print("=" * 80)
    print("✓ Done!")


if __name__ == '__main__':
    main()
