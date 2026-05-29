import os
import json
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# System prompt for the model (same as used in Pixtral/Qwen training)

SYSTEM_PROMPT = """
You are a Senior Lead Engineer specializing in simulation analysis and design optimization.
Your task is to analyze the provided technical description and visual data to generate high-precision engineering insights.

CRITICAL GENERATION RULES:
1. **DIRECTNESS (NO YAPPING):** Do not use conversational fillers (e.g., "The data suggests...", "In conclusion..."). Start every point directly with the technical fact.
2. **PHYSICS OVER CODE:** Focus exclusively on the *physical reality* (e.g., "Increase fillet radius"). Do NOT describe software implementation steps unless explicitly asked for algorithm improvements.
3. **MANDATORY REASONING:** For every optimization strategy or behavior observation, you MUST state the *physical mechanism* or rationale (The "Why") based on the input data.
4. **STRUCTURED OUTPUT:** Use clear headers and dense bullet points. Do not write long, unstructured paragraphs.
5. **PARAMETER GROUPING:** If an optimization involves coupled parameters (e.g., Temperature AND Time), group them into a single strategic point rather than listing them separately.
"""

# User prompts for different keys (output)
OUTPUT_KEY_PROMPTS = {
    "output_1": "**SYSTEM BEHAVIOR, PHYSICS & QUALITY METRICS**\n"
                "Analyze based on the provided problem description and figures:\n"
                "1. **Dominant Fields & KPIs:** \n"
                "   - **Identification:** What are the primary physical variables OR quality metrics? (e.g., 'Von Mises Stress', 'Geometric Deviation', 'Filling Rate').\n"
                "   - **Extremes & Locations:** Where do maximums, minimums, or defects occur? (e.g., 'Max stress at fillet', 'Recirculation at inlet').\n"
                "   - **Reference Figure:** Cite Figure numbers.\n"
                "2. **Critical Phenomena & Patterns:**\n"
                "   - **Observed Behavior:** Describe specific phenomena (e.g., 'Flow separation', 'Pillow defect', 'Numerical instability').\n"
                "   - **Driving Cause:** What physical principle drives this? (e.g., 'Adverse pressure gradient', 'Excessive normal pressure').\n"
                "3. **Failure Modes:**\n"
                "   - **Bottlenecks:** What specifically limits performance? (e.g., 'Fatigue crack initiation', 'Dielectric breakdown').",

    "output_2": "**OPTIMIZATION & IMPROVEMENT STRATEGIES**\n"
                "Based on the problem description and figures, propose improvements for **ANY** aspect (Component, Material, Process, or Simulation Method).\n"
                "**Constraint:** If a strategy involves multiple coupled parameters (e.g., optimal Temp AND Time), group them into ONE entry.\n\n"
                "For each strategy:\n"
                "1. **The Target:** What is being improved? (e.g., 'The Finite Element Model', 'The Casting Process', 'The Geometry').\n"
                "2. **The Specific Modification:** What exact change is recommended? (e.g., 'Set Temp to 594°C and Time to 394s', 'Use adaptive sparsity matching'). **Do NOT describe software code steps.**\n"
                "3. **The Location/Scope:** Where is this applied? (e.g., 'Global domain', 'At the substructure interface').\n"
                "4. **The Mechanism/Rationale:** **WHY** does this work? Use technical reasoning. (e.g., 'Leverages time-temperature equivalence', 'Compensates for stiffness degradation')."
}

# Excluded input cases (hardcoded list)
EXCLUDED_CASES = [
    "10_1016_j_compstruct_2023_116967",
    "10_1016_j_compstruct_2024_118781",
    "10_1016_j_jobe_2022_105346",
    "10_1007_s41104_016_0014_0",
    # Add more cases here as needed
] 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(f'{BASE_DIR}/../final_input.json', 'r') as finput:    
    data_input = json.load(finput)

with open(f'{BASE_DIR}/../final_output.json', 'r') as foutput:
    data_output = json.load(foutput)

assert data_input.keys()==data_output.keys(), "Keys for input and output json files don't match!"

# Get all keys and split into training and testing sets
# IMPORTANT: Do the split FIRST with random_state=42 to match Qwen/Pixtral exactly
all_keys = list(data_input.keys())
train_keys, test_keys = train_test_split(all_keys, test_size=0.15, random_state=42)

# Filter out excluded cases from test sets only (after split to maintain same split as other models)
test_keys = [key for key in test_keys if key not in EXCLUDED_CASES]
excluded_count = len([k for k in all_keys if k in EXCLUDED_CASES and k in test_keys])
if excluded_count > 0:
    print(f"Excluded {excluded_count} cases from test set")
print(f"Total test keys: {len(test_keys)}")

# Define file names for test sets only
test_prompt_jsonl_set1 = f'{BASE_DIR}/test_prompts_set1.jsonl'
test_image_json_set1 = f'{BASE_DIR}/test_images_set1.json'
test_prompt_jsonl_set2 = f'{BASE_DIR}/test_prompts_set2.jsonl'
test_image_json_set2 = f'{BASE_DIR}/test_images_set2.json'

images_dir = os.path.join(BASE_DIR, '../../images_pre_filtered')

def create_test_dataset(keys, prompt_jsonl_set1, image_json_set1, prompt_jsonl_set2, image_json_set2):
    """Create test datasets for Gemini inference (no chat template, just raw text prompts)."""
    prompt_imgs_set1 = {}
    prompt_imgs_set2 = {}

    with open(image_json_set1, 'w') as fwrite_image_set1, \
         open(image_json_set2, 'w') as fwrite_image_set2, \
         open(prompt_jsonl_set1, 'w') as fwrite_prompt_set1, \
         open(prompt_jsonl_set2, 'w') as fwrite_prompt_set2:

        print(f'Writing test_image_json for set 1...')
        for input_case in tqdm(keys):
            prompt_imgs_set1[input_case] = {}
            prompt_imgs_set1[input_case]['paths'] = {
                img_name: f'{images_dir}/{input_case}/{img_name}' 
                for img_name in sorted((f for f in os.listdir(f'{images_dir}/{input_case}') if f.endswith('.jpg')), 
                                       key=lambda x: tuple(int(num) for num in re.findall(r'\d+', x)))}
        json.dump(prompt_imgs_set1, fwrite_image_set1, indent=4)

        print(f'Writing test_image_json for set 2...')
        for input_case in tqdm(keys):
            prompt_imgs_set2[input_case] = {}
            prompt_imgs_set2[input_case]['paths'] = {
                img_name: f'{images_dir}/{input_case}/{img_name}' 
                for img_name in sorted((f for f in os.listdir(f'{images_dir}/{input_case}') if f.endswith('.jpg')), 
                                       key=lambda x: tuple(int(num) for num in re.findall(r'\d+', x)))}
        json.dump(prompt_imgs_set2, fwrite_image_set2, indent=4)

        print(f'Writing test_prompt_jsonl for set 1...')
        for input_case in tqdm(keys):
            # Build user prompt for set 1 (output_1 generation)
            user_prompt = "**Component Information:**\n\n"
            for i in range(1, 5):
                user_prompt += f"{data_input[input_case][f'key_{i}']}\n\n"
            user_prompt += f"**Question:**\n\n{OUTPUT_KEY_PROMPTS['output_1']}"
            
            fwrite_prompt_set1.write(json.dumps({
                "input_case": input_case,
                "system": SYSTEM_PROMPT.strip(),
                "user": user_prompt,
                "expected_output": data_output[input_case]['output_1']
            }) + '\n')

        print(f'Writing test_prompt_jsonl for set 2...')
        for input_case in tqdm(keys):
            # Build user prompt for set 2 (output_2 generation)
            user_prompt = "**Component Information:**\n\n"
            for i in range(1, 5):
                user_prompt += f"{data_input[input_case][f'key_{i}']}\n\n"
            user_prompt += f"**Analysis of the component's behavior:**\n\n{data_output[input_case]['output_1']}\n\n"
            user_prompt += f"**Question:**\n\n{OUTPUT_KEY_PROMPTS['output_2']}"
            
            fwrite_prompt_set2.write(json.dumps({
                "input_case": input_case,
                "system": SYSTEM_PROMPT.strip(),
                "user": user_prompt,
                "expected_output": data_output[input_case]['output_2']
            }) + '\n')

# Create testing datasets only
create_test_dataset(test_keys, 
                    test_prompt_jsonl_set1, test_image_json_set1,
                    test_prompt_jsonl_set2, test_image_json_set2)

print("Successfully created two test sets for Gemini inference.")
