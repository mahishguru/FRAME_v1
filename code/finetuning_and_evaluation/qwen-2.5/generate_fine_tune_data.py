import os
import json
import numpy as np
from transformers import AutoProcessor
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split

# System prompts for different JSON generations
OUTPUT_SYSTEM_PROMPT = """
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(f'{BASE_DIR}/../final_input.json', 'r') as finput:    
    data_input = json.load(finput)

with open(f'{BASE_DIR}/../final_output.json', 'r') as foutput:
    data_output = json.load(foutput)

assert data_input.keys()==data_output.keys(), "Keys for input and output json files don't match!"

# Get all keys and split into training and testing sets
all_keys = list(data_input.keys())
train_keys, test_keys = train_test_split(all_keys, test_size=0.15, random_state=42)

# Define file names for the two sets
train_prompt_jsonl_set1 = f'{BASE_DIR}/train_prompts_set1.jsonl'
train_image_json_set1 = f'{BASE_DIR}/train_images_set1.json'
test_prompt_jsonl_set1 = f'{BASE_DIR}/test_prompts_set1.jsonl'
test_image_json_set1 = f'{BASE_DIR}/test_images_set1.json'

train_prompt_jsonl_set2 = f'{BASE_DIR}/train_prompts_set2.jsonl'
train_image_json_set2 = f'{BASE_DIR}/train_images_set2.json'
test_prompt_jsonl_set2 = f'{BASE_DIR}/test_prompts_set2.jsonl'
test_image_json_set2 = f'{BASE_DIR}/test_images_set2.json'

model_id = "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)

images_dir = os.path.join(BASE_DIR, '../../images_pre_filtered')

def create_dataset(keys, prompt_jsonl_set1, image_json_set1, prompt_jsonl_set2, image_json_set2, set_type):
    prompt_imgs_set1 = {}
    prompt_imgs_set2 = {}

    with open(image_json_set1, 'w') as fwrite_image_set1, \
         open(image_json_set2, 'w') as fwrite_image_set2, \
         open(prompt_jsonl_set1, 'w') as fwrite_prompt_set1, \
         open(prompt_jsonl_set2, 'w') as fwrite_prompt_set2:

        print(f'Writing {set_type}_image_json for set 1...')
        for input_case in tqdm(keys):
            prompt_imgs_set1[input_case] = {}
            prompt_imgs_set1[input_case]['paths'] = {
                img_name: f'{images_dir}/{input_case}/{img_name}' 
                for img_name in sorted((f for f in os.listdir(f'{images_dir}/{input_case}') if f.endswith('.jpg')), 
                                       key=lambda x: tuple(int(num) for num in re.findall(r'\d+', x)))}
        json.dump(prompt_imgs_set1, fwrite_image_set1, indent=4)

        print(f'Writing {set_type}_image_json for set 2...')
        for input_case in tqdm(keys):
            prompt_imgs_set2[input_case] = {}
            prompt_imgs_set2[input_case]['paths'] = {
                img_name: f'{images_dir}/{input_case}/{img_name}' 
                for img_name in sorted((f for f in os.listdir(f'{images_dir}/{input_case}') if f.endswith('.jpg')), 
                                       key=lambda x: tuple(int(num) for num in re.findall(r'\d+', x)))}
        json.dump(prompt_imgs_set2, fwrite_image_set2, indent=4)

        print(f'Writing {set_type}_prompt_jsonl for set 1...')
        for input_case in tqdm(keys):
            conversation_set1 = [
                {"role": "user",
                 "content": [{"type": "image"}] * len(prompt_imgs_set1[input_case]['paths']) +
                            [{"type": "text", "text": f"{data_input[input_case][f'key_{i}']}"} for i in range(1, 5)] + 
                            [{"type": "text", "text": f"{OUTPUT_KEY_PROMPTS['output_1']}"}]
                },
                {"role": "assistant",
                 "content": [{"type": "text", "text": f"{data_output[input_case]['output_1']}"}]
                }
            ]
            prompt_set1 = processor.apply_chat_template(conversation_set1, tokenize=False)
            fwrite_prompt_set1.write(json.dumps({"input_case": input_case, "text": prompt_set1}) + '\n')

        print(f'Writing {set_type}_prompt_jsonl for set 2...')
        for input_case in tqdm(keys):
            conversation_set2 = [
                {"role": "user",
                 "content": [{"type": "image"}] * len(prompt_imgs_set2[input_case]['paths']) +
                            [{"type": "text", "text": f"{data_input[input_case][f'key_{i}']}"} for i in range(1, 5)] +
                            [{"type": "text", "text": f"Analysis of the component's behavior: {data_output[input_case]['output_1']}"}] +
                            [{"type": "text", "text": f"{OUTPUT_KEY_PROMPTS['output_2']}"}]
                },
                {"role": "assistant",
                 "content": [{"type": "text", "text": f"{data_output[input_case]['output_2']}"}]
                }
            ]
            prompt_set2 = processor.apply_chat_template(conversation_set2, tokenize=False)
            fwrite_prompt_set2.write(json.dumps({"input_case": input_case, "text": prompt_set2}) + '\n')

# Create training datasets
create_dataset(train_keys, 
               train_prompt_jsonl_set1, train_image_json_set1,
               train_prompt_jsonl_set2, train_image_json_set2,
               "train")

# Create testing datasets
create_dataset(test_keys, 
               test_prompt_jsonl_set1, test_image_json_set1,
               test_prompt_jsonl_set2, test_image_json_set2,
               "test")

print("Successfully created two training and two test sets.")