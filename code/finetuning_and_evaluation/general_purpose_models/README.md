# Gemini 2.5 Pro Inference via OpenRouter

This script runs inference on Gemini 2.5 Pro (or other vision models) using the OpenRouter API.

## Setup

1. **Install dependencies:**
   ```bash
   pip install requests pillow tqdm
   ```

2. **API Key:**
   - Default key is embedded in the script
   - Or set via `--api_key` argument
   - Or set as environment variable: `export OPENROUTER_API_KEY=sk-or-v1-...`

## Usage

### Basic Usage

```bash
python infer_gemini_openrouter.py \
  --test_prompts_jsonl ../pixtral/test_prompts_set2.jsonl \
  --test_images_json ../pixtral/test_images_set2.json \
  --output_json gemini_inference_results.json \
  --max_tokens 2048 \
  --temperature 0.7
```

### Available Models on OpenRouter

**Default model:** `google/gemini-2.5-pro`

Other vision models you can use with `--model` flag:
- `google/gemini-2.0-flash-exp:free` (free tier, lower quality)
- `google/gemini-pro-1.5` (paid, earlier version)
- `anthropic/claude-3.5-sonnet` (paid, excellent vision)
- `openai/gpt-4o` (paid)
- `meta-llama/llama-3.2-90b-vision-instruct` (paid)

Check [OpenRouter Models](https://openrouter.ai/models) for the latest options.

### All Arguments

```bash
python infer_gemini_openrouter.py \
  --api_key sk-or-v1-... \
  --model google/gemini-2.0-flash-exp:free \
  --test_prompts_jsonl <path_to_prompts.jsonl> \
  --test_images_json <path_to_images.json> \
  --output_json <output_filename.json> \
  --max_tokens 2048 \
  --temperature 0.7 \
  --top_p 0.9 \
  --site_url "https://your-site.com" \
  --site_name "Your App Name"
```

## Features

- ✅ **Multimodal support**: Sends text + images to OpenRouter
- ✅ **Base64 encoding**: Converts images to base64 for API compatibility
- ✅ **Image resizing**: Automatically resizes to 500x500 to reduce payload size
- ✅ **Crash recovery**: Saves results after each sample
- ✅ **Resume capability**: Skips already processed samples
- ✅ **Error handling**: Continues processing even if some samples fail
- ✅ **Progress tracking**: Uses tqdm for visual progress

## Input Format

### Prompts JSONL
```json
{"input_case": "sample_001", "text": "Describe this image in detail."}
{"input_case": "sample_002", "text": "What components are shown?"}
```

### Images JSON
```json
{
  "sample_001": {
    "paths": {
      "img1": "/path/to/image1.jpg",
      "img2": "/path/to/image2.png"
    }
  }
}
```

## Output Format

```json
{
  "sample_001": {
    "input_case": "sample_001",
    "user_prompt": "Describe this image...",
    "generated_output": "The image shows...",
    "num_images_used": 2,
    "model": "google/gemini-2.0-flash-exp:free",
    "usage": {
      "prompt_tokens": 150,
      "completion_tokens": 200,
      "total_tokens": 350
    }
  }
}
```

## Cost Considerations

- **Default model**: `google/gemini-2.5-pro` is a paid model - check [OpenRouter Pricing](https://openrouter.ai/models) for current costs
- **Free alternative**: Use `--model google/gemini-2.0-flash-exp:free` for free tier (with rate limits)
- **Image size**: Script resizes images to 500x500 to reduce token usage and API costs
- **Token tracking**: Output includes usage statistics for cost monitoring

## Notes

- Images are converted to base64 JPEG (quality=85) to minimize payload size
- Default resize is 500x500 pixels - adjust in `image_to_base64()` if needed
- The script automatically handles both Pixtral (`[/INST]`) and Qwen (`<|im_start|>assistant`) prompt formats
- OpenRouter automatically handles model routing and fallbacks

## Troubleshooting

1. **Rate limit errors**: Switch to a paid model or wait before retrying
2. **Image too large**: Reduce resize dimensions in the code
3. **API errors**: Check your API key and model availability
4. **Timeout**: Increase timeout in `call_openrouter()` function
