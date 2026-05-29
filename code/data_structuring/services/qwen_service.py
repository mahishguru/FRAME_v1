import os
from datetime import datetime
from typing import List, Dict, Any
from huggingface_hub import login as hf_login
from config.config import MAX_TOKENS, OUTPUT_DIR
import requests

LOGFILE = os.path.join(OUTPUT_DIR, "api_log.txt")

def log_api_call(message: str):
    """Log API calls with timestamp for tracking and debugging."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

class QwenService:
    """Service for interfacing with the Qwen3-VL model via vLLM's OpenAI-compatible API."""
    
    def __init__(self):
        # Allow user to specify GPU(s) via GPU_DEVICE or CUDA_VISIBLE_DEVICES
        gpu_env = os.getenv("GPU_DEVICE")
        if gpu_env:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_env
            
        # Attempt to login to Hugging Face if token is provided
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            try:
                hf_login(token=hf_token)
            except Exception as e:
                print(f"Warning: Hugging Face login failed: {e}")
                
        # Configure vLLM API endpoint and model
        self.vllm_api_url = os.getenv("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
        self.model_id = os.getenv(
            "VLLM_MODEL_ID",
            "QuantTrio/Qwen3-VL-235B-A22B-Thinking-FP8",
        )

    def process_with_history(
        self,
        messages: list,
        log_context: Dict[str, Any] = None,
        max_new_tokens: int = None
    ) -> str:
        """
        Send multimodal messages to vLLM API and get the model's response.
        
        Args:
            messages: List of message dicts in OpenAI Vision API format
            log_context: Optional context information for logging
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            The text response from the model
        """
        # Log the API call
        log_info = f"Qwen call | "
        if log_context:
            log_info += ", ".join(f"{k}={v}" for k, v in log_context.items())
        log_api_call(log_info)
        
        # Print for debugging
        # print(messages)
        print(f"Sending to vLLM API: {len(messages)} messages with multimodal content")
        
        # Prepare the API request
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_new_tokens or MAX_TOKENS,
            "temperature": 0.7,
            "top_p": 0.95,
            "stop": ["<|endoftext|>"]
        }
        
        # Make the API call
        response = requests.post(self.vllm_api_url, json=payload)
        
        # Handle non-200 responses
        if response.status_code != 200:
            print(f"Error response from API: {response.status_code}")
            print(f"Response content: {response.text}")
            response.raise_for_status()
            
        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {}) or {}

        raw_content = message.get("content")
        if not raw_content:
            raw_content = choice.get("content", "")
        normalized_content = self._normalize_content(raw_content)
        
        if not normalized_content.strip():
            log_api_call("Qwen empty content | payload=" + str(choice))

        log_api_call(f"Qwen call success | {log_context if log_context else ''}")
        return normalized_content

    @staticmethod
    def _normalize_content(content: Any) -> str:
        """Convert OpenAI-style content (string or list of blocks) into plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts: List[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_piece = block.get("text")
                    if text_piece:
                        texts.append(text_piece)
            return "\n".join(texts)
        return "" if content is None else str(content)
