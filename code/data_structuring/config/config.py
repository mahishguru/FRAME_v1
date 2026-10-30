import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# File paths
TEXT_DIR = "data/text"
IMAGE_DIR = "data/images"
OUTPUT_DIR = "output"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model configuration
MAX_TOKENS = 4096