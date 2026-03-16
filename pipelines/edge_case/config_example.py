#!/usr/bin/env python3
"""
Example Configuration for Edge Case Generation
Copy this file to config.py and modify as needed
"""

# API Configuration
API_PROVIDER = "openai"  # Options: "openai", "anthropic", "ollama"
API_KEY = None  # Set your API key here or use environment variable
MODEL_NAME = "gpt-3.5-turbo"  # See README for model options

# Generation Settings
SCENARIOS_PER_CATEGORY = 20  # Number of prompts per category (1-50)
MAX_CONVERSATIONS = 100  # Maximum conversations to generate
OUTPUT_DIR = "edge_case_output"  # Output directory name

# Quality Settings
MIN_QA_PAIRS = 3  # Minimum Q&A pairs per conversation
MAX_RETRIES = 3  # Retry attempts for failed prompts
DELAY_BETWEEN_REQUESTS = 1  # Seconds to wait between API calls

# Model-specific configurations
MODEL_CONFIGS = {
    "openai": {
        "gpt-3.5-turbo": {"max_tokens": 1000, "temperature": 0.7},
        "gpt-4": {"max_tokens": 1200, "temperature": 0.6},
        "gpt-4-turbo-preview": {"max_tokens": 1200, "temperature": 0.6},
    },
    "anthropic": {
        "claude-3-haiku-20240307": {"max_tokens": 1000, "temperature": 0.7},
        "claude-3-sonnet-20240229": {"max_tokens": 1200, "temperature": 0.6},
        "claude-3-opus-20240229": {"max_tokens": 1200, "temperature": 0.5},
    },
    "ollama": {
        "llama2": {"temperature": 0.7},
        "mistral": {"temperature": 0.7},
        "llama2:13b": {"temperature": 0.6},
    },
}

# Categories to focus on (leave empty for all 25 categories)
FOCUS_CATEGORIES = [
    # "suicidality",
    # "substance_abuse_crisis",
    # "borderline_crisis",
    # "trauma_flashback"
]

# Difficulty levels to include
DIFFICULTY_LEVELS = ["moderate", "high", "very_high"]  # Or ["all"]


def get_config():
    """Get configuration dictionary"""
    return {
        "api_provider": API_PROVIDER,
        "api_key": API_KEY,
        "model_name": MODEL_NAME,
        "scenarios_per_category": SCENARIOS_PER_CATEGORY,
        "max_conversations": MAX_CONVERSATIONS,
        "output_dir": OUTPUT_DIR,
        "min_qa_pairs": MIN_QA_PAIRS,
        "max_retries": MAX_RETRIES,
        "delay": DELAY_BETWEEN_REQUESTS,
        "focus_categories": FOCUS_CATEGORIES,
        "difficulty_levels": DIFFICULTY_LEVELS,
        "model_config": MODEL_CONFIGS.get(API_PROVIDER, {}).get(MODEL_NAME, {}),
    }
