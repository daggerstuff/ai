#!/usr/bin/env python3
"""
Fix for Hugging Face authentication issue in training pipeline.

This script addresses the 401 Unauthorized error when loading the
LatitudeGames/Harbinger-24B model. The issue occurs because the training
script doesn't properly handle Hugging Face authentication.
"""

import logging
import os
from typing import Optional

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_huggingface_auth(token: Optional[str] = None) -> bool:
    """
    Set up Hugging Face authentication.

    Args:
        token: Hugging Face API token. If None, will try to get from environment.

    Returns:
        bool: True if authentication was successful, False otherwise.
    """
    try:
        # Get token from environment if not provided
        if token is None:
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

        if token:
            # Login to Hugging Face
            login(token=token)
            logger.info("Successfully authenticated with Hugging Face")
            return True
        else:
            logger.warning("No Hugging Face token found in environment")
            return False

    except Exception as e:
        logger.error(f"Failed to authenticate with Hugging Face: {e}")
        return False


def load_model_with_auth(
    model_name: str = "LatitudeGames/Harbinger-24B",
) -> Optional[AutoModelForCausalLM]:
    """
    Load a Hugging Face model with proper authentication handling.

    Args:
        model_name: Name of the model to load.

    Returns:
        Optional[AutoModelForCausalLM]: Loaded model or None if failed.
    """
    try:
        return _extracted_from_load_model_with_auth_13(model_name)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        logger.error(
            "This might be due to authentication issues or model access restrictions"
        )
        return None


# TODO Rename this here and in `load_model_with_auth`
def _extracted_from_load_model_with_auth_13(model_name):
    # First try to set up authentication
    if not setup_huggingface_auth():
        logger.warning(
            "Proceeding without authentication - some models may fail to load"
        )

    # Load tokenizer first (usually doesn't require auth)
    logger.info(f"Loading tokenizer for {model_name}")
    AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Load model
    logger.info(f"Loading model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, device_map="auto", torch_dtype="auto"
    )

    logger.info(f"Successfully loaded model {model_name}")
    return model


def main():
    """Main function to test the fix."""
    logger.info("Testing Hugging Face authentication fix...")

    # Test with the problematic model
    model_name = "LatitudeGames/Harbinger-24B"
    if load_model_with_auth(model_name):
        logger.info(f"✅ Successfully loaded {model_name}")
        logger.info("Authentication fix is working correctly")
    else:
        _extracted_from_main_11(model_name)


# TODO Rename this here and in `main`
def _extracted_from_main_11(model_name):
    logger.error(f"❌ Failed to load {model_name}")
    logger.error("Please check your Hugging Face token and model access permissions")

    # Provide helpful instructions
    logger.info("\nTo fix this issue:")
    logger.info(
        "1. Get a Hugging Face token from https://huggingface.co/settings/tokens"
    )
    logger.info(
        "2. Set it as an environment variable: export HF_TOKEN='your_token_here'"
    )
    logger.info("3. Make sure you have access to the LatitudeGames/Harbinger-24B model")
    logger.info("4. Run this script again")


if __name__ == "__main__":
    main()
