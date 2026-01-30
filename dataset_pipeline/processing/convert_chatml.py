"""
ChatML conversion utilities for the dataset pipeline.
Converts pandas DataFrames with 'prompt'/'response' columns into
ChatML-formatted lists of dictionaries.
"""

import logging
from typing import Dict, List

import pandas as pd

# Set up logging
logger = logging.getLogger("ai.dataset_pipeline.convert_chatml")


def convert_to_chatml(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Convert a DataFrame with 'prompt' and 'response' columns to a list of
    ChatML messages.

    Args:
        df: Input pandas DataFrame containing 'prompt' and 'response' columns.

    Returns:
        List of dictionaries with 'role' and 'content' keys.

    Raises:
        ValueError: If required columns are missing.
    """
    if df is None:
        return []

    required_cols = ["prompt", "response"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Attempt to redact PII
    try:
        from ai.dataset_pipeline.processing import clean

        # We redact in place or use returned df
        df = clean.redact_pii_in_text_fields(df, required_cols)
    except ImportError:
        logger.warning("Could not import clean module for PII redaction.")
    except Exception as e:
        logger.error(f"PII redaction failed: {e}")

    chatml_messages = []

    for _, row in df.iterrows():
        prompt = row.get("prompt")
        response = row.get("response")

        # Validation: skip nulls or non-strings
        if pd.isna(prompt) or pd.isna(response):
            continue
        if not isinstance(prompt, str) or not isinstance(response, str):
            continue

        chatml_messages.append({"role": "user", "content": prompt})
        chatml_messages.append({"role": "assistant", "content": response})

    return chatml_messages
