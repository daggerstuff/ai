"""
Psychology knowledge base loader for clinical data, personality assessments, and therapy content.
"""

import os

import pandas as pd

from .logger import get_logger

logger = get_logger("dataset_pipeline.psychology_loader")


def load_psychology_knowledge_csv(path: str):
    """
    Load a CSV file containing psychology knowledge base data.
    Args:
        path (str): Path to the psychology knowledge CSV file.
    Returns:
        pd.DataFrame or None if loading fails.
    """
    if not os.path.exists(path):
        logger.error(f"Psychology knowledge base not found: {path}")
        return None
    try:
        df = pd.read_csv(path, keep_default_na=False)
        logger.info(f"Loaded psychology knowledge base '{path}' with {len(df)} records.")
        return df
    except Exception as e:
        logger.error(f"Failed to load psychology knowledge base '{path}': {e}")
        return None
