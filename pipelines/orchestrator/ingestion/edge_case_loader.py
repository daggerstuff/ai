"""
Edge case scenario loader for dataset pipeline.
Loads and validates edge case data for robustness testing.
"""

import os

import pandas as pd

from .logger import get_logger

logger = get_logger("dataset_pipeline.edge_case_loader")


def load_edge_case_csv(path: str):
    """
    Load a CSV file containing edge case scenarios.
    Args:
        path (str): Path to the edge case CSV file.
    Returns:
        pd.DataFrame or None if loading fails.
    """
    if not os.path.exists(path):
        logger.error(f"Edge case dataset not found: {path}")
        return None
    try:
        df = pd.read_csv(path, keep_default_na=False)
        logger.info(f"Loaded edge case dataset '{path}' with {len(df)} records.")
        return df
    except Exception as e:
        logger.error(f"Failed to load edge case dataset '{path}': {e}")
        return None
