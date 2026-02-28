"""
Local dataset loader for existing mental health data.
"""

import os

import pandas as pd

from .logger import get_logger

logger = get_logger("dataset_pipeline.local_loader")


def load_local_csv(path: str, progress_callback=None, error_callback=None):
    """
    Load a local CSV file as a pandas DataFrame with optional progress and error callbacks.
    Args:
        path (str): Path to the CSV file.
        progress_callback (callable, optional): Function called with progress updates.
        error_callback (callable, optional): Function called with error messages.
    Returns:
        pd.DataFrame or None if loading fails.
    """
    if not os.path.exists(path):
        logger.error(f"Local dataset not found: {path}")
        if error_callback:
            error_callback(f"Local dataset not found: {path}")
        return None
    try:
        if progress_callback:
            progress_callback(f"Starting local CSV load: {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded local dataset '{path}' with {len(df)} records.")
        if progress_callback:
            progress_callback(f"Loaded local dataset '{path}' with {len(df)} records.")
        return df
    except Exception as e:
        logger.error(f"Failed to load local dataset '{path}': {e}")
        if error_callback:
            error_callback(f"Failed to load local dataset '{path}': {e}")
        return None
