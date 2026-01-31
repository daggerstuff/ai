"""
Dataset inventory and metadata tracking system for the dataset pipeline.
Scans a directory or list of dataset files and collects metadata such as name, size, columns, and last modified time.
"""

import os

import pandas as pd

from .logger import get_logger

logger = get_logger("dataset_pipeline.dataset_inventory")


def get_dataset_metadata(path: str) -> dict | None:
    """
    Collect metadata for a single dataset file (CSV).
    Args:
        path (str): Path to the dataset file.
    Returns:
        dict with metadata or None if file not found or unreadable.
    """
    if not os.path.exists(path):
        logger.warning(f"Dataset not found: {path}")
        return None
    try:
        stat = os.stat(path)
        df = pd.read_csv(path, nrows=5)
        columns = list(df.columns)
        return {
            "path": path,
            "name": os.path.basename(path),
            "size_bytes": stat.st_size,
            "last_modified": stat.st_mtime,
            "columns": columns,
            "preview_rows": df.to_dict(orient="records"),
        }
    except Exception as e:
        logger.error(f"Failed to collect metadata for {path}: {e}")
        return None


def scan_datasets(directory: str, pattern: str = ".csv") -> list[dict]:
    """
    Scan a directory for dataset files and collect metadata for each.
    Args:
        directory (str): Directory to scan.
        pattern (str): File extension or pattern to match (default: '.csv').
    Returns:
        List of metadata dicts for each dataset file found.
    """
    inventory = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(pattern):
                fpath = os.path.join(root, fname)
                meta = get_dataset_metadata(fpath)
                if meta:
                    inventory.append(meta)
    logger.info(f"Scanned {directory}: found {len(inventory)} dataset(s).")
    return inventory
