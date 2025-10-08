"""
Main dataset loading and acquisition system for the dataset pipeline.
Supports loading from HuggingFace Hub and local sources.
"""

from datasets import load_dataset
from logger import get_logger

logger = get_logger("dataset_pipeline.data_loader")


def load_hf_dataset(
    dataset_name: str, split: str = "train", progress_callback=None, error_callback=None, **kwargs
):
    """
    Load a dataset from the HuggingFace Hub with optional progress and error callbacks.
    Args:
        dataset_name (str): The name or path of the dataset on the Hub.
        split (str): The split to load (default: "train").
        progress_callback (callable, optional): Function called with progress updates.
        error_callback (callable, optional): Function called with error messages.
        **kwargs: Additional arguments for load_dataset.
    Returns:
        Dataset object or None if loading fails.
    """
    try:
        logger.info(f"Loading HuggingFace dataset: {dataset_name} (split={split})")
        if progress_callback:
            progress_callback(f"Starting download: {dataset_name} (split={split})")
        ds = load_dataset(dataset_name, split=split, **kwargs)
        if hasattr(ds, "__len__"):
            dataset_size = len(ds)  # type: ignore[arg-type]
            logger.info(f"Loaded dataset '{dataset_name}' with {dataset_size} records.")
            if progress_callback:
                progress_callback(f"Loaded dataset '{dataset_name}' with {dataset_size} records.")
        else:
            logger.info(f"Loaded iterable dataset '{dataset_name}'.")
            if progress_callback:
                progress_callback(f"Loaded iterable dataset '{dataset_name}'.")
        return ds
    except Exception as e:
        logger.error(f"Failed to load HuggingFace dataset '{dataset_name}': {e}")
        if error_callback:
            error_callback(f"Failed to load HuggingFace dataset '{dataset_name}': {e}")
        return None


def load_local_dataset(path: str, progress_callback=None, error_callback=None):
    """
    Load a dataset from a local file path with optional progress and error callbacks.
    Args:
        path (str): Path to the local dataset.
        progress_callback (callable, optional): Function called with progress updates.
        error_callback (callable, optional): Function called with error messages.
    Returns:
        Loaded dataset object or None.
    """
    try:
        logger.info(f"Loading local dataset from: {path}")
        if progress_callback:
            progress_callback(f"Starting local dataset load: {path}")
        ds = load_dataset(path)

        # Handle different dataset types for logging
        if hasattr(ds, "__len__"):
            # Type checker: we've verified ds has __len__ method
            dataset_size = len(ds)  # type: ignore[arg-type]
            logger.info(f"Loaded local dataset from '{path}' with {dataset_size} records.")
            if progress_callback:
                progress_callback(
                    f"Loaded local dataset from '{path}' with {dataset_size} records."
                )
        else:
            # IterableDataset doesn't support len()
            logger.info(f"Loaded local iterable dataset from '{path}'.")
            if progress_callback:
                progress_callback(f"Loaded local iterable dataset from '{path}'.")

        return ds
    except Exception as e:
        logger.error(f"Failed to load local dataset from '{path}': {e}")
        if error_callback:
            error_callback(f"Failed to load local dataset from '{path}': {e}")
        return None
