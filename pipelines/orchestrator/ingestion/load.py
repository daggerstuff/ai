"""
Modular dataset loader for supervised fine-tuning pipelines.

Loads and validates all required datasets for Pixelated Empathy supervised training.
- No hardcoded secrets or paths; all configs via parameters or secure config.
- Logs all sources and errors.
- Returns loaded data in a modular, testable format.
- PEP8, <500 lines, ready for TDD.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("dataset_pipeline.load")
logger.setLevel(logging.INFO)


class DatasetLoaderError(Exception):
    """Custom exception for dataset loading errors."""


def _default_dataset_config() -> dict[str, str | Path]:
    """
    Returns the default dataset configuration.
    All paths should be overridden via parameters or config for production.
    """
    return {
        "wendy": "ai/wendy/datasets-wendy",
        "amod": "Amod/mental_health_counseling_conversations",
        "mpingale": "mpingale/mental-health-chat-dataset",
        "heliosbrahma": "heliosbrahma/mental_health_chatbot_dataset",
    }


def _find_data_files(dataset_path: Path, exts: list[str] | None = None) -> list[Path]:
    """
    Recursively find all data files in a dataset directory with given extensions.
    """
    if exts is None:
        exts = [".csv", ".json", ".jsonl", ".parquet"]
    files = []
    for ext in exts:
        files.extend(dataset_path.rglob(f"*{ext}"))
    return files


def _load_file(file_path: Path) -> Any:
    """
    Load a single data file into a pandas DataFrame or list of dicts.
    Supports CSV, JSON, JSONL, Parquet.
    """
    try:
        if file_path.suffix == ".csv":
            return pd.read_csv(file_path)
        if file_path.suffix == ".json":
            return pd.read_json(file_path)
        if file_path.suffix == ".jsonl":
            # JSONL: one JSON object per line
            return pd.read_json(file_path, lines=True)
        if file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        raise DatasetLoaderError(f"Unsupported file type: {file_path}")
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise DatasetLoaderError(f"Failed to load {file_path}: {e}") from e


def _validate_dataset_presence(dataset_path: Path) -> None:
    """
    Validate that the dataset path exists and is not empty.
    """
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        raise DatasetLoaderError(f"Dataset path does not exist: {dataset_path}")
    if dataset_path.is_dir():
        files = list(dataset_path.iterdir())
        if not files:
            logger.error(f"Dataset directory is empty: {dataset_path}")
            raise DatasetLoaderError(f"Dataset directory is empty: {dataset_path}")
    elif dataset_path.is_file():
        if dataset_path.stat().st_size == 0:
            logger.error(f"Dataset file is empty: {dataset_path}")
            raise DatasetLoaderError(f"Dataset file is empty: {dataset_path}")
    else:
        logger.error(f"Invalid dataset path: {dataset_path}")
        raise DatasetLoaderError(f"Invalid dataset path: {dataset_path}")


def load_datasets(
    dataset_config: dict[str, str | Path] | None = None,
    logger_override: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Loads all required datasets for supervised fine-tuning.

    Args:
        dataset_config: Dict mapping dataset names to paths (str or Path).
        logger_override: Optional logger to use instead of default.

    Returns:
        Dict mapping dataset names to loaded data (DataFrame or list of DataFrames).

    Raises:
        DatasetLoaderError: If any dataset fails to load or validate.
    """
    log = logger_override or logger
    config = dataset_config or _default_dataset_config()
    loaded: dict[str, Any] = {}

    for name, path in config.items():
        dataset_path = Path(path)
        log.info(f"Validating dataset '{name}' at {dataset_path.resolve()}")
        try:
            _validate_dataset_presence(dataset_path)
        except DatasetLoaderError as e:
            log.error(f"Validation failed for dataset '{name}': {e}")
            raise

        if dataset_path.is_file():
            log.info(f"Loading file for dataset '{name}': {dataset_path.name}")
            loaded[name] = _load_file(dataset_path)
        elif dataset_path.is_dir():
            files = _find_data_files(dataset_path)
            if not files:
                log.error(f"No data files found in directory: {dataset_path}")
                raise DatasetLoaderError(f"No data files found in directory: {dataset_path}")
            log.info(f"Found {len(files)} files for dataset '{name}': {[f.name for f in files]}")
            loaded[name] = []
            for f in files:
                try:
                    loaded[name].append(_load_file(f))
                except DatasetLoaderError as e:
                    log.warning(f"Skipping file {f}: {e}")
        else:
            log.error(f"Invalid dataset path: {dataset_path}")
            raise DatasetLoaderError(f"Invalid dataset path: {dataset_path}")

    log.info(f"Loaded datasets: {list(loaded.keys())}")
    return loaded
