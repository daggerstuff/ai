"""Dataset ingestion utilities for supported corpus formats."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


class DatasetLoaderError(RuntimeError):
    """Raised when datasets cannot be loaded safely."""


def _default_dataset_config() -> dict[str, str]:
    """Return a minimal default dataset config.

    Kept intentionally sparse and overridable in production setups/tests.
    """

    return {}


def _load_single_file(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load a single supported file into a DataFrame."""

    if not path.exists():
        raise DatasetLoaderError(f"Dataset path does not exist: {path}")
    if path.stat().st_size == 0:
        raise DatasetLoaderError(f"Dataset file is empty: {path}")

    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".json":
            return pd.read_json(path)
        if suffix in {".jsonl", ".ndjson"}:
            return pd.read_json(path, lines=True)
        if suffix == ".parquet":
            return pd.read_parquet(path)
    except Exception as exc:
        raise DatasetLoaderError(f"Failed to load {path}: {exc!s}") from exc

    raise DatasetLoaderError(f"Unsupported file type: {path.suffix}")


def _is_supported_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".csv", ".json", ".jsonl", ".ndjson", ".parquet"}


def _load_from_path(path: Path, logger: logging.Logger) -> pd.DataFrame | list[pd.DataFrame]:
    """Load one path entry and return DataFrame or list of DataFrames."""

    if not path.exists():
        raise DatasetLoaderError(f"Dataset path does not exist: {path}")

    if path.is_file():
        return _load_single_file(path, logger)

    if not path.is_dir():
        raise DatasetLoaderError(f"Unsupported dataset path type: {path}")

    loaded: list[pd.DataFrame] = []
    for item in sorted(path.iterdir()):
        if item.name.startswith("."):
            continue
        if not _is_supported_file(item):
            logger.warning("Skipping file (unsupported format) while loading %s", item)
            continue
        try:
            loaded.append(_load_single_file(item, logger))
        except DatasetLoaderError as exc:
            logger.warning("Skipping file %s (%s)", item, exc)

    if not loaded:
        raise DatasetLoaderError(f"No supported files found in {path}")
    return loaded


def load_datasets(
    config: dict[str, str] | None = None,
    *,
    logger_override: logging.Logger | None = None,
) -> dict[str, pd.DataFrame | list[pd.DataFrame]]:
    """Load one or many datasets described by mapping name -> path."""

    logger = logger_override or logging.getLogger(__name__)
    cfg = config if config is not None else _default_dataset_config()

    if not cfg:
        cfg = _default_dataset_config()
    if not cfg:
        raise DatasetLoaderError("No dataset configuration provided")

    results: dict[str, pd.DataFrame | list[pd.DataFrame]] = {}
    for name, path_value in cfg.items():
        path = Path(path_value)
        logger.info("Loading dataset %s from %s", name, path)
        loaded = _load_from_path(path, logger)
        results[name] = loaded

    return results


class _LoadModule(SimpleNamespace):
    """Module-style compatibility object expected by existing imports."""


load = _LoadModule(
    load_datasets=load_datasets,
    DatasetLoaderError=DatasetLoaderError,
    _default_dataset_config=_default_dataset_config,
)


__all__ = ["DatasetLoaderError", "load"]
