"""
Pixelated Empathy - Dataset Cleaning & Deduplication Pipeline

This module provides a modular, testable function to clean, normalize, deduplicate,
and privacy-sanitize loaded datasets for supervised fine-tuning. All configs are
parameterized or loaded from secure config. No secrets are hardcoded.

Author: Pixelated Empathy AI Team
"""

import logging
from typing import Any

import pandas as pd

# Expose logger for test mocking
logger = logging.getLogger("dataset_cleaning")

# Default PII column patterns (can be overridden via config)
DEFAULT_PII_PATTERNS = [
    "email", "phone", "ssn", "name", "address", "dob", "birth", "contact", "pii"
]

def setup_logger(log_level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger for the cleaning pipeline.
    """
    logger = logging.getLogger("dataset_cleaning")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger

def find_pii_columns(
    columns: list[str],
    pii_patterns: list[str] | None = None,
    explicit_pii: set[str] | None = None
) -> set[str]:
    """
    Identifies columns containing PII based on patterns and explicit config.
    """
    patterns = pii_patterns or DEFAULT_PII_PATTERNS
    pii_cols = set()
    for col in columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in patterns):
            pii_cols.add(col)
    if explicit_pii:
        pii_cols.update(explicit_pii)
    return pii_cols

def normalize_text_columns(df: pd.DataFrame, text_columns: list[str] | None = None) -> pd.DataFrame:
    """
    Strips whitespace and lowercases text columns.
    """
    if text_columns is None:
        # Guess text columns as object dtype
        text_columns = df.select_dtypes(include=["object"]).columns.tolist()
    for col in text_columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.lower()
        )
    return df

import re


def redact_pii_in_text_fields(
    df: pd.DataFrame,
    text_columns: list | None = None,
    logger: logging.Logger | None = None
) -> pd.DataFrame:
    """
    Redacts SSN-like patterns in text columns for privacy compliance.
    Emits an info-level privacy audit log if any redaction occurs.
    """
    if text_columns is None:
        text_columns = df.select_dtypes(include=["object"]).columns.tolist()
    ssn_pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    logger = logger or globals().get("logger") or logging.getLogger("dataset_cleaning")
    for col in text_columns:
        def redact_and_log(x):
            redacted = ssn_pattern.sub("[REDACTED-SSN]", str(x))
            if redacted != str(x):
                logger.info(f"privacy audit: redacted PII in field '{col}'")
            return redacted
        df[col] = df[col].astype(str).apply(redact_and_log)
    return df

def remove_pii(
    df: pd.DataFrame,
    pii_columns: set[str],
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Removes PII columns from the DataFrame.
    """
    existing_pii = [col for col in pii_columns if col in df.columns]
    if existing_pii:
        logger.info(f"Removing PII columns: {existing_pii}")
        df = df.drop(columns=existing_pii)
    return df

def clean_and_deduplicate(
    datasets,
    config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
    audit_log: list | None = None
) -> pd.DataFrame:
    """
    Cleans, normalizes, deduplicates, and privacy-sanitizes all records across datasets.

    Minimal implementation for TDD: Accepts a single DataFrame or a list, handles PII removal,
    normalization, and deduplication for basic test coverage.
    """
    # Accept both a single DataFrame or a list for test compatibility
    if isinstance(datasets, pd.DataFrame):
        datasets = [datasets]
    if not datasets or not all(isinstance(df, pd.DataFrame) for df in datasets):
        raise ValueError("Input must be a non-empty list of pandas DataFrames.")

    config = config or {}
    log_level = config.get("log_level", logging.INFO)
    # Use module-level logger if not provided
    logger = logger or globals().get("logger") or setup_logger(log_level)

    # Concatenate all datasets
    df = pd.concat(datasets, ignore_index=True)

    # Check for required columns (e.g., "text")
    required_columns = config.get("required_columns", ["text"])
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Identify and remove PII columns
    pii_patterns = config.get("pii_patterns")
    explicit_pii = set(config.get("explicit_pii", []))
    pii_columns = find_pii_columns(df.columns.tolist(), pii_patterns, explicit_pii)
    df = remove_pii(df, pii_columns, logger)

    # Redact PII patterns in text fields (e.g., SSNs)
    text_columns = config.get("text_columns")
    df = redact_pii_in_text_fields(df, text_columns, logger)

    # Normalize text columns (after redaction, so redacted tokens are lowercased)
    df = normalize_text_columns(df, text_columns)

    # Drop rows with all NaN
    df = df.dropna(how="all")

    # Deduplicate
    dedup_columns = config.get("dedup_columns")
    if dedup_columns:
        df = df.drop_duplicates(subset=dedup_columns, keep="first", ignore_index=True)
    else:
        df = df.drop_duplicates(keep="first", ignore_index=True)

    # Optionally append audit log
    if audit_log is not None:
        audit_log.append({
            "event": "cleaned",
            "rows": len(df),
            "columns": list(df.columns),
        })

    return df

    config = config or {}
    log_level = config.get("log_level", logging.INFO)
    logger = logger or setup_logger(log_level)

    logger.info("Starting dataset cleaning pipeline.")
    stats = {
        "input_datasets": len(datasets),
        "input_rows": sum(len(df) for df in datasets),
        "input_columns": list({col for df in datasets for col in df.columns}),
        "pii_columns": set(),
        "rows_after_concat": 0,
        "rows_after_dedup": 0,
        "columns_after_cleaning": [],
    }

    # Concatenate all datasets
    try:
        df = pd.concat(datasets, ignore_index=True)
        stats["rows_after_concat"] = len(df)
        logger.info(f"Concatenated {stats['input_datasets']} datasets: {stats['rows_after_concat']} rows.")
    except Exception as e:
        logger.error(f"Failed to concatenate datasets: {e}")
        raise

    # Identify and remove PII columns
    pii_patterns = config.get("pii_patterns")
    explicit_pii = set(config.get("explicit_pii", []))
    pii_columns = find_pii_columns(df.columns.tolist(), pii_patterns, explicit_pii)
    stats["pii_columns"] = pii_columns
    df = remove_pii(df, pii_columns, logger)

    # Normalize text columns
    text_columns = config.get("text_columns")
    try:
        df = normalize_text_columns(df, text_columns)
        logger.info(f"Normalized text columns: {text_columns or 'auto-detected'}")
    except Exception as e:
        logger.warning(f"Text normalization failed: {e}")

    # Handle missing values (optional: drop rows with all NaN)
    df = df.dropna(how="all")
    logger.info(f"Rows after dropping all-NaN rows: {len(df)}")

    # Deduplicate
    dedup_columns = config.get("dedup_columns")
    before_dedup = len(df)
    if dedup_columns:
        df = df.drop_duplicates(subset=dedup_columns, keep="first", ignore_index=True)
        logger.info(f"Deduplicated on columns {dedup_columns}: {before_dedup} -> {len(df)} rows.")
    else:
        df = df.drop_duplicates(keep="first", ignore_index=True)
        logger.info(f"Deduplicated on all columns: {before_dedup} -> {len(df)} rows.")
    stats["rows_after_dedup"] = len(df)

    # Final column list
    stats["columns_after_cleaning"] = df.columns.tolist()

    # Log summary statistics
    logger.info(f"Cleaning complete. Stats: {stats}")

    return df
