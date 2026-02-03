"""
Pixelated Empathy - Dataset Cleaning & Deduplication Pipeline

This module provides a modular, testable function to clean, normalize, deduplicate,
and privacy-sanitize loaded datasets for supervised fine-tuning. All configs are
parameterized or loaded from secure config. No secrets are hardcoded.

Author: Pixelated Empathy AI Team
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
from ai.pipelines.orchestrator.processing.nvidia_clients import NemoCuratorClient

# Expose logger for test mocking
logger = logging.getLogger("dataset_cleaning")

# Default PII column patterns (can be overridden via config)
DEFAULT_PII_PATTERNS = [
    "email",
    "phone",
    "ssn",
    "name",
    "address",
    "dob",
    "birth",
    "contact",
    "pii",
]

USE_NVIDIA_CURATOR = os.getenv("USE_NVIDIA_CURATOR", "false").lower() == "true"


def setup_logger(log_level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger for the cleaning pipeline.
    """
    logger_obj = logging.getLogger("dataset_cleaning")
    if not logger_obj.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger_obj.addHandler(handler)
    logger_obj.setLevel(log_level)
    return logger_obj


def find_pii_columns(
    columns: List[str],
    pii_patterns: Optional[List[str]] = None,
    explicit_pii: Optional[Set[str]] = None,
) -> Set[str]:
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


def normalize_text_columns(
    df: pd.DataFrame, text_columns: Optional[List[str]] = None
) -> pd.DataFrame:
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


def redact_pii_in_text_fields(
    df: pd.DataFrame,
    text_columns: Optional[List[str]] = None,
    logger_obj: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Redacts SSN-like patterns in text columns for privacy compliance.
    Emits an info-level privacy audit log if any redaction occurs.
    """
    if text_columns is None:
        text_columns = df.select_dtypes(include=["object"]).columns.tolist()
    ssn_pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    lgr = logger_obj or logger
    for col in text_columns:

        def redact_and_log(x):
            redacted = ssn_pattern.sub("[REDACTED-SSN]", str(x))
            if redacted != str(x):
                lgr.info(f"privacy audit: redacted PII in field '{col}'")
            return redacted

        df[col] = df[col].astype(str).apply(redact_and_log)
    return df


def remove_pii(
    df: pd.DataFrame, pii_columns: Set[str], logger_obj: logging.Logger
) -> pd.DataFrame:
    """
    Removes PII columns from the DataFrame.
    """
    existing_pii = [col for col in pii_columns if col in df.columns]
    if existing_pii:
        logger_obj.info(f"Removing PII columns: {existing_pii}")
        df = df.drop(columns=existing_pii)
    return df


def clean_and_deduplicate(
    datasets: Union[pd.DataFrame, List[pd.DataFrame]],
    config: Optional[Dict[str, Any]] = None,
    logger_obj: Optional[logging.Logger] = None,
    audit_log: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Cleans, normalizes, deduplicates, and privacy-sanitizes all records across datasets.
    """
    # Accept both a single DataFrame or a list
    if isinstance(datasets, pd.DataFrame):
        datasets = [datasets]

    if not datasets or not all(isinstance(df, pd.DataFrame) for df in datasets):
        raise ValueError("Input must be a non-empty list of pandas DataFrames.")

    config = config or {}
    lgr = logger_obj or logger

    # If NeMo Curator is enabled, offload high-performance cleaning
    if USE_NVIDIA_CURATOR:
        lgr.info("Offloading cleaning to NVIDIA NeMo Curator microservice.")
        try:
            client = NemoCuratorClient()
            temp_path = "/workspace/datasets/temp_curate.jsonl"
            lgr.info(
                f"Triggering therapeutic curation for {temp_path} via {client.base_url}"
            )
            # Use the project-specific therapeutic curation method
            client.curate_therapeutic_data(temp_path)
            lgr.info(
                "NeMo Curator: Therapeutic alignment and crisis safety check triggered."
            )
        except Exception as e:
            lgr.error(f"NeMo Curator offloading failed: {e}. Falling back to local.")

    # Concatenate all datasets
    df = pd.concat(datasets, ignore_index=True)

    # Identify and remove PII columns
    pii_patterns = config.get("pii_patterns")
    explicit_pii = set(config.get("explicit_pii", []))
    pii_columns = find_pii_columns(df.columns.tolist(), pii_patterns, explicit_pii)
    df = remove_pii(df, pii_columns, lgr)

    # Redact PII in text fields
    text_columns = config.get("text_columns")
    df = redact_pii_in_text_fields(df, text_columns, lgr)

    # Normalize text columns
    df = normalize_text_columns(df, text_columns)

    # Drop rows with all NaN
    df = df.dropna(how="all")

    # Deduplicate
    dedup_cols = config.get("dedup_columns")
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep="first", ignore_index=True)
    else:
        df = df.drop_duplicates(keep="first", ignore_index=True)

    # Log summary
    if audit_log is not None:
        audit_log.append(
            {
                "event": "cleaned",
                "rows": len(df),
                "columns": list(df.columns),
            }
        )

    return df
