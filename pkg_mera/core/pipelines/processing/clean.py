"""Data cleaning helpers for CSV/JSONL pipelines.

This module centralizes frequently used cleaning operations including:
- PII column discovery
- Text normalization
- Regex-based redaction for obvious PII patterns in text fields
- Deduplication across multiple frames
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

_PII_NAME_PATTERNS = (
    r"ssn",
    r"social",
    r"phone",
    r"telephone",
    r"mobile",
    r"email",
    r"name",
    r"address",
    r"dob",
    r"birth",
)

# Basic regexes for regex-backed redaction
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_PATTERN = re.compile(r"\b(?:\+\d{1,3}[-. ]?)?(?:\(?\d{3}\)?[-. ]?)?\d{3}[-. ]?\d{4}\b")
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


def _is_text_series(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype)


def find_pii_columns(columns: Iterable[str] | pd.Index) -> set[str]:
    """Identify likely PII-bearing columns by name heuristics."""

    pii_columns: set[str] = set()
    for column in columns:
        lower = str(column).strip().lower()
        if any(re.search(pattern, lower) for pattern in _PII_NAME_PATTERNS):
            pii_columns.add(str(column))
    return pii_columns


def normalize_text_columns(frame: pd.DataFrame, text_columns: Iterable[str]) -> pd.DataFrame:
    """Trim, collapse whitespace and normalize case on specified text columns."""

    output = frame.copy()
    for column in text_columns:
        if column not in output.columns:
            continue
        # Convert to string for safety and normalize punctuation spacing
        output[column] = output[column].astype(str).str.strip().str.lower()
        output[column] = output[column].str.replace(r"\s+", " ", regex=True)
        output[column] = output[column].str.replace("\t", " ", regex=False)
        output[column] = output[column].str.replace("\n", " ", regex=False)
        output[column] = output[column].str.strip()
    return output


def redact_pii_in_text_fields(
    frame: pd.DataFrame,
    text_columns: Iterable[str],
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Regex-redact common PII patterns in selected text columns."""

    log = logger or logging.getLogger(__name__)
    output = frame.copy()

    for column in text_columns:
        if column not in output.columns:
            continue
        if not _is_text_series(output[column]):
            continue

        series = output[column].astype(str).copy()
        masked = series.str.replace(_SSN_PATTERN, "[REDACTED-SSN]", regex=True)
        masked = masked.str.replace(_PHONE_PATTERN, "[REDACTED-PHONE]", regex=True)
        masked = masked.str.replace(_EMAIL_PATTERN, "[REDACTED-EMAIL]", regex=True)

        replaced_count = int((series != masked).sum())
        if replaced_count:
            log.debug("Redacted PII-like tokens in %s (%s fields)", replaced_count, column)
        output[column] = masked

    return output


def remove_pii(
    frame: pd.DataFrame,
    pii_columns: set[str] | Iterable[str],
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Drop PII columns identified upstream."""

    log = logger or logging.getLogger(__name__)
    remove = set(pii_columns)
    available = [c for c in frame.columns if c in remove]
    if available:
        log.debug("Dropping PII columns: %s", sorted(available))
    return frame.drop(columns=available)


def _ensure_required_columns(frame: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return frame


def clean_and_deduplicate(
    dataframes: list[pd.DataFrame] | pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """End-to-end cleaning and deduplication helper.

    Args:
        dataframes: One or more pandas DataFrames.
        config: Optional config dict.
            - dedup_columns: list[str], default ["text"]
            - required_columns: optional list[str]
            - additional_text_columns: optional text fields
            - force_lower: bool, default True
    """

    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]

    log = logger or logging.getLogger(__name__)
    cfg = dict(config or {})

    if not dataframes:
        return pd.DataFrame()

    dedup_columns = cfg.get("dedup_columns", ["text"])
    required_columns = cfg.get("required_columns", dedup_columns)
    additional_text_columns = set(cfg.get("additional_text_columns", []))
    remove_pii_columns: bool = bool(cfg.get("remove_pii_columns", True))

    working = pd.concat([df.copy() for df in dataframes], ignore_index=True)
    _ensure_required_columns(working, list(required_columns))

    # Normalize and normalize text fields first.
    text_columns = set(dedup_columns) | set(additional_text_columns)
    text_columns = {c for c in text_columns if c in working.columns}

    if cfg.get("force_lower", True):
        working = normalize_text_columns(working, text_columns)

    # Redact sensitive text patterns from all free-text fields.
    working = redact_pii_in_text_fields(working, text_columns, logger=log)

    # Remove PII columns by name heuristics.
    if remove_pii_columns:
        pii_cols = find_pii_columns(working.columns)
        required_set = set(required_columns)
        # Required columns that match PII patterns can't be dropped,
        # but MUST be redacted to prevent PII leakage.
        required_pii_cols = pii_cols & required_set
        if required_pii_cols:
            working = redact_pii_in_text_fields(working, required_pii_cols, logger=log)
            log.warning(
                "Required columns %s match PII patterns; applied regex redaction instead of column removal",
                sorted(required_pii_cols),
            )
        # Drop non-required PII columns.
        droppable = pii_cols.difference(required_set)
        working = remove_pii(working, droppable, logger=log)

    # Remove duplicate rows on dedup columns while preserving order.
    normalized_dedup = dedup_columns if dedup_columns else working.columns.tolist()
    valid_dedup = [c for c in normalized_dedup if c in working.columns]
    if valid_dedup:
        working = working.drop_duplicates(subset=valid_dedup, keep="first")

    working = working.reset_index(drop=True)
    return working


def clean(*frames: pd.DataFrame, config: dict[str, Any] | None = None, **kwargs: Any) -> pd.DataFrame:
    """Compatibility wrapper around :func:`clean_and_deduplicate`."""

    logger = kwargs.get("logger")
    if not frames:
        return pd.DataFrame()
    return clean_and_deduplicate(list(frames), config=config, logger=logger)


__all__ = [
    "clean",
    "clean_and_deduplicate",
    "find_pii_columns",
    "normalize_text_columns",
    "redact_pii_in_text_fields",
    "remove_pii",
]
