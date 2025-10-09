"""
convert_chatml.py

Modular pipeline component for converting unified datasets to ChatML format
optimized for Wayfarer-2-12B supervised fine-tuning.

- Accepts cleaned, unified dataset (pd.DataFrame or list of dicts)
- Converts all records to ChatML format (role, content, metadata)
- Handles edge cases, missing data, and ensures format compliance
- Logs conversion actions and statistics
- Returns list or DataFrame of ChatML-formatted records, ready for tokenization
- No hardcoded secrets; all configs via parameters or secure config
- Robust error handling and logging
- File <500 lines, PEP8, TDD-ready

Author: Pixelated Empathy AI Team
"""

import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

# ChatML roles allowed for Wayfarer-2-12B
CHATML_ROLES = {"system", "user", "assistant", "function"}

def get_logger(name: str = "convert_chatml") -> logging.Logger:
    """Configure and return a logger for the module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger()

def _validate_and_normalize_record(
    record: Mapping[Any, Any],
    default_role: str = "user",
    default_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Validate and normalize a single record to ChatML format.

    Returns None if the record is invalid or missing required fields.
    """
    if default_metadata is None:
        default_metadata = {}

    # Required fields: role, content
    role = record.get("role", default_role)
    content = record.get("content", "")

    # Edge case: missing content or role
    if not isinstance(content, str) or not content.strip():
        logger.warning("Skipping record with missing or empty content: %s", record)
        return None

    if not isinstance(role, str) or role.lower() not in CHATML_ROLES:
        logger.warning(
            "Invalid or missing role '%s' in record, defaulting to '%s'. Record: %s",
            role, default_role, record
        )
        role = default_role

    # Metadata: merge provided metadata with defaults
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        logger.warning("Metadata is not a dict, using empty metadata. Record: %s", record)
        metadata = {}

    # Merge with default_metadata, user-provided takes precedence
    merged_metadata = {**default_metadata, **metadata}

    # Optionally, add message_id or conversation_id if present
    for key in ("message_id", "conversation_id"):
        if key in record and key not in merged_metadata:
            merged_metadata[key] = record[key]

    return {
        "role": role.lower(),
        "content": content,
        "metadata": merged_metadata,
    }

def convert_to_chatml(
    dataset: pd.DataFrame | list[dict[str, Any]],
    default_role: str = "user",
    default_metadata: dict[str, Any] | None = None,
    log_stats: bool = True,
) -> list[dict[str, Any]]:
    """
    Convert a cleaned, unified dataset to ChatML format for Wayfarer-2-12B.

    Args:
        dataset: Input dataset (pandas DataFrame or list of dicts)
        default_role: Default role to use if missing/invalid
        default_metadata: Default metadata to merge into each record
        log_stats: Whether to log conversion statistics

    Returns:
        List of ChatML-formatted records (dicts with 'role', 'content', 'metadata')
    """
    logger.info("Starting ChatML conversion. Input type: %s", type(dataset).__name__)
    if default_metadata is None:
        default_metadata = {}

    # Minimal TDD: If DataFrame with 'prompt' and 'response', convert to alternating user/assistant messages
    if isinstance(dataset, pd.DataFrame):
        if not all(col in dataset.columns for col in ("prompt", "response")):
            raise ValueError("Input DataFrame must contain 'prompt' and 'response' columns.")
        # Privacy redaction integration (calls redact_pii_in_text_fields if available)
        try:
            from ai.dataset_pipeline.clean import redact_pii_in_text_fields
            redacted, _ = redact_pii_in_text_fields(dataset)
            # Only use the redacted DataFrame if it is a DataFrame and has the same shape
            if isinstance(redacted, pd.DataFrame) and redacted.shape == dataset.shape:
                dataset = redacted
        except ImportError:
            pass  # If not available, skip redaction
        chatml_records = []
        for _, row in dataset.iterrows():
            prompt = row.get("prompt")
            response = row.get("response")
            # Only add if both are present, are strings, and non-empty after stripping
            if isinstance(prompt, str) and prompt.strip() and isinstance(response, str) and response.strip():
                chatml_records.append({"role": "user", "content": prompt})
                chatml_records.append({"role": "assistant", "content": response})
            # If either is missing, skip the row (malformed row handling)
        return chatml_records

    # Fallback to original logic for other types (not used in current tests)
    if isinstance(dataset, list):
        records = dataset
    else:
        logger.error("Unsupported dataset type: %s", type(dataset))
        raise TypeError("Dataset must be a pandas DataFrame or list of dicts.")

    chatml_records = []
    skipped = 0
    for idx, record in enumerate(records):
        if not all(isinstance(k, str) for k in record):
            record = {str(k): v for k, v in record.items()}
        try:
            chatml_record = _validate_and_normalize_record(
                record, default_role=default_role, default_metadata=default_metadata
            )
            if chatml_record is not None:
                chatml_records.append(chatml_record)
            else:
                skipped += 1
        except Exception as e:
            logger.error(
                "Error converting record at index %d: %s. Record: %s", idx, str(e), record
            )
            skipped += 1

    if log_stats:
        logger.info(
            "ChatML conversion complete. Total: %d, Converted: %d, Skipped: %d",
            len(records), len(chatml_records), skipped
        )
        # Log role distribution
        role_counts: dict[str, int] = {}
        for rec in chatml_records:
            role = rec.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
        logger.info("Role distribution: %s", role_counts)

    return chatml_records

# Example usage (for TDD, not run on import)
if __name__ == "__main__":

    # Example input
    sample_data = [
        {"role": "user", "content": "Hello!", "metadata": {"lang": "en"}},
        {"role": "assistant", "content": "Hi, how can I help you?"},
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "", "metadata": {"lang": "en"}},  # Should be skipped
        {"role": "invalid", "content": "Fallback to default role."},
        {"content": "No role provided."},
        {"role": "user", "content": "Another message.", "message_id": "msg-123"},
    ]

    chatml = convert_to_chatml(sample_data)
