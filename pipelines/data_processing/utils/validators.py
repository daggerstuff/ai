"""Validation utilities for adapter output records."""

from __future__ import annotations

from typing import Any

from ai.pipelines.data_processing.utils.converters import TASK_TYPES, VALID_LINGUISTIC_STYLES


def validate_record(record: dict[str, Any]) -> bool:
    """Check if a record meets minimum validity criteria.

    Returns True if valid, False otherwise.
    """
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False

    for msg in messages:
        if not isinstance(msg, dict):
            return False
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("system", "user", "assistant"):
            return False
        if not isinstance(content, str) or not content.strip():
            return False

    # Must have at least one user and one assistant message
    roles = {msg["role"] for msg in messages}
    if "user" not in roles or "assistant" not in roles:
        return False

    # Validate task_type if present
    task_type = record.get("task_type")
    if task_type is not None and task_type not in TASK_TYPES:
        return False

    # Validate linguistic_style if present
    style = record.get("linguistic_style")
    if style is not None and style not in VALID_LINGUISTIC_STYLES:
        return False

    # Validate clinical_reviewed is bool
    clinical_reviewed = record.get("clinical_reviewed")
    if clinical_reviewed is not None and not isinstance(clinical_reviewed, bool):
        return False

    # Validate demographic_tags is list
    demo_tags = record.get("demographic_tags")
    if demo_tags is not None and not isinstance(demo_tags, list):
        return False

    return True


def filter_valid(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter a list of records, returning only valid ones."""
    return [r for r in records if validate_record(r)]
