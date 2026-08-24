"""Standardized ChatML output schema and converter utilities for dataset adapters."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# Extended task types beyond MentalHealthTaskType enum
TASK_TYPES = {
    "symptom_classification",
    "severity_estimation",
    "therapy_response_generation",
    "risk_assessment",
    "empathy_scoring",
    "dpo_preference",
    "adversarial_safety",
}

VALID_ROLES = {"system", "user", "assistant"}
VALID_LINGUISTIC_STYLES = {"formal", "informal", "mixed"}


def csv_to_chatml(
    csv_path: Path,
    user_column: str,
    assistant_column: str,
    *,
    system_prompt: str | None = None,
    source: str,
    task_type: str = "therapy_response_generation",
    diagnostic_tag: str | None = None,
    demographic_tags: list[str] | None = None,
    linguistic_style: str = "mixed",
    extra_fields: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert a CSV file to ChatML-formatted records.

    Args:
        csv_path: Path to the CSV file.
        user_column: Column name for user/seeker utterance.
        assistant_column: Column name for assistant/therapist utterance.
        system_prompt: Optional system message content.
        source: Source dataset name.
        task_type: Task type from TASK_TYPES.
        diagnostic_tag: Optional diagnostic label.
        demographic_tags: Optional demographic tags.
        linguistic_style: formal, informal, or mixed.
        extra_fields: Additional metadata fields to include.

    Returns:
        List of ChatML-formatted dictionaries.
    """
    records: list[dict[str, Any]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_content = (row.get(user_column) or "").strip()
            assistant_content = (row.get(assistant_column) or "").strip()
            if not user_content or not assistant_content:
                continue

            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": assistant_content})

            record: dict[str, Any] = {
                "messages": messages,
                "source": source,
                "task_type": task_type,
                "diagnostic_tag": diagnostic_tag,
                "demographic_tags": demographic_tags or [],
                "linguistic_style": linguistic_style,
                "clinical_reviewed": False,
            }
            if extra_fields:
                record.update(extra_fields)
            records.append(record)
    return records


def json_conversation_to_chatml(
    conversations: list[dict[str, Any]],
    *,
    speaker_key: str = "speaker",
    utterance_key: str = "utterance",
    user_role: str = "seeker",
    assistant_role: str = "supporter",
    system_prompt: str | None = None,
    source: str,
    task_type: str = "therapy_response_generation",
    diagnostic_tag: str | None = None,
    demographic_tags: list[str] | None = None,
    linguistic_style: str = "mixed",
    extra_fields: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert JSON conversation turns to ChatML records.

    Args:
        conversations: List of conversation dicts, each containing a list of turns.
        speaker_key: Key for speaker identification in turns.
        utterance_key: Key for utterance text in turns.
        user_role: Speaker value that maps to "user" role.
        assistant_role: Speaker value that maps to "assistant" role.
        system_prompt: Optional system message.
        source: Source dataset name.
        task_type: Task type from TASK_TYPES.
        diagnostic_tag: Optional diagnostic label.
        demographic_tags: Optional demographic tags.
        linguistic_style: formal, informal, or mixed.
        extra_fields: Additional metadata to merge.

    Returns:
        List of ChatML records.
    """
    records: list[dict[str, Any]] = []
    for conv in conversations:
        turns = conv.get("dialog") or conv.get("turns") or conv.get("conversation") or []
        if not turns:
            continue

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for turn in turns:
            speaker = str(turn.get(speaker_key, "")).lower()
            utterance = (turn.get(utterance_key) or turn.get("text") or "").strip()
            if not utterance:
                continue
            role = "user" if speaker == user_role else "assistant"
            messages.append({"role": role, "content": utterance})

        if len(messages) < 2:
            continue

        record: dict[str, Any] = {
            "messages": messages,
            "source": source,
            "task_type": task_type,
            "diagnostic_tag": diagnostic_tag,
            "demographic_tags": demographic_tags or [],
            "linguistic_style": linguistic_style,
            "clinical_reviewed": False,
        }
        if extra_fields:
            record.update(extra_fields)
        records.append(record)
    return records


def sharegpt_to_chatml(
    conversations: list[dict[str, Any]],
    *,
    turns_key: str = "conversations",
    from_key: str = "from",
    value_key: str = "value",
    user_role: str = "human",
    assistant_role: str = "gpt",
    system_prompt: str | None = None,
    source: str,
    task_type: str = "therapy_response_generation",
    diagnostic_tag: str | None = None,
    demographic_tags: list[str] | None = None,
    linguistic_style: str = "mixed",
    extra_fields: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert ShareGPT-format conversations to ChatML records.

    Args:
        conversations: List of ShareGPT-format conversation dicts.
        turns_key: Key containing the list of turns.
        from_key: Key for speaker identification ("from").
        value_key: Key for utterance text ("value").
        user_role: Value that maps to "user" (default "human").
        assistant_role: Value that maps to "assistant" (default "gpt").
        system_prompt: Optional system message.
        source: Source dataset name.
        task_type: Task type from TASK_TYPES.
        diagnostic_tag: Optional diagnostic label.
        demographic_tags: Optional demographic tags.
        linguistic_style: formal, informal, or mixed.
        extra_fields: Additional metadata to merge.

    Returns:
        List of ChatML records.
    """
    records: list[dict[str, Any]] = []
    for conv in conversations:
        turns = conv.get(turns_key, [])
        if not turns:
            continue

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for turn in turns:
            speaker = str(turn.get(from_key, "")).lower()
            utterance = (turn.get(value_key) or "").strip()
            if not utterance:
                continue
            role = "user" if speaker == user_role else "assistant"
            messages.append({"role": role, "content": utterance})

        if len(messages) < 2:
            continue

        record: dict[str, Any] = {
            "messages": messages,
            "source": source,
            "task_type": task_type,
            "diagnostic_tag": diagnostic_tag,
            "demographic_tags": demographic_tags or [],
            "linguistic_style": linguistic_style,
            "clinical_reviewed": False,
        }
        if extra_fields:
            record.update(extra_fields)
        records.append(record)
    return records


def save_jsonl(records: list[dict[str, Any]], output_path: Path) -> int:
    """Save records as JSONL. Returns count of records written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)
