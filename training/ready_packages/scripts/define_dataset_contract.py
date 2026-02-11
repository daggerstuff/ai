#!/usr/bin/env python3
"""
Define Final Dataset Contract - Creates schema definitions and validation rules
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Note: jsonschema is optional - using basic validation instead

CONTENT_HASH_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
REQUIRED_METADATA_FIELDS = [
    "source_family",
    "source_key",
    "content_hash",
    "pii_status",
    "license_tag",
    "split",
    "phase",
    "provenance",
]

logger = logging.getLogger(__name__)


@dataclass
class ChatMLMessage:
    """Single ChatML message"""

    role: str  # system, user, assistant
    content: str


@dataclass
class ConversationMetadata:
    """Metadata for a conversation"""

    source_family: str
    source_key: str
    content_hash: str
    pii_status: str  # scrubbed, none_detected, requires_review
    license_tag: str
    split: str  # train, val, test
    phase: str  # stage1_foundation, stage2_therapeutic_expertise, etc.
    conversation_length: int | None = None
    total_tokens: int | None = None
    quality_score: float | None = None
    bias_flags: list[str] = field(default_factory=list)


@dataclass
class Provenance:
    """Provenance tracking for a conversation"""

    original_s3_key: str
    processing_pipeline: str
    processed_at: str
    dedup_status: str  # unique, duplicate_removed, near_duplicate_removed
    processing_steps: list[str] = field(default_factory=list)


@dataclass
class FinalDatasetConversation:
    """Complete conversation entry in final dataset"""

    messages: list[dict[str, str]]
    metadata: dict[str, Any]

    def to_chatml_jsonl(self) -> dict[str, Any]:
        """Convert to ChatML JSONL format"""
        return {
            "messages": self.messages,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinalDatasetConversation":
        """Create from dictionary"""
        return cls(
            messages=data["messages"],
            metadata=data["metadata"],
        )


# JSON Schema for validation
CHATML_CONVERSATION_SCHEMA = {
    "type": "object",
    "required": ["messages", "metadata"],
    "properties": {
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["role", "content"],
                "properties": {
                    "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                    "content": {"type": "string"},
                },
            },
        },
        "metadata": {
            "type": "object",
            "required": [
                "source_family",
                "source_key",
                "content_hash",
                "pii_status",
                "license_tag",
                "split",
                "phase",
                "provenance",
            ],
            "properties": {
                "source_family": {"type": "string"},
                "source_key": {"type": "string"},
                "content_hash": {"type": "string", "pattern": "^sha256:[a-f0-9]{64}$"},
                "pii_status": {
                    "type": "string",
                    "enum": ["scrubbed", "none_detected", "requires_review"],
                },
                "license_tag": {"type": "string"},
                "split": {"type": "string", "enum": ["train", "val", "test"]},
                "phase": {"type": "string"},
                "conversation_length": {"type": "integer"},
                "total_tokens": {"type": "integer"},
                "quality_score": {"type": "number", "minimum": 0, "maximum": 1},
                "bias_flags": {"type": "array", "items": {"type": "string"}},
                "provenance": {
                    "type": "object",
                    "required": [
                        "original_s3_key",
                        "processing_pipeline",
                        "processed_at",
                        "dedup_status",
                    ],
                    "properties": {
                        "original_s3_key": {"type": "string"},
                        "processing_pipeline": {"type": "string"},
                        "processed_at": {"type": "string", "format": "date-time"},
                        "dedup_status": {"type": "string"},
                        "processing_steps": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
    },
}

MANIFEST_SCHEMA = {
    "type": "object",
    "required": ["manifest_version", "generated_at", "total_conversations", "splits"],
    "properties": {
        "manifest_version": {"type": "string"},
        "generated_at": {"type": "string", "format": "date-time"},
        "total_conversations": {"type": "integer"},
        "total_tokens_approx": {"type": "integer"},
        "splits": {
            "type": "object",
            "properties": {
                "train": {"type": "object"},
                "val": {"type": "object"},
                "test": {"type": "object"},
            },
        },
        "source_families": {"type": "object"},
        "provenance_map": {"type": "object"},
        "holdout_families": {"type": "object"},
    },
}


def validate_messages(messages: Any) -> list[str]:
    errors: list[str] = []

    if messages is None:
        errors.append("Missing required field: messages")
        return errors

    if not isinstance(messages, list):
        errors.append("messages must be a list")
        return errors

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"messages[{i}] must be a dict")
            continue

        if "role" not in msg or "content" not in msg:
            errors.append(f"messages[{i}] missing required fields: role, content")
            continue

        if msg["role"] not in ["system", "user", "assistant"]:
            errors.append(f"messages[{i}].role must be one of: system, user, assistant")

    return errors


def validate_metadata(metadata: Any) -> list[str]:
    errors: list[str] = []

    if metadata is None:
        errors.append("Missing required field: metadata")
        return errors

    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
        return errors

    missing_required_fields = [
        f"metadata missing required field: {required_field}"
        for required_field in REQUIRED_METADATA_FIELDS
        if required_field not in metadata
    ]
    errors.extend(missing_required_fields)

    pii_status = metadata.get("pii_status")
    if pii_status is not None and pii_status not in [
        "scrubbed",
        "none_detected",
        "requires_review",
    ]:
        errors.append(
            "metadata.pii_status must be one of: scrubbed, none_detected, requires_review"
        )

    split = metadata.get("split")
    if split is not None and split not in ["train", "val", "test"]:
        errors.append("metadata.split must be one of: train, val, test")

    content_hash = metadata.get("content_hash")
    if content_hash is not None and not CONTENT_HASH_RE.match(str(content_hash)):
        errors.append("metadata.content_hash must match pattern: sha256:<64_hex_chars>")

    return errors


def validate_conversation(conversation: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate a conversation against the schema.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = validate_messages(conversation.get("messages")) + validate_metadata(
        conversation.get("metadata"),
    )

    return (not errors), errors


def compute_content_hash(messages: list[dict[str, str]]) -> str:
    """Compute SHA256 hash of normalized conversation content"""
    content_parts = [
        msg["content"].strip().lower()
        for msg in messages
        if isinstance(msg, dict) and isinstance(msg.get("content"), str)
    ]

    normalized = " ".join(sorted(content_parts))
    hash_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sha256:{hash_digest}"


def create_contract_definitions(output_dir: Path) -> None:
    """Create contract definition files"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save schemas
    schema_file = output_dir / "chatml_conversation_schema.json"
    with open(schema_file, "w", encoding="utf-8") as f:
        json.dump(CHATML_CONVERSATION_SCHEMA, f, indent=2)

    manifest_schema_file = output_dir / "manifest_schema.json"
    with open(manifest_schema_file, "w", encoding="utf-8") as f:
        json.dump(MANIFEST_SCHEMA, f, indent=2)

    # Save contract summary
    contract_summary = {
        "contract_version": "1.0",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "schemas": {
            "chatml_conversation": str(schema_file.relative_to(output_dir)),
            "manifest": str(manifest_schema_file.relative_to(output_dir)),
        },
        "required_fields": {
            "messages": ["role", "content"],
            "metadata": REQUIRED_METADATA_FIELDS,
        },
        "validation_rules": {
            "pii_status": ["scrubbed", "none_detected", "requires_review"],
            "split": ["train", "val", "test"],
            "content_hash_format": "sha256:<64_hex_chars>",
        },
        "holdout_families": [
            "long_running_therapy",
            "edge_case_crisis",
            "sarcasm",
            "voice_persona",
        ],
    }

    summary_file = output_dir / "contract_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(contract_summary, f, indent=2)

    logger.info("Contract definitions saved to %s", output_dir)
    logger.info("  - ChatML schema: %s", schema_file)
    logger.info("  - Manifest schema: %s", manifest_schema_file)
    logger.info("  - Contract summary: %s", summary_file)


def main() -> None:
    """Main entry point"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    project_root = Path(__file__).parents[3]
    output_dir = project_root / "ai" / "training_ready" / "data" / "contract_definitions"

    create_contract_definitions(output_dir)
    logger.info("✅ Contract definitions created successfully")


if __name__ == "__main__":
    main()
