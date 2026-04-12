"""
PIX-32: Data Normalizer for Pixelated Empathy AI Dataset Pipeline.

Bridges the canonical JSONL schema (PIX-30) with the Conversation dataclass.
Provides text normalization, key standardization, schema validation, and
provenance metadata attachment for all ingested records.

Canonical JSONL schema fields:
  id, source, license, license_verified, content_type, messages/text,
  metadata (title, authors, doi, topic_tags, therapeutic_modality, quality_score),
  phi_scan_passed, phi_scan_date, pull_date, pix_ticket
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Required fields for the PIX-30 canonical JSONL schema
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: frozenset[str] = frozenset({"id", "source", "content_type"})

# Fields that must appear in every record's metadata dict
REQUIRED_METADATA_FIELDS: frozenset[str] = frozenset(
    {
        "topic_tags",
        "therapeutic_modality",
        "quality_score",
    }
)


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of validating a single JSONL record."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalization result
# ---------------------------------------------------------------------------


@dataclass
class NormalizationResult:
    """Aggregated result of a normalization pass."""

    total_records: int = 0
    valid_records: int = 0
    rejected_records: int = 0
    rejected_reasons: dict[str, int] = field(default_factory=dict)
    normalization_errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DataNormalizer
# ---------------------------------------------------------------------------


class DataNormalizer:
    """
    Normalizes raw JSONL records into the pipeline's Conversation dataclass.

    Pipeline:
      1. Validate required fields exist
      2. Normalize text content (unicode, whitespace)
      3. Standardize dictionary keys to lower_snake_case
      4. Convert to Conversation dataclass
      5. Attach provenance metadata
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        enforce_license: bool = False,
        enforce_phi_scan: bool = False,
    ) -> None:
        """
        Args:
            similarity_threshold: Threshold for deduplication similarity.
            enforce_license: If True, reject records without a license field.
            enforce_phi_scan: If True, reject records without phi_scan_passed.
        """
        self.similarity_threshold = similarity_threshold
        self.enforce_license = enforce_license
        self.enforce_phi_scan = enforce_phi_scan

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_record(self, record: dict[str, Any]) -> ValidationResult:
        """Validate a single JSONL record against the canonical schema."""
        errors: list[str] = []
        warnings: list[str] = []

        # Check required top-level fields
        missing = REQUIRED_FIELDS - set(record.keys())
        if missing:
            errors.append(f"Missing required fields: {sorted(missing)}")

        # Check 'messages' or 'text' — at least one must exist
        has_messages = (
            isinstance(record.get("messages"), list) and len(record["messages"]) > 0
        )
        has_text = (
            isinstance(record.get("text"), str) and len(record["text"].strip()) > 0
        )
        if not has_messages and not has_text:
            errors.append(
                "Record must contain either 'messages' (list) or 'text' (str)"
            )

        # Check metadata structure
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            errors.append("'metadata' must be a dict")
        else:
            missing_meta = REQUIRED_METADATA_FIELDS - set(metadata.keys())
            if missing_meta:
                warnings.append(
                    f"Missing optional metadata fields: {sorted(missing_meta)}"
                )

        # License enforcement
        if self.enforce_license and not record.get("license"):
            errors.append("License enforcement enabled but no license provided")

        # PHI scan enforcement
        if self.enforce_phi_scan and "phi_scan_passed" not in record:
            errors.append("PHI scan enforcement enabled but phi_scan_passed not set")

        return ValidationResult(
            valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    def normalize_text(self, text: str) -> str:
        """Normalize unicode characters and whitespace in a string."""
        if not text or not isinstance(text, str):
            return text
        normalized = unicodedata.normalize("NFKC", text)
        normalized = " ".join(normalized.split())
        return normalized.strip()

    def standardize_keys(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert all dict keys to lower_snake_case recursively."""
        result: dict[str, Any] = {}
        for key, value in data.items():
            clean_key = self._to_snake_case(key)
            if isinstance(value, dict):
                result[clean_key] = self.standardize_keys(value)
            elif isinstance(value, str):
                result[clean_key] = self.normalize_text(value)
            else:
                result[clean_key] = value
        return result

    def normalize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize a single JSONL record: standardize keys, normalize text,
        ensure canonical schema structure.
        """
        normalized = self.standardize_keys(record)

        # Ensure messages field exists (convert text → messages if needed)
        if "messages" not in normalized and "text" in normalized:
            normalized["messages"] = [
                {"role": "user", "content": normalized.pop("text")}
            ]

        # Normalize each message's content
        if isinstance(normalized.get("messages"), list):
            normalized["messages"] = [
                self._normalize_message(msg) for msg in normalized["messages"]
            ]

        # Normalize metadata
        if isinstance(normalized.get("metadata"), dict):
            normalized["metadata"] = self.standardize_keys(normalized["metadata"])

        return normalized

    def record_to_conversation(self, record: dict[str, Any]) -> Conversation:
        """
        Convert a normalized JSONL record into a Conversation dataclass instance.

        Maps PIX-30 canonical fields to Conversation schema:
          - id → conversation_id
          - source → source
          - messages/messages → messages (Message list)
          - metadata + provenance fields → metadata
        """
        # Import here to avoid circular dependency with ai.core.pipelines
        # The Conversation class is defined locally below
        messages: list[Message] = []
        raw_messages = record.get("messages", [])

        if isinstance(raw_messages, list):
            for msg in raw_messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", msg.get("text", ""))
                    msg_metadata = msg.get("metadata", {})
                    timestamp = msg.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    )
                    messages.append(
                        Message(
                            role=self.normalize_text(str(role)),
                            content=self.normalize_text(str(content)),
                            timestamp=timestamp,
                            metadata=msg_metadata
                            if isinstance(msg_metadata, dict)
                            else {},
                        )
                    )

        # Build provenance metadata
        provenance: dict[str, Any] = {}
        for provenance_key in (
            "license",
            "license_verified",
            "content_type",
            "phi_scan_passed",
            "phi_scan_date",
            "pull_date",
            "pix_ticket",
        ):
            if provenance_key in record:
                provenance[provenance_key] = record[provenance_key]

        # Merge user metadata with provenance
        user_metadata = record.get("metadata", {})
        if isinstance(user_metadata, dict):
            provenance.update(user_metadata)

        conversation_id = str(
            record.get("id", hashlib.sha256(repr(record).encode()).hexdigest()[:16])
        )

        return Conversation(
            conversation_id=conversation_id,
            source=self.normalize_text(str(record.get("source", "unknown"))),
            messages=messages,
            metadata=provenance,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def process_jsonl_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        reject_path: str | Path | None = None,
    ) -> NormalizationResult:
        """
        Process an entire JSONL file: validate, normalize, convert.

        Args:
            input_path: Path to input JSONL file.
            output_path: Path for normalized output JSONL. Defaults to
                input_path with .normalized.jsonl suffix.
            reject_path: Path for rejected records JSONL. Defaults to
                input_path with .rejected.jsonl suffix.

        Returns:
            NormalizationResult with counts and rejection reasons.
        """
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.with_suffix(".normalized.jsonl")
        if reject_path is None:
            reject_path = input_path.with_suffix(".rejected.jsonl")

        result = NormalizationResult()


        with (
            input_path.open("r", encoding="utf-8") as infile,
            Path(output_path).open("w", encoding="utf-8") as outfile,
            Path(reject_path).open("w", encoding="utf-8") as rejfile,
        ):
            for line_num, line in enumerate(infile, start=1):
                line = line.strip()
                if not line:
                    continue

                result.total_records += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    result.rejected_records += 1
                    reason = f"JSON parse error line {line_num}: {exc}"
                    result.rejected_reasons["json_parse_error"] = (
                        result.rejected_reasons.get("json_parse_error", 0) + 1
                    )
                    result.normalization_errors.append(reason)
                    rejfile.write(
                        json.dumps(
                            {
                                "line": line_num,
                                "raw": line[:500],
                                "error": str(exc),
                            }
                        )
                        + "\n"
                    )
                    continue

                validation = self.validate_record(record)
                if not validation.valid:
                    result.rejected_records += 1
                    for error in validation.errors:
                        reason_key = (
                            error.split(":")[0].strip().replace(" ", "_").lower()
                        )
                        result.rejected_reasons[reason_key] = (
                            result.rejected_reasons.get(reason_key, 0) + 1
                        )
                    rejfile.write(
                        json.dumps(
                            {
                                "line": line_num,
                                "record_id": record.get("id", "unknown"),
                                "errors": validation.errors,
                                "warnings": validation.warnings,
                            }
                        )
                        + "\n"
                    )
                    continue

                try:
                    normalized = self.normalize_record(record)
                    conversation = self.record_to_conversation(normalized)
                    outfile.write(
                        json.dumps(conversation.to_dict(), ensure_ascii=False) + "\n"
                    )
                    result.valid_records += 1
                except Exception as exc:
                    result.rejected_records += 1
                    reason = f"Normalization error line {line_num}: {exc}"
                    result.rejected_reasons["normalization_error"] = (
                        result.rejected_reasons.get("normalization_error", 0) + 1
                    )
                    result.normalization_errors.append(reason)
                    rejfile.write(
                        json.dumps(
                            {
                                "line": line_num,
                                "record_id": record.get("id", "unknown"),
                                "error": str(exc),
                            }
                        )
                        + "\n"
                    )

        logger.info(
            "Processed %s: %d total, %d valid, %d rejected",
            input_path.name,
            result.total_records,
            result.valid_records,
            result.rejected_records,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_snake_case(key: str) -> str:
        """Convert a string key to lower_snake_case."""
        key = key.strip().lower()
        # Replace spaces and hyphens with underscores
        key = re.sub(r"[\s\-]+", "_", key)
        # Insert underscore before uppercase letters (camelCase → camel_case)
        key = re.sub(r"(?<!^)(?=[A-Z])", "_", key)
        # Collapse multiple underscores
        return re.sub(r"_+", "_", key)

    def _normalize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Normalize a single message dict."""
        if not isinstance(message, dict):
            return {"role": "user", "content": str(message)}
        result = {}
        for key, value in message.items():
            clean_key = self._to_snake_case(key)
            if isinstance(value, str):
                result[clean_key] = self.normalize_text(value)
            else:
                result[clean_key] = value
        # Ensure required message fields
        result.setdefault("role", "user")
        result.setdefault("content", "")
        return result


# ---------------------------------------------------------------------------
# Conversation and Message dataclasses (local copy for ai.core.pipelines)
# Mirrors ai/pipelines/orchestrator/schemas/conversation_schema.py
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """Represents a single message within a conversation."""

    role: str
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the message to a dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass(kw_only=True)
class Conversation:
    """
    Represents a complete conversation adhering to the unified schema.
    """

    conversation_id: str = field(
        default_factory=lambda: hashlib.sha256(
            datetime.now(timezone.utc).isoformat().encode()
        ).hexdigest()[:16]
    )
    source: str | None = None
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Adds a message to the conversation."""
        message = Message(role=role, content=content, **kwargs)
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    @property
    def id(self) -> str:
        return self.conversation_id

    @id.setter
    def id(self, value: str) -> None:
        self.conversation_id = value

    @property
    def meta(self) -> dict[str, Any]:
        return self.metadata

    @meta.setter
    def meta(self, value: dict[str, Any]) -> None:
        self.metadata = value

    def to_dict(self) -> dict[str, Any]:
        """Serializes the conversation to a dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "source": self.source,
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Conversation:
        """Creates a Conversation instance from a dictionary."""
        messages = [Message(**msg_data) for msg_data in data.get("messages", [])]
        return cls(
            conversation_id=data.get(
                "conversation_id",
                hashlib.sha256(
                    datetime.now(timezone.utc).isoformat().encode()
                ).hexdigest()[:16],
            ),
            source=data.get("source"),
            messages=messages,
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
        )


__all__ = [
    "REQUIRED_FIELDS",
    "REQUIRED_METADATA_FIELDS",
    "Conversation",
    "DataNormalizer",
    "Message",
    "NormalizationResult",
    "ValidationResult",
]
