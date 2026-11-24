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

import functools
import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ⚡ Bolt Optimization: Precompile regex patterns globally to avoid the overhead of implicit regex compilation
# or cache lookups on every execution, significantly speeding up dictionary key normalization.
_RE_DASH = re.compile(r"[\s\-]+")
_RE_CAMEL = re.compile(r"(?<!^)(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_RE_MULTI = re.compile(r"_+")


# ---------------------------------------------------------------------------
# Required fields for the PIX-30 canonical JSONL schema
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: frozenset[str] = frozenset({"id", "source", "content_type"})

OPTIONAL_METADATA: frozenset[str] = frozenset(
    {
        "title",
        "authors",
        "doi",
        "topic_tags",
        "therapeutic_modality",
        "quality_score",
    }
)


@dataclass(frozen=True)
class ValidationResult:
    """Represents the outcome of a single record validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class NormalizationSummary:
    """Aggregated statistics for a normalization run."""

    total_records: int = 0
    valid_records: int = 0
    rejected_records: int = 0
    rejected_reasons: dict[str, int] = field(default_factory=dict)


class DataNormalizer:
    """
    Handles normalization of diverse dataset formats into the canonical schema.

    Pipeline stages:
      1. Validate required fields exist
      2. Normalize text content (unicode, whitespace)
      3. Standardize dictionary keys to lower_snake_case
      4. Convert to Conversation dataclass
      5. Attach provenance metadata
    """

    # Version marker for backward compatibility tracking
    NORMALIZATION_VERSION: str = "2.0"

    def __init__(
        self,
        pix_ticket: str | None = None,
        *,
        enforce_license: bool = False,
        enforce_phi_scan: bool = False,
    ) -> None:
        self.pix_ticket = pix_ticket
        self.enforce_license = enforce_license
        self.enforce_phi_scan = enforce_phi_scan

    def validate_record(self, record: dict[str, Any]) -> ValidationResult:
        """Check if a raw record contains the minimum required fields."""
        errors = []
        warnings = []

        # Check top-level required fields
        for field_name in REQUIRED_FIELDS:
            if field_name not in record:
                errors.append(f"Missing required field: {field_name}")

        # Check for message content
        has_messages = isinstance(record.get("messages"), list) and len(record["messages"]) > 0
        has_text = isinstance(record.get("text"), str) and len(record["text"].strip()) > 0

        if not (has_messages or has_text):
            errors.append("Record must contain either 'messages' (list) or 'text' (str)")

        if self.enforce_license and not record.get("license"):
            errors.append("Missing required field: license")

        if self.enforce_phi_scan and record.get("phi_scan_passed") is not True:
            errors.append("PHI scan has not passed")

        # Check metadata
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            errors.append("Metadata must be a dictionary")
        else:
            missing_meta = OPTIONAL_METADATA - set(metadata.keys())
            if missing_meta:
                warnings.append(f"Missing optional metadata fields: {sorted(missing_meta)}")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def normalize_text(self, text: str) -> str:
        """Standardize string encoding and whitespace."""
        if not text:
            return ""

        # Normalize unicode characters to NFC form
        text = unicodedata.normalize("NFC", text)

        # Collapse multiple whitespaces and trim
        return re.sub(r"\s+", " ", text).strip()

    def normalize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize a single JSONL record: standardize keys, normalize text,
        ensure canonical schema structure.
        """
        # Create deep copy for mutation and canonicalize top-level keys first.
        normalized = {self._to_snake_case(str(key)): value for key, value in json.loads(json.dumps(record)).items()}

        # Convert 'text' field to 'messages' if needed
        if "text" in normalized and "messages" not in normalized:
            normalized["messages"] = [{"role": "user", "content": normalized.pop("text")}]

        # Standardize keys and normalize content
        if "messages" in normalized:
            if not isinstance(normalized["messages"], list):
                raise ValueError(f"Expected 'messages' to be a list, got {type(normalized['messages']).__name__}")
            normalized["messages"] = [self._normalize_message(msg) for msg in normalized["messages"]]

        if "metadata" in normalized and isinstance(normalized["metadata"], dict):
            normalized["metadata"] = {self._to_snake_case(k): v for k, v in normalized["metadata"].items()}

        return normalized

    def to_conversation(self, record: dict[str, Any], source_name: str) -> Conversation:
        """
        Map a normalized record to the Conversation dataclass.

        Mappings:
          - id → conversation_id
          - source → source
          - messages/messages → messages (Message list)
          - metadata + provenance fields → metadata
        """
        # Re-normalize to be safe
        if not isinstance(record, dict):
            raise ValueError(f"Expected dict record, got {type(record).__name__}")
        record = self.normalize_record(record)

        messages = []
        for msg in record.get("messages", []):
            if isinstance(msg, dict):
                content = msg.get("content", "")
                role = msg.get("role", "user")
                if content:
                    timestamp = msg.get("timestamp", datetime.now(UTC).isoformat())
                    msg_metadata = msg.get("metadata", {})
                    messages.append(
                        Message(
                            role=role,
                            content=content,
                            timestamp=timestamp,
                            metadata=msg_metadata if isinstance(msg_metadata, dict) else {},
                        )
                    )

        # Build provenance
        provenance = {
            "source_repo": source_name,
            "pull_date": record.get("pull_date", datetime.now(UTC).isoformat()),
            "pix_ticket": self.pix_ticket or record.get("pix_ticket", "N/A"),
            "license": record.get("license", "unknown"),
            "license_verified": record.get("license_verified", False),
            "phi_scan_passed": record.get("phi_scan_passed", False),
            "normalization_version": self.NORMALIZATION_VERSION,
        }

        # Combine existing metadata with provenance
        metadata = {**record.get("metadata", {}), **provenance}

        conversation_id = str(record.get("id", hashlib.sha256(repr(record).encode()).hexdigest()[:16]))

        return Conversation(
            conversation_id=conversation_id,
            source=source_name,
            messages=messages,
            metadata=metadata,
        )

    def process_file(
        self,
        input_path: Path,
        output_path: Path | None = None,
        reject_path: Path | None = None,
    ) -> NormalizationSummary:
        """
        Stream process a JSONL file, normalizing valid records and rejecting others.

        Args:
            input_path: Path to the source JSONL file.
            output_path: Path for normalized output JSONL. Defaults to
                input_path with .normalized.jsonl suffix.
            reject_path: Path for rejected records JSONL. Defaults to
                input_path with .rejected.jsonl suffix.
        """
        if output_path is None:
            output_path = input_path.with_suffix(".normalized.jsonl")
        if reject_path is None:
            reject_path = input_path.with_suffix(".rejected.jsonl")

        result = NormalizationSummary()
        source_name = input_path.stem

        with (
            open(input_path, encoding="utf-8") as infile,
            open(output_path, "w", encoding="utf-8") as outfile,
            open(reject_path, "w", encoding="utf-8") as rejectfile,
        ):
            for line in infile:
                if not line.strip():
                    continue

                result.total_records += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    result.rejected_records += 1
                    result.rejected_reasons["json_parse_error"] = result.rejected_reasons.get("json_parse_error", 0) + 1
                    rejectfile.write(line)
                    continue

                record = self.normalize_record(record)

                # Validation
                validation = self.validate_record(record)
                if not validation.valid:
                    result.rejected_records += 1
                    for error in validation.errors:
                        reason_key = error.split(":")[0].strip().replace(" ", "_").lower()
                        result.rejected_reasons[reason_key] = result.rejected_reasons.get(reason_key, 0) + 1
                    rejectfile.write(line)
                    continue

                # Success path
                try:
                    conversation = self.to_conversation(record, source_name)
                    outfile.write(json.dumps(conversation.to_dict(), ensure_ascii=False) + "\n")
                    result.valid_records += 1
                except Exception as e:
                    logger.error(f"Error processing record {record.get('id', 'N/A')}: {e}")
                    result.rejected_records += 1
                    result.rejected_reasons["processing_error"] = result.rejected_reasons.get("processing_error", 0) + 1
                    rejectfile.write(line)

        logger.info(
            "Process complete: %d total, %d valid, %d rejected",
            result.total_records,
            result.valid_records,
            result.rejected_records,
        )
        return result

    def process_jsonl_file(
        self,
        input_path: Path,
        output_path: Path | None = None,
        reject_path: Path | None = None,
    ) -> NormalizationSummary:
        """Compatibility wrapper for pipeline callers."""
        return self.process_file(input_path, output_path, reject_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def _to_snake_case(key: str) -> str:
        """
        Convert a string key to lower_snake_case.

        ⚡ Bolt Optimization: Uses pre-compiled regular expressions and an LRU cache.
        Since JSONL files often contain thousands of records with identical keys (like 'role', 'content', 'metadata'),
        caching this conversion skips expensive redundant regex operations.
        """
        key = key.strip()
        # Replace spaces and hyphens with underscores
        key = _RE_DASH.sub("_", key)
        # Insert underscore before uppercase letters (camelCase → camel_case)
        # Only split on lowercase-to-uppercase transitions to preserve acronyms
        key = _RE_CAMEL.sub("_", key)
        # Collapse multiple underscores and lower
        return _RE_MULTI.sub("_", key).lower()

    def _normalize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Normalize a single message dict."""
        if not isinstance(message, dict):
            return {"role": "user", "content": str(message)}
        result = {}
        for key, value in message.items():
            clean_key = self._to_snake_case(key)
            if isinstance(value, str):
                result[clean_key] = self.normalize_text(value)
            elif isinstance(value, dict):
                result[clean_key] = {self._to_snake_case(k): v for k, v in value.items()}
            else:
                result[clean_key] = value
        if isinstance(result.get("role"), str):
            result["role"] = result["role"].strip().lower()
        return result


@dataclass
class Message:
    """A single message in a conversation."""

    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class Conversation:
    """Represents a complete conversation adhering to the unified schema."""

    conversation_id: str
    source: str
    messages: list[Message]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "source": self.source,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Conversation:
        """Restore a Conversation from canonical dictionary form."""
        messages = [
            Message(
                role=str(message.get("role", "user")),
                content=str(message.get("content", "")),
                timestamp=str(message.get("timestamp") or datetime.now(UTC).isoformat()),
                metadata=message.get("metadata") if isinstance(message.get("metadata"), dict) else {},
            )
            for message in data.get("messages", [])
            if isinstance(message, dict)
        ]
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        return cls(
            conversation_id=str(data.get("conversation_id") or data.get("id") or ""),
            source=str(data.get("source") or metadata.get("source_repo") or "unknown"),
            messages=messages,
            metadata=metadata,
            created_at=str(data.get("created_at") or datetime.now(UTC).isoformat()),
            updated_at=str(data.get("updated_at") or datetime.now(UTC).isoformat()),
        )
