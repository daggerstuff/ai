#!/usr/bin/env python3
"""
Transcript Ingestor Module for Mental Health AI Training Data

This module provides production-quality functionality for ingesting transcript
data from various formats (JSON, JSONL, plain text with speaker labels, SRT/VTT)
into the PIX-32 normalized schema for mental health AI training.

Key Features:
- Multi-format support (JSON, JSONL, TXT, SRT, VTT)
- Speaker role detection and validation (client/therapist)
- Timestamp extraction and handling
- Batch processing capabilities
- Metadata extraction
- PIX-32 schema compliance
- PII scrubbing integration ready

Usage:
    from ai.pkg_mera.core.pipelines.processing.transcript_ingestor import TranscriptIngestor

    ingestor = TranscriptIngestor()
    result = ingestor.ingest_file("path/to/transcript.json")
    batch_result = ingestor.ingest_batch(["transcript1.jsonl", "transcript2.srt"])
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptFormat(Enum):
    """Supported transcript formats."""

    JSON = "json"
    JSONL = "jsonl"
    TXT = "txt"
    SRT = "srt"
    VTT = "vtt"


class SpeakerRole(Enum):
    """Speaker roles in mental health conversations."""

    CLIENT = "client"
    THERAPIST = "therapist"
    UNKNOWN = "unknown"


@dataclass
class TranscriptMessage:
    """A single message in a transcript."""

    role: str
    content: str
    timestamp: str | None = None
    speaker_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_pix32_dict(self) -> dict[str, Any]:
        """Convert to PIX-32 format dictionary."""
        result = {
            "role": self.role,
            "content": self.content,
        }
        if self.timestamp:
            result["timestamp"] = self.timestamp
        if self.speaker_id:
            result["speaker_id"] = self.speaker_id
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class IngestedTranscript:
    """A fully ingested transcript ready for PIX-32 normalization."""

    conversation_id: str
    messages: list[TranscriptMessage]
    source: str
    source_file: str
    format: TranscriptFormat
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    ingested_at: datetime = field(default_factory=datetime.now)

    @property
    def message_count(self) -> int:
        """Return the number of messages."""
        return len(self.messages)

    @property
    def total_characters(self) -> int:
        """Return total character count across all messages."""
        return sum(len(msg.content) for msg in self.messages)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate duration if timestamps are available."""
        # This is simplified - real implementation would parse timestamps.
        # Duration parsing is intentionally deferred, so return None for now.
        return None

    def to_pix32_dict(self) -> dict[str, Any]:
        """Convert to PIX-32 normalized format."""
        return {
            "conversation_id": self.conversation_id,
            "source": self.source,
            "messages": [msg.to_pix32_dict() for msg in self.messages],
            "metadata": {
                "source_file": self.source_file,
                "format": self.format.value,
                "message_count": self.message_count,
                "total_characters": self.total_characters,
                "ingested_at": self.ingested_at.isoformat(),
                **self.metadata,
            },
        }


@dataclass
class IngestionResult:
    """Result of batch ingestion operation."""

    transcripts: list[IngestedTranscript]
    total_files: int
    successful: int
    failed: int
    errors: list[dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful / self.total_files if self.total_files else 0.0


@dataclass
class TranscriptIngestorConfig:
    """Configuration for TranscriptIngestor."""

    min_message_length: int = 1
    max_message_length: int = 10000
    min_conversation_length: int = 2
    max_conversation_length: int = 1000
    default_source: str = "unknown"
    auto_detect_roles: bool = True
    validate_encoding: bool = True
    skip_empty_messages: bool = True
    warn_on_missing_timestamps: bool = True


class TranscriptIngestor:
    """
    Production-quality transcript ingestor for mental health AI training data.

    Supports multiple formats and provides comprehensive validation,
    role detection, and PIX-32 schema compliance.
    """

    # Patterns for speaker detection
    THERAPIST_PATTERNS = [
        re.compile(r"\b(therapist|counselor|psychologist|clinician|doctor|dr\.?)\b", re.I),
        re.compile(r"^(T|Th|Therapist|Counselor)[:\]]\s*", re.I),
    ]

    CLIENT_PATTERNS = [
        re.compile(r"\b(client|patient|user|caller)\b", re.I),
        re.compile(r"^(C|Cl|Client|P|Pt|Patient|U|User)[:\]]\s*", re.I),
    ]

    def __init__(self, config: TranscriptIngestorConfig | None = None):
        """Initialize the transcript ingestor."""
        self.config = config or TranscriptIngestorConfig()
        self._format_handlers = {
            TranscriptFormat.JSON: self._parse_json,
            TranscriptFormat.JSONL: self._parse_jsonl,
            TranscriptFormat.TXT: self._parse_txt,
            TranscriptFormat.SRT: self._parse_srt,
            TranscriptFormat.VTT: self._parse_vtt,
        }
        logger.info("TranscriptIngestor initialized with config: %s", self.config)

    def ingest_file(self, file_path: str | Path) -> IngestedTranscript:
        """
        Ingest a single transcript file.

        Args:
            file_path: Path to the transcript file.

        Returns:
            IngestedTranscript ready for PIX-32 normalization.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is unsupported or content is invalid.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {file_path}")

        # Detect format
        detected_format = self._detect_format(path)
        logger.info(f"Detected format {detected_format.value} for {file_path}")

        # Read content
        content = self._read_file(path)

        # Parse with appropriate handler
        handler = self._format_handlers[detected_format]
        messages, metadata = handler(content)

        # Validate and filter messages
        messages = self._validate_messages(messages)

        # Auto-detect roles if enabled
        if self.config.auto_detect_roles:
            messages = self._detect_roles(messages)

        # Create transcript
        transcript = IngestedTranscript(
            conversation_id=self._generate_conversation_id(path, metadata),
            messages=messages,
            source=metadata.get("source", self.config.default_source),
            source_file=str(path),
            format=detected_format,
            metadata=metadata,
            warnings=metadata.get("warnings", []),
        )

        logger.info(
            f"Ingested {path.name}: {transcript.message_count} messages, {transcript.total_characters} characters"
        )
        return transcript

    def ingest_batch(self, file_paths: list[str | Path]) -> IngestionResult:
        """
        Ingest multiple transcript files.

        Args:
            file_paths: List of paths to transcript files.

        Returns:
            IngestionResult with all successfully ingested transcripts.
        """
        start_time = datetime.now(UTC)
        transcripts = []
        errors = []

        for path in file_paths:
            try:
                transcript = self.ingest_file(path)
                transcripts.append(transcript)
            except Exception as e:
                errors.append(
                    {
                        "file": str(path),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
                logger.error(f"Failed to ingest {path}: {e}")

        processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

        result = IngestionResult(
            transcripts=transcripts,
            total_files=len(file_paths),
            successful=len(transcripts),
            failed=len(errors),
            errors=errors,
            processing_time_ms=processing_time,
        )

        logger.info(
            f"Batch ingestion complete: {result.successful}/{result.total_files} successful ({result.success_rate:.1%})"
        )
        return result

    def _detect_format(self, path: Path) -> TranscriptFormat:
        """Detect transcript format from file extension."""
        suffix = path.suffix.lower()
        format_map = {
            ".json": TranscriptFormat.JSON,
            ".jsonl": TranscriptFormat.JSONL,
            ".txt": TranscriptFormat.TXT,
            ".srt": TranscriptFormat.SRT,
            ".vtt": TranscriptFormat.VTT,
        }
        if suffix in format_map:
            return format_map[suffix]

        # Try to detect from content
        try:
            content = self._read_file_preview(path)
            if content.startswith(("{", "[")):
                return TranscriptFormat.JSON
            if content.startswith("WEBVTT"):
                return TranscriptFormat.VTT
            if re.match(r"^\d+\s*\n", content):
                return TranscriptFormat.SRT
            return TranscriptFormat.TXT
        except Exception:
            return TranscriptFormat.TXT

    def _read_file(self, path: Path) -> str:
        """Read file content with encoding handling."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue

        # Fallback: read with errors ignored
        return path.read_text(encoding="utf-8", errors="ignore")

    def _read_file_preview(self, path: Path, preview_chars: int = 1000) -> str:
        """Read and normalize a short preview segment for format detection."""
        return path.read_text(encoding="utf-8", errors="ignore")[:preview_chars].strip()

    def _parse_json(self, content: str) -> tuple[list[TranscriptMessage], dict]:
        """Parse JSON format transcript."""
        data = json.loads(content)
        messages = []
        metadata: dict[str, Any] = {}

        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("messages", data.get("transcript", [data]))
            metadata |= {k: v for k, v in data.items() if k not in ["messages", "transcript"]}
        else:
            raise ValueError(f"Unexpected JSON structure: {type(data)}")

        for item in items:
            if isinstance(item, dict):
                msg = TranscriptMessage(
                    role=item.get("role", item.get("speaker", "unknown")),
                    content=item.get("content", item.get("text", item.get("message", ""))),
                    timestamp=item.get("timestamp", item.get("time")),
                    speaker_id=item.get("speaker_id"),
                    metadata=item.get("metadata", {}),
                )
                messages.append(msg)
            elif isinstance(item, str):
                messages.append(TranscriptMessage(role="unknown", content=item))

        return messages, metadata

    def _parse_jsonl(self, content: str) -> tuple[list[TranscriptMessage], dict]:
        """Parse JSONL format transcript."""
        messages = []
        metadata: dict[str, Any] = {"source": "jsonl"}

        for line_num, line in enumerate(content.strip().split("\n"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                msg = TranscriptMessage(
                    role=item.get("role", item.get("speaker", "unknown")),
                    content=item.get("content", item.get("text", "")),
                    timestamp=item.get("timestamp"),
                    speaker_id=item.get("speaker_id"),
                    metadata=item.get("metadata", {}),
                )
                messages.append(msg)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")

        return messages, metadata

    def _parse_txt(self, content: str) -> tuple[list[TranscriptMessage], dict]:
        """Parse plain text transcript with speaker labels."""
        messages = []
        metadata: dict[str, Any] = {"source": "txt"}
        warnings = []

        # Pattern: "Speaker: Message" or "[Speaker] Message"
        speaker_pattern = re.compile(
            r"^(?:(?P<speaker>[^:\]\[]+)[:\]]|(?P<bracketed>\[[^\]]+\]))\s*(?P<content>.*)$",
            re.MULTILINE,
        )

        lines = content.strip().split("\n")
        current_speaker = "unknown"
        current_content = []

        for line in lines:
            if match := speaker_pattern.match(line.strip()):
                # Save previous message
                if current_content:
                    messages.append(
                        TranscriptMessage(
                            role=current_speaker,
                            content=" ".join(current_content),
                        )
                    )
                    current_content = []

                # New speaker
                current_speaker = match["speaker"] or match["bracketed"].strip("[]")
                if match["content"]:
                    current_content.append(match["content"])
            else:
                # Continue current speaker's message
                current_content.append(line.strip())

        # Save final message
        if current_content:
            messages.append(
                TranscriptMessage(
                    role=current_speaker,
                    content=" ".join(current_content),
                )
            )

        if not messages:
            # No speaker labels found - treat as single block
            messages.append(TranscriptMessage(role="unknown", content=content.strip()))
            warnings.append("No speaker labels detected")

        metadata["warnings"] = warnings
        return messages, metadata

    def _parse_srt(self, content: str) -> tuple[list[TranscriptMessage], dict]:
        """Parse SRT subtitle format."""
        messages = []
        metadata: dict[str, Any] = {"source": "srt"}

        # SRT format: index -> timestamp -> text -> blank line
        blocks = re.split(r"\n\s*\n", content.strip())

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue

            # Skip index line (lines[0])
            timestamp = lines[1]
            text = " ".join(lines[2:])

            if text:
                messages.append(
                    TranscriptMessage(
                        role="unknown",
                        content=text,
                        timestamp=timestamp,
                    )
                )

        return messages, metadata

    def _parse_vtt(self, content: str) -> tuple[list[TranscriptMessage], dict]:
        """Parse VTT subtitle format."""
        messages = []
        metadata: dict[str, Any] = {"source": "vtt"}

        # Remove WEBVTT header
        lines = content.strip().split("\n")
        if lines and lines[0].startswith("WEBVTT"):
            lines = lines[1:]

        # Join back and parse similar to SRT
        content = "\n".join(lines)
        blocks = re.split(r"\n\s*\n", content)

        for block in blocks:
            block_lines = block.strip().split("\n")
            if not block_lines:
                continue

            # Find timestamp line (contains -->)
            timestamp = None
            text_lines = []
            for line in block_lines:
                if "-->" in line:
                    timestamp = line
                elif not line.strip().isdigit():  # Skip cue identifiers
                    text_lines.append(line)

            text = " ".join(text_lines)
            if text and timestamp:
                messages.append(
                    TranscriptMessage(
                        role="unknown",
                        content=text,
                        timestamp=timestamp,
                    )
                )

        return messages, metadata

    def _validate_messages(self, messages: list[TranscriptMessage]) -> list[TranscriptMessage]:
        """Validate and filter messages based on configuration."""
        validated = []

        for msg in messages:
            # Check message length
            if len(msg.content) < self.config.min_message_length and self.config.skip_empty_messages:
                continue

            if len(msg.content) > self.config.max_message_length:
                # Truncate with warning
                msg.content = msg.content[: self.config.max_message_length]
                msg.metadata["truncated"] = True

            validated.append(msg)

        # Check conversation length
        if len(validated) < self.config.min_conversation_length:
            logger.warning(
                f"Conversation has only {len(validated)} messages, minimum is {self.config.min_conversation_length}"
            )

        if len(validated) > self.config.max_conversation_length:
            logger.warning(
                f"Conversation has {len(validated)} messages, truncating to {self.config.max_conversation_length}"
            )
            validated = validated[: self.config.max_conversation_length]

        return validated

    def _detect_roles(self, messages: list[TranscriptMessage]) -> list[TranscriptMessage]:
        """Auto-detect speaker roles (client/therapist)."""
        for msg in messages:
            role_lower = msg.role.lower()

            # Check therapist patterns
            for pattern in self.THERAPIST_PATTERNS:
                if pattern.search(role_lower) or pattern.search(msg.content[:100]):
                    msg.role = SpeakerRole.THERAPIST.value
                    break
            else:
                # Check client patterns
                for pattern in self.CLIENT_PATTERNS:
                    if pattern.search(role_lower) or pattern.search(msg.content[:100]):
                        msg.role = SpeakerRole.CLIENT.value
                        break

        return messages

    def _generate_conversation_id(self, path: Path, metadata: dict[str, Any]) -> str:
        """Generate a unique conversation ID."""
        # Use metadata ID if available
        if "id" in metadata:
            return str(metadata["id"])
        if "conversation_id" in metadata:
            return str(metadata["conversation_id"])

        # Generate from filename and hash
        filename_hash = hashlib.md5(path.name.encode()).hexdigest()[:8]
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M")
        return f"transcript_{filename_hash}_{timestamp}"


__all__ = [
    "IngestedTranscript",
    "IngestionResult",
    "SpeakerRole",
    "TranscriptFormat",
    "TranscriptIngestor",
    "TranscriptIngestorConfig",
    "TranscriptMessage",
]
