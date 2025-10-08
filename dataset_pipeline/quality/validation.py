"""Validation module for dataset pipeline Phase 02.

Defines canonical schemas using Pydantic for ingested records, focusing on conversation data.
Provides validators, sanitizers, and initial quality scoring. Integrates with ingestion by
wrapping record creation and providing a validate_record function that can be called post-fetch.

Schemas enforce structure for mental health dialogues: speaker turns, timestamps, metadata.
Supports source mapping (CSV/JSON/audio transcripts) to canonical format.
Quarantine handling: raises ValidationError with details for failed records.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import bleach
from pydantic import BaseModel, Field, ValidationError, validator

from .ingestion_interface import IngestionError, IngestRecord


class SpeakerTurn(BaseModel):
    """A single turn in a conversation."""
    speaker_id: str = Field(..., min_length=1, description="Unique speaker identifier (e.g., 'therapist', 'client')")
    content: str = Field(..., min_length=1, max_length=5000, description="Dialogue content")
    timestamp: datetime | None = Field(None, description="When the turn occurred")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional turn-specific data")

    @validator("content")
    def sanitize_content(self, v):
        """Sanitize text to prevent XSS/HTML injection."""
        # Basic HTML sanitization using bleach
        cleaned = bleach.clean(
            v,
            tags=["p", "br", "strong", "em", "ul", "ol", "li"],
            strip=True
        )
        # Remove any remaining suspicious patterns (e.g., script tags, etc.)
        cleaned = re.sub(r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", "", cleaned, flags=re.IGNORECASE)
        if len(cleaned) != len(v.strip()):
            raise ValueError("Content altered during sanitization - potential security issue")
        return cleaned.strip()

    @validator("speaker_id")
    def validate_speaker_id(self, v):
        """Ensure speaker ID is safe and valid."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Speaker ID must be alphanumeric with underscores/hyphens only")
        return v


class ConversationRecord(BaseModel):
    """Canonical schema for a mental health conversation record."""
    id: str = Field(..., min_length=1, max_length=100, description="Unique record ID")
    title: str | None = Field(None, max_length=200, description="Conversation title")
    turns: list[SpeakerTurn] = Field(..., min_items=1, max_items=1000, description="List of dialogue turns")
    source_type: str = Field(..., description="Original source (e.g., 'csv', 'json', 'audio_transcript')")
    source_id: str | None = Field(None, description="Original source identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provenance: timestamps, source details, quality scores"
    )

    @validator("id")
    def validate_id(self, v):
        """Ensure ID is a valid UUID-like or safe string."""
        if not re.match(r"^[a-zA-Z0-9-_.]+$", v):
            raise ValueError("ID must be alphanumeric with hyphens/underscores/periods only")
        return v

    @validator("turns")
    def validate_turns(self, v):
        """Ensure alternating speakers and basic content checks."""
        if len(v) < 2:
            raise ValueError("Conversation must have at least two turns (dialogue)")
        speakers = [turn.speaker_id for turn in v]
        unique_speakers = set(speakers)
        if len(unique_speakers) < 2:
            raise ValueError("Must have at least two distinct speakers (e.g., therapist/client)")
        # Check for alternating (simple heuristic)
        for i in range(1, len(v)):
            if v[i].speaker_id == v[i-1].speaker_id:
                raise ValueError(f"Consecutive turns by same speaker at index {i}")
        return v

    @validator("source_type")
    def validate_source_type(self, v):
        """Restrict to known source types."""
        allowed = {"csv", "json", "audio_transcript", "youtube", "s3_object", "gcs_object"}
        if v.lower() not in allowed:
            raise ValueError(f"Unknown source_type: {v}. Allowed: {allowed}")
        return v.lower()


class QualityScore(BaseModel):
    """Initial quality scoring model."""
    completeness: float = Field(..., ge=0.0, le=1.0, description="Fraction of required fields present")
    coherence: float = Field(..., ge=0.0, le=1.0, description="Turn coherence score (0-1)")
    relevance: float = Field(..., ge=0.0, le=1.0, description="Mental health topic relevance (0-1)")
    raw_score: float = Field(..., ge=0.0, le=10.0, description="Overall score before normalization")

    @validator("completeness", "coherence", "relevance")
    def normalize_scores(self, v):
        """Normalize to 0-1 scale."""
        return min(max(v, 0.0), 1.0)

    @classmethod
    def compute_initial(cls, record: ConversationRecord) -> QualityScore:
        """Compute basic quality score for a record."""
        # Simple heuristics for now; expand with ML later
        num_turns = len(record.turns)
        completeness = min(1.0, num_turns / 20.0)  # Favor 20+ turns
        avg_turn_length = sum(len(turn.content) for turn in record.turns) / num_turns / 100  # Per 100 chars
        coherence = min(1.0, avg_turn_length)  # Longer turns = more coherent
        relevance = 0.7 if any("mental" in turn.content.lower() or "health" in turn.content.lower() for turn in record.turns) else 0.3
        raw_score = (completeness + coherence + relevance) * 10 / 3
        return cls(completeness=completeness, coherence=coherence, relevance=relevance, raw_score=raw_score)


def validate_record(raw_record: IngestRecord) -> ConversationRecord:
    """Validate and normalize a raw IngestRecord to canonical ConversationRecord.

    Raises ValidationError on failure, with details for quarantine.
    Adds quality score to metadata.
    Maps source_type-specific formats to canonical.
    """
    try:
        # Basic payload check
        if not isinstance(raw_record.payload, dict):
            raise ValueError(f"Payload must be dict, got {type(raw_record.payload)}")

        payload = raw_record.payload
        source_type = payload.get("source_type", "unknown")

        # Source-specific mapping (expand as needed)
        if source_type == "csv":
            # Assume CSV rows map to turns; simple example
            turns_data = [
                {
                    "speaker_id": row.get("speaker", f"speaker_{i}"),
                    "content": row.get("content", ""),
                    "timestamp": datetime.fromisoformat(row.get("timestamp", "")) if row.get("timestamp") else None,
                }
                for i, row in enumerate(payload.get("rows", []))
            ]
            conv_data = {
                "id": raw_record.id,
                "title": payload.get("title"),
                "turns": [SpeakerTurn(**turn) for turn in turns_data],
                "source_type": source_type,
                "source_id": payload.get("source_id"),
                "metadata": raw_record.metadata,
            }
        elif source_type == "json":
            # Assume JSON with turns array
            turns_data = payload.get("turns", [])
            conv_data = {
                "id": raw_record.id,
                "title": payload.get("title"),
                "turns": [SpeakerTurn(**turn) for turn in turns_data],
                "source_type": source_type,
                "source_id": payload.get("source_id"),
                "metadata": raw_record.metadata,
            }
        elif source_type == "audio_transcript":
            # Assume transcript with segments
            segments = payload.get("segments", [])
            turns_data = [
                {
                    "speaker_id": seg.get("speaker", "unknown"),
                    "content": seg.get("text", ""),
                    "timestamp": datetime.fromtimestamp(seg.get("start_time", 0)) if seg.get("start_time") else None,
                }
                for seg in segments
            ]
            conv_data = {
                "id": raw_record.id,
                "title": payload.get("title", "Transcribed Conversation"),
                "turns": [SpeakerTurn(**turn) for turn in turns_data],
                "source_type": source_type,
                "source_id": payload.get("audio_file_id"),
                "metadata": raw_record.metadata,
            }
        else:
            # Default: assume payload is already in canonical form
            conv_data = {
                "id": raw_record.id,
                "title": payload.get("title"),
                "turns": [SpeakerTurn(**turn) for turn in payload.get("turns", [])],
                "source_type": source_type,
                "source_id": payload.get("source_id"),
                "metadata": raw_record.metadata,
            }

        validated = ConversationRecord(**conv_data)

        # Add quality score to metadata
        quality = QualityScore.compute_initial(validated)
        validated.metadata["quality_score"] = quality.dict()

        # Add provenance if missing
        if "ingestion_timestamp" not in validated.metadata:
            validated.metadata["ingestion_timestamp"] = datetime.utcnow()
        if "source_type" not in validated.metadata:
            validated.metadata["source_type"] = source_type

        return validated

    except ValidationError as e:
        # Include details for quarantine
        raise IngestionError(f"Validation failed: {e}") from e
    except Exception as e:
        raise IngestionError(f"Unexpected validation error: {e}") from e


# Integration hook: monkey-patch or wrapper for connectors
def validated_fetch(connector: IngestionConnector) -> Iterable[ConversationRecord]:
    """Wrapper to validate records from a connector's fetch."""
    for raw in connector.fetch():
        if connector.validate(raw):  # Existing connector validate
            yield validate_record(raw)
        else:
            # Skip or quarantine; for now, raise/log
            raise IngestionError(f"Connector rejected record: {raw.id}")


__all__ = [
    "ConversationRecord",
    "QualityScore",
    "SpeakerTurn",
    "validate_record",
    "validated_fetch",
]
