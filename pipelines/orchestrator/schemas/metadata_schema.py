#!/usr/bin/env python3
"""
Unified Metadata Schema for Task 6.6
Standardized metadata schema for all conversation datasets.
"""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetType(Enum):
    """Types of datasets."""
    PRIORITY = "priority"
    COT = "cot"
    REDDIT = "reddit"
    SYNTHETIC = "synthetic"
    EXTERNAL = "external"


class ConversationStatus(Enum):
    """Status of conversation processing."""
    RAW = "raw"
    VALIDATED = "validated"
    PROCESSED = "processed"
    APPROVED = "approved"
    ARCHIVED = "archived"


class QualityTier(Enum):
    """Quality tiers for conversations."""
    PRIORITY = "priority"
    PROFESSIONAL = "professional"
    COT = "cot"
    REDDIT = "reddit"
    SYNTHETIC = "synthetic"
    ARCHIVE = "archive"


@dataclass
class PersonMetadata:
    """Metadata about a person in conversation."""
    person_id: str | None = None
    role: str | None = None  # "user", "therapist", "assistant"
    age_range: str | None = None
    gender: str | None = None
    cultural_background: str | None = None
    mental_health_conditions: list[str] = field(default_factory=list)
    personality_traits: dict[str, float] = field(default_factory=dict)


@dataclass
class TurnMetadata:
    """Metadata for individual conversation turn."""
    turn_id: str
    speaker: str
    content: str
    timestamp: datetime | None = None
    word_count: int = 0
    sentiment_score: float | None = None
    emotion_labels: list[str] = field(default_factory=list)
    therapeutic_techniques: list[str] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    quality_score: float | None = None


@dataclass
class QualityMetrics:
    """Quality assessment metrics."""
    overall_score: float = 0.0
    therapeutic_relevance: float = 0.0
    conversation_coherence: float = 0.0
    emotional_appropriateness: float = 0.0
    safety_compliance: float = 0.0
    linguistic_quality: float = 0.0
    engagement_level: float = 0.0
    professional_standards: float = 0.0
    confidence: float = 0.0
    assessed_by: str | None = None
    assessment_timestamp: datetime | None = None


@dataclass
class ProcessingMetadata:
    """Processing pipeline metadata."""
    pipeline_version: str = "1.0"
    processing_stages: list[str] = field(default_factory=list)
    processing_timestamps: dict[str, datetime] = field(default_factory=dict)
    processing_errors: list[str] = field(default_factory=list)
    validation_results: dict[str, Any] = field(default_factory=dict)
    transformations_applied: list[str] = field(default_factory=list)


@dataclass
class ConversationMetadata:
    """Complete conversation metadata schema."""
    # Core identifiers
    conversation_id: str
    dataset_type: DatasetType
    dataset_name: str
    original_id: str | None = None

    # Content metadata
    title: str | None = None
    summary: str | None = None
    topic_tags: list[str] = field(default_factory=list)
    mental_health_conditions: list[str] = field(default_factory=list)
    therapeutic_approaches: list[str] = field(default_factory=list)

    # Conversation structure
    turn_count: int = 0
    total_word_count: int = 0
    conversation_length_minutes: float | None = None
    turns: list[TurnMetadata] = field(default_factory=list)

    # Participants
    participants: list[PersonMetadata] = field(default_factory=list)

    # Quality and validation
    status: ConversationStatus = ConversationStatus.RAW
    quality_tier: QualityTier | None = None
    quality_metrics: QualityMetrics | None = None

    # Processing information
    processing_metadata: ProcessingMetadata = field(default_factory=ProcessingMetadata)

    # Temporal information
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    conversation_date: datetime | None = None

    # Source information
    source_file: str | None = None
    source_url: str | None = None
    collection_method: str | None = None

    # Privacy and ethics
    privacy_level: str = "standard"
    consent_status: str | None = None
    anonymization_level: str = "full"
    ethical_review_status: str | None = None

    # Usage and licensing
    license: str = "internal_use"
    usage_restrictions: list[str] = field(default_factory=list)
    attribution_required: bool = False

    # Additional metadata
    custom_fields: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str | None = None


class MetadataSchema:
    """
    Unified metadata schema manager.
    """

    def __init__(self):
        """Initialize metadata schema manager."""
        self.schema_version = "1.0"
        self.required_fields = [
            "conversation_id", "dataset_type", "dataset_name",
            "status", "created_at"
        ]

        logger.info("MetadataSchema initialized")

    def create_conversation_metadata(self,
                                   conversation_id: str,
                                   dataset_type: DatasetType,
                                   dataset_name: str,
                                   **kwargs) -> ConversationMetadata:
        """Create new conversation metadata."""
        metadata = ConversationMetadata(
            conversation_id=conversation_id,
            dataset_type=dataset_type,
            dataset_name=dataset_name,
            **kwargs
        )

        # Set processing metadata
        metadata.processing_metadata.processing_stages.append("metadata_creation")
        metadata.processing_metadata.processing_timestamps["metadata_creation"] = datetime.now(timezone.utc)

        return metadata

    def validate_metadata(self, metadata: ConversationMetadata) -> dict[str, Any]:
        """Validate conversation metadata."""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "missing_fields": []
        }

        # Check required fields
        for field in self.required_fields:
            if not hasattr(metadata, field) or getattr(metadata, field) is None:
                validation_result["missing_fields"].append(field)
                validation_result["is_valid"] = False

        # Validate conversation_id format
        if not metadata.conversation_id or len(metadata.conversation_id) < 8:
            validation_result["errors"].append("conversation_id must be at least 8 characters")
            validation_result["is_valid"] = False

        # Validate turn metadata
        if metadata.turns:
            for i, turn in enumerate(metadata.turns):
                if not turn.turn_id:
                    validation_result["errors"].append(f"Turn {i} missing turn_id")
                    validation_result["is_valid"] = False

                if not turn.content:
                    validation_result["warnings"].append(f"Turn {i} has empty content")

        # Validate quality metrics
        if metadata.quality_metrics:
            qm = metadata.quality_metrics
            if not (0 <= qm.overall_score <= 1):
                validation_result["errors"].append("overall_score must be between 0 and 1")
                validation_result["is_valid"] = False

        # Validate participants
        if metadata.participants:
            for i, participant in enumerate(metadata.participants):
                if not participant.role:
                    validation_result["warnings"].append(f"Participant {i} missing role")

        return validation_result

    def enrich_metadata(self, metadata: ConversationMetadata,
                       conversation_data: dict[str, Any]) -> ConversationMetadata:
        """Enrich metadata with conversation data analysis."""
        # Extract turns if not already present
        if not metadata.turns and "turns" in conversation_data:
            for i, turn_data in enumerate(conversation_data["turns"]):
                turn = TurnMetadata(
                    turn_id=f"{metadata.conversation_id}_turn_{i}",
                    speaker=turn_data.get("speaker", "unknown"),
                    content=turn_data.get("content", ""),
                    word_count=len(turn_data.get("content", "").split())
                )
                metadata.turns.append(turn)

        # Update conversation statistics
        metadata.turn_count = len(metadata.turns)
        metadata.total_word_count = sum(turn.word_count for turn in metadata.turns)

        # Extract topics from content
        if not metadata.topic_tags:
            metadata.topic_tags = self._extract_topics(conversation_data)

        # Extract mental health conditions
        if not metadata.mental_health_conditions:
            metadata.mental_health_conditions = self._extract_conditions(conversation_data)

        # Update processing metadata
        metadata.processing_metadata.processing_stages.append("metadata_enrichment")
        metadata.processing_metadata.processing_timestamps["metadata_enrichment"] = datetime.now(timezone.utc)
        metadata.updated_at = datetime.now(timezone.utc)

        return metadata

    def _extract_topics(self, conversation_data: dict[str, Any]) -> list[str]:
        """Extract topics from conversation data."""
        content = self._extract_content(conversation_data)
        topics = []

        topic_keywords = {
            "anxiety": ["anxiety", "anxious", "worry", "panic", "fear"],
            "depression": ["depression", "depressed", "sad", "hopeless"],
            "relationships": ["relationship", "partner", "marriage", "family"],
            "work": ["work", "job", "career", "workplace"],
            "therapy": ["therapy", "counseling", "treatment"],
            "trauma": ["trauma", "abuse", "ptsd", "flashback"]
        }

        content_lower = content.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)

        return topics

    def _extract_conditions(self, conversation_data: dict[str, Any]) -> list[str]:
        """Extract mental health conditions from conversation data."""
        content = self._extract_content(conversation_data)
        conditions = []

        condition_keywords = {
            "anxiety_disorder": ["anxiety disorder", "generalized anxiety"],
            "depression": ["major depression", "clinical depression"],
            "bipolar": ["bipolar", "manic depression"],
            "ptsd": ["ptsd", "post-traumatic stress"],
            "ocd": ["ocd", "obsessive compulsive"],
            "adhd": ["adhd", "attention deficit"]
        }

        content_lower = content.lower()
        for condition, keywords in condition_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                conditions.append(condition)

        return conditions

    def _extract_content(self, conversation_data: dict[str, Any]) -> str:
        """Extract content from conversation data."""
        content = conversation_data.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(turn) for turn in content)
        elif isinstance(content, dict) and "turns" in conversation_data:
            turns = conversation_data["turns"]
            content = " ".join(turn.get("content", "") for turn in turns)
        return content

    def update_quality_metrics(self, metadata: ConversationMetadata,
                              quality_metrics: QualityMetrics) -> ConversationMetadata:
        """Update quality metrics in metadata."""
        quality_metrics.assessment_timestamp = datetime.now(timezone.utc)
        metadata.quality_metrics = quality_metrics

        # Determine quality tier based on overall score
        if quality_metrics.overall_score >= 0.9:
            metadata.quality_tier = QualityTier.PRIORITY
        elif quality_metrics.overall_score >= 0.8:
            metadata.quality_tier = QualityTier.PROFESSIONAL
        elif quality_metrics.overall_score >= 0.7:
            metadata.quality_tier = QualityTier.COT
        elif quality_metrics.overall_score >= 0.6:
            metadata.quality_tier = QualityTier.REDDIT
        else:
            metadata.quality_tier = QualityTier.ARCHIVE

        # Update processing metadata
        metadata.processing_metadata.processing_stages.append("quality_assessment")
        metadata.processing_metadata.processing_timestamps["quality_assessment"] = datetime.now(timezone.utc)
        metadata.updated_at = datetime.now(timezone.utc)

        return metadata

    def update_status(self, metadata: ConversationMetadata,
                     new_status: ConversationStatus,
                     notes: str | None = None) -> ConversationMetadata:
        """Update conversation status."""
        old_status = metadata.status
        metadata.status = new_status
        metadata.updated_at = datetime.now(timezone.utc)

        # Add to processing metadata
        stage_name = f"status_change_{old_status.value}_to_{new_status.value}"
        metadata.processing_metadata.processing_stages.append(stage_name)
        metadata.processing_metadata.processing_timestamps[stage_name] = datetime.now(timezone.utc)

        if notes:
            if not metadata.notes:
                metadata.notes = notes
            else:
                metadata.notes += f"\n{datetime.now(timezone.utc).isoformat()}: {notes}"

        return metadata

    def to_dict(self, metadata: ConversationMetadata) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(metadata)

    def from_dict(self, data: dict[str, Any]) -> ConversationMetadata:
        """Create metadata from dictionary."""
        # Handle enum conversions
        if "dataset_type" in data and isinstance(data["dataset_type"], str):
            data["dataset_type"] = DatasetType(data["dataset_type"])

        if "status" in data and isinstance(data["status"], str):
            data["status"] = ConversationStatus(data["status"])

        if "quality_tier" in data and isinstance(data["quality_tier"], str):
            data["quality_tier"] = QualityTier(data["quality_tier"])

        # Handle datetime conversions
        datetime_fields = ["created_at", "updated_at", "conversation_date"]
        for field in datetime_fields:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])

        return ConversationMetadata(**data)

    def export_schema_documentation(self) -> dict[str, Any]:
        """Export schema documentation."""
        return {
            "schema_version": self.schema_version,
            "description": "Unified metadata schema for conversation datasets",
            "required_fields": self.required_fields,
            "data_types": {
                "DatasetType": [e.value for e in DatasetType],
                "ConversationStatus": [e.value for e in ConversationStatus],
                "QualityTier": [e.value for e in QualityTier]
            },
            "field_descriptions": {
                "conversation_id": "Unique identifier for the conversation",
                "dataset_type": "Type of dataset (priority, cot, reddit, etc.)",
                "dataset_name": "Name of the specific dataset",
                "status": "Current processing status of the conversation",
                "quality_tier": "Quality tier assignment based on assessment",
                "quality_metrics": "Detailed quality assessment scores",
                "turns": "Individual conversation turns with metadata",
                "participants": "Information about conversation participants",
                "processing_metadata": "Pipeline processing information"
            },
            "validation_rules": {
                "conversation_id": "Must be at least 8 characters",
                "quality_scores": "Must be between 0 and 1",
                "required_fields": self.required_fields
            }
        }


def main():
    """Test the metadata schema."""
    schema = MetadataSchema()

    # Create test conversation metadata
    metadata = schema.create_conversation_metadata(
        conversation_id="test_conv_001",
        dataset_type=DatasetType.PRIORITY,
        dataset_name="priority_therapeutic_conversations",
        title="Anxiety Support Session"
    )


    # Test conversation data
    conversation_data = {
        "turns": [
            {"speaker": "user", "content": "I'm feeling really anxious about my presentation tomorrow."},
            {"speaker": "therapist", "content": "I understand that presentations can be anxiety-provoking. Tell me more about what specifically worries you."}
        ]
    }

    # Enrich metadata
    metadata = schema.enrich_metadata(metadata, conversation_data)


    # Validate metadata
    validation = schema.validate_metadata(metadata)
    if validation["errors"]:
        pass
    if validation["warnings"]:
        pass

    # Test quality metrics update
    quality_metrics = QualityMetrics(
        overall_score=0.85,
        therapeutic_relevance=0.9,
        safety_compliance=1.0,
        assessed_by="automated_system"
    )

    metadata = schema.update_quality_metrics(metadata, quality_metrics)

    # Test status update
    metadata = schema.update_status(metadata, ConversationStatus.VALIDATED, "Passed quality assessment")

    # Export to dict and back
    schema.to_dict(metadata)

    # Test schema documentation
    schema.export_schema_documentation()


if __name__ == "__main__":
    main()
