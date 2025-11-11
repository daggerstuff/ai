"""
Pipeline Integrator for Journal Dataset Research

Implements Task 10: Integration with existing training pipeline.
Converts datasets to training pipeline format, validates schema,
merges with existing data, and performs quality checks.

This module provides:
- Format converter using IntegrationPlan
- Schema validation against training pipeline schema
- Dataset merging and deduplication
- Quality checks for integrated data
"""

import hashlib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ai.journal_dataset_research.models.dataset_models import (
    AcquiredDataset,
    IntegrationPlan,
)

# Import training pipeline schemas
try:
    from ai.dataset_pipeline.schemas.conversation_schema import Conversation, Message
    from ai.dataset_pipeline.quality.validation import (
        ConversationRecord,
        SpeakerTurn,
        ValidationError,
    )
except ImportError:
    # Fallback if schemas aren't available
    Conversation = None
    Message = None
    ConversationRecord = None
    SpeakerTurn = None
    ValidationError = Exception

logger = logging.getLogger(__name__)

# Training pipeline schema definition
TRAINING_PIPELINE_SCHEMA = {
    "required_fields": ["messages", "id", "source"],
    "optional_fields": ["timestamp", "quality_score", "tags", "mental_health_condition"],
    "message_structure": {
        "role": ["user", "assistant", "system"],
        "content": "string",
    },
    "alternate_format": {
        "required_fields": ["id", "title", "turns", "source_type", "source_id"],
        "turns_structure": {
            "speaker_id": "string",
            "content": "string",
            "timestamp": "optional_datetime",
            "metadata": "optional_dict",
        },
    },
}


@dataclass
class ConversionResult:
    """Result of dataset conversion to training pipeline format."""

    success: bool
    records_converted: int
    records_failed: int
    output_path: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    conversion_time: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ValidationResult:
    """Result of schema validation."""

    valid: bool
    records_validated: int
    records_passed: int
    records_failed: int
    errors: List[Dict[str, Any]] = field(default_factory=list)
    validation_time: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MergeResult:
    """Result of dataset merging."""

    success: bool
    records_merged: int
    duplicates_removed: int
    conflicts_resolved: int
    output_path: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    merge_time: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class QualityCheckResult:
    """Result of quality checks."""

    passed: bool
    records_checked: int
    records_passed: int
    records_failed: int
    pii_detected: int = 0
    structure_issues: int = 0
    completeness_issues: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0
    check_time: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PipelineFormatConverter:
    """Converts datasets to training pipeline format using IntegrationPlan."""

    def __init__(self, pipeline_schema: Optional[Dict[str, Any]] = None):
        """Initialize the format converter."""
        self.pipeline_schema = pipeline_schema or TRAINING_PIPELINE_SCHEMA
        logger.info("Initialized Pipeline Format Converter")

    def convert_dataset(
        self,
        dataset: AcquiredDataset,
        integration_plan: IntegrationPlan,
        output_path: str,
        target_format: str = "chatml",
    ) -> ConversionResult:
        """
        Convert dataset to training pipeline format using integration plan.

        Args:
            dataset: The acquired dataset to convert
            integration_plan: The integration plan with schema mappings
            output_path: Path to save converted dataset
            target_format: Target format ("chatml" or "conversation_record")

        Returns:
            ConversionResult with conversion statistics
        """
        logger.info(
            f"Converting dataset {dataset.source_id} to {target_format} format"
        )
        start_time = datetime.now(timezone.utc)

        errors = []
        warnings = []
        records_converted = 0
        records_failed = 0

        try:
            # Load dataset based on format
            raw_data = self._load_dataset(dataset.storage_path, integration_plan.dataset_format)

            # Convert records
            converted_records = []
            for idx, record in enumerate(raw_data):
                try:
                    converted = self._convert_record(
                        record, integration_plan, dataset, target_format
                    )
                    if converted:
                        converted_records.append(converted)
                        records_converted += 1
                    else:
                        records_failed += 1
                        warnings.append(f"Record {idx} skipped: conversion returned None")
                except Exception as e:
                    records_failed += 1
                    errors.append(f"Record {idx} conversion failed: {str(e)}")
                    logger.warning(f"Failed to convert record {idx}: {e}")

            # Save converted dataset
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self._save_converted_dataset(converted_records, output_path, target_format)

            conversion_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            result = ConversionResult(
                success=len(errors) == 0,
                records_converted=records_converted,
                records_failed=records_failed,
                output_path=output_path,
                errors=errors,
                warnings=warnings,
                conversion_time=conversion_time,
                timestamp=start_time,
            )

            logger.info(
                f"Conversion complete: {records_converted} converted, "
                f"{records_failed} failed in {conversion_time:.2f}s"
            )
            return result

        except Exception as e:
            conversion_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            errors.append(f"Conversion failed: {str(e)}")
            logger.error(f"Dataset conversion failed: {e}", exc_info=True)

            return ConversionResult(
                success=False,
                records_converted=records_converted,
                records_failed=records_failed,
                output_path=output_path,
                errors=errors,
                warnings=warnings,
                conversion_time=conversion_time,
                timestamp=start_time,
            )

    def _load_dataset(
        self, file_path: str, dataset_format: str
    ) -> List[Dict[str, Any]]:
        """Load dataset from file based on format."""
        if dataset_format == "csv":
            df = pd.read_csv(file_path)
            return df.to_dict("records")
        elif dataset_format == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
                else:
                    raise ValueError(f"Unsupported JSON structure: {type(data)}")
        elif dataset_format == "jsonl":
            records = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
        elif dataset_format == "parquet":
            df = pd.read_parquet(file_path)
            return df.to_dict("records")
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")

    def _convert_record(
        self,
        record: Dict[str, Any],
        integration_plan: IntegrationPlan,
        dataset: AcquiredDataset,
        target_format: str,
    ) -> Optional[Dict[str, Any]]:
        """Convert a single record using integration plan schema mapping."""
        try:
            if target_format == "chatml":
                return self._convert_to_chatml(record, integration_plan, dataset)
            elif target_format == "conversation_record":
                return self._convert_to_conversation_record(
                    record, integration_plan, dataset
                )
            else:
                raise ValueError(f"Unsupported target format: {target_format}")
        except Exception as e:
            logger.warning(f"Record conversion failed: {e}")
            return None

    def _convert_to_chatml(
        self,
        record: Dict[str, Any],
        integration_plan: IntegrationPlan,
        dataset: AcquiredDataset,
    ) -> Dict[str, Any]:
        """Convert record to ChatML format."""
        # Generate ID if not present
        record_id = self._generate_id(record, dataset.source_id)

        # Extract messages based on schema mapping
        messages = self._extract_messages(record, integration_plan.schema_mapping)

        if not messages:
            raise ValueError("No messages found in record")

        # Build ChatML record
        chatml_record = {
            "id": record_id,
            "source": dataset.source_id,
            "messages": messages,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Add optional fields if available
        if "quality_score" in record:
            chatml_record["quality_score"] = record["quality_score"]
        if "tags" in record:
            chatml_record["tags"] = record["tags"]
        if "mental_health_condition" in record:
            chatml_record["mental_health_condition"] = record["mental_health_condition"]

        return chatml_record

    def _convert_to_conversation_record(
        self,
        record: Dict[str, Any],
        integration_plan: IntegrationPlan,
        dataset: AcquiredDataset,
    ) -> Dict[str, Any]:
        """Convert record to ConversationRecord format."""
        # Generate ID if not present
        record_id = self._generate_id(record, dataset.source_id)

        # Extract turns based on schema mapping
        turns = self._extract_turns(record, integration_plan.schema_mapping)

        if not turns:
            raise ValueError("No turns found in record")

        # Build ConversationRecord
        conversation_record = {
            "id": record_id,
            "title": record.get("title") or f"Conversation from {dataset.source_id}",
            "turns": turns,
            "source_type": integration_plan.dataset_format,
            "source_id": dataset.source_id,
            "metadata": {
                "conversion_timestamp": datetime.now(timezone.utc).isoformat(),
                "integration_plan": integration_plan.source_id,
            },
        }

        return conversation_record

    def _extract_messages(
        self, record: Dict[str, Any], schema_mapping: Dict[str, str]
    ) -> List[Dict[str, str]]:
        """Extract messages from record using schema mapping."""
        messages = []

        # Try to find messages field
        messages_field = None
        for dataset_field, pipeline_field in schema_mapping.items():
            if pipeline_field == "messages" or "message" in pipeline_field.lower():
                messages_field = dataset_field
                break

        if messages_field and messages_field in record:
            # Direct messages field
            raw_messages = record[messages_field]
            if isinstance(raw_messages, list):
                for msg in raw_messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if content:
                            messages.append({"role": role, "content": content})
        else:
            # Try to extract from conversation structure
            # Look for role/content fields
            role_field = None
            content_field = None

            for dataset_field, pipeline_field in schema_mapping.items():
                if "role" in pipeline_field.lower() or "speaker" in dataset_field.lower():
                    role_field = dataset_field
                if "content" in pipeline_field.lower() or "text" in dataset_field.lower():
                    content_field = dataset_field

            if role_field and content_field:
                role = record.get(role_field, "user")
                content = record.get(content_field, "")
                if content:
                    messages.append({"role": self._normalize_role(role), "content": content})
            elif content_field:
                # Single message, assume user role
                content = record.get(content_field, "")
                if content:
                    messages.append({"role": "user", "content": content})

        # If no messages found, try prompt/response pattern
        if not messages:
            if "prompt" in record and "response" in record:
                prompt = record.get("prompt", "")
                response = record.get("response", "")
                if prompt:
                    messages.append({"role": "user", "content": prompt})
                if response:
                    messages.append({"role": "assistant", "content": response})

        return messages

    def _extract_turns(
        self, record: Dict[str, Any], schema_mapping: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Extract turns from record using schema mapping."""
        turns = []

        # Try to find turns field
        turns_field = None
        for dataset_field, pipeline_field in schema_mapping.items():
            if pipeline_field == "turns" or "turn" in pipeline_field.lower():
                turns_field = dataset_field
                break

        if turns_field and turns_field in record:
            # Direct turns field
            raw_turns = record[turns_field]
            if isinstance(raw_turns, list):
                for turn in raw_turns:
                    if isinstance(turn, dict):
                        speaker_id = turn.get("speaker_id") or turn.get("speaker", "unknown")
                        content = turn.get("content") or turn.get("text", "")
                        if content:
                            turns.append({
                                "speaker_id": self._normalize_speaker(speaker_id),
                                "content": content,
                                "timestamp": turn.get("timestamp"),
                                "metadata": turn.get("metadata", {}),
                            })
        else:
            # Try to extract from message structure
            messages = self._extract_messages(record, schema_mapping)
            for msg in messages:
                speaker_id = "therapist" if msg["role"] == "assistant" else "client"
                turns.append({
                    "speaker_id": speaker_id,
                    "content": msg["content"],
                    "timestamp": None,
                    "metadata": {},
                })

        return turns

    def _normalize_role(self, role: str) -> str:
        """Normalize role to ChatML format (user, assistant, system)."""
        role_lower = role.lower()
        if role_lower in ["user", "client", "patient", "human"]:
            return "user"
        elif role_lower in ["assistant", "therapist", "counselor", "ai", "bot"]:
            return "assistant"
        elif role_lower in ["system"]:
            return "system"
        else:
            return "user"  # Default to user

    def _normalize_speaker(self, speaker: str) -> str:
        """Normalize speaker ID to therapist/client."""
        speaker_lower = speaker.lower()
        if speaker_lower in ["therapist", "counselor", "assistant", "ai"]:
            return "therapist"
        elif speaker_lower in ["client", "patient", "user", "human"]:
            return "client"
        else:
            return speaker_lower  # Keep original if unclear

    def _generate_id(self, record: Dict[str, Any], source_id: str) -> str:
        """Generate a stable ID for the record."""
        # Try to use existing ID
        for key in ["id", "record_id", "conversation_id"]:
            if key in record and record[key]:
                return str(record[key])

        # Generate UUID5 from content hash
        content_str = json.dumps(record, sort_keys=True)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()
        namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
        record_id = str(uuid.uuid5(namespace, f"{source_id}:{content_hash}"))
        return record_id

    def _save_converted_dataset(
        self, records: List[Dict[str, Any]], output_path: str, target_format: str
    ) -> None:
        """Save converted dataset to file."""
        if target_format == "chatml":
            # Save as JSONL for ChatML format
            with open(output_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        elif target_format == "conversation_record":
            # Save as JSONL for ConversationRecord format
            with open(output_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            raise ValueError(f"Unsupported target format for saving: {target_format}")


class PipelineSchemaValidator:
    """Validates datasets against training pipeline schema."""

    def __init__(self, pipeline_schema: Optional[Dict[str, Any]] = None):
        """Initialize the schema validator."""
        self.pipeline_schema = pipeline_schema or TRAINING_PIPELINE_SCHEMA
        logger.info("Initialized Pipeline Schema Validator")

    def validate_dataset(
        self, dataset_path: str, target_format: str = "chatml"
    ) -> ValidationResult:
        """
        Validate dataset against training pipeline schema.

        Args:
            dataset_path: Path to dataset file (JSONL)
            target_format: Target format ("chatml" or "conversation_record")

        Returns:
            ValidationResult with validation statistics
        """
        logger.info(f"Validating dataset {dataset_path} against {target_format} schema")
        start_time = datetime.now(timezone.utc)

        errors = []
        records_validated = 0
        records_passed = 0
        records_failed = 0

        try:
            # Load dataset
            records = self._load_dataset(dataset_path)

            # Validate each record
            for idx, record in enumerate(records):
                records_validated += 1
                try:
                    if target_format == "chatml":
                        is_valid = self._validate_chatml_record(record)
                    elif target_format == "conversation_record":
                        is_valid = self._validate_conversation_record(record)
                    else:
                        raise ValueError(f"Unsupported target format: {target_format}")

                    if is_valid:
                        records_passed += 1
                    else:
                        records_failed += 1
                        errors.append({
                            "record_index": idx,
                            "record_id": record.get("id", "unknown"),
                            "error": "Record failed validation",
                        })
                except Exception as e:
                    records_failed += 1
                    errors.append({
                        "record_index": idx,
                        "record_id": record.get("id", "unknown"),
                        "error": str(e),
                    })

            validation_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()

            result = ValidationResult(
                valid=records_failed == 0,
                records_validated=records_validated,
                records_passed=records_passed,
                records_failed=records_failed,
                errors=errors,
                validation_time=validation_time,
                timestamp=start_time,
            )

            logger.info(
                f"Validation complete: {records_passed} passed, "
                f"{records_failed} failed in {validation_time:.2f}s"
            )
            return result

        except Exception as e:
            validation_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()
            errors.append({
                "error": f"Validation failed: {str(e)}",
            })
            logger.error(f"Dataset validation failed: {e}", exc_info=True)

            return ValidationResult(
                valid=False,
                records_validated=records_validated,
                records_passed=records_passed,
                records_failed=records_failed,
                errors=errors,
                validation_time=validation_time,
                timestamp=start_time,
            )

    def _load_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """Load dataset from JSONL file."""
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _validate_chatml_record(self, record: Dict[str, Any]) -> bool:
        """Validate a ChatML format record."""
        # Check required fields
        required_fields = self.pipeline_schema["required_fields"]
        for field in required_fields:
            if field not in record:
                raise ValueError(f"Missing required field: {field}")

        # Validate messages structure
        messages = record.get("messages", [])
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("Messages must be a non-empty list")

        valid_roles = self.pipeline_schema["message_structure"]["role"]
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError("Each message must be a dict")
            if "role" not in msg:
                raise ValueError("Message missing 'role' field")
            if msg["role"] not in valid_roles:
                raise ValueError(f"Invalid role: {msg['role']}. Must be one of {valid_roles}")
            if "content" not in msg:
                raise ValueError("Message missing 'content' field")
            if not isinstance(msg["content"], str):
                raise ValueError("Message content must be a string")
            if not msg["content"].strip():
                raise ValueError("Message content cannot be empty")

        # Validate ID
        if not isinstance(record.get("id"), str):
            raise ValueError("ID must be a string")

        # Validate source
        if not isinstance(record.get("source"), str):
            raise ValueError("Source must be a string")

        return True

    def _validate_conversation_record(self, record: Dict[str, Any]) -> bool:
        """Validate a ConversationRecord format record."""
        # Check required fields
        required_fields = self.pipeline_schema["alternate_format"]["required_fields"]
        for field in required_fields:
            if field not in record:
                raise ValueError(f"Missing required field: {field}")

        # Validate turns structure
        turns = record.get("turns", [])
        if not isinstance(turns, list) or len(turns) < 2:
            raise ValueError("Turns must be a list with at least 2 items")

        for turn in turns:
            if not isinstance(turn, dict):
                raise ValueError("Each turn must be a dict")
            if "speaker_id" not in turn:
                raise ValueError("Turn missing 'speaker_id' field")
            if "content" not in turn:
                raise ValueError("Turn missing 'content' field")
            if not isinstance(turn["content"], str):
                raise ValueError("Turn content must be a string")
            if not turn["content"].strip():
                raise ValueError("Turn content cannot be empty")

        # Validate source_type
        if not isinstance(record.get("source_type"), str):
            raise ValueError("source_type must be a string")

        return True


class DatasetMerger:
    """Merges and deduplicates datasets with existing training pipeline data."""

    def __init__(self, similarity_threshold: float = 0.85):
        """Initialize the dataset merger."""
        self.similarity_threshold = similarity_threshold
        logger.info("Initialized Dataset Merger")

    def merge_datasets(
        self,
        new_dataset_path: str,
        existing_dataset_path: str,
        output_path: str,
        target_format: str = "chatml",
    ) -> MergeResult:
        """
        Merge new dataset with existing dataset and remove duplicates.

        Args:
            new_dataset_path: Path to new dataset (JSONL)
            existing_dataset_path: Path to existing dataset (JSONL)
            output_path: Path to save merged dataset
            target_format: Target format ("chatml" or "conversation_record")

        Returns:
            MergeResult with merging statistics
        """
        logger.info(
            f"Merging datasets: {new_dataset_path} + {existing_dataset_path} -> {output_path}"
        )
        start_time = datetime.now(timezone.utc)

        errors = []
        warnings = []
        duplicates_removed = 0
        conflicts_resolved = 0

        try:
            # Load datasets
            new_records = self._load_dataset(new_dataset_path)
            existing_records = self._load_dataset(existing_dataset_path)

            logger.info(
                f"Loaded {len(new_records)} new records and {len(existing_records)} existing records"
            )

            # Create content hashes for deduplication
            existing_hashes = self._create_content_hashes(existing_records, target_format)
            new_hashes = self._create_content_hashes(new_records, target_format)

            # Find duplicates (compare hash keys, not values)
            duplicate_indices = []
            existing_hash_set = set(existing_hashes.keys())
            for record_hash in new_hashes.keys():
                if record_hash in existing_hash_set:
                    duplicate_indices.append(new_hashes[record_hash])
                    duplicates_removed += 1

            # Remove duplicates from new records
            unique_new_records = [
                record
                for idx, record in enumerate(new_records)
                if idx not in duplicate_indices
            ]

            logger.info(
                f"Removed {duplicates_removed} duplicates, {len(unique_new_records)} unique new records"
            )

            # Find similar records (fuzzy deduplication)
            similar_records = self._find_similar_records(
                unique_new_records, existing_records, target_format
            )

            # Remove similar records
            final_new_records = [
                record
                for idx, record in enumerate(unique_new_records)
                if idx not in similar_records
            ]

            similar_removed = len(unique_new_records) - len(final_new_records)
            duplicates_removed += similar_removed

            logger.info(
                f"Removed {similar_removed} similar records, {len(final_new_records)} final new records"
            )

            # Merge datasets (existing first, then new)
            merged_records = existing_records + final_new_records

            # Resolve conflicts (if any)
            conflicts_resolved = self._resolve_conflicts(merged_records, target_format)

            # Save merged dataset
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self._save_merged_dataset(merged_records, output_path, target_format)

            merge_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            result = MergeResult(
                success=len(errors) == 0,
                records_merged=len(merged_records),
                duplicates_removed=duplicates_removed,
                conflicts_resolved=conflicts_resolved,
                output_path=output_path,
                errors=errors,
                warnings=warnings,
                merge_time=merge_time,
                timestamp=start_time,
            )

            logger.info(
                f"Merge complete: {len(merged_records)} records merged, "
                f"{duplicates_removed} duplicates removed in {merge_time:.2f}s"
            )
            return result

        except Exception as e:
            merge_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            errors.append(f"Merge failed: {str(e)}")
            logger.error(f"Dataset merge failed: {e}", exc_info=True)

            return MergeResult(
                success=False,
                records_merged=0,
                duplicates_removed=duplicates_removed,
                conflicts_resolved=conflicts_resolved,
                output_path=output_path,
                errors=errors,
                warnings=warnings,
                merge_time=merge_time,
                timestamp=start_time,
            )

    def _load_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """Load dataset from JSONL file."""
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _create_content_hashes(
        self, records: List[Dict[str, Any]], target_format: str
    ) -> Dict[str, int]:
        """Create content hashes for deduplication."""
        hashes = {}
        for idx, record in enumerate(records):
            # Extract content for hashing
            content = self._extract_content_for_hashing(record, target_format)
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            hashes[content_hash] = idx
        return hashes

    def _extract_content_for_hashing(
        self, record: Dict[str, Any], target_format: str
    ) -> str:
        """Extract content from record for hashing."""
        if target_format == "chatml":
            messages = record.get("messages", [])
            content_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                content_parts.append(f"{role}:{content}")
            return "|".join(content_parts)
        elif target_format == "conversation_record":
            turns = record.get("turns", [])
            content_parts = []
            for turn in turns:
                speaker = turn.get("speaker_id", "")
                content = turn.get("content", "")
                content_parts.append(f"{speaker}:{content}")
            return "|".join(content_parts)
        else:
            # Fallback: use entire record as string
            return json.dumps(record, sort_keys=True)

    def _find_similar_records(
        self,
        new_records: List[Dict[str, Any]],
        existing_records: List[Dict[str, Any]],
        target_format: str,
    ) -> List[int]:
        """Find similar records using content similarity."""
        similar_indices = []

        for new_idx, new_record in enumerate(new_records):
            new_content = self._extract_content_for_hashing(new_record, target_format)
            new_words = set(new_content.lower().split())

            for existing_record in existing_records:
                existing_content = self._extract_content_for_hashing(
                    existing_record, target_format
                )
                existing_words = set(existing_content.lower().split())

                # Calculate Jaccard similarity
                if len(new_words) == 0 or len(existing_words) == 0:
                    continue

                intersection = len(new_words.intersection(existing_words))
                union = len(new_words.union(existing_words))
                similarity = intersection / union if union > 0 else 0.0

                if similarity >= self.similarity_threshold:
                    similar_indices.append(new_idx)
                    break

        return similar_indices

    def _resolve_conflicts(
        self, records: List[Dict[str, Any]], target_format: str
    ) -> int:
        """Resolve conflicts in merged dataset (e.g., duplicate IDs)."""
        conflicts_resolved = 0
        seen_ids = set()

        for record in records:
            record_id = record.get("id")
            if record_id and record_id in seen_ids:
                # Generate new ID for duplicate
                record["id"] = str(uuid.uuid4())
                conflicts_resolved += 1
            elif record_id:
                seen_ids.add(record_id)

        return conflicts_resolved

    def _save_merged_dataset(
        self, records: List[Dict[str, Any]], output_path: str, target_format: str
    ) -> None:
        """Save merged dataset to file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


class QualityChecker:
    """Performs quality checks on integrated datasets."""

    def __init__(self):
        """Initialize the quality checker."""
        logger.info("Initialized Quality Checker")

    def check_quality(
        self, dataset_path: str, target_format: str = "chatml"
    ) -> QualityCheckResult:
        """
        Perform quality checks on integrated dataset.

        Args:
            dataset_path: Path to dataset file (JSONL)
            target_format: Target format ("chatml" or "conversation_record")

        Returns:
            QualityCheckResult with quality statistics
        """
        logger.info(f"Checking quality of dataset {dataset_path}")
        start_time = datetime.now(timezone.utc)

        errors = []
        records_checked = 0
        records_passed = 0
        records_failed = 0
        pii_detected = 0
        structure_issues = 0
        completeness_issues = 0

        try:
            # Load dataset
            records = self._load_dataset(dataset_path)

            # Check each record
            for idx, record in enumerate(records):
                records_checked += 1
                record_errors = []

                # Check for PII
                if self._check_pii(record, target_format):
                    pii_detected += 1
                    record_errors.append("PII detected")

                # Check structure
                if not self._check_structure(record, target_format):
                    structure_issues += 1
                    record_errors.append("Structure issue")

                # Check completeness
                if not self._check_completeness(record, target_format):
                    completeness_issues += 1
                    record_errors.append("Completeness issue")

                if record_errors:
                    records_failed += 1
                    errors.append({
                        "record_index": idx,
                        "record_id": record.get("id", "unknown"),
                        "errors": record_errors,
                    })
                else:
                    records_passed += 1

            # Calculate quality score
            quality_score = (
                records_passed / records_checked if records_checked > 0 else 0.0
            )

            check_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            result = QualityCheckResult(
                passed=records_failed == 0,
                records_checked=records_checked,
                records_passed=records_passed,
                records_failed=records_failed,
                pii_detected=pii_detected,
                structure_issues=structure_issues,
                completeness_issues=completeness_issues,
                errors=errors,
                quality_score=quality_score,
                check_time=check_time,
                timestamp=start_time,
            )

            logger.info(
                f"Quality check complete: {records_passed} passed, "
                f"{records_failed} failed, quality score: {quality_score:.2f}"
            )
            return result

        except Exception as e:
            check_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            errors.append({
                "error": f"Quality check failed: {str(e)}",
            })
            logger.error(f"Quality check failed: {e}", exc_info=True)

            return QualityCheckResult(
                passed=False,
                records_checked=records_checked,
                records_passed=records_passed,
                records_failed=records_failed,
                pii_detected=pii_detected,
                structure_issues=structure_issues,
                completeness_issues=completeness_issues,
                errors=errors,
                quality_score=0.0,
                check_time=check_time,
                timestamp=start_time,
            )

    def _load_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """Load dataset from JSONL file."""
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _check_pii(self, record: Dict[str, Any], target_format: str) -> bool:
        """Check for PII in record."""
        # Basic PII patterns
        pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{3}\.\d{2}\.\d{4}\b",  # SSN
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b\d{3}-\d{3}-\d{4}\b",  # Phone
            r"\b\d{10}\b",  # Phone (10 digits)
            r"\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b",  # Credit card
            r"\b\d{16}\b",  # Credit card (16 digits)
        ]

        # Extract text content
        text_content = self._extract_text_content(record, target_format)

        # Check for PII patterns
        for pattern in pii_patterns:
            if re.search(pattern, text_content, re.IGNORECASE):
                return True

        return False

    def _extract_text_content(self, record: Dict[str, Any], target_format: str) -> str:
        """Extract text content from record for PII checking."""
        text_parts = []

        if target_format == "chatml":
            messages = record.get("messages", [])
            for msg in messages:
                content = msg.get("content", "")
                text_parts.append(content)
        elif target_format == "conversation_record":
            turns = record.get("turns", [])
            for turn in turns:
                content = turn.get("content", "")
                text_parts.append(content)

        return " ".join(text_parts)

    def _check_structure(self, record: Dict[str, Any], target_format: str) -> bool:
        """Check record structure."""
        try:
            if target_format == "chatml":
                # Check required fields
                if "id" not in record or "source" not in record or "messages" not in record:
                    return False

                # Check messages structure
                messages = record.get("messages", [])
                if not isinstance(messages, list) or len(messages) == 0:
                    return False

                for msg in messages:
                    if "role" not in msg or "content" not in msg:
                        return False
                    if not isinstance(msg["content"], str) or not msg["content"].strip():
                        return False

            elif target_format == "conversation_record":
                # Check required fields
                if "id" not in record or "turns" not in record:
                    return False

                # Check turns structure
                turns = record.get("turns", [])
                if not isinstance(turns, list) or len(turns) < 2:
                    return False

                for turn in turns:
                    if "speaker_id" not in turn or "content" not in turn:
                        return False
                    if not isinstance(turn["content"], str) or not turn["content"].strip():
                        return False

            return True

        except Exception:
            return False

    def _check_completeness(self, record: Dict[str, Any], target_format: str) -> bool:
        """Check record completeness."""
        try:
            if target_format == "chatml":
                messages = record.get("messages", [])
                # Check that we have at least user and assistant messages
                roles = [msg.get("role") for msg in messages]
                if "user" not in roles or "assistant" not in roles:
                    return False

                # Check that messages have meaningful content
                for msg in messages:
                    content = msg.get("content", "")
                    if len(content.strip()) < 10:  # Minimum content length
                        return False

            elif target_format == "conversation_record":
                turns = record.get("turns", [])
                # Check that we have at least 2 turns
                if len(turns) < 2:
                    return False

                # Check that turns have meaningful content
                for turn in turns:
                    content = turn.get("content", "")
                    if len(content.strip()) < 10:  # Minimum content length
                        return False

            return True

        except Exception:
            return False

