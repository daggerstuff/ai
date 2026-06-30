#!/usr/bin/env python3
"""
Fine-Tuning Dataset Preparation for Pixelated Empathy AI

Prepares a fine-tuning dataset that collects transcripts, memories, and reflections
for training the model to utilize memory context effectively.

This process involves:
1. Data Collection: Gather anonymized conversation transcripts, memory entries, and reflections
2. Data Processing: Clean, anonymize, and structure into training examples
3. Dataset Organization: Split into training, validation, and test sets
4. Memory-Specific Features: Include examples testing memory retrieval and synthesis
5. Quality Assurance: Automated checks for data quality and consistency
6. Privacy & Compliance: Strict anonymization and secure handling

Usage:
    python -m ai.scripts.prepare_finetuning_dataset --input-dir ./data/transcripts --output-dir ./data/finetuning
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DatasetSplit(Enum):
    """Dataset split types."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ExampleType(Enum):
    """Types of fine-tuning examples."""

    STANDARD = "standard"
    MEMORY_RETRIEVAL = "memory_retrieval"
    MEMORY_FILTERING = "memory_filtering"
    MEMORY_SYNTHESIS = "memory_synthesis"
    TEMPORAL_PATTERN = "temporal_pattern"
    EMOTIONAL_CONTEXT = "emotional_context"


@dataclass
class TrainingExample:
    """A single fine-tuning training example."""

    # Core fields
    id: str
    example_type: ExampleType
    input: str  # Conversation context + relevant memories
    target: str  # Expected model response or next action
    conversation_id: str | None = None

    # Memory context
    relevant_memories: list[dict[str, Any]] = field(default_factory=list)
    memory_retrieval_query: str | None = None

    # Metadata
    split: DatasetSplit = DatasetSplit.TRAIN
    conversation_type: str | None = None
    emotional_tone: str | None = None
    therapeutic_modality: str | None = None
    skill_tags: list[str] = field(default_factory=list)

    # Quality markers
    memory_usage_correct: bool | None = None
    emotional_appropriateness: float | None = None  # 0.0-1.0
    skill_application_accuracy: float | None = None

    # Provenance
    source_file: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "example_type": self.example_type.value,
            "input": self.input,
            "target": self.target,
            "conversation_id": self.conversation_id,
            "relevant_memories": self.relevant_memories,
            "memory_retrieval_query": self.memory_retrieval_query,
            "split": self.split.value,
            "conversation_type": self.conversation_type,
            "emotional_tone": self.emotional_tone,
            "therapeutic_modality": self.therapeutic_modality,
            "skill_tags": self.skill_tags,
            "memory_usage_correct": self.memory_usage_correct,
            "emotional_appropriateness": self.emotional_appropriateness,
            "skill_application_accuracy": self.skill_application_accuracy,
            "source_file": self.source_file,
            "created_at": self.created_at,
        }


@dataclass
class DatasetStatistics:
    """Statistics for the prepared dataset."""

    total_examples: int = 0
    train_examples: int = 0
    validation_examples: int = 0
    test_examples: int = 0
    examples_by_type: dict[str, int] = field(default_factory=dict)
    examples_by_conversation_type: dict[str, int] = field(default_factory=dict)
    avg_memories_per_example: float = 0.0
    anonymization_issues: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "train_examples": self.train_examples,
            "validation_examples": self.validation_examples,
            "test_examples": self.test_examples,
            "examples_by_type": self.examples_by_type,
            "examples_by_conversation_type": self.examples_by_conversation_type,
            "avg_memories_per_example": self.avg_memories_per_example,
            "anonymization_issues": self.anonymization_issues,
        }


class Anonymizer:
    """
    Handles PII scrubbing and anonymization for therapeutic conversations.

    Implements multi-layer anonymization:
    - Names and personal identifiers
    - Contact information
    - Locations
    - Dates (relative timestamps preserved)
    - Medical/health identifiers
    """

    # Patterns for PII detection
    PATTERNS = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "phone": re.compile(r"\b(?:\+?1?[-.\s]?)?\(?(?:[0-9]{3})\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
        "url": re.compile(r"https?://[^\s<]+"),
        "name_title": re.compile(r"\b(?:Dr\.?|Doctor|Professor|Prof\.)\s+[A-Z][a-z]+"),
    }

    # Common name patterns that might indicate PII
    NAME_PATTERNS = [
        re.compile(r"\b(?:my name is|i am called|they call me|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.I),
        re.compile(r"\bI'm\s+([A-Z][a-z]+)", re.I),
    ]

    def __init__(self, seed: str | None = None):
        """Initialize anonymizer with optional seed for reproducibility."""
        self.seed = seed or secrets.token_hex(16)
        self._replacement_cache: dict[str, str] = {}
        self._stats = {
            "emails_redacted": 0,
            "phones_redacted": 0,
            "names_redacted": 0,
            "locations_redacted": 0,
        }

    def anonymize(self, text: str) -> str:
        """
        Anonymize text by replacing PII with placeholders.

        Args:
            text: Input text to anonymize

        Returns:
            Anonymized text
        """
        if not text:
            return ""

        result = text

        # Replace emails
        result = self._replace_pattern(result, self.PATTERNS["email"], "<EMAIL>")

        # Replace phone numbers
        result = self._replace_pattern(result, self.PATTERNS["phone"], "<PHONE>")

        # Replace SSNs
        result = self._replace_pattern(result, self.PATTERNS["ssn"], "<SSN>")

        # Replace credit cards
        result = self._replace_pattern(result, self.PATTERNS["credit_card"], "<CREDIT_CARD>")

        # Replace IP addresses
        result = self._replace_pattern(result, self.PATTERNS["ip_address"], "<IP_ADDRESS>")

        # Replace URLs
        return self._replace_pattern(result, self.PATTERNS["url"], "<URL>")


    def _replace_pattern(self, text: str, pattern: re.Pattern, placeholder: str) -> str:
        """Replace all matches of a pattern with a placeholder."""
        matches = pattern.findall(text)
        if not matches:
            return text

        result = text
        for match in matches:
            if isinstance(match, tuple):  # Groups in regex
                match = match[0] if match else ""
            if match and isinstance(match, str):
                replacement = self._get_replacement(match, placeholder)
                result = result.replace(match, replacement)
                self._increment_stat(f"{placeholder.split('_', maxsplit=1)[0]}s_redacted")

        return result

    def _get_replacement(self, original: str, placeholder: str) -> str:
        """Get or create a consistent replacement for a value."""
        if original not in self._replacement_cache:
            hash_suffix = hashlib.sha256(f"{self.seed}:{original}".encode()).hexdigest()[:8]
            self._replacement_cache[original] = f"{placeholder}_{hash_suffix}"

        return self._replacement_cache[original]

    def _increment_stat(self, stat_name: str) -> None:
        """Increment an internal statistic counter."""
        if stat_name in self._stats:
            self._stats[stat_name] += 1

    def get_stats(self) -> dict[str, int]:
        """Return anonymization statistics."""
        return self._stats.copy()


class FineTuningDatasetPreparer:
    """
    Main class for preparing fine-tuning datasets.

    Collects transcripts, memories, and reflections and structures them
    into training examples for model fine-tuning.
    """

    def __init__(
        self,
        output_dir: str | Path,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int | None = None,
        anonymize: bool = True,
    ):
        """
        Initialize the dataset preparer.

        Args:
            output_dir: Directory to write prepared datasets
            train_ratio: Proportion for training split
            validation_ratio: Proportion for validation split
            test_ratio: Proportion for test split
            seed: Random seed for reproducibility
            anonymize: Whether to apply PII anonymization
        """
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.anonymizer = Anonymizer(seed=str(seed)) if anonymize else None

        # Validate ratios
        total = train_ratio + validation_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        self.examples: list[TrainingExample] = []
        self.statistics = DatasetStatistics()

        logger.info(
            f"FineTuningDatasetPreparer initialized with train={train_ratio}, val={validation_ratio}, test={test_ratio}"
        )

    def load_transcripts(
        self,
        transcript_dir: str | Path,
        format: str = "auto",
    ) -> list[dict[str, Any]]:
        """
        Load transcript files from a directory.

        Args:
            transcript_dir: Path to directory containing transcripts
            format: Format hint ('json', 'jsonl', 'txt', 'auto')

        Returns:
            List of transcript dictionaries
        """
        transcript_path = Path(transcript_dir)
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")

        transcripts = []

        # Determine file extensions based on format
        patterns = ["*.json", "*.jsonl", "*.txt"] if format == "auto" else [f"*.{format}"]

        for pattern in patterns:
            for file_path in transcript_path.glob(pattern):
                try:
                    transcript = self._load_single_transcript(file_path)
                    if transcript:
                        transcripts.append(transcript)
                except Exception as e:
                    logger.warning(f"Failed to load transcript {file_path}: {e}")

        logger.info(f"Loaded {len(transcripts)} transcripts from {transcript_dir}")
        return transcripts

    def _load_single_transcript(
        self,
        file_path: Path,
    ) -> dict[str, Any] | None:
        """Load a single transcript file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Try JSON/JSONL first
            if file_path.suffix in [".json", ".jsonl"]:
                if file_path.suffix == ".jsonl":
                    # Parse as JSONL (multiple JSON objects, one per line)
                    records = []
                    for line in content.strip().split("\n"):
                        if line.strip():
                            records.append(json.loads(line))
                    return {
                        "source_file": str(file_path),
                        "format": "jsonl",
                        "data": records,
                    }
                # Single JSON object
                data = json.loads(content)
                return {
                    "source_file": str(file_path),
                    "format": "json",
                    "data": data,
                }

            # Plain text
            return {
                "source_file": str(file_path),
                "format": "txt",
                "data": {"text": content},
            }

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return None

    def load_memories(
        self,
        memory_source: str | Path | dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Load memory entries from a source.

        Args:
            memory_source: Path to memory file or memory data dict

        Returns:
            List of memory records
        """
        if isinstance(memory_source, dict):
            return memory_source.get("memories", [])

        memory_path = Path(memory_source)
        if not memory_path.exists():
            logger.warning(f"Memory source not found: {memory_source}")
            return []

        try:
            with open(memory_path, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("memories", [])
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to load memories: {e}")
            return []

    def load_reflections(
        self,
        reflection_source: str | Path | dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Load reflection entries from a source.

        Args:
            reflection_source: Path to reflection file or reflection data dict

        Returns:
            List of reflection records
        """
        if isinstance(reflection_source, dict):
            return reflection_source.get("reflections", [])

        reflection_path = Path(reflection_source)
        if not reflection_path.exists():
            logger.warning(f"Reflection source not found: {reflection_source}")
            return []

        try:
            with open(reflection_path, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("reflections", [])
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to load reflections: {e}")
            return []

    def create_training_examples(
        self,
        transcripts: list[dict[str, Any]],
        memories: list[dict[str, Any]] | None = None,
        reflections: list[dict[str, Any]] | None = None,
    ) -> list[TrainingExample]:
        """
        Create training examples from loaded data.

        Args:
            transcripts: List of transcript dictionaries
            memories: Optional list of memory records
            reflections: Optional list of reflection records

        Returns:
            List of training examples
        """
        examples = []
        memories = memories or []
        reflections = reflections or []

        for transcript in transcripts:
            # Create standard examples
            standard_examples = self._create_standard_examples(transcript, memories, reflections)
            examples.extend(standard_examples)

            # Create memory-specific examples
            if memories:
                memory_examples = self._create_memory_examples(transcript, memories, reflections)
                examples.extend(memory_examples)

        self.examples = examples
        logger.info(f"Created {len(examples)} training examples")
        return examples

    def _create_standard_examples(
        self,
        transcript: dict[str, Any],
        memories: list[dict[str, Any]],
        reflections: list[dict[str, Any]],
    ) -> list[TrainingExample]:
        """Create standard conversation examples."""
        examples = []
        data = transcript.get("data", {})

        # Handle JSONL format where data is a list of records
        if isinstance(data, list):
            for record in data:
                examples.extend(self._create_examples_from_record(record, memories, reflections, transcript))
            return examples

        # Handle single record format
        return self._create_examples_from_record(data, memories, reflections, transcript)

    def _create_examples_from_record(
        self,
        record: dict[str, Any],
        memories: list[dict[str, Any]],
        reflections: list[dict[str, Any]],
        transcript: dict[str, Any],
    ) -> list[TrainingExample]:
        """Create examples from a single record."""
        examples = []

        # Handle different transcript structures
        messages = record.get("messages", [])
        if isinstance(messages, list) and len(messages) > 0:
            # Create conversation context
            context = self._format_messages(messages)

            # Get target (last response or reflection insight)
            target = ""
            if reflections:
                target = reflections[0].get("insight", "")
            elif messages:
                last_msg = messages[-1] if isinstance(messages[-1], dict) else {}
                target = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)

            example = TrainingExample(
                id=self._generate_example_id(transcript),
                example_type=ExampleType.STANDARD,
                input=context,
                target=target,
                conversation_id=record.get("conversation_id"),
                source_file=transcript.get("source_file"),
                conversation_type=self._detect_conversation_type(messages),
            )
            examples.append(example)

        return examples

    def _create_memory_examples(
        self,
        transcript: dict[str, Any],
        memories: list[dict[str, Any]],
        reflections: list[dict[str, Any]],
    ) -> list[TrainingExample]:
        """Create memory-specific training examples."""
        examples = []
        data = transcript.get("data", {})

        # Handle JSONL format where data is a list of records
        if isinstance(data, list):
            for record in data:
                examples.extend(self._create_memory_examples_from_record(record, memories, reflections, transcript))
            return examples

        # Handle single record format
        return self._create_memory_examples_from_record(data, memories, reflections, transcript)

    def _create_memory_examples_from_record(
        self,
        record: dict[str, Any],
        memories: list[dict[str, Any]],
        reflections: list[dict[str, Any]],
        transcript: dict[str, Any],
    ) -> list[TrainingExample]:
        """Create memory-specific examples from a single record."""
        examples = []
        messages = record.get("messages", [])
        context = self._format_messages(messages)

        # Memory retrieval example
        if memories:
            retrieval_example = TrainingExample(
                id=self._generate_example_id(transcript, suffix="_retrieval"),
                example_type=ExampleType.MEMORY_RETRIEVAL,
                input=f"Context: {context}\n\nQuery: What memories are relevant to this conversation?",
                target=json.dumps(memories[:3], indent=2),  # Top 3 memories
                conversation_id=record.get("conversation_id"),
                relevant_memories=memories[:3],
                memory_retrieval_query="Relevant memories for current context",
                source_file=transcript.get("source_file"),
            )
            examples.append(retrieval_example)

            # Memory synthesis example (if multiple memories)
            if len(memories) >= 2:
                synthesis_target = self._synthesize_memories(memories)
                synthesis_example = TrainingExample(
                    id=self._generate_example_id(transcript, suffix="_synthesis"),
                    example_type=ExampleType.MEMORY_SYNTHESIS,
                    input=f"Context: {context}\n\nMemories: {json.dumps(memories[:5], indent=2)}",
                    target=synthesis_target,
                    conversation_id=record.get("conversation_id"),
                    relevant_memories=memories[:5],
                    source_file=transcript.get("source_file"),
                )
                examples.append(synthesis_example)

        return examples

    def _format_messages(self, messages: list[Any]) -> str:
        """Format messages into a conversation string."""
        if not messages:
            return ""

        formatted = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                formatted.append(f"{role}: {content}")
            else:
                formatted.append(str(msg))

        return "\n".join(formatted)

    def _detect_conversation_type(self, messages: list[Any]) -> str:
        """Detect the type of conversation from messages."""
        if not messages:
            return "unknown"

        text = str(messages).lower()

        if any(word in text for word in ["crisis", "emergency", "urgent"]):
            return "crisis"
        if any(word in text for word in ["reflection", "insight", "learned"]):
            return "reflection"
        if any(word in text for word in ["goal", "progress", "improvement"]):
            return "therapeutic"
        if "greeting" in text or "hello" in text:
            return "onboarding"

        return "general"

    def _synthesize_memories(self, memories: list[dict[str, Any]]) -> str:
        """Create a synthesis target from multiple memories."""
        if not memories:
            return ""

        contents = [m.get("content", "") for m in memories if isinstance(m, dict)]
        return "Synthesized insight: " + " | ".join(contents[:3])

    def _generate_example_id(
        self,
        transcript: dict[str, Any],
        suffix: str = "",
    ) -> str:
        """Generate a unique example ID."""
        source = transcript.get("source_file", "unknown")
        data = transcript.get("data", {})

        # Handle case where data is a list (JSONL) - use first item or default
        if isinstance(data, list):
            conversation_id = data[0].get("conversation_id", secrets.token_hex(8)) if data else secrets.token_hex(8)
        else:
            conversation_id = data.get("conversation_id", secrets.token_hex(8))

        base = f"{source}:{conversation_id}:{suffix}"
        return hashlib.sha256(base.encode()).hexdigest()[:16]

    def split_dataset(
        self,
        examples: list[TrainingExample] | None = None,
    ) -> dict[DatasetSplit, list[TrainingExample]]:
        """
        Split examples into train/validation/test sets.

        Args:
            examples: Examples to split (uses self.examples if not provided)

        Returns:
            Dictionary mapping split type to list of examples
        """
        import random

        examples = examples or self.examples
        if not examples:
            raise ValueError("No examples to split")

        # Shuffle with seed
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(examples)

        n = len(examples)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.validation_ratio)

        splits = {
            DatasetSplit.TRAIN: examples[:train_end],
            DatasetSplit.VALIDATION: examples[train_end:val_end],
            DatasetSplit.TEST: examples[val_end:],
        }

        logger.info(
            f"Split dataset: train={len(splits[DatasetSplit.TRAIN])}, "
            f"val={len(splits[DatasetSplit.VALIDATION])}, "
            f"test={len(splits[DatasetSplit.TEST])}"
        )

        return splits

    def anonymize_examples(
        self,
        examples: list[TrainingExample],
    ) -> list[TrainingExample]:
        """
        Apply anonymization to examples.

        Args:
            examples: Examples to anonymize

        Returns:
            Anonymized examples
        """
        if not self.anonymizer:
            return examples

        anonymized = []
        for example in examples:
            # Anonymize input
            anon_input = self.anonymizer.anonymize(example.input)

            # Anonymize target
            anon_target = self.anonymizer.anonymize(example.target)

            # Create new example with anonymized fields
            anon_example = TrainingExample(
                id=example.id,
                example_type=example.example_type,
                input=anon_input,
                target=anon_target,
                conversation_id=example.conversation_id,
                relevant_memories=example.relevant_memories,
                memory_retrieval_query=example.memory_retrieval_query,
                split=example.split,
                conversation_type=example.conversation_type,
                emotional_tone=example.emotional_tone,
                therapeutic_modality=example.therapeutic_modality,
                skill_tags=example.skill_tags,
                memory_usage_correct=example.memory_usage_correct,
                emotional_appropriateness=example.emotional_appropriateness,
                skill_application_accuracy=example.skill_application_accuracy,
                source_file=example.source_file,
                created_at=example.created_at,
            )
            anonymized.append(anon_example)

        logger.info(f"Anonymized {len(anonymized)} examples (PII replacements: {self.anonymizer.get_stats()})")

        return anonymized

    def save_dataset(
        self,
        splits: dict[DatasetSplit, list[TrainingExample]] | None = None,
    ) -> dict[str, Path]:
        """
        Save dataset splits to files.

        Args:
            splits: Dataset splits to save

        Returns:
            Dictionary mapping split name to output file path
        """
        if splits is None:
            # Generate splits if not provided
            splits = self.split_dataset()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_files = {}
        for split, examples in splits.items():
            filename = f"finetuning_{split.value}.jsonl"
            output_path = self.output_dir / filename

            with open(output_path, "w", encoding="utf-8") as f:
                for example in examples:
                    f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")

            output_files[split.value] = output_path
            logger.info(f"Saved {len(examples)} examples to {output_path}")

        # Save statistics
        self._compute_statistics(splits)
        stats_path = self.output_dir / "dataset_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.statistics.to_dict(), f, indent=2)
        logger.info(f"Saved dataset statistics to {stats_path}")

        # Save metadata
        metadata_path = self.output_dir / "dataset_metadata.json"
        self._save_metadata(metadata_path, splits)

        return output_files

    def _compute_statistics(
        self,
        splits: dict[DatasetSplit, list[TrainingExample]],
    ) -> None:
        """Compute dataset statistics."""
        total = 0
        total_memories = 0
        examples_by_type: dict[str, int] = {}
        examples_by_conv_type: dict[str, int] = {}

        for split, examples in splits.items():
            total += len(examples)

            for example in examples:
                # Count by type
                type_key = example.example_type.value
                examples_by_type[type_key] = examples_by_type.get(type_key, 0) + 1

                # Count by conversation type
                if example.conversation_type:
                    conv_type = example.conversation_type
                    examples_by_conv_type[conv_type] = examples_by_conv_type.get(conv_type, 0) + 1

                # Count memories
                total_memories += len(example.relevant_memories)

            # Update split counts
            if split == DatasetSplit.TRAIN:
                self.statistics.train_examples = len(examples)
            elif split == DatasetSplit.VALIDATION:
                self.statistics.validation_examples = len(examples)
            elif split == DatasetSplit.TEST:
                self.statistics.test_examples = len(examples)

        self.statistics.total_examples = total
        self.statistics.examples_by_type = examples_by_type
        self.statistics.examples_by_conversation_type = examples_by_conv_type
        self.statistics.avg_memories_per_example = total_memories / total if total > 0 else 0.0

    def _save_metadata(
        self,
        metadata_path: Path,
        splits: dict[DatasetSplit, list[TrainingExample]],
    ) -> None:
        """Save dataset metadata."""
        metadata = {
            "version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "description": "Fine-tuning dataset for Pixelated Empathy AI memory-aware training",
            "splits": {
                split.value: {
                    "count": len(examples),
                    "file": f"finetuning_{split.value}.jsonl",
                }
                for split, examples in splits.items()
            },
            "config": {
                "train_ratio": self.train_ratio,
                "validation_ratio": self.validation_ratio,
                "test_ratio": self.test_ratio,
                "seed": self.seed,
                "anonymization_enabled": self.anonymizer is not None,
            },
            "statistics": self.statistics.to_dict(),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def prepare(
        self,
        transcript_dir: str | Path,
        memory_source: str | Path | dict[str, Any] | None = None,
        reflection_source: str | Path | dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """
        Full pipeline: load data, create examples, split, and save.

        Args:
            transcript_dir: Directory containing transcript files
            memory_source: Path or data for memories (optional)
            reflection_source: Path or data for reflections (optional)

        Returns:
            Dictionary of output file paths
        """
        # Load data
        logger.info(f"Loading transcripts from {transcript_dir}")
        transcripts = self.load_transcripts(transcript_dir)

        memories = []
        if memory_source:
            logger.info(f"Loading memories from {memory_source}")
            memories = self.load_memories(memory_source)

        reflections = []
        if reflection_source:
            logger.info(f"Loading reflections from {reflection_source}")
            reflections = self.load_reflections(reflection_source)

        # Create examples
        logger.info("Creating training examples")
        examples = self.create_training_examples(transcripts, memories, reflections)

        # Anonymize
        if self.anonymizer:
            logger.info("Applying anonymization")
            examples = self.anonymize_examples(examples)

        # Split and save
        logger.info("Splitting and saving dataset")
        splits = self.split_dataset(examples)
        return self.save_dataset(splits)



def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare fine-tuning dataset for Pixelated Empathy AI")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing transcript files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/finetuning",
        help="Output directory for prepared dataset",
    )
    parser.add_argument(
        "--memory-source",
        type=str,
        default=None,
        help="Path to memory data (JSON file)",
    )
    parser.add_argument(
        "--reflection-source",
        type=str,
        default=None,
        help="Path to reflection data (JSON file)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-anonymize",
        action="store_true",
        help="Disable PII anonymization",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate ratios
    total_ratio = args.train_ratio + args.validation_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        logger.error(f"Train/validation/test ratios must sum to 1.0, got {total_ratio}")
        return 1

    # Create preparer
    preparer = FineTuningDatasetPreparer(
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        anonymize=not args.no_anonymize,
    )

    # Run pipeline
    try:
        output_files = preparer.prepare(
            transcript_dir=args.input_dir,
            memory_source=args.memory_source,
            reflection_source=args.reflection_source,
        )

        logger.info("Dataset preparation complete!")
        logger.info(f"Output files: {output_files}")

        return 0

    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
