#!/usr/bin/env python3
"""
Production-Ready Dataset Export with Tiered Access
Exports datasets in multiple formats with access control and versioning.
"""

import csv
import hashlib
import json
import logging
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# Optional imports for specific export formats
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from datasets import Dataset
except ImportError:
    Dataset = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats."""

    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    HUGGINGFACE = "huggingface"
    JSONL = "jsonl"
    OPENAI_FINE_TUNING = "openai_fine_tuning"


class AccessTier(Enum):
    """Dataset access tiers."""

    PRIORITY = ("priority", 1)
    PROFESSIONAL = ("professional", 2)
    COT = ("cot", 3)
    REDDIT = ("reddit", 4)
    RESEARCH = ("research", 5)
    ARCHIVE = ("archive", 6)

    def __init__(self, tier_name: str, priority: int):
        self.tier_name = tier_name
        self.priority = priority


@dataclass
class ExportMetadata:
    """Export metadata information."""

    export_id: str
    version: str
    format: ExportFormat
    access_tier: AccessTier
    total_conversations: int
    quality_threshold: float
    export_timestamp: datetime
    file_paths: list[str] = field(default_factory=list)
    checksums: dict[str, str] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportConfig:
    """Export configuration."""

    formats: list[ExportFormat]
    access_tiers: list[AccessTier]
    output_directory: str
    include_metadata: bool = True
    compress_output: bool = True
    validate_export: bool = True
    max_conversations_per_file: int = 10000
    quality_threshold: float = 0.7


class ProductionExporter:
    """
    Production-ready dataset export system with tiered access.
    """

    def __init__(self, base_output_dir: str = "./exports"):
        """Initialize the production exporter."""
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        self.export_history: list[ExportMetadata] = []
        self.version_counter = 1

        # Load export history if exists
        self._load_export_history()

    def export_dataset(
        self, conversations: list[dict[str, Any]], config: ExportConfig
    ) -> list[ExportMetadata]:
        """
        Export dataset with specified configuration.

        Args:
            conversations: list of conversations to export
            config: Export configuration

        Returns:
            list of export metadata for each tier/format combination
        """
        logger.info(f"Starting dataset export with {len(conversations)} conversations")

        export_results = []

        # Filter and organize conversations by tier
        tiered_conversations = self._organize_by_tier(conversations, config.access_tiers)

        # Export each tier in each format
        for access_tier in config.access_tiers:
            tier_conversations = tiered_conversations.get(access_tier, [])

            if not tier_conversations:
                logger.warning(f"No conversations found for tier {access_tier.tier_name}")
                continue

            # Filter by quality threshold
            filtered_conversations = self._filter_by_quality(
                tier_conversations, config.quality_threshold
            )

            logger.info(
                f"Exporting {len(filtered_conversations)} conversations for tier {access_tier.tier_name}"
            )

            for export_format in config.formats:
                try:
                    metadata = self._export_tier_format(
                        filtered_conversations, access_tier, export_format, config
                    )
                    export_results.append(metadata)

                except Exception as e:
                    logger.error(
                        f"Failed to export tier {access_tier.tier_name} in format {export_format.value}: {e}"
                    )

        # Save export history
        self.export_history.extend(export_results)
        self._save_export_history()

        logger.info(f"Dataset export completed. Generated {len(export_results)} export packages")
        return export_results

    def _organize_by_tier(
        self, conversations: list[dict[str, Any]], access_tiers: list[AccessTier]
    ) -> dict[AccessTier, list[dict[str, Any]]]:
        """Organize conversations by access tier."""
        tiered_conversations = {tier: [] for tier in access_tiers}

        for conversation in conversations:
            # Determine conversation tier based on metadata
            conv_tier = self._determine_conversation_tier(conversation)

            # Add to appropriate tiers (hierarchical access)
            for tier in access_tiers:
                if tier.priority >= conv_tier.priority:
                    tiered_conversations[tier].append(conversation)

        return tiered_conversations

    def _determine_conversation_tier(self, conversation: dict[str, Any]) -> AccessTier:
        """Determine the access tier for a conversation."""
        metadata = conversation.get("metadata", {})
        source = metadata.get("source", "").lower()
        quality_score = metadata.get("quality_score", 0.0)

        # Tier determination logic
        if source == "priority" or quality_score >= 0.95:
            return AccessTier.PRIORITY
        if source == "cot" or "chain_of_thought" in source:
            return AccessTier.COT
        if source == "reddit":
            return AccessTier.REDDIT
        return AccessTier.RESEARCH if source == "research" else AccessTier.PROFESSIONAL

    def _filter_by_quality(
        self, conversations: list[dict[str, Any]], quality_threshold: float
    ) -> list[dict[str, Any]]:
        """Filter conversations by quality threshold."""
        filtered = []

        for conversation in conversations:
            quality_score = conversation.get("metadata", {}).get("quality_score", 0.0)
            if quality_score >= quality_threshold:
                filtered.append(conversation)

        return filtered

    def _export_tier_format(
        self,
        conversations: list[dict[str, Any]],
        access_tier: AccessTier,
        export_format: ExportFormat,
        config: ExportConfig,
    ) -> ExportMetadata:
        """Export conversations for a specific tier and format."""
        # Generate export metadata
        export_id = f"{access_tier.tier_name}_{export_format.value}_{self.version_counter}"
        version = f"v{self.version_counter}.0"

        # Create output directory
        output_dir = self.base_output_dir / version / access_tier.tier_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export based on format
        file_paths = []
        checksums = {}

        if export_format == ExportFormat.JSON:
            file_paths, checksums = self._export_json(conversations, output_dir, config)
        elif export_format == ExportFormat.CSV:
            file_paths, checksums = self._export_csv(conversations, output_dir, config)
        elif export_format == ExportFormat.PARQUET:
            file_paths, checksums = self._export_parquet(conversations, output_dir, config)
        elif export_format == ExportFormat.HUGGINGFACE:
            file_paths, checksums = self._export_huggingface(conversations, output_dir, config)
        elif export_format == ExportFormat.JSONL:
            file_paths, checksums = self._export_jsonl(conversations, output_dir, config)
        elif export_format == ExportFormat.OPENAI_FINE_TUNING:
            file_paths, checksums = self._export_openai_fine_tuning(
                conversations, output_dir, config
            )

        # Calculate statistics
        statistics = self._calculate_export_statistics(conversations)

        # Create metadata
        metadata = ExportMetadata(
            export_id=export_id,
            version=version,
            format=export_format,
            access_tier=access_tier,
            total_conversations=len(conversations),
            quality_threshold=config.quality_threshold,
            export_timestamp=datetime.now(timezone.utc),
            file_paths=file_paths,
            checksums=checksums,
            statistics=statistics,
        )

        # Save metadata file
        if config.include_metadata:
            metadata_path = output_dir / f"metadata_{export_format.value}.json"
            self._save_metadata(metadata, metadata_path)
            file_paths.append(str(metadata_path))

        # Compress if requested
        if config.compress_output:
            compressed_path = self._compress_export(output_dir, export_id)
            metadata.file_paths = [compressed_path]

        # Validate export if requested
        if config.validate_export:
            self._validate_export(metadata, conversations)

        return metadata

    def _export_json(
        self, conversations: list[dict[str, Any]], output_dir: Path, config: ExportConfig
    ) -> tuple[list[str], dict[str, str]]:
        """Export conversations as JSON."""
        file_paths = []
        checksums = {}

        # Split into chunks if needed
        chunks = self._chunk_conversations(conversations, config.max_conversations_per_file)

        for i, chunk in enumerate(chunks):
            filename = (
                f"conversations_part_{i + 1:03d}.json" if len(chunks) > 1 else "conversations.json"
            )
            file_path = output_dir / filename

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(chunk, f, indent=2, ensure_ascii=False)

            file_paths.append(str(file_path))
            checksums[str(file_path)] = self._calculate_checksum(file_path)

        return file_paths, checksums

    def _export_csv(
        self, conversations: list[dict[str, Any]], output_dir: Path, config: ExportConfig
    ) -> tuple[list[str], dict[str, str]]:
        """Export conversations as CSV."""
        file_paths = []
        checksums = {}

        # Flatten conversations for CSV
        flattened_data = self._flatten_conversations_for_csv(conversations)

        # Split into chunks if needed
        chunks = self._chunk_conversations(flattened_data, config.max_conversations_per_file)

        for i, chunk in enumerate(chunks):
            filename = (
                f"conversations_part_{i + 1:03d}.csv" if len(chunks) > 1 else "conversations.csv"
            )
            file_path = output_dir / filename

            if chunk:
                fieldnames = chunk[0].keys()
                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(chunk)

                file_paths.append(str(file_path))
                checksums[str(file_path)] = self._calculate_checksum(file_path)

        return file_paths, checksums

    def _export_parquet(
        self, conversations: list[dict[str, Any]], output_dir: Path, config: ExportConfig
    ) -> tuple[list[str], dict[str, str]]:
        """Export conversations as Parquet (requires pandas and pyarrow)."""
        if pd is None:
            logger.warning("pandas not available, falling back to JSON export")
            return self._export_json(conversations, output_dir, config)

        try:
            return self._export_parquet_data(conversations, config, output_dir)
        except ImportError:
            logger.warning("pandas/pyarrow not available, falling back to JSON export")
            return self._export_json(conversations, output_dir, config)

    def _export_parquet_data(self, conversations, config, output_dir):
        # Assert that pandas is available (checked in calling method)
        assert pd is not None, "pandas is required for parquet export"

        file_paths = []
        checksums = {}

        # Flatten conversations for tabular format
        flattened_data = self._flatten_conversations_for_csv(conversations)

        # Split into chunks if needed
        chunks = self._chunk_conversations(flattened_data, config.max_conversations_per_file)

        for i, chunk in enumerate(chunks):
            filename = (
                f"conversations_part_{i + 1:03d}.parquet"
                if len(chunks) > 1
                else "conversations.parquet"
            )
            file_path = output_dir / filename

            if chunk:
                df = pd.DataFrame(chunk)
                df.to_parquet(file_path, index=False)

                file_paths.append(str(file_path))
                checksums[str(file_path)] = self._calculate_checksum(file_path)

        return file_paths, checksums

    def _export_huggingface(
        self, conversations: list[dict[str, Any]], output_dir: Path, config: ExportConfig
    ) -> tuple[list[str], dict[str, str]]:
        """Export conversations in HuggingFace datasets format."""
        if Dataset is None:
            logger.warning("datasets library not available, falling back to JSONL export")
            return self._export_jsonl(conversations, output_dir, config)

        try:
            return self._export_huggingface_data(conversations, output_dir)
        except ImportError:
            logger.warning("datasets library not available, falling back to JSONL export")
            return self._export_jsonl(conversations, output_dir, config)

    def _export_huggingface_data(self, conversations, output_dir):
        # Assert that datasets library is available (checked in calling method)
        assert Dataset is not None, "datasets library is required for HuggingFace export"

        # Prepare data for HuggingFace format
        hf_data = self._prepare_huggingface_format(conversations)

        # Create dataset
        dataset = Dataset.from_list(hf_data)

        # Save dataset
        dataset_path = output_dir / "dataset"
        dataset.save_to_disk(str(dataset_path))

        # Calculate checksums for all files in dataset directory
        file_paths = []
        checksums = {}

        for file_path in dataset_path.rglob("*"):
            if file_path.is_file():
                file_paths.append(str(file_path))
                checksums[str(file_path)] = self._calculate_checksum(file_path)

        return file_paths, checksums

    def _export_jsonl(
        self, conversations: list[dict[str, Any]], output_dir: Path, config: ExportConfig
    ) -> tuple[list[str], dict[str, str]]:
        """Export conversations as JSONL."""
        file_paths = []
        checksums = {}

        # Split into chunks if needed
        chunks = self._chunk_conversations(conversations, config.max_conversations_per_file)

        for i, chunk in enumerate(chunks):
            filename = (
                f"conversations_part_{i + 1:03d}.jsonl"
                if len(chunks) > 1
                else "conversations.jsonl"
            )
            file_path = output_dir / filename

            with open(file_path, "w", encoding="utf-8") as f:
                for conversation in chunk:
                    f.write(json.dumps(conversation, ensure_ascii=False) + "\n")

            file_paths.append(str(file_path))
            checksums[str(file_path)] = self._calculate_checksum(file_path)

        return file_paths, checksums

    def _export_openai_fine_tuning(
        self, conversations: list[dict[str, Any]], output_dir: Path, config: ExportConfig
    ) -> tuple[list[str], dict[str, str]]:
        """Export conversations in OpenAI fine-tuning format."""
        file_paths = []
        checksums = {}

        # Prepare data for OpenAI fine-tuning format
        openai_data = self._prepare_openai_fine_tuning_format(conversations)

        # Split into chunks if needed
        chunks = self._chunk_conversations(openai_data, config.max_conversations_per_file)

        for i, chunk in enumerate(chunks):
            filename = (
                f"conversations_part_{i + 1:03d}.jsonl"
                if len(chunks) > 1
                else "conversations.jsonl"
            )
            file_path = output_dir / filename

            with open(file_path, "w", encoding="utf-8") as f:
                for item in chunk:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            file_paths.append(str(file_path))
            checksums[str(file_path)] = self._calculate_checksum(file_path)

        return file_paths, checksums

    def _prepare_openai_fine_tuning_format(
        self, conversations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prepare conversations for OpenAI fine-tuning format."""
        openai_data = []

        for conversation in conversations:
            # Convert conversation to OpenAI chat format
            messages = []

            # Add system message if available
            if conversation.get("system_prompt"):
                messages.append({"role": "system", "content": conversation["system_prompt"]})

            # Convert turns to messages
            turns = conversation.get("turns", [])
            messages.extend(
                {
                    "role": turn.get("speaker", "user").lower(),
                    "content": turn.get("text", turn.get("content", "")),
                }
                for turn in turns
            )

            # Create OpenAI fine-tuning format item
            openai_item = {"messages": messages}

            # Add metadata as custom fields if needed
            metadata = conversation.get("metadata", {})
            if metadata:
                openai_item["metadata"] = [
                    {"quality_score": metadata.get("quality_score", 0.0)},
                    {"condition": metadata.get("condition", "")},
                    {"approach": metadata.get("approach", "")},
                    {"source": metadata.get("source", "")},
                ]

            openai_data.append(openai_item)

        return openai_data

    def _flatten_conversations_for_csv(
        self, conversations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Flatten conversations for CSV export."""
        flattened = []

        for conversation in conversations:
            flat_conv = {
                "id": conversation.get("id", ""),
                "content": conversation.get("content", ""),
                "quality_score": conversation.get("metadata", {}).get("quality_score", 0.0),
                "source": conversation.get("metadata", {}).get("source", ""),
                "condition": conversation.get("metadata", {}).get("condition", ""),
                "approach": conversation.get("metadata", {}).get("approach", ""),
                "turn_count": len(conversation.get("turns", [])),
                "timestamp": conversation.get("metadata", {}).get("timestamp", ""),
            }

            # Add turn information
            turns = conversation.get("turns", [])
            for i, turn in enumerate(turns[:5]):  # Limit to first 5 turns for CSV
                flat_conv[f"turn_{i + 1}_speaker"] = turn.get("speaker", "")
                flat_conv[f"turn_{i + 1}_text"] = turn.get("text", "")

            flattened.append(flat_conv)

        return flattened

    def _prepare_huggingface_format(
        self, conversations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prepare conversations for HuggingFace datasets format."""
        hf_data = []

        for conversation in conversations:
            hf_conv = {
                "id": conversation.get("id", ""),
                "text": conversation.get("content", ""),
                "turns": [
                    {"speaker": turn.get("speaker", ""), "text": turn.get("text", "")}
                    for turn in conversation.get("turns", [])
                ],
                "metadata": conversation.get("metadata", {}),
                "quality_score": conversation.get("metadata", {}).get("quality_score", 0.0),
            }
            hf_data.append(hf_conv)

        return hf_data

    def _chunk_conversations(self, conversations: list[Any], max_per_file: int) -> list[list[Any]]:
        """Split conversations into chunks."""
        if len(conversations) <= max_per_file:
            return [conversations]

        return [
            conversations[i : i + max_per_file] for i in range(0, len(conversations), max_per_file)
        ]

    def _count_distribution(self, items: list[str]) -> dict[str, int]:
        """Count the distribution of items."""
        counts = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return counts

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _calculate_export_statistics(self, conversations: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate export statistics."""
        if not conversations:
            return {}

        # Basic statistics
        total_conversations = len(conversations)
        total_turns = sum(len(conv.get("turns", [])) for conv in conversations)

        # Quality statistics
        quality_scores = [
            conv.get("metadata", {}).get("quality_score", 0.0) for conv in conversations
        ]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        min_quality = min(quality_scores, default=0.0)
        max_quality = max(quality_scores, default=0.0)

        # Source distribution
        sources = [conv.get("metadata", {}).get("source", "unknown") for conv in conversations]
        source_counts = self._count_distribution(sources)

        # Condition distribution
        conditions = [
            conv.get("metadata", {}).get("condition", "unknown") for conv in conversations
        ]
        condition_counts = self._count_distribution(conditions)

        return {
            "total_conversations": total_conversations,
            "total_turns": total_turns,
            "average_turns_per_conversation": total_turns / total_conversations
            if total_conversations > 0
            else 0,
            "quality_statistics": {
                "average": avg_quality,
                "minimum": min_quality,
                "maximum": max_quality,
            },
            "source_distribution": source_counts,
            "condition_distribution": condition_counts,
        }

    def _save_metadata(self, metadata: ExportMetadata, file_path: Path):
        """Save export metadata to file."""
        metadata_dict = {
            "export_id": metadata.export_id,
            "version": metadata.version,
            "format": metadata.format.value,
            "access_tier": metadata.access_tier.tier_name,
            "total_conversations": metadata.total_conversations,
            "quality_threshold": metadata.quality_threshold,
            "export_timestamp": metadata.export_timestamp.isoformat(),
            "file_paths": metadata.file_paths,
            "checksums": metadata.checksums,
            "statistics": metadata.statistics,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

    def _compress_export(self, output_dir: Path, export_id: str) -> str:
        """Compress export directory."""
        zip_path = output_dir.parent / f"{export_id}.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(output_dir)
                    zipf.write(file_path, arcname)

        # Remove original directory
        shutil.rmtree(output_dir)

        return str(zip_path)

    def _validate_export(
        self, metadata: ExportMetadata, original_conversations: list[dict[str, Any]]
    ):
        """Validate exported data."""
        logger.info(f"Validating export {metadata.export_id}")

        # Basic validation
        if metadata.total_conversations != len(original_conversations):
            logger.warning(f"Conversation count mismatch in export {metadata.export_id}")

        # File existence validation
        for file_path in metadata.file_paths:
            if not Path(file_path).exists():
                logger.error(f"Export file missing: {file_path}")

        # Checksum validation
        for file_path, expected_checksum in metadata.checksums.items():
            if Path(file_path).exists():
                actual_checksum = self._calculate_checksum(Path(file_path))
                if actual_checksum != expected_checksum:
                    logger.error(f"Checksum mismatch for {file_path}")

        logger.info(f"Export validation completed for {metadata.export_id}")

    def _load_export_history(self):
        """Load export history from file."""
        history_file = self.base_output_dir / "export_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    history_data = json.load(f)
                    self.version_counter = history_data.get("version_counter", 1)
                    # Note: Full history reconstruction would require more complex deserialization
            except Exception as e:
                logger.warning(f"Could not load export history: {e}")

    def _save_export_history(self):
        """Save export history to file."""
        history_file = self.base_output_dir / "export_history.json"

        history_data = {
            "version_counter": self.version_counter,
            "total_exports": len(self.export_history),
            "last_export": self.export_history[-1].export_timestamp.isoformat()
            if self.export_history
            else None,
        }

        with open(history_file, "w") as f:
            json.dump(history_data, f, indent=2)

        # Increment version counter for next export
        self.version_counter += 1

    def get_export_summary(self) -> dict[str, Any]:
        """Get export summary statistics."""
        if not self.export_history:
            return {"message": "No exports performed yet"}

        total_exports = len(self.export_history)
        total_conversations = sum(exp.total_conversations for exp in self.export_history)

        # Format distribution
        format_counts = {}
        for exp in self.export_history:
            format_counts[exp.format.value] = format_counts.get(exp.format.value, 0) + 1

        # Tier distribution
        tier_counts = {}
        for exp in self.export_history:
            tier_counts[exp.access_tier.tier_name] = (
                tier_counts.get(exp.access_tier.tier_name, 0) + 1
            )

        return {
            "total_exports": total_exports,
            "total_conversations_exported": total_conversations,
            "format_distribution": format_counts,
            "tier_distribution": tier_counts,
            "current_version": self.version_counter,
            "last_export": self.export_history[-1].export_timestamp.isoformat(),
        }


def main():
    """Example usage of the ProductionExporter."""
    exporter = ProductionExporter("./production_exports")

    # Sample conversations with different tiers
    sample_conversations = [
        {
            "id": "priority_001",
            "content": "High-quality therapeutic conversation with excellent clinical accuracy.",
            "turns": [
                {"speaker": "user", "text": "I need help with anxiety."},
                {
                    "speaker": "therapist",
                    "text": "I understand. Let's explore evidence-based techniques.",
                },
            ],
            "metadata": {
                "source": "priority",
                "quality_score": 0.98,
                "condition": "anxiety",
                "approach": "CBT",
                "timestamp": "2025-08-10T07:30:00Z",
            },
        },
        {
            "id": "reddit_001",
            "content": "Community-sourced conversation with moderate quality.",
            "turns": [
                {"speaker": "user", "text": "Feeling down lately."},
                {"speaker": "helper", "text": "That sounds tough. Want to talk about it?"},
            ],
            "metadata": {
                "source": "reddit",
                "quality_score": 0.75,
                "condition": "depression",
                "timestamp": "2025-08-10T07:25:00Z",
            },
        },
    ]

    # Configure export - Include all formats required for Task 3A.1
    config = ExportConfig(
        formats=[
            ExportFormat.JSON,
            ExportFormat.CSV,
            ExportFormat.PARQUET,
            ExportFormat.HUGGINGFACE,
            ExportFormat.JSONL,
            ExportFormat.OPENAI_FINE_TUNING,
        ],
        access_tiers=[AccessTier.PRIORITY, AccessTier.REDDIT, AccessTier.ARCHIVE],
        output_directory="./production_exports",
        include_metadata=True,
        compress_output=True,
        validate_export=True,
        quality_threshold=0.7,
    )

    # Perform export
    export_results = exporter.export_dataset(sample_conversations, config)

    # Print results
    logger.info("=== EXPORT COMPLETED ===")
    logger.info(f"Generated {len(export_results)} export packages")

    for result in export_results:
        logger.info(f"Export: {result.export_id}")
        logger.info(f"  Tier: {result.access_tier.tier_name}")
        logger.info(f"  Format: {result.format.value}")
        logger.info(f"  Conversations: {result.total_conversations}")
        logger.info(f"  Files: {len(result.file_paths)}")

    # Print summary
    logger.info("=== EXPORT SUMMARY ===")
    summary = exporter.get_export_summary()
    logger.info(f"Total Exports: {summary['total_exports']}")
    logger.info(f"Total Conversations: {summary['total_conversations']}")
    logger.info(f"Format Distribution: {summary['format_distribution']}")
    logger.info(f"Tier Distribution: {summary['tier_distribution']}")


if __name__ == "__main__":
    main()
