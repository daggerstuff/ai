"""
Dataset versioning system for production dataset management.
Manages dataset versions, metadata, and change tracking.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from conversation_schema import Conversation
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetVersion:
    """Dataset version information."""
    version_id: str
    version_number: str  # e.g., "1.0.0", "1.1.0"
    created_at: datetime
    description: str
    conversation_count: int
    quality_metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    file_paths: dict[str, str] = field(default_factory=dict)  # format -> path
    parent_version: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class VersionComparison:
    """Comparison between two dataset versions."""
    version_a: str
    version_b: str
    conversation_diff: dict[str, int]  # added, removed, modified
    quality_diff: dict[str, float]
    metadata_changes: list[str]


class DatasetVersioningSystem:
    """
    Dataset versioning system for production management.

    Manages dataset versions, tracks changes, and provides
    version control capabilities for therapeutic AI datasets.
    """

    def __init__(self, base_path: str = "./dataset_versions"):
        """Initialize the dataset versioning system."""
        self.logger = get_logger(__name__)

        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        self.versions_file = self.base_path / "versions.json"
        self.versions = self._load_versions()

        self.logger.info(f"DatasetVersioningSystem initialized at {self.base_path}")

    def create_version(self, conversations: list[Conversation],
                      version_number: str,
                      description: str,
                      export_formats: list[str] | None = None,
                      parent_version: str | None = None,
                      tags: list[str] | None = None) -> DatasetVersion:
        """Create a new dataset version."""
        if export_formats is None:
            export_formats = ["jsonl"]
        if tags is None:
            tags = []

        self.logger.info(f"Creating dataset version {version_number}")

        # Generate version ID
        version_id = self._generate_version_id(version_number, conversations)

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(conversations)

        # Create version directory
        version_dir = self.base_path / version_id
        version_dir.mkdir(exist_ok=True)

        # Export dataset in specified formats
        file_paths = {}
        for format_type in export_formats:
            file_path = version_dir / f"dataset.{format_type}"
            success = self._export_conversations(conversations, file_path, format_type)
            if success:
                file_paths[format_type] = str(file_path)

        # Create version metadata
        version = DatasetVersion(
            version_id=version_id,
            version_number=version_number,
            created_at=datetime.now(),
            description=description,
            conversation_count=len(conversations),
            quality_metrics=quality_metrics,
            metadata={
                "export_formats": export_formats,
                "total_messages": sum(len(conv.messages) for conv in conversations),
                "average_quality": quality_metrics.get("average_quality", 0),
                "creation_method": "manual"
            },
            file_paths=file_paths,
            parent_version=parent_version,
            tags=tags
        )

        # Save version metadata
        self._save_version_metadata(version)

        # Add to versions registry
        self.versions[version_id] = version
        self._save_versions()

        self.logger.info(f"Created dataset version {version_number} ({version_id})")
        return version

    def get_version(self, version_identifier: str) -> DatasetVersion | None:
        """Get version by ID or version number."""
        # Try by version ID first
        if version_identifier in self.versions:
            return self.versions[version_identifier]

        # Try by version number
        for version in self.versions.values():
            if version.version_number == version_identifier:
                return version

        return None

    def list_versions(self, tags: list[str] | None = None) -> list[DatasetVersion]:
        """List all versions, optionally filtered by tags."""
        versions = list(self.versions.values())

        if tags:
            versions = [
                version for version in versions
                if any(tag in version.tags for tag in tags)
            ]

        # Sort by creation date (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions

    def compare_versions(self, version_a_id: str, version_b_id: str) -> VersionComparison | None:
        """Compare two dataset versions."""
        version_a = self.get_version(version_a_id)
        version_b = self.get_version(version_b_id)

        if not version_a or not version_b:
            return None

        # Calculate conversation differences
        conversation_diff = {
            "version_a_count": version_a.conversation_count,
            "version_b_count": version_b.conversation_count,
            "difference": version_b.conversation_count - version_a.conversation_count
        }

        # Calculate quality differences
        quality_diff = {}
        for metric in version_a.quality_metrics:
            if metric in version_b.quality_metrics:
                quality_diff[metric] = version_b.quality_metrics[metric] - version_a.quality_metrics[metric]

        # Identify metadata changes
        metadata_changes = []
        for key in version_a.metadata:
            if key in version_b.metadata:
                if version_a.metadata[key] != version_b.metadata[key]:
                    metadata_changes.append(f"Changed {key}: {version_a.metadata[key]} -> {version_b.metadata[key]}")
            else:
                metadata_changes.append(f"Removed {key}")

        for key in version_b.metadata:
            if key not in version_a.metadata:
                metadata_changes.append(f"Added {key}: {version_b.metadata[key]}")

        return VersionComparison(
            version_a=version_a.version_id,
            version_b=version_b.version_id,
            conversation_diff=conversation_diff,
            quality_diff=quality_diff,
            metadata_changes=metadata_changes
        )

    def load_conversations(self, version_identifier: str, format_type: str = "jsonl") -> list[Conversation] | None:
        """Load conversations from a specific version."""
        version = self.get_version(version_identifier)
        if not version:
            return None

        file_path = version.file_paths.get(format_type)
        if not file_path or not Path(file_path).exists():
            return None

        try:
            conversations = []
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    conv_data = json.loads(line.strip())
                    conv = self._dict_to_conversation(conv_data)
                    conversations.append(conv)

            self.logger.info(f"Loaded {len(conversations)} conversations from version {version.version_number}")
            return conversations

        except Exception as e:
            self.logger.error(f"Failed to load conversations from version {version_identifier}: {e}")
            return None

    def delete_version(self, version_identifier: str) -> bool:
        """Delete a dataset version."""
        version = self.get_version(version_identifier)
        if not version:
            return False

        try:
            # Remove version directory
            version_dir = self.base_path / version.version_id
            if version_dir.exists():
                import shutil
                shutil.rmtree(version_dir)

            # Remove from registry
            del self.versions[version.version_id]
            self._save_versions()

            self.logger.info(f"Deleted version {version.version_number} ({version.version_id})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete version {version_identifier}: {e}")
            return False

    def _generate_version_id(self, version_number: str, conversations: list[Conversation]) -> str:
        """Generate unique version ID."""
        # Create hash based on version number and conversation content
        content_hash = hashlib.md5(
            f"{version_number}_{len(conversations)}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]

        return f"v_{version_number.replace('.', '_')}_{content_hash}"

    def _calculate_quality_metrics(self, conversations: list[Conversation]) -> dict[str, float]:
        """Calculate quality metrics for conversations."""
        if not conversations:
            return {}

        quality_scores = [conv.quality_score for conv in conversations if conv.quality_score is not None]

        return {
            "average_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "conversations_with_scores": len(quality_scores),
            "score_coverage": len(quality_scores) / len(conversations)
        }

    def _export_conversations(self, conversations: list[Conversation], file_path: Path, format_type: str) -> bool:
        """Export conversations to file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for conv in conversations:
                    conv_dict = self._conversation_to_dict(conv)
                    f.write(json.dumps(conv_dict, ensure_ascii=False, default=str) + "\\n")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export conversations: {e}")
            return False

    def _conversation_to_dict(self, conversation: Conversation) -> dict[str, Any]:
        """Convert conversation to dictionary."""
        return {
            "id": conversation.id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
                }
                for msg in conversation.messages
            ],
            "title": conversation.title,
            "metadata": conversation.metadata,
            "tags": conversation.tags,
            "quality_score": conversation.quality_score
        }

    def _dict_to_conversation(self, conv_dict: dict[str, Any]) -> Conversation:
        """Convert dictionary to conversation."""
        from conversation_schema import Message

        messages = []
        for msg_data in conv_dict.get("messages", []):
            timestamp = None
            if msg_data.get("timestamp"):
                timestamp = datetime.fromisoformat(msg_data["timestamp"])

            messages.append(Message(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=timestamp
            ))

        return Conversation(
            id=conv_dict["id"],
            messages=messages,
            title=conv_dict.get("title"),
            metadata=conv_dict.get("metadata", {}),
            tags=conv_dict.get("tags", []),
            quality_score=conv_dict.get("quality_score")
        )

    def _save_version_metadata(self, version: DatasetVersion) -> None:
        """Save version metadata to file."""
        version_dir = self.base_path / version.version_id
        metadata_file = version_dir / "metadata.json"

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump({
                "version_id": version.version_id,
                "version_number": version.version_number,
                "created_at": version.created_at.isoformat(),
                "description": version.description,
                "conversation_count": version.conversation_count,
                "quality_metrics": version.quality_metrics,
                "metadata": version.metadata,
                "file_paths": version.file_paths,
                "parent_version": version.parent_version,
                "tags": version.tags
            }, f, indent=2, ensure_ascii=False)

    def _load_versions(self) -> dict[str, DatasetVersion]:
        """Load versions registry."""
        if not self.versions_file.exists():
            return {}

        try:
            with open(self.versions_file, encoding="utf-8") as f:
                versions_data = json.load(f)

            versions = {}
            for version_id, version_data in versions_data.items():
                version = DatasetVersion(
                    version_id=version_data["version_id"],
                    version_number=version_data["version_number"],
                    created_at=datetime.fromisoformat(version_data["created_at"]),
                    description=version_data["description"],
                    conversation_count=version_data["conversation_count"],
                    quality_metrics=version_data["quality_metrics"],
                    metadata=version_data.get("metadata", {}),
                    file_paths=version_data.get("file_paths", {}),
                    parent_version=version_data.get("parent_version"),
                    tags=version_data.get("tags", [])
                )
                versions[version_id] = version

            return versions

        except Exception as e:
            self.logger.error(f"Failed to load versions: {e}")
            return {}

    def _save_versions(self) -> None:
        """Save versions registry."""
        try:
            versions_data = {}
            for version_id, version in self.versions.items():
                versions_data[version_id] = {
                    "version_id": version.version_id,
                    "version_number": version.version_number,
                    "created_at": version.created_at.isoformat(),
                    "description": version.description,
                    "conversation_count": version.conversation_count,
                    "quality_metrics": version.quality_metrics,
                    "metadata": version.metadata,
                    "file_paths": version.file_paths,
                    "parent_version": version.parent_version,
                    "tags": version.tags
                }

            with open(self.versions_file, "w", encoding="utf-8") as f:
                json.dump(versions_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Failed to save versions: {e}")


def validate_dataset_versioning_system():
    """Validate the DatasetVersioningSystem functionality."""
    try:
        versioning = DatasetVersioningSystem()
        assert hasattr(versioning, "create_version")
        assert hasattr(versioning, "get_version")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    if validate_dataset_versioning_system():
        pass
    else:
        pass
