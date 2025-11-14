"""
Tier 6 Knowledge Base & Reference Materials Loader

Loads foundational therapeutic knowledge (DSM-5, psychology-10k, etc.)
and converts to training format (instruction-following examples).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

# Import conversation schema
import sys
from pathlib import Path as PathType

loader_path = PathType(__file__).parent
pipeline_root = loader_path.parent.parent.parent
sys.path.insert(0, str(pipeline_root))

try:
    from schemas.conversation_schema import Conversation, Message
except ImportError:
    try:
        from ai.dataset_pipeline.schemas.conversation_schema import Conversation, Message
    except ImportError:
        from conversation_schema import Conversation, Message

from ai.dataset_pipeline.ingestion.tier_loaders.base_tier_loader import (
    BaseTierLoader,
)

logger = logging.getLogger(__name__)


class Tier6KnowledgeLoader(BaseTierLoader):
    """
    Loader for Tier 6: Knowledge Base & Reference Materials.

    Tier 6 datasets are:
    - Foundational therapeutic knowledge
    - Reference materials (DSM-5, psychology textbooks)
    - Used for validation, not primary training data
    - Quality threshold: reference (not scored)
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        quality_threshold: float = 1.0,  # Reference quality
    ):
        """
        Initialize Tier 6 knowledge loader.

        Args:
            base_path: Optional base path to datasets directory
            quality_threshold: Quality threshold for Tier 6 (default: 1.0 = reference)
        """
        super().__init__(tier=6, quality_threshold=quality_threshold, base_path=base_path)
        self.base_path = Path(base_path) if base_path else Path("ai/datasets")
        self.training_data_path = Path("ai/training_data_consolidated")

        # Known Tier 6 knowledge base datasets
        self.dataset_paths = {
            "dsm5": self.training_data_path,  # DSM-5 already processed
            "psychology_10k": self.training_data_path / "psychology-10k",
            "psych_101": self.base_path / "Psych-101",
            "xmu_psych_books": self.base_path / "xmu_psych_books",
            "mental_health_snli": self.base_path / "customized-mental-health-snli2",
        }

        logger.info(
            f"Initialized Tier6KnowledgeLoader: quality_threshold={quality_threshold}"
        )

    def load_datasets(self) -> Dict[str, List[Conversation]]:
        """
        Load all Tier 6 knowledge base datasets.

        Converts knowledge base content to training format (instruction-following examples).

        Returns:
            Dictionary mapping dataset name to list of conversations
        """
        datasets = {}

        for dataset_name, dataset_path in self.dataset_paths.items():
            if not dataset_path.exists():
                logger.warning(f"Tier 6 dataset not found: {dataset_path}")
                continue

            logger.info(f"Loading Tier 6 knowledge base: {dataset_name}")

            try:
                conversations = self._load_knowledge_base(dataset_path, dataset_name)

                # Add tier metadata
                self.add_tier_metadata(conversations, {
                    "dataset_name": dataset_name,
                    "source": f"tier6_knowledge_{dataset_name}",
                    "data_type": "knowledge_base",
                    "usage": "reference_validation",
                })

                datasets[dataset_name] = conversations
                logger.info(
                    f"Loaded {len(conversations)} knowledge entries from {dataset_name}"
                )

            except Exception as e:
                logger.error(
                    f"Error loading Tier 6 dataset {dataset_name}: {e}",
                    exc_info=True,
                )
                continue

        total_conversations = sum(len(convs) for convs in datasets.values())
        logger.info(
            f"Tier 6 loading complete: {len(datasets)} datasets, "
            f"{total_conversations} total knowledge entries"
        )

        return datasets

    def _load_knowledge_base(
        self, knowledge_path: Path, dataset_name: str
    ) -> List[Conversation]:
        """
        Load knowledge base and convert to instruction-following format.

        Args:
            knowledge_path: Path to knowledge base directory or file
            dataset_name: Name of the knowledge base

        Returns:
            List of Conversation objects in instruction-following format
        """
        conversations = []

        # Look for processed knowledge files
        jsonl_files = list(knowledge_path.rglob("*.jsonl"))
        json_files = list(knowledge_path.rglob("*.json"))

        for file_path in jsonl_files + json_files:
            file_conversations = self._convert_knowledge_to_instructions(
                file_path, dataset_name
            )
            conversations.extend(file_conversations)

        return conversations

    def _convert_knowledge_to_instructions(
        self, file_path: Path, dataset_name: str
    ) -> List[Conversation]:
        """
        Convert knowledge base entries to instruction-following format.

        Args:
            file_path: Path to knowledge file
            dataset_name: Name of the knowledge base

        Returns:
            List of Conversation objects in instruction format
        """
        conversations = []

        # Load knowledge entries
        knowledge_entries = self.load_jsonl_file(file_path)

        # Convert each entry to instruction-following format
        for entry in knowledge_entries:
            # Extract knowledge content
            knowledge_text = ""
            if entry.messages:
                knowledge_text = " ".join(msg.content for msg in entry.messages)
            else:
                knowledge_text = str(entry.metadata.get("content", ""))

            if not knowledge_text:
                continue

            # Create instruction-following conversation
            # Format: "What is [concept]?" -> "[Knowledge content]"
            concept = entry.metadata.get("concept") or entry.metadata.get("title") or "this concept"
            instruction = f"What is {concept}?"

            conversation = Conversation(
                conversation_id=f"{dataset_name}_{len(conversations)}",
                source=f"tier6_knowledge_{dataset_name}",
                messages=[
                    Message(role="user", content=instruction),
                    Message(role="assistant", content=knowledge_text),
                ],
                metadata={
                    "tier": self.tier,
                    "quality_threshold": self.quality_threshold,
                    "knowledge_type": dataset_name,
                    "original_entry": entry.metadata,
                },
            )

            conversations.append(conversation)

        return conversations


