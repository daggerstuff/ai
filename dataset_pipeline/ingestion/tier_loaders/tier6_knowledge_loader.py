"""
Tier 6 Knowledge Base & Reference Materials Loader

Loads foundational therapeutic knowledge (DSM-5, psychology-10k, etc.)
and converts to training format (instruction-following examples).
"""

import json
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
            "enhanced_psychology_knowledge": self.base_path.parent / "pixel" / "knowledge" / "enhanced_psychology_knowledge_base.json",
            "further_enhanced_psychology_knowledge": self.base_path.parent / "pixel" / "knowledge" / "further_enhanced_psychology_knowledge_base.json",
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

        # Handle single JSON file (like enhanced_psychology_knowledge_base.json)
        if knowledge_path.is_file() and knowledge_path.suffix == ".json":
            file_conversations = self._convert_json_knowledge_to_instructions(
                knowledge_path, dataset_name
            )
            conversations.extend(file_conversations)
        else:
            # Look for processed knowledge files in directory
            jsonl_files = list(knowledge_path.rglob("*.jsonl"))
            json_files = list(knowledge_path.rglob("*.json"))

            for file_path in jsonl_files + json_files:
                file_conversations = self._convert_knowledge_to_instructions(
                    file_path, dataset_name
                )
                conversations.extend(file_conversations)

        return conversations

    def _convert_json_knowledge_to_instructions(
        self, file_path: Path, dataset_name: str
    ) -> List[Conversation]:
        """
        Convert JSON knowledge base to instruction-following format.

        Args:
            file_path: Path to JSON knowledge base file
            dataset_name: Name of the knowledge base

        Returns:
            List of Conversation objects in instruction format
        """
        conversations = []

        try:
            # Load knowledge base from JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                knowledge_data = json.load(f)

            # Handle different JSON structures
            concepts = []
            if isinstance(knowledge_data, dict):
                # Check if it's the enhanced knowledge base format
                if "concepts" in knowledge_data:
                    concepts_data = knowledge_data["concepts"]
                    if isinstance(concepts_data, dict):
                        concepts = list(concepts_data.values())
                    elif isinstance(concepts_data, list):
                        concepts = concepts_data
                else:
                    # Assume it's a list of concepts directly
                    concepts = list(knowledge_data.values()) if isinstance(knowledge_data, dict) else knowledge_data
            elif isinstance(knowledge_data, list):
                concepts = knowledge_data

            # Convert each concept to instruction-following format
            for i, concept in enumerate(concepts):
                if not isinstance(concept, dict):
                    continue

                # Extract knowledge content
                concept_name = concept.get("name", concept.get("title", f"Concept {i}"))
                concept_definition = concept.get("definition", concept.get("content", concept.get("description", "")))

                if not concept_definition:
                    continue

                # Create instruction-following conversation
                # Format: "What is [concept]?" -> "[Knowledge content]"
                instruction = f"What is {concept_name}?"

                conversation = Conversation(
                    conversation_id=f"{dataset_name}_{i}",
                    source=f"tier6_knowledge_{dataset_name}",
                    messages=[
                        Message(role="user", content=instruction),
                        Message(role="assistant", content=concept_definition),
                    ],
                    metadata={
                        "tier": self.tier,
                        "quality_threshold": self.quality_threshold,
                        "knowledge_type": dataset_name,
                        "original_entry": concept,
                    },
                )

                conversations.append(conversation)

        except Exception as e:
            logger.error(f"Error converting JSON knowledge base {file_path}: {e}", exc_info=True)

        logger.info(f"Converted {len(conversations)} entries from JSON knowledge base {file_path}")
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


