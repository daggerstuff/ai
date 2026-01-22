"""
Tier 3 Chain-of-Thought Reasoning Dataset Loader

Loads CoT reasoning datasets for advanced therapeutic reasoning patterns.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from ai.dataset_pipeline.ingestion.tier_loaders.base_tier_loader import (
    BaseTierLoader,
)
from conversation_schema import Conversation

logger = logging.getLogger(__name__)


class Tier3CoTLoader(BaseTierLoader):
    """
    Loader for Tier 3: Chain-of-Thought Reasoning Datasets.

    Tier 3 datasets are:
    - Advanced therapeutic reasoning patterns
    - Chain-of-thought reasoning examples
    - Quality threshold: 90%
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        quality_threshold: float = 0.90,
        dataset_registry_path: str = "ai/data/dataset_registry.json",
    ):
        """
        Initialize Tier 3 CoT loader.

        Args:
            base_path: Optional base path to datasets directory
            quality_threshold: Quality threshold for Tier 3 (default: 0.90 = 90%)
            dataset_registry_path: Path to dataset_registry.json
        """
        super().__init__(
            tier=3,
            quality_threshold=quality_threshold,
            base_path=base_path,
            dataset_registry_path=dataset_registry_path,
        )
        self.base_path = Path(base_path) if base_path else Path("ai/datasets")

        # Consolidated dataset key in registry
        self.consolidated_key = "cot_reasoning_consolidated"
        self.dataset_paths = {
            self.consolidated_key: self.base_path / "cot_reasoning_filtered.json"
        }

        logger.info(
            f"Initialized Tier3CoTLoader: quality_threshold={quality_threshold}, "
            "configured for consolidated CoT dataset"
        )

    def load_datasets(self) -> Dict[str, List[Conversation]]:
        """
        Load all Tier 3 CoT reasoning datasets.

        Returns:
            Dictionary mapping dataset name to list of conversations
        """
        datasets = {}

        # Load consolidated dataset
        dataset_name = self.consolidated_key
        dataset_path = self.dataset_paths[dataset_name]

        # Ensure dataset is available locally (downloads from S3 if needed)
        try:
            dataset_path = self._ensure_dataset_locally(
                dataset_name, dataset_path, registry_category="cot_reasoning"
            )

            if not dataset_path.exists():
                logger.warning(f"Tier 3 consolidated dataset not found: {dataset_path}")
                return datasets

            logger.info(
                f"Loading Tier 3 CoT dataset: {dataset_name} from {dataset_path}"
            )

            # Handle single file or directory
            if dataset_path.suffix == ".jsonl":
                conversations = self.load_jsonl_file(dataset_path)
            else:
                conversations = self.load_json_file(dataset_path)

            # Add tier metadata with CoT-specific info
            self.add_tier_metadata(
                conversations,
                {
                    "dataset_name": dataset_name,
                    "source": f"tier3_cot_{dataset_name}",
                    "reasoning_type": "chain_of_thought",
                },
            )

            datasets[dataset_name] = conversations
            logger.info(
                f"Loaded {len(conversations)} conversations from {dataset_name}"
            )

        except Exception as e:
            logger.error(
                f"Error loading Tier 3 dataset {dataset_name}: {e}",
                exc_info=True,
            )

        total_conversations = sum(len(convs) for convs in datasets.values())
        logger.info(
            f"Tier 3 loading complete: {len(datasets)} datasets, "
            f"{total_conversations} total conversations"
        )

        return datasets
