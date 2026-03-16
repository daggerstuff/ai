"""
Tier 5 Research & Specialized Dataset Loader

Loads research datasets and specialized multi-modal data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from ai.pipelines.orchestrator.ingestion.tier_loaders.base_tier_loader import (
    BaseTierLoader,
)
from conversation_schema import Conversation

logger = logging.getLogger(__name__)


class Tier5ResearchLoader(BaseTierLoader):
    """
    Loader for Tier 5: Research & Specialized Datasets.

    Tier 5 datasets are:
    - Academic research datasets
    - Multi-modal data
    - Specialized training scenarios
    - Quality threshold: 80%
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        quality_threshold: float = 0.80,
        dataset_registry_path: str = "ai/data/dataset_registry.json",
    ):
        """
        Initialize Tier 5 research loader.

        Args:
            base_path: Optional base path to datasets directory
            quality_threshold: Quality threshold for Tier 5 (default: 0.80 = 80%)
            dataset_registry_path: Path to dataset_registry.json
        """
        super().__init__(
            tier=5,
            quality_threshold=quality_threshold,
            base_path=base_path,
            dataset_registry_path=dataset_registry_path,
        )
        self.base_path = Path(base_path) if base_path else Path("ai/datasets")

        # Consolidated dataset key in registry
        self.consolidated_key = "research_consolidated"
        self.dataset_paths = {
            self.consolidated_key: self.base_path / "research_datasets_filtered.json"
        }

        logger.info(
            f"Initialized Tier5ResearchLoader: quality_threshold={quality_threshold}, "
            "configured for consolidated research dataset"
        )

    def load_datasets(self) -> Dict[str, List[Conversation]]:
        """
        Load all Tier 5 research datasets.

        Returns:
            Dictionary mapping dataset name to list of conversations
        """
        datasets = {}

        # Load consolidated dataset
        dataset_name = self.consolidated_key
        dataset_path = self.dataset_paths[dataset_name]

        # Ensure available locally (S3 fetch if needed)
        try:
            dataset_path = self._ensure_dataset_locally(
                dataset_name, dataset_path, registry_category="supplementary"
            )

            if not dataset_path.exists():
                logger.warning(f"Tier 5 consolidated dataset not found: {dataset_path}")
                return datasets

            logger.info(
                f"Loading Tier 5 research dataset: {dataset_name} from {dataset_path}"
            )

            if dataset_path.suffix == ".csv":
                logger.warning(
                    "Consolidated dataset is CSV, generic loading not fully "
                    "supported for Tier 5 CSV yet"
                )
                # Fallback to generic CSV? Or error?
                # The consolidated file is .json, so this branch might be dead code,
                # but valid for safety.
            elif dataset_path.is_file():
                if dataset_path.suffix == ".jsonl":
                    conversations = self.load_jsonl_file(dataset_path)
                else:
                    conversations = self.load_json_file(dataset_path)
            else:
                conversations = self._load_dataset_directory(dataset_path, dataset_name)

            # Add tier metadata
            self.add_tier_metadata(
                conversations,
                {
                    "dataset_name": dataset_name,
                    "source": f"tier5_research_{dataset_name}",
                    "data_type": "research",
                },
            )

            datasets[dataset_name] = conversations
            logger.info(
                f"Loaded {len(conversations)} conversations from {dataset_name}"
            )

        except Exception as e:
            logger.error(
                f"Error loading Tier 5 dataset {dataset_name}: {e}",
                exc_info=True,
            )

        total_conversations = sum(len(convs) for convs in datasets.values())
        logger.info(
            f"Tier 5 loading complete: {len(datasets)} datasets, "
            f"{total_conversations} total conversations"
        )

        return datasets
