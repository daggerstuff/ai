"""
Tier 2 Professional Therapeutic Dataset Loader

Loads professional therapeutic datasets (Psych8k, mental_health_counseling, etc.)
with clinical-grade quality validation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from ai.dataset_pipeline.ingestion.tier_loaders.base_tier_loader import (
    BaseTierLoader,
)
from conversation_schema import Conversation

logger = logging.getLogger(__name__)


class Tier2ProfessionalLoader(BaseTierLoader):
    """
    Loader for Tier 2: Professional Therapeutic Datasets.

    Tier 2 datasets are:
    - Clinical-grade conversation data
    - Professional therapy sessions
    - Quality threshold: 95%
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        quality_threshold: float = 0.95,
        dataset_registry_path: str = "ai/data/dataset_registry.json",
    ):
        """
        Initialize Tier 2 professional loader.

        Args:
            base_path: Optional base path to datasets directory
            quality_threshold: Quality threshold for Tier 2 (default: 0.95 = 95%)
            dataset_registry_path: Path to dataset_registry.json
        """
        super().__init__(
            tier=2,
            quality_threshold=quality_threshold,
            base_path=base_path,
            dataset_registry_path=dataset_registry_path,
        )
        self.base_path = Path(base_path) if base_path else Path("ai/datasets")

        # Known Tier 2 datasets
        self.dataset_paths = {
            "psych8k": self.base_path / "Psych8k",
            "mental_health_counseling": self.base_path
            / "mental_health_counseling_conversations",
            "soulchat": self.base_path / "SoulChat2.0",
            "counsel_chat": self.base_path / "counsel-chat",
            "llama3_mental": self.base_path / "LLAMA3_Mental_Counseling_Data",
            "therapist_sft": self.base_path / "therapist-sft-format",
            "neuro_qa": self.base_path / "neuro_qa_SFT_Trainer",
        }

        # Discover more from registry
        self._discover_professional_datasets()

        logger.info(
            f"Initialized Tier2ProfessionalLoader: "
            f"quality_threshold={quality_threshold}, "
            f"{len(self.dataset_paths)} datasets configured"
        )

    def _discover_professional_datasets(self) -> None:
        """Discover professional datasets from registry."""
        prof_datasets = self.registry.get("datasets", {}).get(
            "professional_therapeutic", {}
        )
        for reg_name, entry in prof_datasets.items():
            if not isinstance(entry, dict):
                continue

            key = reg_name.lower().replace("-", "_")
            if key not in self.dataset_paths:
                path_val = entry.get("path", "")
                if path_val and not self._is_s3_path(path_val):
                    self.dataset_paths[key] = Path(path_val).expanduser()
                else:
                    self.dataset_paths[key] = self.base_path / reg_name

    def load_datasets(self) -> Dict[str, List[Conversation]]:
        """
        Load all Tier 2 professional datasets.

        Returns:
            Dictionary mapping dataset name to list of conversations
        """
        datasets = {}

        for dataset_name, dataset_path in self.dataset_paths.items():
            # Ensure available locally (S3 fetch if needed)
            dataset_path = self._ensure_dataset_locally(
                dataset_name, dataset_path, registry_category="professional_therapeutic"
            )

            if not dataset_path.exists():
                logger.warning(f"Tier 2 dataset not found: {dataset_path}")
                continue

            logger.info(f"Loading Tier 2 dataset: {dataset_name} from {dataset_path}")

            try:
                # Use unified directory loader from base class
                if dataset_path.is_file():
                    if dataset_path.suffix == ".jsonl":
                        conversations = self.load_jsonl_file(dataset_path)
                    else:
                        conversations = self.load_json_file(dataset_path)
                else:
                    conversations = self._load_dataset_directory(
                        dataset_path, dataset_name
                    )

                # Add tier metadata
                self.add_tier_metadata(
                    conversations,
                    {
                        "dataset_name": dataset_name,
                        "source": f"tier2_{dataset_name}",
                    },
                )

                datasets[dataset_name] = conversations
                logger.info(
                    f"Loaded {len(conversations)} conversations from {dataset_name}"
                )

            except Exception as e:
                logger.error(
                    f"Error loading Tier 2 dataset {dataset_name}: {e}",
                    exc_info=True,
                )
                continue

        total_conversations = sum(len(convs) for convs in datasets.values())
        logger.info(
            f"Tier 2 loading complete: {len(datasets)} datasets, "
            f"{total_conversations} total conversations"
        )

        return datasets
