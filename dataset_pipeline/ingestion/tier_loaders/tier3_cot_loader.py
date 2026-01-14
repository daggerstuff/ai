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

        # Mapping of internal names to predicted local paths
        self.dataset_paths = {
            "cot_clinical_diagnosis": self.base_path
            / "CoT_Reasoning_Clinical_Diagnosis_Mental_Health",
            "cot_neurodivergent": self.base_path
            / "CoT_Neurodivergent_vs_Neurotypical_Interactions",
            "cot_heartbreak": self.base_path / "CoT_Heartbreak_and_Breakups",
            "cot_mens_mental_health": self.base_path
            / "CoT_Reasoning_Mens_Mental_Health",
            "cot_legal": self.base_path / "CoT_Legal_Issues_And_Laws",
            "cot_philosophical": self.base_path / "CoT_Philosophical_Understanding",
            "cot_rare_diseases": self.base_path
            / "CoT_Rare-Diseases_And_Health-Conditions",
            "cot_temporal": self.base_path / "CoT_Temporal_Reasoning_Dataset",
            "cot_scientific": self.base_path
            / "CoT_Reasoning_Scientific_Discovery_and_Research",
            "cot_cultural": self.base_path / "CoT-Reasoning_Cultural_Nuances",
        }

        # Add dynamic datasets from registry
        self._discover_cot_datasets()

        logger.info(
            f"Initialized Tier3CoTLoader: quality_threshold={quality_threshold}, "
            f"{len(self.dataset_paths)} datasets configured"
        )

    def _discover_cot_datasets(self) -> None:
        """Discover all CoT datasets from registry."""
        cot_datasets = self.registry.get("datasets", {}).get("cot_reasoning", {})
        for reg_name, entry in cot_datasets.items():
            if not isinstance(entry, dict):
                continue

            # Create a consistent key
            key = reg_name.lower().replace("-", "_")
            if not key.startswith("cot_") and "cot" not in key:
                key = f"cot_{key}"

            if key not in self.dataset_paths:
                path_val = entry.get("path", "")
                if path_val and not self._is_s3_path(path_val):
                    self.dataset_paths[key] = Path(path_val).expanduser()
                else:
                    # Predict a path if not found
                    self.dataset_paths[key] = self.base_path / reg_name

    def load_datasets(self) -> Dict[str, List[Conversation]]:
        """
        Load all Tier 3 CoT reasoning datasets.

        Returns:
            Dictionary mapping dataset name to list of conversations
        """
        datasets = {}

        for dataset_name, dataset_path in self.dataset_paths.items():
            # Ensure dataset is available locally (downloads from S3 if needed)
            dataset_path = self._ensure_dataset_locally(
                dataset_name, dataset_path, registry_category="cot_reasoning"
            )

            if not dataset_path.exists():
                logger.warning(f"Tier 3 dataset not found: {dataset_path}")
                continue

            logger.info(
                f"Loading Tier 3 CoT dataset: {dataset_name} from {dataset_path}"
            )

            try:
                # Handle single file or directory
                if dataset_path.is_file():
                    if dataset_path.suffix == ".jsonl":
                        conversations = self.load_jsonl_file(dataset_path)
                    else:
                        conversations = self.load_json_file(dataset_path)
                else:
                    conversations = self._load_dataset_directory(
                        dataset_path, dataset_name
                    )

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
                continue

        total_conversations = sum(len(convs) for convs in datasets.values())
        logger.info(
            f"Tier 3 loading complete: {len(datasets)} datasets, "
            f"{total_conversations} total conversations"
        )

        return datasets
