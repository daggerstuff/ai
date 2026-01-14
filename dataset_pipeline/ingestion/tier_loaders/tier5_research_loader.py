"""
Tier 5 Research & Specialized Dataset Loader

Loads research datasets and specialized multi-modal data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from ai.dataset_pipeline.ingestion.tier_loaders.base_tier_loader import (
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

        # Known Tier 5 research datasets
        self.dataset_paths = {
            "empathy_mental_health": self.base_path / "Empathy-Mental-Health",
            "reccon": self.base_path / "RECCON",
            "iemocap": self.base_path / "IEMOCAP_EMOTION_Recognition",
            "modma": self.base_path / "MODMA-Dataset",
            "unalignment_toxic": self.base_path / "unalignment_toxic-dpo-v0.2-ShareGPT",
            "data_final": self.base_path / "data-final.csv",
            "depression_detection": self.base_path / "DepressionDetection",
            "reddit_raw": self.base_path / "Original Reddit Data" / "raw data",
        }

        # Discover more from registry
        self._discover_research_datasets()

        logger.info(
            f"Initialized Tier5ResearchLoader: quality_threshold={quality_threshold}, "
            f"{len(self.dataset_paths)} datasets configured"
        )

    def _discover_research_datasets(self) -> None:
        """Discover research datasets from registry."""
        research_datasets = self.registry.get("datasets", {}).get("supplementary", {})
        for reg_name, entry in research_datasets.items():
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
        Load all Tier 5 research datasets.

        Returns:
            Dictionary mapping dataset name to list of conversations
        """
        datasets = {}

        for dataset_name, dataset_path in self.dataset_paths.items():
            # Ensure available locally (S3 fetch if needed)
            dataset_path = self._ensure_dataset_locally(
                dataset_name, dataset_path, registry_category="supplementary"
            )

            if not dataset_path.exists():
                logger.warning(f"Tier 5 dataset not found: {dataset_path}")
                continue

            logger.info(
                f"Loading Tier 5 research dataset: {dataset_name} from {dataset_path}"
            )

            try:
                if dataset_path.suffix == ".csv":
                    # Custom CSV loader for Tier 5 (not in base class yet)
                    conversations = self._load_csv_dataset(dataset_path, dataset_name)
                elif dataset_path.is_file():
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
                continue

        total_conversations = sum(len(convs) for convs in datasets.values())
        logger.info(
            f"Tier 5 loading complete: {len(datasets)} datasets, "
            f"{total_conversations} total conversations"
        )

        return datasets

    def _load_csv_dataset(
        self, csv_path: Path, dataset_name: str
    ) -> List[Conversation]:
        """
        Load conversations from a CSV file (for data-final.csv).

        Args:
            csv_path: Path to CSV file
            dataset_name: Name of the dataset

        Returns:
            List of Conversation objects
        """
        import csv

        from conversation_schema import Message

        conversations = []

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader, 1):
                    try:
                        # Convert CSV row to conversation format
                        text = row.get("text") or row.get("content") or str(row)

                        if not text:
                            continue

                        conversation = Conversation(
                            conversation_id=f"{dataset_name}_{row_num}",
                            source=f"tier5_research_{dataset_name}",
                            messages=[Message(role="user", content=text)],
                            metadata={
                                "tier": self.tier,
                                "quality_threshold": self.quality_threshold,
                                "row_data": row,
                            },
                        )

                        conversations.append(conversation)

                    except Exception as e:
                        logger.warning(
                            f"Error processing row {row_num} in {csv_path}: {e}"
                        )
                        continue

        except Exception as e:
            logger.error(f"Error loading CSV dataset {csv_path}: {e}", exc_info=True)
            raise

        return conversations
