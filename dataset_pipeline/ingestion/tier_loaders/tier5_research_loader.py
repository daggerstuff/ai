"""
Tier 5 Research & Specialized Dataset Loader

Loads research datasets and specialized multi-modal data.
"""

import csv
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
    ):
        """
        Initialize Tier 5 research loader.

        Args:
            base_path: Optional base path to datasets directory
            quality_threshold: Quality threshold for Tier 5 (default: 0.80 = 80%)
        """
        super().__init__(tier=5, quality_threshold=quality_threshold, base_path=base_path)
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

        logger.info(
            f"Initialized Tier5ResearchLoader: quality_threshold={quality_threshold}"
        )

    def load_datasets(self) -> Dict[str, List[Conversation]]:
        """
        Load all Tier 5 research datasets.

        Returns:
            Dictionary mapping dataset name to list of conversations
        """
        datasets = {}

        for dataset_name, dataset_path in self.dataset_paths.items():
            if not dataset_path.exists():
                logger.warning(f"Tier 5 dataset not found: {dataset_path}")
                continue

            logger.info(f"Loading Tier 5 research dataset: {dataset_name}")

            try:
                if dataset_path.suffix == ".csv":
                    conversations = self._load_csv_dataset(dataset_path, dataset_name)
                else:
                    conversations = self._load_dataset_directory(dataset_path, dataset_name)

                # Add tier metadata
                self.add_tier_metadata(conversations, {
                    "dataset_name": dataset_name,
                    "source": f"tier5_research_{dataset_name}",
                    "data_type": "research",
                })

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

    def _load_dataset_directory(
        self, dataset_path: Path, dataset_name: str
    ) -> List[Conversation]:
        """
        Load conversations from a research dataset directory.

        Args:
            dataset_path: Path to dataset directory
            dataset_name: Name of the dataset

        Returns:
            List of Conversation objects
        """
        conversations = []

        # Look for JSONL files, fallback to JSON files
        jsonl_files = list(dataset_path.rglob("*.jsonl")) or list(
            dataset_path.rglob("*.json")
        )

        for jsonl_file in jsonl_files:
            file_conversations = self.load_jsonl_file(jsonl_file)
            conversations.extend(file_conversations)

        return conversations

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
        conversations = []

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader, 1):
                    try:
                        # Convert CSV row to conversation format
                        # This is dataset-specific and may need customization
                        text = (
                            row.get("text")
                            or row.get("content")
                            or str(row)
                        )

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


