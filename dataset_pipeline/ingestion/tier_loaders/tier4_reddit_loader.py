"""
Tier 4 Reddit Mental Health Archive Loader

Loads comprehensive Reddit mental health archive (50+ condition-specific datasets).
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Import conversation schema - adjust path based on where this file is located
import sys
from pathlib import Path as PathType

# Add parent directory to path to find conversation_schema
loader_path = PathType(__file__).parent
pipeline_root = loader_path.parent.parent.parent
sys.path.insert(0, str(pipeline_root))

try:
    from schemas.conversation_schema import Conversation, Message
except ImportError:
    # Fallback: try relative import
    try:
        from ai.dataset_pipeline.schemas.conversation_schema import Conversation, Message
    except ImportError:
        # Last resort: try direct import
        from conversation_schema import Conversation, Message

from ai.dataset_pipeline.ingestion.tier_loaders.base_tier_loader import (
    BaseTierLoader,
)

logger = logging.getLogger(__name__)


class Tier4RedditLoader(BaseTierLoader):
    """
    Loader for Tier 4: Reddit Mental Health Archive.

    Tier 4 datasets are:
    - Real-world mental health conversations
    - Large-scale data (millions of posts)
    - Diverse conditions and populations
    - Quality threshold: 85%
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        quality_threshold: float = 0.85,
    ):
        """
        Initialize Tier 4 Reddit loader.

        Args:
            base_path: Optional base path to datasets directory
            quality_threshold: Quality threshold for Tier 4 (default: 0.85 = 85%)
        """
        super().__init__(tier=4, quality_threshold=quality_threshold, base_path=base_path)
        self.base_path = Path(base_path) if base_path else Path("ai/datasets/old-datasets")

        # Condition-specific datasets
        self.condition_datasets = [
            "addiction", "ADHD", "anxiety", "autism", "bipolar", "BPD",
            "depression", "PTSD", "schizophrenia", "social_anxiety",
            "health_anxiety", "eating_disorders", "loneliness",
            "parenting_stress", "divorce_recovery",
        ]

        logger.info(
            f"Initialized Tier4RedditLoader: quality_threshold={quality_threshold}"
        )

    def load_datasets(self) -> Dict[str, List[Conversation]]:
        """
        Load all Tier 4 Reddit datasets.

        Returns:
            Dictionary mapping dataset name to list of conversations
        """
        datasets = {}

        # Load condition-specific datasets
        for condition in self.condition_datasets:
            condition_path = self.base_path / f"{condition}.csv"
            if condition_path.exists():
                logger.info(f"Loading Tier 4 Reddit dataset: {condition}")
                try:
                    conversations = self._load_csv_dataset(condition_path, condition)

                    # Add tier metadata
                    self.add_tier_metadata(conversations, {
                        "condition": condition,
                        "source": f"tier4_reddit_{condition}",
                        "data_type": "reddit_archive",
                    })

                    datasets[condition] = conversations
                    logger.info(
                        f"Loaded {len(conversations)} conversations from {condition}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error loading Tier 4 dataset {condition}: {e}",
                        exc_info=True,
                    )
                    continue

        # Load special datasets
        special_datasets = {
            "suicide_detection": self.base_path / "Suicide_Detection.csv",
            "covid19_support": self.base_path / "COVID19_support_post_features",
            "adhd_women": self.base_path / "adhdwomen.csv",
        }

        for dataset_name, dataset_path in special_datasets.items():
            if dataset_path.exists():
                logger.info(f"Loading Tier 4 special dataset: {dataset_name}")
                try:
                    if dataset_path.suffix == ".csv":
                        conversations = self._load_csv_dataset(dataset_path, dataset_name)
                    else:
                        conversations = self._load_jsonl_file(dataset_path)

                    self.add_tier_metadata(conversations, {
                        "dataset_name": dataset_name,
                        "source": f"tier4_reddit_{dataset_name}",
                    })

                    datasets[dataset_name] = conversations
                    logger.info(
                        f"Loaded {len(conversations)} conversations from {dataset_name}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error loading Tier 4 dataset {dataset_name}: {e}",
                        exc_info=True,
                    )
                    continue

        total_conversations = sum(len(convs) for convs in datasets.values())
        logger.info(
            f"Tier 4 loading complete: {len(datasets)} datasets, "
            f"{total_conversations} total conversations"
        )

        return datasets

    def _load_csv_dataset(
        self, csv_path: Path, dataset_name: str
    ) -> List[Conversation]:
        """
        Load conversations from a CSV file.

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
                        # Try to extract conversation from CSV row
                        # Common CSV formats: text, post, content, message, etc.
                        text = (
                            row.get("text")
                            or row.get("post")
                            or row.get("content")
                            or row.get("message")
                            or row.get("body")
                            or ""
                        )

                        if not text:
                            continue

                        # Create conversation from text
                        # Assume single message format for Reddit posts
                        conversation = Conversation(
                            conversation_id=f"{dataset_name}_{row_num}",
                            source=f"tier4_reddit_{dataset_name}",
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

