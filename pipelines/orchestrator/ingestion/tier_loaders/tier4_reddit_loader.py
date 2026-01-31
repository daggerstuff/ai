"""
Tier 4 Reddit Mental Health Archive Loader

Loads comprehensive Reddit mental health archive (50+ condition-specific datasets).
"""

import logging

# Import conversation schema - adjust path based on where this file is located
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ai.pipelines.orchestrator.ingestion.tier_loaders.base_tier_loader import (
    BaseTierLoader,
)

# Add parent directory to path to find conversation_schema
loader_path = Path(__file__).parent
pipeline_root = loader_path.parent.parent.parent
sys.path.insert(0, str(pipeline_root))

try:
    from schemas.conversation_schema import Conversation
except ImportError:
    # Fallback: try relative import
    try:
        from ai.pipelines.orchestrator.schemas.conversation_schema import Conversation
    except ImportError:
        # Last resort: try direct import
        from conversation_schema import Conversation

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
        dataset_registry_path: str = "ai/data/dataset_registry.json",
    ):
        """
        Initialize Tier 4 Reddit loader.

        Args:
            base_path: Optional base path to datasets directory
            quality_threshold: Quality threshold for Tier 4 (default: 0.85 = 85%)
            dataset_registry_path: Path to dataset_registry.json
        """
        super().__init__(
            tier=4,
            quality_threshold=quality_threshold,
            base_path=base_path,
            dataset_registry_path=dataset_registry_path,
        )
        self.base_path = Path(base_path) if base_path else Path("ai/datasets")

        # Consolidated dataset key in registry
        self.consolidated_key = "reddit_consolidated"
        self.dataset_paths = {
            self.consolidated_key: self.base_path / "reddit_mental_health_filtered.json"
        }

        logger.info(
            f"Initialized Tier4RedditLoader: quality_threshold={quality_threshold}, "
            "configured for consolidated Reddit dataset"
        )

    def load_datasets(self) -> Dict[str, List[Conversation]]:
        """
        Load all Tier 4 Reddit datasets.

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
                dataset_name, dataset_path, registry_category="reddit_mental_health"
            )

            if not dataset_path.exists():
                logger.warning(f"Tier 4 consolidated dataset not found: {dataset_path}")
                return datasets

            logger.info(
                f"Loading Tier 4 Reddit dataset: {dataset_name} from {dataset_path}"
            )

            # Handle single file or directory
            if dataset_path.suffix == ".jsonl":
                conversations = self.load_jsonl_file(dataset_path)
            else:
                conversations = self.load_json_file(dataset_path)

            # Add tier metadata
            self.add_tier_metadata(
                conversations,
                {
                    "dataset_name": dataset_name,
                    "source": f"tier4_reddit_{dataset_name}",
                    "data_type": "reddit_archive",
                },
            )

            datasets[dataset_name] = conversations
            logger.info(
                f"Loaded {len(conversations)} conversations from {dataset_name}"
            )

        except Exception as e:
            logger.error(
                f"Error loading Tier 4 dataset {dataset_name}: {e}",
                exc_info=True,
            )

        total_conversations = sum(len(convs) for convs in datasets.values())
        logger.info(
            f"Tier 4 loading complete: {len(datasets)} datasets, "
            f"{total_conversations} total conversations"
        )

        return datasets
