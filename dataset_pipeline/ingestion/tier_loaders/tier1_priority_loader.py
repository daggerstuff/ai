"""
Tier 1 Priority Dataset Loader

Loads curated Wendy datasets with highest quality validation.
Tier 1 datasets are production-ready and used as gold standard for quality validation.
"""

import json
import logging
from pathlib import Path
from typing import Any

from conversation_schema import Conversation

from ai.dataset_pipeline.ingestion.tier_loaders.base_tier_loader import (
    BaseTierLoader,
)

logger = logging.getLogger(__name__)


class Tier1PriorityLoader(BaseTierLoader):
    """
    Loader for Tier 1: Curated Priority Datasets.

    Tier 1 datasets are:
    - Highest quality, production-ready data
    - Pre-curated and validated
    - Used as gold standard for quality validation
    - Quality threshold: 99%
    """

    def __init__(
        self,
        base_path: Path = Path("ai/datasets/datasets-wendy"),
        quality_threshold: float = 0.99,
        priority_numbers: list[int] | None = None,
    ):
        """
        Initialize Tier 1 priority loader.

        Args:
            base_path: Base path to datasets-wendy directory
            quality_threshold: Quality threshold for Tier 1 (default: 0.99 = 99%)
            priority_numbers: Optional list of priority numbers to load (1-5).
                             If None, loads all available priority datasets.
        """
        super().__init__(tier=1, quality_threshold=quality_threshold, base_path=base_path)
        # Ensure base_path is a Path object (base class may set it to None)
        if self.base_path is None:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path(self.base_path)

        # Validate and set priority numbers
        if priority_numbers is not None:
            self._validate_priority_numbers(priority_numbers)
            self.priority_numbers = priority_numbers
        else:
            self.priority_numbers = None

        logger.info(
            f"Initialized Tier1PriorityLoader: base_path={self.base_path}, "
            f"quality_threshold={quality_threshold}, "
            f"priority_numbers={self.priority_numbers}"
        )

    def load_datasets(self) -> dict[str, list[Conversation]]:
        """
        Load priority datasets (Wendy priority set).

        Uses priority_numbers from instance configuration if set,
        otherwise loads all available priority datasets.

        Returns:
            Dictionary mapping priority number to list of conversations
        """
        priority_numbers = self.priority_numbers or [1, 2, 3, 4, 5]
        return self._load_priority_datasets(priority_numbers)

    def _load_priority_datasets(self, priority_numbers: list[int]) -> dict[str, list[Conversation]]:
        """
        Load priority datasets (Wendy priority set).

        Args:
            priority_numbers: List of priority numbers to load (1-5).

        Returns:
            Dictionary mapping priority number to list of conversations
        """
        self._validate_priority_numbers(priority_numbers)

        datasets = {}

        filename_candidates: dict[int, list[str]] = {
            1: ["wendy_set_alpha_therapeutic_core.jsonl"],
            2: ["wendy_set_beta_high_quality_core.jsonl"],
            3: ["wendy_set_gamma_specialized_therapy.jsonl"],
            4: ["wendy_set_delta.jsonl"],
            5: ["wendy_set_epsilon.jsonl"],
        }

        for priority_num in priority_numbers:
            candidates = filename_candidates.get(priority_num, [])
            if not candidates:
                logger.warning("No filename candidates configured for set %s", priority_num)
                continue
            priority_file = next(
                (self.base_path / name for name in candidates if (self.base_path / name).exists()),
                self.base_path / candidates[0],
            )
            summary_file = self.base_path / "summary.json"

            if not priority_file.exists():
                logger.warning(
                    f"Priority dataset file not found for {priority_num}: {priority_file}"
                )
                continue

            logger.info("Loading Tier 1 priority dataset: %s", priority_file.name)

            try:
                # Use base class method to load JSONL file
                conversations = self.load_jsonl_file(priority_file)

                # Update source and add priority-specific metadata to each conversation
                source_name = f"tier1_wendy_priority_{priority_num}"
                for conv in conversations:
                    conv.source = source_name

                # Load summary if available
                metadata = {}
                if summary_file.exists():
                    metadata = self._load_summary(summary_file)

                # Add tier metadata to all conversations (includes priority_number)
                self.add_tier_metadata(
                    conversations,
                    {
                        "priority_number": priority_num,
                        "source": f"wendy_priority_{priority_num}",
                        **metadata,
                    },
                )

                datasets[f"priority_{priority_num}"] = conversations
                logger.info(
                    f"Loaded {len(conversations)} conversations from priority_{priority_num}"
                )

            except Exception as e:
                logger.error(
                    f"Error loading priority_{priority_num}: {e}",
                    exc_info=True,
                )
                continue

        total_conversations = sum(len(convs) for convs in datasets.values())
        logger.info(
            f"Tier 1 loading complete: {len(datasets)} datasets, "
            f"{total_conversations} total conversations"
        )

        return datasets

    def _validate_priority_numbers(self, priority_numbers: list[int]) -> None:
        """
        Validate that priority numbers are in valid range (1-5).

        Args:
            priority_numbers: List of priority numbers to validate

        Raises:
            ValueError: If any priority number is outside valid range
        """
        valid_range = set(range(1, 6))  # 1-5
        invalid = [p for p in priority_numbers if p not in valid_range]
        if invalid:
            raise ValueError(
                f"Invalid priority numbers: {invalid}. Priority numbers must be in range 1-5."
            )

    def _load_summary(self, summary_file: Path) -> dict[str, Any]:
        """
        Load summary.json file for metadata.

        Args:
            summary_file: Path to summary.json file

        Returns:
            Dictionary with summary metadata
        """
        try:
            with open(summary_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load summary file {summary_file}: {e}")
            return {}
