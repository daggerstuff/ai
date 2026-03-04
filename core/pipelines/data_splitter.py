"""
Data splitter for the Pixelated Empathy AI dataset pipeline.
Implements the mandatory 70/15/15 train/val/test split ratio.
"""

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ai.core.pipelines.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetSplit:
    """Represents a split dataset with metadata."""

    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]
    metadata: dict[str, Any]


class DataSplitter:
    """
    Handles dataset splitting into Train, Validation, and Test sets.
    Enforces the mandatory 70/15/15 split ratio.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        """
        Initialize the splitter with specified ratios.
        Ratios must sum to 1.0.
        """
        if not (0.99 <= (train_ratio + val_ratio + test_ratio) <= 1.01):
            raise ValueError(
                f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
            )

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        logger.info(
            f"DataSplitter initialized with ratios: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}"
        )

    def split(
        self, data: list[dict[str, Any]], shuffle: bool = True, seed: int = 42
    ) -> DatasetSplit:
        """
        Splits a list of records into train, val, and test sets.
        Expects a list of dictionaries (standard JSONL record format).
        """
        total_count = len(data)
        if total_count == 0:
            logger.warning("Attempted to split an empty dataset.")
            return DatasetSplit([], [], [], {"total": 0, "status": "empty"})

        if shuffle:
            random.seed(seed)
            random.shuffle(data)

        train_end = int(total_count * self.train_ratio)
        val_end = train_end + int(total_count * self.val_ratio)

        train_set = data[:train_end]
        val_set = data[train_end:val_end]
        test_set = data[val_end:]

        metadata = {
            "total_records": total_count,
            "train_count": len(train_set),
            "val_count": len(val_set),
            "test_count": len(test_set),
            "ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
            "seed": seed if shuffle else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"Dataset split complete: {metadata['train_count']} train, {metadata['val_count']} val, {metadata['test_count']} test"
        )

        return DatasetSplit(
            train=train_set, val=val_set, test=test_set, metadata=metadata
        )

    def split_by_source(
        self, data: list[dict[str, Any]], source_key: str = "source"
    ) -> dict[str, DatasetSplit]:
        """
        Splits data independently for each source to maintain balance.
        """
        by_source = {}
        for record in data:
            source = record.get(source_key, "unknown")
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(record)

        results = {}
        for source, source_data in by_source.items():
            logger.info(
                f"Splitting data for source: {source} ({len(source_data)} records)"
            )
            results[source] = self.split(source_data)

        return results


def run_data_splitting(data: list[dict[str, Any]]) -> DatasetSplit:
    """Helper function to run splitting with default parameters."""
    splitter = DataSplitter()
    return splitter.split(data)
