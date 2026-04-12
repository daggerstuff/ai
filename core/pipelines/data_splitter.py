# Stub for ai.core.pipelines.data_splitter
# Generated for test compatibility

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SplitResult:
    """Result of data splitting."""
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


class DataSplitter:
    """Stub implementation for DataSplitter."""

    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        """Initialize data splitter with ratios."""
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(self, data: list[dict[str, Any]], shuffle: bool = True, seed: int | None = None) -> SplitResult:
        """Split data into train/val/test sets."""
        if seed is not None:
            random.seed(seed)

        # Make a copy to avoid modifying original
        data_copy = data.copy()

        if shuffle:
            random.shuffle(data_copy)

        # Calculate split indices
        n = len(data_copy)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train = data_copy[:train_end]
        val = data_copy[train_end:val_end]
        test = data_copy[val_end:]

        return SplitResult(
            train=train,
            val=val,
            test=test,
            metadata={
                "total_records": n,
                "train_count": len(train),
                "val_count": len(val),
                "test_count": len(test)
            }
        )

    def split_by_source(self, data: list[dict[str, Any]], source_field: str = "source") -> dict[str, SplitResult]:
        """Split data by source field."""
        # Group by source
        by_source: dict[str, list[dict[str, Any]]] = {}
        for item in data:
            source = item.get(source_field, "unknown")
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(item)

        # Split each source group
        results: dict[str, SplitResult] = {}
        for source, source_data in by_source.items():
            results[source] = self.split(source_data)

        return results


__all__ = ["DataSplitter", "SplitResult"]
