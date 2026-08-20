"""Deterministic dataset split utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class SplitResult:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]
    metadata: dict[str, Any]


class DataSplitter:
    """Split list-like datasets into deterministic train/val/test partitions."""

    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        total = train_ratio + val_ratio + test_ratio
        if total <= 0:
            raise ValueError("Ratios must sum to a positive value")
        self.train_ratio = train_ratio / total
        self.val_ratio = val_ratio / total
        self.test_ratio = test_ratio / total

    def split(self, data: list[dict[str, Any]], *, shuffle: bool = True, seed: int | None = None) -> SplitResult:
        if not isinstance(data, list):
            raise TypeError("data must be a list")

        rng = random.Random(seed)
        payload = data.copy()
        if shuffle:
            rng.shuffle(payload)

        n = len(payload)
        if n == 0:
            return SplitResult(
                train=[],
                val=[],
                test=[],
                metadata={"counts": {"train": 0, "val": 0, "test": 0}, "ratios": self._ratios()},
            )

        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train = payload[:train_end]
        val = payload[train_end:val_end]
        test = payload[val_end:]

        metadata = {
            "counts": {"train": len(train), "val": len(val), "test": len(test)},
            "ratios": {
                "train": len(train) / n,
                "val": len(val) / n,
                "test": len(test) / n,
            },
            "requested_ratios": {"train": self.train_ratio, "val": self.val_ratio, "test": self.test_ratio},
            "total_records": n,
        }
        return SplitResult(train=train, val=val, test=test, metadata=metadata)

    def split_by_source(
        self,
        data: list[dict[str, Any]],
        *,
        source_field: str = "source",
        min_per_field: int = 0,
    ) -> dict[Any, SplitResult]:
        return self.split_by_field(
            data,
            field=source_field,
            min_per_field=min_per_field,
        )

    def split_by_field(
        self,
        data: list[dict[str, Any]],
        *,
        field: str,
        min_per_field: int = 0,
    ) -> dict[Any, SplitResult]:
        if not data:
            return {}

        grouped: dict[Any, list[dict[str, Any]]] = {}
        for item in data:
            key = item.get(field, "unknown") if isinstance(item, dict) else "unknown"
            grouped.setdefault(key, []).append(item)

        results: dict[Any, SplitResult] = {}
        for key, records in grouped.items():
            if len(records) < min_per_field:
                continue
            results[key] = self.split(records)
        return results

    def _ratios(self) -> dict[str, float]:
        return {"train": self.train_ratio, "val": self.val_ratio, "test": self.test_ratio}

    def report(self, split_result: SplitResult) -> str:
        counts = split_result.metadata.get("counts", {})
        return (
            f"train={counts.get('train', 0)}, val={counts.get('val', 0)}, "
            f"test={counts.get('test', 0)}, total={split_result.metadata.get('total_records', 0)}"
        )


__all__ = ["DataSplitter", "SplitResult"]
