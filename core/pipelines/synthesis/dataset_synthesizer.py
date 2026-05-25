"""Dataset synthesis orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data_splitter import DataSplitter


@dataclass
class DatasetSplit:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]


class DatasetSynthesizer:
    """Create synthetic dataset payloads and split by provided policy."""

    def __init__(self, output_path: str | Path | None = None) -> None:
        self.output_path = Path(output_path or "ai/training_ready/data/generated")
        self.output_path.mkdir(parents=True, exist_ok=True)

    def synthesize_dataset(self, format_type: str = "alpaca", *, count: int = 50) -> list[dict[str, Any]]:
        if format_type not in {"alpaca", "jsonl", "conversation"}:
            raise ValueError(f"Unsupported format_type: {format_type}")

        samples: list[dict[str, Any]] = []
        for idx in range(count):
            if format_type == "alpaca":
                samples.append(
                    {
                        "instruction": f"Synthesize therapeutic response {idx}",
                        "input": f"Client issue #{idx}",
                        "output": "Provide empathetic support and coping steps.",
                    }
                )
            elif format_type == "conversation":
                samples.append(
                    {
                        "conversation_id": str(idx),
                        "messages": [
                            {"role": "user", "content": f"I feel stressed #{idx}"},
                            {"role": "assistant", "content": "Let's pause and explore this together."},
                        ],
                    }
                )
            else:
                samples.append({"id": idx, "text": f"Synthetic sample {idx}"})

        return samples

    def split_dataset(
        self,
        dataset: list[dict[str, Any]],
        *,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        splitter = DataSplitter(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        result = splitter.split(dataset, seed=seed)
        return {"train": result.train, "val": result.val, "test": result.test}


__all__ = ["DatasetSplit", "DatasetSynthesizer"]
