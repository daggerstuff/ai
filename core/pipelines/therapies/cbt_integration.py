"""CBT integration helper for deterministic sample generation."""

from __future__ import annotations

import random


class CbtIntegration:
    """Generate CBT-style synthetic conversation prompts."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def generate_batch_content(self, count: int = 10) -> list[dict]:
        return [
            {
                "technique": "CBT",
                "prompt": f"Challenge thought #{idx}",
            }
            for idx in range(count)
        ]

    def export_data(self, data: list[dict]) -> str:
        return f"Exported {len(data)} CBT items"


__all__ = ["CbtIntegration"]
