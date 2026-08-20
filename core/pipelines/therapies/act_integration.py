"""ACT integration helper for synthetic sample generation."""

from __future__ import annotations

import random


class ActIntegration:
    """Generate ACT-style therapeutic prompts and lightweight exports."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def generate_batch_content(self, count: int = 10) -> list[dict]:
        return [{"technique": "ACT", "prompt": f"Values-based action plan #{idx}"} for idx in range(count)]

    def export_data(self, data: list[dict]) -> str:
        return f"Exported {len(data)} ACT items"


__all__ = ["ActIntegration"]
