"""DBT integration helper for synthetic sample generation."""

from __future__ import annotations

import random


class DbtIntegration:
    """Generate DBT-style synthetic conversation prompts."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def generate_batch_content(self, count: int = 10) -> list[dict]:
        return [{"technique": "DBT", "prompt": f"Distress tolerance exercise #{idx}"} for idx in range(count)]

    def export_data(self, data: list[dict]) -> str:
        return f"Exported {len(data)} DBT items"


__all__ = ["DbtIntegration"]
