"""EMDR integration helper for synthetic sample generation."""

from __future__ import annotations

import random


class EmdrIntegration:
    """Generate EMDR-style synthetic conversation prompts."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def generate_batch_content(self, count: int = 10) -> list[dict]:
        return [{"technique": "EMDR", "prompt": f"Bilateral stimulation scenario #{idx}"} for idx in range(count)]

    def export_data(self, data: list[dict]) -> str:
        return f"Exported {len(data)} EMDR items"


__all__ = ["EmdrIntegration"]
