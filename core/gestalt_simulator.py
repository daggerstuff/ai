"""Small, deterministic simulation helpers for gestalt-level workflow states."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class GestaltState:
    state: str
    confidence: float
    metadata: dict[str, Any]


class GestaltSimulator:
    """Generate lightweight synthetic gestalt states for sessions."""

    STATES = ["stable", "escalating", "frustrated", "calm", "resolved"]

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def simulate(self, turn_count: int, *, seed: int | None = None) -> list[GestaltState]:
        if turn_count <= 0:
            return []
        rng = random.Random(seed) if seed is not None else self._rng

        out = []
        for idx in range(turn_count):
            state = rng.choice(self.STATES)
            confidence = round(0.5 + rng.random() * 0.5, 3)
            out.append(
                GestaltState(
                    state=state,
                    confidence=confidence,
                    metadata={"turn_index": idx, "seeded": seed is not None},
                )
            )
        return out

    def summarize(self, trajectory: list[GestaltState]) -> dict[str, Any]:
        if not trajectory:
            return {"length": 0, "most_common": None, "avg_confidence": 0.0}
        counts: dict[str, int] = {}
        for item in trajectory:
            counts[item.state] = counts.get(item.state, 0) + 1
        most_common = max(counts, key=counts.get)
        avg_confidence = sum(item.confidence for item in trajectory) / len(trajectory)
        return {
            "length": len(trajectory),
            "most_common": most_common,
            "avg_confidence": round(avg_confidence, 3),
        }


__all__ = ["GestaltSimulator", "GestaltState"]
