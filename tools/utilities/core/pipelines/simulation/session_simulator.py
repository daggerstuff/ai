"""Session simulation helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Session:
    session_id: str
    turns: list[dict]


class SessionSimulator:
    """Generate synthetic journaling sessions for pipeline tests."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def generate_session(self, index: int) -> Session:
        turns = [
            {"role": "user", "content": f"I am feeling mood {self.rng.randint(1, 10)} at turn {i}"} for i in range(2)
        ]
        turns.append({"role": "assistant", "content": "Thank you for sharing"})
        return Session(session_id=f"sim-{index}", turns=turns)

    def generate_batch(self, count: int = 10) -> list[Session]:
        return [self.generate_session(i) for i in range(count)]


__all__ = ["Session", "SessionSimulator"]
