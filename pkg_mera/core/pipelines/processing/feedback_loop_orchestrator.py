"""Feedback loop coordination used by production training pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass
class FeedbackEvent:
    name: str
    payload: dict
    created_at: str


class FeedbackLoopOrchestrator:
    """Small helper for collecting and replaying feedback artifacts."""

    def __init__(self) -> None:
        self._events: list[FeedbackEvent] = []

    def record(self, name: str, payload: dict) -> None:
        self._events.append(
            FeedbackEvent(
                name=name,
                payload=payload,
                created_at=datetime.now(tz=UTC).isoformat(),
            )
        )

    def snapshot(self) -> list[FeedbackEvent]:
        """Return a copy of current feedback events."""

        return list(self._events)

    def clear(self) -> None:
        self._events.clear()


__all__ = ["FeedbackEvent", "FeedbackLoopOrchestrator"]
