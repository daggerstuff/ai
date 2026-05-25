"""
Therapist voice extraction data models.

Dataclasses used throughout the extraction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChannelResult:
    """Aggregated results for a single therapist channel."""

    name: str
    transcripts: list[str] = field(default_factory=list)
    transcript_titles: list[str] = field(default_factory=list)
    voice_profile: dict = field(default_factory=dict)
    conversations: list[dict] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    score_detail: list[dict] = field(default_factory=list)
    validation_report: dict = field(default_factory=dict)

    @property
    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    @property
    def pass_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s >= 0.5) / len(self.scores)

    @property
    def high_quality_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s >= 0.7) / len(self.scores)
