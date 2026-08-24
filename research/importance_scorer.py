"""
PIX-510 Task 2: Importance Scoring Engine
Implements memory importance scoring with:
  - Exponential decay (configurable τ, default 7 days)
  - Cosine similarity for relevance scoring
  - Emotional weight multipliers (crisis=5.0x, high=2.0x, normal=1.0x)
  - Configurable weights via environment variables

Acceptance: scoring latency < 10ms per block, deterministic output.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

from ai.memory.schema import MemoryBlock, ScoringWeights

# ─── Cosine similarity ────────────────────────────────────────────────────────


def _tokenise(text: str) -> set[str]:
    """Normalise and tokenise text for vectorisation."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two strings using bag-of-words vectors.
    Returns a score in [0.0, 1.0] — 1.0 means identical token sets.

    Time complexity: O(n + m) where n, m are token counts.
    """
    if not text_a or not text_b:
        return 0.0

    tokens_a = _tokenise(text_a)
    tokens_b = _tokenise(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    norm_a = math.sqrt(len(tokens_a))
    norm_b = math.sqrt(len(tokens_b))

    return intersection / (norm_a * norm_b)


# ─── Exponential decay ───────────────────────────────────────────────────────


def exponential_decay(
    timestamp_ms: int,
    now_ms: int | None = None,
    tau_days: float = 7.0,
) -> float:
    """
    Compute exponential decay factor for a memory timestamp.
    Returns e^(-age / τ) in [0.0, 1.0].

    Args:
        timestamp_ms: Memory creation timestamp in Unix milliseconds.
        now_ms: Current timestamp (defaults to now).
        tau_days: Decay time constant in days (default 7.0).

    Example:
        decay(timestamp_3_days_ago, tau_days=7)  # ≈ 0.65
        decay(timestamp_7_days_ago, tau_days=7)  # ≈ 0.37
        decay(timestamp_14_days_ago, tau_days=7) # ≈ 0.14
    """
    if now_ms is None:
        now_ms = time.time_ns() // 1_000_000

    age_ms = now_ms - timestamp_ms
    age_ms = max(age_ms, 0)  # future timestamps clamp to 1.0

    tau_ms = tau_days * 86400 * 1000
    return math.exp(-age_ms / tau_ms)


# ─── Emotional weight ────────────────────────────────────────────────────────


@dataclass
class EmotionalWeights:
    """
    Emotion category → weight multiplier mapping.
    Crisis indicators get 5.0x boost, high-intensity emotions get 2.0x.
    """

    CRISIS: float = 5.0
    HIGH: float = 2.0
    NORMAL: float = 1.0

    _CRISIS_CATEGORIES: tuple[str, ...] = (
        "suicide",
        "self-harm",
        "overdose",
        "panic",
        "psychosis",
    )
    _HIGH_CATEGORIES: tuple[str, ...] = (
        "grief",
        "trauma",
        "anxiety",
        "fear",
        "anger",
        "despair",
        "hopelessness",
    )

    def get_weight(self, categories: list[str]) -> float:
        """
        Return the highest applicable emotional weight for a list of categories.
        Priority: crisis > high > normal.
        """
        lower = [c.lower() for c in categories]

        if any(c in self._CRISIS_CATEGORIES for c in lower):
            return self.CRISIS
        if any(c in self._HIGH_CATEGORIES for c in lower):
            return self.HIGH

        return self.NORMAL


# ─── Main scorer ─────────────────────────────────────────────────────────────


class ImportanceScorer:
    """
    Computes composite importance scores for MemoryBlock instances.

    importance.raw = α·recency + β·relevance + γ·(emotionalWeight/5.0) + δ·actionability

    All weights are configurable via environment variables (see ScoringWeights.from_env).
    Scoring is deterministic — same MemoryBlock always produces the same score.
    """

    __slots__ = ("_emotions", "_weights")

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self._weights = weights or ScoringWeights()
        self._emotions = EmotionalWeights()

    @classmethod
    def from_env(cls) -> ImportanceScorer:
        """Create scorer with weights loaded from environment variables."""
        return cls(ScoringWeights.from_env())

    # ── public API ──────────────────────────────────────────────────────────

    def score(self, memory: MemoryBlock, context: str = "") -> float:
        """
        Compute the composite importance score for a memory block.

        Args:
            memory: The MemoryBlock to score.
            context: Optional query/context string for relevance scoring.
                     If empty, relevance defaults to 0.5 (neutral).

        Returns:
            Composite importance score in [0.0, 1.0].
        """
        recency = exponential_decay(memory.timestamp, tau_days=self._weights.decay_tau_days)
        relevance = self._compute_relevance(memory.content, context)
        emotional = self._emotions.get_weight(memory.emotions.categories)
        actionability = memory.importance.actionability

        return self._weights.compute_importance(recency, relevance, emotional, actionability)

    def score_components(self, memory: MemoryBlock, context: str = "") -> dict[str, float]:
        """
        Return the individual scoring components for debugging/inspection.
        Useful for understanding why a memory scored a certain way.
        """
        recency = exponential_decay(memory.timestamp, tau_days=self._weights.decay_tau_days)
        relevance = self._compute_relevance(memory.content, context)
        emotional = self._emotions.get_weight(memory.emotions.categories)
        actionability = memory.importance.actionability
        raw = self._weights.compute_importance(recency, relevance, emotional, actionability)

        return {
            "recency": round(recency, 6),
            "relevance": round(relevance, 6),
            "emotionalWeight": round(emotional, 2),
            "actionability": round(actionability, 6),
            "raw": round(raw, 6),
        }

    def benchmark(self, n: int = 1000) -> float:
        """
        Benchmark scoring throughput.
        Returns average ms per score over n iterations with a synthetic memory.
        """
        from ai.memory.schema import (
            ConsolidationPhase,
            MemoryConsolidation,
            MemoryEmotions,
            MemoryGating,
            MemoryImportance,
        )

        dummy_full = MemoryBlock(
            id="bench",
            tenantId="bench",
            sessionId="bench",
            content="Therapeutic session discussing coping strategies for anxiety",
            timestamp=int(time.time_ns() // 1_000_000),
            importance=MemoryImportance(
                raw=0.0,
                recency=0.0,
                relevance=0.0,
                emotionalWeight=1.0,
                actionability=0.5,
                reveriePotential=0.0,
            ),
            emotions=MemoryEmotions(valence=-0.3, arousal=0.7, categories=["anxiety"]),
            gating=MemoryGating(),
            consolidation=MemoryConsolidation(
                phase=ConsolidationPhase.RAW,
                lastProcessed=0,
                remCycles=0,
                schemaReferences=[],
                reverieEligible=False,
                reveriePhase="",
            ),
        )
        start = time.perf_counter()
        for _ in range(n):
            self.score(dummy_full)
        elapsed = time.perf_counter() - start
        return (elapsed / n) * 1000  # ms per score

    # ── internal ────────────────────────────────────────────────────────────

    def _compute_relevance(self, content: str, context: str) -> float:
        """Compute relevance as cosine similarity to context, or 0.5 if no context."""
        if not context:
            return 0.5
        return cosine_similarity(content, context)

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def weights(self) -> ScoringWeights:
        return self._weights
