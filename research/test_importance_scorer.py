"""Unit tests for ai/research/importance_scorer.py — PIX-510 Task 2."""

from __future__ import annotations

import math
import time

from ai.research.importance_scorer import (
    EmotionalWeights,
    ImportanceScorer,
    cosine_similarity,
    exponential_decay,
)
from ai.research.schema import (
    ConsolidationPhase,
    MemoryBlock,
    MemoryConsolidation,
    MemoryEmotions,
    MemoryGating,
    MemoryImportance,
    ScoringWeights,
)

# ─── Cosine similarity ────────────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_texts(self) -> None:
        assert math.isclose(cosine_similarity("hello world", "hello world"), 1.0, rel_tol=1e-5)

    def test_same_tokens_different_order(self) -> None:
        assert math.isclose(cosine_similarity("hello world", "world hello"), 1.0, rel_tol=1e-5)

    def test_partial_overlap(self) -> None:
        score = cosine_similarity("hello world", "world")
        assert 0.3 < score < 0.8  # partial intersection

    def test_no_common_tokens(self) -> None:
        score = cosine_similarity("hello world", "foo bar baz qux")
        assert score < 0.01

    def test_empty_string_a(self) -> None:
        assert cosine_similarity("", "hello world") == 0.0

    def test_empty_string_b(self) -> None:
        assert cosine_similarity("hello world", "") == 0.0

    def test_both_empty(self) -> None:
        assert cosine_similarity("", "") == 0.0

    def test_tokenisation_lowercases(self) -> None:
        assert math.isclose(cosine_similarity("HELLO WORLD", "hello world"), 1.0, rel_tol=1e-5)


# ─── Exponential decay ────────────────────────────────────────────────────────


class TestExponentialDecay:
    def test_fresh_memory_full_value(self) -> None:
        now = int(time.time() * 1000)
        assert math.isclose(exponential_decay(now, now, 7.0), 1.0, rel_tol=1e-3)

    def test_seven_day_old_memory(self) -> None:
        now = int(time.time() * 1000)
        seven_days_ms = 7 * 86_400 * 1000
        assert math.isclose(exponential_decay(now - seven_days_ms, now, 7.0), 0.3679, abs_tol=0.01)

    def test_fourteen_day_old_memory(self) -> None:
        now = int(time.time() * 1000)
        fourteen_days_ms = 14 * 86_400 * 1000
        assert math.isclose(exponential_decay(now - fourteen_days_ms, now, 7.0), 0.1353, abs_tol=0.01)

    def test_future_timestamp_clamps_to_one(self) -> None:
        now = int(time.time() * 1000)
        future = now + 86_400_000  # 1 day in future
        assert math.isclose(exponential_decay(future, now, 7.0), 1.0, rel_tol=1e-3)

    def test_custom_tau(self) -> None:
        now = int(time.time() * 1000)
        one_day_ms = 86_400 * 1000
        # With τ=1 day, 1-day-old memory should have decay ≈ e^-1 ≈ 0.3679
        assert math.isclose(exponential_decay(now - one_day_ms, now, tau_days=1.0), 0.3679, abs_tol=0.01)

    def test_zero_age(self) -> None:
        now = int(time.time() * 1000)
        assert exponential_decay(now, now) == 1.0


# ─── Emotional weights ────────────────────────────────────────────────────────


class TestEmotionalWeights:
    def test_crisis_categories(self) -> None:
        ew = EmotionalWeights()
        for cat in ["suicide", "self-harm", "overdose", "panic", "psychosis"]:
            assert ew.get_weight([cat]) == 5.0, f"{cat} should be crisis weight"

    def test_high_categories(self) -> None:
        ew = EmotionalWeights()
        for cat in ["grief", "trauma", "anxiety", "fear", "anger", "despair", "hopelessness"]:
            assert ew.get_weight([cat]) == 2.0, f"{cat} should be high weight"

    def test_normal_categories(self) -> None:
        ew = EmotionalWeights()
        for cat in ["joy", "trust", "anticipation", "surprise", "disgust"]:
            assert ew.get_weight([cat]) == 1.0, f"{cat} should be normal weight"

    def test_mixed_categories_highest_wins(self) -> None:
        ew = EmotionalWeights()
        assert ew.get_weight(["joy", "anxiety"]) == 2.0  # high overrides normal
        assert ew.get_weight(["anxiety", "suicide"]) == 5.0  # crisis overrides high

    def test_empty_categories(self) -> None:
        assert EmotionalWeights().get_weight([]) == 1.0

    def test_case_insensitive(self) -> None:
        ew = EmotionalWeights()
        assert ew.get_weight(["ANXIETY"]) == 2.0
        assert ew.get_weight(["Suicide"]) == 5.0


# ─── ImportanceScorer ────────────────────────────────────────────────────────


def _make_memory(
    categories: list[str] | None = None,
    actionability: float = 0.5,
    timestamp_ms: int | None = None,
) -> MemoryBlock:
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    return MemoryBlock(
        id="test",
        tenantId="t1",
        sessionId="s1",
        content="Therapeutic session discussing coping strategies for anxiety",
        timestamp=timestamp_ms,
        importance=MemoryImportance(
            raw=0.0, recency=0.0, relevance=0.0, emotionalWeight=1.0, actionability=actionability
        ),
        emotions=MemoryEmotions(valence=-0.3, arousal=0.7, categories=categories or []),
        gating=MemoryGating(),
        consolidation=MemoryConsolidation(
            phase=ConsolidationPhase.RAW,
            lastProcessed=0,
            remCycles=0,
            schemaReferences=[],
        ),
    )


class TestImportanceScorer:
    def test_score_in_range(self) -> None:
        scorer = ImportanceScorer()
        memory = _make_memory()
        score = scorer.score(memory)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self) -> None:
        scorer = ImportanceScorer()
        memory = _make_memory()
        s1 = scorer.score(memory)
        s2 = scorer.score(memory)
        assert s1 == s2

    def test_deterministic_with_context(self) -> None:
        scorer = ImportanceScorer()
        memory = _make_memory()
        ctx = "coping strategies for anxiety"
        s1 = scorer.score(memory, ctx)
        s2 = scorer.score(memory, ctx)
        assert s1 == s2

    def test_anxiety_categories_get_high_weight(self) -> None:
        scorer = ImportanceScorer()
        memory_anxiety = _make_memory(categories=["anxiety"])
        memory_joy = _make_memory(categories=["joy"])
        assert scorer.score(memory_anxiety) > scorer.score(memory_joy)

    def test_crisis_categories_top_weight(self) -> None:
        scorer = ImportanceScorer()
        memory_crisis = _make_memory(categories=["suicide"])
        memory_high = _make_memory(categories=["anxiety"])
        assert scorer.score(memory_crisis) > scorer.score(memory_high)

    def test_context_affects_relevance(self) -> None:
        scorer = ImportanceScorer()
        memory = _make_memory()
        s_no_ctx = scorer.score(memory, "")
        s_with_ctx = scorer.score(memory, "coping strategies anxiety therapy")
        assert s_with_ctx > s_no_ctx

    def test_score_components(self) -> None:
        scorer = ImportanceScorer()
        memory = _make_memory(categories=["grief"])
        comps = scorer.score_components(memory)
        assert "recency" in comps
        assert "relevance" in comps
        assert comps["emotionalWeight"] == 2.0
        assert 0 <= comps["raw"] <= 1.0

    def test_from_env(self) -> None:
        scorer = ImportanceScorer.from_env()
        assert hasattr(scorer, "score")

    def test_latency_under_10ms(self) -> None:
        scorer = ImportanceScorer()
        ms = scorer.benchmark(500)
        assert ms < 10.0, f"Latency {ms:.3f}ms exceeds 10ms threshold"


class TestScoringWeights:
    def test_default_weights(self) -> None:
        w = ScoringWeights()
        assert w.alpha == 0.25
        assert w.beta == 0.25
        assert w.gamma == 0.30
        assert w.delta == 0.20
        assert w.decay_tau_days == 7.0

    def test_compute_importance_bounds(self) -> None:
        w = ScoringWeights()
        score = w.compute_importance(1.0, 1.0, 5.0, 1.0)
        assert 0.0 <= score <= 1.0

    def test_compute_importance_minimum(self) -> None:
        # emotionalWeight is clamped to min 1.0 in the formula, so minimum
        # non-zero contribution from gamma is 0.30 * (1.0/5.0) = 0.06
        w = ScoringWeights()
        score = w.compute_importance(0.0, 0.0, 1.0, 0.0)
        assert 0.05 < score < 0.07

    def test_decay_factor_static(self) -> None:
        decay = ScoringWeights.decay_factor(7 * 86_400, tau_days=7.0)
        assert math.isclose(decay, 0.3679, abs_tol=0.01)
