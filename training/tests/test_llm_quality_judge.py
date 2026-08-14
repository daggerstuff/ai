"""Tests for DualModelQualityJudge — dual-model LLM quality judge.

Covers: rubric scoring (weighted mean, 4-bin), multi-turn recency-decay
weighting, self-consistency (k=3, variance check), cross-model consistency
rule, calibration (Pearson + Cohen's kappa), and async interface.

All LLM calls are mocked — no vLLM or GPU required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.llm_quality_judge import (
    DualModelQualityJudge,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_llm_response(
    relevance: float = 0.5,
    accuracy: float = 0.5,
    helpfulness: float = 0.5,
    style: float = 0.5,
    safety: float = 0.5,
) -> dict:
    """Create a mock LLM structured response."""
    return {
        "overall_score": 0.5,
        "reasoning": "Mock response",
        "dimension_scores": {
            "relevance": relevance,
            "accuracy": accuracy,
            "helpfulness": helpfulness,
            "style": style,
            "safety": safety,
        },
    }


def make_mock_client(response: dict | list[dict]) -> MagicMock:
    """Create a mock LLMClient that returns the given response(s).

    If a list is given, returns each element in sequence (for self-consistency
    variance testing).
    """
    client = MagicMock()
    if isinstance(response, list):
        client.generate_structured.side_effect = response
    else:
        client.generate_structured.return_value = response
    return client


def make_conversation(n_turns: int = 1) -> list[dict[str, str]]:
    """Create a conversation with n_turns assistant responses."""
    conv: list[dict[str, str]] = []
    for i in range(n_turns):
        conv.append({"role": "user", "content": f"Question {i}"})
        conv.append({"role": "assistant", "content": f"Answer {i}"})
    return conv


@pytest.fixture
def uniform_response() -> dict:
    """All dimensions = 0.6 (good range)."""
    return make_llm_response(0.6, 0.6, 0.6, 0.6, 0.6)


@pytest.fixture
def high_response() -> dict:
    """All dimensions = 0.9 (excellent range)."""
    return make_llm_response(0.9, 0.9, 0.9, 0.9, 0.9)


@pytest.fixture
def low_response() -> dict:
    """All dimensions = 0.1 (poor range)."""
    return make_llm_response(0.1, 0.1, 0.1, 0.1, 0.1)


@pytest.fixture
def judge_with_mock_clients(uniform_response):
    """Judge with both clients returning uniform 0.6 scores."""
    primary = make_mock_client(uniform_response)
    secondary = make_mock_client(uniform_response)
    return DualModelQualityJudge(
        primary_client=primary,
        secondary_client=secondary,
    )


# ===========================================================================
# Test 1: Rubric scoring — weighted mean and 4-bin classification
# ===========================================================================


class TestRubricScoring:
    """Verify weighted mean computation and 4-bin classification."""

    def test_weighted_mean_with_default_weights(self, judge_with_mock_clients, uniform_response):
        """Overall = weighted mean of dimension scores."""
        result = judge_with_mock_clients.judge(make_conversation())
        # All dims = 0.6, weighted mean = 0.6
        ts = result["turn_scores"][0]
        assert ts.primary_overall == pytest.approx(0.6, abs=0.01)
        assert ts.secondary_overall == pytest.approx(0.6, abs=0.01)

    def test_weighted_mean_with_custom_weights(self, uniform_response):
        """Custom weights produce correct weighted mean."""
        custom_weights = {
            "relevance": 0.10,
            "accuracy": 0.40,
            "helpfulness": 0.20,
            "style": 0.20,
            "safety": 0.10,
        }
        primary = make_mock_client(uniform_response)
        secondary = make_mock_client(uniform_response)
        judge = DualModelQualityJudge(
            primary_client=primary,
            secondary_client=secondary,
            rubric_weights=custom_weights,
        )
        result = judge.judge(make_conversation())
        # All dims = 0.6 → weighted mean = 0.6 regardless of weights
        assert result["overall_score"] == pytest.approx(0.6, abs=0.02)

    def test_weighted_mean_with_asymmetric_scores(self):
        """Asymmetric dimension scores → correct weighted mean."""
        # relevance=1.0, accuracy=0.0, helpfulness=0.0, style=0.0, safety=0.0
        # Default weights: 0.25*1.0 + 0.30*0 + 0.20*0 + 0.15*0 + 0.10*0 = 0.25
        response = make_llm_response(1.0, 0.0, 0.0, 0.0, 0.0)
        primary = make_mock_client(response)
        secondary = make_mock_client(response)
        judge = DualModelQualityJudge(primary, secondary)
        result = judge.judge(make_conversation())
        assert result["turn_scores"][0].primary_overall == pytest.approx(0.25, abs=0.01)

    def test_classify_score_poor(self):
        """Score < 0.25 → poor."""
        assert DualModelQualityJudge.classify_score(0.0) == "poor"
        assert DualModelQualityJudge.classify_score(0.10) == "poor"
        assert DualModelQualityJudge.classify_score(0.24) == "poor"

    def test_classify_score_fair(self):
        """0.25 <= score < 0.50 → fair."""
        assert DualModelQualityJudge.classify_score(0.25) == "fair"
        assert DualModelQualityJudge.classify_score(0.30) == "fair"
        assert DualModelQualityJudge.classify_score(0.49) == "fair"

    def test_classify_score_good(self):
        """0.50 <= score < 0.75 → good."""
        assert DualModelQualityJudge.classify_score(0.50) == "good"
        assert DualModelQualityJudge.classify_score(0.60) == "good"
        assert DualModelQualityJudge.classify_score(0.74) == "good"

    def test_classify_score_excellent(self):
        """0.75 <= score <= 1.0 → excellent."""
        assert DualModelQualityJudge.classify_score(0.75) == "excellent"
        assert DualModelQualityJudge.classify_score(0.90) == "excellent"
        assert DualModelQualityJudge.classify_score(1.0) == "excellent"

    def test_bin_in_result_matches_classify(self, judge_with_mock_clients, uniform_response):
        """Result bin matches classify_score(overall_score)."""
        result = judge_with_mock_clients.judge(make_conversation())
        assert result["bin"] == DualModelQualityJudge.classify_score(result["overall_score"])

    def test_invalid_rubric_weights_rejected(self):
        """Weights not summing to ~1.0 raise ValueError."""
        bad_weights = {"relevance": 0.5, "accuracy": 0.5, "helpfulness": 0.5, "style": 0.5, "safety": 0.5}
        primary = MagicMock()
        secondary = MagicMock()
        with pytest.raises(ValueError, match="must sum to ~1.0"):
            DualModelQualityJudge(primary, secondary, rubric_weights=bad_weights)

    def test_missing_dimension_in_weights_rejected(self):
        """Missing dimension in weights raises ValueError."""
        bad_weights = {"relevance": 0.5, "accuracy": 0.5}
        primary = MagicMock()
        secondary = MagicMock()
        with pytest.raises(ValueError, match="missing dimensions"):
            DualModelQualityJudge(primary, secondary, rubric_weights=bad_weights)


# ===========================================================================
# Test 2: Multi-turn weighting — recency-decay formula
# ===========================================================================


class TestMultiTurnWeighting:
    """Verify recency-decay weighted mean across turns."""

    def test_single_turn_weighted_mean(self, judge_with_mock_clients, uniform_response):
        """Single turn: overall = turn score (weight = 1.0)."""
        result = judge_with_mock_clients.judge(make_conversation(n_turns=1))
        # decay^(1-1-0) = decay^0 = 1.0, weight_sum = 1.0
        assert result["overall_score"] == pytest.approx(0.6, abs=0.01)

    def test_two_turn_decay_weighting(self):
        """Two turns: newer turn weighted more than older."""
        # Turn 0: score 0.3 (older), Turn 1: score 0.8 (newer)
        # decay=0.85: weights = [0.85^1, 0.85^0] = [0.85, 1.0]
        # primary weighted = (0.85*0.3 + 1.0*0.8) / (0.85+1.0) = (0.255+0.8)/1.85 = 0.5703
        low_resp = make_llm_response(0.3, 0.3, 0.3, 0.3, 0.3)
        high_resp = make_llm_response(0.8, 0.8, 0.8, 0.8, 0.8)
        # k=3 samples per turn: [low,low,low, high,high,high]
        primary = make_mock_client([low_resp] * 3 + [high_resp] * 3)
        secondary = make_mock_client([low_resp] * 3 + [high_resp] * 3)
        judge = DualModelQualityJudge(primary, secondary, decay=0.85)
        result = judge.judge(make_conversation(n_turns=2))

        # Both models return same scores, so overall = primary_weighted
        expected = (0.85 * 0.3 + 1.0 * 0.8) / (0.85 + 1.0)
        assert result["overall_score"] == pytest.approx(expected, abs=0.02)
        assert result["primary_overall"] == pytest.approx(expected, abs=0.02)

    def test_three_turn_decay_weighting(self):
        """Three turns: verify decay formula weight_i = decay^(n-1-i)."""
        # Turn 0 (oldest): 0.2, Turn 1: 0.5, Turn 2 (newest): 0.9
        # decay=0.85: weights = [0.85^2, 0.85^1, 0.85^0] = [0.7225, 0.85, 1.0]
        # weighted = (0.7225*0.2 + 0.85*0.5 + 1.0*0.9) / (0.7225+0.85+1.0)
        #           = (0.1445 + 0.425 + 0.9) / 2.5725 = 1.4695 / 2.5725 = 0.5711
        resp_low = make_llm_response(0.2, 0.2, 0.2, 0.2, 0.2)
        resp_mid = make_llm_response(0.5, 0.5, 0.5, 0.5, 0.5)
        resp_high = make_llm_response(0.9, 0.9, 0.9, 0.9, 0.9)
        # k=3 per turn: [low×3, mid×3, high×3]
        primary = make_mock_client([resp_low] * 3 + [resp_mid] * 3 + [resp_high] * 3)
        secondary = make_mock_client([resp_low] * 3 + [resp_mid] * 3 + [resp_high] * 3)
        judge = DualModelQualityJudge(primary, secondary, decay=0.85)
        result = judge.judge(make_conversation(n_turns=3))

        expected = (0.7225 * 0.2 + 0.85 * 0.5 + 1.0 * 0.9) / (0.7225 + 0.85 + 1.0)
        assert result["overall_score"] == pytest.approx(expected, abs=0.02)

    def test_custom_decay(self):
        """Custom decay factor works correctly."""
        # decay=1.0 → equal weighting (no decay)
        resp_low = make_llm_response(0.3, 0.3, 0.3, 0.3, 0.3)
        resp_high = make_llm_response(0.7, 0.7, 0.7, 0.7, 0.7)
        # k=3 per turn: [low×3, high×3]
        primary = make_mock_client([resp_low] * 3 + [resp_high] * 3)
        secondary = make_mock_client([resp_low] * 3 + [resp_high] * 3)
        judge = DualModelQualityJudge(primary, secondary, decay=1.0)
        result = judge.judge(make_conversation(n_turns=2))

        # Equal weighting: (0.3 + 0.7) / 2 = 0.5
        assert result["overall_score"] == pytest.approx(0.5, abs=0.02)

    def test_turn_scores_returned(self, judge_with_mock_clients, uniform_response):
        """Result includes per-turn scores."""
        result = judge_with_mock_clients.judge(make_conversation(n_turns=3))
        assert len(result["turn_scores"]) == 3
        for i, ts in enumerate(result["turn_scores"]):
            assert ts.turn_index == i


# ===========================================================================
# Test 3: Self-consistency — k=3, variance check
# ===========================================================================


class TestSelfConsistency:
    """Verify k=3 self-consistency sampling and variance flagging."""

    def test_low_variance_no_flag(self, uniform_response):
        """All 3 samples identical → variance=0, no flag."""
        primary = make_mock_client(uniform_response)
        secondary = make_mock_client(uniform_response)
        judge = DualModelQualityJudge(primary, secondary, k_samples=3)
        result = judge.judge(make_conversation())
        assert result["turn_scores"][0].primary_variance == pytest.approx(0.0, abs=0.001)
        assert "turn_0_primary_high_variance" not in result["flags"]

    def test_high_variance_flags_primary(self):
        """Variance > 0.05 in primary samples → flag."""
        # 3 samples: 0.2, 0.8, 0.5 → variance = pvariance([0.2,0.8,0.5])
        # mean = 0.5, pvariance = ((-0.3)^2 + 0.3^2 + 0^2) / 3 = (0.09+0.09+0)/3 = 0.06
        samples = [
            make_llm_response(0.2, 0.2, 0.2, 0.2, 0.2),
            make_llm_response(0.8, 0.8, 0.8, 0.8, 0.8),
            make_llm_response(0.5, 0.5, 0.5, 0.5, 0.5),
        ]
        primary = make_mock_client(samples)
        secondary = make_mock_client(make_llm_response(0.5, 0.5, 0.5, 0.5, 0.5))
        judge = DualModelQualityJudge(primary, secondary, k_samples=3)
        result = judge.judge(make_conversation())
        assert result["turn_scores"][0].primary_variance > 0.05
        assert "turn_0_primary_high_variance" in result["flags"]

    def test_high_variance_flags_secondary(self):
        """Variance > 0.05 in secondary samples → flag."""
        samples = [
            make_llm_response(0.2, 0.2, 0.2, 0.2, 0.2),
            make_llm_response(0.8, 0.8, 0.8, 0.8, 0.8),
            make_llm_response(0.5, 0.5, 0.5, 0.5, 0.5),
        ]
        primary = make_mock_client(make_llm_response(0.5, 0.5, 0.5, 0.5, 0.5))
        secondary = make_mock_client(samples)
        judge = DualModelQualityJudge(primary, secondary, k_samples=3)
        result = judge.judge(make_conversation())
        assert result["turn_scores"][0].secondary_variance > 0.05
        assert "turn_0_secondary_high_variance" in result["flags"]

    def test_variance_reported_in_metadata(self, uniform_response):
        """Variance values are reported in metadata."""
        primary = make_mock_client(uniform_response)
        secondary = make_mock_client(uniform_response)
        judge = DualModelQualityJudge(primary, secondary)
        result = judge.judge(make_conversation())
        assert "primary_variances" in result["metadata"]
        assert "secondary_variances" in result["metadata"]
        assert len(result["metadata"]["primary_variances"]) == 1

    def test_custom_variance_threshold(self):
        """Custom variance threshold works."""
        # pvariance([0.2, 0.8, 0.5]) = 0.06 > 0.05 default threshold
        # With threshold 0.10, should NOT flag
        samples = [
            make_llm_response(0.2, 0.2, 0.2, 0.2, 0.2),
            make_llm_response(0.8, 0.8, 0.8, 0.8, 0.8),
            make_llm_response(0.5, 0.5, 0.5, 0.5, 0.5),
        ]
        primary = make_mock_client(samples)
        secondary = make_mock_client(make_llm_response(0.5, 0.5, 0.5, 0.5, 0.5))
        judge = DualModelQualityJudge(
            primary,
            secondary,
            k_samples=3,
            self_consistency_variance_threshold=0.10,
        )
        result = judge.judge(make_conversation())
        assert "turn_0_primary_high_variance" not in result["flags"]


# ===========================================================================
# Test 4: Cross-model consistency rule
# ===========================================================================


class TestConsistencyRule:
    """Verify primary/secondary agreement/disagreement paths."""

    def test_models_agree_no_flag(self, uniform_response):
        """Both models return same scores → no cross_model_inconsistent flag."""
        primary = make_mock_client(uniform_response)
        secondary = make_mock_client(uniform_response)
        judge = DualModelQualityJudge(primary, secondary)
        result = judge.judge(make_conversation())
        assert "cross_model_inconsistent" not in result["flags"]
        assert result["consistency_diff"] <= 0.15

    def test_models_disagree_flags(self):
        """Models disagree by > 0.15 → cross_model_inconsistent flag."""
        high_resp = make_llm_response(0.9, 0.9, 0.9, 0.9, 0.9)
        low_resp = make_llm_response(0.3, 0.3, 0.3, 0.3, 0.3)
        primary = make_mock_client([high_resp] * 3)
        secondary = make_mock_client([low_resp] * 3)
        judge = DualModelQualityJudge(primary, secondary)
        result = judge.judge(make_conversation())
        # diff = |0.9 - 0.3| = 0.6 > 0.15
        assert result["consistency_diff"] > 0.15
        assert "cross_model_inconsistent" in result["flags"]

    def test_models_marginally_agree(self):
        """Models differ by exactly 0.15 → no flag (boundary case)."""
        # 0.6 and 0.75 → diff = 0.15
        resp1 = make_llm_response(0.6, 0.6, 0.6, 0.6, 0.6)
        resp2 = make_llm_response(0.75, 0.75, 0.75, 0.75, 0.75)
        primary = make_mock_client([resp1] * 3)
        secondary = make_mock_client([resp2] * 3)
        judge = DualModelQualityJudge(primary, secondary)
        result = judge.judge(make_conversation())
        # diff = 0.15, threshold is 0.15 → NOT flagged (> not >=)
        assert result["consistency_diff"] == pytest.approx(0.15, abs=0.01)
        assert "cross_model_inconsistent" not in result["flags"]

    def test_custom_consistency_threshold(self):
        """Custom consistency threshold works."""
        # diff = 0.1, threshold = 0.05 → should flag
        resp1 = make_llm_response(0.6, 0.6, 0.6, 0.6, 0.6)
        resp2 = make_llm_response(0.7, 0.7, 0.7, 0.7, 0.7)
        primary = make_mock_client([resp1] * 3)
        secondary = make_mock_client([resp2] * 3)
        judge = DualModelQualityJudge(
            primary,
            secondary,
            consistency_threshold=0.05,
        )
        result = judge.judge(make_conversation())
        assert "cross_model_inconsistent" in result["flags"]


# ===========================================================================
# Test 5: Empty conversation handling
# ===========================================================================


class TestEmptyConversation:
    """Empty conversations return safe defaults."""

    def test_empty_list(self, judge_with_mock_clients):
        """Empty conversation list → 0.0 score, empty_conversation flag."""
        result = judge_with_mock_clients.judge([])
        assert result["overall_score"] == 0.0
        assert "empty_conversation" in result["flags"]
        assert result["turn_scores"] == []

    def test_no_assistant_turns(self, judge_with_mock_clients):
        """Conversation with only user messages → empty result."""
        conv = [{"role": "user", "content": "Hello?"}]
        result = judge_with_mock_clients.judge(conv)
        assert result["overall_score"] == 0.0
        assert "empty_conversation" in result["flags"]


# ===========================================================================
# Test 6: Calibration — Pearson r and Cohen's kappa
# ===========================================================================


class TestCalibration:
    """Verify calibration against the golden 200-sample set."""

    def test_calibrate_with_perfect_mock(self, tmp_path):
        """Mock LLM returns human scores exactly → Pearson=1.0, kappa=1.0."""
        # Load golden file, create a mock that returns each sample's human_scores
        golden_path = Path(__file__).resolve().parent.parent / "data" / "golden_judge_calib.jsonl"
        if not golden_path.exists():
            pytest.skip(f"Golden file not found: {golden_path}")

        samples = []
        with golden_path.open() as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        # Create mock clients that return human_scores as dimension_scores
        # The mock needs to return based on the prompt — we use side_effect
        # Since _call_model_sync is called with the turn text, and the mock
        # returns the same response for all calls, we need to cycle through
        # the responses per sample.
        # But actually, judge.judge() is called per sample, and each sample
        # has 1 assistant turn with k=3 samples per model → 6 calls per sample.
        # We need to map each sample to its human scores.

        # Strategy: mock _call_model_sync to return the correct response
        # based on which sample is currently being judged.
        # We'll use a queue approach.

        judge = DualModelQualityJudge(
            primary_client=MagicMock(),
            secondary_client=MagicMock(),
        )

        # Patch _call_model_sync to return human scores for each sample
        call_idx = [0]

        def mock_call_model(client, turn_text):
            # Each sample calls _call_model_sync 6 times (k=3 × 2 models)
            # We cycle through samples: calls 0-5 → sample 0, 6-11 → sample 1, etc.
            sample_idx = call_idx[0] // 6
            call_idx[0] += 1
            if sample_idx >= len(samples):
                return make_llm_response(0.5, 0.5, 0.5, 0.5, 0.5)
            hs = samples[sample_idx]["human_scores"]
            return make_llm_response(
                hs.get("relevance", 0.5),
                hs.get("accuracy", 0.5),
                hs.get("helpfulness", 0.5),
                hs.get("style", 0.5),
                hs.get("safety", 0.5),
            )

        with patch.object(judge, "_call_model_sync", side_effect=mock_call_model):
            report = judge.calibrate(golden_path)

        assert report["pearson_r"] >= 0.80
        assert report["cohens_kappa"] >= 0.65
        assert report["pass"] is True
        assert report["sample_count"] == 200

    def test_calibrate_returns_per_dimension(self, uniform_response):
        """calibrate() returns per-dimension correlations."""
        golden_path = Path(__file__).resolve().parent.parent / "data" / "golden_judge_calib.jsonl"
        if not golden_path.exists():
            pytest.skip(f"Golden file not found: {golden_path}")

        primary = make_mock_client(uniform_response)
        secondary = make_mock_client(uniform_response)
        judge = DualModelQualityJudge(primary, secondary)

        report = judge.calibrate(golden_path)
        assert "per_dimension_correlations" in report
        for dim in DualModelQualityJudge.DIMENSIONS:
            assert dim in report["per_dimension_correlations"]

    def test_calibrate_missing_file_raises(self, tmp_path):
        """Missing golden file raises FileNotFoundError."""
        primary = MagicMock()
        secondary = MagicMock()
        judge = DualModelQualityJudge(primary, secondary)
        with pytest.raises(FileNotFoundError):
            judge.calibrate(tmp_path / "nonexistent.jsonl")


# ===========================================================================
# Test 7: Async interface — concurrent batching
# ===========================================================================


class TestAsyncInterface:
    """Verify async ajudge() with concurrent batching."""

    @pytest.mark.asyncio
    async def test_ajudge_returns_same_structure(self, uniform_response):
        """ajudge() returns same result structure as judge()."""
        primary = make_mock_client(uniform_response)
        secondary = make_mock_client(uniform_response)
        judge = DualModelQualityJudge(primary, secondary)
        result = await judge.ajudge(make_conversation())
        assert "overall_score" in result
        assert "bin" in result
        assert "turn_scores" in result
        assert len(result["turn_scores"]) == 1

    @pytest.mark.asyncio
    async def test_ajudge_matches_sync_score(self, uniform_response):
        """ajudge() and judge() produce the same overall score."""
        primary = make_mock_client(uniform_response)
        secondary = make_mock_client(uniform_response)
        judge = DualModelQualityJudge(primary, secondary)
        sync_result = judge.judge(make_conversation())

        # Reset mocks (side_effect consumed by sync call)
        primary = make_mock_client(uniform_response)
        secondary = make_mock_client(uniform_response)
        judge.primary_client = primary
        judge.secondary_client = secondary

        async_result = await judge.ajudge(make_conversation())
        assert async_result["overall_score"] == pytest.approx(sync_result["overall_score"], abs=0.01)

    @pytest.mark.asyncio
    async def test_ajudge_empty_conversation(self, judge_with_mock_clients):
        """ajudge() handles empty conversation gracefully."""
        result = await judge_with_mock_clients.ajudge([])
        assert result["overall_score"] == 0.0
        assert "empty_conversation" in result["flags"]

    @pytest.mark.asyncio
    async def test_ajudge_multi_turn(self):
        """ajudge() handles multiple turns correctly."""
        resp = make_llm_response(0.7, 0.7, 0.7, 0.7, 0.7)
        primary = make_mock_client([resp] * 6)  # 2 turns × 3 samples
        secondary = make_mock_client([resp] * 6)
        judge = DualModelQualityJudge(primary, secondary)
        result = await judge.ajudge(make_conversation(n_turns=2))
        assert len(result["turn_scores"]) == 2
        assert result["overall_score"] == pytest.approx(0.7, abs=0.01)

    @pytest.mark.asyncio
    async def test_ajudge_runs_concurrently(self):
        """Verify that async batches calls concurrently (not sequentially)."""
        import time

        call_times: list[float] = []

        def slow_generate_structured(prompt, schema, system_prompt):
            call_times.append(time.monotonic())
            time.sleep(0.05)  # 50ms per call
            return make_llm_response(0.5, 0.5, 0.5, 0.5, 0.5)

        primary = MagicMock()
        primary.generate_structured.side_effect = slow_generate_structured
        secondary = MagicMock()
        secondary.generate_structured.side_effect = slow_generate_structured
        judge = DualModelQualityJudge(primary, secondary, k_samples=3)

        start = time.monotonic()
        result = await judge.ajudge(make_conversation(n_turns=1))
        elapsed = time.monotonic() - start

        # 6 calls (3 primary + 3 secondary) × 50ms = 300ms sequential
        # Concurrent should be ~50ms (all run in parallel)
        # Allow generous margin for thread pool overhead
        assert elapsed < 0.25  # Should be well under 300ms


def test_temperature_passed_when_client_supports_it():
    class TempClient:
        def __init__(self):
            self.last_kwargs: dict | None = None

        def generate_structured(self, prompt, schema, system_prompt=None, **kwargs):
            self.last_kwargs = kwargs
            return make_llm_response(0.7, 0.7, 0.7, 0.7, 0.7)

    primary = TempClient()
    secondary = TempClient()
    judge = DualModelQualityJudge(primary, secondary, temperature=0.1)
    judge.judge(make_conversation(n_turns=1))
    assert primary.last_kwargs is not None and primary.last_kwargs.get("temperature") == 0.1
    assert secondary.last_kwargs is not None and secondary.last_kwargs.get("temperature") == 0.1


def test_llm_call_failed_flag_when_all_samples_are_none():
    primary = MagicMock()
    primary.generate_structured.return_value = None
    secondary = MagicMock()
    secondary.generate_structured.return_value = None
    judge = DualModelQualityJudge(primary, secondary)
    result = judge.judge(make_conversation(n_turns=1))
    assert "llm_call_failed" in result["flags"]
    assert result["needs_human_review"] is True
    assert 0 in result["metadata"]["failed_turns"]
