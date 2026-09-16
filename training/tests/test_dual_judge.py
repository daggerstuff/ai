"""Tests for dual_judge — standalone dual-model LLM QA judge (PIX-4343).

Covers: JSON parsing of judge output, recency-decay weighted mean, multi-turn
aggregation, self-consistency (k=3, variance check), dual-model reconciliation,
calibration (Pearson r + Cohen's kappa), and async judge_single / judge_record_turns
with mocked aiohttp transport.

All HTTP calls are mocked — no vLLM or GPU required.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from training.dual_judge import (
    ACCEPT_THRESHOLD,
    CALIB_KAPPA_MIN,
    CALIB_PEARSON_MIN,
    DIMENSIONS,
    DUAL_CONSISTENCY_DIFF_MAX,
    RECENCY_DECAY,
    SELF_CONSISTENCY_VARIANCE_MAX,
    CalibrationReport,
    JudgeVerdict,
    aggregate_turn_verdicts,
    cohen_kappa,
    evaluate_calibration,
    judge_record_turns,
    judge_single,
    parse_judge_json,
    pearson_r,
    recency_weighted_mean,
    reconcile_dual,
    runs_self_consistent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_verdict(
    quality: float = 0.5,
    reject_reason: str = "",
    dims: dict[str, float] | None = None,
    reasoning: str = "test",
) -> JudgeVerdict:
    """Create a JudgeVerdict with all five dimensions filled."""
    if dims is None:
        dims = dict.fromkeys(DIMENSIONS, quality)
    return JudgeVerdict(
        quality_score=quality,
        reject_reason=reject_reason,
        dim_scores=dims,
        reasoning=reasoning,
    )


def make_judge_json_response(
    quality: float = 0.7,
    dims: dict[str, float] | None = None,
    reject_reason: str = "",
    reasoning: str = "looks good",
) -> str:
    """Produce the JSON string a judge model would return."""
    if dims is None:
        dims = dict.fromkeys(DIMENSIONS, quality)
    return json.dumps(
        {
            "quality_score": quality,
            "reject_reason": reject_reason,
            "dim_scores": dims,
            "reasoning": reasoning,
        }
    )


def make_mock_aiohttp_response(content: str, status: int = 200) -> MagicMock:
    """Create a mock aiohttp response object."""
    resp = MagicMock()
    resp.status = status
    resp.text = AsyncMock(return_value="error body" if status != 200 else "")
    resp.json = AsyncMock(return_value={"choices": [{"message": {"content": content}}]})
    return resp


def make_mock_session(post_side_effect) -> MagicMock:
    """Create a mock aiohttp.ClientSession with the given post side_effect."""
    session = MagicMock()
    session.post = MagicMock(side_effect=post_side_effect)
    session.close = AsyncMock()
    return session


class _AsyncCM:
    """Minimal async context manager wrapping a value."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *args):
        return False


_async_cm = _AsyncCM  # alias for lambda use in test mocks


# ---------------------------------------------------------------------------
# Test 1: parse_judge_json — JSON parsing from LLM output
# ===========================================================================


class TestParseJudgeJson:
    """Verify robust parsing of judge model JSON output."""

    def test_clean_json(self):
        """Straightforward JSON payload parses correctly."""
        content = make_judge_json_response(quality=0.85)
        v = parse_judge_json(content)
        assert v.quality_score == pytest.approx(0.85, abs=0.001)
        assert v.reject_reason == ""
        assert v.reasoning == "looks good"
        for d in DIMENSIONS:
            assert v.dim_scores[d] == pytest.approx(0.85, abs=0.001)

    def test_fenced_json_block(self):
        """```json fenced block is extracted and parsed."""
        content = f"```json\n{make_judge_json_response(0.9)}\n```"
        v = parse_judge_json(content)
        assert v.quality_score == pytest.approx(0.9, abs=0.001)

    def test_fenced_plain_block(self):
        """``` (no language tag) fenced block is extracted."""
        content = f"```\n{make_judge_json_response(0.6)}\n```"
        v = parse_judge_json(content)
        assert v.quality_score == pytest.approx(0.6, abs=0.001)

    def test_leading_prose(self):
        """Leading prose before JSON is tolerated."""
        content = f"Here is my evaluation:\n{make_judge_json_response(0.75)}"
        v = parse_judge_json(content)
        assert v.quality_score == pytest.approx(0.75, abs=0.001)

    def test_trailing_prose(self):
        """Trailing prose after JSON is tolerated."""
        content = f"{make_judge_json_response(0.5)}\nHope this helps!"
        v = parse_judge_json(content)
        assert v.quality_score == pytest.approx(0.5, abs=0.001)

    def test_empty_content(self):
        """Empty string returns zero verdict."""
        v = parse_judge_json("")
        assert v.quality_score == 0.0
        assert v.reject_reason == "empty_judge_output"

    def test_whitespace_only(self):
        """Whitespace-only content returns zero verdict."""
        v = parse_judge_json("   \n\t  ")
        assert v.quality_score == 0.0
        assert v.reject_reason == "empty_judge_output"

    def test_malformed_json(self):
        """Invalid JSON returns zero verdict with parse error reason."""
        v = parse_judge_json("{bad json}")
        assert v.quality_score == 0.0
        assert "json_parse_error" in v.reject_reason

    def test_missing_quality_score(self):
        """Missing quality_score defaults to 0.0."""
        content = json.dumps({"dim_scores": dict.fromkeys(DIMENSIONS, 0.5)})
        v = parse_judge_json(content)
        assert v.quality_score == 0.0

    def test_missing_dim_scores(self):
        """Missing dim_scores defaults all dims to 0.0."""
        content = json.dumps({"quality_score": 0.7, "reasoning": "ok"})
        v = parse_judge_json(content)
        assert v.quality_score == pytest.approx(0.7, abs=0.001)
        for d in DIMENSIONS:
            assert v.dim_scores[d] == 0.0

    def test_missing_dimension_key(self):
        """Missing dimension key defaults to 0.0."""
        content = json.dumps(
            {
                "quality_score": 0.8,
                "dim_scores": {"relevance": 0.9, "accuracy": 0.8, "helpfulness": 0.7, "style": 0.6},
                "reasoning": "missing safety",
            }
        )
        v = parse_judge_json(content)
        assert v.dim_scores["safety"] == 0.0
        assert v.dim_scores["relevance"] == pytest.approx(0.9, abs=0.001)

    def test_non_numeric_quality_score(self):
        """Non-numeric quality_score defaults to 0.0."""
        content = json.dumps({"quality_score": "high", "dim_scores": {}})
        v = parse_judge_json(content)
        assert v.quality_score == 0.0

    def test_non_numeric_dim_score(self):
        """Non-numeric dim_score defaults to 0.0."""
        content = json.dumps(
            {
                "quality_score": 0.7,
                "dim_scores": {"relevance": "high", "accuracy": 0.8},
            }
        )
        v = parse_judge_json(content)
        assert v.dim_scores["relevance"] == 0.0
        assert v.dim_scores["accuracy"] == pytest.approx(0.8, abs=0.001)

    def test_quality_score_clamped_to_range(self):
        """Quality score > 1.0 is clamped to 1.0."""
        content = json.dumps({"quality_score": 1.5, "dim_scores": {}})
        v = parse_judge_json(content)
        assert v.quality_score == 1.0

    def test_quality_score_clamped_negative(self):
        """Quality score < 0.0 is clamped to 0.0."""
        content = json.dumps({"quality_score": -0.5, "dim_scores": {}})
        v = parse_judge_json(content)
        assert v.quality_score == 0.0

    def test_dim_scores_not_a_dict(self):
        """dim_scores that is a list instead of dict is tolerated."""
        content = json.dumps({"quality_score": 0.7, "dim_scores": [0.5, 0.5]})
        v = parse_judge_json(content)
        assert v.quality_score == pytest.approx(0.7, abs=0.001)
        for d in DIMENSIONS:
            assert v.dim_scores[d] == 0.0

    def test_reject_reason_empty_string(self):
        """Empty reject_reason string is preserved."""
        content = make_judge_json_response(0.8, reject_reason="")
        v = parse_judge_json(content)
        assert v.reject_reason == ""

    def test_reject_reason_none(self):
        """reject_reason: null in JSON is treated as empty string."""
        content = json.dumps({"quality_score": 0.3, "reject_reason": None, "dim_scores": {}})
        v = parse_judge_json(content)
        assert v.reject_reason == ""

    def test_to_dict_roundtrip(self):
        """JudgeVerdict.to_dict() produces correct structure."""
        v = make_verdict(0.7, dims={"relevance": 0.8, "accuracy": 0.7, "helpfulness": 0.6, "style": 0.7, "safety": 0.9})
        d = v.to_dict()
        assert d["quality_score"] == pytest.approx(0.7, abs=0.001)
        assert d["dim_scores"]["relevance"] == pytest.approx(0.8, abs=0.001)
        assert d["dim_scores"]["safety"] == pytest.approx(0.9, abs=0.001)

    def test_accepted_property_true(self):
        """JudgeVerdict.accepted is True when score >= threshold."""
        v = JudgeVerdict(quality_score=0.60)
        assert v.accepted is True

    def test_accepted_property_false(self):
        """JudgeVerdict.accepted is False when score < threshold."""
        v = JudgeVerdict(quality_score=0.59)
        assert v.accepted is False


# ---------------------------------------------------------------------------
# Test 2: recency_weighted_mean — decay formula
# ===========================================================================


class TestRecencyWeightedMean:
    """Verify recency-decay weighted mean computation."""

    def test_empty_scores(self):
        """Empty list returns 0.0."""
        assert recency_weighted_mean([]) == 0.0

    def test_single_score(self):
        """Single score returns itself (weight = decay^0 = 1.0)."""
        assert recency_weighted_mean([0.7]) == pytest.approx(0.7, abs=0.001)

    def test_two_scores_default_decay(self):
        """Two scores with default decay=0.85: turn_0 weighted less."""
        # scores[0]=0.3 (older), scores[1]=0.8 (newer)
        # weights = [0.85^1, 0.85^0] = [0.85, 1.0]
        # weighted = (0.85*0.3 + 1.0*0.8) / (0.85+1.0) = 1.055/1.85 = 0.5703
        result = recency_weighted_mean([0.3, 0.8])
        expected = (0.85 * 0.3 + 1.0 * 0.8) / (0.85 + 1.0)
        assert result == pytest.approx(expected, abs=0.001)

    def test_three_scores_default_decay(self):
        """Three scores: weights = [0.85^2, 0.85^1, 0.85^0] = [0.7225, 0.85, 1.0]."""
        scores = [0.2, 0.5, 0.9]
        weights = [0.85 ** (3 - 1 - i) for i in range(3)]
        expected = sum(s * w for s, w in zip(scores, weights, strict=False)) / sum(weights)
        assert recency_weighted_mean(scores) == pytest.approx(expected, abs=0.001)

    def test_decay_one_equal_weighting(self):
        """decay=1.0 → equal weighting (no decay)."""
        result = recency_weighted_mean([0.3, 0.7], decay=1.0)
        assert result == pytest.approx(0.5, abs=0.001)

    def test_custom_decay(self):
        """Custom decay factor works correctly."""
        result = recency_weighted_mean([0.3, 0.8], decay=0.5)
        expected = (0.5 * 0.3 + 1.0 * 0.8) / 1.5
        assert result == pytest.approx(expected, abs=0.001)

    def test_newest_turn_highest_weight(self):
        """The last score (newest) always has the highest weight."""
        # With decay<1, the last element should have weight 1.0
        scores = [0.1, 0.9]
        result = recency_weighted_mean(scores, decay=0.85)
        # Result should be closer to 0.9 (newer) than simple mean of 0.5
        assert result > 0.5


# ---------------------------------------------------------------------------
# Test 3: aggregate_turn_verdicts — multi-turn aggregation
# ===========================================================================


class TestAggregateTurnVerdicts:
    """Verify aggregation of per-turn verdicts."""

    def test_empty_verdicts(self):
        """Empty list returns zero verdict with no_turns reason."""
        v = aggregate_turn_verdicts([])
        assert v.quality_score == 0.0
        assert v.reject_reason == "no_turns"

    def test_single_verdict(self):
        """Single verdict: aggregated score = verdict score."""
        v = make_verdict(0.7)
        result = aggregate_turn_verdicts([v])
        assert result.quality_score == pytest.approx(0.7, abs=0.001)

    def test_multiple_verdicts_decay(self):
        """Multiple verdicts aggregated with recency decay."""
        v1 = make_verdict(0.3)
        v2 = make_verdict(0.8)
        result = aggregate_turn_verdicts([v1, v2])
        expected = recency_weighted_mean([0.3, 0.8])
        assert result.quality_score == pytest.approx(expected, abs=0.001)

    def test_dim_scores_aggregated(self):
        """Dimension scores are also aggregated with decay."""
        v1 = make_verdict(0.3, dims=dict.fromkeys(DIMENSIONS, 0.3))
        v2 = make_verdict(0.8, dims=dict.fromkeys(DIMENSIONS, 0.8))
        result = aggregate_turn_verdicts([v1, v2])
        expected = recency_weighted_mean([0.3, 0.8])
        for d in DIMENSIONS:
            assert result.dim_scores[d] == pytest.approx(expected, abs=0.001)

    def test_reject_reasons_joined(self):
        """Reject reasons from verdicts are joined (max 3)."""
        verdicts = [
            make_verdict(0.3, reject_reason="reason_a"),
            make_verdict(0.4, reject_reason="reason_b"),
            make_verdict(0.5, reject_reason="reason_c"),
            make_verdict(0.6, reject_reason="reason_d"),
        ]
        result = aggregate_turn_verdicts(verdicts)
        assert "reason_a" in result.reject_reason
        assert "reason_b" in result.reject_reason
        assert "reason_c" in result.reject_reason
        # Only first 3 reasons
        assert "reason_d" not in result.reject_reason

    def test_reasoning_mentions_turn_count(self):
        """Reasoning field mentions the number of turns."""
        verdicts = [make_verdict(0.5), make_verdict(0.6), make_verdict(0.7)]
        result = aggregate_turn_verdicts(verdicts)
        assert "3 turns" in result.reasoning

    def test_custom_decay(self):
        """Custom decay parameter is passed through."""
        v1 = make_verdict(0.3)
        v2 = make_verdict(0.8)
        result = aggregate_turn_verdicts([v1, v2], decay=1.0)
        assert result.quality_score == pytest.approx(0.55, abs=0.001)


# ---------------------------------------------------------------------------
# Test 4: runs_self_consistent — k-run variance check
# ===========================================================================


class TestRunsSelfConsistent:
    """Verify self-consistency variance check."""

    def test_single_verdict_consistent(self):
        """Single verdict (no variance) is always consistent."""
        v = make_verdict(0.7)
        assert runs_self_consistent([v]) is True

    def test_identical_verdicts_consistent(self):
        """Identical verdicts have zero variance → consistent."""
        verdicts = [make_verdict(0.7)] * 3
        assert runs_self_consistent(verdicts) is True

    def test_low_variance_consistent(self):
        """Small score differences stay under threshold."""
        # scores = [0.70, 0.72, 0.71]
        # variance = mean of squared deviations from mean(0.71)
        # = ((0.01)^2 + (0.01)^2 + 0) / 3 ≈ 0.000067
        verdicts = [make_verdict(0.70), make_verdict(0.72), make_verdict(0.71)]
        assert runs_self_consistent(verdicts) is True

    def test_high_variance_inconsistent(self):
        """Large score differences exceed threshold."""
        # scores = [0.2, 0.8, 0.5]
        # mean = 0.5, variance = (0.09+0.09+0)/3 = 0.06 > 0.05
        verdicts = [make_verdict(0.2), make_verdict(0.8), make_verdict(0.5)]
        assert runs_self_consistent(verdicts) is False

    def test_exact_threshold_boundary(self):
        """Variance exactly at threshold → consistent (<=)."""
        # variance = 0.05 exactly → should be consistent
        # mean = 0.5, find scores with pvariance = 0.05
        # scores = [0.5-x, 0.5+x, 0.5] → variance = 2x²/3 = 0.05 → x = sqrt(0.075) ≈ 0.2739
        x = math.sqrt(0.075)
        verdicts = [make_verdict(0.5 - x), make_verdict(0.5 + x), make_verdict(0.5)]
        assert runs_self_consistent(verdicts) is True

    def test_custom_variance_threshold(self):
        """Custom variance threshold works."""
        # variance = 0.06 > default 0.05 but < custom 0.10
        verdicts = [make_verdict(0.2), make_verdict(0.8), make_verdict(0.5)]
        assert runs_self_consistent(verdicts, variance_max=0.10) is True
        assert runs_self_consistent(verdicts, variance_max=0.05) is False

    def test_empty_list_consistent(self):
        """Empty list returns True (no evidence of inconsistency)."""
        assert runs_self_consistent([]) is True


# ---------------------------------------------------------------------------
# Test 5: reconcile_dual — dual-model consistency rule
# ===========================================================================


class TestReconcileDual:
    """Verify B.3.2 dual-judge reconciliation."""

    def test_models_agree_accept(self):
        """|primary - secondary| <= 0.15 → accept primary."""
        primary = make_verdict(0.8)
        secondary = make_verdict(0.7)
        result = reconcile_dual(primary, secondary)
        assert result.accepted is True
        assert result.needs_human_review is False
        assert "dual_consistent" in result.reason

    def test_models_disagree_human_review(self):
        """|primary - secondary| > 0.15 → human review."""
        primary = make_verdict(0.9)
        secondary = make_verdict(0.3)
        result = reconcile_dual(primary, secondary)
        assert result.accepted is False
        assert result.needs_human_review is True
        assert "dual_inconsistent" in result.reason

    def test_boundary_exactly_threshold(self):
        """|diff| = 0.15 exactly → accept (<=)."""
        primary = make_verdict(0.75)
        secondary = make_verdict(0.60)
        result = reconcile_dual(primary, secondary)
        assert result.accepted is True
        assert result.needs_human_review is False

    def test_just_above_threshold(self):
        """|diff| = 0.151 → human review."""
        primary = make_verdict(0.751)
        secondary = make_verdict(0.60)
        result = reconcile_dual(primary, secondary)
        assert result.needs_human_review is True

    def test_primary_rejected_even_when_consistent(self):
        """If primary score < threshold, accepted=False even if models agree."""
        primary = make_verdict(0.4)
        secondary = make_verdict(0.5)
        result = reconcile_dual(primary, secondary)
        assert result.accepted is False
        assert result.needs_human_review is False  # consistent, just low score

    def test_custom_diff_max(self):
        """Custom diff_max threshold."""
        primary = make_verdict(0.7)
        secondary = make_verdict(0.6)
        # diff = 0.1 > 0.05 custom → human review
        result = reconcile_dual(primary, secondary, diff_max=0.05)
        assert result.needs_human_review is True

    def test_dual_result_to_dict(self):
        """DualJudgeResult.to_dict() produces correct structure."""
        primary = make_verdict(0.8)
        secondary = make_verdict(0.75)
        result = reconcile_dual(primary, secondary)
        d = result.to_dict()
        assert "primary" in d
        assert "secondary" in d
        assert "accepted" in d
        assert "needs_human_review" in d
        assert "reason" in d
        assert d["primary"]["quality_score"] == pytest.approx(0.8, abs=0.001)

    def test_diff_rounded_in_reason(self):
        """Diff is rounded to 3 decimal places in the reason string."""
        primary = make_verdict(0.723)
        secondary = make_verdict(0.612)
        result = reconcile_dual(primary, secondary)
        # diff = 0.111 → should appear as 0.111 in reason
        assert "0.111" in result.reason


# ---------------------------------------------------------------------------
# Test 6: pearson_r — correlation
# ===========================================================================


class TestPearsonR:
    """Verify Pearson correlation computation."""

    def test_perfect_positive(self):
        """Perfect positive correlation → r = 1.0."""
        a = [0.1, 0.2, 0.3, 0.4, 0.5]
        b = [0.2, 0.4, 0.6, 0.8, 1.0]
        assert pearson_r(a, b) == pytest.approx(1.0, abs=0.001)

    def test_perfect_negative(self):
        """Perfect negative correlation → r = -1.0."""
        a = [0.1, 0.2, 0.3, 0.4, 0.5]
        b = [1.0, 0.8, 0.6, 0.4, 0.2]
        assert pearson_r(a, b) == pytest.approx(-1.0, abs=0.001)

    def test_no_correlation(self):
        """No linear correlation → r ≈ 0."""
        a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        b = [0.5, 0.8, 0.2, 0.6, 0.1, 0.4, 0.9, 0.3]
        r = pearson_r(a, b)
        assert abs(r) < 0.2  # weak/no correlation

    def test_empty_vectors(self):
        """Empty vectors → r = 0.0."""
        assert pearson_r([], []) == 0.0

    def test_single_element(self):
        """Single element vectors → r = 0.0 (can't compute)."""
        assert pearson_r([0.5], [0.5]) == 0.0

    def test_unequal_lengths(self):
        """Unequal length vectors use min length."""
        a = [0.1, 0.2, 0.3, 0.4, 0.5]
        b = [0.2, 0.4, 0.6]
        r = pearson_r(a, b)
        # First 3 elements are perfectly correlated
        assert r == pytest.approx(1.0, abs=0.001)

    def test_zero_variance(self):
        """Constant vector (zero variance) → r = 0.0."""
        a = [0.5, 0.5, 0.5]
        b = [0.1, 0.5, 0.9]
        assert pearson_r(a, b) == 0.0


# ---------------------------------------------------------------------------
# Test 7: cohen_kappa — agreement metric
# ===========================================================================


class TestCohenKappa:
    """Verify Cohen's kappa on accept/reject decisions."""

    def test_perfect_agreement(self):
        """Perfect agreement → kappa = 1.0."""
        golden = [True, True, False, False, True]
        judge = [True, True, False, False, True]
        assert cohen_kappa(golden, judge) == pytest.approx(1.0, abs=0.001)

    def test_perfect_disagreement(self):
        """Perfect disagreement → kappa < 0."""
        golden = [True, True, False, False]
        judge = [False, False, True, True]
        k = cohen_kappa(golden, judge)
        assert k < 0

    def test_random_agreement(self):
        """Random agreement → kappa ≈ 0."""
        # 50/50 split with 50% agreement
        golden = [True, False, True, False, True, False, True, False]
        judge = [True, True, False, False, True, True, False, False]
        k = cohen_kappa(golden, judge)
        assert abs(k) < 0.3  # near zero

    def test_empty_lists(self):
        """Empty lists → kappa = 0.0."""
        assert cohen_kappa([], []) == 0.0

    def test_all_accept(self):
        """All accept → p_e = 1.0 → kappa = 1.0 (edge case)."""
        golden = [True, True, True]
        judge = [True, True, True]
        assert cohen_kappa(golden, judge) == pytest.approx(1.0, abs=0.001)

    def test_all_reject(self):
        """All reject → p_e = 1.0 → kappa = 1.0 (edge case)."""
        golden = [False, False, False]
        judge = [False, False, False]
        assert cohen_kappa(golden, judge) == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# Test 8: evaluate_calibration — release gate
# ===========================================================================


class TestEvaluateCalibration:
    """Verify calibration evaluation against golden set."""

    def test_perfect_calibration_passes(self):
        """Judge scores = golden scores → passes gate."""
        golden = [0.9, 0.8, 0.7, 0.3, 0.2] * 40  # 200 samples
        judge = golden[:]
        report = evaluate_calibration(judge, golden)
        assert report.passes is True
        assert report.pearson_r == pytest.approx(1.0, abs=0.001)
        assert report.cohen_kappa == pytest.approx(1.0, abs=0.001)
        assert report.sample_count == 200
        assert report.reason == "release_ready"

    def test_below_pearson_fails(self):
        """Pearson below threshold → fails."""
        golden = [0.9] * 10 + [0.1] * 10
        judge = [0.9] * 10 + [0.9] * 10  # No correlation with golden for second half
        report = evaluate_calibration(judge, golden, pearson_min=0.80)
        assert report.passes is False
        assert "below_gate" in report.reason
        assert "pearson" in report.reason

    def test_below_kappa_fails(self):
        """Kappa below threshold → fails."""
        # Pearson might be high but kappa low if accept/reject boundary differs
        golden = [0.59, 0.61] * 50  # alternating around 0.60 threshold
        judge = [0.59, 0.59] * 50  # always rejects → kappa low
        report = evaluate_calibration(judge, golden, kappa_min=0.65)
        assert report.passes is False
        assert "below_gate" in report.reason

    def test_empty_sets(self):
        """Empty calibration set → fails with empty_calibration_set."""
        report = evaluate_calibration([], [])
        assert report.passes is False
        assert report.reason == "empty_calibration_set"
        assert report.sample_count == 0

    def test_custom_thresholds(self):
        """Custom thresholds work."""
        golden = [0.9, 0.1, 0.9, 0.1]
        judge = [0.85, 0.15, 0.85, 0.15]
        # Should pass with loose thresholds
        report = evaluate_calibration(judge, golden, pearson_min=0.5, kappa_min=0.3)
        assert report.passes is True

    def test_unequal_lengths(self):
        """Unequal length vectors use min length."""
        golden = [0.9, 0.8, 0.7, 0.6, 0.5]
        judge = [0.9, 0.8, 0.7]
        report = evaluate_calibration(judge, golden)
        assert report.sample_count == 3

    def test_custom_accept_threshold(self):
        """Custom accept threshold for kappa computation."""
        golden = [0.7, 0.5, 0.7, 0.5]
        judge = [0.7, 0.5, 0.7, 0.5]
        # With threshold=0.6: golden=[True,False,True,False], judge same
        report = evaluate_calibration(judge, golden, accept_threshold=0.6)
        assert report.passes is True
        assert report.cohen_kappa == pytest.approx(1.0, abs=0.001)

    def test_calibration_report_dataclass(self):
        """CalibrationReport has all expected fields."""
        report = CalibrationReport(
            pearson_r=0.85,
            cohen_kappa=0.70,
            sample_count=200,
            passes=True,
            reason="release_ready",
        )
        assert report.pearson_r == 0.85
        assert report.cohen_kappa == 0.70
        assert report.sample_count == 200
        assert report.passes is True
        assert report.reason == "release_ready"


# ---------------------------------------------------------------------------
# Test 9: judge_single — async with mocked HTTP
# ===========================================================================


class TestJudgeSingle:
    """Verify judge_single with mocked aiohttp transport."""

    @pytest.mark.asyncio
    async def test_consistent_models_accept(self):
        """Both models return same score → accepted."""
        json_response = make_judge_json_response(0.8)
        mock_resp = make_mock_aiohttp_response(json_response)
        mock_session = make_mock_session(post_side_effect=lambda *a, **kw: _async_cm(mock_resp))
        result = await judge_single(
            candidate="test answer",
            reference="reference answer",
            session=mock_session,
        )
        assert result.accepted is True
        assert result.needs_human_review is False
        assert result.primary.quality_score == pytest.approx(0.8, abs=0.001)

    @pytest.mark.asyncio
    async def test_models_disagree_human_review(self):
        """Models disagree → needs_human_review."""
        high_json = make_judge_json_response(0.9)
        low_json = make_judge_json_response(0.3)
        # Primary gets 0.9 (3 runs), secondary gets 0.3
        responses = [
            make_mock_aiohttp_response(high_json),
            make_mock_aiohttp_response(high_json),
            make_mock_aiohttp_response(high_json),
            make_mock_aiohttp_response(low_json),
        ]
        mock_session = make_mock_session(post_side_effect=lambda *a, **kw: _async_cm(responses.pop(0)))
        result = await judge_single(
            candidate="test",
            reference="ref",
            session=mock_session,
        )
        assert result.needs_human_review is True
        assert result.accepted is False
        assert "dual_inconsistent" in result.reason

    @pytest.mark.asyncio
    async def test_self_consistency_variance_flag(self):
        """High variance in primary runs → human review."""
        # 3 primary runs: 0.2, 0.8, 0.5 → variance = 0.06 > 0.05
        responses = [
            make_mock_aiohttp_response(make_judge_json_response(0.2)),
            make_mock_aiohttp_response(make_judge_json_response(0.8)),
            make_mock_aiohttp_response(make_judge_json_response(0.5)),
            make_mock_aiohttp_response(make_judge_json_response(0.5)),  # secondary
        ]
        mock_session = make_mock_session(post_side_effect=lambda *a, **kw: _async_cm(responses.pop(0)))
        result = await judge_single(
            candidate="test",
            reference="ref",
            session=mock_session,
        )
        assert result.needs_human_review is True
        assert "self_consistency_variance_exceeded" in result.reason

    @pytest.mark.asyncio
    async def test_self_consistency_no_variance(self):
        """Zero variance in primary runs → no flag."""
        json_resp = make_judge_json_response(0.7)
        responses = [make_mock_aiohttp_response(json_resp)] * 4  # 3 primary + 1 secondary
        mock_session = make_mock_session(post_side_effect=lambda *a, **kw: _async_cm(responses.pop(0)))
        result = await judge_single(
            candidate="test",
            reference="ref",
            session=mock_session,
        )
        assert "self_consistency" not in result.reason
        assert result.needs_human_review is False

    @pytest.mark.asyncio
    async def test_http_error_returns_zero_verdict(self):
        """HTTP error from judge model → zero verdict with error reason."""
        mock_resp = make_mock_aiohttp_response("error", status=500)
        mock_session = make_mock_session(post_side_effect=lambda *a, **kw: _async_cm(mock_resp))
        result = await judge_single(
            candidate="test",
            reference="ref",
            session=mock_session,
        )
        assert result.primary.quality_score == 0.0
        assert "http_500" in result.primary.reject_reason

    @pytest.mark.asyncio
    async def test_custom_self_consistency_runs(self):
        """Custom number of self-consistency runs."""
        json_resp = make_judge_json_response(0.7)
        # 2 primary + 1 secondary = 3 calls
        responses = [make_mock_aiohttp_response(json_resp)] * 3
        mock_session = make_mock_session(post_side_effect=lambda *a, **kw: _async_cm(responses.pop(0)))
        result = await judge_single(
            candidate="test",
            reference="ref",
            session=mock_session,
            self_consistency_runs=2,
        )
        assert result.accepted is True
        assert mock_session.post.call_count == 3  # 2 primary + 1 secondary

    @pytest.mark.asyncio
    async def test_primary_score_is_aggregate_of_runs(self):
        """When k>1, primary score is the aggregate of all k runs."""
        # 3 runs with scores 0.6, 0.7, 0.8
        responses = [
            make_mock_aiohttp_response(make_judge_json_response(0.6)),
            make_mock_aiohttp_response(make_judge_json_response(0.7)),
            make_mock_aiohttp_response(make_judge_json_response(0.8)),
            make_mock_aiohttp_response(make_judge_json_response(0.7)),  # secondary
        ]
        mock_session = make_mock_session(post_side_effect=lambda *a, **kw: _async_cm(responses.pop(0)))
        result = await judge_single(
            candidate="test",
            reference="ref",
            session=mock_session,
        )
        # Primary = aggregate of [0.6, 0.7, 0.8] → mean since no decay for k runs
        # recency_weighted_mean with 3 scores: weights = [0.85^2, 0.85^1, 1.0]
        expected = recency_weighted_mean([0.6, 0.7, 0.8])
        assert result.primary.quality_score == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# Test 10: judge_record_turns — multi-turn with mocked HTTP
# ===========================================================================


class TestJudgeRecordTurns:
    """Verify judge_record_turns with mocked aiohttp transport."""

    @pytest.mark.asyncio
    async def test_single_turn_pair(self):
        """Single user/assistant pair is judged."""
        json_resp = make_judge_json_response(0.8)
        mock_resp = make_mock_aiohttp_response(json_resp)
        mock_session = make_mock_session(post_side_effect=lambda *a, **kw: _async_cm(mock_resp))
        record = {
            "messages": [
                {"role": "user", "content": "What is CBT?"},
                {"role": "assistant", "content": "CBT is a therapeutic approach..."},
            ]
        }
        result = await judge_record_turns(record, session=mock_session)
        assert result.accepted is True
        assert result.primary.quality_score == pytest.approx(0.8, abs=0.001)

    @pytest.mark.asyncio
    async def test_multi_turn_per_turn_pairs(self):
        """per_turn=True: multiple user/assistant turns are judged per-turn."""
        json_resp = make_judge_json_response(0.8)
        mock_resp = make_mock_aiohttp_response(json_resp)
        mock_session = make_mock_session(post_side_effect=lambda *a, **kw: _async_cm(mock_resp))
        record = {
            "messages": [
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"},
                {"role": "user", "content": "Question 2"},
                {"role": "assistant", "content": "Answer 2"},
            ]
        }
        result = await judge_record_turns(record, session=mock_session, per_turn=True)
        assert result.accepted is True
        # 2 primary turn calls + 1 secondary call = 3 total
        assert mock_session.post.call_count == 3

    @pytest.mark.asyncio
    async def test_aggregate_single_call_per_model(self):
        """Default aggregate mode: one primary + one secondary call."""
        json_resp = make_judge_json_response(0.8)
        mock_resp = make_mock_aiohttp_response(json_resp)
        mock_session = make_mock_session(post_side_effect=lambda *a, **kw: _async_cm(mock_resp))
        record = {
            "messages": [
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"},
                {"role": "user", "content": "Question 2"},
                {"role": "assistant", "content": "Answer 2"},
                {"role": "user", "content": "Question 3"},
                {"role": "assistant", "content": "Answer 3"},
            ]
        }
        result = await judge_record_turns(record, session=mock_session)
        assert result.accepted is True
        assert result.primary.quality_score == pytest.approx(0.8, abs=0.001)
        # Aggregate: exactly 2 calls regardless of turn count (1 primary + 1 secondary)
        assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_aggregate_passes_joined_transcript(self):
        """Aggregate mode sends the newline-joined transcript to the primary judge."""
        captured: list[dict] = []

        def side_effect(*_args, **kw):
            captured.append(kw)
            return _async_cm(make_mock_aiohttp_response(make_judge_json_response(0.8)))

        mock_session = make_mock_session(post_side_effect=side_effect)
        record = {
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ]
        }
        result = await judge_record_turns(record, session=mock_session)
        assert result.accepted is True
        assert mock_session.post.call_count == 2
        payloads = [kw["json"] for kw in captured]
        primary_user_prompt = payloads[0]["messages"][1]["content"]
        assert "Q1\nQ2" in primary_user_prompt
        assert "A1\nA2" in primary_user_prompt

    @pytest.mark.asyncio
    async def test_empty_record(self):
        """Record with no messages → human review."""
        mock_session = make_mock_session(post_side_effect=lambda *a, **kw: _async_cm(MagicMock()))
        result = await judge_record_turns({"messages": []}, session=mock_session)
        assert result.accepted is False
        assert result.needs_human_review is True
        assert result.reason == "record_has_no_turns"

    @pytest.mark.asyncio
    async def test_single_message_no_pair(self):
        """Record with only one message → human review."""
        mock_session = make_mock_session(post_side_effect=lambda *a, **kw: _async_cm(MagicMock()))
        record = {"messages": [{"role": "user", "content": "Hello?"}]}
        result = await judge_record_turns(record, session=mock_session)
        assert result.accepted is False
        assert result.needs_human_review is True
        assert result.reason == "record_has_no_turns"

    @pytest.mark.asyncio
    async def test_no_user_assistant_pairs(self):
        """Messages that don't form user/assistant pairs → human review."""
        mock_session = make_mock_session(post_side_effect=lambda *a, **kw: _async_cm(MagicMock()))
        record = {
            "messages": [
                {"role": "system", "content": "You are a therapist"},
                {"role": "assistant", "content": "Hello"},
            ]
        }
        result = await judge_record_turns(record, session=mock_session)
        assert result.accepted is False
        assert result.needs_human_review is True
        assert result.reason == "no_user_assistant_pairs"

    @pytest.mark.asyncio
    async def test_missing_messages_key(self):
        """Record without 'messages' key → human review."""
        mock_session = make_mock_session(post_side_effect=lambda *a, **kw: _async_cm(MagicMock()))
        result = await judge_record_turns({}, session=mock_session)
        assert result.accepted is False
        assert result.needs_human_review is True

    @pytest.mark.asyncio
    async def test_turn_scores_aggregated_with_decay(self):
        """Multi-turn scores are aggregated with recency decay."""
        # Turn 1: 0.3, Turn 2: 0.8
        responses = [
            make_mock_aiohttp_response(make_judge_json_response(0.3)),
            make_mock_aiohttp_response(make_judge_json_response(0.8)),
            make_mock_aiohttp_response(make_judge_json_response(0.8)),  # secondary sees joined
        ]
        mock_session = make_mock_session(post_side_effect=lambda *a, **kw: _async_cm(responses.pop(0)))
        record = {
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ]
        }
        result = await judge_record_turns(record, session=mock_session, per_turn=True)
        expected = recency_weighted_mean([0.3, 0.8])
        assert result.primary.quality_score == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# Test 11: Constants and module-level configuration
# ===========================================================================


class TestConstants:
    """Verify module constants match blueprint B.3 specifications."""

    def test_recency_decay_value(self):
        """RECENCY_DECAY = 0.85 per B.3.3."""
        assert RECENCY_DECAY == 0.85

    def test_accept_threshold(self):
        """ACCEPT_THRESHOLD = 0.60 per B.3."""
        assert ACCEPT_THRESHOLD == 0.60

    def test_self_consistency_variance_max(self):
        """SELF_CONSISTENCY_VARIANCE_MAX = 0.05 per B.3.4."""
        assert SELF_CONSISTENCY_VARIANCE_MAX == 0.05

    def test_dual_consistency_diff_max(self):
        """DUAL_CONSISTENCY_DIFF_MAX = 0.15 per B.3.2."""
        assert DUAL_CONSISTENCY_DIFF_MAX == 0.15

    def test_calib_pearson_min(self):
        """CALIB_PEARSON_MIN = 0.80 per B.3.4."""
        assert CALIB_PEARSON_MIN == 0.80

    def test_calib_kappa_min(self):
        """CALIB_KAPPA_MIN = 0.65 per B.3.4."""
        assert CALIB_KAPPA_MIN == 0.65

    def test_dimensions(self):
        """DIMENSIONS = 5 per B.3.1."""
        assert DIMENSIONS == ("relevance", "accuracy", "helpfulness", "style", "safety")
        assert len(DIMENSIONS) == 5


# ---------------------------------------------------------------------------
# Test 12: Golden calibration file integration
# ===========================================================================


class TestGoldenCalibration:
    """Verify calibration against the real golden file (if present)."""

    def test_golden_file_exists(self):
        """Golden calibration file exists with 90 samples."""
        golden_path = Path(__file__).resolve().parent.parent / "data" / "golden_vera_mh_v1.jsonl"
        if not golden_path.exists():
            pytest.skip("Golden file not found")
        count = 0
        with golden_path.open() as f:
            for line in f:
                if line.strip():
                    count += 1
        assert count == 90

    def test_golden_file_format(self):
        """Golden samples have required fields."""
        golden_path = Path(__file__).resolve().parent.parent / "data" / "golden_vera_mh_v1.jsonl"
        if not golden_path.exists():
            pytest.skip("Golden file not found")
        with golden_path.open() as f:
            first_line = f.readline().strip()
            sample = json.loads(first_line)
            assert "id" in sample
            assert "conversation" in sample
            assert "human_scores" in sample
            assert isinstance(sample["conversation"], list)
            assert all(d in sample["human_scores"] for d in DIMENSIONS)

    def test_evaluate_calibration_with_golden_file(self):
        """evaluate_calibration works with golden file data."""
        golden_path = Path(__file__).resolve().parent.parent / "data" / "golden_vera_mh_v1.jsonl"
        if not golden_path.exists():
            pytest.skip("Golden file not found")
        golden_scores: list[float] = []
        with golden_path.open() as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    hs = sample["human_scores"]
                    # Compute weighted mean from human scores
                    golden_scores.append(sum(hs.values()) / len(hs))
        # Simulate judge scores = golden (perfect calibration)
        judge_scores = golden_scores[:]
        report = evaluate_calibration(judge_scores, golden_scores)
        assert report.passes is True
        assert report.pearson_r >= CALIB_PEARSON_MIN
        assert report.cohen_kappa >= CALIB_KAPPA_MIN
        assert report.sample_count == 90
