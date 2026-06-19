#!/usr/bin/env python3
"""Property tests for Phase 27 reward score."""
from __future__ import annotations

import math

import numpy as np
import pytest

from training.reward_score import RewardScoreCalculator, compute_reward, filter_by_threshold


class TestRewardScoreCalculator:
    def test_reward_range(self) -> None:
        calculator = RewardScoreCalculator()
        assert -1.0 <= calculator.reward("ok", 0.5, 0.5) <= 1.0

    def test_unsafe_response_excluded(self) -> None:
        calculator = RewardScoreCalculator(safety_weight=1.0)
        assert calculator.is_unsafe("")
        assert calculator._safety_score("") == 0.0

    def test_safe_response_safety_score_is_1(self) -> None:
        calculator = RewardScoreCalculator(safety_weight=1.0)
        assert not calculator.is_unsafe("safe response")
        assert calculator._safety_score("safe response") == 1.0

    def test_unsafe_safety_score_crisis_zero(self) -> None:
        """When safety_score == 0, crisis_score should be 0 (not 1.0)."""
        calc = RewardScoreCalculator(crisis_weight=0.4, empathy_weight=0.0)
        assert calc.reward("", 0.5, 0.5) == 0.0

    def test_safety_weighted_zero_when_safety_zero(self) -> None:
        calc = RewardScoreCalculator()
        assert calc._safety_weighted(1.0, 0.4, 0.0) == 0.0

    def test_safety_weighted_zero_when_weight_zero(self) -> None:
        calc = RewardScoreCalculator()
        assert calc._safety_weighted(1.0, 0.0, 1.0) == 0.0

    def test_safety_weighted_positive(self) -> None:
        calc = RewardScoreCalculator(crisis_weight=1.0, empathy_weight=0.0, clinical_weight=0.0, safety_weight=0.0)
        result = calc._safety_weighted(0.5, 1.0, 1.0)
        assert result == pytest.approx(0.5)
        assert 0.0 <= result <= 1.0

    def test_weighted_sum(self) -> None:
        calc = RewardScoreCalculator(crisis_weight=0.3, empathy_weight=0.7)
        assert calc.weighted_sum() == pytest.approx(1.0)


class TestComputeReward:
    def test_returns_float(self) -> None:
        result = compute_reward(0.8, 0.6)
        assert isinstance(result, float)

    def test_zero_weights_return_zero(self) -> None:
        result = compute_reward(0.8, 0.6, crisis_weight=0.0, empathy_weight=0.0)
        assert result == pytest.approx(0.0)

    def test_zero_score_zero_weight(self) -> None:
        result = compute_reward(0.0, 0.0)
        assert result == 0.0

    def test_directional_sign(self) -> None:
        """Positive scores yield positive reward; negative scores negative."""
        pos = compute_reward(1.0, 1.0)
        assert pos >= 0
        neg = compute_reward(-1.0, -1.0)
        assert neg <= 0

    def test_reward_bounded(self) -> None:
        for w1 in (0.3, 0.5):
            for w2 in (0.7, 0.5):
                r = compute_reward(1.0, 1.0, crisis_weight=w1, empathy_weight=w2)
                assert -1.0 <= r <= 1.0, f"w1={w1} w2={w2} r={r}"

    def test_reward_extreme_inputs(self) -> None:
        # Standard inputs never crash or raise
        compute_reward(float("inf"), 0.5)
        compute_reward(-float("inf"), 0.5)
        # Test with edge-case weights
        r = compute_reward(1.0, 1.0, crisis_weight=0.0, empathy_weight=0.0,
                           clinical_weight=0.0, safety_weight=0.0)
        assert r == 0.0

    def test_clinical_and_safety_weights(self) -> None:
        r = compute_reward(1.0, 1.0, clinical_score=0.8, safety_score=0.9,
                           crisis_weight=0.3, empathy_weight=0.3,
                           clinical_weight=0.2, safety_weight=0.2)
        assert -1.0 <= r <= 1.0


class TestFilterByThreshold:
    def test_filters_below_threshold(self) -> None:
        result = filter_by_threshold(
            ["p1", "p2"],
            ["good response", "bad response"],
            threshold=0.8,
        )
        assert isinstance(result, list)

    def test_empty_inputs_returns_empty(self) -> None:
        result = filter_by_threshold([], [], threshold=0.5)
        assert result == []

    def test_single_element(self) -> None:
        result = filter_by_threshold(["test"], ["response"], threshold=0.0)
        assert len(result) == 1
        assert result[0]["prompt"] == "test"
        assert "composite_score" in result[0]


if __name__ == "__main__":
    pytest.main([__file__])
