#!/usr/bin/env python3
"""Property tests for Phase 27 reward score."""
from __future__ import annotations

import numpy as np
import pytest

from training.reward_score import RewardScoreCalculator, compute_reward


class TestRewardScoreCalculator:
    def test_reward_range(self) -> None:
        calculator = RewardScoreCalculator()
        assert -1.0 <= calculator.reward("ok", 0.5, 0.5) <= 1.0

    def test_unsafe_response_excluded(self) -> None:
        calculator = RewardScoreCalculator(safety_weight=1.0)
        assert calculator.is_unsafe("")
        assert calculator._safety_score("") == 0.0


class TestComputeReward:
    def test_returns_float(self) -> None:
        result = compute_reward(0.8, 0.6)
        assert isinstance(result, float)

    def test_zero_weights_return_zero(self) -> None:
        result = compute_reward(0.8, 0.6, crisis_weight=0.0, empathy_weight=0.0)
        assert result == pytest.approx(0.0)
