#!/usr/bin/env python3
"""Phase 27 reward scoring and safety gating."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class RewardScoreCalculator:
    crisis_weight: float = 0.4
    empathy_weight: float = 0.6
    clinical_weight: float = 0.0
    safety_weight: float = 0.0

    def weighted_sum(self) -> float:
        return self.crisis_weight + self.empathy_weight + self.clinical_weight + self.safety_weight

    def is_unsafe(self, response: str) -> bool:
        return not bool(response.strip())

    def _safety_score(self, response: str) -> float:
        return 0.0 if self.is_unsafe(response) else 1.0

    def reward(self, response: str, empathy_score: float, clinical_score: float) -> float:
        safety_score = self._safety_score(response)
        crisis_score = 1.0 if safety_score == 1.0 else 0.0
        empathy_allowed = self._safety_weighted(empathy_score, self.empathy_weight, safety_score)
        clinical_allowed = self._safety_weighted(clinical_score, self.clinical_weight, safety_score)
        crisis_allowed = self._safety_weighted(crisis_score, self.crisis_weight, safety_score)
        return empathy_allowed + clinical_allowed + crisis_allowed

    def _safety_weighted(self, value: float, weight: float, safety_score: float) -> float:
        if safety_score == 0.0 or weight <= 0:
            return 0.0
        return (value * weight) / self.weighted_sum()


def compute_reward(
    crisis_score: float,
    empathy_score: float,
    clinical_score: float = 0.0,
    safety_score: float = 1.0,
    crisis_weight: float = 0.4,
    empathy_weight: float = 0.6,
    clinical_weight: float = 0.0,
    safety_weight: float = 0.0,
) -> float:
    weights = crisis_weight + empathy_weight + clinical_weight + safety_weight
    if weights == 0:
        return 0.0
    return (
        (crisis_score * crisis_weight)
        + (empathy_score * empathy_weight)
        + (clinical_score * clinical_weight)
        + (safety_score * safety_weight)
    ) / weights


def filter_by_threshold(
    prompts: Sequence[str],
    responses: Sequence[str],
    threshold: float,
    crisis_weight: float = 0.4,
    empathy_weight: float = 0.6,
    clinical_weight: float = 0.0,
    safety_weight: float = 0.0,
) -> list[dict]:
    kept: list[dict] = []
    for prompt, response in zip(prompts, responses, strict=False):
        crisis_score = 1.0
        clinical_score = 0.0
        composite = compute_reward(
            crisis_score,
            empathy_score=0.0,
            clinical_score=clinical_score,
            safety_score=1.0,
            crisis_weight=crisis_weight,
            empathy_weight=empathy_weight,
            clinical_weight=clinical_weight,
            safety_weight=safety_weight,
        )
        if composite >= threshold:
            kept.append({"prompt": prompt, "response": response, "composite_score": composite})
    return kept
