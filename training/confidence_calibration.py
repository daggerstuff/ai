"""Confidence calibration for closed-set decision probes.

A 51% call and a 90% call may both land on the right label, but they should not
be treated the same. The Fireworks Glimmer recipe fits a separate temperature
per question type on a held-out calibration set and rescales the model's
probabilities without touching weights; the tuned model's >=80% calls came true
86.8% of the time.

This module holds that post-step: reliability bins, expected/maximum
calibration error, Brier score, and a one-parameter temperature scaler fitted by
direct NLL minimisation (no optimiser dependency, fully deterministic).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_N_BINS = 10
PROBABILITY_EPSILON = 1e-12
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0
GRID_STEPS = 400
REFINE_STEPS = 80
REFINE_SHRINK = 0.6180339887498949
BINARY_LABELS = 2


def _clip_probability(value: float) -> float:
    return min(1.0 - PROBABILITY_EPSILON, max(PROBABILITY_EPSILON, value))


def _logit(probability: float) -> float:
    probability = _clip_probability(probability)
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


@dataclass(frozen=True)
class ReliabilityBin:
    """One bucket of predicted confidence versus observed accuracy."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Signed calibration gap; positive means underconfident."""
        return self.accuracy - self.mean_confidence


def _validate(confidences: Sequence[float], outcomes: Sequence[int | bool]) -> None:
    if len(confidences) != len(outcomes):
        raise ValueError("confidences and outcomes must have the same length")
    if not confidences:
        raise ValueError("need at least one observation")
    for value in confidences:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"confidence out of range: {value}")
    for value in outcomes:
        if int(value) not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1: {value}")


def reliability_bins(
    confidences: Sequence[float],
    outcomes: Sequence[int | bool],
    n_bins: int = DEFAULT_N_BINS,
) -> list[ReliabilityBin]:
    """Bucket observations by predicted confidence and report observed accuracy."""
    _validate(confidences, outcomes)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    counts = [0] * n_bins
    confidence_sums = [0.0] * n_bins
    correct_sums = [0] * n_bins
    for confidence, outcome in zip(confidences, outcomes, strict=True):
        index = min(n_bins - 1, int(float(confidence) * n_bins))
        counts[index] += 1
        confidence_sums[index] += float(confidence)
        correct_sums[index] += int(bool(outcome))

    bins: list[ReliabilityBin] = []
    for index, count in enumerate(counts):
        if count == 0:
            continue
        bins.append(
            ReliabilityBin(
                lower=index / n_bins,
                upper=(index + 1) / n_bins,
                count=count,
                mean_confidence=confidence_sums[index] / count,
                accuracy=correct_sums[index] / count,
            )
        )
    return bins


def expected_calibration_error(
    confidences: Sequence[float],
    outcomes: Sequence[int | bool],
    n_bins: int = DEFAULT_N_BINS,
) -> float:
    """Return the count-weighted mean gap between confidence and accuracy."""
    bins = reliability_bins(confidences, outcomes, n_bins)
    total = sum(item.count for item in bins)
    return sum(item.count * abs(item.gap) for item in bins) / total


def maximum_calibration_error(
    confidences: Sequence[float],
    outcomes: Sequence[int | bool],
    n_bins: int = DEFAULT_N_BINS,
) -> float:
    """Return the worst per-bin gap between confidence and accuracy."""
    bins = reliability_bins(confidences, outcomes, n_bins)
    return max(abs(item.gap) for item in bins)


def brier_score(confidences: Sequence[float], outcomes: Sequence[int | bool]) -> float:
    """Return the mean squared error of the confidence estimates."""
    _validate(confidences, outcomes)
    total = sum(
        (float(confidence) - float(bool(outcome))) ** 2
        for confidence, outcome in zip(confidences, outcomes, strict=True)
    )
    return total / len(confidences)


def negative_log_likelihood(confidences: Sequence[float], outcomes: Sequence[int | bool], temperature: float) -> float:
    """Return mean binary NLL after rescaling confidence by ``temperature``."""
    _validate(confidences, outcomes)
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    total = 0.0
    for confidence, outcome in zip(confidences, outcomes, strict=True):
        scaled = _sigmoid(_logit(float(confidence)) / temperature)
        target = float(bool(outcome))
        total -= target * math.log(_clip_probability(scaled))
        total -= (1.0 - target) * math.log(_clip_probability(1.0 - scaled))
    return total / len(confidences)


def _search_temperature(confidences: Sequence[float], outcomes: Sequence[int | bool]) -> float:
    log_min = math.log(MIN_TEMPERATURE)
    log_max = math.log(MAX_TEMPERATURE)
    step = (log_max - log_min) / (GRID_STEPS - 1)

    best_log, best_nll = log_min, math.inf
    for index in range(GRID_STEPS):
        candidate = log_min + step * index
        value = negative_log_likelihood(confidences, outcomes, math.exp(candidate))
        if value < best_nll:
            best_log, best_nll = candidate, value

    low, high = best_log - step, best_log + step
    for _ in range(REFINE_STEPS):
        width = (high - low) * REFINE_SHRINK
        left = high - width
        right = low + width
        if negative_log_likelihood(confidences, outcomes, math.exp(left)) <= (
            negative_log_likelihood(confidences, outcomes, math.exp(right))
        ):
            high = right
        else:
            low = left
    return math.exp((low + high) / 2.0)


@dataclass(frozen=True)
class TemperatureScaler:
    """One-parameter confidence rescaling fitted on a held-out set."""

    temperature: float
    samples: int

    @classmethod
    def fit(cls, confidences: Sequence[float], outcomes: Sequence[int | bool]) -> TemperatureScaler:
        """Fit the temperature that minimises NLL on ``(confidence, outcome)``."""
        _validate(confidences, outcomes)
        return cls(temperature=_search_temperature(confidences, outcomes), samples=len(confidences))

    def transform(self, confidence: float) -> float:
        """Return the rescaled confidence for a single prediction."""
        return _sigmoid(_logit(float(confidence)) / self.temperature)

    def transform_all(self, confidences: Sequence[float]) -> list[float]:
        """Return rescaled confidences for a batch."""
        return [self.transform(value) for value in confidences]
