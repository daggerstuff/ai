"""Tests for confidence calibration."""

from __future__ import annotations

import math

import pytest

from training.confidence_calibration import (
    TemperatureScaler,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_bins,
)


class TestReliabilityBins:
    def test_buckets_by_predicted_confidence(self) -> None:
        bins = reliability_bins([0.05, 0.15, 0.95, 0.99], [0, 0, 1, 1], n_bins=10)
        # 0.05 and 0.15 fall in the first two bins; 0.95 and 0.99 share the last.
        assert [item.count for item in bins] == [1, 1, 2]
        assert bins[0].lower == pytest.approx(0.0)
        assert bins[-1].upper == pytest.approx(1.0)

    def test_confidence_of_one_lands_in_the_last_bin(self) -> None:
        bins = reliability_bins([1.0], [1], n_bins=10)
        assert bins[-1].lower == pytest.approx(0.9)
        assert bins[-1].accuracy == pytest.approx(1.0)

    def test_reports_signed_gap(self) -> None:
        bins = reliability_bins([0.9, 0.9, 0.9, 0.9], [1, 0, 0, 0], n_bins=10)
        assert bins[0].gap == pytest.approx(0.25 - 0.9)

    def test_empty_bins_are_dropped(self) -> None:
        bins = reliability_bins([0.5], [1], n_bins=10)
        assert len(bins) == 1

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            reliability_bins([0.5, 0.6], [1])

    def test_rejects_out_of_range_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence out of range"):
            reliability_bins([1.2], [1])

    def test_rejects_non_binary_outcome(self) -> None:
        with pytest.raises(ValueError, match="outcome must be 0 or 1"):
            reliability_bins([0.5], [2])

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="at least one observation"):
            reliability_bins([], [])

    def test_rejects_non_positive_bin_count(self) -> None:
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            reliability_bins([0.5], [1], n_bins=0)


class TestCalibrationErrors:
    def test_perfectly_calibrated_predictions_have_zero_error(self) -> None:
        confidences = [0.0, 0.0, 1.0, 1.0]
        outcomes = [0, 0, 1, 1]
        assert expected_calibration_error(confidences, outcomes) == pytest.approx(0.0, abs=1e-9)
        assert maximum_calibration_error(confidences, outcomes) == pytest.approx(0.0, abs=1e-9)

    def test_overconfidence_shows_up_as_positive_error(self) -> None:
        confidences = [0.9, 0.9, 0.9, 0.9]
        outcomes = [1, 0, 0, 0]
        assert expected_calibration_error(confidences, outcomes) == pytest.approx(0.65)

    def test_error_is_count_weighted_across_bins(self) -> None:
        confidences = [0.1, 0.1, 0.1, 0.9]
        outcomes = [1, 1, 1, 0]
        bins = reliability_bins(confidences, outcomes, n_bins=10)
        weighted = sum(item.count * abs(item.gap) for item in bins) / len(confidences)
        assert expected_calibration_error(confidences, outcomes) == pytest.approx(weighted)

    def test_brier_score_of_a_confident_mistake_is_one(self) -> None:
        assert brier_score([1.0], [0]) == pytest.approx(1.0)

    def test_brier_score_of_a_hedged_call(self) -> None:
        assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


class TestNegativeLogLikelihood:
    def test_temperature_of_one_preserves_confidence(self) -> None:
        value = negative_log_likelihood([0.8], [1], temperature=1.0)
        assert value == pytest.approx(-math.log(0.8))

    def test_high_temperature_softens_confident_predictions(self) -> None:
        softened = negative_log_likelihood([0.99], [0], temperature=5.0)
        raw = negative_log_likelihood([0.99], [0], temperature=1.0)
        assert softened < raw

    def test_low_temperature_sharpens_confident_predictions(self) -> None:
        sharpened = negative_log_likelihood([0.99], [0], temperature=0.5)
        raw = negative_log_likelihood([0.99], [0], temperature=1.0)
        assert sharpened > raw

    def test_rejects_non_positive_temperature(self) -> None:
        with pytest.raises(ValueError, match="temperature must be > 0"):
            negative_log_likelihood([0.5], [1], temperature=0.0)


FIT_SAMPLE_COUNT = 40
OVERCONFIDENT_TEMPERATURE_RANGE = (2.0, 5.0)
CALIBRATED_TEMPERATURE_RANGE = (0.7, 1.4)


class TestTemperatureScaler:
    def test_recovers_a_temperature_that_undoes_overconfidence(self) -> None:
        # These are sigmoid(logit(0.9) / 3) -> about 0.72, observed at 72%.
        scaler = TemperatureScaler.fit([0.9] * FIT_SAMPLE_COUNT, [1] * 29 + [0] * 11)
        low, high = OVERCONFIDENT_TEMPERATURE_RANGE
        assert low < scaler.temperature < high
        assert scaler.samples == FIT_SAMPLE_COUNT

    def test_keeps_a_temperature_near_one_when_already_calibrated(self) -> None:
        scaler = TemperatureScaler.fit([0.7] * FIT_SAMPLE_COUNT, [1] * 28 + [0] * 12)
        low, high = CALIBRATED_TEMPERATURE_RANGE
        assert low < scaler.temperature < high

    def test_fitted_temperature_lowers_nll(self) -> None:
        confidences = [0.95] * 30 + [0.6] * 30
        outcomes = [1] * 12 + [0] * 18 + [1] * 12 + [0] * 18
        scaler = TemperatureScaler.fit(confidences, outcomes)
        fitted = negative_log_likelihood(confidences, outcomes, scaler.temperature)
        baseline = negative_log_likelihood(confidences, outcomes, 1.0)
        assert fitted <= baseline

    def test_transform_is_monotonic(self) -> None:
        scaler = TemperatureScaler.fit([0.9] * 20, [1] * 10 + [0] * 10)
        values = scaler.transform_all([0.1, 0.4, 0.8, 0.95])
        assert values == sorted(values)

    def test_transform_bounds_stay_in_probability_range(self) -> None:
        scaler = TemperatureScaler(temperature=3.0, samples=10)
        assert 0.0 < scaler.transform(0.0) < 1.0
        assert 0.0 < scaler.transform(1.0) < 1.0

    def test_rejects_empty_fit(self) -> None:
        with pytest.raises(ValueError, match="at least one observation"):
            TemperatureScaler.fit([], [])
