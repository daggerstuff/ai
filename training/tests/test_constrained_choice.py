"""Tests for the closed-set decision read-out."""

from __future__ import annotations

import io
import math
from unittest.mock import patch

import pytest

from training.constrained_choice import (
    ChoiceProbe,
    ChoiceReadout,
    normalize_label_token,
    read_choice_distribution,
)


def _entries(**logprobs: float) -> list[dict[str, object]]:
    return [{"token": token, "logprob": value} for token, value in logprobs.items()]


class TestNormalizeLabelToken:
    def test_folds_case_whitespace_and_punctuation(self) -> None:
        cases: list[tuple[str, str]] = [
            ("A", "a"),
            (" A", "a"),
            ("a", "a"),
            ("A)", "a"),
            ("(b)", "b"),
            ("**C**", "c"),
            ("D.", "d"),
            ("\n A \n", "a"),
        ]
        for raw, expected in cases:
            assert normalize_label_token(raw) == expected, raw


class TestChoiceProbe:
    def test_post_returns_json_object(self) -> None:
        probe = ChoiceProbe(model="test-model", api_key="")
        with patch(
            "training.constrained_choice.urllib.request.urlopen",
            return_value=io.BytesIO(b'{"choices": []}'),
        ):
            assert probe._post({}) == {"choices": []}

    def test_post_rejects_non_object_json(self) -> None:
        probe = ChoiceProbe(model="test-model", api_key="")
        for body in (b"[]", b"null", b'"text"', b"1", b"true"):
            with (
                patch("training.constrained_choice.urllib.request.urlopen", return_value=io.BytesIO(body)),
                pytest.raises(ValueError, match="probe response must be a JSON object"),
            ):
                probe._post({})


class TestReadChoiceDistribution:
    def test_renormalises_over_the_option_set(self) -> None:
        readout = read_choice_distribution(
            _entries(A=math.log(0.5), B=math.log(0.25), The=math.log(0.25)),
            ["A", "B"],
        )
        assert readout.probs["A"] == pytest.approx(2 / 3)
        assert readout.probs["B"] == pytest.approx(1 / 3)
        assert readout.best() == "A"
        assert readout.confidence() == pytest.approx(2 / 3)

    def test_reports_mass_outside_the_option_set_as_residual(self) -> None:
        readout = read_choice_distribution(
            _entries(A=math.log(0.5), The=math.log(0.25)),
            ["A", "B"],
        )
        assert readout.residual == pytest.approx(0.5)
        assert readout.masses["B"] == 0.0

    def test_aggregates_surface_variants_of_the_same_label(self) -> None:
        readout = read_choice_distribution(
            _entries(A=math.log(0.4), **{" A": math.log(0.3), "a)": math.log(0.1)}),
            ["A", "B"],
        )
        assert readout.masses["A"] == pytest.approx(0.8)
        assert readout.probs["A"] == pytest.approx(1.0)

    def test_without_normalisation_returns_raw_mass(self) -> None:
        readout = read_choice_distribution(
            _entries(A=math.log(0.5), B=math.log(0.25)),
            ["A", "B"],
            normalize=False,
        )
        assert readout.probs["A"] == pytest.approx(0.5)
        assert readout.probs["B"] == pytest.approx(0.25)

    def test_flags_unscoreable_when_no_label_appears(self) -> None:
        readout = read_choice_distribution(_entries(The=math.log(0.9)), ["A", "B"])
        assert readout.unscoreable is True
        assert readout.best() is None
        assert readout.confidence() == 0.0

    def test_ties_break_on_label_order_not_dict_order(self) -> None:
        readout = read_choice_distribution(
            _entries(B=math.log(0.5), A=math.log(0.5)),
            ["A", "B"],
        )
        assert readout.best() == "A"

    def test_ignores_entries_missing_fields(self) -> None:
        readout = read_choice_distribution(
            [{"token": "A"}, {"logprob": -0.1}, {"token": "B", "logprob": math.log(0.5)}],
            ["A", "B"],
        )
        assert readout.probs["B"] == pytest.approx(1.0)
        assert readout.unscoreable is False

    def test_rejects_empty_label_set(self) -> None:
        with pytest.raises(ValueError, match="labels must not be empty"):
            read_choice_distribution(_entries(A=0.0), [])

    def test_rejects_labels_that_collide_after_normalisation(self) -> None:
        with pytest.raises(ValueError, match="distinct after normalisation"):
            read_choice_distribution(_entries(A=0.0), ["A", "a"])


class TestChoiceReadout:
    def test_probability_of_unknown_label_is_zero(self) -> None:
        readout = ChoiceReadout(masses={"A": 1.0}, probs={"A": 1.0}, residual=0.0, unscoreable=False)
        assert readout.probability_of("Z") == 0.0

    def test_unscoreable_readout_has_no_winner(self) -> None:
        readout = ChoiceReadout(masses={"A": 0.0}, probs={"A": 0.0}, residual=1.0, unscoreable=True)
        assert readout.best() is None
        assert readout.confidence() == 0.0
        assert readout.raw_confidence() == 0.0

    def test_raw_confidence_keeps_the_residual(self) -> None:
        readout = read_choice_distribution(
            _entries(A=math.log(0.05), B=math.log(0.03), The=math.log(0.92)),
            ["A", "B"],
        )
        # Renormalised, A looks like a decision; raw mass shows it is a refusal.
        assert readout.confidence() == pytest.approx(0.625)
        assert readout.raw_confidence() == pytest.approx(0.05)
        assert readout.residual == pytest.approx(0.92)

    def test_raw_confidence_matches_confidence_when_nothing_is_left_over(self) -> None:
        readout = read_choice_distribution(
            _entries(A=math.log(0.7), B=math.log(0.3)),
            ["A", "B"],
        )
        assert readout.raw_confidence() == pytest.approx(readout.confidence())
