"""Tests for the constrained-choice decision probe.

The probe's job is to read a behaviour distribution off one token. These tests
pin the two things that can silently invalidate that read-out: which label the
correct behaviour lands on, and how confidence is reported.
"""

from __future__ import annotations

from training.constrained_choice import ChoiceReadout
from training.probe_scenario_choices import (
    MENUS,
    REFUSAL_RESIDUAL,
    CaseResult,
    rotate_menu,
)

REFUSALS_IN_FIXTURE = 2


def _readout(residual: float) -> ChoiceReadout:
    mass = 1.0 - residual
    return ChoiceReadout(
        masses={"A": mass},
        probs={"A": 1.0},
        residual=residual,
        unscoreable=False,
    )


class TestRotateMenu:
    def test_offset_zero_is_the_identity(self) -> None:
        menu = MENUS["research"]
        rotated = rotate_menu(menu, 0)
        assert rotated.labels == menu.labels
        assert rotated.correct_label == menu.correct_label
        assert [item.summary for item in rotated.behaviours] == [item.summary for item in menu.behaviours]

    def test_rotation_moves_the_correct_label(self) -> None:
        menu = MENUS["research"]
        assert menu.correct_label == "A"
        assert rotate_menu(menu, 1).correct_label == "C"
        assert rotate_menu(menu, 2).correct_label == "B"

    def test_rotation_wraps_on_the_menu_size(self) -> None:
        menu = MENUS["research"]
        assert rotate_menu(menu, 3).correct_label == menu.correct_label

    def test_correct_behaviour_keeps_its_summary_after_rotation(self) -> None:
        menu = MENUS["research"]
        expected = next(item.summary for item in menu.behaviours if item.label == menu.correct_label)
        for offset in range(len(menu.behaviours)):
            rotated = rotate_menu(menu, offset)
            landed = next(item.summary for item in rotated.behaviours if item.label == rotated.correct_label)
            assert landed == expected

    def test_every_menu_keeps_distinct_labels(self) -> None:
        for menu in MENUS.values():
            rotated = rotate_menu(menu, 1)
            assert sorted(rotated.labels) == sorted(menu.labels)
            assert rotated.correct_label in rotated.labels


class TestCaseResultRefusals:
    def test_counts_readouts_that_spent_mass_outside_the_menu(self) -> None:
        case = CaseResult(
            case_id="case",
            scenario="support",
            correct_label="A",
            readouts=[_readout(0.01), _readout(REFUSAL_RESIDUAL), _readout(0.9)],
        )
        assert case.refusals() == REFUSALS_IN_FIXTURE

    def test_reports_zero_when_every_readout_answered(self) -> None:
        case = CaseResult(
            case_id="case",
            scenario="support",
            correct_label="A",
            readouts=[_readout(0.02), _readout(0.1)],
        )
        assert case.refusals() == 0
