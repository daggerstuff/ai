"""Tests for the clinical accuracy validator."""

import pytest

from ai.tools.utilities.pipelines.clinical_accuracy_validator import ClinicalAccuracyValidator


@pytest.fixture
def validator() -> ClinicalAccuracyValidator:
    return ClinicalAccuracyValidator()


def test_a_grounded_answer_is_accepted(validator: ClinicalAccuracyValidator) -> None:
    """Text with clinical vocabulary passes the accuracy check."""
    result = validator.process({"text": "The patient reports anxiety", "diagnosis": "GAD"})

    assert result.is_accurate is True
    assert 0.0 <= result.score <= 1.0


def test_the_result_reports_its_issues(validator: ClinicalAccuracyValidator) -> None:
    """Whatever the verdict, the reasons travel with it."""
    result = validator.process({"text": "The patient reports anxiety", "diagnosis": "GAD"})

    assert isinstance(result.issues, list)


def test_the_score_is_bounded_for_bare_text(validator: ClinicalAccuracyValidator) -> None:
    """A payload without clinical grounding still scores inside the range."""
    result = validator.process({"text": "hello"})

    assert 0.0 <= result.score <= 1.0
    assert isinstance(result.is_accurate, bool)
