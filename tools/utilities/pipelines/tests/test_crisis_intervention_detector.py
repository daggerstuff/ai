"""Tests for the crisis intervention detector."""

import pytest

from ai.tools.utilities.pipelines.crisis_intervention_detector import CrisisInterventionDetector


@pytest.fixture
def detector() -> CrisisInterventionDetector:
    return CrisisInterventionDetector()


def test_ordinary_text_is_not_flagged(detector: CrisisInterventionDetector) -> None:
    """A harmless sentence yields no crisis type and no escalation."""
    result = detector.process("I had a lovely walk today")

    assert result.flagged is False
    assert result.crisis_type == "none"
    assert result.matches == []
    assert result.score == 0.0


def test_suicidal_language_is_flagged_as_critical(detector: CrisisInterventionDetector) -> None:
    """Explicit self-harm language is flagged with escalation recommendations."""
    result = detector.process("I want to kill myself")

    assert result.flagged is True
    assert result.severity == "critical"
    assert "suicidal_ideation" in result.matches
    assert result.recommendations


def test_the_score_is_bounded(detector: CrisisInterventionDetector) -> None:
    """Scores stay inside the documented range whatever the input."""
    for text in ("I had a lovely walk today", "I want to kill myself", ""):
        score = detector.process(text).score
        assert 0.0 <= score <= 1.0


def test_mapping_payloads_are_read_like_strings(detector: CrisisInterventionDetector) -> None:
    """A dict payload is flattened before detection."""
    result = detector.process({"text": "I want to kill myself"})

    assert result.flagged is True


def test_toxicity_check_reports_its_own_shape(detector: CrisisInterventionDetector) -> None:
    """The toxicity detector returns categories, matches and a score."""
    result = detector.detect_toxicity("you are a worthless idiot")

    assert isinstance(result.categories, list)
    assert isinstance(result.matches, list)
    assert 0.0 <= result.score <= 1.0


def test_audit_event_is_serialisable(detector: CrisisInterventionDetector) -> None:
    """The audit record for a detection is a plain mapping."""
    record = detector.audit_event(detector.process("I want to kill myself"))

    assert isinstance(record, dict)
    assert record
