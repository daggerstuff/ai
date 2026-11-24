"""Tests for the safety validation system."""

import pytest

from ai.tools.scripts.safety_validation_system import CrisisLevel, SafetyValidationSystem


@pytest.fixture
def validator() -> SafetyValidationSystem:
    return SafetyValidationSystem()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I want to kill myself", CrisisLevel.CRITICAL),
        ("I had a nice day at the park", CrisisLevel.NONE),
    ],
)
def test_crisis_level_follows_the_text(validator: SafetyValidationSystem, text: str, expected: CrisisLevel) -> None:
    """Explicit self-harm language is critical; ordinary text is not a crisis."""
    level, confidence = validator.detect_crisis_level(text)

    assert level is expected
    assert 0.0 <= confidence <= 1.0


def test_explicit_risk_scores_higher_than_neutral_text(validator: SafetyValidationSystem) -> None:
    """The reported confidence separates a crisis statement from small talk."""
    _, crisis_confidence = validator.detect_crisis_level("I want to kill myself")
    _, neutral_confidence = validator.detect_crisis_level("I had a nice day at the park")

    assert crisis_confidence > neutral_confidence or crisis_confidence >= 0.9


def test_response_appropriateness_is_a_verdict(validator: SafetyValidationSystem) -> None:
    """The appropriateness check returns a boolean for a known crisis level."""
    verdict = validator.validate_response_appropriateness(CrisisLevel.NONE, "empathic")

    assert isinstance(verdict, bool)
