"""Tests for the safety and ethics validator."""

import pytest

from ai.tools.utilities.pipelines.safety_ethics_validator import SafetyEthicsValidator


@pytest.fixture
def validator() -> SafetyEthicsValidator:
    return SafetyEthicsValidator()


def test_clean_payload_passes_with_a_full_score(validator: SafetyEthicsValidator) -> None:
    """Ordinary text raises no violation and keeps the score at one."""
    result = validator.process({"text": "I had a lovely walk today"})

    assert result.is_safe is True
    assert result.score == 1.0
    assert result.violations == []
    assert result.suggestions == []


def test_self_harm_language_is_flagged_with_an_escalation(validator: SafetyEthicsValidator) -> None:
    """Self-harm wording blocks the payload and routes it to the crisis protocol."""
    result = validator.process({"text": "I want to kill myself"})

    assert result.is_safe is False
    assert "self_harm" in result.violations
    assert "Escalate to safety and crisis protocol" in result.suggestions
    assert result.score < 1.0


def test_score_falls_with_each_violated_category(validator: SafetyEthicsValidator) -> None:
    """One of four categories violated leaves three quarters of the score."""
    result = validator.process({"text": "I want to kill myself"})

    assert result.score == pytest.approx(0.75)


def test_privacy_identifiers_are_flagged_and_advised(validator: SafetyEthicsValidator) -> None:
    """Personal identifiers are a privacy violation with its own remediation."""
    result = validator.process({"text": "my ssn is on the form"})

    assert "privacy" in result.violations
    assert "Remove personal identifiers before processing" in result.suggestions


def test_message_lists_are_read_as_content(validator: SafetyEthicsValidator) -> None:
    """A chat payload is flattened from its message list before scanning."""
    result = validator.process({"messages": [{"content": "I want to kill myself"}]})

    assert "self_harm" in result.violations


def test_none_is_rejected(validator: SafetyEthicsValidator) -> None:
    """A missing payload is a programming error, not a safe payload."""
    with pytest.raises(ValueError):
        validator.process(None)


def test_unsupported_payload_type_is_rejected(validator: SafetyEthicsValidator) -> None:
    """Only strings, mappings and message lists are accepted."""
    with pytest.raises(TypeError):
        validator.process(42)
