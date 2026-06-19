"""Tests for the English clinical content analyzer."""

from __future__ import annotations

import re

import pytest

try:
    from hypothesis import given, settings, strategies as st
except ImportError:
    given = None
    settings = None
    st = None

from training.clinical_safety_checker import ClinicalContentAnalyzer


def test_version_format():
    assert re.match(r"^\d+\.\d+\.\d+$", ClinicalContentAnalyzer.VERSION)


def test_empty_string_returns_false():
    assert not ClinicalContentAnalyzer.contains_crisis_keywords("")


def test_whitespace_only_returns_false():
    assert not ClinicalContentAnalyzer.contains_crisis_keywords("   \t\n  ")


def test_none_returns_false():
    assert not ClinicalContentAnalyzer.contains_crisis_keywords(None)


def test_non_string_returns_false():
    assert not ClinicalContentAnalyzer.contains_crisis_keywords(123)
    assert not ClinicalContentAnalyzer.contains_crisis_keywords(["text"])
    assert not ClinicalContentAnalyzer.contains_crisis_keywords({"text": "value"})


CRISIS_CASES = (
    "I want to die tonight.",
    "I want to kill myself",
    "end my life is the only solution",
    "no reason to live anymore",
    "suicide is the best option",
    "I've made my final decision to end it all",
    "I've already decided to end my life",
)

NEGATED_CASES = (
    "I don't want to die, I'm just feeling sad.",
    "No plans to kill myself, I'm okay.",
    "I don't want to end my life.",
    "No reason to worry about suicide.",
    "I want to live, I just need support.",
    "I am NOT suicidal.",
)

SAFE_CASES = (
    "I had a wonderful walk in the park today.",
    "The training session was very productive.",
    "I learned a new technique for therapy.",
    "My client made great progress today.",
    "The weather is lovely this morning.",
    "I am grateful for the support I have.",
)


@pytest.mark.parametrize("text", CRISIS_CASES)
def test_crisis_keywords_detected(text: str):
    assert ClinicalContentAnalyzer.contains_crisis_keywords(text)


@pytest.mark.parametrize("text", NEGATED_CASES)
def test_negated_crisis_not_detected(text: str):
    assert not ClinicalContentAnalyzer.contains_crisis_keywords(text)


@pytest.mark.parametrize("text", SAFE_CASES)
def test_safe_text_not_detected(text: str):
    assert not ClinicalContentAnalyzer.contains_crisis_keywords(text)


if st is not None:

    @given(st.sampled_from(CRISIS_CASES))
    def test_hypothesis_crisis_detected(sample: str):
        assert ClinicalContentAnalyzer.contains_crisis_keywords(sample)

    @given(st.sampled_from(NEGATED_CASES))
    def test_hypothesis_negated_not_detected(sample: str):
        assert not ClinicalContentAnalyzer.contains_crisis_keywords(sample)

    @given(st.text(min_size=1, max_size=1000))
    @settings(max_examples=1000)
    def test_all_inputs_return_bool(text: str):
        """Every non-None string returns a bool, never raises. With 1000
        examples this consistently hits the implicit return False branch
        (line 85) for safe text."""
        result = ClinicalContentAnalyzer.contains_crisis_keywords(text)
        assert isinstance(result, bool)

    def test_implicit_false_return_on_safe_text():
        """Explicitly exercise the implicit 'return False' at line 85: safe text
        that matches no crisis patterns must return False."""
        safe_long_text = (
            "The cognitive behavioral therapy session went well today. "
            "The client practiced thought challenging and identified several "
            "automatic negative thoughts. We worked on evidence evaluation "
            "and developing alternative perspectives. The client showed good "
            "insight and commitment to the treatment plan. Next week we will "
            "continue working on behavioral activation techniques."
        )
        result = ClinicalContentAnalyzer.contains_crisis_keywords(safe_long_text)
        assert result is False, "safe text should not trigger crisis detection"

    def test_short_random_not_detected():
        """Short, safe strings that contain no crisis-related words must return
        False, exercising the implicit return on line 85."""
        for text in ("abc", "xyz", "qwerty", "hello world", "testing one two three"):
            assert not ClinicalContentAnalyzer.contains_crisis_keywords(text)


if st is None:

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_crisis_detected():
        raise AssertionError("Skipped when hypothesis is unavailable")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_negated_not_detected():
        raise AssertionError("Skipped when hypothesis is unavailable")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_all_inputs_return_bool():
        raise AssertionError("Skipped when hypothesis is unavailable")
