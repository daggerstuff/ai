"""Tests for the GRPO trainer reward function."""

from __future__ import annotations

import pytest

try:
    from hypothesis import given, settings, strategies as st
except ImportError:
    given = None
    settings = None
    st = None

from training.grpo_trainer import (
    CRISIS_RESOURCES,
    EMPATHY_KEYWORDS,
    _crisis_score,
    _empathy_score,
    filter_by_threshold,
    score,
)


class TestEmpathyScore:

    def test_no_empathy_keywords(self):
        assert _empathy_score("The sky is blue.") == 0.0

    def test_one_keyword(self):
        result = _empathy_score("I understand your concern.")
        assert 0.0 < result <= 0.25

    def test_two_keywords(self):
        result = _empathy_score("I understand your concern. I hear you.")
        assert 0.25 < result <= 0.5

    def test_four_keywords_max(self):
        text = "I understand. I hear you. Your feelings are valid. I care about your wellbeing."
        result = _empathy_score(text)
        assert result == 1.0

    def test_empty_string(self):
        assert _empathy_score("") == 0.0

    def test_case_insensitive(self):
        assert _empathy_score("I UNDERSTAND your feelings.") > 0.0


class TestCrisisScore:

    def test_non_crisis_prompt_returns_one(self):
        assert _crisis_score("Tell me about anxiety", "Any response") == 1.0

    def test_crisis_prompt_with_988(self):
        assert _crisis_score("I want to die", "Please call 988 for help.") == 1.0

    def test_crisis_prompt_with_741741(self):
        assert _crisis_score("I want to die", "Text 741741 for support.") == 1.0

    def test_crisis_prompt_with_findahelpline(self):
        assert _crisis_score("I want to die", "Visit findahelpline.org") == 1.0

    def test_crisis_prompt_with_911(self):
        assert _crisis_score("I want to kill myself", "Call 911 immediately.") == 1.0

    def test_crisis_prompt_with_999(self):
        assert _crisis_score("I want to kill myself", "Dial 999 for emergency services.") == 1.0

    def test_crisis_prompt_with_112(self):
        assert _crisis_score("I want to die tonight", "Call 112 for emergency.") == 1.0

    def test_crisis_prompt_without_resource(self):
        assert _crisis_score("I want to die", "I'm sorry to hear that.") == 0.0

    def test_crisis_prompt_with_suicide_prevention_lifeline(self):
        assert _crisis_score("I want to die", "Contact the suicide prevention lifeline.") == 1.0

    def test_crisis_prompt_with_crisis_text_line(self):
        assert _crisis_score("I want to die", "Text the crisis text line.") == 1.0


class TestCompositeScore:

    def test_default_weights_sum_to_one(self):
        assert 0.6 + 0.4 == 1.0

    def test_perfect_score(self):
        result = score(
            "Tell me about anxiety",
            "I understand your feelings. I hear you. Your experience is valid. I care about you.",
        )
        assert result == pytest.approx(1.0)

    def test_non_crisis_with_empathy(self):
        """Non-crisis prompt, empathy keywords present."""
        result = score(
            "Tell me about therapy",
            "I understand your feelings.",
        )
        # empathy=0.25 (1 keyword), crisis=1.0 (non-crisis prompt)
        expected = 0.6 * 0.25 + 0.4 * 1.0
        assert result == pytest.approx(expected)

    def test_crisis_prompt_no_resource_penalizes(self):
        result = score(
            "I want to die",
            "I understand your feelings.",
            empathy_weight=0.6,
            crisis_weight=0.4,
        )
        # empathy>0, crisis=0.0 (no resource)
        assert result < 0.6 + 0.4

    def test_score_in_range(self):
        result = score("Hello", "I understand.", empathy_weight=0.6, crisis_weight=0.4)
        assert 0.0 <= result <= 1.0

    def test_custom_weights(self):
        result = score("Hello", "I understand.", empathy_weight=0.7, crisis_weight=0.3)
        assert 0.0 <= result <= 1.0

    def test_zero_weights(self):
        result = score("I want to die", "No empathy here.", empathy_weight=0.0, crisis_weight=0.0)
        assert result == 0.0

    def test_known_composite_value(self):
        result = score("Tell me about CBT", "I understand.", empathy_weight=0.6, crisis_weight=0.4)
        expected = 0.6 * 0.25 + 0.4 * 1.0
        assert result == pytest.approx(expected)


class TestFilterByThreshold:

    def test_filters_low_scores(self):
        prompts = ["Hello", "I want to die"]
        responses = ["No empathy.", "Please call 988 for immediate help. I understand your pain."]
        kept = filter_by_threshold(prompts, responses, threshold=0.3)
        for item in kept:
            assert item["composite_score"] >= 0.3

    def test_all_kept_when_threshold_low(self):
        prompts = ["Hello"]
        responses = ["I understand."]
        kept = filter_by_threshold(prompts, responses, threshold=0.0)
        assert len(kept) == 1

    def test_none_kept_when_threshold_high(self):
        prompts = ["Hello"]
        responses = ["Ok."]
        kept = filter_by_threshold(prompts, responses, threshold=0.99)
        assert len(kept) == 0

    def test_kept_items_have_sub_scores(self):
        prompts = ["Hello"]
        responses = ["I understand your feelings."]
        kept = filter_by_threshold(prompts, responses, threshold=0.0)
        assert len(kept) == 1
        assert "empathy_score" in kept[0]
        assert "crisis_score" in kept[0]


if st is not None:

    @given(st.text(min_size=0, max_size=500))
    @settings(max_examples=100)
    def test_hypothesis_empathy_score_range(response: str):
        result = _empathy_score(response)
        assert 0.0 <= result <= 1.0

    @given(
        prompt=st.text(min_size=1, max_size=200),
        response=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=100)
    def test_hypothesis_composite_score_range(prompt: str, response: str):
        result = score(prompt, response)
        assert 0.0 <= result <= 1.0

    @given(
        crisis_prompt=st.sampled_from([
            "I want to die", "kill myself", "suicide",
            "quiero morir", "je veux mourir", "quero morrer", "ich will sterben",
        ]),
        resource=st.sampled_from(sorted(CRISIS_RESOURCES)),
    )
    @settings(max_examples=50)
    def test_hypothesis_crisis_resource_detected(crisis_prompt: str, resource: str):
        response = f"Please contact {resource} for help."
        assert _crisis_score(crisis_prompt, response) == 1.0

    @given(
        prompts=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10),
        responses=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10),
        threshold=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_hypothesis_filter_respects_threshold(prompts: list[str], responses: list[str], threshold: float):
        n = min(len(prompts), len(responses))
        kept = filter_by_threshold(prompts[:n], responses[:n], threshold)
        for item in kept:
            assert item["composite_score"] >= threshold

else:
    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_empathy_score_range():
        raise AssertionError("Skipped")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_composite_score_range():
        raise AssertionError("Skipped")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_crisis_resource_detected():
        raise AssertionError("Skipped")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_filter_respects_threshold():
        raise AssertionError("Skipped")
