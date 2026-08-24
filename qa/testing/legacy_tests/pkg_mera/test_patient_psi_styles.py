"""Tests for patient_psi conversational style templates."""

from __future__ import annotations

import pytest

from ai.tools.utilities.platform.patient_psi.styles import (
    ConversationalStyle,
    StyleRegistry,
    StyleTemplate,
)


class TestConversationalStyle:
    """Verify the ConversationalStyle enum."""

    def test_all_six_styles_exist(self) -> None:
        styles = list(ConversationalStyle)
        assert len(styles) == 6
        expected = {
            "neutral",
            "friendly",
            "hostile",
            "anxious",
            "melancholic",
            "manic",
        }
        assert {s.value for s in styles} == expected


class TestStyleRegistry:
    """Verify StyleRegistry behaviour."""

    def setup_method(self) -> None:
        self.registry = StyleRegistry()

    def test_registry_has_all_six_styles(self) -> None:
        styles = self.registry.list_styles()
        assert len(styles) == 6
        assert set(styles) == set(ConversationalStyle)

    def test_each_style_has_non_empty_templates(self) -> None:
        categories = [
            "greeting_templates",
            "question_templates",
            "response_templates",
            "counter_question_templates",
            "closure_templates",
        ]
        for style in ConversationalStyle:
            template = self.registry.get_style(style)
            for category in categories:
                assert getattr(template, category), f"{style}.{category} is empty"

    def test_get_utterance_returns_non_empty_string(self) -> None:
        for style in ConversationalStyle:
            for utterance_type in (
                "greeting",
                "question",
                "response",
                "counter_question",
                "closure",
            ):
                result = self.registry.get_utterance(style, utterance_type)
                assert isinstance(result, str)
                assert result

    def test_get_utterance_renders_jinja2_variables(self) -> None:
        context = {"patient_name": "Alex"}
        result = self.registry.get_utterance(ConversationalStyle.NEUTRAL, "greeting", context)
        assert "Alex" in result

    def test_get_utterance_random_selection(self) -> None:
        """Two calls may differ due to random template selection."""
        results = {self.registry.get_utterance(ConversationalStyle.FRIENDLY, "greeting") for _ in range(50)}
        assert len(results) >= 1
        # With 4 templates, 50 draws should almost certainly yield >1 unique.
        assert len(results) > 1, "Expected random selection to vary across 50 draws"

    def test_get_utterance_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError) as cm:
            self.registry.get_utterance(ConversationalStyle.NEUTRAL, "invalid_type")
        assert "Unknown utterance_type" in str(cm.value)

    def test_get_utterance_empty_context_produces_output(self) -> None:
        result = self.registry.get_utterance(ConversationalStyle.ANXIOUS, "response", context={})
        assert isinstance(result, str)
        assert result

    def test_get_style_markers_returns_expected_keys(self) -> None:
        expected_keys = {
            "formality",
            "emotional_valence",
            "assertiveness",
            "verbosity",
            "insight_level",
        }
        for style in ConversationalStyle:
            markers = self.registry.get_style_markers(style)
            assert set(markers.keys()) == expected_keys
            for value in markers.values():
                assert isinstance(value, float)
                assert 0.0 <= value <= 1.0

    def test_styles_produce_qualitatively_different_output(self) -> None:
        """Ensure each style produces recognisably different text."""
        samples = {style: self.registry.get_utterance(style, "greeting") for style in ConversationalStyle}
        # All samples should be unique (probabilistic but highly likely).
        assert len(set(samples.values())) == 6, samples


class TestStyleTemplateModel:
    """Verify StyleTemplate pydantic model."""

    def test_instantiation(self) -> None:
        template = StyleTemplate(
            name="Test",
            style=ConversationalStyle.NEUTRAL,
            greeting_templates=["Hello, {{patient_name}}."],
            question_templates=["What about {{context}}?"],
            response_templates=["I see."],
            counter_question_templates=["Why do you ask?"],
            closure_templates=["Goodbye."],
            style_markers={
                "formality": 0.5,
                "emotional_valence": 0.0,
                "assertiveness": 0.5,
                "verbosity": 0.5,
                "insight_level": 0.5,
            },
        )
        assert template.name == "Test"
        assert template.style == ConversationalStyle.NEUTRAL
