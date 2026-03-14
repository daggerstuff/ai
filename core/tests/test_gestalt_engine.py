"""
Tests for PIX-149: Gestalt Fusion Engine (ai/core/gestalt_engine.py)

Verifies all behavioral branches without requiring a real model checkpoint.
The PsyDefDetect model is mocked at the DefenseClassifier level so the full
GestaltEngine business logic runs — only inference is faked.

Run:
    uv run pytest ai/core/tests/test_gestalt_engine.py -v
"""

from __future__ import annotations

import math
from typing import Optional
from unittest.mock import MagicMock

import pytest

from ai.core.gestalt_engine import (
    OCEAN_TRAITS,
    PLUTCHIK_EMOTIONS,
    CrisisLevel,
    GestaltEngine,
    GestaltState,
    _behavioral_prediction,
    _breakthrough_score,
    _compute_crisis_level,
    _dominant_emotion,
    _persona_directive,
    _validate_scores,
)
from ai.training.defense_mechanisms.model import DefensePrediction

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

MINIMAL_PLUTCHIK: dict[str, float] = {e: 0.0 for e in PLUTCHIK_EMOTIONS}
MINIMAL_OCEAN: dict[str, float] = {t: 0.5 for t in OCEAN_TRAITS}

SAMPLE_DIALOGUE = [
    {"speaker": "Supporter", "text": "How are you feeling today?"},
    {"speaker": "Seeker", "text": "Fine. Everything is fine."},
]
SAMPLE_TARGET = "Fine. Everything is fine."


def make_defense_prediction(
    label: int = 0,
    confidence: float = 0.9,
    maturity_score: Optional[float] = None,
) -> DefensePrediction:
    """Build a minimal DefensePrediction for injection into mocks."""
    from ai.training.defense_mechanisms.constants import DEFENSE_LABELS, DEFENSE_MATURITY

    probs = [0.0] * 9
    probs[label] = confidence
    return DefensePrediction(
        label=label,
        label_name=DEFENSE_LABELS.get(label, "Unknown"),
        confidence=confidence,
        probabilities=probs,
        maturity_score=maturity_score
        if maturity_score is not None
        else DEFENSE_MATURITY.get(label),
        raw_logits=[0.0] * 9,
    )


def make_engine_with_mock_model(prediction: DefensePrediction) -> GestaltEngine:
    """
    Return a GestaltEngine whose internal model is replaced by a mock that
    returns a fixed DefensePrediction.

    This bypasses the need for a real checkpoint while exercising all real
    GestaltEngine logic.
    """
    engine = GestaltEngine()

    mock_model = MagicMock()
    mock_model.predict.return_value = [prediction]
    # Make next(model.parameters()).device return a CPU device-like string
    mock_param = MagicMock()
    mock_param.device = "cpu"
    mock_model.parameters.side_effect = lambda: iter([mock_param])

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }

    engine._defense_model = mock_model
    engine._defense_tokenizer = mock_tokenizer
    return engine


# ---------------------------------------------------------------------------
# Unit tests: pure functions
# ---------------------------------------------------------------------------


class TestDominantEmotion:
    def test_returns_highest_score(self):
        scores = {**MINIMAL_PLUTCHIK, "sadness": 0.8, "fear": 0.3}
        emotion, intensity = _dominant_emotion(scores)
        assert emotion == "sadness"
        assert math.isclose(intensity, 0.8)

    def test_empty_dict_returns_unknown(self):
        emotion, intensity = _dominant_emotion({})
        assert emotion == "unknown"
        assert intensity == 0.0

    def test_all_zero_picks_first_alphabetically(self):
        scores = {e: 0.0 for e in PLUTCHIK_EMOTIONS}
        emotion, intensity = _dominant_emotion(scores)
        assert emotion in PLUTCHIK_EMOTIONS
        assert intensity == 0.0


class TestComputeCrisisLevel:
    """Verify the priority-ordered crisis classification rules."""

    def test_action_defense_high_sadness_is_acute(self):
        level = _compute_crisis_level(
            defense_label=1,  # Action
            defense_maturity=0.0,
            dominant_emotion="sadness",
            dominant_intensity=0.75,
            ocean_neuroticism=0.5,
        )
        assert level == CrisisLevel.ACUTE

    def test_action_defense_high_neuroticism_is_acute(self):
        level = _compute_crisis_level(
            defense_label=1,
            defense_maturity=0.0,
            dominant_emotion="joy",  # Not a crisis emotion
            dominant_intensity=0.9,
            ocean_neuroticism=0.80,
        )
        assert level == CrisisLevel.ACUTE

    def test_action_defense_alone_is_high(self):
        level = _compute_crisis_level(
            defense_label=1,
            defense_maturity=0.0,
            dominant_emotion="trust",  # Not crisis-amplifying
            dominant_intensity=0.5,
            ocean_neuroticism=0.4,
        )
        assert level == CrisisLevel.HIGH

    def test_major_image_distorting_with_fear_is_high(self):
        level = _compute_crisis_level(
            defense_label=2,  # Major Image-Distorting
            defense_maturity=0.14,
            dominant_emotion="fear",
            dominant_intensity=0.6,
            ocean_neuroticism=0.5,
        )
        assert level == CrisisLevel.HIGH

    def test_low_maturity_with_crisis_emotion_is_elevated(self):
        level = _compute_crisis_level(
            defense_label=3,  # Disavowal
            defense_maturity=0.20,
            dominant_emotion="anger",
            dominant_intensity=0.5,
            ocean_neuroticism=0.4,
        )
        assert level == CrisisLevel.ELEVATED

    def test_mature_defense_is_none(self):
        level = _compute_crisis_level(
            defense_label=7,  # High-Adaptive
            defense_maturity=1.0,
            dominant_emotion="joy",
            dominant_intensity=0.8,
            ocean_neuroticism=0.2,
        )
        assert level == CrisisLevel.NONE

    def test_neurotic_defense_with_low_emotion_is_none(self):
        level = _compute_crisis_level(
            defense_label=5,  # Neurotic
            defense_maturity=0.57,
            dominant_emotion="anticipation",
            dominant_intensity=0.3,
            ocean_neuroticism=0.4,
        )
        assert level == CrisisLevel.NONE


class TestBehavioralPrediction:
    def test_acute_mentions_deescalate(self):
        result = _behavioral_prediction(
            "Action Defenses", "sadness", CrisisLevel.ACUTE, 0.0
        )
        assert "de-escalate" in result.lower()

    def test_high_mentions_validate(self):
        result = _behavioral_prediction(
            "Action Defenses", "fear", CrisisLevel.HIGH, 0.0
        )
        assert "validate" in result.lower()

    def test_elevated_mentions_reflective(self):
        result = _behavioral_prediction(
            "Disavowal", "anger", CrisisLevel.ELEVATED, 0.20
        )
        assert "reflective" in result.lower()

    def test_mature_mentions_adaptive(self):
        result = _behavioral_prediction("High-Adaptive", "joy", CrisisLevel.NONE, 1.0)
        assert "adaptive" in result.lower()

    def test_intermediate_mentions_exploration(self):
        result = _behavioral_prediction("Obsessional", "trust", CrisisLevel.NONE, 0.57)
        assert "exploration" in result.lower()


class TestPersonaDirective:
    def test_action_defense_yields_injection_clause(self):
        directive = _persona_directive(1, "Action Defenses", 0.0)
        assert "[System:" in directive
        assert "Action Defense" in directive

    def test_major_image_distorting_yields_splitting_clause(self):
        directive = _persona_directive(2, "Major Image-Distorting", 0.14)
        assert "Splitting" in directive or "Image-Distorting" in directive

    def test_disavowal_yields_denial_clause(self):
        directive = _persona_directive(3, "Disavowal", 0.29)
        assert "Deny" in directive or "Disavowal" in directive

    def test_high_adaptive_yields_empty_string(self):
        directive = _persona_directive(7, "High-Adaptive", 1.0)
        assert directive == ""

    def test_neurotic_at_threshold_yields_empty_string(self):
        # 0.57 >= 0.43 threshold → no injection
        directive = _persona_directive(5, "Neurotic", 0.57)
        assert directive == ""

    def test_none_maturity_yields_empty_string(self):
        directive = _persona_directive(0, "Neutral", None)
        assert directive == ""


class TestBreakthroughScore:
    def test_maturity_increase_scores_delta(self):
        score = _breakthrough_score(defense_maturity=0.71, previous_maturity=0.43)
        assert math.isclose(score, 0.28, abs_tol=1e-9)

    def test_maturity_decrease_scores_zero(self):
        score = _breakthrough_score(defense_maturity=0.14, previous_maturity=0.57)
        assert score == 0.0

    def test_no_previous_baseline_scores_zero(self):
        assert _breakthrough_score(defense_maturity=0.71, previous_maturity=None) == 0.0

    def test_none_maturity_scores_zero(self):
        assert _breakthrough_score(defense_maturity=None, previous_maturity=0.5) == 0.0

    def test_capped_at_one(self):
        score = _breakthrough_score(defense_maturity=1.0, previous_maturity=0.0)
        assert score == 1.0


class TestValidateScores:
    def test_valid_scores_returned_as_floats(self):
        result = _validate_scores(
            {"joy": 0.8, "sadness": 0.2}, PLUTCHIK_EMOTIONS, "test"
        )
        assert result["joy"] == 0.8
        assert isinstance(result["joy"], float)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            _validate_scores({"joy": 1.1}, PLUTCHIK_EMOTIONS, "test")

    def test_negative_score_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            _validate_scores({"joy": -0.1}, PLUTCHIK_EMOTIONS, "test")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="must be numeric"):
            _validate_scores({"joy": "high"}, PLUTCHIK_EMOTIONS, "test")

    def test_unknown_keys_logged_but_accepted(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            result = _validate_scores(
                {"joy": 0.5, "custom_emotion": 0.3},
                PLUTCHIK_EMOTIONS,
                "test",
            )
        assert "custom_emotion" in result
        assert any("unknown" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Integration tests: GestaltEngine with mocked model
# ---------------------------------------------------------------------------


class TestGestaltEngineAnalyze:
    """Full analyze_gestalt() integration — model mocked, logic real."""

    @pytest.fixture
    def action_engine(self):
        """Engine configured to predict Action Defenses (label=1, maturity=0.0)."""
        pred = make_defense_prediction(label=1, confidence=0.88, maturity_score=0.0)
        return make_engine_with_mock_model(pred)

    @pytest.fixture
    def high_adaptive_engine(self):
        """Engine configured to predict High-Adaptive (label=7, maturity=1.0)."""
        pred = make_defense_prediction(label=7, confidence=0.92, maturity_score=1.0)
        return make_engine_with_mock_model(pred)

    def test_returns_gestalt_state(self, action_engine):
        state = action_engine.analyze_gestalt(
            dialogue=SAMPLE_DIALOGUE,
            target_utterance=SAMPLE_TARGET,
            plutchik_scores={**MINIMAL_PLUTCHIK, "sadness": 0.7},
            ocean_scores=MINIMAL_OCEAN,
        )
        assert isinstance(state, GestaltState)

    def test_action_defense_high_sadness_is_acute(self, action_engine):
        state = action_engine.analyze_gestalt(
            dialogue=SAMPLE_DIALOGUE,
            target_utterance=SAMPLE_TARGET,
            plutchik_scores={**MINIMAL_PLUTCHIK, "sadness": 0.75},
            ocean_scores=MINIMAL_OCEAN,
        )
        assert state.crisis_level == CrisisLevel.ACUTE

    def test_action_defense_yields_persona_directive(self, action_engine):
        state = action_engine.analyze_gestalt(
            dialogue=SAMPLE_DIALOGUE,
            target_utterance=SAMPLE_TARGET,
            plutchik_scores=MINIMAL_PLUTCHIK,
            ocean_scores=MINIMAL_OCEAN,
        )
        assert "[System:" in state.persona_directive

    def test_high_adaptive_crisis_is_none(self, high_adaptive_engine):
        state = high_adaptive_engine.analyze_gestalt(
            dialogue=SAMPLE_DIALOGUE,
            target_utterance=SAMPLE_TARGET,
            plutchik_scores={**MINIMAL_PLUTCHIK, "joy": 0.8},
            ocean_scores=MINIMAL_OCEAN,
        )
        assert state.crisis_level == CrisisLevel.NONE
        assert state.persona_directive == ""

    def test_breakthrough_score_positive_after_maturity_increase(self):
        """Simulate two turns where maturity rises from 0 → 1."""
        engine = make_engine_with_mock_model(
            make_defense_prediction(label=1, maturity_score=0.0)
        )
        # First turn: baseline established at 0.0
        engine.analyze_gestalt(
            SAMPLE_DIALOGUE,
            SAMPLE_TARGET,
            plutchik_scores=MINIMAL_PLUTCHIK,
            ocean_scores=MINIMAL_OCEAN,
        )

        # Second turn: replace model to return High-Adaptive
        engine._defense_model.predict.return_value = [
            make_defense_prediction(label=7, maturity_score=1.0)
        ]
        state2 = engine.analyze_gestalt(
            SAMPLE_DIALOGUE,
            SAMPLE_TARGET,
            plutchik_scores=MINIMAL_PLUTCHIK,
            ocean_scores=MINIMAL_OCEAN,
        )
        assert math.isclose(state2.breakthrough_score, 1.0, abs_tol=1e-9)

    def test_reset_session_clears_previous_maturity(self):
        engine = make_engine_with_mock_model(
            make_defense_prediction(label=7, maturity_score=1.0)
        )
        engine.analyze_gestalt(
            SAMPLE_DIALOGUE,
            SAMPLE_TARGET,
            plutchik_scores=MINIMAL_PLUTCHIK,
            ocean_scores=MINIMAL_OCEAN,
        )
        engine.reset_session()
        assert engine._previous_maturity is None

    def test_defense_model_not_loaded_raises_runtime_error(self):
        engine = GestaltEngine()  # No model loaded
        with pytest.raises(RuntimeError, match="load_defense_model"):
            engine.analyze_gestalt(
                SAMPLE_DIALOGUE,
                SAMPLE_TARGET,
                plutchik_scores=MINIMAL_PLUTCHIK,
                ocean_scores=MINIMAL_OCEAN,
            )

    def test_invalid_plutchik_score_raises_value_error(self, action_engine):
        with pytest.raises(ValueError, match="out of range"):
            action_engine.analyze_gestalt(
                SAMPLE_DIALOGUE,
                SAMPLE_TARGET,
                plutchik_scores={"sadness": 1.5},
                ocean_scores=MINIMAL_OCEAN,
            )

    def test_invalid_ocean_score_raises_value_error(self, action_engine):
        with pytest.raises(ValueError, match="out of range"):
            action_engine.analyze_gestalt(
                SAMPLE_DIALOGUE,
                SAMPLE_TARGET,
                plutchik_scores=MINIMAL_PLUTCHIK,
                ocean_scores={"neuroticism": -0.1},
            )

    def test_partial_plutchik_fills_missing_with_zero(self, action_engine):
        state = action_engine.analyze_gestalt(
            SAMPLE_DIALOGUE,
            SAMPLE_TARGET,
            plutchik_scores={"sadness": 0.5},  # Only one emotion provided
            ocean_scores=MINIMAL_OCEAN,
        )
        assert state.plutchik_scores["anger"] == 0.0
        assert state.plutchik_scores["sadness"] == 0.5

    def test_partial_ocean_fills_missing_with_half(self, action_engine):
        state = action_engine.analyze_gestalt(
            SAMPLE_DIALOGUE,
            SAMPLE_TARGET,
            plutchik_scores=MINIMAL_PLUTCHIK,
            ocean_scores={"neuroticism": 0.8},  # Only one trait provided
        )
        assert state.ocean_scores["openness"] == 0.5
        assert state.ocean_scores["neuroticism"] == 0.8

    def test_defense_confidence_in_range(self, action_engine):
        state = action_engine.analyze_gestalt(
            SAMPLE_DIALOGUE,
            SAMPLE_TARGET,
            plutchik_scores=MINIMAL_PLUTCHIK,
            ocean_scores=MINIMAL_OCEAN,
        )
        assert 0.0 <= state.defense_confidence <= 1.0

    def test_dominant_emotion_reflects_max_score(self, action_engine):
        state = action_engine.analyze_gestalt(
            SAMPLE_DIALOGUE,
            SAMPLE_TARGET,
            plutchik_scores={**MINIMAL_PLUTCHIK, "fear": 0.9, "sadness": 0.6},
            ocean_scores=MINIMAL_OCEAN,
        )
        assert state.dominant_emotion == "fear"
        assert math.isclose(state.dominant_emotion_intensity, 0.9)

    def test_state_has_all_defense_labels_in_probabilities(self, action_engine):
        state = action_engine.analyze_gestalt(
            SAMPLE_DIALOGUE,
            SAMPLE_TARGET,
            plutchik_scores=MINIMAL_PLUTCHIK,
            ocean_scores=MINIMAL_OCEAN,
        )
        from ai.training.defense_mechanisms.constants import DEFENSE_LABELS

        # Ensure all defense labels are present in the probabilities map
        assert set(DEFENSE_LABELS.values()).issubset(state.defense_probabilities.keys())

    def test_defense_model_loaded_property(self, action_engine):
        assert action_engine.defense_model_loaded is True

    def test_defense_model_not_loaded_property(self):
        engine = GestaltEngine()
        assert engine.defense_model_loaded is False

    def test_orchestrates_all_models_without_scores(self, action_engine):
        """
        analyze_gestalt should still return a coherent GestaltState when callers omit
        explicit Plutchik and OCEAN payloads.
        """
        state = action_engine.analyze_gestalt(
            dialogue=SAMPLE_DIALOGUE,
            target_utterance=SAMPLE_TARGET,
        )

        assert state.behavioral_pattern
        assert 0.0 <= state.behavioral_pattern_confidence <= 1.0
        assert set(state.plutchik_scores).issuperset(PLUTCHIK_EMOTIONS)
        assert set(state.ocean_scores).issuperset(OCEAN_TRAITS)
