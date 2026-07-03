"""Tests for the PATIENT-Ψ rational-emotive coherence model."""

from __future__ import annotations

import pytest

from ai.platform.patient_psi.coherence import CoherenceModel, CoherenceScore
from ai.platform.patient_psi.profiles import ProfileRegistry


class TestCoherenceScore:
    """CoherenceScore model validation."""

    def test_default_values(self) -> None:
        score = CoherenceScore()
        assert score.overall == 0.5
        assert score.belief_consistency == 0.5
        assert score.emotional_congruence == 0.5
        assert score.narrative_coherence == 0.5
        assert score.cognitive_dissonance == 0.0

    def test_fields_clamped_to_0_1(self) -> None:
        with pytest.raises(ValueError):
            CoherenceScore(overall=1.5)  # type: ignore[arg-type]


class TestCoherenceModel:
    """CoherenceModel evaluation logic."""

    def setup_method(self) -> None:
        registry = ProfileRegistry()
        self.depression = registry.get_profile("major_depressive_disorder")
        self.bpd = registry.get_profile("borderline_personality")
        self.model_dep = CoherenceModel(self.depression)
        self.model_bpd = CoherenceModel(self.bpd)

    def test_evaluate_returns_coherence_score(self) -> None:
        score = self.model_dep.evaluate("I feel hopeless about everything.")
        assert isinstance(score, CoherenceScore)

    def test_empty_response_returns_default(self) -> None:
        score = self.model_dep.evaluate("")
        assert score.overall == 0.5

    def test_whitespace_response_returns_default(self) -> None:
        score = self.model_dep.evaluate("   ")
        assert score.overall == 0.5

    def test_belief_consistent_response_scores_higher(self) -> None:
        score = self.model_dep.evaluate(
            "I feel completely worthless. Nothing will ever get better. "
            "The world just feels empty and meaningless to me now."
        )
        assert score.belief_consistency >= 0.3

    def test_dissonance_detected(self) -> None:
        score = self.model_dep.evaluate(
            "I feel hopeless but I also know I should be grateful. "
            "Part of me wants to get better but another part doesn't believe I can."
        )
        assert score.cognitive_dissonance > 0.1

    def test_depression_response_shows_negative_emotional_congruence(self) -> None:
        score = self.model_dep.evaluate("I am so sad and depressed. Nothing matters.")
        assert score.emotional_congruence >= 0.5

    def test_manic_response_incongruent_for_depression(self) -> None:
        score = self.model_dep.evaluate("I feel amazing! Everything is wonderful and I'm so excited!")
        assert score.emotional_congruence <= 0.5

    def test_long_narrative_scores_higher_coherence(self) -> None:
        score_long = self.model_dep.evaluate(
            "I've been feeling really down lately. Every morning I wake up and "
            "it takes all my energy just to get out of bed. I keep thinking about "
            "all the things I've done wrong and it just spirals from there."
        )
        score_short = self.model_dep.evaluate("I don't know.")
        assert score_long.narrative_coherence > score_short.narrative_coherence

    def test_dissonance_amplified_for_bpd(self) -> None:
        score = self.model_bpd.evaluate("I hate them but I also love them. I don't know what to feel.")
        assert score.cognitive_dissonance > 0.0

    def test_distressed_phase_modulation(self) -> None:
        score = self.model_dep.evaluate(
            "I feel terrible. Everything is falling apart.",
            context={"phase": "distressed"},
        )
        assert isinstance(score, CoherenceScore)

    def test_resistant_phase_modulation(self) -> None:
        score = self.model_dep.evaluate(
            "I don't want to talk about this.",
            context={"phase": "resistant"},
        )
        assert isinstance(score, CoherenceScore)

    def test_insight_phase_modulation(self) -> None:
        score = self.model_dep.evaluate(
            "I think I understand now why I react that way. It's because of what happened when I was young.",
            context={"phase": "insight"},
        )
        assert isinstance(score, CoherenceScore)

    def test_predict_coherence_range(self) -> None:
        low, high = self.model_dep.predict_coherence_range()
        assert low >= 0.0
        assert high <= 1.0
        assert low < high

    def test_different_profiles_have_different_ranges(self) -> None:
        _low_bpd, _high_bpd = self.model_bpd.predict_coherence_range()
        _low_dep, _high_dep = self.model_dep.predict_coherence_range()

    def test_all_dimensions_return_float(self) -> None:
        score = self.model_dep.evaluate("I feel completely lost and alone.")
        for val in (
            score.overall,
            score.belief_consistency,
            score.emotional_congruence,
            score.narrative_coherence,
            score.cognitive_dissonance,
        ):
            assert isinstance(val, float)

    def test_score_range_0_to_1(self) -> None:
        score = self.model_dep.evaluate(
            "I don't understand anything anymore. Everything is confusing and nothing makes sense. "
            "Maybe I should just give up. I don't know what to do."
        )
        for val in (
            score.overall,
            score.belief_consistency,
            score.emotional_congruence,
            score.narrative_coherence,
            score.cognitive_dissonance,
        ):
            assert val >= 0.0
            assert val <= 1.0

    def test_all_profiles_return_valid_scores(self) -> None:
        registry = ProfileRegistry()
        for name in registry.list_profiles():
            profile = registry.get_profile(name)
            model = CoherenceModel(profile)
            score = model.evaluate("I am feeling something today but I'm not sure what.")
            for val in (
                score.overall,
                score.belief_consistency,
                score.emotional_congruence,
                score.narrative_coherence,
            ):
                assert val >= 0.0
                assert val <= 1.0
            assert score.cognitive_dissonance >= 0.0
            assert score.cognitive_dissonance <= 1.0

    def test_hedging_does_not_artificially_inflate_coherence(self) -> None:
        score = self.model_dep.evaluate("I guess maybe sort of feel kind of okay I suppose perhaps")
        assert score.narrative_coherence <= 0.5
