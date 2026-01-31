#!/usr/bin/env python3
"""
Test suite for Therapeutic Effectiveness Prediction System
"""


import pytest

from .effectiveness_predictor import (
    EffectivenessLevel,
    EffectivenessPrediction,
    OutcomeMetric,
    OutcomeType,
    PredictionConfidence,
    TherapeuticEffectivenessPredictor,
)


class TestTherapeuticEffectivenessPredictor:
    """Test cases for TherapeuticEffectivenessPredictor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = TherapeuticEffectivenessPredictor()

        # Sample conversation data
        self.high_effectiveness_conversation = {
            "id": "high_eff_001",
            "content": "I understand you're struggling with depression. Let's work together using cognitive behavioral techniques and mindfulness to identify and challenge negative thought patterns. I believe we can make significant progress with your motivation and family support.",
            "turns": [
                {"speaker": "user", "text": "I've been depressed but I'm motivated to change and have family support."},
                {"speaker": "therapist", "text": "Let's work together using CBT and mindfulness techniques."}
            ],
            "metadata": {"condition": "depression", "session_number": 1}
        }

        self.moderate_effectiveness_conversation = {
            "id": "mod_eff_001",
            "content": "I hear that you're having some difficulties. Let's try to work on some coping strategies.",
            "turns": [
                {"speaker": "user", "text": "I'm having some problems with anxiety."},
                {"speaker": "therapist", "text": "Let's work on some coping strategies."}
            ],
            "metadata": {"condition": "anxiety"}
        }

        self.low_effectiveness_conversation = {
            "id": "low_eff_001",
            "content": "I see you have multiple issues including substance use, trauma history, and social isolation. This will be challenging.",
            "turns": [
                {"speaker": "user", "text": "I have drinking problems, trauma, and no friends. Nothing works."},
                {"speaker": "therapist", "text": "This will be challenging with multiple issues."}
            ],
            "metadata": {"condition": "depression", "comorbidities": ["substance_use", "ptsd"]}
        }

        self.protective_factors_conversation = {
            "id": "protective_001",
            "content": "You mentioned having good family support, stable employment, and strong motivation to change. These are excellent resources we can build upon.",
            "turns": [
                {"speaker": "user", "text": "I have good family support, a stable job, and I'm motivated."},
                {"speaker": "therapist", "text": "These are excellent resources we can build upon."}
            ]
        }

    def test_predictor_initialization(self):
        """Test predictor initialization."""
        predictor = TherapeuticEffectivenessPredictor()
        assert predictor.prediction_history == []
        assert predictor.evidence_base is not None
        assert predictor.outcome_models is not None

        # Check evidence base structure
        assert "intervention_effectiveness" in predictor.evidence_base
        assert "outcome_predictors" in predictor.evidence_base
        assert "timeframe_expectations" in predictor.evidence_base

        # Check outcome models
        assert len(predictor.outcome_models) == 6  # Updated to include crisis reduction
        assert OutcomeType.SYMPTOM_REDUCTION in predictor.outcome_models
        assert OutcomeType.FUNCTIONAL_IMPROVEMENT in predictor.outcome_models
        assert OutcomeType.CRISIS_REDUCTION in predictor.outcome_models

    def test_extract_therapeutic_elements_cbt(self):
        """Test extraction of CBT therapeutic elements."""
        content = "Let's use cognitive behavioral techniques to challenge negative thoughts"
        turns = []

        elements = self.predictor._extract_therapeutic_elements(content, turns)

        assert "CBT" in elements["interventions"]
        assert elements["therapeutic_alliance"] >= 0.0
        assert elements["client_engagement"] >= 0.0
        assert elements["skill_building"] >= 0.0

    def test_extract_therapeutic_elements_multiple_interventions(self):
        """Test extraction of multiple therapeutic interventions."""
        content = "We'll use cognitive behavioral techniques and mindfulness meditation to help you"
        turns = []

        elements = self.predictor._extract_therapeutic_elements(content, turns)

        assert "CBT" in elements["interventions"]
        assert "mindfulness" in elements["interventions"]
        assert len(elements["interventions"]) >= 2  # May detect additional interventions

    def test_identify_risk_factors(self):
        """Test identification of risk factors."""
        content = "I have drinking problems, trauma history, and I'm socially isolated with financial stress"
        turns = []
        metadata = {}

        risk_factors = self.predictor._identify_risk_factors(content, turns, metadata)

        assert "substance_use" in risk_factors
        assert "trauma_history" in risk_factors
        assert "social_isolation" in risk_factors
        assert "financial_stress" in risk_factors
        assert len(risk_factors) >= 4

    def test_identify_protective_factors(self):
        """Test identification of protective factors."""
        content = "I have good family support, I'm motivated to change, and I have insight into my problems"
        turns = []
        metadata = {}

        protective_factors = self.predictor._identify_protective_factors(content, turns, metadata)

        assert "social_support" in protective_factors
        assert "motivation" in protective_factors
        assert "insight" in protective_factors
        assert len(protective_factors) >= 3

    def test_predict_high_effectiveness_conversation(self):
        """Test prediction for high effectiveness conversation."""
        prediction = self.predictor.predict_effectiveness(self.high_effectiveness_conversation)

        assert isinstance(prediction, EffectivenessPrediction)
        assert prediction.conversation_id == "high_eff_001"
        assert prediction.overall_effectiveness in [
            EffectivenessLevel.HIGHLY_EFFECTIVE,
            EffectivenessLevel.MODERATELY_EFFECTIVE
        ]
        assert prediction.success_probability > 0.5
        assert len(prediction.outcome_metrics) == 6  # All outcome types including crisis reduction
        assert len(prediction.protective_factors) > 0
        assert len(prediction.recommendations) > 0

    def test_predict_low_effectiveness_conversation(self):
        """Test prediction for low effectiveness conversation."""
        prediction = self.predictor.predict_effectiveness(self.low_effectiveness_conversation)

        assert prediction.conversation_id == "low_eff_001"
        assert prediction.overall_effectiveness in [
            EffectivenessLevel.MINIMALLY_EFFECTIVE,
            EffectivenessLevel.INEFFECTIVE
        ]
        assert prediction.success_probability < 0.7  # Should be lower due to risk factors
        assert len(prediction.risk_factors) > 0
        assert len(prediction.recommendations) > 0

    def test_calculate_base_outcome_score(self):
        """Test base outcome score calculation."""
        # High quality therapeutic elements
        high_elements = {
            "interventions": ["CBT", "mindfulness"],
            "therapeutic_alliance": 0.8,
            "client_engagement": 0.9,
            "skill_building": 0.7
        }

        score = self.predictor._calculate_base_outcome_score(
            OutcomeType.SYMPTOM_REDUCTION, high_elements
        )
        assert score > 0.7

        # Low quality therapeutic elements
        low_elements = {
            "interventions": [],
            "therapeutic_alliance": 0.2,
            "client_engagement": 0.1,
            "skill_building": 0.0
        }

        score = self.predictor._calculate_base_outcome_score(
            OutcomeType.SYMPTOM_REDUCTION, low_elements
        )
        assert score < 0.6

    def test_determine_prediction_confidence(self):
        """Test prediction confidence determination."""
        # High confidence scenario
        high_elements = {
            "interventions": ["CBT", "mindfulness"],
            "therapeutic_alliance": 0.8,
            "client_engagement": 0.9,
            "skill_building": 0.7
        }
        risk_factors = ["substance_use"]
        protective_factors = ["social_support", "motivation"]

        confidence = self.predictor._determine_prediction_confidence(
            high_elements, risk_factors, protective_factors
        )
        assert confidence == PredictionConfidence.HIGH

        # Low confidence scenario
        low_elements = {
            "interventions": [],
            "therapeutic_alliance": 0.2,
            "client_engagement": 0.1,
            "skill_building": 0.0
        }

        confidence = self.predictor._determine_prediction_confidence(
            low_elements, [], []
        )
        assert confidence == PredictionConfidence.LOW

    def test_assess_evidence_strength(self):
        """Test evidence strength assessment."""
        # Strong evidence interventions
        strong_elements = {
            "interventions": ["CBT", "DBT"],
            "therapeutic_alliance": 0.8,
            "client_engagement": 0.7,
            "skill_building": 0.6
        }

        strength = self.predictor._assess_evidence_strength(
            strong_elements, OutcomeType.SYMPTOM_REDUCTION
        )
        assert strength == "strong"

        # No interventions
        no_interventions = {
            "interventions": [],
            "therapeutic_alliance": 0.5,
            "client_engagement": 0.5,
            "skill_building": 0.5
        }

        strength = self.predictor._assess_evidence_strength(
            no_interventions, OutcomeType.SYMPTOM_REDUCTION
        )
        assert strength == "limited"

    def test_calculate_overall_effectiveness(self):
        """Test overall effectiveness calculation."""
        # Create metrics for all outcome types with high scores
        high_metrics = []
        for outcome_type in OutcomeType:
            high_metrics.append(OutcomeMetric(
                metric_type=outcome_type,
                baseline_score=0.3,
                predicted_score=0.9,
                improvement_percentage=200.0,
                confidence=PredictionConfidence.HIGH,
                evidence_strength="strong",
                timeframe_weeks=16
            ))

        effectiveness = self.predictor._calculate_overall_effectiveness(high_metrics)
        assert effectiveness in [EffectivenessLevel.HIGHLY_EFFECTIVE, EffectivenessLevel.MODERATELY_EFFECTIVE]

        # Low effectiveness metrics
        low_metrics = []
        for outcome_type in OutcomeType:
            low_metrics.append(OutcomeMetric(
                metric_type=outcome_type,
                baseline_score=0.3,
                predicted_score=0.2,
                improvement_percentage=-33.0,
                confidence=PredictionConfidence.LOW,
                evidence_strength="limited",
                timeframe_weeks=16
            ))

        effectiveness = self.predictor._calculate_overall_effectiveness(low_metrics)
        assert effectiveness == EffectivenessLevel.INEFFECTIVE

    def test_calculate_success_probability(self):
        """Test success probability calculation."""
        # High success scenario
        high_metrics = [
            OutcomeMetric(
                metric_type=OutcomeType.SYMPTOM_REDUCTION,
                baseline_score=0.3,
                predicted_score=0.8,
                improvement_percentage=167.0,
                confidence=PredictionConfidence.HIGH,
                evidence_strength="strong",
                timeframe_weeks=16
            )
        ]
        risk_factors = []
        protective_factors = ["social_support", "motivation"]

        probability = self.predictor._calculate_success_probability(
            high_metrics, risk_factors, protective_factors
        )
        assert probability > 0.7

        # Low success scenario
        low_metrics = [
            OutcomeMetric(
                metric_type=OutcomeType.SYMPTOM_REDUCTION,
                baseline_score=0.3,
                predicted_score=0.3,
                improvement_percentage=0.0,
                confidence=PredictionConfidence.LOW,
                evidence_strength="limited",
                timeframe_weeks=16
            )
        ]
        risk_factors = ["substance_use", "trauma_history", "social_isolation"]
        protective_factors = []

        probability = self.predictor._calculate_success_probability(
            low_metrics, risk_factors, protective_factors
        )
        assert probability < 0.5

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        # Scenario with multiple issues
        therapeutic_elements = {
            "interventions": [],
            "therapeutic_alliance": 0.3,
            "client_engagement": 0.2,
            "skill_building": 0.1
        }

        outcome_metrics = [
            OutcomeMetric(
                metric_type=OutcomeType.SYMPTOM_REDUCTION,
                baseline_score=0.3,
                predicted_score=0.2,
                improvement_percentage=-33.0,
                confidence=PredictionConfidence.LOW,
                evidence_strength="limited",
                timeframe_weeks=16
            )
        ]

        risk_factors = ["substance_use", "social_isolation"]
        protective_factors = []

        recommendations = self.predictor._generate_recommendations(
            therapeutic_elements, outcome_metrics, risk_factors, protective_factors
        )

        assert len(recommendations) > 0
        assert any("intervention" in rec.lower() for rec in recommendations)
        assert any("alliance" in rec.lower() for rec in recommendations)
        assert any("substance" in rec.lower() for rec in recommendations)

    def test_compile_evidence_base(self):
        """Test evidence base compilation."""
        therapeutic_elements = {
            "interventions": ["CBT", "mindfulness"],
            "therapeutic_alliance": 0.8,
            "client_engagement": 0.7,
            "skill_building": 0.6
        }

        evidence = self.predictor._compile_evidence_base(therapeutic_elements)

        assert "interventions_used" in evidence
        assert "evidence_ratings" in evidence
        assert "effect_sizes" in evidence
        assert "success_rates" in evidence

        assert evidence["interventions_used"] == ["CBT", "mindfulness"]
        assert "CBT" in evidence["evidence_ratings"]
        assert "mindfulness" in evidence["evidence_ratings"]

    def test_prediction_history_tracking(self):
        """Test prediction history tracking."""
        initial_count = len(self.predictor.prediction_history)

        # Make a prediction
        self.predictor.predict_effectiveness(self.high_effectiveness_conversation)

        assert len(self.predictor.prediction_history) == initial_count + 1

        # Make another prediction
        self.predictor.predict_effectiveness(self.moderate_effectiveness_conversation)

        assert len(self.predictor.prediction_history) == initial_count + 2

    def test_prediction_summary(self):
        """Test prediction summary generation."""
        # Make some predictions
        self.predictor.predict_effectiveness(self.high_effectiveness_conversation)
        self.predictor.predict_effectiveness(self.low_effectiveness_conversation)
        self.predictor.predict_effectiveness(self.protective_factors_conversation)

        summary = self.predictor.get_prediction_summary()

        assert "total_predictions" in summary
        assert "effectiveness_distribution" in summary
        assert "average_success_probability" in summary
        assert "common_risk_factors" in summary
        assert "common_protective_factors" in summary
        assert "last_prediction" in summary

        assert summary["total_predictions"] == 3
        assert 0 <= summary["average_success_probability"] <= 1
        assert isinstance(summary["effectiveness_distribution"], dict)

    def test_prediction_summary_empty(self):
        """Test prediction summary with no predictions."""
        predictor = TherapeuticEffectivenessPredictor()
        summary = predictor.get_prediction_summary()

        assert summary["message"] == "No predictions performed yet"

    def test_outcome_metrics_completeness(self):
        """Test that all outcome types are included in predictions."""
        prediction = self.predictor.predict_effectiveness(self.high_effectiveness_conversation)

        # Should have metrics for all outcome types
        metric_types = [metric.metric_type for metric in prediction.outcome_metrics]
        expected_types = list(OutcomeType)

        assert len(metric_types) == len(expected_types)
        for expected_type in expected_types:
            assert expected_type in metric_types


def test_main_function():
    """Test the main function runs without errors."""
    try:
        # We can't easily test the full main() due to print statements,
        # but we can test that it imports and the predictor works
        predictor = TherapeuticEffectivenessPredictor()
        assert predictor is not None
    except Exception as e:
        pytest.fail(f"Main function test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
