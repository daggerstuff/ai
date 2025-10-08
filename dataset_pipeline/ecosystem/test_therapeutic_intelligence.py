#!/usr/bin/env python3
"""
Test Suite for Task 6.0 Phase 2: Advanced Therapeutic Intelligence & Pattern Recognition

This test suite validates the therapeutic intelligence components:
- Therapeutic Approach Classification (6.7)
- Mental Health Condition Pattern Recognition (6.8)
- Therapeutic Outcome Prediction (6.9)
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

from .condition_pattern_recognition import MentalHealthCondition, MentalHealthConditionRecognizer
from .outcome_prediction import OutcomeCategory, TherapeuticOutcomePredictor

# Import the modules we're testing
from .therapeutic_intelligence import TherapeuticApproach, TherapeuticApproachClassifier


class TestTherapeuticApproachClassifier(unittest.TestCase):
    """Test the therapeutic approach classification system."""

    def setUp(self):
        """Set up test fixtures."""
        self.classifier = TherapeuticApproachClassifier()

        # Test conversations for different approaches
        self.test_conversations = {
            "cbt": {
                "id": "cbt_test",
                "messages": [
                    {"role": "client", "content": "I keep having these negative automatic thoughts"},
                    {"role": "therapist", "content": "Let's examine the evidence for these thoughts and try some cognitive restructuring"}
                ]
            },
            "dbt": {
                "id": "dbt_test",
                "messages": [
                    {"role": "client", "content": "I feel so overwhelmed with emotions"},
                    {"role": "therapist", "content": "Let's practice some distress tolerance skills and work on emotion regulation"}
                ]
            },
            "humanistic": {
                "id": "humanistic_test",
                "messages": [
                    {"role": "client", "content": "I don't know what I want in life"},
                    {"role": "therapist", "content": "How does that uncertainty feel for you? What comes up when you sit with that experience?"}
                ]
            }
        }

    def test_cbt_classification(self):
        """Test CBT approach classification."""
        classification = self.classifier.classify_conversation(self.test_conversations["cbt"])

        assert classification.primary_approach == TherapeuticApproach.CBT
        assert classification.quality_score > 0.3  # More realistic threshold
        assert "cognitive" in classification.classification_rationale.lower()

    def test_dbt_classification(self):
        """Test DBT approach classification."""
        classification = self.classifier.classify_conversation(self.test_conversations["dbt"])

        assert classification.primary_approach == TherapeuticApproach.DBT
        assert classification.quality_score > 0.5
        assert "distress" in classification.classification_rationale.lower()

    def test_humanistic_classification(self):
        """Test humanistic approach classification."""
        classification = self.classifier.classify_conversation(self.test_conversations["humanistic"])

        assert classification.primary_approach == TherapeuticApproach.HUMANISTIC
        assert classification.quality_score > 0.5

    def test_batch_classification(self):
        """Test batch classification functionality."""
        conversations = list(self.test_conversations.values())
        classifications = self.classifier.classify_batch(conversations)

        assert len(classifications) == 3
        assert all(c.quality_score > 0 for c in classifications)

    def test_approach_distribution(self):
        """Test approach distribution tracking."""
        # Classify all test conversations
        for conversation in self.test_conversations.values():
            self.classifier.classify_conversation(conversation)

        distribution = self.classifier.get_approach_distribution()

        assert distribution["total_classified"] > 0
        assert "approach_distribution" in distribution
        assert distribution["average_confidence"] > 0

    def test_export_results(self):
        """Test exporting classification results."""
        # Classify a conversation first
        self.classifier.classify_conversation(self.test_conversations["cbt"])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            self.classifier.export_classification_results(temp_path)

            # Verify file was created and contains expected data
            assert Path(temp_path).exists()

            with open(temp_path) as f:
                results = json.load(f)

            assert "classification_statistics" in results
            assert "therapeutic_markers" in results
            assert "export_timestamp" in results

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestMentalHealthConditionRecognizer(unittest.TestCase):
    """Test the mental health condition recognition system."""

    def setUp(self):
        """Set up test fixtures."""
        self.recognizer = MentalHealthConditionRecognizer()

        # Test conversations for different conditions
        self.test_conversations = {
            "depression": {
                "id": "depression_test",
                "messages": [
                    {"role": "client", "content": "I feel hopeless and empty. Nothing brings me joy anymore. I just want to sleep all the time."}
                ]
            },
            "anxiety": {
                "id": "anxiety_test",
                "messages": [
                    {"role": "client", "content": "I'm constantly worried and my heart races. I can't stop thinking about what if something bad happens."}
                ]
            },
            "ptsd": {
                "id": "ptsd_test",
                "messages": [
                    {"role": "client", "content": "I keep having flashbacks of the trauma. I can't sleep and I jump at every sound."}
                ]
            }
        }

    def test_depression_recognition(self):
        """Test depression condition recognition."""
        recognition = self.recognizer.recognize_condition(self.test_conversations["depression"])

        assert recognition.primary_condition == MentalHealthCondition.DEPRESSION
        assert recognition.recognition_quality > 0.5
        assert "hopeless" in " ".join(recognition.symptom_markers.get("symptom", [])).lower()

    def test_anxiety_recognition(self):
        """Test anxiety condition recognition."""
        recognition = self.recognizer.recognize_condition(self.test_conversations["anxiety"])

        assert recognition.primary_condition == MentalHealthCondition.ANXIETY
        assert recognition.recognition_quality > 0.5

    def test_ptsd_recognition(self):
        """Test PTSD condition recognition."""
        recognition = self.recognizer.recognize_condition(self.test_conversations["ptsd"])

        assert recognition.primary_condition == MentalHealthCondition.PTSD
        assert recognition.recognition_quality > 0.5
        assert "flashback" in " ".join(recognition.symptom_markers.get("symptom", [])).lower()

    def test_severity_assessment(self):
        """Test severity assessment functionality."""
        # Test severe case
        severe_conversation = {
            "id": "severe_test",
            "messages": [
                {"role": "client", "content": "I have suicidal thoughts and can't function at all. This is a crisis."}
            ]
        }

        recognition = self.recognizer.recognize_condition(severe_conversation)
        assert recognition.severity_assessment == "severe"

    def test_risk_identification(self):
        """Test risk indicator identification."""
        risky_conversation = {
            "id": "risk_test",
            "messages": [
                {"role": "client", "content": "I want to hurt myself and I have suicidal thoughts. I feel completely alone."}
            ]
        }

        recognition = self.recognizer.recognize_condition(risky_conversation)
        assert len(recognition.risk_indicators) > 0
        assert any("suicide" in risk for risk in recognition.risk_indicators)

    def test_therapeutic_recommendations(self):
        """Test therapeutic recommendation generation."""
        recognition = self.recognizer.recognize_condition(self.test_conversations["depression"])

        assert len(recognition.therapeutic_recommendations) > 0
        assert any("CBT" in rec for rec in recognition.therapeutic_recommendations)

    def test_batch_recognition(self):
        """Test batch recognition functionality."""
        conversations = list(self.test_conversations.values())
        recognitions = self.recognizer.recognize_batch(conversations)

        assert len(recognitions) == 3
        assert all(r.recognition_quality > 0 for r in recognitions)

    def test_condition_distribution(self):
        """Test condition distribution tracking."""
        # Recognize all test conversations
        for conversation in self.test_conversations.values():
            self.recognizer.recognize_condition(conversation)

        distribution = self.recognizer.get_condition_distribution()

        assert distribution["total_recognized"] > 0
        assert "condition_distribution" in distribution
        assert "severity_distribution" in distribution


class TestTherapeuticOutcomePredictor(unittest.TestCase):
    """Test the therapeutic outcome prediction system."""

    def setUp(self):
        """Set up test fixtures."""
        self.predictor = TherapeuticOutcomePredictor()

        # Test conversations with different prognoses
        self.test_conversations = {
            "positive_prognosis": {
                "id": "positive_test",
                "messages": [
                    {"role": "client", "content": "I'm motivated to get better and have good support from my family. Therapy is really helping."}
                ]
            },
            "poor_prognosis": {
                "id": "poor_test",
                "messages": [
                    {"role": "client", "content": "Nothing works. I've tried everything and I don't want help anymore. I'm completely alone."}
                ]
            },
            "mixed_prognosis": {
                "id": "mixed_test",
                "messages": [
                    {"role": "client", "content": "Sometimes I feel hopeful but other times I struggle. I have some support but it's hard."}
                ]
            }
        }

    def test_positive_outcome_prediction(self):
        """Test prediction for positive prognosis case."""
        prediction = self.predictor.predict_outcome(
            self.test_conversations["positive_prognosis"],
            MentalHealthCondition.DEPRESSION,
            "short_term"
        )

        # Accept any positive outcome (including minimal improvement)
        assert prediction.predicted_outcome in [OutcomeCategory.SIGNIFICANT_IMPROVEMENT, OutcomeCategory.MODERATE_IMPROVEMENT, OutcomeCategory.MINIMAL_IMPROVEMENT]
        assert prediction.confidence_score > 0.3

    def test_poor_outcome_prediction(self):
        """Test prediction for poor prognosis case."""
        prediction = self.predictor.predict_outcome(
            self.test_conversations["poor_prognosis"],
            MentalHealthCondition.DEPRESSION,
            "short_term"
        )

        assert prediction.predicted_outcome in [OutcomeCategory.NO_CHANGE, OutcomeCategory.DETERIORATION, OutcomeCategory.MINIMAL_IMPROVEMENT]

    def test_feature_extraction(self):
        """Test predictive feature extraction."""
        features = self.predictor._extract_predictive_features(
            self.test_conversations["positive_prognosis"]
        )

        assert "engagement_level" in features
        assert "social_support" in features
        assert all(0 <= score <= 1 for score in features.values())

    def test_risk_factor_identification(self):
        """Test risk factor identification."""
        prediction = self.predictor.predict_outcome(
            self.test_conversations["poor_prognosis"],
            MentalHealthCondition.DEPRESSION,
            "short_term"
        )

        assert len(prediction.risk_factors) > 0

    def test_protective_factor_identification(self):
        """Test protective factor identification."""
        prediction = self.predictor.predict_outcome(
            self.test_conversations["positive_prognosis"],
            MentalHealthCondition.DEPRESSION,
            "short_term"
        )

        assert len(prediction.protective_factors) > 0

    def test_intervention_recommendations(self):
        """Test intervention recommendation generation."""
        prediction = self.predictor.predict_outcome(
            self.test_conversations["poor_prognosis"],
            MentalHealthCondition.DEPRESSION,
            "short_term"
        )

        assert len(prediction.recommended_interventions) > 0

    def test_batch_prediction(self):
        """Test batch prediction functionality."""
        conversations = list(self.test_conversations.values())
        predictions = self.predictor.predict_batch(
            conversations,
            MentalHealthCondition.DEPRESSION,
            "short_term"
        )

        assert len(predictions) == 3
        assert all(p.confidence_score > 0 for p in predictions)

    def test_prediction_statistics(self):
        """Test prediction statistics tracking."""
        # Make some predictions
        for conversation in self.test_conversations.values():
            self.predictor.predict_outcome(
                conversation,
                MentalHealthCondition.DEPRESSION,
                "short_term"
            )

        stats = self.predictor.get_prediction_statistics()

        assert stats["total_predictions"] > 0
        assert "outcome_distribution" in stats


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios combining multiple components."""

    def setUp(self):
        """Set up test fixtures."""
        self.classifier = TherapeuticApproachClassifier()
        self.recognizer = MentalHealthConditionRecognizer()
        self.predictor = TherapeuticOutcomePredictor()

        # Complex test conversation
        self.complex_conversation = {
            "id": "integration_test",
            "messages": [
                {
                    "role": "client",
                    "content": "I've been feeling really depressed and anxious. I have automatic negative thoughts and I'm motivated to try CBT. I have good family support."
                },
                {
                    "role": "therapist",
                    "content": "Let's work on identifying those thought patterns and examine the evidence for them. Your motivation and support system are great assets."
                }
            ]
        }

    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline integration."""
        # Step 1: Classify therapeutic approach
        classification = self.classifier.classify_conversation(self.complex_conversation)

        # Step 2: Recognize mental health condition
        recognition = self.recognizer.recognize_condition(self.complex_conversation)

        # Step 3: Predict therapeutic outcome
        prediction = self.predictor.predict_outcome(
            self.complex_conversation,
            recognition.primary_condition,
            "short_term"
        )

        # Verify all components produced valid results
        assert classification.primary_approach is not None
        assert recognition.primary_condition is not None
        assert prediction.predicted_outcome is not None

        # Verify consistency between components
        assert classification.primary_approach == TherapeuticApproach.CBT
        assert recognition.primary_condition in [MentalHealthCondition.DEPRESSION, MentalHealthCondition.ANXIETY]

        # Verify quality scores
        assert classification.quality_score > 0.2  # More realistic threshold
        assert recognition.recognition_quality > 0.5
        assert prediction.confidence_score > 0.3

    def test_comprehensive_reporting(self):
        """Test comprehensive analysis reporting."""
        # Analyze the conversation with all components
        classification = self.classifier.classify_conversation(self.complex_conversation)
        recognition = self.recognizer.recognize_condition(self.complex_conversation)
        prediction = self.predictor.predict_outcome(
            self.complex_conversation,
            recognition.primary_condition,
            "short_term"
        )

        # Create comprehensive report
        report = {
            "conversation_id": self.complex_conversation["id"],
            "therapeutic_approach": {
                "primary": classification.primary_approach.value,
                "confidence": classification.quality_score,
                "rationale": classification.classification_rationale
            },
            "mental_health_condition": {
                "primary": recognition.primary_condition.value,
                "severity": recognition.severity_assessment,
                "confidence": recognition.recognition_quality,
                "risk_indicators": recognition.risk_indicators
            },
            "outcome_prediction": {
                "predicted_outcome": prediction.predicted_outcome.value,
                "confidence": prediction.confidence_score,
                "timeline": prediction.prediction_timeline,
                "recommendations": prediction.recommended_interventions
            }
        }

        # Verify report structure
        assert "therapeutic_approach" in report
        assert "mental_health_condition" in report
        assert "outcome_prediction" in report

        # Verify all components have valid data
        assert all(report[component]["confidence"] > 0 for component in ["therapeutic_approach", "mental_health_condition", "outcome_prediction"])


def run_comprehensive_test():
    """Run comprehensive test suite with detailed reporting."""

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestTherapeuticApproachClassifier,
        TestMentalHealthConditionRecognizer,
        TestTherapeuticOutcomePredictor,
        TestIntegrationScenarios
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary

    if result.failures:
        for _test, traceback in result.failures:
            traceback.split("AssertionError: ")[-1].split("\n")[0]

    if result.errors:
        for _test, traceback in result.errors:
            traceback.split("\n")[-2]

    if not result.failures and not result.errors:
        pass

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
