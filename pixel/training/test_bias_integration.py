"""
Integration tests for bias detection with other pipeline components
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from bias_detection import AlertLevel, BiasDetector, BiasType
from crisis_detection import CrisisDetector
from data_augmentation import DataAugmentationPipeline


class TestBiasDetectionIntegration:
    """Integration tests for bias detection system"""

    @pytest.fixture
    def bias_detector(self):
        return BiasDetector()

    @pytest.fixture
    def crisis_detector(self):
        return CrisisDetector()

    @pytest.fixture
    def augmentation_pipeline(self):
        return DataAugmentationPipeline()

    def test_bias_detection_on_augmented_data(self, bias_detector, augmentation_pipeline):
        """Test bias detection works on augmented data"""
        original_record = {
            "user_message": "I'm struggling with my identity",
            "assistant_response": "That's a common experience for young people.",
            "category": "identity",
        }

        augmented_list = augmentation_pipeline.augment_record(original_record)

        # Test bias detection on original
        original_result = bias_detector.detect_bias(original_record["assistant_response"])
        assert original_result is not None
        assert 0.0 <= original_result.overall_bias_score <= 1.0

        # Test bias detection on augmented records
        for augmented in augmented_list:
            augmented_result = bias_detector.detect_bias(augmented["assistant_response"])
            assert augmented_result is not None
            assert 0.0 <= augmented_result.overall_bias_score <= 1.0

    def test_bias_and_crisis_detection_together(self, bias_detector, crisis_detector):
        """Test bias and crisis detection work together"""
        text = "I want to kill myself because I'm old and useless."

        crisis_result = crisis_detector.detect_crisis(text)
        bias_result = bias_detector.detect_bias(text)

        # Should detect crisis
        assert crisis_result.overall_severity.value in ["high", "imminent"]

        # Should detect age bias
        assert any(i.type == BiasType.AGE_BIAS for i in bias_result.indicators)

    def test_batch_bias_detection(self, bias_detector):
        """Test batch processing of bias detection"""
        texts = [
            "The patient is doing well.",
            "Women are too emotional for leadership.",
            "Old people can't learn new things.",
            "Black people are criminals.",
        ]

        results = [bias_detector.detect_bias(text) for text in texts]

        assert len(results) == len(texts)
        assert all(0.0 <= r.overall_bias_score <= 1.0 for r in results)

        # First should be low bias
        assert results[0].alert_level in [AlertLevel.LOW, AlertLevel.MEDIUM]

        # Others should be higher bias
        assert all(r.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL] for r in results[1:])

    def test_bias_detection_preserves_crisis_context(self, bias_detector, crisis_detector):
        """Test that bias detection doesn't interfere with crisis detection"""
        crisis_text = "I'm going to kill myself"
        biased_crisis_text = "I'm going to kill myself because I'm old and stupid and useless"

        crisis_result1 = crisis_detector.detect_crisis(crisis_text)
        crisis_result2 = crisis_detector.detect_crisis(biased_crisis_text)

        # Both should detect crisis
        assert crisis_result1.overall_severity.value in ["high", "imminent"]
        assert crisis_result2.overall_severity.value in ["high", "imminent"]

        # Second should also detect bias
        bias_result = bias_detector.detect_bias(biased_crisis_text)
        assert len(bias_result.indicators) > 0 or bias_result.alert_level in [
            AlertLevel.HIGH,
            AlertLevel.CRITICAL,
        ]

    def test_demographic_context_in_bias_detection(self, bias_detector):
        """Test bias detection with demographic context"""
        text = "The patient is making good progress."
        demographics = {"age": "65", "gender": "female", "race": "Black", "socioeconomic": "low"}

        result = bias_detector.detect_bias(text, demographics)
        assert result.metadata["demographics_provided"] is True

    def test_bias_detection_on_therapeutic_dialogue(self, bias_detector):
        """Test bias detection on realistic therapeutic dialogue"""
        dialogue = """
        Therapist: How are you feeling today?
        Patient: I'm struggling with my identity as a woman in tech.
        Therapist: That's a common experience. Many women face challenges in tech fields.
        Patient: Yes, people assume I'm not technical because I'm a woman.
        Therapist: That's a form of gender bias. Let's explore how it affects you.
        """

        result = bias_detector.detect_bias(dialogue)
        assert result is not None
        assert 0.0 <= result.overall_bias_score <= 1.0

    def test_bias_detection_on_diverse_scenarios(self, bias_detector):
        """Test bias detection on diverse mental health scenarios"""
        scenarios = [
            ("Young patient", "I'm a teenager struggling with anxiety."),
            ("Elderly patient", "I'm an older adult dealing with depression."),
            ("Immigrant patient", "I'm struggling with cultural adjustment."),
            ("Low-income patient", "I can't afford therapy but I need help."),
            ("Disabled patient", "My disability affects my mental health."),
        ]

        for scenario_name, text in scenarios:
            result = bias_detector.detect_bias(text)
            assert result is not None
            assert 0.0 <= result.overall_bias_score <= 1.0

    def test_mitigation_strategies_are_actionable(self, bias_detector):
        """Test that mitigation strategies are actionable"""
        text = "Women are too emotional for leadership."
        result = bias_detector.detect_bias(text)

        if result.requires_mitigation:
            assert len(result.mitigation_strategies) > 0
            for strategy in result.mitigation_strategies:
                assert isinstance(strategy, str)
                assert len(strategy) > 0

    def test_confidence_score_reflects_detection_strength(self, bias_detector):
        """Test that confidence score reflects detection strength"""
        weak_bias = "The patient is young."
        strong_bias = "Young people are lazy and irresponsible."

        weak_result = bias_detector.detect_bias(weak_bias)
        strong_result = bias_detector.detect_bias(strong_bias)

        # Strong bias should have higher confidence
        assert strong_result.confidence_score >= weak_result.confidence_score


class TestBiasDetectionEdgeCases:
    """Test edge cases in bias detection"""

    @pytest.fixture
    def detector(self):
        return BiasDetector()

    def test_very_long_text(self, detector):
        """Test handling of very long text"""
        long_text = "The patient is doing well. " * 1000
        result = detector.detect_bias(long_text)
        assert result is not None
        assert 0.0 <= result.overall_bias_score <= 1.0

    def test_special_characters_and_unicode(self, detector):
        """Test handling of special characters and unicode"""
        texts = [
            "Patient: 你好 (Hello in Chinese)",
            "Therapist: Café, naïve, résumé",
            "Patient: I'm feeling @#$% today",
            "Therapist: Let's discuss your feelings... okay?",
        ]

        for text in texts:
            result = detector.detect_bias(text)
            assert result is not None

    def test_mixed_case_sensitivity(self, detector):
        """Test case-insensitive bias detection"""
        texts = [
            "WOMEN ARE EMOTIONAL",
            "Women are emotional",
            "women are emotional",
            "WoMeN aRe EmOtIoNaL",
        ]

        results = [detector.detect_bias(text) for text in texts]

        # All should detect similar bias levels
        bias_scores = [r.overall_bias_score for r in results]
        assert max(bias_scores) - min(bias_scores) < 0.2  # Similar scores

    def test_repeated_bias_patterns(self, detector):
        """Test handling of repeated bias patterns"""
        text = "Women are too emotional and weak. Women are irrational. Women are too weak for leadership."
        result = detector.detect_bias(text)

        assert result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]
        assert len(result.indicators) > 0

    def test_negated_bias_statements(self, detector):
        """Test handling of negated bias statements"""
        text = "It's not true that women are emotional."
        result = detector.detect_bias(text)

        # Should still detect the bias pattern even if negated
        # (because the pattern itself is present)
        assert result is not None

    def test_bias_in_quotes(self, detector):
        """Test detection of bias in quoted text"""
        text = 'The patient said: "Old people are useless."'
        result = detector.detect_bias(text)

        # Should detect bias even in quotes
        assert any(i.type == BiasType.AGE_BIAS for i in result.indicators)

    def test_multiple_bias_types_in_one_text(self, detector):
        """Test detection of multiple bias types"""
        text = "Old women from poor communities are lazy and emotional."
        result = detector.detect_bias(text)

        bias_types = {i.type for i in result.indicators}
        assert len(bias_types) >= 2  # Should detect multiple types

    def test_fairness_metrics_with_no_demographics(self, detector):
        """Test fairness metrics calculation without demographic context"""
        text = "The patient is making progress."
        result = detector.detect_bias(text)

        assert result.fairness_metrics is not None
        assert 0.0 <= result.fairness_metrics.demographic_parity <= 1.0
        assert 0.0 <= result.fairness_metrics.equalized_odds <= 1.0
        assert 0.0 <= result.fairness_metrics.calibration <= 1.0
        assert 0.0 <= result.fairness_metrics.representation_balance <= 1.0

    def test_performance_on_large_batch(self, detector):
        """Test performance on large batch of texts"""
        texts = [f"Patient {i}: I'm struggling with my mental health." for i in range(100)]

        results = [detector.detect_bias(text) for text in texts]
        assert len(results) == 100
        assert all(r is not None for r in results)

    def test_consistency_across_multiple_runs(self, detector):
        """Test that results are consistent across multiple runs"""
        text = "Women are too emotional for leadership."

        result1 = detector.detect_bias(text)
        result2 = detector.detect_bias(text)

        assert result1.overall_bias_score == result2.overall_bias_score
        assert result1.alert_level == result2.alert_level
        assert len(result1.indicators) == len(result2.indicators)


class TestBiasDetectionWithRealWorldData:
    """Test bias detection with realistic mental health data"""

    @pytest.fixture
    def detector(self):
        return BiasDetector()

    def test_depression_scenario_with_age_bias(self, detector):
        """Test depression scenario with age bias"""
        text = "That's common for older adults. Old people are lazy and should just accept it."
        result = detector.detect_bias(text)
        assert any(
            i.type == BiasType.AGE_BIAS for i in result.indicators
        ) or result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]

    def test_anxiety_scenario_with_gender_bias(self, detector):
        """Test anxiety scenario with gender bias"""
        text = "Women are naturally more anxious and too emotional than men."
        result = detector.detect_bias(text)
        assert any(
            i.type == BiasType.GENDER_BIAS for i in result.indicators
        ) or result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]

    def test_trauma_scenario_with_cultural_bias(self, detector):
        """Test trauma scenario with cultural bias"""
        text = """
        Patient: I experienced trauma in my home country.
        Therapist: Western therapy approaches are superior to your traditional methods.
        """
        result = detector.detect_bias(text)
        assert any(i.type == BiasType.CULTURAL_BIAS for i in result.indicators)

    def test_addiction_scenario_with_socioeconomic_bias(self, detector):
        """Test addiction scenario with socioeconomic bias"""
        text = """
        Patient: I'm struggling with substance abuse.
        Therapist: Poor people are more likely to become addicts.
        """
        result = detector.detect_bias(text)
        assert any(i.type == BiasType.SOCIOECONOMIC_BIAS for i in result.indicators)

    def test_disability_scenario_with_ability_bias(self, detector):
        """Test disability scenario with ability bias"""
        text = """
        Patient: I have a disability that affects my mental health.
        Therapist: It's inspiring that you can do normal things despite your disability.
        """
        result = detector.detect_bias(text)
        assert any(i.type == BiasType.ABILITY_BIAS for i in result.indicators)
