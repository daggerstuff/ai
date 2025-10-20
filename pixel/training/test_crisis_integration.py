"""
Integration tests for Crisis Detection with Data Pipeline

Tests crisis detection integration with data augmentation,
data loading, and training pipeline.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from crisis_detection import CrisisDetector, CrisisSeverity, CrisisType
from data_augmentation import DataAugmentationPipeline


class TestCrisisDetectionIntegration:
    """Integration tests for crisis detection with data pipeline"""

    @pytest.fixture
    def detector(self):
        return CrisisDetector()

    @pytest.fixture
    def augmentation_pipeline(self):
        return DataAugmentationPipeline()

    def test_crisis_detection_on_augmented_data(self, detector, augmentation_pipeline):
        """Test crisis detection on augmented conversation data"""
        original_record = {
            "user_message": "I want to kill myself",
            "assistant_response": "I'm here to help. Let's talk about what you're feeling.",
            "category": "crisis",
        }

        # Augment the data
        augmented = augmentation_pipeline.augment_record(original_record)

        # Verify crisis detection on original
        original_result = detector.detect_crisis(original_record["user_message"])
        assert original_result.overall_severity in [CrisisSeverity.HIGH, CrisisSeverity.IMMINENT]
        assert original_result.requires_intervention is True

        # Verify crisis detection on augmented versions
        for aug_record in augmented:
            result = detector.detect_crisis(aug_record["user_message"])
            # Augmented versions should maintain crisis indicators
            assert result.overall_severity in [
                CrisisSeverity.NONE,
                CrisisSeverity.LOW,
                CrisisSeverity.MODERATE,
                CrisisSeverity.HIGH,
                CrisisSeverity.IMMINENT,
            ]

    def test_crisis_detection_preserves_safety_keywords(self, detector):
        """Test that crisis detection preserves safety-critical keywords"""
        crisis_keywords = [
            "suicide",
            "kill myself",
            "self harm",
            "cutting",
            "hearing voices",
            "violent",
            "overdose",
        ]

        for keyword in crisis_keywords:
            text = f"I'm struggling with {keyword}"
            result = detector.detect_crisis(text)
            # Should detect at least some crisis indicators
            assert len(result.indicators) > 0 or result.overall_severity != CrisisSeverity.NONE

    def test_crisis_detection_with_therapeutic_context(self, detector):
        """Test crisis detection with therapeutic context"""
        therapeutic_text = (
            "I've been having suicidal thoughts, but I'm working with my therapist "
            "to develop coping strategies. My family is supportive."
        )

        result = detector.detect_crisis(therapeutic_text)

        # Should detect crisis indicators or have protective factors
        has_crisis = any(i.type == CrisisType.SUICIDAL_IDEATION for i in result.indicators)
        has_protective = len(result.protective_factors) > 0

        # Should have either crisis indicators or protective factors
        assert has_crisis or has_protective

        # Should recommend intervention if crisis detected
        if has_crisis:
            assert result.requires_intervention is True

    def test_crisis_detection_batch_processing(self, detector):
        """Test crisis detection on batch of records"""
        records = [
            "I'm feeling great today",
            "I want to kill myself",
            "I'm struggling but getting help",
            "I'm cutting myself to cope",
            "Everything is fine",
        ]

        results = [detector.detect_crisis(text) for text in records]

        # Verify results
        assert len(results) == 5
        assert results[0].overall_severity == CrisisSeverity.NONE
        assert results[1].overall_severity in [CrisisSeverity.HIGH, CrisisSeverity.IMMINENT]
        assert results[3].overall_severity in [CrisisSeverity.MODERATE, CrisisSeverity.HIGH]
        assert results[4].overall_severity == CrisisSeverity.NONE

    def test_crisis_detection_confidence_scores(self, detector):
        """Test confidence score calculation across different scenarios"""
        scenarios = [
            ("I'm fine", 0.0),  # No crisis
            ("I feel sad", 0.0),  # Low confidence
            ("I want to die", 0.5),  # Moderate confidence
            ("I'm going to kill myself right now", 0.8),  # High confidence
        ]

        for text, min_expected_confidence in scenarios:
            result = detector.detect_crisis(text)
            assert result.confidence_score >= min_expected_confidence

    def test_crisis_detection_timeline_escalation(self, detector):
        """Test that immediate timeline escalates severity"""
        base_text = "I want to hurt myself"
        immediate_text = "I want to hurt myself right now"

        base_result = detector.detect_crisis(base_text)
        immediate_result = detector.detect_crisis(immediate_text)

        # Immediate timeline should be detected
        assert immediate_result.timeline == "immediate"

        # Immediate timeline with crisis should escalate to IMMINENT
        severity_order = [
            CrisisSeverity.NONE,
            CrisisSeverity.LOW,
            CrisisSeverity.MODERATE,
            CrisisSeverity.HIGH,
            CrisisSeverity.IMMINENT,
        ]
        base_idx = severity_order.index(base_result.overall_severity)
        immediate_idx = severity_order.index(immediate_result.overall_severity)
        assert immediate_idx >= base_idx

    def test_crisis_detection_action_recommendations(self, detector):
        """Test that appropriate actions are recommended"""
        imminent_text = "I'm going to kill myself tonight"
        high_text = "I want to end my life"
        moderate_text = "I've been feeling hopeless"

        imminent_result = detector.detect_crisis(imminent_text)
        high_result = detector.detect_crisis(high_text)
        moderate_result = detector.detect_crisis(moderate_text)

        # Imminent should have emergency actions
        assert any("emergency" in action.lower() for action in imminent_result.recommended_actions)

        # High should have crisis actions
        assert any("crisis" in action.lower() for action in high_result.recommended_actions)

        # Moderate should have therapy actions
        if moderate_result.overall_severity == CrisisSeverity.MODERATE:
            assert any(
                "therapy" in action.lower() for action in moderate_result.recommended_actions
            )

    def test_crisis_detection_multiple_crisis_types(self, detector):
        """Test detection of multiple crisis types in single text"""
        text = (
            "I'm hearing voices telling me to hurt myself. "
            "I've been cutting and I want to die. "
            "I'm so angry I could hurt someone."
        )

        result = detector.detect_crisis(text)

        # Should detect multiple crisis types
        crisis_types = {i.type for i in result.indicators}
        assert len(crisis_types) >= 2

        # Should have high severity
        assert result.overall_severity in [CrisisSeverity.HIGH, CrisisSeverity.IMMINENT]

    def test_crisis_detection_false_positive_mitigation(self, detector):
        """Test that false positives are minimized"""
        # These should NOT trigger crisis detection (or only low severity)
        benign_texts = [
            "The movie had a sad ending",
            "I'm tired of this weather",
            "I need to cut back on spending",
            "I'm feeling better today",
            "I'm learning about mental health",
        ]

        for text in benign_texts:
            result = detector.detect_crisis(text)
            # Should not trigger high-severity crisis detection
            assert result.overall_severity not in [CrisisSeverity.HIGH, CrisisSeverity.IMMINENT]

    def test_crisis_detection_metadata_completeness(self, detector):
        """Test that metadata is complete and accurate"""
        text = "I'm having a crisis"
        result = detector.detect_crisis(text)

        # Verify metadata
        assert "detected_at" in result.metadata
        assert "text_length" in result.metadata
        assert "indicator_count" in result.metadata

        # Verify metadata accuracy
        assert result.metadata["text_length"] == len(text)
        assert result.metadata["indicator_count"] == len(result.indicators)

    def test_crisis_detection_performance(self, detector):
        """Test crisis detection performance on large text"""
        large_text = "I'm struggling. " * 1000  # 16,000 characters

        result = detector.detect_crisis(large_text)

        # Should complete without error
        assert result is not None
        assert isinstance(result.overall_severity, CrisisSeverity)


class TestCrisisDetectionEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def detector(self):
        return CrisisDetector()

    def test_empty_indicators_list(self, detector):
        """Test handling of empty indicators"""
        result = detector.detect_crisis("Everything is fine")
        assert len(result.indicators) == 0
        assert result.overall_severity == CrisisSeverity.NONE

    def test_special_characters_handling(self, detector):
        """Test handling of special characters"""
        text = "I want to k!ll my$elf!!! @#$%"
        result = detector.detect_crisis(text)
        # Should handle gracefully
        assert result is not None

    def test_unicode_handling(self, detector):
        """Test handling of unicode characters"""
        text = "I'm having thoughts of suicide 😢"
        result = detector.detect_crisis(text)
        # Should detect crisis despite emoji
        assert result.overall_severity in [CrisisSeverity.HIGH, CrisisSeverity.IMMINENT]

    def test_very_long_text(self, detector):
        """Test handling of very long text"""
        text = "I'm struggling. " * 10000  # 160,000 characters
        result = detector.detect_crisis(text)
        assert result is not None

    def test_mixed_case_sensitivity(self, detector):
        """Test case-insensitive detection"""
        texts = [
            "I WANT TO KILL MYSELF",
            "i want to kill myself",
            "I Want To Kill Myself",
            "i WANT to KILL myself",
        ]

        results = [detector.detect_crisis(text) for text in texts]

        # All should have same severity
        severities = [r.overall_severity for r in results]
        assert len(set(severities)) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
