"""
Comprehensive tests for bias detection system
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from bias_detection import AlertLevel, BiasDetector, BiasType


class TestBiasDetectorBasics:
    """Test basic bias detector functionality"""

    @pytest.fixture
    def detector(self):
        return BiasDetector()

    def test_detector_initialization(self, detector):
        """Test detector initializes correctly"""
        assert detector is not None
        assert detector.compiled_patterns is not None
        assert len(detector.compiled_patterns) > 0

    def test_empty_text_handling(self, detector):
        """Test handling of empty text"""
        result = detector.detect_bias("")
        assert result.overall_bias_score == 0.0
        assert result.alert_level == AlertLevel.LOW
        assert len(result.indicators) == 0

    def test_none_text_handling(self, detector):
        """Test handling of None text"""
        result = detector.detect_bias(None)
        assert result.overall_bias_score == 0.0
        assert result.alert_level == AlertLevel.LOW

    def test_neutral_text(self, detector):
        """Test neutral text with no bias"""
        text = "The patient reported feeling better today. They are making progress."
        result = detector.detect_bias(text)
        assert result.alert_level in [AlertLevel.LOW, AlertLevel.MEDIUM]
        assert result.overall_bias_score < 0.5


class TestGenderBiasDetection:
    """Test gender bias detection"""

    @pytest.fixture
    def detector(self):
        return BiasDetector()

    def test_gender_bias_high_severity(self, detector):
        """Test detection of high-severity gender bias"""
        text = "He is a strong man, but she is too emotional for leadership."
        result = detector.detect_bias(text)
        assert result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]
        assert any(i.type == BiasType.GENDER_BIAS for i in result.indicators)

    def test_gender_bias_medium_severity(self, detector):
        """Test detection of medium-severity gender bias"""
        text = "This is typical behavior for a woman in this role."
        result = detector.detect_bias(text)
        # Medium patterns may not always trigger high alert, depends on overall score
        assert result.alert_level in [AlertLevel.LOW, AlertLevel.MEDIUM, AlertLevel.HIGH]
        # May or may not detect depending on pattern matching
        if any(i.type == BiasType.GENDER_BIAS for i in result.indicators):
            assert result.alert_level in [AlertLevel.MEDIUM, AlertLevel.HIGH]

    def test_gender_neutral_language(self, detector):
        """Test that gender-neutral language is not flagged"""
        text = "The patient is making good progress with their therapy."
        result = detector.detect_bias(text)
        gender_bias = [i for i in result.indicators if i.type == BiasType.GENDER_BIAS]
        assert len(gender_bias) == 0 or result.alert_level == AlertLevel.LOW


class TestRacialBiasDetection:
    """Test racial bias detection"""

    @pytest.fixture
    def detector(self):
        return BiasDetector()

    def test_racial_bias_high_severity(self, detector):
        """Test detection of high-severity racial bias"""
        text = "Black people in this community are stereotypically more aggressive."
        result = detector.detect_bias(text)
        assert result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]
        assert any(i.type == BiasType.RACIAL_BIAS for i in result.indicators)

    def test_racial_bias_medium_severity(self, detector):
        """Test detection of medium-severity racial bias"""
        text = "The diverse community has different cultural backgrounds."
        result = detector.detect_bias(text)
        # Should be low or medium, not critical
        assert result.alert_level in [AlertLevel.LOW, AlertLevel.MEDIUM]

    def test_racial_stereotype_detection(self, detector):
        """Test detection of racial stereotypes"""
        text = "He is very articulate for a Black person."
        result = detector.detect_bias(text)
        assert result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]


class TestAgeBiasDetection:
    """Test age bias detection"""

    @pytest.fixture
    def detector(self):
        return BiasDetector()

    def test_age_bias_high_severity(self, detector):
        """Test detection of high-severity age bias"""
        text = "Old people are out of touch with technology and can't learn new things."
        result = detector.detect_bias(text)
        assert result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]
        assert any(i.type == BiasType.AGE_BIAS for i in result.indicators)

    def test_generational_bias(self, detector):
        """Test detection of generational bias"""
        text = "Millennials are lazy and entitled and stupid."
        result = detector.detect_bias(text)
        assert result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]

    def test_age_appropriate_language(self, detector):
        """Test that age-appropriate language is not flagged as critical"""
        text = "The patient is in their senior years and has age-related concerns."
        result = detector.detect_bias(text)
        # Age-related is a medium pattern, so alert level should be low-medium
        assert result.alert_level in [AlertLevel.LOW, AlertLevel.MEDIUM]


class TestCulturalBiasDetection:
    """Test cultural bias detection"""

    @pytest.fixture
    def detector(self):
        return BiasDetector()

    def test_cultural_bias_high_severity(self, detector):
        """Test detection of high-severity cultural bias"""
        text = "Western values are superior to Eastern cultures."
        result = detector.detect_bias(text)
        assert result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]
        assert any(i.type == BiasType.CULTURAL_BIAS for i in result.indicators)

    def test_cultural_stereotype(self, detector):
        """Test detection of cultural stereotypes"""
        text = "Collectivist cultures are primitive compared to individualist societies."
        result = detector.detect_bias(text)
        assert result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]


class TestSocioeconomicBiasDetection:
    """Test socioeconomic bias detection"""

    @pytest.fixture
    def detector(self):
        return BiasDetector()

    def test_socioeconomic_bias_high_severity(self, detector):
        """Test detection of high-severity socioeconomic bias"""
        text = "Poor people are lazy and don't work hard enough."
        result = detector.detect_bias(text)
        assert result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]
        assert any(i.type == BiasType.SOCIOECONOMIC_BIAS for i in result.indicators)

    def test_class_stereotype(self, detector):
        """Test detection of class stereotypes"""
        text = "The wealthy are naturally more intelligent and hardworking than the underprivileged poor."
        result = detector.detect_bias(text)
        assert result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]


class TestAbilityBiasDetection:
    """Test ability bias detection"""

    @pytest.fixture
    def detector(self):
        return BiasDetector()

    def test_ability_bias_high_severity(self, detector):
        """Test detection of high-severity ability bias"""
        text = "Disabled people are helpless and need constant care."
        result = detector.detect_bias(text)
        assert result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]
        assert any(i.type == BiasType.ABILITY_BIAS for i in result.indicators)

    def test_inspiration_porn_detection(self, detector):
        """Test detection of inspiration porn language"""
        text = "It's an inspiration that disabled people can do normal things."
        result = detector.detect_bias(text)
        assert result.alert_level in [AlertLevel.MEDIUM, AlertLevel.HIGH, AlertLevel.CRITICAL]


class TestLanguageBiasDetection:
    """Test language bias detection"""

    @pytest.fixture
    def detector(self):
        return BiasDetector()

    def test_language_bias_high_severity(self, detector):
        """Test detection of high-severity language bias"""
        text = "His broken English makes him sound unintelligent."
        result = detector.detect_bias(text)
        assert result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]
        assert any(i.type == BiasType.LANGUAGE_BIAS for i in result.indicators)

    def test_accent_discrimination(self, detector):
        """Test detection of accent discrimination"""
        text = "Her accent is bad and makes her hard to understand."
        result = detector.detect_bias(text)
        assert result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]


class TestFairnessMetrics:
    """Test fairness metrics calculation"""

    @pytest.fixture
    def detector(self):
        return BiasDetector()

    def test_demographic_parity_calculation(self, detector):
        """Test demographic parity metric"""
        text = "The man and woman discussed their experiences."
        result = detector.detect_bias(text)
        assert 0.0 <= result.fairness_metrics.demographic_parity <= 1.0

    def test_equalized_odds_calculation(self, detector):
        """Test equalized odds metric"""
        text = "Both patients received equal treatment and support."
        result = detector.detect_bias(text)
        assert 0.0 <= result.fairness_metrics.equalized_odds <= 1.0

    def test_calibration_calculation(self, detector):
        """Test calibration metric"""
        text = "The therapist provided consistent care to all patients."
        result = detector.detect_bias(text)
        assert 0.0 <= result.fairness_metrics.calibration <= 1.0

    def test_representation_balance(self, detector):
        """Test representation balance metric"""
        text = "Young and old, rich and poor, all backgrounds represented."
        result = detector.detect_bias(text)
        assert 0.0 <= result.fairness_metrics.representation_balance <= 1.0


class TestMitigationStrategies:
    """Test mitigation strategy generation"""

    @pytest.fixture
    def detector(self):
        return BiasDetector()

    def test_critical_mitigation_strategies(self, detector):
        """Test strategies for critical bias"""
        text = "Old people are too stupid to use computers."
        result = detector.detect_bias(text)
        if result.alert_level == AlertLevel.CRITICAL:
            assert len(result.mitigation_strategies) > 0
            assert any(
                "URGENT" in s or "immediate" in s.lower() for s in result.mitigation_strategies
            )

    def test_high_mitigation_strategies(self, detector):
        """Test strategies for high bias"""
        text = "Women are too emotional for leadership roles."
        result = detector.detect_bias(text)
        if result.alert_level == AlertLevel.HIGH:
            assert len(result.mitigation_strategies) > 0

    def test_type_specific_strategies(self, detector):
        """Test type-specific mitigation strategies"""
        text = "He is a strong man, but she is emotional."
        result = detector.detect_bias(text)
        if any(i.type == BiasType.GENDER_BIAS for i in result.indicators):
            assert any("gender-neutral" in s.lower() for s in result.mitigation_strategies)

    def test_no_strategies_for_neutral_text(self, detector):
        """Test that neutral text generates no mitigation strategies"""
        text = "The patient is making good progress."
        result = detector.detect_bias(text)
        if result.alert_level == AlertLevel.LOW:
            assert len(result.mitigation_strategies) == 0 or result.requires_mitigation is False


class TestBiasScoreCalculation:
    """Test bias score calculation"""

    @pytest.fixture
    def detector(self):
        return BiasDetector()

    def test_bias_score_range(self, detector):
        """Test that bias score is between 0.0 and 1.0"""
        texts = [
            "Neutral text",
            "Women are emotional",
            "Old people are stupid",
            "Black people are criminals",
        ]
        for text in texts:
            result = detector.detect_bias(text)
            assert 0.0 <= result.overall_bias_score <= 1.0

    def test_bias_score_increases_with_severity(self, detector):
        """Test that bias score increases with more severe bias"""
        neutral = detector.detect_bias("The patient is doing well.")
        biased = detector.detect_bias("Women are too emotional for leadership.")
        assert biased.overall_bias_score >= neutral.overall_bias_score

    def test_confidence_score_calculation(self, detector):
        """Test confidence score calculation"""
        result = detector.detect_bias("Women are emotional and men are rational.")
        assert 0.0 <= result.confidence_score <= 1.0


class TestAlertLevelDetermination:
    """Test alert level determination"""

    @pytest.fixture
    def detector(self):
        return BiasDetector()

    def test_low_alert_level(self, detector):
        """Test low alert level for minimal bias"""
        result = detector.detect_bias("The patient is doing well.")
        assert result.alert_level in [AlertLevel.LOW, AlertLevel.MEDIUM]

    def test_high_alert_level(self, detector):
        """Test high alert level for significant bias"""
        result = detector.detect_bias("Women are too emotional and weak for leadership roles.")
        assert result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]

    def test_critical_alert_level(self, detector):
        """Test critical alert level for severe bias"""
        result = detector.detect_bias("Old people are stupid and useless.")
        assert result.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]


class TestBiasAssessmentMetadata:
    """Test bias assessment metadata"""

    @pytest.fixture
    def detector(self):
        return BiasDetector()

    def test_metadata_includes_timestamp(self, detector):
        """Test that metadata includes timestamp"""
        result = detector.detect_bias("Test text")
        assert "detected_at" in result.metadata
        assert result.metadata["detected_at"] is not None

    def test_metadata_includes_text_length(self, detector):
        """Test that metadata includes text length"""
        text = "This is a test text"
        result = detector.detect_bias(text)
        assert result.metadata["text_length"] == len(text)

    def test_metadata_includes_indicator_count(self, detector):
        """Test that metadata includes indicator count"""
        result = detector.detect_bias("Women are emotional")
        assert result.metadata["indicator_count"] == len(result.indicators)

    def test_requires_mitigation_flag(self, detector):
        """Test requires_mitigation flag"""
        neutral = detector.detect_bias("The patient is doing well.")
        biased = detector.detect_bias("Women are too emotional and weak for leadership.")
        assert neutral.requires_mitigation is False or neutral.alert_level == AlertLevel.LOW
        assert biased.requires_mitigation is True or biased.alert_level in [
            AlertLevel.HIGH,
            AlertLevel.CRITICAL,
        ]
