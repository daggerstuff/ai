"""
Comprehensive test suite for Crisis Detection System

Tests pattern matching, risk assessment, protective factors,
and response protocols.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from crisis_detection import CrisisDetector, CrisisSeverity, CrisisType


class TestCrisisDetectorBasics:
    """Test basic crisis detection functionality"""

    @pytest.fixture
    def detector(self):
        return CrisisDetector()

    def test_detector_initialization(self, detector):
        """Test detector initializes correctly"""
        assert detector is not None
        assert detector.compiled_patterns is not None
        assert len(detector.compiled_patterns) > 0

    def test_empty_text_handling(self, detector):
        """Test handling of empty text"""
        result = detector.detect_crisis("")
        assert result.overall_severity == CrisisSeverity.NONE
        assert len(result.indicators) == 0
        assert result.confidence_score == 0.0

    def test_none_text_handling(self, detector):
        """Test handling of None text"""
        result = detector.detect_crisis(None)
        assert result.overall_severity == CrisisSeverity.NONE
        assert result.requires_intervention is False


class TestSuicidalIdeationDetection:
    """Test suicidal ideation detection"""

    @pytest.fixture
    def detector(self):
        return CrisisDetector()

    def test_high_severity_suicidal_ideation(self, detector):
        """Test detection of high-severity suicidal ideation"""
        text = "I want to kill myself tonight"
        result = detector.detect_crisis(text)
        assert result.overall_severity in [CrisisSeverity.HIGH, CrisisSeverity.IMMINENT]
        assert any(i.type == CrisisType.SUICIDAL_IDEATION for i in result.indicators)

    def test_moderate_severity_suicidal_ideation(self, detector):
        """Test detection of moderate suicidal ideation"""
        text = "I wish I was dead, life has no point"
        result = detector.detect_crisis(text)
        assert result.overall_severity in [CrisisSeverity.MODERATE, CrisisSeverity.HIGH]

    def test_multiple_suicidal_indicators(self, detector):
        """Test detection with multiple indicators"""
        text = "I can't go on. I want to end my life. Better off dead."
        result = detector.detect_crisis(text)
        # Multiple high-severity matches should result in HIGH or IMMINENT
        assert result.overall_severity in [CrisisSeverity.HIGH, CrisisSeverity.IMMINENT]
        assert result.requires_intervention is True

    def test_suicidal_with_immediate_timeline(self, detector):
        """Test suicidal ideation with immediate timeline"""
        text = "I'm going to kill myself right now"
        result = detector.detect_crisis(text)
        assert result.timeline == "immediate"
        assert result.overall_severity == CrisisSeverity.IMMINENT


class TestSelfHarmDetection:
    """Test self-harm detection"""

    @pytest.fixture
    def detector(self):
        return CrisisDetector()

    def test_high_severity_self_harm(self, detector):
        """Test detection of high-severity self-harm"""
        text = "I've been cutting myself for weeks"
        result = detector.detect_crisis(text)
        assert any(i.type == CrisisType.SELF_HARM for i in result.indicators)

    def test_moderate_self_harm(self, detector):
        """Test detection of moderate self-harm"""
        text = "I feel numb and need to hurt myself to feel something"
        result = detector.detect_crisis(text)
        assert any(i.type == CrisisType.SELF_HARM for i in result.indicators)


class TestPsychosisDetection:
    """Test psychosis detection"""

    @pytest.fixture
    def detector(self):
        return CrisisDetector()

    def test_psychosis_detection(self, detector):
        """Test detection of psychotic symptoms"""
        text = "I'm hearing voices telling me to hurt myself"
        result = detector.detect_crisis(text)
        assert any(i.type == CrisisType.PSYCHOSIS for i in result.indicators)

    def test_paranoia_detection(self, detector):
        """Test detection of paranoid ideation"""
        text = "Everyone is watching me, they're following me"
        result = detector.detect_crisis(text)
        assert any(i.type == CrisisType.PSYCHOSIS for i in result.indicators)


class TestAgitationDetection:
    """Test agitation and violence detection"""

    @pytest.fixture
    def detector(self):
        return CrisisDetector()

    def test_agitation_detection(self, detector):
        """Test detection of agitation"""
        text = "I'm so angry I could hurt someone"
        result = detector.detect_crisis(text)
        assert any(i.type == CrisisType.AGITATION for i in result.indicators)

    def test_rage_detection(self, detector):
        """Test detection of rage"""
        text = "I'm in a rage and losing control"
        result = detector.detect_crisis(text)
        assert any(i.type == CrisisType.AGITATION for i in result.indicators)


class TestProtectiveFactors:
    """Test protective factor detection"""

    @pytest.fixture
    def detector(self):
        return CrisisDetector()

    def test_social_support_detection(self, detector):
        """Test detection of social support"""
        text = "My family and friends are supporting me through this"
        result = detector.detect_crisis(text)
        assert len(result.protective_factors) > 0

    def test_coping_skills_detection(self, detector):
        """Test detection of coping skills"""
        text = "I use meditation and breathing exercises to manage stress"
        result = detector.detect_crisis(text)
        assert len(result.protective_factors) > 0

    def test_hope_and_future_detection(self, detector):
        """Test detection of hope and future orientation"""
        text = "I have goals and dreams for my future"
        result = detector.detect_crisis(text)
        assert len(result.protective_factors) > 0


class TestTimelineDetection:
    """Test crisis timeline detection"""

    @pytest.fixture
    def detector(self):
        return CrisisDetector()

    def test_immediate_timeline(self, detector):
        """Test detection of immediate timeline"""
        text = "I'm going to do it right now"
        result = detector.detect_crisis(text)
        assert result.timeline == "immediate"

    def test_short_term_timeline(self, detector):
        """Test detection of short-term timeline"""
        text = "I'm planning to do it this week"
        result = detector.detect_crisis(text)
        assert result.timeline == "short_term"

    def test_ongoing_timeline(self, detector):
        """Test detection of ongoing crisis"""
        text = "I've been struggling with these thoughts"
        result = detector.detect_crisis(text)
        assert result.timeline == "ongoing"


class TestSeverityCalculation:
    """Test severity calculation logic"""

    @pytest.fixture
    def detector(self):
        return CrisisDetector()

    def test_no_indicators_severity(self, detector):
        """Test severity with no indicators"""
        text = "I'm having a good day"
        result = detector.detect_crisis(text)
        assert result.overall_severity == CrisisSeverity.NONE

    def test_single_moderate_indicator(self, detector):
        """Test severity with single moderate indicator"""
        text = "I feel sad and hopeless"
        result = detector.detect_crisis(text)
        assert result.overall_severity in [CrisisSeverity.MODERATE, CrisisSeverity.LOW]

    def test_multiple_high_indicators(self, detector):
        """Test severity with multiple high indicators"""
        text = "I want to kill myself and I'm cutting myself right now"
        result = detector.detect_crisis(text)
        assert result.overall_severity == CrisisSeverity.IMMINENT


class TestRecommendedActions:
    """Test recommended action generation"""

    @pytest.fixture
    def detector(self):
        return CrisisDetector()

    def test_imminent_actions(self, detector):
        """Test actions for imminent crisis"""
        text = "I'm going to kill myself right now"
        result = detector.detect_crisis(text)
        assert len(result.recommended_actions) > 0
        assert any("emergency" in action.lower() for action in result.recommended_actions)

    def test_high_severity_actions(self, detector):
        """Test actions for high severity"""
        text = "I want to end my life"
        result = detector.detect_crisis(text)
        assert len(result.recommended_actions) > 0
        assert any("crisis" in action.lower() for action in result.recommended_actions)

    def test_moderate_severity_actions(self, detector):
        """Test actions for moderate severity"""
        text = "I've been feeling hopeless lately"
        result = detector.detect_crisis(text)
        if result.overall_severity == CrisisSeverity.MODERATE:
            assert len(result.recommended_actions) > 0


class TestConfidenceScoring:
    """Test confidence score calculation"""

    @pytest.fixture
    def detector(self):
        return CrisisDetector()

    def test_confidence_with_no_indicators(self, detector):
        """Test confidence score with no indicators"""
        text = "I'm fine"
        result = detector.detect_crisis(text)
        assert result.confidence_score == 0.0

    def test_confidence_with_indicators(self, detector):
        """Test confidence score with indicators"""
        text = "I want to kill myself"
        result = detector.detect_crisis(text)
        assert result.confidence_score > 0.0
        assert result.confidence_score <= 1.0

    def test_confidence_increases_with_multiple_indicators(self, detector):
        """Test confidence increases with multiple indicators"""
        text1 = "I want to die"
        text2 = "I want to die and I'm cutting myself"
        result1 = detector.detect_crisis(text1)
        result2 = detector.detect_crisis(text2)
        assert result2.confidence_score >= result1.confidence_score


class TestIntegration:
    """Integration tests for crisis detection"""

    @pytest.fixture
    def detector(self):
        return CrisisDetector()

    def test_complex_scenario_with_protective_factors(self, detector):
        """Test complex scenario with both risk and protective factors"""
        text = (
            "I've been having suicidal thoughts, but my family is supporting me. "
            "I'm going to therapy and using coping strategies."
        )
        result = detector.detect_crisis(text)
        assert len(result.indicators) > 0
        assert len(result.protective_factors) > 0

    def test_case_insensitivity(self, detector):
        """Test case-insensitive detection"""
        text1 = "I want to KILL MYSELF"
        text2 = "i want to kill myself"
        result1 = detector.detect_crisis(text1)
        result2 = detector.detect_crisis(text2)
        assert result1.overall_severity == result2.overall_severity

    def test_metadata_generation(self, detector):
        """Test metadata is properly generated"""
        text = "I'm having a crisis"
        result = detector.detect_crisis(text)
        assert "detected_at" in result.metadata
        assert "text_length" in result.metadata
        assert "indicator_count" in result.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
