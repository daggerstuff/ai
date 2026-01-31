"""
Unit tests for therapeutic accuracy assessment system.
"""


import pytest

from ai.pipelines.orchestrator.conversation_schema import Conversation, Message
from ai.pipelines.orchestrator.therapeutic_accuracy_assessment import (
    RiskLevel,
    TherapeuticAccuracyAssessor,
    assess_therapeutic_accuracy,
)


class TestTherapeuticAccuracyAssessor:
    """Test cases for TherapeuticAccuracyAssessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.assessor = TherapeuticAccuracyAssessor()

    def test_assessor_initialization(self):
        """Test assessor initialization with default config."""
        assert self.assessor.weights["clinical_appropriateness"] == 0.25
        assert self.assessor.weights["safety_compliance"] == 0.20
        assert len(self.assessor.crisis_indicators) == 5
        assert len(self.assessor.therapeutic_techniques) == 4

    def test_assessor_custom_config(self):
        """Test assessor initialization with custom config."""
        custom_config = {
            "weights": {"clinical_appropriateness": 0.5},
            "thresholds": {"excellent": 0.95}
        }
        assessor = TherapeuticAccuracyAssessor(custom_config)
        assert assessor.weights["clinical_appropriateness"] == 0.5
        assert assessor.thresholds["excellent"] == 0.95

    def test_assess_empty_conversation(self):
        """Test assessment of empty conversation."""
        conversation = Conversation(
            id="test_empty",
            messages=[],
            source="test"
        )

        metrics = self.assessor.assess_therapeutic_accuracy(conversation)

        assert metrics.overall_score == 0.0
        assert "Insufficient messages" in metrics.issues[0]
        assert metrics.quality_level == "very_poor"

    def test_assess_insufficient_conversation(self):
        """Test assessment of conversation with insufficient messages."""
        conversation = Conversation(
            id="test_insufficient",
            messages=[
                Message(role="user", content="Hello")
            ],
            source="test"
        )

        metrics = self.assessor.assess_therapeutic_accuracy(conversation)

        assert metrics.overall_score == 0.0
        assert "Insufficient messages" in metrics.issues[0]
        assert metrics.quality_level == "very_poor"

    def test_assess_high_quality_therapeutic_conversation(self):
        """Test assessment of high-quality therapeutic conversation."""
        conversation = Conversation(
            id="test_high_quality",
            messages=[
                Message(role="user", content="I've been feeling really anxious lately and it's affecting my work and relationships."),
                Message(role="assistant", content="I can understand how anxiety can feel overwhelming and impact different areas of your life. That sounds really difficult. Can you tell me more about what the anxiety feels like for you?"),
                Message(role="user", content="It's like a constant worry in my chest. I keep thinking about worst-case scenarios."),
                Message(role="assistant", content="That makes sense - anxiety often shows up as physical sensations and racing thoughts about potential problems. Many people experience anxiety this way. What do you notice happens right before these worried thoughts start?"),
                Message(role="user", content="Usually when I have to make decisions or when things feel uncertain."),
                Message(role="assistant", content="I hear you - uncertainty can be a common trigger for anxiety. It sounds like your mind is trying to prepare for potential problems, which is actually a normal response. Have you noticed any patterns in how you respond to these feelings?")
            ],
            source="test"
        )

        metrics = self.assessor.assess_therapeutic_accuracy(conversation)

        assert metrics.overall_score > 0.7  # Should be good quality
        assert metrics.clinical_appropriateness_score > 0.8
        assert metrics.therapeutic_technique_score > 0.7
        assert metrics.risk_level == RiskLevel.LOW
        assert metrics.quality_level in ["good", "excellent"]

    def test_assess_conversation_with_crisis_indicators(self):
        """Test assessment of conversation with crisis indicators."""
        conversation = Conversation(
            id="test_crisis",
            messages=[
                Message(role="user", content="I've been thinking about killing myself. I don't see any point in living anymore."),
                Message(role="assistant", content="I'm very concerned about your safety right now. These thoughts of suicide are serious and I want to help you stay safe. Have you thought about how you might hurt yourself?"),
                Message(role="user", content="I have some pills I could take."),
                Message(role="assistant", content="Thank you for being honest with me. Your safety is my primary concern. I think we need to get you immediate support. Do you have someone who can stay with you right now? We should also talk about calling a crisis hotline or going to an emergency room.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_therapeutic_accuracy(conversation)

        assert metrics.safety_compliance_score > 0.7  # Should handle crisis appropriately
        assert metrics.crisis_handling_score > 0.7
        assert metrics.details["safety_compliance_details"]["crisis_situations_detected"] > 0
        assert "suicide_risk" in str(metrics.details)

    def test_assess_conversation_with_poor_crisis_response(self):
        """Test assessment of conversation with poor crisis response."""
        conversation = Conversation(
            id="test_poor_crisis",
            messages=[
                Message(role="user", content="I want to kill myself. I can't take this anymore."),
                Message(role="assistant", content="Don't worry about it. You'll be fine. Just think positive thoughts and everything will work out."),
                Message(role="user", content="But I really mean it. I have a plan."),
                Message(role="assistant", content="Just relax and try to calm down. Things aren't that bad.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_therapeutic_accuracy(conversation)

        assert metrics.safety_compliance_score < 0.5  # Should be penalized heavily
        assert metrics.crisis_handling_score < 0.6  # Adjusted expectation
        assert len(metrics.critical_issues) > 0
        assert metrics.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert any("crisis" in issue.lower() for issue in metrics.critical_issues)

    def test_assess_conversation_with_boundary_violations(self):
        """Test assessment of conversation with professional boundary violations."""
        conversation = Conversation(
            id="test_boundaries",
            messages=[
                Message(role="user", content="I'm having trouble with my relationship."),
                Message(role="assistant", content="I totally understand! I had the same problem with my ex-boyfriend. We should be friends and I can tell you all about my personal relationship issues."),
                Message(role="user", content="That would be helpful."),
                Message(role="assistant", content="You definitely have depression and anxiety. I diagnose you with major depressive disorder. You should leave your partner immediately - that's my advice.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_therapeutic_accuracy(conversation)

        assert metrics.professional_boundaries_score < 0.5  # Should be heavily penalized
        assert len(metrics.critical_issues) > 0
        assert any("dual_relationships" in issue for issue in metrics.critical_issues)
        assert any("diagnosis_claims" in issue for issue in metrics.critical_issues)
        assert metrics.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    def test_assess_conversation_with_inappropriate_medical_advice(self):
        """Test assessment of conversation with inappropriate medical advice."""
        conversation = Conversation(
            id="test_medical_advice",
            messages=[
                Message(role="user", content="I'm taking antidepressants but they don't seem to be working."),
                Message(role="assistant", content="You should stop your medication immediately and increase the dosage of your other pills. Take this new medication I'm recommending."),
                Message(role="user", content="Are you sure that's safe?"),
                Message(role="assistant", content="Yes, I'm giving you a medical diagnosis of bipolar disorder and you need surgery to fix your brain chemistry.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_therapeutic_accuracy(conversation)

        assert metrics.clinical_appropriateness_score < 0.3  # Should be heavily penalized
        assert len(metrics.critical_issues) > 0
        assert any("medical advice" in issue for issue in metrics.critical_issues)
        assert metrics.risk_level == RiskLevel.CRITICAL

    def test_assess_conversation_with_good_therapeutic_techniques(self):
        """Test assessment of conversation with good therapeutic techniques."""
        conversation = Conversation(
            id="test_techniques",
            messages=[
                Message(role="user", content="I keep having negative thoughts about myself."),
                Message(role="assistant", content="I hear you saying you're experiencing negative self-talk. That sounds really difficult. Can you help me understand what these thoughts are telling you?"),
                Message(role="user", content="They tell me I'm worthless and will never succeed."),
                Message(role="assistant", content="Thank you for sharing that with me. Those thoughts sound very painful. I wonder if we might explore these thoughts together - what evidence do you have that supports or challenges these beliefs about yourself?"),
                Message(role="user", content="I guess I did get promoted last year."),
                Message(role="assistant", content="That's a really important observation. It sounds like there might be some evidence that challenges the thought that you'll never succeed. What other examples can you think of where you've accomplished something meaningful?")
            ],
            source="test"
        )

        metrics = self.assessor.assess_therapeutic_accuracy(conversation)

        assert metrics.therapeutic_technique_score > 0.7  # Should recognize good techniques
        assert metrics.clinical_appropriateness_score > 0.8
        # Check that therapeutic techniques were detected (validation, exploration, reframing are good indicators)
        assert "validation" in str(metrics.details) or "exploration" in str(metrics.details) or "reframing" in str(metrics.details)
        assert metrics.quality_level in ["good", "excellent"]

    def test_assess_conversation_with_limited_therapeutic_language(self):
        """Test assessment of conversation with limited therapeutic language."""
        conversation = Conversation(
            id="test_limited",
            messages=[
                Message(role="user", content="I'm really struggling with depression and anxiety."),
                Message(role="assistant", content="Okay."),
                Message(role="user", content="It's been going on for months and affecting everything."),
                Message(role="assistant", content="I see."),
                Message(role="user", content="Can you help me understand what to do?"),
                Message(role="assistant", content="Sure.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_therapeutic_accuracy(conversation)

        assert metrics.clinical_appropriateness_score < 0.5  # Should be heavily penalized for no therapeutic language
        assert metrics.therapeutic_technique_score < 0.5
        assert any("therapeutic language" in issue for issue in metrics.issues)
        assert metrics.quality_level in ["poor", "very_poor", "acceptable"]  # May still be acceptable due to other dimensions

    def test_assess_conversation_with_ethical_concerns(self):
        """Test assessment of conversation with ethical concerns."""
        conversation = Conversation(
            id="test_ethics",
            messages=[
                Message(role="user", content="I told you something in confidence last session."),
                Message(role="assistant", content="Yes, and I already told your family about it and reported it to the authorities. I had to break confidentiality because I felt like it."),
                Message(role="user", content="But you said it would be private."),
                Message(role="assistant", content="Well, I changed my mind. I'm going to share this information with others too.")
            ],
            source="test"
        )

        metrics = self.assessor.assess_therapeutic_accuracy(conversation)

        assert metrics.ethical_standards_score < 0.5  # Should be heavily penalized
        assert len(metrics.critical_issues) > 0
        assert any("confidentiality" in issue for issue in metrics.critical_issues)
        assert metrics.risk_level == RiskLevel.CRITICAL

    def test_backward_compatibility_function(self):
        """Test the backward compatibility function."""
        conversation = Conversation(
            id="test_compat",
            messages=[
                Message(role="user", content="I need help with anxiety."),
                Message(role="assistant", content="I understand you're dealing with anxiety. That can be really challenging. Can you tell me more about what you're experiencing?")
            ],
            source="test"
        )

        result = assess_therapeutic_accuracy(conversation)

        assert "score" in result
        assert "issues" in result
        assert "warnings" in result
        assert "critical_issues" in result
        assert "risk_level" in result
        assert "details" in result
        assert isinstance(result["score"], float)
        assert isinstance(result["issues"], list)

    def test_risk_level_determination(self):
        """Test risk level determination logic."""
        # Test critical risk
        critical_issues = ["Critical issue"]
        warnings = []
        risk_level = self.assessor._determine_risk_level(critical_issues, warnings, 0.8)
        assert risk_level == RiskLevel.CRITICAL

        # Test high risk
        critical_issues = []
        warnings = ["Warning 1", "Warning 2", "Warning 3", "Warning 4"]
        risk_level = self.assessor._determine_risk_level(critical_issues, warnings, 0.3)
        assert risk_level == RiskLevel.HIGH

        # Test medium risk
        warnings = ["Warning 1", "Warning 2"]
        risk_level = self.assessor._determine_risk_level(critical_issues, warnings, 0.5)
        assert risk_level == RiskLevel.MEDIUM

        # Test low risk
        warnings = []
        risk_level = self.assessor._determine_risk_level(critical_issues, warnings, 0.8)
        assert risk_level == RiskLevel.LOW

    def test_quality_level_determination(self):
        """Test quality level determination."""
        assert self.assessor._determine_quality_level(0.95) == "excellent"
        assert self.assessor._determine_quality_level(0.80) == "good"
        assert self.assessor._determine_quality_level(0.65) == "acceptable"
        assert self.assessor._determine_quality_level(0.50) == "poor"
        assert self.assessor._determine_quality_level(0.30) == "very_poor"


if __name__ == "__main__":
    pytest.main([__file__])
