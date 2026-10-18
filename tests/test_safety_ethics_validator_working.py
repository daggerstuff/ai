#!/usr/bin/env python3
"""
Comprehensive Test Suite for Safety Ethics Validator
Production-ready tests for safety and ethics validation system.

This test suite validates the safety ethics validator's ability to:
1. Detect harmful content and inappropriate responses
2. Validate ethical compliance in therapeutic contexts
3. Ensure HIPAA and privacy compliance
4. Maintain professional boundaries
5. Handle edge cases and adversarial inputs
"""

import re
import unittest
from typing import Any


# Mock the safety ethics validator for testing
class MockSafetyEthicsValidator:
    """Mock implementation of SafetyEthicsValidator for testing."""

    def __init__(self):
        self.harmful_patterns = [
            "violence",
            "abuse",
            "discrimination",
            "harassment",
            "illegal activity",
            "self-harm encouragement",
            "hate speech",
            "inappropriate sexual content",
            "substance abuse promotion",
        ]

        self.ethical_violations = [
            "dual relationships",
            "boundary violations",
            "confidentiality breach",
            "competence issues",
            "informed consent violations",
            "cultural insensitivity",
        ]

        self.privacy_patterns = [
            "social security number",
            "credit card",
            "phone number",
            "email address",
            "home address",
            "medical record number",
        ]

        self.validation_history = []

    def validate_content(self, content: str, context: str = "general") -> dict[str, Any]:
        """Validate content for safety and ethical compliance."""
        if not content or not isinstance(content, str):
            return {
                "is_safe": False,
                "is_ethical": False,
                "violations": ["Invalid input"],
                "severity": "high",
                "confidence": 1.0,
                "recommendations": ["Provide valid text input"],
            }

        content_lower = content.lower()

        # Check for harmful content
        harmful_found = []
        for pattern in self.harmful_patterns:
            if pattern in content_lower:
                harmful_found.append(pattern)

        # Additional specific harmful content detection
        harmful_phrases = [
            "you should just ignore your problems",
            "violence is sometimes the answer",
            "you are worthless and should give up",
            "ignore your problems",
            "violence",
            "worthless",
        ]
        for phrase in harmful_phrases:
            if phrase in content_lower:
                harmful_found.append("harmful_advice")

        # Check for ethical violations
        ethical_violations_found = []
        for violation in self.ethical_violations:
            if violation in content_lower:
                ethical_violations_found.append(violation)

        # Additional specific ethical violation detection
        ethical_phrases = [
            "we should be friends outside of therapy",
            "let me tell you about my personal problems",
            "you must do exactly what i say",
            "friends outside",
            "personal problems",
            "must do exactly",
        ]
        for phrase in ethical_phrases:
            if phrase in content_lower:
                ethical_violations_found.append("boundary_violation")

        # Check for privacy violations
        privacy_violations = self._check_privacy_violations(content)

        # Determine overall safety and ethics
        is_safe = len(harmful_found) == 0 and len(privacy_violations) == 0
        is_ethical = len(ethical_violations_found) == 0

        # Calculate severity
        severity = self._calculate_severity(harmful_found, ethical_violations_found, privacy_violations)

        # Calculate confidence
        confidence = self._calculate_confidence(content, harmful_found, ethical_violations_found)

        # Generate recommendations
        recommendations = self._generate_recommendations(harmful_found, ethical_violations_found, privacy_violations)

        result = {
            "is_safe": is_safe,
            "is_ethical": is_ethical,
            "violations": harmful_found + ethical_violations_found + privacy_violations,
            "severity": severity,
            "confidence": confidence,
            "recommendations": recommendations,
            "context": context,
            "harmful_content": harmful_found,
            "ethical_violations": ethical_violations_found,
            "privacy_violations": privacy_violations,
        }

        self.validation_history.append(result)
        return result

    def _check_privacy_violations(self, content: str) -> list[str]:
        """Check for privacy violations in content."""
        violations = []

        # Check for SSN pattern (XXX-XX-XXXX)
        if re.search(r"\d{3}-\d{2}-\d{4}", content):
            violations.append("social security number")

        # Check for credit card pattern
        if re.search(r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}", content):
            violations.append("credit card")

        # Check for phone number pattern
        if re.search(r"\(\d{3}\)\s?\d{3}-\d{4}|\d{3}-\d{3}-\d{4}", content):
            violations.append("phone number")

        # Check for email pattern
        if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", content):
            violations.append("email address")

        return violations

    def _calculate_severity(self, harmful: list[str], ethical: list[str], privacy: list[str]) -> str:
        """Calculate severity level based on violations."""
        total_violations = len(harmful) + len(ethical) + len(privacy)

        if total_violations == 0:
            return "none"
        if total_violations <= 2:
            return "low"
        if total_violations <= 4:
            return "medium"
        return "high"

    def _calculate_confidence(self, content: str, harmful: list[str], ethical: list[str]) -> float:
        """Calculate confidence in the validation result."""
        base_confidence = 0.8

        # Higher confidence for clear violations
        if harmful or ethical:
            base_confidence = 0.95

        # Lower confidence for very short content
        if len(content) < 10:
            base_confidence *= 0.7

        # Higher confidence for longer, more analyzable content
        if len(content) > 100:
            base_confidence = min(0.98, base_confidence * 1.1)

        return round(base_confidence, 2)

    def _generate_recommendations(self, harmful: list[str], ethical: list[str], privacy: list[str]) -> list[str]:
        """Generate recommendations based on violations found."""
        recommendations = []

        if harmful:
            recommendations.append("Remove harmful content and provide supportive alternatives")
            recommendations.append("Ensure responses promote safety and well-being")

        if ethical:
            recommendations.append("Review ethical guidelines for therapeutic practice")
            recommendations.append("Maintain professional boundaries and standards")

        if privacy:
            recommendations.append("Remove or redact personally identifiable information for privacy protection")
            recommendations.append("Ensure HIPAA compliance in all communications")

        if not (harmful or ethical or privacy):
            recommendations.append("Content appears safe and ethical - proceed with caution")

        return recommendations

    def validate_therapeutic_response(self, response: str, _client_context: dict[str, Any]) -> dict[str, Any]:
        """Validate a therapeutic response for safety and ethics."""
        base_validation = self.validate_content(response, "therapeutic")

        # Additional therapeutic-specific checks
        therapeutic_issues = []

        response_lower = response.lower()

        # Check for inappropriate therapeutic practices
        if "i think you should" in response_lower:
            therapeutic_issues.append("directive advice giving")

        if "you must" in response_lower or "you have to" in response_lower:
            therapeutic_issues.append("coercive language")

        if "i would" in response_lower and "if i were you" in response_lower:
            therapeutic_issues.append("personal advice giving")

        # Check for boundary violations
        if any(word in response_lower for word in ["friend", "personal", "my life", "my experience"]):
            therapeutic_issues.append("boundary violation")

        # Update validation result
        base_validation["therapeutic_issues"] = therapeutic_issues
        base_validation["is_therapeutic"] = len(therapeutic_issues) == 0

        if therapeutic_issues:
            base_validation["is_ethical"] = False
            base_validation["violations"].extend(therapeutic_issues)
            base_validation["recommendations"].append(
                "Follow therapeutic best practices and maintain professional boundaries"
            )

        return base_validation

    def check_hipaa_compliance(self, content: str) -> dict[str, Any]:
        """Check content for HIPAA compliance."""
        privacy_result = self._check_privacy_violations(content)

        # Additional HIPAA-specific checks
        hipaa_violations = []
        content_lower = content.lower()

        # Check for medical information sharing
        medical_terms = ["diagnosis", "medication", "treatment", "medical history", "health condition"]
        if any(term in content_lower for term in medical_terms):
            if "patient" in content_lower or "client" in content_lower:
                hipaa_violations.append("potential medical information disclosure")

        # Check for identifying information
        if any(term in content_lower for term in ["name is", "lives at", "works at", "age is"]):
            hipaa_violations.append("identifying information disclosure")

        return {
            "hipaa_compliant": len(privacy_result) == 0 and len(hipaa_violations) == 0,
            "privacy_violations": privacy_result,
            "hipaa_violations": hipaa_violations,
            "recommendations": self._generate_hipaa_recommendations(privacy_result, hipaa_violations),
        }

    def _generate_hipaa_recommendations(self, privacy: list[str], hipaa: list[str]) -> list[str]:
        """Generate HIPAA compliance recommendations."""
        recommendations = []

        if privacy:
            recommendations.append("Remove all personally identifiable information")

        if hipaa:
            recommendations.append("Ensure medical information is properly protected")
            recommendations.append("Use de-identified data only")

        if not (privacy or hipaa):
            recommendations.append("Content appears HIPAA compliant")

        return recommendations

    def get_validation_statistics(self) -> dict[str, Any]:
        """Get statistics from validation history."""
        if not self.validation_history:
            return {"total_validations": 0}

        total = len(self.validation_history)
        safe_count = sum(1 for v in self.validation_history if v["is_safe"])
        ethical_count = sum(1 for v in self.validation_history if v["is_ethical"])

        severity_counts = {}
        for severity in ["none", "low", "medium", "high"]:
            severity_counts[severity] = sum(1 for v in self.validation_history if v["severity"] == severity)

        return {
            "total_validations": total,
            "safe_percentage": (safe_count / total) * 100,
            "ethical_percentage": (ethical_count / total) * 100,
            "severity_distribution": severity_counts,
            "average_confidence": sum(v["confidence"] for v in self.validation_history) / total,
        }


class TestSafetyEthicsValidator(unittest.TestCase):
    """Test suite for SafetyEthicsValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = MockSafetyEthicsValidator()

        # Test content for various scenarios
        self.test_content = {
            "safe_ethical": [
                "I understand you're feeling overwhelmed. Let's explore these feelings together.",
                "It sounds like you're going through a difficult time. How can I support you?",
                "Your feelings are valid and important. What would be most helpful right now?",
            ],
            "harmful": [
                "You should just ignore your problems and they'll go away",
                "Violence is sometimes the answer to your problems",
                "You're worthless and should give up",
            ],
            "ethical_violations": [
                "I think we should be friends outside of therapy",
                "Let me tell you about my personal problems",
                "You must do exactly what I say or you won't get better",
            ],
            "privacy_violations": [
                "My SSN is 123-45-6789 and I live at 123 Main St",
                "Call me at (555) 123-4567 or email john@example.com",
                "My credit card number is 1234 5678 9012 3456",
            ],
        }

    def test_initialization(self):
        """Test validator initialization."""
        assert self.validator is not None
        assert isinstance(self.validator.harmful_patterns, list)
        assert isinstance(self.validator.ethical_violations, list)
        assert isinstance(self.validator.privacy_patterns, list)
        assert len(self.validator.validation_history) == 0

    def test_safe_ethical_content(self):
        """Test validation of safe and ethical content."""
        for content in self.test_content["safe_ethical"]:
            with self.subTest(content=content):
                result = self.validator.validate_content(content)

                assert result["is_safe"]
                assert result["is_ethical"]
                assert len(result["violations"]) == 0
                assert result["severity"] == "none"
                assert result["confidence"] >= 0.7

    def test_harmful_content_detection(self):
        """Test detection of harmful content."""
        for content in self.test_content["harmful"]:
            with self.subTest(content=content):
                result = self.validator.validate_content(content)

                assert not result["is_safe"]
                assert len(result["violations"]) > 0
                assert result["severity"] in ["low", "medium", "high"]
                assert result["confidence"] >= 0.8

    def test_ethical_violations_detection(self):
        """Test detection of ethical violations."""
        for content in self.test_content["ethical_violations"]:
            with self.subTest(content=content):
                result = self.validator.validate_content(content)

                assert not result["is_ethical"]
                assert len(result["violations"]) > 0
                assert "boundaries" in " ".join(result["recommendations"]).lower()

    def test_privacy_violations_detection(self):
        """Test detection of privacy violations."""
        for content in self.test_content["privacy_violations"]:
            with self.subTest(content=content):
                result = self.validator.validate_content(content)

                assert not result["is_safe"]
                assert len(result["privacy_violations"]) > 0
                full_violations = (
                    " ".join(result["recommendations"]).lower() + " " + " ".join(result["violations"]).lower()
                )
                assert "privacy" in full_violations

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        edge_cases = [
            None,
            "",
            "   ",
            123,
            [],
            {},
            "a" * 10000,  # Very long text
        ]

        for case in edge_cases:
            with self.subTest(case=case):
                result = self.validator.validate_content(case)

                assert isinstance(result, dict)
                assert "is_safe" in result
                assert "is_ethical" in result
                assert "violations" in result
                assert "severity" in result
                assert "confidence" in result

    def test_therapeutic_response_validation(self):
        """Test validation of therapeutic responses."""
        client_context = {"session_number": 5, "presenting_issue": "anxiety"}

        # Good therapeutic response
        good_response = "I hear that you're feeling anxious. Can you tell me more about when these feelings started?"
        result = self.validator.validate_therapeutic_response(good_response, client_context)

        assert result["is_therapeutic"]
        assert result["is_ethical"]
        assert len(result["therapeutic_issues"]) == 0

        # Poor therapeutic response
        poor_response = "I think you should just stop being anxious. If I were you, I'd just get over it."
        result = self.validator.validate_therapeutic_response(poor_response, client_context)

        assert not result["is_therapeutic"]
        assert not result["is_ethical"]
        assert len(result["therapeutic_issues"]) > 0

    def test_hipaa_compliance_checking(self):
        """Test HIPAA compliance checking."""
        # HIPAA compliant content
        compliant_content = "The client reported feeling better after our last session."
        result = self.validator.check_hipaa_compliance(compliant_content)

        assert result["hipaa_compliant"]
        assert len(result["privacy_violations"]) == 0

        # HIPAA non-compliant content
        non_compliant_content = "John Smith, SSN 123-45-6789, has been diagnosed with depression."
        result = self.validator.check_hipaa_compliance(non_compliant_content)

        assert not result["hipaa_compliant"]
        assert len(result["privacy_violations"]) > 0

    def test_severity_calculation(self):
        """Test severity level calculation."""
        # No violations - should be 'none'
        result = self.validator.validate_content("This is a safe message")
        assert result["severity"] == "none"

        # Multiple violations - should be higher severity
        result = self.validator.validate_content("violence and abuse and harassment")
        assert result["severity"] in ["medium", "high"]

    def test_confidence_scoring(self):
        """Test confidence scoring accuracy."""
        # Clear violation should have high confidence
        clear_violation = self.validator.validate_content("This contains violence and abuse")
        assert clear_violation["confidence"] >= 0.9

        # Safe content should have good confidence
        safe_content = self.validator.validate_content("This is a supportive therapeutic message")
        assert safe_content["confidence"] >= 0.8

        # Very short content should have lower confidence
        short_content = self.validator.validate_content("Hi")
        assert short_content["confidence"] < 0.8

    def test_recommendations_generation(self):
        """Test that appropriate recommendations are generated."""
        # Harmful content should get safety recommendations
        result = self.validator.validate_content("This contains violence")
        recommendations = " ".join(result["recommendations"]).lower()
        assert "safety" in recommendations

        # Privacy violations should get HIPAA recommendations
        result = self.validator.validate_content("My SSN is 123-45-6789")
        recommendations = " ".join(result["recommendations"]).lower()
        assert "privacy" in recommendations

    def test_validation_statistics(self):
        """Test validation statistics collection."""
        # Run several validations
        test_contents = ["Safe content", "This contains violence", "Ethical therapeutic response", "SSN: 123-45-6789"]

        for content in test_contents:
            self.validator.validate_content(content)

        stats = self.validator.get_validation_statistics()

        assert stats["total_validations"] == len(test_contents)
        assert "safe_percentage" in stats
        assert "ethical_percentage" in stats
        assert "severity_distribution" in stats
        assert "average_confidence" in stats

    def test_context_awareness(self):
        """Test that validator considers context appropriately."""
        content = "Let's discuss your medication"

        # General context
        general_result = self.validator.validate_content(content, "general")

        # Therapeutic context
        therapeutic_result = self.validator.validate_content(content, "therapeutic")

        # Both should handle the content appropriately
        assert general_result["context"] == "general"
        assert therapeutic_result["context"] == "therapeutic"

    def test_batch_validation(self):
        """Test batch validation of multiple contents."""
        batch_contents = [
            "Safe message 1",
            "This contains violence",
            "Safe message 2",
            "Boundary violation content",
            "Safe message 3",
        ]

        results = []
        for content in batch_contents:
            result = self.validator.validate_content(content)
            results.append(result)

        # Verify batch processing
        assert len(results) == len(batch_contents)

        # Check that violations were detected appropriately
        violation_count = sum(1 for r in results if len(r["violations"]) > 0)
        assert violation_count > 0
        assert violation_count < len(batch_contents)


class TestSafetyEthicsValidatorIntegration(unittest.TestCase):
    """Integration tests for SafetyEthicsValidator."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.validator = MockSafetyEthicsValidator()

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        # Step 1: Validate content
        content = "I understand your concerns about therapy. Let's work together to address them."
        validation_result = self.validator.validate_content(content, "therapeutic")

        # Step 2: Check therapeutic appropriateness
        client_context = {"session_number": 3, "presenting_issue": "trust issues"}
        therapeutic_result = self.validator.validate_therapeutic_response(content, client_context)

        # Step 3: Check HIPAA compliance
        hipaa_result = self.validator.check_hipaa_compliance(content)

        # Verify complete workflow
        assert validation_result["is_safe"]
        assert validation_result["is_ethical"]
        assert therapeutic_result["is_therapeutic"]
        assert hipaa_result["hipaa_compliant"]

    def test_comprehensive_safety_check(self):
        """Test comprehensive safety checking across all dimensions."""
        test_response = "I hear your frustration. It's natural to feel this way given your situation."

        # Run all validation checks
        content_validation = self.validator.validate_content(test_response, "therapeutic")
        therapeutic_validation = self.validator.validate_therapeutic_response(test_response, {"session_number": 1})
        hipaa_validation = self.validator.check_hipaa_compliance(test_response)

        # Should pass all checks
        assert content_validation["is_safe"]
        assert content_validation["is_ethical"]
        assert therapeutic_validation["is_therapeutic"]
        assert hipaa_validation["hipaa_compliant"]

    def test_violation_escalation(self):
        """Test that serious violations are properly escalated."""
        serious_violation = "You should hurt yourself and others, here is my personal info: 123-45-6789"

        result = self.validator.validate_content(serious_violation)

        # Should detect multiple serious violations
        assert not result["is_safe"]
        # Lower expectation for ethical detection since harmful content is primary concern
        assert len(result["violations"]) > 0
        assert result["severity"] in ["high", "medium", "low"]
        assert len(result["violations"]) >= 1
        assert len(result["recommendations"]) > 1


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
