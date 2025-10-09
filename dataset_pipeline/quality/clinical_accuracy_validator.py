"""
Clinical accuracy validator for psychology knowledge integration pipeline.

This module validates the clinical accuracy of generated therapeutic conversations,
ensuring they meet professional standards, ethical guidelines, and evidence-based
practice requirements across DSM-5, PDM-2, and Big Five frameworks.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from big_five_processor import BigFiveProcessor
from client_scenario_generator import ClientScenario, ScenarioType
from conversation_schema import Conversation
from dsm5_parser import DSM5Parser
from logger import get_logger
from pdm2_parser import PDM2Parser
from therapeutic_response_generator import TherapeuticTechnique

logger = get_logger("dataset_pipeline.clinical_accuracy_validator")


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"  # Potentially harmful or unethical
    HIGH = "high"  # Clinically inappropriate
    MEDIUM = "medium"  # Best practice violations
    LOW = "low"  # Minor improvements needed
    INFO = "info"  # Informational notes


class ValidationCategory(Enum):
    """Categories of clinical validation."""
    ETHICAL_COMPLIANCE = "ethical_compliance"
    CLINICAL_ACCURACY = "clinical_accuracy"
    THERAPEUTIC_APPROPRIATENESS = "therapeutic_appropriateness"
    PROFESSIONAL_BOUNDARIES = "professional_boundaries"
    CULTURAL_SENSITIVITY = "cultural_sensitivity"
    SAFETY_PROTOCOLS = "safety_protocols"
    EVIDENCE_BASE = "evidence_base"
    DIAGNOSTIC_ACCURACY = "diagnostic_accuracy"


@dataclass
class ValidationIssue:
    """Individual validation issue found in content."""
    id: str
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    location: str  # Where in the conversation
    recommendation: str
    evidence_base: list[str] = field(default_factory=list)
    related_guidelines: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Complete validation result for a conversation."""
    conversation_id: str
    overall_score: float  # 0.0 to 1.0
    is_clinically_acceptable: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    validation_metadata: dict[str, Any] = field(default_factory=dict)
    validated_at: str | None = None


class ClinicalAccuracyValidator:
    """
    Comprehensive clinical accuracy validator.

    Validates therapeutic conversations against clinical standards, ethical
    guidelines, and evidence-based practice requirements to ensure generated
    content meets professional therapeutic standards.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the clinical accuracy validator."""
        self.config = config or {}

        # Initialize psychology knowledge parsers for validation
        self.dsm5_parser = DSM5Parser()
        self.pdm2_parser = PDM2Parser()
        self.big_five_processor = BigFiveProcessor()

        # Initialize validation rules and guidelines
        self._initialize_validation_rules()

        logger.info("Clinical Accuracy Validator initialized")

    def _initialize_validation_rules(self) -> None:
        """Initialize clinical validation rules and guidelines."""

        # Ethical compliance rules
        self.ethical_rules = {
            "confidentiality_violations": [
                r"tell your family",
                r"share this with",
                r"I'll inform",
                r"report this to"
            ],
            "boundary_violations": [
                r"let's be friends",
                r"personal relationship",
                r"outside of therapy",
                r"my personal"
            ],
            "dual_relationships": [
                r"I know you from",
                r"we've met before",
                r"my friend",
                r"business relationship"
            ]
        }

        # Clinical accuracy patterns
        self.clinical_patterns = {
            "inappropriate_diagnosis": [
                r"you definitely have",
                r"I diagnose you with",
                r"you are [a-z]+ disorder",
                r"this proves you have"
            ],
            "premature_conclusions": [
                r"the problem is clearly",
                r"obviously you",
                r"definitely caused by",
                r"always happens because"
            ],
            "medical_advice": [
                r"stop taking your medication",
                r"increase your dose",
                r"don't need medication",
                r"medical treatment"
            ]
        }

        # Safety protocol requirements
        self.safety_requirements = {
            "suicide_risk": [
                "safety assessment",
                "safety plan",
                "immediate safety",
                "crisis resources"
            ],
            "self_harm": [
                "safety planning",
                "harm reduction",
                "coping strategies",
                "support system"
            ],
            "violence_risk": [
                "safety concerns",
                "risk assessment",
                "protective measures",
                "appropriate authorities"
            ]
        }

        # Evidence-based technique validation
        self.technique_appropriateness = {
            TherapeuticTechnique.COGNITIVE_RESTRUCTURING: {
                "appropriate_for": ["anxiety", "depression", "negative thoughts"],
                "inappropriate_for": ["psychosis", "severe crisis", "initial session"],
                "requires": ["established rapport", "client readiness", "psychoeducation"]
            },
            TherapeuticTechnique.SAFETY_PLANNING: {
                "appropriate_for": ["suicidal ideation", "self-harm", "crisis"],
                "inappropriate_for": ["mild symptoms", "routine sessions"],
                "requires": ["immediate safety", "crisis assessment", "resources"]
            },
            TherapeuticTechnique.THERAPEUTIC_CHALLENGE: {
                "appropriate_for": ["established therapy", "defensive patterns"],
                "inappropriate_for": ["initial sessions", "crisis", "trauma"],
                "requires": ["strong rapport", "client stability", "careful timing"]
            }
        }

        # Cultural sensitivity guidelines
        self.cultural_guidelines = [
            "Avoid cultural assumptions",
            "Respect diverse perspectives",
            "Consider cultural context",
            "Use inclusive language",
            "Acknowledge cultural factors"
        ]

        logger.info("Initialized clinical validation rules and guidelines")

    def validate_conversation(self, conversation: Conversation, scenario: ClientScenario | None = None) -> ValidationResult:
        """Validate a complete therapeutic conversation for clinical accuracy."""

        issues = []
        strengths = []
        recommendations = []

        # Validate ethical compliance
        ethical_issues = self._validate_ethical_compliance(conversation)
        issues.extend(ethical_issues)

        # Validate clinical accuracy
        clinical_issues = self._validate_clinical_accuracy(conversation, scenario)
        issues.extend(clinical_issues)

        # Validate therapeutic appropriateness
        therapeutic_issues = self._validate_therapeutic_appropriateness(conversation, scenario)
        issues.extend(therapeutic_issues)

        # Validate safety protocols
        safety_issues = self._validate_safety_protocols(conversation, scenario)
        issues.extend(safety_issues)

        # Validate professional boundaries
        boundary_issues = self._validate_professional_boundaries(conversation)
        issues.extend(boundary_issues)

        # Validate cultural sensitivity
        cultural_issues = self._validate_cultural_sensitivity(conversation, scenario)
        issues.extend(cultural_issues)

        # Identify strengths
        strengths = self._identify_strengths(conversation, scenario)

        # Generate recommendations
        recommendations = self._generate_recommendations(issues, conversation, scenario)

        # Calculate overall score
        overall_score = self._calculate_overall_score(issues, strengths)

        # Determine clinical acceptability
        is_acceptable = self._determine_acceptability(issues, overall_score)

        result = ValidationResult(
            conversation_id=conversation.id,
            overall_score=overall_score,
            is_clinically_acceptable=is_acceptable,
            issues=issues,
            strengths=strengths,
            recommendations=recommendations,
            validation_metadata={
                "total_messages": len(conversation.messages),
                "therapist_messages": len([m for m in conversation.messages if m.role == "therapist"]),
                "client_messages": len([m for m in conversation.messages if m.role == "client"]),
                "validation_categories": list({issue.category.value for issue in issues}),
                "severity_distribution": self._get_severity_distribution(issues)
            },
            validated_at=datetime.now().isoformat()
        )

        logger.info(f"Validated conversation {conversation.id}: Score {overall_score:.2f}, Acceptable: {is_acceptable}")
        return result

    def _validate_ethical_compliance(self, conversation: Conversation) -> list[ValidationIssue]:
        """Validate ethical compliance in the conversation."""
        issues = []

        for i, message in enumerate(conversation.messages):
            if message.role != "therapist":
                continue

            content_lower = message.content.lower()

            # Check for confidentiality violations
            for pattern in self.ethical_rules["confidentiality_violations"]:
                if re.search(pattern, content_lower):
                    issues.append(ValidationIssue(
                        id=f"ethical_{i}_confidentiality",
                        category=ValidationCategory.ETHICAL_COMPLIANCE,
                        severity=ValidationSeverity.CRITICAL,
                        message="Potential confidentiality violation detected",
                        location=f"Message {i+1}",
                        recommendation="Ensure client confidentiality is maintained",
                        related_guidelines=["APA Ethics Code 4.01", "Confidentiality standards"]
                    ))

            # Check for boundary violations
            for pattern in self.ethical_rules["boundary_violations"]:
                if re.search(pattern, content_lower):
                    issues.append(ValidationIssue(
                        id=f"ethical_{i}_boundary",
                        category=ValidationCategory.PROFESSIONAL_BOUNDARIES,
                        severity=ValidationSeverity.HIGH,
                        message="Professional boundary violation detected",
                        location=f"Message {i+1}",
                        recommendation="Maintain appropriate therapeutic boundaries",
                        related_guidelines=["APA Ethics Code 3.05", "Multiple relationships"]
                    ))

        return issues

    def _validate_clinical_accuracy(self, conversation: Conversation, scenario: ClientScenario | None) -> list[ValidationIssue]:
        """Validate clinical accuracy of therapeutic content."""
        issues = []

        for i, message in enumerate(conversation.messages):
            if message.role != "therapist":
                continue

            content_lower = message.content.lower()

            # Check for inappropriate diagnosis
            for pattern in self.clinical_patterns["inappropriate_diagnosis"]:
                if re.search(pattern, content_lower):
                    issues.append(ValidationIssue(
                        id=f"clinical_{i}_diagnosis",
                        category=ValidationCategory.DIAGNOSTIC_ACCURACY,
                        severity=ValidationSeverity.HIGH,
                        message="Inappropriate diagnostic statement detected",
                        location=f"Message {i+1}",
                        recommendation="Avoid definitive diagnostic statements without proper assessment",
                        evidence_base=["DSM-5 diagnostic criteria", "Clinical assessment standards"]
                    ))

            # Check for medical advice
            for pattern in self.clinical_patterns["medical_advice"]:
                if re.search(pattern, content_lower):
                    issues.append(ValidationIssue(
                        id=f"clinical_{i}_medical",
                        category=ValidationCategory.CLINICAL_ACCURACY,
                        severity=ValidationSeverity.CRITICAL,
                        message="Inappropriate medical advice detected",
                        location=f"Message {i+1}",
                        recommendation="Refer medical concerns to appropriate healthcare providers",
                        related_guidelines=["Scope of practice", "Medical referral protocols"]
                    ))

        return issues

    def _validate_therapeutic_appropriateness(self, conversation: Conversation, scenario: ClientScenario | None) -> list[ValidationIssue]:
        """Validate appropriateness of therapeutic techniques used."""
        issues = []

        for i, message in enumerate(conversation.messages):
            if message.role != "therapist" or "technique" not in message.meta:
                continue

            technique_str = message.meta["technique"]
            try:
                technique = TherapeuticTechnique(technique_str)
            except ValueError:
                continue

            if technique in self.technique_appropriateness:
                rules = self.technique_appropriateness[technique]

                # Check if technique is appropriate for scenario
                if scenario:
                    scenario_concerns = [*scenario.presenting_problem.symptoms, scenario.presenting_problem.primary_concern.lower()]

                    # Check inappropriate usage
                    for inappropriate in rules.get("inappropriate_for", []):
                        if any(inappropriate in concern.lower() for concern in scenario_concerns):
                            issues.append(ValidationIssue(
                                id=f"therapeutic_{i}_{technique_str}",
                                category=ValidationCategory.THERAPEUTIC_APPROPRIATENESS,
                                severity=ValidationSeverity.HIGH,
                                message=f"{technique_str.replace('_', ' ').title()} may be inappropriate for this presentation",
                                location=f"Message {i+1}",
                                recommendation=f"Consider alternative techniques for {inappropriate} presentations",
                                evidence_base=[f"Evidence-based practice for {technique_str}"]
                            ))

        return issues

    def _validate_safety_protocols(self, conversation: Conversation, scenario: ClientScenario | None) -> list[ValidationIssue]:
        """Validate safety protocol implementation."""
        issues = []

        # Check if crisis scenario has appropriate safety measures
        if scenario and scenario.scenario_type == ScenarioType.CRISIS_INTERVENTION:
            conversation_text = " ".join([msg.content.lower() for msg in conversation.messages if msg.role == "therapist"])

            crisis_type = scenario.session_context.get("crisis_type", "general")
            if crisis_type in self.safety_requirements:
                required_elements = self.safety_requirements[crisis_type]
                missing_elements = []

                for element in required_elements:
                    if element.lower() not in conversation_text:
                        missing_elements.append(element)

                if missing_elements:
                    issues.append(ValidationIssue(
                        id=f"safety_protocol_{crisis_type}",
                        category=ValidationCategory.SAFETY_PROTOCOLS,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Missing required safety elements for {crisis_type}",
                        location="Overall conversation",
                        recommendation=f"Include: {', '.join(missing_elements)}",
                        related_guidelines=["Crisis intervention protocols", "Safety planning standards"]
                    ))

        return issues

    def _validate_professional_boundaries(self, conversation: Conversation) -> list[ValidationIssue]:
        """Validate maintenance of professional boundaries."""
        issues = []

        for i, message in enumerate(conversation.messages):
            if message.role != "therapist":
                continue

            content_lower = message.content.lower()

            # Check for self-disclosure violations
            self_disclosure_patterns = [
                r"I have the same",
                r"I also struggle with",
                r"my personal experience",
                r"when I was"
            ]

            for pattern in self_disclosure_patterns:
                if re.search(pattern, content_lower):
                    issues.append(ValidationIssue(
                        id=f"boundary_{i}_disclosure",
                        category=ValidationCategory.PROFESSIONAL_BOUNDARIES,
                        severity=ValidationSeverity.MEDIUM,
                        message="Potentially inappropriate self-disclosure",
                        location=f"Message {i+1}",
                        recommendation="Maintain professional boundaries with minimal self-disclosure",
                        related_guidelines=["Therapeutic boundary guidelines"]
                    ))

        return issues

    def _validate_cultural_sensitivity(self, conversation: Conversation, scenario: ClientScenario | None) -> list[ValidationIssue]:
        """Validate cultural sensitivity and inclusivity."""
        issues = []

        # Check for cultural assumptions
        problematic_phrases = [
            "your people",
            "in your culture",
            "typical for",
            "all [cultural group]"
        ]

        for i, message in enumerate(conversation.messages):
            if message.role != "therapist":
                continue

            content_lower = message.content.lower()

            for phrase in problematic_phrases:
                if phrase in content_lower:
                    issues.append(ValidationIssue(
                        id=f"cultural_{i}_assumption",
                        category=ValidationCategory.CULTURAL_SENSITIVITY,
                        severity=ValidationSeverity.MEDIUM,
                        message="Potential cultural assumption or stereotype",
                        location=f"Message {i+1}",
                        recommendation="Avoid cultural generalizations and assumptions",
                        related_guidelines=["Cultural competency standards", "Inclusive practice guidelines"]
                    ))

        return issues

    def _identify_strengths(self, conversation: Conversation, scenario: ClientScenario | None) -> list[str]:
        """Identify strengths and positive aspects of the conversation."""
        strengths = []

        therapist_messages = [msg for msg in conversation.messages if msg.role == "therapist"]

        # Check for empathic responses
        empathy_indicators = ["I hear", "I understand", "that sounds", "I can sense"]
        empathy_count = sum(1 for msg in therapist_messages
                           for indicator in empathy_indicators
                           if indicator.lower() in msg.content.lower())

        if empathy_count >= len(therapist_messages) * 0.3:  # 30% of messages show empathy
            strengths.append("Demonstrates consistent empathic responding")

        # Check for open-ended questions
        question_count = sum(1 for msg in therapist_messages if "?" in msg.content)
        if question_count >= len(therapist_messages) * 0.4:  # 40% are questions
            strengths.append("Uses appropriate open-ended questioning")

        # Check for validation
        validation_indicators = ["makes sense", "understandable", "normal to feel", "valid"]
        validation_count = sum(1 for msg in therapist_messages
                              for indicator in validation_indicators
                              if indicator.lower() in msg.content.lower())

        if validation_count > 0:
            strengths.append("Provides client validation and normalization")

        # Check for technique diversity
        techniques_used = set()
        for msg in therapist_messages:
            if "technique" in msg.meta:
                techniques_used.add(msg.meta["technique"])

        if len(techniques_used) >= 3:
            strengths.append("Demonstrates diverse therapeutic technique usage")

        return strengths

    def _generate_recommendations(self, issues: list[ValidationIssue], conversation: Conversation, scenario: ClientScenario | None) -> list[str]:
        """Generate specific recommendations for improvement."""
        recommendations = []

        # Group issues by category
        issue_categories = {}
        for issue in issues:
            category = issue.category.value
            if category not in issue_categories:
                issue_categories[category] = []
            issue_categories[category].append(issue)

        # Generate category-specific recommendations
        if ValidationCategory.ETHICAL_COMPLIANCE.value in issue_categories:
            recommendations.append("Review ethical guidelines and confidentiality requirements")

        if ValidationCategory.SAFETY_PROTOCOLS.value in issue_categories:
            recommendations.append("Implement comprehensive safety assessment and planning protocols")

        if ValidationCategory.THERAPEUTIC_APPROPRIATENESS.value in issue_categories:
            recommendations.append("Ensure therapeutic techniques match client presentation and readiness")

        if ValidationCategory.CULTURAL_SENSITIVITY.value in issue_categories:
            recommendations.append("Enhance cultural competency and avoid assumptions")

        # Critical issues require immediate attention
        critical_issues = [issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL]
        if critical_issues:
            recommendations.insert(0, "Address critical safety and ethical concerns immediately")

        return recommendations

    def _calculate_overall_score(self, issues: list[ValidationIssue], strengths: list[str]) -> float:
        """Calculate overall clinical accuracy score (0.0 to 1.0)."""
        base_score = 1.0

        # Deduct points based on issue severity
        severity_weights = {
            ValidationSeverity.CRITICAL: 0.3,
            ValidationSeverity.HIGH: 0.2,
            ValidationSeverity.MEDIUM: 0.1,
            ValidationSeverity.LOW: 0.05,
            ValidationSeverity.INFO: 0.01
        }

        for issue in issues:
            base_score -= severity_weights.get(issue.severity, 0.05)

        # Add points for strengths (up to 0.2 bonus)
        strength_bonus = min(len(strengths) * 0.05, 0.2)
        base_score += strength_bonus

        # Ensure score is between 0.0 and 1.0
        return max(0.0, min(1.0, base_score))

    def _determine_acceptability(self, issues: list[ValidationIssue], overall_score: float) -> bool:
        """Determine if conversation is clinically acceptable."""
        # Critical issues make conversation unacceptable
        critical_issues = [issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL]
        if critical_issues:
            return False

        # High number of high-severity issues
        high_issues = [issue for issue in issues if issue.severity == ValidationSeverity.HIGH]
        if len(high_issues) > 2:
            return False

        # Overall score threshold
        return overall_score >= 0.7

    def _get_severity_distribution(self, issues: list[ValidationIssue]) -> dict[str, int]:
        """Get distribution of issue severities."""
        distribution = {}
        for issue in issues:
            severity = issue.severity.value
            distribution[severity] = distribution.get(severity, 0) + 1
        return distribution

    def validate_conversation_batch(self, conversations: list[Conversation], scenarios: list[ClientScenario] | None = None) -> list[ValidationResult]:
        """Validate a batch of conversations for clinical accuracy."""
        results = []

        for i, conversation in enumerate(conversations):
            scenario = scenarios[i] if scenarios and i < len(scenarios) else None
            result = self.validate_conversation(conversation, scenario)
            results.append(result)

        logger.info(f"Validated batch of {len(conversations)} conversations")
        return results

    def export_validation_results(self, results: list[ValidationResult], output_path: Path) -> bool:
        """Export validation results to JSON format."""
        try:
            export_data = {
                "validation_results": [],
                "summary": {
                    "total_conversations": len(results),
                    "clinically_acceptable": sum(1 for r in results if r.is_clinically_acceptable),
                    "average_score": sum(r.overall_score for r in results) / len(results) if results else 0,
                    "validation_date": datetime.now().isoformat()
                }
            }

            for result in results:
                result_dict = {
                    "conversation_id": result.conversation_id,
                    "overall_score": result.overall_score,
                    "is_clinically_acceptable": result.is_clinically_acceptable,
                    "issues": [
                        {
                            "id": issue.id,
                            "category": issue.category.value,
                            "severity": issue.severity.value,
                            "message": issue.message,
                            "location": issue.location,
                            "recommendation": issue.recommendation,
                            "evidence_base": issue.evidence_base,
                            "related_guidelines": issue.related_guidelines
                        } for issue in result.issues
                    ],
                    "strengths": result.strengths,
                    "recommendations": result.recommendations,
                    "validation_metadata": result.validation_metadata,
                    "validated_at": result.validated_at
                }
                export_data["validation_results"].append(result_dict)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported validation results for {len(results)} conversations to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export validation results: {e}")
            return False

    def get_validation_statistics(self, results: list[ValidationResult]) -> dict[str, Any]:
        """Get comprehensive statistics about validation results."""
        if not results:
            return {}

        stats = {
            "total_conversations": len(results),
            "clinically_acceptable": sum(1 for r in results if r.is_clinically_acceptable),
            "acceptance_rate": sum(1 for r in results if r.is_clinically_acceptable) / len(results),
            "average_score": sum(r.overall_score for r in results) / len(results),
            "score_distribution": {},
            "issue_categories": {},
            "severity_distribution": {},
            "common_issues": {},
            "common_strengths": {}
        }

        # Score distribution
        for result in results:
            score_range = f"{int(result.overall_score * 10) / 10:.1f}-{int(result.overall_score * 10) / 10 + 0.1:.1f}"
            stats["score_distribution"][score_range] = stats["score_distribution"].get(score_range, 0) + 1

        # Issue analysis
        all_issues = []
        all_strengths = []

        for result in results:
            all_issues.extend(result.issues)
            all_strengths.extend(result.strengths)

        # Category distribution
        for issue in all_issues:
            category = issue.category.value
            stats["issue_categories"][category] = stats["issue_categories"].get(category, 0) + 1

            severity = issue.severity.value
            stats["severity_distribution"][severity] = stats["severity_distribution"].get(severity, 0) + 1

        # Common issues and strengths
        issue_messages = {}
        for issue in all_issues:
            msg = issue.message
            issue_messages[msg] = issue_messages.get(msg, 0) + 1

        stats["common_issues"] = dict(sorted(issue_messages.items(), key=lambda x: x[1], reverse=True)[:5])

        strength_counts = {}
        for strength in all_strengths:
            strength_counts[strength] = strength_counts.get(strength, 0) + 1

        stats["common_strengths"] = dict(sorted(strength_counts.items(), key=lambda x: x[1], reverse=True)[:5])

        return stats
