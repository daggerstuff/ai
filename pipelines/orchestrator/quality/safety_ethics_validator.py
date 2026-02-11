#!/usr/bin/env python3
"""
Conversation Safety and Ethics Validation System
Validates conversations for harmful content, ethical boundaries, and safety protocols.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafetyRiskLevel(Enum):
    """Safety risk levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    MINIMAL = "minimal"


class EthicsViolationType(Enum):
    """Types of ethics violations."""
    BOUNDARY_VIOLATION = "boundary_violation"
    CONFIDENTIALITY_BREACH = "confidentiality_breach"
    DUAL_RELATIONSHIP = "dual_relationship"
    COMPETENCE_ISSUE = "competence_issue"
    INFORMED_CONSENT = "informed_consent"
    CULTURAL_INSENSITIVITY = "cultural_insensitivity"
    DISCRIMINATION = "discrimination"
    EXPLOITATION = "exploitation"


class HarmfulContentType(Enum):
    """Types of harmful content."""
    SELF_HARM = "self_harm"
    VIOLENCE = "violence"
    SUBSTANCE_ABUSE = "substance_abuse"
    SEXUAL_CONTENT = "sexual_content"
    HATE_SPEECH = "hate_speech"
    MISINFORMATION = "misinformation"
    MANIPULATION = "manipulation"
    GROOMING = "grooming"


@dataclass
class SafetyIssue:
    """Individual safety issue."""
    issue_type: HarmfulContentType
    risk_level: SafetyRiskLevel
    description: str
    content_snippet: str
    mitigation_required: bool
    recommended_actions: list[str] = field(default_factory=list)


@dataclass
class EthicsViolation:
    """Individual ethics violation."""
    violation_type: EthicsViolationType
    severity: str
    description: str
    content_snippet: str
    professional_standard: str
    corrective_actions: list[str] = field(default_factory=list)


@dataclass
class SafetyEthicsResult:
    """Complete safety and ethics validation result."""
    conversation_id: str
    overall_safety_score: float
    overall_ethics_score: float
    safety_issues: list[SafetyIssue] = field(default_factory=list)
    ethics_violations: list[EthicsViolation] = field(default_factory=list)
    risk_assessment: dict[str, Any] = field(default_factory=dict)
    compliance_status: str = "compliant"
    recommendations: list[str] = field(default_factory=list)
    requires_intervention: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class SafetyEthicsValidator:
    """
    Conversation safety and ethics validation system.
    """

    def __init__(self):
        """Initialize the safety and ethics validator."""
        self.validation_history: list[SafetyEthicsResult] = []
        self.harmful_content_patterns = self._load_harmful_content_patterns()
        self.ethics_guidelines = self._load_ethics_guidelines()
        self.safety_protocols = self._load_safety_protocols()

    def _load_harmful_content_patterns(self) -> dict[HarmfulContentType, dict[str, Any]]:
        """Load harmful content detection patterns."""
        return {
            HarmfulContentType.SELF_HARM: {
                "keywords": [
                    "suicide", "kill myself", "end my life", "self-harm", "cutting",
                    "overdose", "jump off", "hang myself", "want to die", "better off dead"
                ],
                "severity_indicators": {
                    "critical": ["plan to", "method", "tonight", "today", "now", "going to kill"],
                    "high": ["want to die", "thinking of killing", "considering suicide"],
                    "moderate": ["sometimes think", "wondered about", "thoughts of"]
                },
                "protective_factors": ["safety plan", "crisis line", "support system", "support", "family", "friends", "help"]
            },
            HarmfulContentType.VIOLENCE: {
                "keywords": [
                    "hurt someone", "kill them", "violence", "attack", "assault",
                    "weapon", "gun", "knife", "fight", "revenge"
                ],
                "severity_indicators": {
                    "critical": ["plan to", "going to", "will hurt"],
                    "high": ["want to hurt", "thinking of hurting"],
                    "moderate": ["angry enough to", "fantasize about"]
                },
                "protective_factors": ["anger management", "conflict resolution"]
            },
            HarmfulContentType.SUBSTANCE_ABUSE: {
                "keywords": [
                    "overdose", "addiction", "drugs", "alcohol problem", "drinking too much",
                    "high all the time", "can't stop using", "withdrawal"
                ],
                "severity_indicators": {
                    "critical": ["overdosed", "can't function", "lost everything"],
                    "high": ["using daily", "can't control"],
                    "moderate": ["drinking more", "concerned about use"]
                },
                "protective_factors": ["treatment", "recovery", "sobriety"]
            },
            HarmfulContentType.SEXUAL_CONTENT: {
                "keywords": [
                    "sexual abuse", "inappropriate touching", "sexual assault",
                    "unwanted advances", "sexual harassment"
                ],
                "severity_indicators": {
                    "critical": ["ongoing abuse", "current danger"],
                    "high": ["recent incident", "trauma response"],
                    "moderate": ["past experience", "processing trauma"]
                },
                "protective_factors": ["safety planning", "trauma therapy", "support"]
            },
            HarmfulContentType.HATE_SPEECH: {
                "keywords": [
                    "racial slur", "hate", "discrimination", "prejudice",
                    "bigotry", "supremacy", "inferior race"
                ],
                "severity_indicators": {
                    "critical": ["violence against", "eliminate"],
                    "high": ["hate them", "don't deserve"],
                    "moderate": ["don't like", "different from us"]
                },
                "protective_factors": ["education", "diversity training", "empathy"]
            }
        }

    def _load_ethics_guidelines(self) -> dict[EthicsViolationType, dict[str, Any]]:
        """Load professional ethics guidelines."""
        return {
            EthicsViolationType.BOUNDARY_VIOLATION: {
                "indicators": [
                    "personal relationship", "friendship", "dating", "romantic",
                    "my personal life", "let's be friends", "outside of therapy"
                ],
                "standards": "Maintain professional boundaries at all times",
                "severity_factors": {
                    "high": ["romantic", "sexual", "dating", "date"],
                    "moderate": ["friendship", "personal disclosure"],
                    "low": ["minor boundary crossing"]
                }
            },
            EthicsViolationType.CONFIDENTIALITY_BREACH: {
                "indicators": [
                    "told someone about", "shared your information", "discussed with others",
                    "my supervisor knows", "other clients"
                ],
                "standards": "Maintain strict confidentiality except as legally required",
                "severity_factors": {
                    "high": ["shared details", "identified client"],
                    "moderate": ["general discussion", "anonymized"],
                    "low": ["appropriate consultation"]
                }
            },
            EthicsViolationType.DUAL_RELATIONSHIP: {
                "indicators": [
                    "business relationship", "social connection", "family friend",
                    "work together", "mutual friends"
                ],
                "standards": "Avoid dual relationships that could impair judgment",
                "severity_factors": {
                    "high": ["business dealings", "social relationship"],
                    "moderate": ["community connection", "indirect relationship"],
                    "low": ["unavoidable minimal contact"]
                }
            },
            EthicsViolationType.COMPETENCE_ISSUE: {
                "indicators": [
                    "not sure how to help", "outside my expertise", "never dealt with this",
                    "don't know", "not trained in"
                ],
                "standards": "Practice within scope of competence",
                "severity_factors": {
                    "high": ["serious condition", "safety risk"],
                    "moderate": ["specialized treatment", "complex case"],
                    "low": ["learning opportunity", "consultation available"]
                }
            },
            EthicsViolationType.CULTURAL_INSENSITIVITY: {
                "indicators": [
                    "your people", "that culture", "weird customs", "strange beliefs",
                    "normal people", "civilized society"
                ],
                "standards": "Demonstrate cultural competence and sensitivity",
                "severity_factors": {
                    "high": ["discriminatory", "prejudicial"],
                    "moderate": ["insensitive", "stereotyping"],
                    "low": ["lack of awareness", "unintentional"]
                }
            }
        }

    def _load_safety_protocols(self) -> dict[str, dict[str, Any]]:
        """Load safety protocols and procedures."""
        return {
            "crisis_intervention": {
                "triggers": ["suicide", "self-harm", "violence", "abuse"],
                "required_actions": [
                    "immediate safety assessment",
                    "crisis intervention protocol",
                    "safety planning",
                    "emergency contacts",
                    "follow-up procedures"
                ],
                "documentation": "detailed crisis notes required"
            },
            "mandatory_reporting": {
                "triggers": ["child abuse", "elder abuse", "imminent danger"],
                "required_actions": [
                    "immediate reporting to authorities",
                    "documentation of incident",
                    "client notification of limits",
                    "safety planning"
                ],
                "legal_requirements": "mandated by law"
            },
            "risk_assessment": {
                "factors": [
                    "suicide risk", "violence risk", "self-harm risk",
                    "substance abuse", "psychosis", "impaired judgment"
                ],
                "assessment_tools": [
                    "risk assessment scales",
                    "clinical interview",
                    "collateral information"
                ]
            }
        }

    def validate_conversation(self, conversation: dict[str, Any]) -> SafetyEthicsResult:
        """
        Validate conversation for safety and ethics compliance.

        Args:
            conversation: Conversation data to validate

        Returns:
            SafetyEthicsResult with detailed assessment
        """
        conversation_id = conversation.get("id", "unknown")
        logger.info(f"Validating safety and ethics for conversation {conversation_id}")

        content = str(conversation.get("content", ""))
        turns = conversation.get("turns", [])

        # Detect safety issues
        safety_issues = self._detect_safety_issues(content, turns)

        # Detect ethics violations
        ethics_violations = self._detect_ethics_violations(content, turns)

        # Calculate safety score
        safety_score = self._calculate_safety_score(safety_issues)

        # Calculate ethics score
        ethics_score = self._calculate_ethics_score(ethics_violations)

        # Perform risk assessment
        risk_assessment = self._perform_risk_assessment(content, turns, safety_issues)

        # Determine compliance status
        compliance_status = self._determine_compliance_status(safety_issues, ethics_violations)

        # Check if intervention is required
        requires_intervention = self._requires_intervention(safety_issues, ethics_violations)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            safety_issues, ethics_violations, risk_assessment
        )

        result = SafetyEthicsResult(
            conversation_id=conversation_id,
            overall_safety_score=safety_score,
            overall_ethics_score=ethics_score,
            safety_issues=safety_issues,
            ethics_violations=ethics_violations,
            risk_assessment=risk_assessment,
            compliance_status=compliance_status,
            recommendations=recommendations,
            requires_intervention=requires_intervention
        )

        self.validation_history.append(result)
        return result

    def _detect_safety_issues(self, content: str, turns: list[dict]) -> list[SafetyIssue]:
        """Detect safety issues in conversation."""
        issues = []
        content_lower = content.lower()

        for harm_type, patterns in self.harmful_content_patterns.items():
            # Check for harmful content keywords
            detected_keywords = [
                keyword for keyword in patterns["keywords"]
                if keyword in content_lower
            ]

            if detected_keywords:
                # Determine risk level based on severity indicators
                risk_level = self._assess_risk_level(content_lower, patterns["severity_indicators"])

                # Check for protective factors
                protective_factors = [
                    factor for factor in patterns.get("protective_factors", [])
                    if factor in content_lower
                ]

                # Adjust risk level based on protective factors
                if protective_factors:
                    risk_level = self._reduce_risk_level(risk_level)

                # Create safety issue
                issue = SafetyIssue(
                    issue_type=harm_type,
                    risk_level=risk_level,
                    description=f"Detected {harm_type.value} content",
                    content_snippet=", ".join(detected_keywords[:3]),
                    mitigation_required=risk_level in [SafetyRiskLevel.CRITICAL, SafetyRiskLevel.HIGH],
                    recommended_actions=self._get_safety_actions(harm_type, risk_level)
                )
                issues.append(issue)

        return issues

    def _assess_risk_level(self, content: str, severity_indicators: dict[str, list[str]]) -> SafetyRiskLevel:
        """Assess risk level based on severity indicators."""
        # Check for critical indicators
        if any(indicator in content for indicator in severity_indicators.get("critical", [])):
            return SafetyRiskLevel.CRITICAL

        # Check for high indicators
        if any(indicator in content for indicator in severity_indicators.get("high", [])):
            return SafetyRiskLevel.HIGH

        # Check for moderate indicators
        if any(indicator in content for indicator in severity_indicators.get("moderate", [])):
            return SafetyRiskLevel.MODERATE

        # Default to low risk if keywords present but no specific indicators
        return SafetyRiskLevel.LOW

    def _reduce_risk_level(self, current_level: SafetyRiskLevel) -> SafetyRiskLevel:
        """Reduce risk level when protective factors are present."""
        risk_hierarchy = [
            SafetyRiskLevel.CRITICAL,
            SafetyRiskLevel.HIGH,
            SafetyRiskLevel.MODERATE,
            SafetyRiskLevel.LOW,
            SafetyRiskLevel.MINIMAL
        ]

        current_index = risk_hierarchy.index(current_level)
        if current_index < len(risk_hierarchy) - 1:
            return risk_hierarchy[current_index + 1]
        return current_level

    def _get_safety_actions(self, harm_type: HarmfulContentType, risk_level: SafetyRiskLevel) -> list[str]:
        """Get recommended safety actions for specific harm type and risk level."""
        actions = []

        if risk_level == SafetyRiskLevel.CRITICAL:
            actions.extend([
                "Immediate safety assessment required",
                "Crisis intervention protocol activation",
                "Emergency services contact if necessary",
                "Continuous monitoring"
            ])

        if harm_type == HarmfulContentType.SELF_HARM:
            actions.extend([
                "Suicide risk assessment",
                "Safety planning",
                "Crisis hotline information",
                "Remove means of harm"
            ])
        elif harm_type == HarmfulContentType.VIOLENCE:
            actions.extend([
                "Violence risk assessment",
                "Threat assessment",
                "Potential victim notification",
                "Law enforcement contact if imminent"
            ])
        elif harm_type == HarmfulContentType.SUBSTANCE_ABUSE:
            actions.extend([
                "Substance abuse assessment",
                "Detox evaluation",
                "Treatment referral",
                "Support group information"
            ])

        return actions

    def _detect_ethics_violations(self, content: str, turns: list[dict]) -> list[EthicsViolation]:
        """Detect ethics violations in conversation."""
        violations = []
        content_lower = content.lower()

        for violation_type, guidelines in self.ethics_guidelines.items():
            # Check for violation indicators
            detected_indicators = [
                indicator for indicator in guidelines["indicators"]
                if indicator in content_lower
            ]

            if detected_indicators:
                # Determine severity
                severity = self._assess_violation_severity(content_lower, guidelines["severity_factors"])

                # Create ethics violation
                violation = EthicsViolation(
                    violation_type=violation_type,
                    severity=severity,
                    description=f"Potential {violation_type.value.replace('_', ' ')}",
                    content_snippet=", ".join(detected_indicators[:2]),
                    professional_standard=guidelines["standards"],
                    corrective_actions=self._get_corrective_actions(violation_type, severity)
                )
                violations.append(violation)

        return violations

    def _assess_violation_severity(self, content: str, severity_factors: dict[str, list[str]]) -> str:
        """Assess severity of ethics violation."""
        if any(factor in content for factor in severity_factors.get("high", [])):
            return "high"
        if any(factor in content for factor in severity_factors.get("moderate", [])):
            return "moderate"
        return "low"

    def _get_corrective_actions(self, violation_type: EthicsViolationType, severity: str) -> list[str]:
        """Get corrective actions for ethics violations."""
        actions = []

        if severity == "high":
            actions.extend([
                "Immediate supervision consultation",
                "Ethics committee review",
                "Corrective action plan",
                "Additional training required"
            ])

        if violation_type == EthicsViolationType.BOUNDARY_VIOLATION:
            actions.extend([
                "Boundary clarification with client",
                "Professional relationship redefinition",
                "Supervision for boundary management"
            ])
        elif violation_type == EthicsViolationType.CONFIDENTIALITY_BREACH:
            actions.extend([
                "Client notification of breach",
                "Damage mitigation measures",
                "Confidentiality protocol review"
            ])
        elif violation_type == EthicsViolationType.COMPETENCE_ISSUE:
            actions.extend([
                "Immediate consultation",
                "Referral to specialist",
                "Additional training",
                "Competence assessment"
            ])

        return actions

    def _calculate_safety_score(self, safety_issues: list[SafetyIssue]) -> float:
        """Calculate overall safety score."""
        if not safety_issues:
            return 1.0

        # Weight issues by risk level
        risk_weights = {
            SafetyRiskLevel.CRITICAL: 1.0,
            SafetyRiskLevel.HIGH: 0.8,
            SafetyRiskLevel.MODERATE: 0.6,
            SafetyRiskLevel.LOW: 0.4,
            SafetyRiskLevel.MINIMAL: 0.2
        }

        total_risk = sum(risk_weights[issue.risk_level] for issue in safety_issues)
        max_possible_risk = len(safety_issues) * 1.0

        # Convert to safety score (inverse of risk)
        safety_score = 1.0 - (total_risk / max_possible_risk)
        return max(0.0, safety_score)

    def _calculate_ethics_score(self, ethics_violations: list[EthicsViolation]) -> float:
        """Calculate overall ethics score."""
        if not ethics_violations:
            return 1.0

        # Weight violations by severity
        severity_weights = {
            "high": 1.0,
            "moderate": 0.6,
            "low": 0.3
        }

        total_violation_weight = sum(
            severity_weights[violation.severity] for violation in ethics_violations
        )
        max_possible_weight = len(ethics_violations) * 1.0

        # Convert to ethics score (inverse of violations)
        ethics_score = 1.0 - (total_violation_weight / max_possible_weight)
        return max(0.0, ethics_score)

    def _perform_risk_assessment(
        self,
        content: str,
        turns: list[dict],
        safety_issues: list[SafetyIssue]
    ) -> dict[str, Any]:
        """Perform comprehensive risk assessment."""
        assessment = {
            "overall_risk_level": "low",
            "immediate_intervention_needed": False,
            "risk_factors": [],
            "protective_factors": [],
            "monitoring_required": False
        }

        # Determine overall risk level
        if any(issue.risk_level == SafetyRiskLevel.CRITICAL for issue in safety_issues):
            assessment["overall_risk_level"] = "critical"
            assessment["immediate_intervention_needed"] = True
        elif any(issue.risk_level == SafetyRiskLevel.HIGH for issue in safety_issues):
            assessment["overall_risk_level"] = "high"
            assessment["monitoring_required"] = True
        elif any(issue.risk_level == SafetyRiskLevel.MODERATE for issue in safety_issues):
            assessment["overall_risk_level"] = "moderate"
            assessment["monitoring_required"] = True

        # Identify risk factors
        content_lower = content.lower()
        risk_indicators = [
            "isolated", "hopeless", "no support", "financial problems",
            "relationship issues", "job loss", "health problems"
        ]
        assessment["risk_factors"] = [
            factor for factor in risk_indicators if factor in content_lower
        ]

        # Identify protective factors
        protective_indicators = [
            "support system", "family", "friends", "therapy", "treatment",
            "coping skills", "hope", "future plans", "reasons to live"
        ]
        assessment["protective_factors"] = [
            factor for factor in protective_indicators if factor in content_lower
        ]

        return assessment

    def _determine_compliance_status(
        self,
        safety_issues: list[SafetyIssue],
        ethics_violations: list[EthicsViolation]
    ) -> str:
        """Determine overall compliance status."""
        # Check for critical safety issues
        if any(issue.risk_level == SafetyRiskLevel.CRITICAL for issue in safety_issues):
            return "non_compliant_critical"

        # Check for high severity ethics violations
        if any(violation.severity == "high" for violation in ethics_violations):
            return "non_compliant_ethics"

        # Check for multiple moderate issues
        moderate_issues = (
            len([issue for issue in safety_issues if issue.risk_level == SafetyRiskLevel.MODERATE]) +
            len([violation for violation in ethics_violations if violation.severity == "moderate"])
        )

        if moderate_issues >= 3:
            return "non_compliant_multiple"

        # Check for any issues
        if safety_issues or ethics_violations:
            return "conditional_compliance"

        return "compliant"

    def _requires_intervention(
        self,
        safety_issues: list[SafetyIssue],
        ethics_violations: list[EthicsViolation]
    ) -> bool:
        """Determine if immediate intervention is required."""
        # Critical safety issues require intervention
        if any(issue.risk_level == SafetyRiskLevel.CRITICAL for issue in safety_issues):
            return True

        # High severity ethics violations require intervention
        if any(violation.severity == "high" for violation in ethics_violations):
            return True

        # Multiple high-risk safety issues require intervention
        high_risk_count = len([
            issue for issue in safety_issues
            if issue.risk_level == SafetyRiskLevel.HIGH
        ])
        return high_risk_count >= 2

    def _generate_recommendations(
        self,
        safety_issues: list[SafetyIssue],
        ethics_violations: list[EthicsViolation],
        risk_assessment: dict[str, Any]
    ) -> list[str]:
        """Generate safety and ethics recommendations."""
        recommendations = []

        # Critical safety recommendations
        if risk_assessment["immediate_intervention_needed"]:
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED: Critical safety issue detected",
                "Activate crisis intervention protocol immediately",
                "Ensure client safety before proceeding",
                "Document all safety measures taken"
            ])

        # Safety issue recommendations
        for issue in safety_issues:
            if issue.mitigation_required:
                recommendations.extend(issue.recommended_actions)

        # Ethics violation recommendations
        for violation in ethics_violations:
            if violation.severity in ["high", "moderate"]:
                recommendations.extend(violation.corrective_actions)

        # General recommendations based on risk level
        if risk_assessment["overall_risk_level"] in ["high", "critical"]:
            recommendations.extend([
                "Increase session frequency",
                "Develop comprehensive safety plan",
                "Coordinate with other healthcare providers",
                "Consider higher level of care"
            ])

        # Monitoring recommendations
        if risk_assessment["monitoring_required"]:
            recommendations.extend([
                "Implement regular safety check-ins",
                "Monitor for escalation of risk factors",
                "Maintain detailed documentation"
            ])

        return list(set(recommendations))  # Remove duplicates

    def get_validation_summary(self) -> dict[str, Any]:
        """Get validation summary statistics."""
        if not self.validation_history:
            return {"message": "No validations performed yet"}

        total_validations = len(self.validation_history)

        # Safety statistics
        avg_safety_score = sum(r.overall_safety_score for r in self.validation_history) / total_validations
        safety_issues_count = sum(len(r.safety_issues) for r in self.validation_history)

        # Ethics statistics
        avg_ethics_score = sum(r.overall_ethics_score for r in self.validation_history) / total_validations
        ethics_violations_count = sum(len(r.ethics_violations) for r in self.validation_history)

        # Compliance statistics
        compliance_distribution = {}
        for result in self.validation_history:
            status = result.compliance_status
            compliance_distribution[status] = compliance_distribution.get(status, 0) + 1

        # Intervention statistics
        interventions_required = sum(1 for r in self.validation_history if r.requires_intervention)

        return {
            "total_validations": total_validations,
            "average_safety_score": avg_safety_score,
            "average_ethics_score": avg_ethics_score,
            "total_safety_issues": safety_issues_count,
            "total_ethics_violations": ethics_violations_count,
            "compliance_distribution": compliance_distribution,
            "interventions_required": interventions_required,
            "intervention_rate": interventions_required / total_validations,
            "last_validation": self.validation_history[-1].timestamp.isoformat()
        }


def main():
    """Example usage of the SafetyEthicsValidator."""
    validator = SafetyEthicsValidator()

    # Example conversations
    sample_conversations = [
        {
            "id": "safety_001",
            "content": "I understand you're having thoughts of suicide. This is very concerning and I want to help you stay safe. Let's develop a safety plan and connect you with crisis resources immediately.",
            "turns": [
                {"speaker": "user", "text": "I've been thinking about killing myself lately."},
                {"speaker": "therapist", "text": "I'm very concerned about your safety. Let's develop a safety plan right now."}
            ]
        },
        {
            "id": "ethics_001",
            "content": "I think we should be friends outside of therapy. Maybe we could grab coffee sometime and I could tell you about my personal problems.",
            "turns": [
                {"speaker": "therapist", "text": "I think we should be friends outside of therapy."},
                {"speaker": "user", "text": "That sounds nice."}
            ]
        },
        {
            "id": "safe_001",
            "content": "I understand you're feeling anxious. Let's work on some coping strategies that can help you manage these feelings in a healthy way.",
            "turns": [
                {"speaker": "user", "text": "I'm feeling really anxious lately."},
                {"speaker": "therapist", "text": "Let's work on some healthy coping strategies."}
            ]
        }
    ]

    # Validate conversations
    for conversation in sample_conversations:
        result = validator.validate_conversation(conversation)


        if result.safety_issues:
            for _issue in result.safety_issues:
                pass

        if result.ethics_violations:
            for _violation in result.ethics_violations:
                pass

        if result.recommendations:
            for _rec in result.recommendations[:3]:
                pass

    # Print summary
    validator.get_validation_summary()


if __name__ == "__main__":
    main()
