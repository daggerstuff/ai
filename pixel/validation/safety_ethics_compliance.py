"""
Safety and Ethics Compliance Validation System

This module provides comprehensive validation of clinical content for safety
and ethical compliance according to professional standards and guidelines.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplianceStandard(Enum):
    """Professional compliance standards."""

    APA_ETHICS = "apa_ethics"  # American Psychological Association
    NASW_CODE = "nasw_code"  # National Association of Social Workers
    ACA_CODE = "aca_code"  # American Counseling Association
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    FERPA = "ferpa"  # Family Educational Rights and Privacy Act
    SAMHSA_GUIDELINES = "samhsa_guidelines"  # Substance Abuse and Mental Health Services
    JOINT_COMMISSION = "joint_commission"  # Joint Commission Standards
    STATE_LICENSING = "state_licensing"  # State licensing requirements


class ComplianceLevel(Enum):
    """Compliance assessment levels."""

    FULLY_COMPLIANT = "fully_compliant"
    MOSTLY_COMPLIANT = "mostly_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    CRITICAL_VIOLATION = "critical_violation"


class ViolationSeverity(Enum):
    """Severity levels for compliance violations."""

    CRITICAL = "critical"  # Immediate risk to client safety or severe ethical breach
    HIGH = "high"  # Significant compliance issue requiring immediate attention
    MEDIUM = "medium"  # Moderate issue that should be addressed
    LOW = "low"  # Minor issue, good practice to address
    ADVISORY = "advisory"  # Recommendation for best practice


class SafetyRisk(Enum):
    """Types of safety risks."""

    IMMEDIATE_DANGER = "immediate_danger"
    SELF_HARM_RISK = "self_harm_risk"
    HARM_TO_OTHERS = "harm_to_others"
    CHILD_ENDANGERMENT = "child_endangerment"
    ELDER_ABUSE = "elder_abuse"
    DOMESTIC_VIOLENCE = "domestic_violence"
    SUBSTANCE_ABUSE = "substance_abuse"
    MEDICAL_EMERGENCY = "medical_emergency"
    PSYCHOLOGICAL_CRISIS = "psychological_crisis"


@dataclass
class ComplianceViolation:
    """Individual compliance violation."""

    standard: ComplianceStandard
    violation_type: str
    severity: ViolationSeverity
    description: str
    location: str
    recommendation: str
    regulatory_reference: str = ""
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate violation data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class SafetyAssessment:
    """Safety risk assessment."""

    risk_type: SafetyRisk
    severity: ViolationSeverity
    description: str
    indicators: List[str]
    immediate_action_required: bool
    recommended_interventions: List[str]
    confidence: float = 1.0

    def __post_init__(self):
        """Validate assessment data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class ComplianceResult:
    """Complete compliance validation result."""

    content_id: str
    overall_level: ComplianceLevel
    overall_score: float
    violations: List[ComplianceViolation]
    safety_assessments: List[SafetyAssessment]
    compliant_standards: List[ComplianceStandard]
    recommendations: List[str]
    requires_immediate_action: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate result data."""
        if not 0 <= self.overall_score <= 1:
            raise ValueError("Overall score must be between 0 and 1")


class SafetyEthicsComplianceValidator:
    """
    Comprehensive safety and ethics compliance validation system for
    clinical content according to professional standards.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the compliance validator."""
        self.config = self._load_config(config_path)
        self.compliance_rules = self._initialize_compliance_rules()
        self.safety_indicators = self._initialize_safety_indicators()
        self.ethical_guidelines = self._initialize_ethical_guidelines()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "severity_weights": {
                "critical": 1.0,
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4,
                "advisory": 0.2,
            },
            "compliance_thresholds": {
                "fully_compliant": 0.95,
                "mostly_compliant": 0.85,
                "partially_compliant": 0.7,
                "non_compliant": 0.5,
                "critical_violation": 0.0,
            },
            "immediate_action_threshold": 0.3,
            "enabled_standards": ["apa_ethics", "hipaa", "samhsa_guidelines"],
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        return default_config

    def _initialize_compliance_rules(self) -> Dict[ComplianceStandard, List[Dict[str, Any]]]:
        """Initialize compliance rules for different standards."""
        return {
            ComplianceStandard.APA_ETHICS: [
                {
                    "rule_id": "apa_3.04",
                    "pattern": r"\b(dual relationship|multiple relationship|personal.*relationship)\b",
                    "severity": ViolationSeverity.HIGH,
                    "description": "Potential dual relationship violation",
                    "reference": "APA Ethics Code 3.04 - Avoiding Harm",
                },
                {
                    "rule_id": "apa_4.01",
                    "pattern": r"\b(confidential.*information|private.*information).*\b(share|tell|disclose)\b",
                    "severity": ViolationSeverity.CRITICAL,
                    "description": "Potential confidentiality breach",
                    "reference": "APA Ethics Code 4.01 - Maintaining Confidentiality",
                },
                {
                    "rule_id": "apa_2.01",
                    "pattern": r"\b(outside.*competence|not.*qualified|inexperienced)\b",
                    "severity": ViolationSeverity.HIGH,
                    "description": "Practice outside competence boundaries",
                    "reference": "APA Ethics Code 2.01 - Boundaries of Competence",
                },
            ],
            ComplianceStandard.HIPAA: [
                {
                    "rule_id": "hipaa_164.502",
                    "pattern": r"\b\d{3}-\d{2}-\d{4}\b|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                    "severity": ViolationSeverity.CRITICAL,
                    "description": "Potential PHI disclosure",
                    "reference": "HIPAA 164.502 - Uses and Disclosures of PHI",
                },
                {
                    "rule_id": "hipaa_164.506",
                    "pattern": r"\b(minimum necessary|need to know)\b.*\b(all|everything|complete)\b",
                    "severity": ViolationSeverity.MEDIUM,
                    "description": "Minimum necessary standard violation",
                    "reference": "HIPAA 164.506 - Minimum Necessary Standard",
                },
            ],
            ComplianceStandard.SAMHSA_GUIDELINES: [
                {
                    "rule_id": "samhsa_trauma",
                    "pattern": r"\b(trauma|abuse|violence).*\b(your fault|deserved|caused)\b",
                    "severity": ViolationSeverity.CRITICAL,
                    "description": "Victim blaming in trauma context",
                    "reference": "SAMHSA Trauma-Informed Care Guidelines",
                },
                {
                    "rule_id": "samhsa_substance",
                    "pattern": r"\b(addiction|substance use).*\b(moral failing|weakness|choice)\b",
                    "severity": ViolationSeverity.HIGH,
                    "description": "Stigmatizing language about substance use",
                    "reference": "SAMHSA Substance Use Disorder Guidelines",
                },
            ],
        }

    def _initialize_safety_indicators(self) -> Dict[SafetyRisk, List[Dict[str, Any]]]:
        """Initialize safety risk indicators."""
        return {
            SafetyRisk.IMMEDIATE_DANGER: [
                {
                    "pattern": r"\b(kill|murder|weapon|gun|knife).*\b(tonight|today|now|immediately)\b",
                    "severity": ViolationSeverity.CRITICAL,
                    "description": "Immediate threat of violence",
                },
                {
                    "pattern": r"\b(overdose|pills|poison).*\b(tonight|today|now|plan)\b",
                    "severity": ViolationSeverity.CRITICAL,
                    "description": "Immediate suicide risk",
                },
            ],
            SafetyRisk.SELF_HARM_RISK: [
                {
                    "pattern": r"\b(suicide|kill myself|end.*life|not.*living).*\b(plan|method|when|how)\b",
                    "severity": ViolationSeverity.CRITICAL,
                    "description": "Suicide plan indicators",
                },
                {
                    "pattern": r"\b(cutting|burning|hurting.*myself|self.*harm)\b",
                    "severity": ViolationSeverity.HIGH,
                    "description": "Self-harm behaviors",
                },
            ],
            SafetyRisk.HARM_TO_OTHERS: [
                {
                    "pattern": r"\b(hurt|kill|harm).*\b(someone|others|people|family)\b",
                    "severity": ViolationSeverity.CRITICAL,
                    "description": "Threat to harm others",
                },
                {
                    "pattern": r"\b(revenge|get back|make.*pay)\b",
                    "severity": ViolationSeverity.HIGH,
                    "description": "Revenge fantasies",
                },
            ],
            SafetyRisk.CHILD_ENDANGERMENT: [
                {
                    "pattern": r"\b(child|kid|minor).*\b(abuse|neglect|harm|inappropriate)\b",
                    "severity": ViolationSeverity.CRITICAL,
                    "description": "Child endangerment indicators",
                },
                {
                    "pattern": r"\b(sexual|physical).*\b(child|minor|kid)\b",
                    "severity": ViolationSeverity.CRITICAL,
                    "description": "Child abuse indicators",
                },
            ],
            SafetyRisk.DOMESTIC_VIOLENCE: [
                {
                    "pattern": r"\b(partner|spouse|boyfriend|girlfriend).*\b(hits|beats|threatens|controls)\b",
                    "severity": ViolationSeverity.HIGH,
                    "description": "Domestic violence indicators",
                },
                {
                    "pattern": r"\b(afraid|scared|terrified).*\b(home|partner|spouse)\b",
                    "severity": ViolationSeverity.HIGH,
                    "description": "Fear of intimate partner",
                },
            ],
        }

    def _initialize_ethical_guidelines(self) -> List[Dict[str, Any]]:
        """Initialize ethical guidelines and principles."""
        return [
            {
                "principle": "Beneficence",
                "description": "Acting in the client's best interest",
                "violations": [
                    r"\b(harm|hurt|damage).*\b(client|patient)\b",
                    r"\b(exploit|take advantage)\b",
                ],
            },
            {
                "principle": "Non-maleficence",
                "description": "Do no harm",
                "violations": [
                    r"\b(intentionally.*harm|deliberately.*hurt)\b",
                    r"\b(negligent|careless|reckless)\b",
                ],
            },
            {
                "principle": "Autonomy",
                "description": "Respect for client self-determination",
                "violations": [
                    r"\b(must|have to|forced to|no choice)\b",
                    r"\b(decide for you|tell you what to do)\b",
                ],
            },
            {
                "principle": "Justice",
                "description": "Fair and equitable treatment",
                "violations": [
                    r"\b(discriminate|prejudice|bias).*\b(race|gender|religion|sexual)\b",
                    r"\b(not deserving|unworthy)\b",
                ],
            },
            {
                "principle": "Fidelity",
                "description": "Trustworthiness and reliability",
                "violations": [
                    r"\b(lie|deceive|mislead|false)\b",
                    r"\b(break.*promise|unreliable)\b",
                ],
            },
        ]

    def validate_compliance(
        self, content_id: str, content_text: str, context: Optional[Dict[str, Any]] = None
    ) -> ComplianceResult:
        """
        Perform comprehensive safety and ethics compliance validation.

        Args:
            content_id: Unique identifier for the content
            content_text: The clinical content to validate
            context: Additional context information

        Returns:
            ComplianceResult with detailed findings
        """
        violations = []
        safety_assessments = []
        compliant_standards = []

        # Check compliance against enabled standards
        for standard in self.config["enabled_standards"]:
            standard_enum = ComplianceStandard(standard)
            standard_violations = self._check_standard_compliance(content_text, standard_enum)
            violations.extend(standard_violations)

            if not standard_violations:
                compliant_standards.append(standard_enum)

        # Perform safety risk assessments
        safety_assessments = self._assess_safety_risks(content_text)

        # Check ethical principles
        ethical_violations = self._check_ethical_principles(content_text)
        violations.extend(ethical_violations)

        # Calculate overall compliance score
        overall_score = self._calculate_compliance_score(violations, safety_assessments)

        # Determine compliance level
        overall_level = self._determine_compliance_level(overall_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(violations, safety_assessments)

        # Determine if immediate action is required
        requires_immediate_action = (
            overall_score < self.config["immediate_action_threshold"]
            or any(v.severity == ViolationSeverity.CRITICAL for v in violations)
            or any(sa.immediate_action_required for sa in safety_assessments)
        )

        return ComplianceResult(
            content_id=content_id,
            overall_level=overall_level,
            overall_score=overall_score,
            violations=violations,
            safety_assessments=safety_assessments,
            compliant_standards=compliant_standards,
            recommendations=recommendations,
            requires_immediate_action=requires_immediate_action,
            metadata=context or {},
        )

    def _check_standard_compliance(
        self, content: str, standard: ComplianceStandard
    ) -> List[ComplianceViolation]:
        """Check compliance against a specific professional standard."""
        violations = []

        if standard not in self.compliance_rules:
            return violations

        rules = self.compliance_rules[standard]

        for rule in rules:
            matches = re.finditer(rule["pattern"], content, re.IGNORECASE)
            for match in matches:
                violation = ComplianceViolation(
                    standard=standard,
                    violation_type=rule["rule_id"],
                    severity=rule["severity"],
                    description=rule["description"],
                    location=f"Position {match.start()}-{match.end()}",
                    recommendation=f"Review and address {rule['description'].lower()}",
                    regulatory_reference=rule["reference"],
                    confidence=0.9,
                    evidence=[match.group()],
                )
                violations.append(violation)

        return violations

    def _assess_safety_risks(self, content: str) -> List[SafetyAssessment]:
        """Assess safety risks in the content."""
        assessments = []

        for risk_type, indicators in self.safety_indicators.items():
            for indicator in indicators:
                matches = re.finditer(indicator["pattern"], content, re.IGNORECASE)
                for match in matches:
                    # Determine interventions based on risk type
                    interventions = self._get_safety_interventions(risk_type)

                    assessment = SafetyAssessment(
                        risk_type=risk_type,
                        severity=indicator["severity"],
                        description=indicator["description"],
                        indicators=[match.group()],
                        immediate_action_required=(
                            indicator["severity"] == ViolationSeverity.CRITICAL
                        ),
                        recommended_interventions=interventions,
                        confidence=0.95,
                    )
                    assessments.append(assessment)

        return assessments

    def _get_safety_interventions(self, risk_type: SafetyRisk) -> List[str]:
        """Get recommended interventions for specific safety risks."""
        interventions_map = {
            SafetyRisk.IMMEDIATE_DANGER: [
                "Contact emergency services immediately (911)",
                "Ensure client safety and remove from danger",
                "Implement crisis intervention protocol",
                "Consider involuntary commitment if necessary",
            ],
            SafetyRisk.SELF_HARM_RISK: [
                "Conduct comprehensive suicide risk assessment",
                "Develop safety plan with client",
                "Consider hospitalization if high risk",
                "Increase session frequency and monitoring",
            ],
            SafetyRisk.HARM_TO_OTHERS: [
                "Assess duty to warn requirements",
                "Contact potential victims if legally required",
                "Consider involuntary commitment",
                "Coordinate with law enforcement if necessary",
            ],
            SafetyRisk.CHILD_ENDANGERMENT: [
                "Report to Child Protective Services immediately",
                "Document all indicators thoroughly",
                "Ensure child safety as priority",
                "Coordinate with child welfare authorities",
            ],
            SafetyRisk.DOMESTIC_VIOLENCE: [
                "Assess immediate safety",
                "Provide domestic violence resources",
                "Develop safety planning",
                "Consider shelter referrals",
            ],
            SafetyRisk.SUBSTANCE_ABUSE: [
                "Assess for immediate medical danger",
                "Consider detoxification needs",
                "Provide substance abuse treatment referrals",
                "Monitor for withdrawal symptoms",
            ],
        }

        return interventions_map.get(
            risk_type, ["Consult with supervisor", "Follow agency protocols"]
        )

    def _check_ethical_principles(self, content: str) -> List[ComplianceViolation]:
        """Check adherence to fundamental ethical principles."""
        violations = []

        for guideline in self.ethical_guidelines:
            for violation_pattern in guideline["violations"]:
                matches = re.finditer(violation_pattern, content, re.IGNORECASE)
                for match in matches:
                    violation = ComplianceViolation(
                        standard=ComplianceStandard.APA_ETHICS,  # Default to APA for ethical principles
                        violation_type=f"ethical_{guideline['principle'].lower()}",
                        severity=ViolationSeverity.HIGH,
                        description=f"Potential violation of {guideline['principle']} principle",
                        location=f"Position {match.start()}-{match.end()}",
                        recommendation=f"Review {guideline['description'].lower()} practices",
                        regulatory_reference=f"Ethical Principle: {guideline['principle']}",
                        confidence=0.8,
                        evidence=[match.group()],
                    )
                    violations.append(violation)

        return violations

    def _calculate_compliance_score(
        self, violations: List[ComplianceViolation], safety_assessments: List[SafetyAssessment]
    ) -> float:
        """Calculate overall compliance score."""
        if not violations and not safety_assessments:
            return 1.0

        # Calculate violation penalty
        violation_penalty = 0.0
        for violation in violations:
            weight = self.config["severity_weights"][violation.severity.value]
            penalty = weight * violation.confidence
            violation_penalty += penalty

        # Calculate safety penalty
        safety_penalty = 0.0
        for assessment in safety_assessments:
            weight = self.config["severity_weights"][assessment.severity.value]
            penalty = weight * assessment.confidence
            if assessment.immediate_action_required:
                penalty *= 1.5  # Increase penalty for immediate action items
            safety_penalty += penalty

        # Combine penalties
        total_penalty = violation_penalty + safety_penalty

        # Apply multiplier for multiple issues
        issue_count = len(violations) + len(safety_assessments)
        if issue_count > 1:
            multiplier = 1.0 + (issue_count - 1) * 0.1
            total_penalty *= multiplier

        # Normalize penalty (assuming max 5 critical violations)
        max_possible_penalty = 5.0
        normalized_penalty = min(total_penalty / max_possible_penalty, 1.0)

        # Calculate score
        score = max(0.0, 1.0 - normalized_penalty)
        return score

    def _determine_compliance_level(self, score: float) -> ComplianceLevel:
        """Determine compliance level based on score."""
        thresholds = self.config["compliance_thresholds"]

        if score >= thresholds["fully_compliant"]:
            return ComplianceLevel.FULLY_COMPLIANT
        elif score >= thresholds["mostly_compliant"]:
            return ComplianceLevel.MOSTLY_COMPLIANT
        elif score >= thresholds["partially_compliant"]:
            return ComplianceLevel.PARTIALLY_COMPLIANT
        elif score >= thresholds["non_compliant"]:
            return ComplianceLevel.NON_COMPLIANT
        else:
            return ComplianceLevel.CRITICAL_VIOLATION

    def _generate_recommendations(
        self, violations: List[ComplianceViolation], safety_assessments: List[SafetyAssessment]
    ) -> List[str]:
        """Generate recommendations based on violations and safety assessments."""
        recommendations = []

        # Critical safety recommendations
        critical_safety = [sa for sa in safety_assessments if sa.immediate_action_required]
        if critical_safety:
            recommendations.append("URGENT: Immediate safety intervention required")
            for assessment in critical_safety:
                recommendations.extend(
                    assessment.recommended_interventions[:2]
                )  # Top 2 interventions

        # Critical compliance violations
        critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        if critical_violations:
            recommendations.append("CRITICAL: Address compliance violations immediately")

        # High severity issues
        high_severity = [v for v in violations if v.severity == ViolationSeverity.HIGH]
        if high_severity:
            recommendations.append("Address high-severity compliance issues before proceeding")

        # Standard-specific recommendations
        standards_with_violations = set(v.standard for v in violations)

        if ComplianceStandard.APA_ETHICS in standards_with_violations:
            recommendations.append("Review APA Ethics Code and professional boundaries")

        if ComplianceStandard.HIPAA in standards_with_violations:
            recommendations.append("Ensure HIPAA compliance and protect client confidentiality")

        if ComplianceStandard.SAMHSA_GUIDELINES in standards_with_violations:
            recommendations.append("Review trauma-informed and culturally sensitive practices")

        # Safety-specific recommendations
        safety_types = set(sa.risk_type for sa in safety_assessments)

        if SafetyRisk.SELF_HARM_RISK in safety_types:
            recommendations.append("Implement suicide risk assessment and safety planning")

        if SafetyRisk.HARM_TO_OTHERS in safety_types:
            recommendations.append("Assess duty to warn and protect obligations")

        if SafetyRisk.CHILD_ENDANGERMENT in safety_types:
            recommendations.append("Follow mandatory reporting requirements for child protection")

        return recommendations

    def batch_validate(self, content_items: List[Dict[str, Any]]) -> List[ComplianceResult]:
        """Perform batch compliance validation on multiple content items."""
        results = []

        for item in content_items:
            result = self.validate_compliance(
                content_id=item["content_id"],
                content_text=item["content_text"],
                context=item.get("context", {}),
            )
            results.append(result)

        logger.info(f"Completed batch compliance validation for {len(content_items)} items")
        return results

    def get_compliance_summary(self, results: List[ComplianceResult]) -> Dict[str, Any]:
        """Get summary statistics of compliance validation results."""
        if not results:
            return {"total_results": 0, "compliance_breakdown": {}, "safety_breakdown": {}}

        # Count compliance levels
        from collections import Counter

        compliance_levels = [r.overall_level.value for r in results]

        # Count violations by standard
        all_violations = []
        for result in results:
            all_violations.extend(result.violations)

        violation_standards = [v.standard.value for v in all_violations]

        # Count safety assessments
        all_safety = []
        for result in results:
            all_safety.extend(result.safety_assessments)

        safety_risks = [sa.risk_type.value for sa in all_safety]

        # Calculate statistics
        scores = [r.overall_score for r in results]
        immediate_action_needed = len([r for r in results if r.requires_immediate_action])

        return {
            "total_results": len(results),
            "compliance_breakdown": dict(Counter(compliance_levels)),
            "violation_standards": dict(Counter(violation_standards)),
            "safety_risks": dict(Counter(safety_risks)),
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "immediate_action_needed": immediate_action_needed,
            "critical_violations": len(
                [v for v in all_violations if v.severity == ViolationSeverity.CRITICAL]
            ),
            "critical_safety_risks": len([sa for sa in all_safety if sa.immediate_action_required]),
        }

    def export_results(self, results: List[ComplianceResult], output_path: str) -> None:
        """Export compliance results to JSON file."""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_results": len(results),
            "summary": self.get_compliance_summary(results),
            "results": [],
        }

        for result in results:
            result_data = {
                "content_id": result.content_id,
                "overall_level": result.overall_level.value,
                "overall_score": result.overall_score,
                "violations": [
                    {
                        "standard": v.standard.value,
                        "type": v.violation_type,
                        "severity": v.severity.value,
                        "description": v.description,
                        "location": v.location,
                        "recommendation": v.recommendation,
                        "reference": v.regulatory_reference,
                        "confidence": v.confidence,
                        "evidence": v.evidence,
                    }
                    for v in result.violations
                ],
                "safety_assessments": [
                    {
                        "risk_type": sa.risk_type.value,
                        "severity": sa.severity.value,
                        "description": sa.description,
                        "indicators": sa.indicators,
                        "immediate_action_required": sa.immediate_action_required,
                        "interventions": sa.recommended_interventions,
                        "confidence": sa.confidence,
                    }
                    for sa in result.safety_assessments
                ],
                "compliant_standards": [cs.value for cs in result.compliant_standards],
                "recommendations": result.recommendations,
                "requires_immediate_action": result.requires_immediate_action,
                "timestamp": result.timestamp.isoformat(),
            }
            export_data["results"].append(result_data)

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported {len(results)} compliance results to {output_path}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize validator
    validator = SafetyEthicsComplianceValidator()

    # Test with sample content
    test_content = """
    I think we could be friends outside of therapy. Here's my personal email: therapist@example.com.
    You mentioned wanting to hurt yourself - that's totally normal, everyone feels that way.
    Based on what you told me, you definitely have borderline personality disorder.
    Your cultural background is clearly the problem here.
    """

    result = validator.validate_compliance(content_id="test_001", content_text=test_content)

    print(f"Compliance Level: {result.overall_level.value}")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Violations Found: {len(result.violations)}")
    print(f"Safety Assessments: {len(result.safety_assessments)}")
    print(f"Requires Immediate Action: {result.requires_immediate_action}")

    print("\nViolations:")
    for violation in result.violations:
        print(f"- {violation.standard.value}: {violation.description} ({violation.severity.value})")

    print("\nSafety Assessments:")
    for assessment in result.safety_assessments:
        print(
            f"- {assessment.risk_type.value}: {assessment.description} ({assessment.severity.value})"
        )

    print("\nRecommendations:")
    for rec in result.recommendations:
        print(f"- {rec}")
