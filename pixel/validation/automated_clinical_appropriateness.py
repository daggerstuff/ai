"""
Automated Clinical Appropriateness Checking System

This module provides automated validation of clinical content for appropriateness,
safety, and adherence to professional standards before expert review.
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
import json
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppropriatenessLevel(Enum):
    """Clinical appropriateness levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    CONCERNING = "concerning"
    INAPPROPRIATE = "inappropriate"


class ViolationType(Enum):
    """Types of clinical appropriateness violations."""

    BOUNDARY_VIOLATION = "boundary_violation"
    CRISIS_MISHANDLING = "crisis_mishandling"
    DIAGNOSTIC_ERROR = "diagnostic_error"
    THERAPEUTIC_OVERREACH = "therapeutic_overreach"
    CULTURAL_INSENSITIVITY = "cultural_insensitivity"
    ETHICAL_VIOLATION = "ethical_violation"
    SAFETY_CONCERN = "safety_concern"
    SCOPE_VIOLATION = "scope_violation"
    CONFIDENTIALITY_BREACH = "confidentiality_breach"
    DUAL_RELATIONSHIP = "dual_relationship"


class SeverityLevel(Enum):
    """Severity levels for violations."""

    CRITICAL = "critical"  # Immediate danger or severe ethical violation
    HIGH = "high"  # Significant concern requiring attention
    MEDIUM = "medium"  # Moderate concern, should be addressed
    LOW = "low"  # Minor issue, good practice to fix
    INFO = "info"  # Informational, not necessarily problematic


@dataclass
class AppropriatenessViolation:
    """Individual appropriateness violation."""

    violation_type: ViolationType
    severity: SeverityLevel
    description: str
    location: str  # Where in the content the violation occurs
    recommendation: str
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate violation data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class AppropriatenessResult:
    """Complete appropriateness assessment result."""

    content_id: str
    overall_level: AppropriatenessLevel
    overall_score: float
    violations: List[AppropriatenessViolation]
    passed_checks: List[str]
    recommendations: List[str]
    requires_expert_review: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate result data."""
        if not 0 <= self.overall_score <= 1:
            raise ValueError("Overall score must be between 0 and 1")


class ClinicalAppropriatenessChecker:
    """
    Automated clinical appropriateness checking system that validates
    therapeutic content against professional standards and safety guidelines.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the clinical appropriateness checker."""
        self.config = self._load_config(config_path)
        self.violation_patterns = self._initialize_violation_patterns()
        self.safety_keywords = self._initialize_safety_keywords()
        self.boundary_indicators = self._initialize_boundary_indicators()
        self.crisis_indicators = self._initialize_crisis_indicators()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "severity_weights": {
                "critical": 1.0,
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4,
                "info": 0.2,
            },
            "appropriateness_thresholds": {
                "excellent": 0.9,
                "good": 0.8,
                "acceptable": 0.7,
                "concerning": 0.5,
                "inappropriate": 0.0,
            },
            "expert_review_threshold": 0.7,
            "auto_reject_threshold": 0.3,
            "confidence_threshold": 0.8,
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        return default_config

    def _initialize_violation_patterns(self) -> Dict[ViolationType, List[Dict[str, Any]]]:
        """Initialize patterns for detecting various violation types."""
        return {
            ViolationType.BOUNDARY_VIOLATION: [
                {
                    "pattern": r"\b(personal relationship|friendship|dating|romantic|sexual|friends|friend)\b",
                    "severity": SeverityLevel.CRITICAL,
                    "description": "Inappropriate personal relationship language",
                },
                {
                    "pattern": r"\b(meet outside|personal contact|phone number|social media|outside.*therapy)\b",
                    "severity": SeverityLevel.HIGH,
                    "description": "Boundary crossing communication",
                },
                {
                    "pattern": r"\b(you remind me of|like my|similar to my)\b",
                    "severity": SeverityLevel.MEDIUM,
                    "description": "Inappropriate personal disclosure",
                },
            ],
            ViolationType.CRISIS_MISHANDLING: [
                {
                    "pattern": r"\b(suicide|kill myself|end it all|not worth living).*\b(that's normal|everyone feels|just relax|totally normal)\b",
                    "severity": SeverityLevel.CRITICAL,
                    "description": "Minimizing suicidal ideation",
                },
                {
                    "pattern": r"\b(self-harm|cutting|hurting myself).*\b(stop doing that|just don't|try not to)\b",
                    "severity": SeverityLevel.HIGH,
                    "description": "Inadequate self-harm response",
                },
            ],
            ViolationType.DIAGNOSTIC_ERROR: [
                {
                    "pattern": r"\byou (definitely have|clearly have|obviously have|have).*\b(bipolar|schizophrenia|personality disorder|depression|anxiety|PTSD)\b",
                    "severity": SeverityLevel.HIGH,
                    "description": "Premature or definitive diagnosis",
                },
                {
                    "pattern": r"\byou don't have.*\b(depression|anxiety|PTSD|trauma)\b",
                    "severity": SeverityLevel.MEDIUM,
                    "description": "Dismissive diagnostic language",
                },
            ],
            ViolationType.THERAPEUTIC_OVERREACH: [
                {
                    "pattern": r"\byou (should|must|need to).*(leave|divorce|quit|break up|immediately)",
                    "severity": SeverityLevel.HIGH,
                    "description": "Directive advice about major life decisions",
                },
                {
                    "pattern": r"\b(take this medication|stop taking|increase your dose|medications I'm recommending)\b",
                    "severity": SeverityLevel.CRITICAL,
                    "description": "Unauthorized medication advice",
                },
            ],
            ViolationType.CULTURAL_INSENSITIVITY: [
                {
                    "pattern": r"\b(your culture|your people|typical for|common in your).*\b(problem|issue|wrong)\b",
                    "severity": SeverityLevel.HIGH,
                    "description": "Cultural stereotyping or bias",
                },
                {
                    "pattern": r"\b(just assimilate|fit in better|be more American|assimilate better)\b",
                    "severity": SeverityLevel.MEDIUM,
                    "description": "Cultural insensitivity",
                },
            ],
            ViolationType.ETHICAL_VIOLATION: [
                {
                    "pattern": r"\b(don't tell anyone|keep this secret|between us|confidential).*\b(illegal|harmful|dangerous)\b",
                    "severity": SeverityLevel.CRITICAL,
                    "description": "Inappropriate confidentiality handling",
                },
                {
                    "pattern": r"\b(I'll help you|we can work together).*\b(hide|cover up|not report)\b",
                    "severity": SeverityLevel.CRITICAL,
                    "description": "Collusion in harmful behavior",
                },
            ],
            ViolationType.SAFETY_CONCERN: [
                {
                    "pattern": r"\b(violence|hitting|abuse).*\b(normal|acceptable|understandable)\b",
                    "severity": SeverityLevel.CRITICAL,
                    "description": "Normalizing violence or abuse",
                },
                {
                    "pattern": r"\b(substance use|drinking|drugs).*\b(not a problem|fine|okay)\b",
                    "severity": SeverityLevel.MEDIUM,
                    "description": "Minimizing substance use concerns",
                },
            ],
        }

    def _initialize_safety_keywords(self) -> Dict[str, SeverityLevel]:
        """Initialize safety-related keywords and their severity levels."""
        return {
            # Critical safety keywords
            "suicide": SeverityLevel.CRITICAL,
            "kill myself": SeverityLevel.CRITICAL,
            "end my life": SeverityLevel.CRITICAL,
            "homicide": SeverityLevel.CRITICAL,
            "murder": SeverityLevel.CRITICAL,
            "child abuse": SeverityLevel.CRITICAL,
            "sexual abuse": SeverityLevel.CRITICAL,
            # High severity keywords
            "self-harm": SeverityLevel.HIGH,
            "cutting": SeverityLevel.HIGH,
            "overdose": SeverityLevel.HIGH,
            "domestic violence": SeverityLevel.HIGH,
            "rape": SeverityLevel.HIGH,
            # Medium severity keywords
            "depression": SeverityLevel.MEDIUM,
            "anxiety": SeverityLevel.MEDIUM,
            "panic": SeverityLevel.MEDIUM,
            "trauma": SeverityLevel.MEDIUM,
            "PTSD": SeverityLevel.MEDIUM,
        }

    def _initialize_boundary_indicators(self) -> List[str]:
        """Initialize boundary violation indicators."""
        return [
            "personal information",
            "outside of session",
            "social relationship",
            "dual relationship",
            "gift giving",
            "physical contact",
            "personal disclosure",
            "role confusion",
            "friendship",
            "romantic feelings",
        ]

    def _initialize_crisis_indicators(self) -> List[str]:
        """Initialize crisis situation indicators."""
        return [
            "suicidal thoughts",
            "suicide plan",
            "self-harm",
            "homicidal ideation",
            "psychosis",
            "severe depression",
            "manic episode",
            "substance overdose",
            "domestic violence",
            "child endangerment",
        ]

    def check_appropriateness(
        self, content_id: str, content_text: str, context: Optional[Dict[str, Any]] = None
    ) -> AppropriatenessResult:
        """
        Perform comprehensive appropriateness checking on clinical content.

        Args:
            content_id: Unique identifier for the content
            content_text: The clinical content to check
            context: Additional context information

        Returns:
            AppropriatenessResult with detailed findings
        """
        violations = []
        passed_checks = []

        # Run all appropriateness checks
        violations.extend(self._check_boundary_violations(content_text))
        violations.extend(self._check_crisis_handling(content_text))
        violations.extend(self._check_diagnostic_appropriateness(content_text))
        violations.extend(self._check_therapeutic_scope(content_text))
        violations.extend(self._check_cultural_sensitivity(content_text))
        violations.extend(self._check_ethical_compliance(content_text))
        violations.extend(self._check_safety_concerns(content_text))
        violations.extend(self._check_confidentiality(content_text))

        # Track passed checks
        if not any(v.violation_type == ViolationType.BOUNDARY_VIOLATION for v in violations):
            passed_checks.append("Boundary appropriateness")
        if not any(v.violation_type == ViolationType.CRISIS_MISHANDLING for v in violations):
            passed_checks.append("Crisis handling")
        if not any(v.violation_type == ViolationType.DIAGNOSTIC_ERROR for v in violations):
            passed_checks.append("Diagnostic appropriateness")
        if not any(v.violation_type == ViolationType.THERAPEUTIC_OVERREACH for v in violations):
            passed_checks.append("Therapeutic scope")
        if not any(v.violation_type == ViolationType.CULTURAL_INSENSITIVITY for v in violations):
            passed_checks.append("Cultural sensitivity")
        if not any(v.violation_type == ViolationType.ETHICAL_VIOLATION for v in violations):
            passed_checks.append("Ethical compliance")
        if not any(v.violation_type == ViolationType.SAFETY_CONCERN for v in violations):
            passed_checks.append("Safety standards")
        if not any(v.violation_type == ViolationType.CONFIDENTIALITY_BREACH for v in violations):
            passed_checks.append("Confidentiality")

        # Calculate overall score
        overall_score = self._calculate_overall_score(violations)

        # Determine appropriateness level
        overall_level = self._determine_appropriateness_level(overall_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(violations, overall_score)

        # Determine if expert review is required
        requires_expert_review = overall_score < self.config["expert_review_threshold"] or any(
            v.severity == SeverityLevel.CRITICAL for v in violations
        )

        return AppropriatenessResult(
            content_id=content_id,
            overall_level=overall_level,
            overall_score=overall_score,
            violations=violations,
            passed_checks=passed_checks,
            recommendations=recommendations,
            requires_expert_review=requires_expert_review,
            metadata=context or {},
        )

    def _check_boundary_violations(self, content: str) -> List[AppropriatenessViolation]:
        """Check for therapeutic boundary violations."""
        violations = []
        patterns = self.violation_patterns[ViolationType.BOUNDARY_VIOLATION]

        for pattern_info in patterns:
            matches = re.finditer(pattern_info["pattern"], content, re.IGNORECASE)
            for match in matches:
                violation = AppropriatenessViolation(
                    violation_type=ViolationType.BOUNDARY_VIOLATION,
                    severity=pattern_info["severity"],
                    description=pattern_info["description"],
                    location=f"Position {match.start()}-{match.end()}",
                    recommendation="Review and revise to maintain appropriate therapeutic boundaries",
                    confidence=0.9,
                    evidence=[match.group()],
                )
                violations.append(violation)

        return violations

    def _check_crisis_handling(self, content: str) -> List[AppropriatenessViolation]:
        """Check for appropriate crisis situation handling."""
        violations = []
        patterns = self.violation_patterns[ViolationType.CRISIS_MISHANDLING]

        for pattern_info in patterns:
            matches = re.finditer(pattern_info["pattern"], content, re.IGNORECASE)
            for match in matches:
                violation = AppropriatenessViolation(
                    violation_type=ViolationType.CRISIS_MISHANDLING,
                    severity=pattern_info["severity"],
                    description=pattern_info["description"],
                    location=f"Position {match.start()}-{match.end()}",
                    recommendation="Ensure appropriate crisis intervention protocols are followed",
                    confidence=0.95,
                    evidence=[match.group()],
                )
                violations.append(violation)

        return violations

    def _check_diagnostic_appropriateness(self, content: str) -> List[AppropriatenessViolation]:
        """Check for appropriate diagnostic language and practices."""
        violations = []
        patterns = self.violation_patterns[ViolationType.DIAGNOSTIC_ERROR]

        for pattern_info in patterns:
            matches = re.finditer(pattern_info["pattern"], content, re.IGNORECASE)
            for match in matches:
                violation = AppropriatenessViolation(
                    violation_type=ViolationType.DIAGNOSTIC_ERROR,
                    severity=pattern_info["severity"],
                    description=pattern_info["description"],
                    location=f"Position {match.start()}-{match.end()}",
                    recommendation="Use tentative, collaborative language for diagnostic impressions",
                    confidence=0.85,
                    evidence=[match.group()],
                )
                violations.append(violation)

        return violations

    def _check_therapeutic_scope(self, content: str) -> List[AppropriatenessViolation]:
        """Check for therapeutic scope violations."""
        violations = []
        patterns = self.violation_patterns[ViolationType.THERAPEUTIC_OVERREACH]

        for pattern_info in patterns:
            matches = re.finditer(pattern_info["pattern"], content, re.IGNORECASE)
            for match in matches:
                violation = AppropriatenessViolation(
                    violation_type=ViolationType.THERAPEUTIC_OVERREACH,
                    severity=pattern_info["severity"],
                    description=pattern_info["description"],
                    location=f"Position {match.start()}-{match.end()}",
                    recommendation="Stay within appropriate therapeutic scope and avoid directive advice",
                    confidence=0.9,
                    evidence=[match.group()],
                )
                violations.append(violation)

        return violations

    def _check_cultural_sensitivity(self, content: str) -> List[AppropriatenessViolation]:
        """Check for cultural sensitivity issues."""
        violations = []
        patterns = self.violation_patterns[ViolationType.CULTURAL_INSENSITIVITY]

        for pattern_info in patterns:
            matches = re.finditer(pattern_info["pattern"], content, re.IGNORECASE)
            for match in matches:
                violation = AppropriatenessViolation(
                    violation_type=ViolationType.CULTURAL_INSENSITIVITY,
                    severity=pattern_info["severity"],
                    description=pattern_info["description"],
                    location=f"Position {match.start()}-{match.end()}",
                    recommendation="Ensure culturally sensitive and respectful language",
                    confidence=0.8,
                    evidence=[match.group()],
                )
                violations.append(violation)

        return violations

    def _check_ethical_compliance(self, content: str) -> List[AppropriatenessViolation]:
        """Check for ethical compliance issues."""
        violations = []
        patterns = self.violation_patterns[ViolationType.ETHICAL_VIOLATION]

        for pattern_info in patterns:
            matches = re.finditer(pattern_info["pattern"], content, re.IGNORECASE)
            for match in matches:
                violation = AppropriatenessViolation(
                    violation_type=ViolationType.ETHICAL_VIOLATION,
                    severity=pattern_info["severity"],
                    description=pattern_info["description"],
                    location=f"Position {match.start()}-{match.end()}",
                    recommendation="Review ethical guidelines and professional standards",
                    confidence=0.95,
                    evidence=[match.group()],
                )
                violations.append(violation)

        return violations

    def _check_safety_concerns(self, content: str) -> List[AppropriatenessViolation]:
        """Check for safety-related concerns."""
        violations = []
        patterns = self.violation_patterns[ViolationType.SAFETY_CONCERN]

        for pattern_info in patterns:
            matches = re.finditer(pattern_info["pattern"], content, re.IGNORECASE)
            for match in matches:
                violation = AppropriatenessViolation(
                    violation_type=ViolationType.SAFETY_CONCERN,
                    severity=pattern_info["severity"],
                    description=pattern_info["description"],
                    location=f"Position {match.start()}-{match.end()}",
                    recommendation="Address safety concerns appropriately and consider risk assessment",
                    confidence=0.9,
                    evidence=[match.group()],
                )
                violations.append(violation)

        return violations

    def _check_confidentiality(self, content: str) -> List[AppropriatenessViolation]:
        """Check for confidentiality breaches."""
        violations = []

        # Check for specific identifying information
        patterns = [
            (r"\b\d{3}-\d{2}-\d{4}\b", "Social Security Number"),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "Email address"),
            (r"\b\d{3}-\d{3}-\d{4}\b", "Phone number"),
            (
                r"\b\d{1,5}\s\w+\s(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b",
                "Street address",
            ),
        ]

        for pattern, info_type in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                violation = AppropriatenessViolation(
                    violation_type=ViolationType.CONFIDENTIALITY_BREACH,
                    severity=SeverityLevel.HIGH,
                    description=f"Potential {info_type} disclosure",
                    location=f"Position {match.start()}-{match.end()}",
                    recommendation="Remove or redact identifying information",
                    confidence=0.95,
                    evidence=[match.group()],
                )
                violations.append(violation)

        return violations

    def _calculate_overall_score(self, violations: List[AppropriatenessViolation]) -> float:
        """Calculate overall appropriateness score based on violations."""
        if not violations:
            return 1.0

        # Calculate weighted penalty based on severity
        total_penalty = 0.0
        for violation in violations:
            weight = self.config["severity_weights"][violation.severity.value]
            penalty = weight * violation.confidence
            total_penalty += penalty

        # More aggressive penalty scaling for multiple violations
        violation_count_multiplier = (
            1.0 + (len(violations) - 1) * 0.1
        )  # 10% penalty per additional violation
        total_penalty *= violation_count_multiplier

        # Normalize penalty (assuming max 5 violations of critical severity)
        max_possible_penalty = 5.0
        normalized_penalty = min(total_penalty / max_possible_penalty, 1.0)

        # Calculate score (1.0 - penalty)
        score = max(0.0, 1.0 - normalized_penalty)
        return score

    def _determine_appropriateness_level(self, score: float) -> AppropriatenessLevel:
        """Determine appropriateness level based on score."""
        thresholds = self.config["appropriateness_thresholds"]

        if score >= thresholds["excellent"]:
            return AppropriatenessLevel.EXCELLENT
        elif score >= thresholds["good"]:
            return AppropriatenessLevel.GOOD
        elif score >= thresholds["acceptable"]:
            return AppropriatenessLevel.ACCEPTABLE
        elif score >= thresholds["concerning"]:
            return AppropriatenessLevel.CONCERNING
        else:
            return AppropriatenessLevel.INAPPROPRIATE

    def _generate_recommendations(
        self, violations: List[AppropriatenessViolation], overall_score: float
    ) -> List[str]:
        """Generate recommendations based on violations and score."""
        recommendations = []

        # Critical violations
        critical_violations = [v for v in violations if v.severity == SeverityLevel.CRITICAL]
        if critical_violations:
            recommendations.append(
                "URGENT: Address critical violations immediately before proceeding"
            )

        # High severity violations
        high_violations = [v for v in violations if v.severity == SeverityLevel.HIGH]
        if high_violations:
            recommendations.append("Address high-severity concerns before expert review")

        # Specific violation type recommendations
        violation_types = set(v.violation_type for v in violations)

        if ViolationType.BOUNDARY_VIOLATION in violation_types:
            recommendations.append(
                "Review therapeutic boundary guidelines and maintain professional relationships"
            )

        if ViolationType.CRISIS_MISHANDLING in violation_types:
            recommendations.append("Ensure proper crisis intervention protocols are followed")

        if ViolationType.DIAGNOSTIC_ERROR in violation_types:
            recommendations.append(
                "Use collaborative, tentative language for diagnostic impressions"
            )

        if ViolationType.THERAPEUTIC_OVERREACH in violation_types:
            recommendations.append(
                "Stay within appropriate therapeutic scope and avoid directive advice"
            )

        if ViolationType.CULTURAL_INSENSITIVITY in violation_types:
            recommendations.append(
                "Enhance cultural competency and use respectful, inclusive language"
            )

        if ViolationType.ETHICAL_VIOLATION in violation_types:
            recommendations.append("Review professional ethical guidelines and standards")

        if ViolationType.SAFETY_CONCERN in violation_types:
            recommendations.append("Prioritize client safety and consider risk assessment")

        if ViolationType.CONFIDENTIALITY_BREACH in violation_types:
            recommendations.append(
                "Remove identifying information and protect client confidentiality"
            )

        # Overall score recommendations
        if overall_score < 0.5:
            recommendations.append("Significant revision required before clinical use")
        elif overall_score < 0.7:
            recommendations.append("Moderate improvements needed for clinical appropriateness")
        elif overall_score < 0.9:
            recommendations.append("Minor refinements would improve clinical quality")

        return recommendations

    def batch_check(self, content_items: List[Dict[str, Any]]) -> List[AppropriatenessResult]:
        """Perform batch appropriateness checking on multiple content items."""
        results = []

        for item in content_items:
            result = self.check_appropriateness(
                content_id=item["content_id"],
                content_text=item["content_text"],
                context=item.get("context", {}),
            )
            results.append(result)

        logger.info(f"Completed batch appropriateness checking for {len(content_items)} items")
        return results

    def get_violation_summary(self, results: List[AppropriatenessResult]) -> Dict[str, Any]:
        """Get summary statistics of violations across multiple results."""
        all_violations = []
        for result in results:
            all_violations.extend(result.violations)

        if not all_violations:
            return {"total_violations": 0, "violation_breakdown": {}, "severity_breakdown": {}}

        # Count violations by type
        from collections import Counter

        violation_types = [v.violation_type.value for v in all_violations]
        severity_levels = [v.severity.value for v in all_violations]

        return {
            "total_violations": len(all_violations),
            "violation_breakdown": dict(Counter(violation_types)),
            "severity_breakdown": dict(Counter(severity_levels)),
            "average_confidence": np.mean([v.confidence for v in all_violations]),
            "critical_violations": len(
                [v for v in all_violations if v.severity == SeverityLevel.CRITICAL]
            ),
        }

    def export_results(self, results: List[AppropriatenessResult], output_path: str) -> None:
        """Export appropriateness results to JSON file."""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_results": len(results),
            "summary": self.get_violation_summary(results),
            "results": [],
        }

        for result in results:
            result_data = {
                "content_id": result.content_id,
                "overall_level": result.overall_level.value,
                "overall_score": result.overall_score,
                "violations": [
                    {
                        "type": v.violation_type.value,
                        "severity": v.severity.value,
                        "description": v.description,
                        "location": v.location,
                        "recommendation": v.recommendation,
                        "confidence": v.confidence,
                        "evidence": v.evidence,
                    }
                    for v in result.violations
                ],
                "passed_checks": result.passed_checks,
                "recommendations": result.recommendations,
                "requires_expert_review": result.requires_expert_review,
                "timestamp": result.timestamp.isoformat(),
            }
            export_data["results"].append(result_data)

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported {len(results)} appropriateness results to {output_path}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize checker
    checker = ClinicalAppropriatenessChecker()

    # Test with sample content
    test_content = """
    I understand you're feeling depressed. You definitely have major depression based on what you've told me.
    You should leave your husband immediately - he's clearly the problem. 
    I think we could be good friends outside of therapy. Here's my personal phone number.
    """

    result = checker.check_appropriateness(content_id="test_001", content_text=test_content)

    print(f"Overall Level: {result.overall_level.value}")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Violations Found: {len(result.violations)}")
    print(f"Requires Expert Review: {result.requires_expert_review}")

    for violation in result.violations:
        print(
            f"- {violation.violation_type.value} ({violation.severity.value}): {violation.description}"
        )

    print("\nRecommendations:")
    for rec in result.recommendations:
        print(f"- {rec}")
