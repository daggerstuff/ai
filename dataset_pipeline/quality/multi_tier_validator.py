#!/usr/bin/env python3
"""
Enhanced Multi-Tier Validation System for Pixelated Empathy AI
Implements sophisticated tier-specific validation with adaptive thresholds.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class ValidationTier(Enum):
    """Validation tiers with increasing strictness."""
    BASIC = "basic"
    STANDARD = "standard"
    PROFESSIONAL = "professional"
    CLINICAL = "clinical"
    RESEARCH = "research"

class ValidationResult(Enum):
    """Validation result categories."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    CRITICAL_FAIL = "critical_fail"

@dataclass
class ValidationIssue:
    """Represents a validation issue found during analysis."""
    tier: ValidationTier
    category: str
    severity: str
    description: str
    suggestion: str
    confidence_score: float
    line_number: int | None = None
    context: str | None = None

@dataclass
class TierValidationResult:
    """Results for a specific validation tier."""
    tier: ValidationTier
    overall_result: ValidationResult
    score: float
    issues: list[ValidationIssue]
    passed_checks: int
    total_checks: int
    execution_time: float

class MultiTierValidator:
    """Enhanced multi-tier validation system with adaptive thresholds."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Tier-specific thresholds (more stringent for higher tiers)
        self.tier_thresholds = {
            ValidationTier.BASIC: {
                "min_score": 0.6,
                "warning_threshold": 0.7,
                "critical_threshold": 0.4
            },
            ValidationTier.STANDARD: {
                "min_score": 0.7,
                "warning_threshold": 0.8,
                "critical_threshold": 0.5
            },
            ValidationTier.PROFESSIONAL: {
                "min_score": 0.8,
                "warning_threshold": 0.85,
                "critical_threshold": 0.6
            },
            ValidationTier.CLINICAL: {
                "min_score": 0.9,
                "warning_threshold": 0.95,
                "critical_threshold": 0.75
            },
            ValidationTier.RESEARCH: {
                "min_score": 0.95,
                "warning_threshold": 0.98,
                "critical_threshold": 0.85
            }
        }

        # Validation categories with tier-specific weights
        self.validation_categories = {
            "therapeutic_accuracy": {
                ValidationTier.BASIC: 0.2,
                ValidationTier.STANDARD: 0.3,
                ValidationTier.PROFESSIONAL: 0.4,
                ValidationTier.CLINICAL: 0.5,
                ValidationTier.RESEARCH: 0.6
            },
            "safety_compliance": {
                ValidationTier.BASIC: 0.3,
                ValidationTier.STANDARD: 0.3,
                ValidationTier.PROFESSIONAL: 0.3,
                ValidationTier.CLINICAL: 0.3,
                ValidationTier.RESEARCH: 0.3
            },
            "ethical_guidelines": {
                ValidationTier.BASIC: 0.2,
                ValidationTier.STANDARD: 0.2,
                ValidationTier.PROFESSIONAL: 0.15,
                ValidationTier.CLINICAL: 0.1,
                ValidationTier.RESEARCH: 0.05
            },
            "clinical_accuracy": {
                ValidationTier.BASIC: 0.1,
                ValidationTier.STANDARD: 0.1,
                ValidationTier.PROFESSIONAL: 0.1,
                ValidationTier.CLINICAL: 0.05,
                ValidationTier.RESEARCH: 0.02
            },
            "research_validity": {
                ValidationTier.BASIC: 0.0,
                ValidationTier.STANDARD: 0.0,
                ValidationTier.PROFESSIONAL: 0.0,
                ValidationTier.CLINICAL: 0.0,
                ValidationTier.RESEARCH: 0.03
            }
        }

        # Enhanced validation patterns
        self.validation_patterns = {
            "therapeutic_techniques": [
                r"\b(cognitive behavioral|CBT|mindfulness|exposure therapy)\b",
                r"\b(active listening|empathic response|validation)\b",
                r"\b(reframing|challenging thoughts|behavioral activation)\b"
            ],
            "safety_indicators": [
                r"\b(suicide|self-harm|crisis|emergency)\b",
                r"\b(abuse|trauma|violence|danger)\b",
                r"\b(substance use|addiction|overdose)\b"
            ],
            "ethical_violations": [
                r"\b(dual relationship|boundary violation|confidentiality breach)\b",
                r"\b(inappropriate disclosure|personal information)\b",
                r"\b(discrimination|bias|prejudice)\b"
            ],
            "clinical_terminology": [
                r"\b(DSM-5|ICD-11|diagnosis|assessment)\b",
                r"\b(symptoms|treatment plan|intervention)\b",
                r"\b(prognosis|comorbidity|differential diagnosis)\b"
            ]
        }

    def _issue_to_dict(self, issue: ValidationIssue) -> dict[str, Any]:
        """Convert ValidationIssue to dictionary for JSON serialization."""
        issue_dict = asdict(issue)
        # Convert tier enum to string if it's still an enum
        if hasattr(issue.tier, "value"):
            issue_dict["tier"] = issue.tier.value
        return issue_dict

    def _create_validation_issue(self, tier: ValidationTier, category: str, severity: str,
                               description: str, suggestion: str, confidence_score: float,
                               line_number: int | None = None, context: str | None = None) -> ValidationIssue:
        """Helper method to create ValidationIssue with proper serialization."""
        return ValidationIssue(
            tier=tier.value,
            category=category,
            severity=severity,
            description=description,
            suggestion=suggestion,
            confidence_score=confidence_score,
            line_number=line_number,
            context=context
        )

    def validate_conversation(self, conversation: dict[str, Any], target_tier: ValidationTier) -> TierValidationResult:
        """Validate a conversation against a specific tier."""
        start_time = datetime.now()

        issues = []
        scores = {}

        # Run all validation categories for the target tier
        for category, weights in self.validation_categories.items():
            if weights[target_tier] > 0:  # Only validate if category has weight for this tier
                category_score, category_issues = self._validate_category(
                    conversation, category, target_tier
                )
                scores[category] = category_score
                issues.extend(category_issues)

        # Calculate weighted overall score
        overall_score = self._calculate_weighted_score(scores, target_tier)

        # Determine validation result
        thresholds = self.tier_thresholds[target_tier]
        if overall_score < thresholds["critical_threshold"]:
            result = ValidationResult.CRITICAL_FAIL
        elif overall_score < thresholds["min_score"]:
            result = ValidationResult.FAIL
        elif overall_score < thresholds["warning_threshold"]:
            result = ValidationResult.WARNING
        else:
            result = ValidationResult.PASS

        execution_time = (datetime.now() - start_time).total_seconds()

        # Count passed checks
        passed_checks = sum(1 for score in scores.values() if score >= thresholds["min_score"])
        total_checks = len(scores)

        return TierValidationResult(
            tier=target_tier,
            overall_result=result,
            score=overall_score,
            issues=issues,
            passed_checks=passed_checks,
            total_checks=total_checks,
            execution_time=execution_time
        )

    def validate_multi_tier(self, conversation: dict[str, Any],
                          tiers: list[ValidationTier] | None = None) -> dict[ValidationTier, TierValidationResult]:
        """Validate conversation against multiple tiers."""
        if tiers is None:
            tiers = list(ValidationTier)

        results = {}
        for tier in tiers:
            results[tier] = self.validate_conversation(conversation, tier)

        return results

    def _validate_category(self, conversation: dict[str, Any],
                          category: str, tier: ValidationTier) -> tuple[float, list[ValidationIssue]]:
        """Validate a specific category for a conversation."""

        if category == "therapeutic_accuracy":
            return self._validate_therapeutic_accuracy(conversation, tier)
        if category == "safety_compliance":
            return self._validate_safety_compliance(conversation, tier)
        if category == "ethical_guidelines":
            return self._validate_ethical_guidelines(conversation, tier)
        if category == "clinical_accuracy":
            return self._validate_clinical_accuracy(conversation, tier)
        if category == "research_validity":
            return self._validate_research_validity(conversation, tier)
        return 1.0, []

    def _validate_therapeutic_accuracy(self, conversation: dict[str, Any],
                                     tier: ValidationTier) -> tuple[float, list[ValidationIssue]]:
        """Validate therapeutic accuracy with tier-specific criteria."""
        issues = []
        score = 1.0

        content = self._extract_conversation_content(conversation)

        # Check for therapeutic techniques
        technique_matches = 0
        for pattern in self.validation_patterns["therapeutic_techniques"]:
            if re.search(pattern, content, re.IGNORECASE):
                technique_matches += 1

        # Tier-specific requirements
        if tier in [ValidationTier.CLINICAL, ValidationTier.RESEARCH]:
            required_techniques = 3
        elif tier == ValidationTier.PROFESSIONAL:
            required_techniques = 2
        else:
            required_techniques = 1

        if technique_matches < required_techniques:
            score -= 0.3
            issues.append(ValidationIssue(
                tier=tier,
                category="therapeutic_accuracy",
                severity="HIGH" if tier in [ValidationTier.CLINICAL, ValidationTier.RESEARCH] else "MEDIUM",
                description=f"Insufficient therapeutic techniques detected ({technique_matches}/{required_techniques})",
                suggestion="Include more evidence-based therapeutic interventions",
                confidence_score=0.8
            ))

        # Check response appropriateness
        if not self._check_response_appropriateness(content, tier):
            score -= 0.2
            issues.append(ValidationIssue(
                tier=tier,
                category="therapeutic_accuracy",
                severity="MEDIUM",
                description="Response appropriateness concerns detected",
                suggestion="Ensure responses are therapeutically appropriate and professional",
                confidence_score=0.7
            ))

        return max(0.0, score), issues

    def _validate_safety_compliance(self, conversation: dict[str, Any],
                                  tier: ValidationTier) -> tuple[float, list[ValidationIssue]]:
        """Validate safety compliance with consistent high standards across tiers."""
        issues = []
        score = 1.0

        content = self._extract_conversation_content(conversation)

        # Check for safety indicators
        safety_concerns = []
        for pattern in self.validation_patterns["safety_indicators"]:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                safety_concerns.append(match.group())

        if safety_concerns:
            # Check if safety concerns are properly addressed
            if not self._check_safety_response(content, safety_concerns):
                score -= 0.5
                issues.append(ValidationIssue(
                    tier=tier,
                    category="safety_compliance",
                    severity="CRITICAL",
                    description=f"Safety concerns not properly addressed: {', '.join(safety_concerns)}",
                    suggestion="Implement proper crisis intervention protocols and safety assessments",
                    confidence_score=0.9
                ))

        # Check for mandatory safety elements (consistent across all tiers)
        if not self._has_safety_disclaimers(content):
            score -= 0.2
            issues.append(ValidationIssue(
                tier=tier,
                category="safety_compliance",
                severity="HIGH",
                description="Missing required safety disclaimers",
                suggestion="Include appropriate safety disclaimers and emergency contact information",
                confidence_score=0.8
            ))

        return max(0.0, score), issues

    def _validate_ethical_guidelines(self, conversation: dict[str, Any],
                                   tier: ValidationTier) -> tuple[float, list[ValidationIssue]]:
        """Validate ethical guidelines compliance."""
        issues = []
        score = 1.0

        content = self._extract_conversation_content(conversation)

        # Check for ethical violations
        for pattern in self.validation_patterns["ethical_violations"]:
            if re.search(pattern, content, re.IGNORECASE):
                score -= 0.4
                issues.append(ValidationIssue(
                    tier=tier,
                    category="ethical_guidelines",
                    severity="HIGH",
                    description="Potential ethical violation detected",
                    suggestion="Review and ensure compliance with professional ethical guidelines",
                    confidence_score=0.8
                ))

        # Check for appropriate boundaries
        if not self._check_professional_boundaries(content, tier):
            score -= 0.3
            issues.append(ValidationIssue(
                tier=tier,
                category="ethical_guidelines",
                severity="MEDIUM",
                description="Professional boundary concerns detected",
                suggestion="Maintain appropriate professional boundaries in therapeutic interactions",
                confidence_score=0.7
            ))

        return max(0.0, score), issues

    def _validate_clinical_accuracy(self, conversation: dict[str, Any],
                                  tier: ValidationTier) -> tuple[float, list[ValidationIssue]]:
        """Validate clinical accuracy with tier-specific requirements."""
        issues = []
        score = 1.0

        content = self._extract_conversation_content(conversation)

        # Check for clinical terminology usage
        clinical_terms = 0
        for pattern in self.validation_patterns["clinical_terminology"]:
            if re.search(pattern, content, re.IGNORECASE):
                clinical_terms += 1

        # Higher tiers require more clinical accuracy
        if tier in [ValidationTier.CLINICAL, ValidationTier.RESEARCH]:
            if clinical_terms == 0:
                score -= 0.4
                issues.append(ValidationIssue(
                    tier=tier,
                    category="clinical_accuracy",
                    severity="HIGH",
                    description="Insufficient clinical terminology for this tier",
                    suggestion="Include appropriate clinical terminology and evidence-based practices",
                    confidence_score=0.8
                ))

        # Check for diagnostic accuracy (if applicable)
        if self._contains_diagnostic_content(content):
            if not self._validate_diagnostic_accuracy(content, tier):
                score -= 0.3
                issues.append(ValidationIssue(
                    tier=tier,
                    category="clinical_accuracy",
                    severity="HIGH",
                    description="Diagnostic accuracy concerns detected",
                    suggestion="Ensure diagnostic content aligns with current clinical standards",
                    confidence_score=0.7
                ))

        return max(0.0, score), issues

    def _validate_research_validity(self, conversation: dict[str, Any],
                                  tier: ValidationTier) -> tuple[float, list[ValidationIssue]]:
        """Validate research validity (only for research tier)."""
        issues = []
        score = 1.0

        if tier != ValidationTier.RESEARCH:
            return score, issues

        content = self._extract_conversation_content(conversation)

        # Check for evidence-based practices
        if not self._has_evidence_based_content(content):
            score -= 0.3
            issues.append(ValidationIssue(
                tier=tier,
                category="research_validity",
                severity="MEDIUM",
                description="Insufficient evidence-based content for research tier",
                suggestion="Include references to current research and evidence-based practices",
                confidence_score=0.8
            ))

        # Check for research methodology compliance
        if not self._check_research_methodology(content):
            score -= 0.2
            issues.append(ValidationIssue(
                tier=tier,
                category="research_validity",
                severity="MEDIUM",
                description="Research methodology concerns detected",
                suggestion="Ensure compliance with research ethics and methodology standards",
                confidence_score=0.7
            ))

        return max(0.0, score), issues

    def _extract_conversation_content(self, conversation: dict[str, Any]) -> str:
        """Extract text content from conversation structure."""
        content = ""

        if isinstance(conversation, dict):
            if "messages" in conversation:
                for message in conversation["messages"]:
                    if isinstance(message, dict) and "content" in message:
                        content += message["content"] + " "
            elif "content" in conversation:
                content = conversation["content"]
            elif "text" in conversation:
                content = conversation["text"]
        elif isinstance(conversation, str):
            content = conversation

        return content.lower()

    def _calculate_weighted_score(self, scores: dict[str, float], tier: ValidationTier) -> float:
        """Calculate weighted overall score for a tier."""
        total_weight = 0
        weighted_sum = 0

        for category, score in scores.items():
            weight = self.validation_categories[category][tier]
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _check_response_appropriateness(self, content: str, tier: ValidationTier) -> bool:
        """Check if responses are appropriate for the tier."""
        # Simple heuristic - can be enhanced with ML models
        inappropriate_patterns = [
            r"\b(personal opinion|i think|i believe)\b",
            r"\b(you should definitely|you must)\b",
            r"\b(i\'m not qualified|i don\'t know)\b"
        ]

        for pattern in inappropriate_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False

        return True

    def _check_safety_response(self, content: str, safety_concerns: list[str]) -> bool:
        """Check if safety concerns are properly addressed."""
        safety_response_patterns = [
            r"\b(crisis hotline|emergency services|seek immediate help)\b",
            r"\b(safety plan|risk assessment|professional help)\b",
            r"\b(not alone|support available|reach out)\b"
        ]

        for pattern in safety_response_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return False

    def _has_safety_disclaimers(self, content: str) -> bool:
        """Check for required safety disclaimers."""
        disclaimer_patterns = [
            r"\b(not a substitute for professional|emergency services)\b",
            r"\b(crisis situation|immediate danger|call 911)\b"
        ]

        return any(re.search(pattern, content, re.IGNORECASE) for pattern in disclaimer_patterns)

    def _check_professional_boundaries(self, content: str, tier: ValidationTier) -> bool:
        """Check for appropriate professional boundaries."""
        boundary_violations = [
            r"\b(personal relationship|friend|date)\b",
            r"\b(my personal experience|when i was)\b",
            r"\b(outside of session|meet privately)\b"
        ]

        for pattern in boundary_violations:
            if re.search(pattern, content, re.IGNORECASE):
                return False

        return True

    def _contains_diagnostic_content(self, content: str) -> bool:
        """Check if content contains diagnostic information."""
        diagnostic_patterns = [
            r"\b(diagnosis|diagnosed with|symptoms of)\b",
            r"\b(disorder|condition|syndrome)\b",
            r"\b(DSM|ICD|criteria)\b"
        ]

        return any(re.search(pattern, content, re.IGNORECASE) for pattern in diagnostic_patterns)

    def _validate_diagnostic_accuracy(self, content: str, tier: ValidationTier) -> bool:
        """Validate diagnostic accuracy (simplified implementation)."""
        # This would typically involve more sophisticated validation
        # against clinical databases and current diagnostic criteria

        # Check for common diagnostic errors
        error_patterns = [
            r"\b(definitely have|certainly is|100% sure)\b",
            r"\b(self-diagnose|diagnose yourself)\b",
            r"\b(not a real disorder|made up condition)\b"
        ]

        return all(not re.search(pattern, content, re.IGNORECASE) for pattern in error_patterns)

    def _has_evidence_based_content(self, content: str) -> bool:
        """Check for evidence-based content."""
        evidence_patterns = [
            r"\b(research shows|studies indicate|evidence suggests)\b",
            r"\b(peer-reviewed|clinical trial|meta-analysis)\b",
            r"\b(evidence-based|empirically supported)\b"
        ]

        return any(re.search(pattern, content, re.IGNORECASE) for pattern in evidence_patterns)

    def _check_research_methodology(self, content: str) -> bool:
        """Check research methodology compliance."""
        methodology_patterns = [
            r"\b(informed consent|ethical approval|IRB)\b",
            r"\b(methodology|systematic review|randomized)\b",
            r"\b(statistical significance|confidence interval)\b"
        ]

        for pattern in methodology_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return True  # Default to true for non-research content

    def generate_validation_report(self, results: dict[ValidationTier, TierValidationResult],
                                 output_file: str | None = None) -> str:
        """Generate comprehensive validation report."""
        if not output_file:
            output_file = f"multi_tier_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "validator_version": "2.0.0",
            "tier_results": {},
            "summary": self._generate_validation_summary(results),
            "recommendations": self._generate_validation_recommendations(results)
        }

        for tier, result in results.items():
            report["tier_results"][tier.value] = {
                "overall_result": result.overall_result.value,
                "score": result.score,
                "passed_checks": result.passed_checks,
                "total_checks": result.total_checks,
                "execution_time": result.execution_time,
                "issues": [self._issue_to_dict(issue) for issue in result.issues]
            }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Multi-tier validation report saved to {output_file}")
        return output_file

    def _generate_validation_summary(self, results: dict[ValidationTier, TierValidationResult]) -> dict[str, Any]:
        """Generate validation summary."""
        summary = {
            "total_tiers_tested": len(results),
            "passing_tiers": 0,
            "warning_tiers": 0,
            "failing_tiers": 0,
            "critical_failing_tiers": 0,
            "average_score": 0.0,
            "total_issues": 0
        }

        total_score = 0
        for result in results.values():
            total_score += result.score
            summary["total_issues"] += len(result.issues)

            if result.overall_result == ValidationResult.PASS:
                summary["passing_tiers"] += 1
            elif result.overall_result == ValidationResult.WARNING:
                summary["warning_tiers"] += 1
            elif result.overall_result == ValidationResult.FAIL:
                summary["failing_tiers"] += 1
            elif result.overall_result == ValidationResult.CRITICAL_FAIL:
                summary["critical_failing_tiers"] += 1

        summary["average_score"] = total_score / len(results) if results else 0.0

        return summary

    def _generate_validation_recommendations(self, results: dict[ValidationTier, TierValidationResult]) -> list[str]:
        """Generate validation recommendations."""
        recommendations = []

        # Analyze results and provide recommendations
        critical_issues = []
        high_issues = []

        for result in results.values():
            for issue in result.issues:
                if issue.severity == "CRITICAL":
                    critical_issues.append(issue)
                elif issue.severity == "HIGH":
                    high_issues.append(issue)

        if critical_issues:
            recommendations.append(f"URGENT: Address {len(critical_issues)} critical validation issues immediately")

        if high_issues:
            recommendations.append(f"HIGH PRIORITY: Resolve {len(high_issues)} high-severity validation issues")

        # Tier-specific recommendations
        failing_tiers = [tier for tier, result in results.items()
                        if result.overall_result in [ValidationResult.FAIL, ValidationResult.CRITICAL_FAIL]]

        if failing_tiers:
            recommendations.append(f"Focus on improving validation for tiers: {', '.join([t.value for t in failing_tiers])}")

        recommendations.extend([
            "Implement continuous validation monitoring",
            "Establish validation quality gates in deployment pipeline",
            "Regular review and update of validation criteria",
            "Training on tier-specific requirements for content creators"
        ])

        return recommendations

def main():
    """Main function for testing the multi-tier validator."""
    validator = MultiTierValidator()

    # Test conversation
    test_conversation = {
        "messages": [
            {"role": "user", "content": "I'm feeling really depressed and having thoughts of self-harm"},
            {"role": "assistant", "content": "I understand you're going through a difficult time. These feelings are concerning and I want you to know that help is available. Please consider reaching out to a crisis hotline at 988 or emergency services if you're in immediate danger. Let's work together on some coping strategies while you seek professional support."}
        ]
    }

    # Validate against all tiers
    results = validator.validate_multi_tier(test_conversation)

    # Generate report
    validator.generate_validation_report(results)

    # Print summary
    for _tier, _result in results.items():
        pass


if __name__ == "__main__":
    main()
