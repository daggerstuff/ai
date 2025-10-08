#!/usr/bin/env python3
"""
Enhanced DSM-5 Accuracy Validation System for Pixelated Empathy AI
Validates clinical accuracy and compliance with DSM-5 diagnostic standards.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class DSMCategory(Enum):
    """DSM-5 diagnostic categories."""
    NEURODEVELOPMENTAL = "neurodevelopmental_disorders"
    SCHIZOPHRENIA_SPECTRUM = "schizophrenia_spectrum"
    BIPOLAR_RELATED = "bipolar_related_disorders"
    DEPRESSIVE = "depressive_disorders"
    ANXIETY = "anxiety_disorders"
    OBSESSIVE_COMPULSIVE = "obsessive_compulsive_related"
    TRAUMA_STRESSOR = "trauma_stressor_related"
    DISSOCIATIVE = "dissociative_disorders"
    SOMATIC_SYMPTOM = "somatic_symptom_disorders"
    FEEDING_EATING = "feeding_eating_disorders"
    ELIMINATION = "elimination_disorders"
    SLEEP_WAKE = "sleep_wake_disorders"
    SEXUAL_DYSFUNCTIONS = "sexual_dysfunctions"
    GENDER_DYSPHORIA = "gender_dysphoria"
    DISRUPTIVE_IMPULSE = "disruptive_impulse_control"
    SUBSTANCE_RELATED = "substance_related_addictive"
    NEUROCOGNITIVE = "neurocognitive_disorders"
    PERSONALITY = "personality_disorders"
    PARAPHILIC = "paraphilic_disorders"
    OTHER_MENTAL = "other_mental_disorders"

class ValidationSeverity(Enum):
    """Severity levels for DSM-5 validation issues."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DSMValidationIssue:
    """Represents a DSM-5 validation issue."""
    category: DSMCategory
    severity: ValidationSeverity
    issue_type: str
    description: str
    suggestion: str
    confidence_score: float
    line_number: int | None = None
    context: str | None = None
    dsm_reference: str | None = None

@dataclass
class DSMValidationResult:
    """Results of DSM-5 validation."""
    overall_accuracy_score: float
    category_scores: dict[str, float]
    issues: list[DSMValidationIssue]
    validated_disorders: list[str]
    accuracy_by_category: dict[str, float]
    compliance_score: float
    total_checks: int
    passed_checks: int

class DSM5Validator:
    """Enhanced DSM-5 accuracy validation system."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # DSM-5 disorder patterns and criteria
        self.dsm5_disorders = {
            DSMCategory.DEPRESSIVE: {
                "major_depressive_disorder": {
                    "patterns": [
                        r"\b(major depressive disorder|MDD|major depression)\b",
                        r"\b(depressed mood|anhedonia|significant weight)\b",
                        r"\b(insomnia|hypersomnia|psychomotor agitation)\b",
                        r"\b(fatigue|worthlessness|diminished concentration)\b",
                        r"\b(recurrent thoughts of death|suicidal ideation)\b"
                    ],
                    "criteria_count": 5,
                    "duration": "2 weeks",
                    "exclusions": ["manic episode", "hypomanic episode"]
                },
                "persistent_depressive_disorder": {
                    "patterns": [
                        r"\b(persistent depressive disorder|dysthymia|dysthymic)\b",
                        r"\b(depressed mood.*most.*day.*2.*years)\b"
                    ],
                    "criteria_count": 2,
                    "duration": "2 years",
                    "exclusions": []
                }
            },
            DSMCategory.ANXIETY: {
                "generalized_anxiety_disorder": {
                    "patterns": [
                        r"\b(generalized anxiety disorder|GAD)\b",
                        r"\b(excessive anxiety.*worry.*6.*months)\b",
                        r"\b(restlessness|easily fatigued|difficulty concentrating)\b",
                        r"\b(irritability|muscle tension|sleep disturbance)\b"
                    ],
                    "criteria_count": 3,
                    "duration": "6 months",
                    "exclusions": ["substance use", "medical condition"]
                },
                "panic_disorder": {
                    "patterns": [
                        r"\b(panic disorder|panic attacks)\b",
                        r"\b(recurrent.*unexpected.*panic attacks)\b",
                        r"\b(palpitations|sweating|trembling|shortness of breath)\b",
                        r"\b(chest pain|nausea|dizziness|chills)\b",
                        r"\b(fear of dying|fear of losing control)\b"
                    ],
                    "criteria_count": 4,
                    "duration": "1 month",
                    "exclusions": ["substance use", "medical condition"]
                }
            },
            DSMCategory.TRAUMA_STRESSOR: {
                "ptsd": {
                    "patterns": [
                        r"\b(post-traumatic stress disorder|PTSD)\b",
                        r"\b(traumatic event|exposure to death|threatened death)\b",
                        r"\b(intrusive memories|recurrent dreams|flashbacks)\b",
                        r"\b(avoidance.*trauma.*related.*stimuli)\b",
                        r"\b(negative alterations.*cognitions.*mood)\b",
                        r"\b(alterations.*arousal.*reactivity)\b"
                    ],
                    "criteria_count": 4,
                    "duration": "1 month",
                    "exclusions": ["substance use", "medical condition"]
                }
            },
            DSMCategory.SUBSTANCE_RELATED: {
                "alcohol_use_disorder": {
                    "patterns": [
                        r"\b(alcohol use disorder|alcohol dependence|alcoholism)\b",
                        r"\b(tolerance|withdrawal|larger amounts)\b",
                        r"\b(unsuccessful efforts.*cut down|craving)\b",
                        r"\b(failure.*fulfill.*obligations)\b",
                        r"\b(continued use.*social.*problems)\b"
                    ],
                    "criteria_count": 2,
                    "duration": "12 months",
                    "exclusions": []
                }
            }
        }

        # Common diagnostic errors and misconceptions
        self.diagnostic_errors = {
            "overgeneralization": [
                r"\b(everyone has|all people with|always)\b",
                r"\b(never|impossible|definitely)\b"
            ],
            "outdated_terminology": [
                r"\b(multiple personality disorder|manic depressive)\b",
                r"\b(asperger\'s syndrome|PDD-NOS)\b",
                r"\b(mental retardation|organic brain syndrome)\b"
            ],
            "inappropriate_certainty": [
                r"\b(definitely has|certainly is|100% sure)\b",
                r"\b(without a doubt|absolutely)\b"
            ],
            "missing_differential": [
                r"\b(only possible|single explanation)\b"
            ],
            "cultural_insensitivity": [
                r"\b(cultural bound|primitive|backward)\b",
                r"\b(normal for their culture|expected behavior)\b"
            ]
        }

        # Severity specifiers
        self.severity_specifiers = {
            "mild": r"\b(mild|slight|minimal)\b",
            "moderate": r"\b(moderate|medium|average)\b",
            "severe": r"\b(severe|extreme|intense)\b"
        }

        # Course specifiers
        self.course_specifiers = {
            "episodic": r"\b(episodic|episodes|recurrent)\b",
            "continuous": r"\b(continuous|persistent|chronic)\b",
            "partial_remission": r"\b(partial remission|improving)\b",
            "full_remission": r"\b(full remission|recovered)\b"
        }

    def validate_dsm5_accuracy(self, content: str) -> DSMValidationResult:
        """Validate DSM-5 accuracy of clinical content."""
        start_time = datetime.now()

        issues = []
        category_scores = {}
        validated_disorders = []
        accuracy_by_category = {}

        content_lower = content.lower()

        # Check each DSM-5 category
        for category, disorders in self.dsm5_disorders.items():
            category_issues, category_score, found_disorders = self._validate_category(
                content_lower, category, disorders
            )
            issues.extend(category_issues)
            category_scores[category.value] = category_score
            validated_disorders.extend(found_disorders)
            accuracy_by_category[category.value] = category_score

        # Check for common diagnostic errors
        error_issues = self._check_diagnostic_errors(content_lower)
        issues.extend(error_issues)

        # Check for proper use of specifiers
        specifier_issues = self._check_specifiers(content_lower)
        issues.extend(specifier_issues)

        # Calculate overall scores
        overall_accuracy = self._calculate_overall_accuracy(category_scores, issues)
        compliance_score = self._calculate_compliance_score(issues)

        # Count checks
        total_checks = len(self.dsm5_disorders) * 3 + len(self.diagnostic_errors) + 2
        passed_checks = total_checks - len([i for i in issues if i.severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]])

        execution_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"DSM-5 validation completed in {execution_time:.3f}s")

        return DSMValidationResult(
            overall_accuracy_score=overall_accuracy,
            category_scores=category_scores,
            issues=issues,
            validated_disorders=validated_disorders,
            accuracy_by_category=accuracy_by_category,
            compliance_score=compliance_score,
            total_checks=total_checks,
            passed_checks=passed_checks
        )

    def _validate_category(self, content: str, category: DSMCategory,
                          disorders: dict[str, Any]) -> tuple[list[DSMValidationIssue], float, list[str]]:
        """Validate a specific DSM-5 category."""
        issues = []
        found_disorders = []
        category_score = 1.0

        for disorder_name, disorder_info in disorders.items():
            # Check if disorder is mentioned
            disorder_mentioned = False
            for pattern in disorder_info["patterns"]:
                if re.search(pattern, content, re.IGNORECASE):
                    disorder_mentioned = True
                    found_disorders.append(disorder_name)
                    break

            if disorder_mentioned:
                # Validate diagnostic criteria
                criteria_issues = self._validate_diagnostic_criteria(
                    content, disorder_name, disorder_info, category
                )
                issues.extend(criteria_issues)

                # Check for proper duration specification
                if "duration" in disorder_info:
                    if not self._check_duration_specification(content, disorder_info["duration"]):
                        issues.append(DSMValidationIssue(
                            category=category,
                            severity=ValidationSeverity.MEDIUM,
                            issue_type="MISSING_DURATION",
                            description=f"Missing duration specification for {disorder_name}",
                            suggestion=f"Include duration requirement: {disorder_info['duration']}",
                            confidence_score=0.8,
                            dsm_reference=f"DSM-5 criteria for {disorder_name}"
                        ))
                        category_score -= 0.2

                # Check for exclusion criteria
                if "exclusions" in disorder_info:
                    exclusion_issues = self._check_exclusion_criteria(
                        content, disorder_info["exclusions"], disorder_name, category
                    )
                    issues.extend(exclusion_issues)
                    if exclusion_issues:
                        category_score -= 0.1 * len(exclusion_issues)

        return issues, max(0.0, category_score), found_disorders

    def _validate_diagnostic_criteria(self, content: str, disorder_name: str,
                                    disorder_info: dict[str, Any], category: DSMCategory) -> list[DSMValidationIssue]:
        """Validate diagnostic criteria for a specific disorder."""
        issues = []

        # Count how many criteria patterns are mentioned
        criteria_found = 0
        for pattern in disorder_info["patterns"][1:]:  # Skip the disorder name pattern
            if re.search(pattern, content, re.IGNORECASE):
                criteria_found += 1

        required_criteria = disorder_info.get("criteria_count", 3)

        if criteria_found < required_criteria:
            issues.append(DSMValidationIssue(
                category=category,
                severity=ValidationSeverity.HIGH,
                issue_type="INSUFFICIENT_CRITERIA",
                description=f"Insufficient diagnostic criteria for {disorder_name} ({criteria_found}/{required_criteria})",
                suggestion="Include more specific diagnostic criteria as per DSM-5",
                confidence_score=0.9,
                dsm_reference=f"DSM-5 diagnostic criteria for {disorder_name}"
            ))

        return issues

    def _check_duration_specification(self, content: str, required_duration: str) -> bool:
        """Check if duration is properly specified."""
        duration_patterns = [
            r"\b(\d+)\s*(weeks?|months?|years?)\b",
            r"\b(at least|minimum|more than)\s*(\d+)\s*(weeks?|months?|years?)\b",
            r"\b(persistent|chronic|ongoing)\b"
        ]

        return any(re.search(pattern, content, re.IGNORECASE) for pattern in duration_patterns)

    def _check_exclusion_criteria(self, content: str, exclusions: list[str],
                                disorder_name: str, category: DSMCategory) -> list[DSMValidationIssue]:
        """Check for proper consideration of exclusion criteria."""
        issues = []

        for exclusion in exclusions:
            if re.search(exclusion, content, re.IGNORECASE):
                # Check if exclusion is properly addressed
                exclusion_addressed = self._check_exclusion_addressed(content, exclusion)
                if not exclusion_addressed:
                    issues.append(DSMValidationIssue(
                        category=category,
                        severity=ValidationSeverity.HIGH,
                        issue_type="UNADDRESSED_EXCLUSION",
                        description=f"Exclusion criterion '{exclusion}' mentioned but not properly addressed for {disorder_name}",
                        suggestion="Address exclusion criteria in differential diagnosis",
                        confidence_score=0.8,
                        dsm_reference=f"DSM-5 exclusion criteria for {disorder_name}"
                    ))

        return issues

    def _check_exclusion_addressed(self, content: str, exclusion: str) -> bool:
        """Check if an exclusion criterion is properly addressed."""
        addressing_patterns = [
            r"\b(ruled out|excluded|not due to|not caused by)\b",
            r"\b(differential diagnosis|consider|distinguish)\b",
            r"\b(not better explained|not attributable)\b"
        ]

        # Look for addressing patterns near the exclusion mention
        exclusion_pos = content.find(exclusion.lower())
        if exclusion_pos != -1:
            # Check 200 characters around the exclusion mention
            context = content[max(0, exclusion_pos-100):exclusion_pos+100]
            for pattern in addressing_patterns:
                if re.search(pattern, context, re.IGNORECASE):
                    return True

        return False

    def _check_diagnostic_errors(self, content: str) -> list[DSMValidationIssue]:
        """Check for common diagnostic errors."""
        issues = []

        for error_type, patterns in self.diagnostic_errors.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    severity = ValidationSeverity.HIGH if error_type in ["inappropriate_certainty", "outdated_terminology"] else ValidationSeverity.MEDIUM

                    issues.append(DSMValidationIssue(
                        category=DSMCategory.OTHER_MENTAL,  # General category for errors
                        severity=severity,
                        issue_type=error_type.upper(),
                        description=f"Diagnostic error detected: {error_type.replace('_', ' ')}",
                        suggestion=self._get_error_suggestion(error_type),
                        confidence_score=0.7,
                        line_number=content[:match.start()].count("\n") + 1,
                        context=match.group()
                    ))

        return issues

    def _get_error_suggestion(self, error_type: str) -> str:
        """Get suggestion for specific error type."""
        suggestions = {
            "overgeneralization": "Avoid absolute statements; use qualified language",
            "outdated_terminology": "Use current DSM-5 terminology and classifications",
            "inappropriate_certainty": "Use appropriate clinical language with uncertainty when warranted",
            "missing_differential": "Consider differential diagnosis and alternative explanations",
            "cultural_insensitivity": "Consider cultural factors and avoid cultural bias"
        }
        return suggestions.get(error_type, "Review diagnostic accuracy and clinical standards")

    def _check_specifiers(self, content: str) -> list[DSMValidationIssue]:
        """Check for proper use of severity and course specifiers."""
        issues = []

        # Check if disorders are mentioned without appropriate specifiers
        disorder_mentions = re.finditer(r"\b\w+\s+(disorder|syndrome)\b", content, re.IGNORECASE)

        for match in disorder_mentions:
            disorder_context = content[max(0, match.start()-50):match.end()+50]

            # Check for severity specifiers
            has_severity = any(re.search(pattern, disorder_context, re.IGNORECASE)
                             for pattern in self.severity_specifiers.values())

            # Check for course specifiers
            has_course = any(re.search(pattern, disorder_context, re.IGNORECASE)
                           for pattern in self.course_specifiers.values())

            if not has_severity and not has_course:
                issues.append(DSMValidationIssue(
                    category=DSMCategory.OTHER_MENTAL,
                    severity=ValidationSeverity.LOW,
                    issue_type="MISSING_SPECIFIERS",
                    description="Disorder mentioned without severity or course specifiers",
                    suggestion="Include appropriate severity and/or course specifiers as per DSM-5",
                    confidence_score=0.6,
                    context=match.group()
                ))

        return issues

    def _calculate_overall_accuracy(self, category_scores: dict[str, float],
                                  issues: list[DSMValidationIssue]) -> float:
        """Calculate overall DSM-5 accuracy score."""
        if not category_scores:
            return 1.0

        # Base score from category averages
        base_score = sum(category_scores.values()) / len(category_scores)

        # Penalty for issues
        penalty = 0
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                penalty += 0.3
            elif issue.severity == ValidationSeverity.HIGH:
                penalty += 0.2
            elif issue.severity == ValidationSeverity.MEDIUM:
                penalty += 0.1
            elif issue.severity == ValidationSeverity.LOW:
                penalty += 0.05

        return max(0.0, base_score - penalty)

    def _calculate_compliance_score(self, issues: list[DSMValidationIssue]) -> float:
        """Calculate DSM-5 compliance score."""
        if not issues:
            return 1.0

        # Weight issues by severity
        total_weight = 0
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                total_weight += 4
            elif issue.severity == ValidationSeverity.HIGH:
                total_weight += 3
            elif issue.severity == ValidationSeverity.MEDIUM:
                total_weight += 2
            elif issue.severity == ValidationSeverity.LOW:
                total_weight += 1

        # Convert to 0-1 score (assuming max 20 weighted issues for full penalty)
        return max(0.0, 1.0 - (total_weight / 20))

    def generate_dsm5_report(self, result: DSMValidationResult, output_file: str | None = None) -> str:
        """Generate comprehensive DSM-5 validation report."""
        if not output_file:
            output_file = f"dsm5_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert enums to strings for JSON serialization
        issues_dict = []
        for issue in result.issues:
            issue_dict = asdict(issue)
            issue_dict["category"] = issue.category.value
            issue_dict["severity"] = issue.severity.value
            issues_dict.append(issue_dict)

        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "validator_version": "1.0.0",
            "overall_accuracy_score": result.overall_accuracy_score,
            "compliance_score": result.compliance_score,
            "category_scores": result.category_scores,
            "accuracy_by_category": result.accuracy_by_category,
            "validated_disorders": result.validated_disorders,
            "total_checks": result.total_checks,
            "passed_checks": result.passed_checks,
            "issues": issues_dict,
            "summary": self._generate_dsm5_summary(result),
            "recommendations": self._generate_dsm5_recommendations(result)
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"DSM-5 validation report saved to {output_file}")
        return output_file

    def _generate_dsm5_summary(self, result: DSMValidationResult) -> dict[str, Any]:
        """Generate DSM-5 validation summary."""
        severity_counts = {}
        category_counts = {}

        for issue in result.issues:
            severity_counts[issue.severity.value] = severity_counts.get(issue.severity.value, 0) + 1
            category_counts[issue.category.value] = category_counts.get(issue.category.value, 0) + 1

        return {
            "total_issues": len(result.issues),
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "accuracy_grade": self._get_accuracy_grade(result.overall_accuracy_score),
            "compliance_grade": self._get_accuracy_grade(result.compliance_score),
            "disorders_validated": len(result.validated_disorders),
            "check_pass_rate": f"{result.passed_checks}/{result.total_checks}"
        }

    def _get_accuracy_grade(self, score: float) -> str:
        """Convert accuracy score to letter grade."""
        if score >= 0.95:
            return "A+"
        if score >= 0.90:
            return "A"
        if score >= 0.85:
            return "B+"
        if score >= 0.80:
            return "B"
        if score >= 0.75:
            return "C+"
        if score >= 0.70:
            return "C"
        if score >= 0.65:
            return "D+"
        if score >= 0.60:
            return "D"
        return "F"

    def _generate_dsm5_recommendations(self, result: DSMValidationResult) -> list[str]:
        """Generate DSM-5 validation recommendations."""
        recommendations = []

        if result.overall_accuracy_score < 0.7:
            recommendations.append("CRITICAL: DSM-5 accuracy is below acceptable standards. Immediate review required.")
        elif result.overall_accuracy_score < 0.8:
            recommendations.append("WARNING: DSM-5 accuracy needs improvement. Review diagnostic criteria.")

        if result.compliance_score < 0.8:
            recommendations.append("Improve DSM-5 compliance by addressing diagnostic errors and terminology.")

        # Category-specific recommendations
        low_scoring_categories = [cat for cat, score in result.category_scores.items() if score < 0.8]
        if low_scoring_categories:
            recommendations.append(f"Focus on improving accuracy in: {', '.join(low_scoring_categories)}")

        # Issue-specific recommendations
        critical_issues = [issue for issue in result.issues if issue.severity == ValidationSeverity.CRITICAL]
        high_issues = [issue for issue in result.issues if issue.severity == ValidationSeverity.HIGH]

        if critical_issues:
            recommendations.append(f"Address {len(critical_issues)} critical DSM-5 issues immediately")

        if high_issues:
            recommendations.append(f"Resolve {len(high_issues)} high-priority diagnostic accuracy issues")

        recommendations.extend([
            "Regular training on current DSM-5 criteria and updates",
            "Implement peer review process for diagnostic content",
            "Use structured diagnostic interviews and assessments",
            "Stay updated with DSM-5-TR revisions and clarifications",
            "Consider cultural formulation in diagnostic assessments"
        ])

        return recommendations

def main():
    """Main function for testing the DSM-5 validator."""
    validator = DSM5Validator()

    # Test content with various DSM-5 elements
    test_content = """
    The patient presents with major depressive disorder, experiencing depressed mood,
    anhedonia, significant weight loss, insomnia, and fatigue for the past 3 weeks.
    The symptoms have persisted for more than 2 weeks and cause significant impairment
    in social and occupational functioning. We need to rule out substance use and
    medical conditions. The severity appears to be moderate based on the number of
    symptoms and functional impairment.
    """

    # Validate DSM-5 accuracy
    result = validator.validate_dsm5_accuracy(test_content)

    # Generate report
    validator.generate_dsm5_report(result)

    # Print summary

    if result.issues:
        for _issue in result.issues[:5]:  # Show first 5 issues
            pass


if __name__ == "__main__":
    main()
