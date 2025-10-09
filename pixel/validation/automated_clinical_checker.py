"""
Automated Clinical Appropriateness Checking System

This module provides automated checking of clinical appropriateness for
therapeutic responses, including rule-based validation, pattern detection,
and compliance verification against clinical standards.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set, Pattern
import json
from pathlib import Path

from .clinical_accuracy_validator import (
    ClinicalAccuracyResult,
    ClinicalContext,
    TherapeuticModality,
    SafetyRiskLevel,
    ClinicalAccuracyLevel,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppropriatenessLevel(Enum):
    """Clinical appropriateness levels"""

    HIGHLY_APPROPRIATE = "highly_appropriate"
    APPROPRIATE = "appropriate"
    QUESTIONABLE = "questionable"
    INAPPROPRIATE = "inappropriate"
    DANGEROUS = "dangerous"


class ViolationType(Enum):
    """Types of clinical appropriateness violations"""

    BOUNDARY_VIOLATION = "boundary_violation"
    ETHICAL_VIOLATION = "ethical_violation"
    SAFETY_VIOLATION = "safety_violation"
    THERAPEUTIC_VIOLATION = "therapeutic_violation"
    CULTURAL_INSENSITIVITY = "cultural_insensitivity"
    SCOPE_VIOLATION = "scope_violation"
    CONFIDENTIALITY_BREACH = "confidentiality_breach"
    DUAL_RELATIONSHIP = "dual_relationship"
    INAPPROPRIATE_DISCLOSURE = "inappropriate_disclosure"
    HARMFUL_INTERVENTION = "harmful_intervention"


class CheckCategory(Enum):
    """Categories of clinical checks"""

    BOUNDARY_CHECK = "boundary_check"
    SAFETY_CHECK = "safety_check"
    ETHICAL_CHECK = "ethical_check"
    THERAPEUTIC_CHECK = "therapeutic_check"
    CULTURAL_CHECK = "cultural_check"
    LEGAL_CHECK = "legal_check"
    PROFESSIONAL_CHECK = "professional_check"


@dataclass
class ClinicalRule:
    """Clinical appropriateness rule"""

    rule_id: str
    name: str
    description: str
    category: CheckCategory
    violation_type: ViolationType
    severity: int  # 1-10, 10 being most severe
    patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    context_requirements: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    is_active: bool = True


@dataclass
class AppropriatenessViolation:
    """Clinical appropriateness violation"""

    violation_id: str
    rule_id: str
    violation_type: ViolationType
    severity: int
    description: str
    matched_text: str
    context: str
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    location: Dict[str, int] = field(default_factory=dict)  # start, end positions


@dataclass
class AppropriatenessCheckResult:
    """Result of clinical appropriateness check"""

    check_id: str
    timestamp: datetime
    response_text: str
    clinical_context: ClinicalContext
    overall_level: AppropriatenessLevel
    confidence_score: float
    violations: List[AppropriatenessViolation] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutomatedClinicalChecker:
    """
    Automated clinical appropriateness checking system

    This class provides comprehensive automated checking of clinical
    appropriateness using rule-based validation, pattern matching,
    and contextual analysis.
    """

    def __init__(self, rules_path: Optional[Path] = None):
        """Initialize the automated clinical checker"""
        self.rules: Dict[str, ClinicalRule] = {}
        self.compiled_patterns: Dict[str, Pattern] = {}

        # Load clinical rules
        self._load_clinical_rules(rules_path)

        # Compile regex patterns for efficiency
        self._compile_patterns()

        # Initialize check statistics
        self.check_stats = {
            "total_checks": 0,
            "violations_found": 0,
            "by_category": {},
            "by_severity": {},
        }

        logger.info("Automated clinical checker initialized")

    def _load_clinical_rules(self, rules_path: Optional[Path]) -> None:
        """Load clinical appropriateness rules"""
        # Default rules - in production, these would be loaded from a database
        default_rules = [
            ClinicalRule(
                rule_id="BR001",
                name="Personal Information Sharing",
                description="Therapist should not share personal information",
                category=CheckCategory.BOUNDARY_CHECK,
                violation_type=ViolationType.BOUNDARY_VIOLATION,
                severity=8,
                patterns=[
                    r"my (wife|husband|partner|child|family)",
                    r"when I was (married|divorced|dating)",
                    r"my personal (experience|story|life)",
                    r"let me tell you about my",
                ],
                keywords=[
                    "my personal",
                    "my family",
                    "my relationship",
                    "my experience",
                    "my story",
                    "my life",
                ],
            ),
            ClinicalRule(
                rule_id="BR002",
                name="Dual Relationship Boundaries",
                description="Avoid dual relationships with clients",
                category=CheckCategory.BOUNDARY_CHECK,
                violation_type=ViolationType.DUAL_RELATIONSHIP,
                severity=9,
                patterns=[
                    r"let's be friends",
                    r"we could (hang out|meet up|grab coffee)",
                    r"outside of (therapy|session|office)",
                    r"my (phone number|address|social media)",
                ],
                keywords=[
                    "friends",
                    "hang out",
                    "meet up",
                    "coffee",
                    "outside therapy",
                    "personal relationship",
                ],
            ),
            ClinicalRule(
                rule_id="SF001",
                name="Suicide Risk Minimization",
                description="Do not minimize or dismiss suicidal thoughts",
                category=CheckCategory.SAFETY_CHECK,
                violation_type=ViolationType.SAFETY_VIOLATION,
                severity=10,
                patterns=[
                    r"(everyone|people) (thinks|feels) that way",
                    r"you don't really (want to|mean to) (die|kill yourself)",
                    r"(just|only) (attention|cry for help)",
                    r"things aren't that bad",
                ],
                keywords=[
                    "everyone feels",
                    "attention seeking",
                    "not that bad",
                    "don't really mean",
                    "just dramatic",
                ],
            ),
            ClinicalRule(
                rule_id="SF002",
                name="Crisis Response Requirement",
                description="Must address safety concerns immediately",
                category=CheckCategory.SAFETY_CHECK,
                violation_type=ViolationType.SAFETY_VIOLATION,
                severity=9,
                patterns=[
                    r"we'll talk about (that|suicide|self-harm) next (time|session)",
                    r"let's move on to",
                    r"that's not important right now",
                ],
                keywords=[
                    "next session",
                    "move on",
                    "not important",
                    "talk later",
                    "different topic",
                ],
            ),
            ClinicalRule(
                rule_id="ET001",
                name="Confidentiality Maintenance",
                description="Must maintain client confidentiality",
                category=CheckCategory.ETHICAL_CHECK,
                violation_type=ViolationType.CONFIDENTIALITY_BREACH,
                severity=9,
                patterns=[
                    r"I had another client who",
                    r"someone else told me",
                    r"another patient of mine",
                    r"I can tell (your|my) (family|friends|colleagues)",
                ],
                keywords=[
                    "another client",
                    "other patient",
                    "someone else",
                    "tell family",
                    "share with",
                ],
            ),
            ClinicalRule(
                rule_id="TH001",
                name="Therapeutic Competence",
                description="Stay within scope of competence",
                category=CheckCategory.THERAPEUTIC_CHECK,
                violation_type=ViolationType.SCOPE_VIOLATION,
                severity=7,
                patterns=[
                    r"I don't know (anything|much) about",
                    r"I'm not (trained|qualified) in",
                    r"this is outside my (expertise|area)",
                    r"I'll just (guess|try|wing it)",
                ],
                keywords=[
                    "don't know",
                    "not trained",
                    "outside expertise",
                    "just guess",
                    "wing it",
                    "not qualified",
                ],
            ),
            ClinicalRule(
                rule_id="CU001",
                name="Cultural Sensitivity",
                description="Avoid cultural assumptions and stereotypes",
                category=CheckCategory.CULTURAL_CHECK,
                violation_type=ViolationType.CULTURAL_INSENSITIVITY,
                severity=6,
                patterns=[
                    r"people like you (always|usually|typically)",
                    r"in your culture, (everyone|people|you all)",
                    r"that's just how (your people|they) are",
                    r"you should be more like",
                ],
                keywords=[
                    "people like you",
                    "your culture",
                    "your people",
                    "they all",
                    "typical for",
                    "should be more",
                ],
            ),
            ClinicalRule(
                rule_id="PR001",
                name="Professional Language",
                description="Maintain professional therapeutic language",
                category=CheckCategory.PROFESSIONAL_CHECK,
                violation_type=ViolationType.THERAPEUTIC_VIOLATION,
                severity=5,
                patterns=[
                    r"that's (crazy|insane|nuts|weird)",
                    r"you're (overreacting|being dramatic)",
                    r"just (get over it|move on|forget about it)",
                    r"(suck it up|deal with it|toughen up)",
                ],
                keywords=[
                    "crazy",
                    "insane",
                    "nuts",
                    "weird",
                    "overreacting",
                    "dramatic",
                    "get over it",
                    "suck it up",
                ],
            ),
        ]

        # Load default rules
        for rule in default_rules:
            self.rules[rule.rule_id] = rule

        # Load custom rules if path provided
        if rules_path and rules_path.exists():
            try:
                with open(rules_path, "r") as f:
                    custom_rules_data = json.load(f)
                    for rule_data in custom_rules_data:
                        rule = ClinicalRule(**rule_data)
                        self.rules[rule.rule_id] = rule
                logger.info(f"Loaded {len(custom_rules_data)} custom rules")
            except Exception as e:
                logger.error(f"Failed to load custom rules: {e}")

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching"""
        for rule in self.rules.values():
            for pattern in rule.patterns:
                try:
                    compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                    pattern_key = f"{rule.rule_id}_{pattern}"
                    self.compiled_patterns[pattern_key] = compiled
                except re.error as e:
                    logger.error(f"Failed to compile pattern '{pattern}': {e}")

    async def check_appropriateness(
        self, response_text: str, clinical_context: ClinicalContext
    ) -> AppropriatenessCheckResult:
        """
        Perform comprehensive clinical appropriateness check

        Args:
            response_text: The therapeutic response to check
            clinical_context: Clinical context for the response

        Returns:
            Complete appropriateness check result
        """
        check_id = f"ac_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize result
        result = AppropriatenessCheckResult(
            check_id=check_id,
            timestamp=datetime.now(),
            response_text=response_text,
            clinical_context=clinical_context,
            overall_level=AppropriatenessLevel.APPROPRIATE,
            confidence_score=0.0,
        )

        # Run all active rules
        violations = []
        passed_checks = []

        for rule in self.rules.values():
            if not rule.is_active:
                continue

            rule_violations = await self._check_rule(rule, response_text, clinical_context)

            if rule_violations:
                violations.extend(rule_violations)
            else:
                passed_checks.append(rule.rule_id)

        # Store results
        result.violations = violations
        result.passed_checks = passed_checks

        # Calculate overall appropriateness level
        result.overall_level, result.confidence_score = self._calculate_overall_appropriateness(
            violations, response_text
        )

        # Generate warnings and recommendations
        result.warnings = self._generate_warnings(violations)
        result.recommendations = self._generate_recommendations(violations)

        # Update statistics
        self._update_statistics(result)

        logger.info(f"Appropriateness check completed: {check_id}")
        return result

    async def _check_rule(
        self, rule: ClinicalRule, response_text: str, clinical_context: ClinicalContext
    ) -> List[AppropriatenessViolation]:
        """Check a specific rule against the response"""
        violations = []

        # Check context requirements
        if not self._meets_context_requirements(rule, clinical_context):
            return violations

        # Check patterns
        for pattern in rule.patterns:
            pattern_key = f"{rule.rule_id}_{pattern}"
            compiled_pattern = self.compiled_patterns.get(pattern_key)

            if compiled_pattern:
                matches = compiled_pattern.finditer(response_text)
                for match in matches:
                    # Check for exceptions
                    if self._is_exception(rule, match.group(), response_text):
                        continue

                    violation = AppropriatenessViolation(
                        violation_id=f"v_{rule.rule_id}_{len(violations)}",
                        rule_id=rule.rule_id,
                        violation_type=rule.violation_type,
                        severity=rule.severity,
                        description=f"{rule.name}: {rule.description}",
                        matched_text=match.group(),
                        context=self._extract_context(response_text, match.span()),
                        confidence=0.9,  # High confidence for pattern matches
                        location={"start": match.start(), "end": match.end()},
                    )

                    violations.append(violation)

        # Check keywords (lower confidence)
        for keyword in rule.keywords:
            if keyword.lower() in response_text.lower():
                # Find keyword position
                start_pos = response_text.lower().find(keyword.lower())
                end_pos = start_pos + len(keyword)

                # Check for exceptions
                if self._is_exception(rule, keyword, response_text):
                    continue

                violation = AppropriatenessViolation(
                    violation_id=f"v_{rule.rule_id}_kw_{len(violations)}",
                    rule_id=rule.rule_id,
                    violation_type=rule.violation_type,
                    severity=max(1, rule.severity - 2),  # Lower severity for keywords
                    description=f"{rule.name}: Potential issue detected",
                    matched_text=keyword,
                    context=self._extract_context(response_text, (start_pos, end_pos)),
                    confidence=0.6,  # Lower confidence for keyword matches
                    location={"start": start_pos, "end": end_pos},
                )

                violations.append(violation)

        return violations

    def _meets_context_requirements(
        self, rule: ClinicalRule, clinical_context: ClinicalContext
    ) -> bool:
        """Check if clinical context meets rule requirements"""
        if not rule.context_requirements:
            return True

        # Check context requirements (simplified implementation)
        for requirement in rule.context_requirements:
            if requirement == "crisis" and "crisis" not in clinical_context.crisis_indicators:
                return False
            elif requirement == "initial_session" and clinical_context.session_phase != "initial":
                return False
            # Add more context requirement checks as needed

        return True

    def _is_exception(self, rule: ClinicalRule, matched_text: str, full_text: str) -> bool:
        """Check if the match is an exception to the rule"""
        if not rule.exceptions:
            return False

        # Check if any exception patterns match
        for exception in rule.exceptions:
            if exception.lower() in matched_text.lower():
                return True

            # Check surrounding context for exceptions
            match_pos = full_text.lower().find(matched_text.lower())
            if match_pos != -1:
                context_start = max(0, match_pos - 50)
                context_end = min(len(full_text), match_pos + len(matched_text) + 50)
                context = full_text[context_start:context_end]

                if exception.lower() in context.lower():
                    return True

        return False

    def _extract_context(self, text: str, span: Tuple[int, int]) -> str:
        """Extract context around a match"""
        start, end = span
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)

        context = text[context_start:context_end]

        # Add ellipsis if truncated
        if context_start > 0:
            context = "..." + context
        if context_end < len(text):
            context = context + "..."

        return context

    def _calculate_overall_appropriateness(
        self, violations: List[AppropriatenessViolation], response_text: str
    ) -> Tuple[AppropriatenessLevel, float]:
        """Calculate overall appropriateness level and confidence"""
        if not violations:
            return AppropriatenessLevel.HIGHLY_APPROPRIATE, 0.95

        # Calculate severity score
        max_severity = max(v.severity for v in violations)
        avg_severity = sum(v.severity for v in violations) / len(violations)
        violation_count = len(violations)

        # Determine appropriateness level based on severity and count
        if max_severity >= 9 or violation_count >= 5:
            level = AppropriatenessLevel.DANGEROUS
            confidence = 0.9
        elif max_severity >= 7 or violation_count >= 3:
            level = AppropriatenessLevel.INAPPROPRIATE
            confidence = 0.85
        elif max_severity >= 5 or violation_count >= 2:
            level = AppropriatenessLevel.QUESTIONABLE
            confidence = 0.8
        elif max_severity >= 3 or violation_count >= 1:
            level = AppropriatenessLevel.APPROPRIATE
            confidence = 0.75
        else:
            level = AppropriatenessLevel.HIGHLY_APPROPRIATE
            confidence = 0.9

        # Adjust confidence based on violation confidence
        if violations:
            avg_violation_confidence = sum(v.confidence for v in violations) / len(violations)
            confidence = (confidence + avg_violation_confidence) / 2

        return level, confidence

    def _generate_warnings(self, violations: List[AppropriatenessViolation]) -> List[str]:
        """Generate warnings based on violations"""
        warnings = []

        # Group violations by type
        violation_types = {}
        for violation in violations:
            vtype = violation.violation_type
            if vtype not in violation_types:
                violation_types[vtype] = []
            violation_types[vtype].append(violation)

        # Generate type-specific warnings
        for vtype, type_violations in violation_types.items():
            if vtype == ViolationType.SAFETY_VIOLATION:
                warnings.append("CRITICAL: Safety violations detected - immediate review required")
            elif vtype == ViolationType.BOUNDARY_VIOLATION:
                warnings.append(
                    "WARNING: Boundary violations may compromise therapeutic relationship"
                )
            elif vtype == ViolationType.ETHICAL_VIOLATION:
                warnings.append(
                    "WARNING: Ethical violations detected - review professional standards"
                )
            elif vtype == ViolationType.CONFIDENTIALITY_BREACH:
                warnings.append(
                    "WARNING: Potential confidentiality breach - review disclosure policies"
                )
            elif len(type_violations) > 2:
                warnings.append(
                    f"WARNING: Multiple {vtype.value.replace('_', ' ')} violations detected"
                )

        return warnings

    def _generate_recommendations(self, violations: List[AppropriatenessViolation]) -> List[str]:
        """Generate recommendations based on violations"""
        recommendations = []

        # Group violations by category
        categories = {}
        for violation in violations:
            # Get rule to determine category
            rule = self.rules.get(violation.rule_id)
            if rule:
                category = rule.category
                if category not in categories:
                    categories[category] = []
                categories[category].append(violation)

        # Generate category-specific recommendations
        for category, cat_violations in categories.items():
            if category == CheckCategory.BOUNDARY_CHECK:
                recommendations.append(
                    "Review professional boundaries and dual relationship policies"
                )
                recommendations.append(
                    "Consider consultation with supervisor about boundary issues"
                )
            elif category == CheckCategory.SAFETY_CHECK:
                recommendations.append("Implement immediate safety assessment protocol")
                recommendations.append("Consider crisis intervention procedures")
            elif category == CheckCategory.ETHICAL_CHECK:
                recommendations.append("Review ethical guidelines and professional standards")
                recommendations.append("Consider ethics consultation if needed")
            elif category == CheckCategory.THERAPEUTIC_CHECK:
                recommendations.append("Review therapeutic techniques and interventions")
                recommendations.append("Consider additional training in relevant areas")
            elif category == CheckCategory.CULTURAL_CHECK:
                recommendations.append("Increase cultural sensitivity and awareness")
                recommendations.append("Consider cultural competency training")

        # Add general recommendations
        if len(violations) > 3:
            recommendations.append("Consider comprehensive review of therapeutic approach")
            recommendations.append("Seek supervision or consultation for complex cases")

        return list(set(recommendations))  # Remove duplicates

    def _update_statistics(self, result: AppropriatenessCheckResult) -> None:
        """Update check statistics"""
        self.check_stats["total_checks"] += 1
        self.check_stats["violations_found"] += len(result.violations)

        # Update category statistics
        for violation in result.violations:
            rule = self.rules.get(violation.rule_id)
            if rule:
                category = rule.category.value
                if category not in self.check_stats["by_category"]:
                    self.check_stats["by_category"][category] = 0
                self.check_stats["by_category"][category] += 1

                # Update severity statistics
                severity = str(violation.severity)
                if severity not in self.check_stats["by_severity"]:
                    self.check_stats["by_severity"][severity] = 0
                self.check_stats["by_severity"][severity] += 1

    def add_rule(self, rule: ClinicalRule) -> bool:
        """Add a new clinical rule"""
        try:
            self.rules[rule.rule_id] = rule

            # Compile patterns for the new rule
            for pattern in rule.patterns:
                try:
                    compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                    pattern_key = f"{rule.rule_id}_{pattern}"
                    self.compiled_patterns[pattern_key] = compiled
                except re.error as e:
                    logger.error(f"Failed to compile pattern '{pattern}': {e}")

            logger.info(f"Added clinical rule: {rule.rule_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add rule {rule.rule_id}: {e}")
            return False

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a clinical rule"""
        if rule_id in self.rules:
            # Remove compiled patterns
            patterns_to_remove = [
                key for key in self.compiled_patterns.keys() if key.startswith(f"{rule_id}_")
            ]
            for pattern_key in patterns_to_remove:
                del self.compiled_patterns[pattern_key]

            # Remove rule
            del self.rules[rule_id]
            logger.info(f"Removed clinical rule: {rule_id}")
            return True

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get check statistics"""
        return self.check_stats.copy()

    def export_rules(self, output_path: Path) -> None:
        """Export clinical rules to file"""
        try:
            rules_data = []
            for rule in self.rules.values():
                rule_dict = {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "description": rule.description,
                    "category": rule.category.value,
                    "violation_type": rule.violation_type.value,
                    "severity": rule.severity,
                    "patterns": rule.patterns,
                    "keywords": rule.keywords,
                    "context_requirements": rule.context_requirements,
                    "exceptions": rule.exceptions,
                    "is_active": rule.is_active,
                }
                rules_data.append(rule_dict)

            with open(output_path, "w") as f:
                json.dump(rules_data, f, indent=2)

            logger.info(f"Exported {len(rules_data)} rules to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export rules: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":

    async def main():
        # Initialize checker
        checker = AutomatedClinicalChecker()

        # Example clinical context
        context = ClinicalContext(
            client_presentation="Client with depression seeking help",
            therapeutic_modality=TherapeuticModality.CBT,
            session_phase="initial",
        )

        # Example inappropriate response
        inappropriate_response = """
        I understand you're feeling depressed. You know, my wife went through 
        something similar when we got divorced. Let me tell you about my personal 
        experience with depression. Everyone feels that way sometimes, you're just 
        being dramatic. Why don't we grab coffee outside of therapy to talk more?
        """

        # Check appropriateness
        result = await checker.check_appropriateness(inappropriate_response, context)

        # Print results
        print(f"Check ID: {result.check_id}")
        print(f"Overall Level: {result.overall_level.value}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Violations Found: {len(result.violations)}")

        for violation in result.violations:
            print(f"\n- {violation.violation_type.value} (Severity: {violation.severity})")
            print(f"  Matched: '{violation.matched_text}'")
            print(f"  Description: {violation.description}")

        print(f"\nWarnings: {result.warnings}")
        print(f"Recommendations: {result.recommendations}")

        # Get statistics
        stats = checker.get_statistics()
        print(f"\nStatistics: {stats}")

    # Run example
    asyncio.run(main())
