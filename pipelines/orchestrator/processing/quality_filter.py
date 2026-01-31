"""
Quality filtering system for conversation datasets.

This module provides a comprehensive quality filtering system that integrates
all quality assessment components (coherence, emotional authenticity, therapeutic
accuracy, and language quality) with configurable thresholds for dataset curation.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from conversation_coherence_assessment import ConversationCoherenceAssessor
from conversation_schema import Conversation
from emotional_authenticity_assessment import EmotionalAuthenticityAssessor
from language_quality_assessment import LanguageQualityAssessor
from therapeutic_accuracy_assessment import TherapeuticAccuracyAssessor

# Set up logging
logger = logging.getLogger(__name__)


class FilterDecision(Enum):
    """Filter decision outcomes."""

    ACCEPT = "accept"
    REJECT = "reject"
    REVIEW = "review"


@dataclass
class QualityThresholds:
    """Configurable quality thresholds for filtering."""

    # Individual component thresholds
    coherence_threshold: float = 0.6
    emotional_authenticity_threshold: float = 0.6
    therapeutic_accuracy_threshold: float = 0.6
    language_quality_threshold: float = 0.6

    # Overall quality threshold
    overall_threshold: float = 0.65

    # Review thresholds (between reject and accept)
    review_threshold: float = 0.55

    # Critical issue handling
    reject_on_critical_issues: bool = True
    max_warnings: int = 5

    # Component weights for overall score calculation
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "coherence": 0.25,
            "emotional_authenticity": 0.25,
            "therapeutic_accuracy": 0.30,
            "language_quality": 0.20,
        }
    )


@dataclass
class FilterResult:
    """Result of quality filtering for a single conversation."""

    conversation_id: str
    decision: FilterDecision
    overall_score: float
    component_scores: dict[str, float]
    issues: list[str]
    warnings: list[str]
    critical_issues: list[str]
    details: dict[str, Any]
    passed: bool


@dataclass
class FilterReport:
    """Comprehensive filtering report for a dataset."""

    total_conversations: int
    accepted: int
    rejected: int
    review_needed: int
    acceptance_rate: float
    average_quality_score: float
    component_averages: dict[str, float]
    common_issues: list[tuple[str, int]]
    quality_distribution: dict[str, int]
    results: list[FilterResult]


class QualityFilter:
    """
    Comprehensive quality filtering system for conversation datasets.

    Integrates all quality assessment components:
    - Conversation coherence assessment
    - Emotional authenticity assessment
    - Therapeutic accuracy assessment
    - Language quality assessment

    Provides configurable thresholds and detailed filtering reports.
    """

    def __init__(
        self,
        thresholds: QualityThresholds | None = None,
        assessor_configs: dict[str, dict[str, Any]] | None = None,
    ):
        """
        Initialize the quality filter.

        Args:
            thresholds: Quality thresholds configuration
            assessor_configs: Configuration for individual assessors
        """
        self.thresholds = thresholds or QualityThresholds()
        self.assessor_configs = assessor_configs or {}

        # Initialize assessors
        self.coherence_assessor = ConversationCoherenceAssessor(
            self.assessor_configs.get("coherence", {})
        )
        self.emotional_assessor = EmotionalAuthenticityAssessor(
            self.assessor_configs.get("emotional_authenticity", {})
        )
        self.therapeutic_assessor = TherapeuticAccuracyAssessor(
            self.assessor_configs.get("therapeutic_accuracy", {})
        )
        self.language_assessor = LanguageQualityAssessor(
            self.assessor_configs.get("language_quality", {})
        )

        logger.info("Quality filter initialized with thresholds: %s", self.thresholds)

    def filter_conversation(self, conversation: Conversation) -> FilterResult:
        """
        Filter a single conversation through quality assessment.

        Args:
            conversation: The conversation to filter

        Returns:
            FilterResult with detailed assessment and decision
        """
        logger.debug(f"Filtering conversation {conversation.id}")

        # Run all quality assessments
        coherence_metrics = self.coherence_assessor.assess_conversation_coherence(
            conversation
        )
        emotional_metrics = self.emotional_assessor.assess_emotional_authenticity(
            conversation
        )
        therapeutic_metrics = self.therapeutic_assessor.assess_therapeutic_accuracy(
            conversation
        )
        language_metrics = self.language_assessor.assess_language_quality(conversation)

        # Extract scores
        component_scores = {
            "coherence": coherence_metrics.overall_score,
            "emotional_authenticity": emotional_metrics.overall_score,
            "therapeutic_accuracy": therapeutic_metrics.overall_score,
            "language_quality": language_metrics.overall_score,
        }

        # Calculate weighted overall score
        overall_score = sum(
            component_scores[component] * self.thresholds.weights[component]
            for component in component_scores
        )

        # Collect all issues and warnings
        all_issues = []
        all_warnings = []
        all_critical_issues = []

        # Coherence issues
        all_issues.extend([f"Coherence: {issue}" for issue in coherence_metrics.issues])
        # Note: CoherenceMetrics doesn't have warnings attribute

        # Emotional authenticity issues
        all_issues.extend([f"Emotional: {issue}" for issue in emotional_metrics.issues])

        # Therapeutic accuracy issues
        all_issues.extend(
            [f"Therapeutic: {issue}" for issue in therapeutic_metrics.issues]
        )
        all_warnings.extend(
            [f"Therapeutic: {warning}" for warning in therapeutic_metrics.warnings]
        )
        all_critical_issues.extend(
            [f"Therapeutic: {issue}" for issue in therapeutic_metrics.critical_issues]
        )

        # Language quality issues
        all_issues.extend([f"Language: {issue}" for issue in language_metrics.issues])
        all_warnings.extend(
            [f"Language: {warning}" for warning in language_metrics.warnings]
        )

        # Make filtering decision
        decision = self._make_filter_decision(
            overall_score, component_scores, all_critical_issues, all_warnings
        )

        # Compile detailed results
        details = {
            "coherence_details": coherence_metrics.details,
            "emotional_details": emotional_metrics.details,
            "therapeutic_details": therapeutic_metrics.details,
            "language_details": language_metrics.details,
            "risk_level": (
                therapeutic_metrics.risk_level.value
                if hasattr(therapeutic_metrics, "risk_level")
                else "unknown"
            ),
            "complexity_level": (
                language_metrics.complexity_level.value
                if hasattr(language_metrics, "complexity_level")
                else "unknown"
            ),
        }

        return FilterResult(
            conversation_id=conversation.id,
            decision=decision,
            overall_score=overall_score,
            component_scores=component_scores,
            issues=all_issues,
            warnings=all_warnings,
            critical_issues=all_critical_issues,
            details=details,
            passed=(decision == FilterDecision.ACCEPT),
        )

    def filter_conversations(self, conversations: list[Conversation]) -> FilterReport:
        """
        Filter a list of conversations and generate a comprehensive report.

        Args:
            conversations: List of conversations to filter

        Returns:
            FilterReport with detailed statistics and results
        """
        logger.info(f"Filtering {len(conversations)} conversations")

        results = []
        for conversation in conversations:
            try:
                result = self.filter_conversation(conversation)
                results.append(result)
            except Exception as e:
                logger.error(f"Error filtering conversation {conversation.id}: {e}")
                # Create a failed result
                results.append(
                    FilterResult(
                        conversation_id=conversation.id,
                        decision=FilterDecision.REJECT,
                        overall_score=0.0,
                        component_scores={
                            "coherence": 0.0,
                            "emotional_authenticity": 0.0,
                            "therapeutic_accuracy": 0.0,
                            "language_quality": 0.0,
                        },
                        issues=[f"Processing error: {e!s}"],
                        warnings=[],
                        critical_issues=[f"Processing error: {e!s}"],
                        details={},
                        passed=False,
                    )
                )

        # Generate comprehensive report
        return self._generate_report(results)

    def _make_filter_decision(
        self,
        overall_score: float,
        component_scores: dict[str, float],
        critical_issues: list[str],
        warnings: list[str],
    ) -> FilterDecision:
        """Make filtering decision based on scores and issues."""

        # Reject if critical issues and configured to do so
        if critical_issues and self.thresholds.reject_on_critical_issues:
            return FilterDecision.REJECT

        # Reject if too many warnings
        if len(warnings) > self.thresholds.max_warnings:
            return FilterDecision.REJECT

        # Check individual component thresholds
        if component_scores["coherence"] < self.thresholds.coherence_threshold:
            return FilterDecision.REJECT
        if (
            component_scores["emotional_authenticity"]
            < self.thresholds.emotional_authenticity_threshold
        ):
            return FilterDecision.REJECT
        if (
            component_scores["therapeutic_accuracy"]
            < self.thresholds.therapeutic_accuracy_threshold
        ):
            return FilterDecision.REJECT
        if (
            component_scores["language_quality"]
            < self.thresholds.language_quality_threshold
        ):
            return FilterDecision.REJECT

        # Check overall score
        if overall_score >= self.thresholds.overall_threshold:
            return FilterDecision.ACCEPT
        if overall_score >= self.thresholds.review_threshold:
            return FilterDecision.REVIEW
        return FilterDecision.REJECT

    def _generate_report(self, results: list[FilterResult]) -> FilterReport:
        """Generate comprehensive filtering report."""
        total = len(results)
        accepted = sum(1 for r in results if r.decision == FilterDecision.ACCEPT)
        rejected = sum(1 for r in results if r.decision == FilterDecision.REJECT)
        review_needed = sum(1 for r in results if r.decision == FilterDecision.REVIEW)

        acceptance_rate = accepted / total if total > 0 else 0.0

        # Calculate average scores
        avg_quality = (
            sum(r.overall_score for r in results) / total if total > 0 else 0.0
        )

        component_averages = {}
        for component in [
            "coherence",
            "emotional_authenticity",
            "therapeutic_accuracy",
            "language_quality",
        ]:
            component_averages[component] = (
                sum(r.component_scores.get(component, 0.0) for r in results) / total
                if total > 0
                else 0.0
            )

        # Analyze common issues
        issue_counts = {}
        for result in results:
            for issue in result.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        # Quality distribution
        quality_distribution = {
            "excellent": sum(1 for r in results if r.overall_score >= 0.85),
            "good": sum(1 for r in results if 0.70 <= r.overall_score < 0.85),
            "acceptable": sum(1 for r in results if 0.55 <= r.overall_score < 0.70),
            "poor": sum(1 for r in results if 0.40 <= r.overall_score < 0.55),
            "very_poor": sum(1 for r in results if r.overall_score < 0.40),
        }

        return FilterReport(
            total_conversations=total,
            accepted=accepted,
            rejected=rejected,
            review_needed=review_needed,
            acceptance_rate=acceptance_rate,
            average_quality_score=avg_quality,
            component_averages=component_averages,
            common_issues=common_issues,
            quality_distribution=quality_distribution,
            results=results,
        )


# Backward compatibility function
def filter_conversations(
    conversations: list[Conversation], thresholds: dict[str, float] | None = None
) -> list[dict[str, Any]]:
    """
    Backward compatibility function for conversation filtering.

    Args:
        conversations: List of conversations to filter
        thresholds: Dictionary with threshold values

    Returns:
        List of dictionaries with filtering results
    """
    # Convert old threshold format to new format
    if thresholds:
        quality_thresholds = QualityThresholds(
            coherence_threshold=thresholds.get("coherence", 0.6),
            emotional_authenticity_threshold=thresholds.get("emotional", 0.6),
            therapeutic_accuracy_threshold=thresholds.get("therapeutic", 0.6),
            language_quality_threshold=thresholds.get("language", 0.6),
        )
    else:
        quality_thresholds = QualityThresholds()

    filter_system = QualityFilter(quality_thresholds)
    report = filter_system.filter_conversations(conversations)

    # Convert to old format
    return [
        {
            "conversation": conversations[i],
            "passed": result.passed,
            "overall_score": result.overall_score,
            "coherence_score": result.component_scores.get("coherence", 0.0),
            "emotional_score": result.component_scores.get(
                "emotional_authenticity", 0.0
            ),
            "therapeutic_score": result.component_scores.get(
                "therapeutic_accuracy", 0.0
            ),
            "language_score": result.component_scores.get("language_quality", 0.0),
            "issues": result.issues,
            "warnings": result.warnings,
            "critical_issues": result.critical_issues,
        }
        for i, result in enumerate(report.results)
    ]
