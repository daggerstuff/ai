"""
Dataset Evaluation Engine

Implements systematic quality assessment and prioritization of identified datasets
across four dimensions: therapeutic relevance, data structure quality, training
integration potential, and ethical accessibility.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from ai.journal_dataset_research.models.dataset_models import (
    DatasetEvaluation,
    DatasetSource,
)

# Optional compliance module imports
try:
    from ai.journal_dataset_research.compliance.compliance_checker import (
        ComplianceChecker,
    )
    COMPLIANCE_AVAILABLE = True
except ImportError:
    COMPLIANCE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation engine."""

    # Scoring weights for each dimension
    therapeutic_relevance_weight: float = 0.35
    data_structure_quality_weight: float = 0.25
    training_integration_weight: float = 0.20
    ethical_accessibility_weight: float = 0.20

    # Priority tier thresholds
    high_priority_threshold: float = 7.5
    medium_priority_threshold: float = 5.0

    # Therapeutic relevance keywords
    therapeutic_keywords: List[str] = field(
        default_factory=lambda: [
            "therapy",
            "therapeutic",
            "counseling",
            "counselor",
            "psychotherapy",
            "mental health",
            "psychological",
            "clinical",
            "intervention",
            "treatment",
            "patient",
            "client",
            "session",
            "dialogue",
            "conversation",
            "transcript",
            "crisis",
            "suicide",
            "trauma",
            "depression",
            "anxiety",
            "ptsd",
        ]
    )

    # Evidence-based practice keywords
    evidence_based_keywords: List[str] = field(
        default_factory=lambda: [
            "cbt",
            "cognitive behavioral",
            "dbt",
            "dialectical behavioral",
            "act",
            "acceptance and commitment",
            "emdr",
            "evidence-based",
            "randomized controlled trial",
            "rct",
            "meta-analysis",
            "systematic review",
        ]
    )

    # Content type indicators
    transcript_keywords: List[str] = field(
        default_factory=lambda: [
            "transcript",
            "dialogue",
            "conversation",
            "session",
            "interview",
        ]
    )

    outcome_keywords: List[str] = field(
        default_factory=lambda: [
            "outcome",
            "effectiveness",
            "efficacy",
            "improvement",
            "progress",
            "before and after",
            "pre-post",
        ]
    )

    protocol_keywords: List[str] = field(
        default_factory=lambda: [
            "protocol",
            "procedure",
            "guideline",
            "manual",
            "intervention",
            "treatment protocol",
        ]
    )

    def validate(self) -> List[str]:
        """Validate the configuration and return list of errors."""
        errors = []
        total_weight = (
            self.therapeutic_relevance_weight
            + self.data_structure_quality_weight
            + self.training_integration_weight
            + self.ethical_accessibility_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Evaluation weights must sum to 1.0, got {total_weight}")
        if (
            self.high_priority_threshold
            <= self.medium_priority_threshold
        ):
            errors.append(
                "high_priority_threshold must be greater than medium_priority_threshold"
            )
        return errors


class DatasetEvaluationEngine:
    """
    Main evaluation engine for assessing dataset quality and priority.

    Evaluates datasets across four dimensions:
    1. Therapeutic Relevance (1-10)
    2. Data Structure Quality (1-10)
    3. Training Integration Potential (1-10)
    4. Ethical Accessibility (1-10)

    Calculates overall score using weighted average and assigns priority tier.
    """

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        compliance_checker: Optional[ComplianceChecker] = None,
    ):
        """
        Initialize the evaluation engine.

        Args:
            config: Evaluation configuration. If None, uses default config.
            compliance_checker: Optional compliance checker instance for enhanced compliance checking.
        """
        self.config = config or EvaluationConfig()
        config_errors = self.config.validate()
        if config_errors:
            raise ValueError(f"Invalid evaluation config: {', '.join(config_errors)}")

        # Compliance checker (optional)
        self.compliance_checker = compliance_checker
        if compliance_checker and not COMPLIANCE_AVAILABLE:
            logger.warning("Compliance checker provided but compliance module not available")
            self.compliance_checker = None

    def evaluate_dataset(
        self, source: DatasetSource, evaluator: str = "system"
    ) -> DatasetEvaluation:
        """
        Evaluate a dataset source across all dimensions.

        Args:
            source: The dataset source to evaluate
            evaluator: Name of the evaluator (default: "system")

        Returns:
            DatasetEvaluation with all scores and notes
        """
        logger.info(f"Evaluating dataset: {source.source_id} - {source.title}")

        # Evaluate each dimension
        therapeutic_relevance, therapeutic_notes = (
            self._assess_therapeutic_relevance(source)
        )
        data_structure_quality, structure_notes = (
            self._assess_data_structure_quality(source)
        )
        training_integration, integration_notes = (
            self._assess_training_integration(source)
        )
        ethical_accessibility, ethical_notes = (
            self._assess_ethical_accessibility(source)
        )

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            therapeutic_relevance,
            data_structure_quality,
            training_integration,
            ethical_accessibility,
        )

        # Determine priority tier
        priority_tier = self._calculate_priority_tier(overall_score)

        # Identify competitive advantages
        competitive_advantages = self._identify_competitive_advantages(
            source, therapeutic_relevance, data_structure_quality
        )

        # Perform compliance check if compliance checker is available
        compliance_checked = False
        compliance_status = "unknown"
        compliance_score = 0.0
        license_compatible = False
        privacy_compliant = False
        hipaa_compliant = False

        if self.compliance_checker:
            try:
                compliance_result = self.compliance_checker.check_compliance(
                    source=source,
                    dataset_sample=None,  # Could be provided if available
                    license_text=None,  # Could be extracted from source
                    metadata={"abstract": source.abstract, "keywords": source.keywords},
                )

                compliance_checked = True
                compliance_status = compliance_result.compliance_status
                compliance_score = compliance_result.overall_compliance_score

                # Update ethical accessibility score based on compliance results
                if compliance_result.license_check:
                    license_compatible = compliance_result.license_check.is_usable()
                    # Adjust ethical accessibility score based on license compatibility
                    if not license_compatible:
                        ethical_accessibility = max(1, ethical_accessibility - 3)
                        ethical_notes += "; License incompatible or requires review"

                if compliance_result.privacy_assessment:
                    privacy_compliant = compliance_result.privacy_assessment.is_compliant()
                    if not privacy_compliant:
                        ethical_accessibility = max(1, ethical_accessibility - 2)
                        ethical_notes += "; Privacy compliance issues detected"

                if compliance_result.hipaa_compliance:
                    hipaa_compliant = compliance_result.hipaa_compliance.is_compliant()
                    if compliance_result.hipaa_compliance.contains_phi and not hipaa_compliant:
                        ethical_accessibility = max(1, ethical_accessibility - 2)
                        ethical_notes += "; HIPAA compliance issues detected"

                # Recalculate overall score with updated ethical accessibility
                overall_score = self._calculate_overall_score(
                    therapeutic_relevance,
                    data_structure_quality,
                    training_integration,
                    ethical_accessibility,
                )
                priority_tier = self._calculate_priority_tier(overall_score)

            except Exception as e:
                logger.warning(f"Error performing compliance check: {e}")

        evaluation = DatasetEvaluation(
            source_id=source.source_id,
            therapeutic_relevance=therapeutic_relevance,
            therapeutic_relevance_notes=therapeutic_notes,
            data_structure_quality=data_structure_quality,
            data_structure_notes=structure_notes,
            training_integration=training_integration,
            integration_notes=integration_notes,
            ethical_accessibility=ethical_accessibility,
            ethical_notes=ethical_notes,
            overall_score=overall_score,
            priority_tier=priority_tier,
            evaluation_date=datetime.now(),
            evaluator=evaluator,
            competitive_advantages=competitive_advantages,
            compliance_checked=compliance_checked,
            compliance_status=compliance_status,
            compliance_score=compliance_score,
            license_compatible=license_compatible,
            privacy_compliant=privacy_compliant,
            hipaa_compliant=hipaa_compliant,
        )

        # Validate evaluation
        eval_errors = evaluation.validate()
        if eval_errors:
            raise ValueError(f"Invalid evaluation: {', '.join(eval_errors)}")

        logger.info(
            f"Evaluation complete for {source.source_id}: "
            f"overall_score={overall_score:.2f}, priority={priority_tier}"
        )

        return evaluation

    def _assess_therapeutic_relevance(
        self, source: DatasetSource
    ) -> tuple[int, str]:
        """
        Assess therapeutic relevance of the dataset (1-10).

        Evaluates:
        - Direct applicability to counseling/therapy contexts
        - Quality of therapeutic dialogue or interventions
        - Alignment with evidence-based therapeutic practices

        Args:
            source: The dataset source to evaluate

        Returns:
            Tuple of (score: int, notes: str)
        """
        score = 0
        notes_parts = []

        # Check title and abstract for therapeutic keywords
        text_to_check = f"{source.title} {source.abstract}".lower()
        keyword_matches = sum(
            1 for keyword in self.config.therapeutic_keywords if keyword in text_to_check
        )
        # More generous scoring: base score on keyword density
        if keyword_matches > 0:
            keyword_density = min(1.0, keyword_matches / 10.0)  # Normalize to 0-1
            keyword_score = 3 + (keyword_density * 4)  # Score between 3-7
            score += keyword_score
            notes_parts.append(
                f"Found {keyword_matches} therapeutic keywords in title/abstract"
            )

        # Check for evidence-based practice indicators (high value)
        evidence_matches = sum(
            1
            for keyword in self.config.evidence_based_keywords
            if keyword in text_to_check
        )
        if evidence_matches > 0:
            evidence_score = min(3, evidence_matches * 0.8)  # Up to 3 points
            score += evidence_score
            notes_parts.append(
                f"Found {evidence_matches} evidence-based practice indicators"
            )

        # Check content type (transcripts are highly relevant - base score)
        content_type_score = 0
        if any(keyword in text_to_check for keyword in self.config.transcript_keywords):
            content_type_score = 4  # High base score for transcripts
            notes_parts.append("Contains therapy transcripts/conversations")
        elif any(keyword in text_to_check for keyword in self.config.outcome_keywords):
            content_type_score = 3
            notes_parts.append("Contains therapy outcome data")
        elif any(keyword in text_to_check for keyword in self.config.protocol_keywords):
            content_type_score = 2
            notes_parts.append("Contains therapy protocols/guidelines")
        else:
            content_type_score = 1
            notes_parts.append("Therapeutic content type unclear")

        score += content_type_score

        # Normalize to 1-10 scale
        final_score = max(1, min(10, int(round(score))))

        # Boost score for open access datasets with available data
        if source.open_access and source.data_availability == "available":
            final_score = min(10, final_score + 1)
            notes_parts.append("Open access with available data (+1)")

        notes = "; ".join(notes_parts) if notes_parts else "No specific therapeutic indicators found"

        return final_score, notes

    def _assess_data_structure_quality(
        self, source: DatasetSource
    ) -> tuple[int, str]:
        """
        Assess data structure quality of the dataset (1-10).

        Evaluates:
        - Organization and standardization of data
        - Completeness of therapeutic conversations
        - Metadata and contextual information availability

        Args:
            source: The dataset source to evaluate

        Returns:
            Tuple of (score: int, notes: str)
        """
        score = 5  # Start with neutral score
        notes_parts = []

        # Check for DOI (indicates formal publication with metadata)
        if source.doi:
            score += 2
            notes_parts.append("Has DOI (formal publication)")

        # Check for comprehensive metadata
        metadata_score = 0
        if source.abstract:
            metadata_score += 1
        if source.keywords:
            metadata_score += 1
        if source.authors:
            metadata_score += 1
        if source.publication_date:
            metadata_score += 1

        if metadata_score >= 3:
            score += 2
            notes_parts.append("Comprehensive metadata available")
        elif metadata_score >= 2:
            score += 1
            notes_parts.append("Partial metadata available")
        else:
            notes_parts.append("Limited metadata")

        # Check data availability
        if source.data_availability == "available":
            score += 2
            notes_parts.append("Data directly available")
        elif source.data_availability == "upon_request":
            score += 1
            notes_parts.append("Data available upon request")
        elif source.data_availability == "restricted":
            score -= 1
            notes_parts.append("Data access restricted")
        else:
            score -= 1
            notes_parts.append("Data availability unknown")

        # Check source type (repositories often have better structure)
        if source.source_type == "repository":
            score += 1
            notes_parts.append("Repository source (typically well-structured)")
        elif source.source_type == "clinical_trial":
            score += 1
            notes_parts.append("Clinical trial (structured data expected)")

        # Normalize to 1-10 scale
        final_score = max(1, min(10, int(round(score))))

        notes = "; ".join(notes_parts) if notes_parts else "Standard data structure"

        return final_score, notes

    def _assess_training_integration(
        self, source: DatasetSource
    ) -> tuple[int, str]:
        """
        Assess training integration potential of the dataset (1-10).

        Evaluates:
        - Compatibility with existing training pipeline
        - Format alignment with current datasets
        - Integration complexity and preprocessing requirements

        Args:
            source: The dataset source to evaluate

        Returns:
            Tuple of (score: int, notes: str)
        """
        score = 5  # Start with neutral score
        notes_parts = []

        # Check data availability (need data to integrate)
        if source.data_availability == "available":
            score += 3
            notes_parts.append("Data available for integration")
        elif source.data_availability == "upon_request":
            score += 1
            notes_parts.append("Data available upon request (may delay integration)")
        else:
            score -= 2
            notes_parts.append("Data access uncertain (integration risk)")

        # Check open access status
        if source.open_access:
            score += 2
            notes_parts.append("Open access (no licensing barriers)")
        else:
            score -= 1
            notes_parts.append("May have licensing restrictions")

        # Repository sources often have standardized formats
        if source.source_type == "repository":
            score += 1
            notes_parts.append("Repository format (typically standardized)")
        elif source.source_type == "clinical_trial":
            score += 1
            notes_parts.append("Clinical trial data (may require preprocessing)")

        # Check if source has URL (needed for download/integration)
        if source.url:
            score += 1
            notes_parts.append("Has accessible URL")
        else:
            score -= 2
            notes_parts.append("No accessible URL (integration difficulty)")

        # Normalize to 1-10 scale
        final_score = max(1, min(10, int(round(score))))

        notes = "; ".join(notes_parts) if notes_parts else "Standard integration requirements"

        return final_score, notes

    def _assess_ethical_accessibility(
        self, source: DatasetSource
    ) -> tuple[int, str]:
        """
        Assess ethical accessibility of the dataset (1-10).

        Evaluates:
        - Legal availability for AI training purposes
        - Privacy and anonymization standards
        - Licensing compatibility with commercial use

        Args:
            source: The dataset source to evaluate

        Returns:
            Tuple of (score: int, notes: str)
        """
        score = 5  # Start with neutral score
        notes_parts = []

        # Check open access status (open access usually permits AI training)
        if source.open_access:
            score += 3
            notes_parts.append("Open access (likely permits AI training)")
        else:
            score -= 1
            notes_parts.append("Not open access (license verification needed)")

        # Check data availability
        if source.data_availability == "available":
            score += 2
            notes_parts.append("Data available (no access restrictions)")
        elif source.data_availability == "upon_request":
            score += 1
            notes_parts.append(
                "Data upon request (may have usage restrictions to verify)"
            )
        elif source.data_availability == "restricted":
            score -= 2
            notes_parts.append("Data restricted (ethical/legal review required)")
        else:
            score -= 1
            notes_parts.append("Data availability unknown (ethical review needed)")

        # Check if source is from reputable source (journals, repositories)
        if source.source_type in ["journal", "repository"]:
            score += 1
            notes_parts.append("Reputable source (likely has proper anonymization)")
        elif source.source_type == "clinical_trial":
            score += 1
            notes_parts.append(
                "Clinical trial (should have IRB approval and anonymization)"
            )

        # Check for DOI (formal publication usually has proper licensing)
        if source.doi:
            score += 1
            notes_parts.append("Formal publication (licensing typically clear)")

        # Normalize to 1-10 scale
        final_score = max(1, min(10, int(round(score))))

        # Flag for manual review if score is low
        if final_score < 5:
            notes_parts.append("⚠️ Manual ethical review recommended")

        notes = "; ".join(notes_parts) if notes_parts else "Standard ethical considerations"

        return final_score, notes

    def _calculate_overall_score(
        self,
        therapeutic_relevance: int,
        data_structure_quality: int,
        training_integration: int,
        ethical_accessibility: int,
    ) -> float:
        """
        Calculate overall quality score using weighted average.

        Args:
            therapeutic_relevance: Therapeutic relevance score (1-10)
            data_structure_quality: Data structure quality score (1-10)
            training_integration: Training integration score (1-10)
            ethical_accessibility: Ethical accessibility score (1-10)

        Returns:
            Overall score (0-10)
        """
        overall = (
            therapeutic_relevance * self.config.therapeutic_relevance_weight
            + data_structure_quality * self.config.data_structure_quality_weight
            + training_integration * self.config.training_integration_weight
            + ethical_accessibility * self.config.ethical_accessibility_weight
        )

        return round(overall, 2)

    def _calculate_priority_tier(self, overall_score: float) -> str:
        """
        Calculate priority tier based on overall score.

        Args:
            overall_score: Overall quality score (0-10)

        Returns:
            Priority tier: "high", "medium", or "low"
        """
        if overall_score >= self.config.high_priority_threshold:
            return "high"
        elif overall_score >= self.config.medium_priority_threshold:
            return "medium"
        else:
            return "low"

    def _identify_competitive_advantages(
        self, source: DatasetSource, therapeutic_relevance: int, data_structure_quality: int
    ) -> List[str]:
        """
        Identify competitive advantages of the dataset.

        Args:
            source: The dataset source
            therapeutic_relevance: Therapeutic relevance score
            data_structure_quality: Data structure quality score

        Returns:
            List of competitive advantage descriptions
        """
        advantages = []

        # High therapeutic relevance
        if therapeutic_relevance >= 8:
            advantages.append("High therapeutic relevance for counseling contexts")

        # High data structure quality
        if data_structure_quality >= 8:
            advantages.append("High-quality structured data with comprehensive metadata")

        # Contains transcripts
        text_to_check = f"{source.title} {source.abstract}".lower()
        if any(keyword in text_to_check for keyword in self.config.transcript_keywords):
            advantages.append("Contains therapy session transcripts (rare and valuable)")

        # Evidence-based practices
        if any(
            keyword in text_to_check
            for keyword in self.config.evidence_based_keywords
        ):
            advantages.append("Aligned with evidence-based therapeutic practices")

        # Open access with available data
        if source.open_access and source.data_availability == "available":
            advantages.append("Open access with directly available data (low integration barrier)")

        # Crisis intervention content
        if "crisis" in text_to_check or "suicide" in text_to_check:
            advantages.append("Crisis intervention content (specialized and valuable)")

        # Clinical trial data
        if source.source_type == "clinical_trial":
            advantages.append("Clinical trial data (rigorous and evidence-based)")

        return advantages

    def rank_datasets(
        self, evaluations: List[DatasetEvaluation]
    ) -> List[DatasetEvaluation]:
        """
        Rank datasets by overall score (descending).

        Args:
            evaluations: List of dataset evaluations

        Returns:
            List of evaluations sorted by overall score (highest first)
        """
        return sorted(evaluations, key=lambda e: e.overall_score, reverse=True)

    def generate_evaluation_report(
        self, evaluation: DatasetEvaluation, source: Optional[DatasetSource] = None
    ) -> str:
        """
        Generate a structured evaluation report in markdown format.

        Args:
            evaluation: The dataset evaluation
            source: Optional dataset source for additional context

        Returns:
            Markdown-formatted evaluation report
        """
        report_lines = [
            "# Dataset Evaluation Report",
            "",
            f"**Source ID**: {evaluation.source_id}",
            f"**Evaluation Date**: {evaluation.evaluation_date.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Evaluator**: {evaluation.evaluator}",
            "",
        ]

        if source:
            report_lines.extend(
                [
                    "## Dataset Information",
                    f"**Title**: {source.title}",
                    f"**Source Type**: {source.source_type}",
                    f"**URL**: {source.url}",
                    f"**DOI**: {source.doi or 'N/A'}",
                    f"**Open Access**: {source.open_access}",
                    f"**Data Availability**: {source.data_availability}",
                    "",
                ]
            )

        report_lines.extend(
            [
                "## Evaluation Scores",
                "",
                f"### Overall Score: {evaluation.overall_score:.2f}/10",
                f"**Priority Tier**: {evaluation.priority_tier.upper()}",
                "",
                "### Dimension Scores",
                "",
                f"1. **Therapeutic Relevance**: {evaluation.therapeutic_relevance}/10",
                f"   - {evaluation.therapeutic_relevance_notes}",
                "",
                f"2. **Data Structure Quality**: {evaluation.data_structure_quality}/10",
                f"   - {evaluation.data_structure_notes}",
                "",
                f"3. **Training Integration**: {evaluation.training_integration}/10",
                f"   - {evaluation.integration_notes}",
                "",
                f"4. **Ethical Accessibility**: {evaluation.ethical_accessibility}/10",
                f"   - {evaluation.ethical_notes}",
                "",
            ]
        )

        if evaluation.competitive_advantages:
            report_lines.extend(
                [
                    "## Competitive Advantages",
                    "",
                ]
            )
            for advantage in evaluation.competitive_advantages:
                report_lines.append(f"- {advantage}")
            report_lines.append("")

        return "\n".join(report_lines)

