"""
Clinical Accuracy Assessment Framework

This module provides comprehensive clinical accuracy assessment capabilities
for validating AI-generated therapeutic responses against expert standards.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalDomain(Enum):
    """Clinical domains for assessment categorization."""

    DSM5_DIAGNOSTIC = "dsm5_diagnostic"
    PDM2_PSYCHODYNAMIC = "pdm2_psychodynamic"
    THERAPEUTIC_INTERVENTION = "therapeutic_intervention"
    CRISIS_MANAGEMENT = "crisis_management"
    ETHICAL_BOUNDARIES = "ethical_boundaries"
    CULTURAL_COMPETENCY = "cultural_competency"
    TRAUMA_INFORMED = "trauma_informed"
    SUBSTANCE_USE = "substance_use"
    DEVELOPMENTAL = "developmental"
    FAMILY_SYSTEMS = "family_systems"


class AccuracyLevel(Enum):
    """Clinical accuracy assessment levels."""

    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"  # 80-89%
    ADEQUATE = "adequate"  # 70-79%
    NEEDS_IMPROVEMENT = "needs_improvement"  # 60-69%
    INADEQUATE = "inadequate"  # Below 60%


class ValidationStatus(Enum):
    """Validation status for clinical assessments."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REQUIRES_REVISION = "requires_revision"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ClinicalCriteria:
    """Clinical assessment criteria definition."""

    domain: ClinicalDomain
    criterion_id: str
    name: str
    description: str
    weight: float = 1.0
    required_accuracy: float = 0.8
    assessment_rubric: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate criteria configuration."""
        if not 0 <= self.weight <= 1:
            raise ValueError("Weight must be between 0 and 1")
        if not 0 <= self.required_accuracy <= 1:
            raise ValueError("Required accuracy must be between 0 and 1")


@dataclass
class AssessmentResult:
    """Individual assessment result."""

    criterion_id: str
    score: float
    accuracy_level: AccuracyLevel
    feedback: str
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    assessor_id: Optional[str] = None

    def __post_init__(self):
        """Validate assessment result."""
        if not 0 <= self.score <= 1:
            raise ValueError("Score must be between 0 and 1")


@dataclass
class ClinicalAssessment:
    """Complete clinical accuracy assessment."""

    assessment_id: str
    content_id: str
    domain: ClinicalDomain
    overall_score: float
    accuracy_level: AccuracyLevel
    status: ValidationStatus
    individual_results: List[AssessmentResult]
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    assessor_ids: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate clinical assessment."""
        if not 0 <= self.overall_score <= 1:
            raise ValueError("Overall score must be between 0 and 1")


class ClinicalAccuracyAssessmentFramework:
    """
    Comprehensive clinical accuracy assessment framework for validating
    AI-generated therapeutic content against expert clinical standards.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the clinical accuracy assessment framework."""
        self.criteria_registry: Dict[str, ClinicalCriteria] = {}
        self.assessments: Dict[str, ClinicalAssessment] = {}
        self.expert_profiles: Dict[str, Dict[str, Any]] = {}
        self.config = self._load_config(config_path)
        self._initialize_default_criteria()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "assessment_timeout_hours": 48,
            "minimum_assessors_per_domain": 2,
            "consensus_threshold": 0.8,
            "auto_approval_threshold": 0.9,
            "quality_gates": {
                "dsm5_diagnostic": 0.85,
                "crisis_management": 0.95,
                "ethical_boundaries": 0.9,
            },
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        return default_config

    def _initialize_default_criteria(self):
        """Initialize default clinical assessment criteria."""
        # DSM-5 Diagnostic Accuracy Criteria
        dsm5_criteria = [
            ClinicalCriteria(
                domain=ClinicalDomain.DSM5_DIAGNOSTIC,
                criterion_id="dsm5_diagnostic_accuracy",
                name="DSM-5 Diagnostic Accuracy",
                description="Accuracy of diagnostic impressions and criteria application",
                weight=1.0,
                required_accuracy=0.85,
                assessment_rubric={
                    "excellent": "Perfect application of DSM-5 criteria with nuanced understanding",
                    "good": "Correct application with minor gaps in nuance",
                    "adequate": "Generally correct with some misapplications",
                    "needs_improvement": "Significant gaps in diagnostic accuracy",
                    "inadequate": "Incorrect or harmful diagnostic impressions",
                },
            ),
            ClinicalCriteria(
                domain=ClinicalDomain.DSM5_DIAGNOSTIC,
                criterion_id="differential_diagnosis",
                name="Differential Diagnosis Consideration",
                description="Appropriate consideration of alternative diagnoses",
                weight=0.8,
                required_accuracy=0.8,
            ),
        ]

        # Crisis Management Criteria
        crisis_criteria = [
            ClinicalCriteria(
                domain=ClinicalDomain.CRISIS_MANAGEMENT,
                criterion_id="crisis_recognition",
                name="Crisis Recognition",
                description="Accurate identification of crisis situations",
                weight=1.0,
                required_accuracy=0.95,
                assessment_rubric={
                    "excellent": "Immediate and accurate crisis identification with appropriate urgency",
                    "good": "Timely crisis recognition with appropriate response",
                    "adequate": "Crisis recognized but with some delay or uncertainty",
                    "needs_improvement": "Delayed or uncertain crisis recognition",
                    "inadequate": "Failed to recognize clear crisis indicators",
                },
            ),
            ClinicalCriteria(
                domain=ClinicalDomain.CRISIS_MANAGEMENT,
                criterion_id="safety_planning",
                name="Safety Planning",
                description="Appropriate safety planning and intervention",
                weight=1.0,
                required_accuracy=0.95,
            ),
        ]

        # Therapeutic Intervention Criteria
        therapeutic_criteria = [
            ClinicalCriteria(
                domain=ClinicalDomain.THERAPEUTIC_INTERVENTION,
                criterion_id="intervention_appropriateness",
                name="Intervention Appropriateness",
                description="Appropriateness of therapeutic interventions for presenting concerns",
                weight=1.0,
                required_accuracy=0.8,
            ),
            ClinicalCriteria(
                domain=ClinicalDomain.THERAPEUTIC_INTERVENTION,
                criterion_id="timing_pacing",
                name="Timing and Pacing",
                description="Appropriate timing and pacing of interventions",
                weight=0.8,
                required_accuracy=0.75,
            ),
        ]

        # Register all criteria
        all_criteria = dsm5_criteria + crisis_criteria + therapeutic_criteria
        for criterion in all_criteria:
            self.register_criterion(criterion)

    def register_criterion(self, criterion: ClinicalCriteria) -> None:
        """Register a clinical assessment criterion."""
        self.criteria_registry[criterion.criterion_id] = criterion
        logger.info(f"Registered criterion: {criterion.name}")

    def get_criteria_by_domain(self, domain: ClinicalDomain) -> List[ClinicalCriteria]:
        """Get all criteria for a specific clinical domain."""
        return [
            criterion for criterion in self.criteria_registry.values() if criterion.domain == domain
        ]

    def create_assessment(
        self,
        content_id: str,
        domain: ClinicalDomain,
        assessor_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new clinical accuracy assessment."""
        assessment_id = (
            f"assessment_{content_id}_{domain.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        assessment = ClinicalAssessment(
            assessment_id=assessment_id,
            content_id=content_id,
            domain=domain,
            overall_score=0.0,
            accuracy_level=AccuracyLevel.INADEQUATE,
            status=ValidationStatus.PENDING,
            individual_results=[],
            assessor_ids={assessor_id},
            metadata=metadata or {},
        )

        self.assessments[assessment_id] = assessment
        logger.info(f"Created assessment {assessment_id} for content {content_id}")
        return assessment_id

    def conduct_assessment(
        self,
        assessment_id: str,
        content: str,
        assessor_id: str,
        individual_scores: Dict[str, float],
        feedback: Dict[str, str],
    ) -> ClinicalAssessment:
        """Conduct clinical accuracy assessment."""
        if assessment_id not in self.assessments:
            raise ValueError(f"Assessment {assessment_id} not found")

        assessment = self.assessments[assessment_id]
        assessment.status = ValidationStatus.IN_PROGRESS
        assessment.assessor_ids.add(assessor_id)

        # Get relevant criteria for this domain
        domain_criteria = self.get_criteria_by_domain(assessment.domain)

        # Create individual assessment results
        individual_results = []
        total_weighted_score = 0.0
        total_weight = 0.0

        for criterion in domain_criteria:
            if criterion.criterion_id in individual_scores:
                score = individual_scores[criterion.criterion_id]
                accuracy_level = self._determine_accuracy_level(score)

                result = AssessmentResult(
                    criterion_id=criterion.criterion_id,
                    score=score,
                    accuracy_level=accuracy_level,
                    feedback=feedback.get(criterion.criterion_id, ""),
                    assessor_id=assessor_id,
                )

                individual_results.append(result)
                total_weighted_score += score * criterion.weight
                total_weight += criterion.weight

        # Calculate overall score
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        assessment.overall_score = overall_score
        assessment.accuracy_level = self._determine_accuracy_level(overall_score)
        assessment.individual_results.extend(individual_results)
        assessment.completed_at = datetime.now()
        assessment.status = ValidationStatus.COMPLETED

        logger.info(f"Completed assessment {assessment_id} with score {overall_score:.3f}")
        return assessment

    def _determine_accuracy_level(self, score: float) -> AccuracyLevel:
        """Determine accuracy level based on score."""
        if score >= 0.9:
            return AccuracyLevel.EXCELLENT
        elif score >= 0.8:
            return AccuracyLevel.GOOD
        elif score >= 0.7:
            return AccuracyLevel.ADEQUATE
        elif score >= 0.6:
            return AccuracyLevel.NEEDS_IMPROVEMENT
        else:
            return AccuracyLevel.INADEQUATE

    def get_assessment(self, assessment_id: str) -> Optional[ClinicalAssessment]:
        """Get assessment by ID."""
        return self.assessments.get(assessment_id)

    def get_assessments_by_domain(self, domain: ClinicalDomain) -> List[ClinicalAssessment]:
        """Get all assessments for a specific domain."""
        return [
            assessment for assessment in self.assessments.values() if assessment.domain == domain
        ]

    def get_assessment_statistics(self, domain: Optional[ClinicalDomain] = None) -> Dict[str, Any]:
        """Get assessment statistics."""
        assessments = (
            self.get_assessments_by_domain(domain) if domain else list(self.assessments.values())
        )

        if not assessments:
            return {"total": 0, "average_score": 0.0, "accuracy_distribution": {}}

        scores = [a.overall_score for a in assessments if a.status == ValidationStatus.COMPLETED]
        accuracy_levels = [
            a.accuracy_level.value for a in assessments if a.status == ValidationStatus.COMPLETED
        ]

        from collections import Counter

        accuracy_distribution = Counter(accuracy_levels)

        return {
            "total": len(assessments),
            "completed": len(scores),
            "average_score": np.mean(scores) if scores else 0.0,
            "median_score": np.median(scores) if scores else 0.0,
            "accuracy_distribution": dict(accuracy_distribution),
            "pass_rate": len([s for s in scores if s >= 0.7]) / len(scores) if scores else 0.0,
        }

    def generate_assessment_report(self, assessment_id: str) -> Dict[str, Any]:
        """Generate comprehensive assessment report."""
        assessment = self.get_assessment(assessment_id)
        if not assessment:
            raise ValueError(f"Assessment {assessment_id} not found")

        # Get domain criteria for context
        domain_criteria = self.get_criteria_by_domain(assessment.domain)
        criteria_map = {c.criterion_id: c for c in domain_criteria}

        # Analyze individual results
        criterion_analysis = []
        for result in assessment.individual_results:
            criterion = criteria_map.get(result.criterion_id)
            if criterion:
                criterion_analysis.append(
                    {
                        "criterion_name": criterion.name,
                        "score": result.score,
                        "accuracy_level": result.accuracy_level.value,
                        "meets_requirement": result.score >= criterion.required_accuracy,
                        "feedback": result.feedback,
                        "weight": criterion.weight,
                    }
                )

        # Generate recommendations
        recommendations = self._generate_recommendations(assessment)

        return {
            "assessment_id": assessment_id,
            "content_id": assessment.content_id,
            "domain": assessment.domain.value,
            "overall_score": assessment.overall_score,
            "accuracy_level": assessment.accuracy_level.value,
            "status": assessment.status.value,
            "created_at": assessment.created_at.isoformat(),
            "completed_at": (
                assessment.completed_at.isoformat() if assessment.completed_at else None
            ),
            "criterion_analysis": criterion_analysis,
            "recommendations": recommendations,
            "metadata": assessment.metadata,
        }

    def _generate_recommendations(self, assessment: ClinicalAssessment) -> List[str]:
        """Generate improvement recommendations based on assessment results."""
        recommendations = []

        # Analyze weak areas
        weak_criteria = [result for result in assessment.individual_results if result.score < 0.7]

        if weak_criteria:
            recommendations.append(
                f"Focus on improving {len(weak_criteria)} criteria that scored below 70%"
            )

        # Domain-specific recommendations
        if assessment.domain == ClinicalDomain.CRISIS_MANAGEMENT and assessment.overall_score < 0.9:
            recommendations.append(
                "Crisis management requires exceptional accuracy - consider additional training"
            )

        if assessment.domain == ClinicalDomain.DSM5_DIAGNOSTIC and assessment.overall_score < 0.8:
            recommendations.append(
                "Review DSM-5 diagnostic criteria and differential diagnosis procedures"
            )

        # General recommendations based on accuracy level
        if assessment.accuracy_level == AccuracyLevel.NEEDS_IMPROVEMENT:
            recommendations.append("Significant improvement needed - recommend supervised practice")
        elif assessment.accuracy_level == AccuracyLevel.ADEQUATE:
            recommendations.append("Good foundation - focus on refining clinical judgment")

        return recommendations

    async def batch_assess(
        self, content_items: List[Dict[str, Any]], domain: ClinicalDomain, assessor_id: str
    ) -> List[str]:
        """Conduct batch assessment of multiple content items."""
        assessment_ids = []

        for item in content_items:
            assessment_id = self.create_assessment(
                content_id=item["content_id"],
                domain=domain,
                assessor_id=assessor_id,
                metadata=item.get("metadata", {}),
            )
            assessment_ids.append(assessment_id)

        logger.info(f"Created {len(assessment_ids)} assessments for batch processing")
        return assessment_ids

    def export_assessments(self, output_path: str, domain: Optional[ClinicalDomain] = None) -> None:
        """Export assessments to JSON file."""
        assessments = (
            self.get_assessments_by_domain(domain) if domain else list(self.assessments.values())
        )

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "domain_filter": domain.value if domain else None,
            "total_assessments": len(assessments),
            "assessments": [self.generate_assessment_report(a.assessment_id) for a in assessments],
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported {len(assessments)} assessments to {output_path}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize framework
    framework = ClinicalAccuracyAssessmentFramework()

    # Create sample assessment
    assessment_id = framework.create_assessment(
        content_id="sample_response_001",
        domain=ClinicalDomain.DSM5_DIAGNOSTIC,
        assessor_id="expert_001",
    )

    # Conduct assessment
    sample_scores = {"dsm5_diagnostic_accuracy": 0.85, "differential_diagnosis": 0.78}

    sample_feedback = {
        "dsm5_diagnostic_accuracy": "Good application of criteria with minor gaps",
        "differential_diagnosis": "Could consider additional differential diagnoses",
    }

    assessment = framework.conduct_assessment(
        assessment_id=assessment_id,
        content="Sample therapeutic response content",
        assessor_id="expert_001",
        individual_scores=sample_scores,
        feedback=sample_feedback,
    )

    # Generate report
    report = framework.generate_assessment_report(assessment_id)
    print(json.dumps(report, indent=2))

    # Get statistics
    stats = framework.get_assessment_statistics(ClinicalDomain.DSM5_DIAGNOSTIC)
    print(f"Assessment Statistics: {stats}")
