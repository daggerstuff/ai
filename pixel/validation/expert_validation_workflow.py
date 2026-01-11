"""
Expert Validation Interface and Workflow System

This module provides a comprehensive interface for expert validation of clinical
content, including workflow management, expert assignment, and consensus building.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from clinical_accuracy_assessment import (
    ClinicalAccuracyAssessmentFramework,
    ClinicalDomain,
    ValidationStatus,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpertLevel(Enum):
    """Expert qualification levels."""

    JUNIOR = "junior"  # 1-3 years experience
    SENIOR = "senior"  # 4-10 years experience
    PRINCIPAL = "principal"  # 10+ years experience
    DISTINGUISHED = "distinguished"  # 15+ years, recognized expertise


class ExpertSpecialty(Enum):
    """Expert specialty areas."""

    CLINICAL_PSYCHOLOGY = "clinical_psychology"
    PSYCHIATRY = "psychiatry"
    COUNSELING = "counseling"
    SOCIAL_WORK = "social_work"
    MARRIAGE_FAMILY_THERAPY = "marriage_family_therapy"
    SUBSTANCE_ABUSE = "substance_abuse"
    TRAUMA_THERAPY = "trauma_therapy"
    CHILD_ADOLESCENT = "child_adolescent"
    GERIATRIC = "geriatric"
    NEUROPSYCHOLOGY = "neuropsychology"


class ValidationPriority(Enum):
    """Validation request priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class WorkflowStatus(Enum):
    """Workflow status tracking."""

    CREATED = "created"
    EXPERT_ASSIGNED = "expert_assigned"
    IN_REVIEW = "in_review"
    AWAITING_CONSENSUS = "awaiting_consensus"
    CONSENSUS_REACHED = "consensus_reached"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"


@dataclass
class ExpertProfile:
    """Expert profile and qualifications."""

    expert_id: str
    name: str
    email: str
    level: ExpertLevel
    specialties: List[ExpertSpecialty]
    domains: List[ClinicalDomain]
    years_experience: int
    certifications: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: ["en"])
    availability_hours: Dict[str, List[str]] = field(default_factory=dict)  # day -> hours
    max_concurrent_reviews: int = 5
    current_workload: int = 0
    rating: float = 5.0  # 1-5 scale
    total_reviews: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True

    def __post_init__(self):
        """Validate expert profile."""
        if not 0 <= self.years_experience <= 50:
            raise ValueError("Years of experience must be between 0 and 50")
        if not 1.0 <= self.rating <= 5.0:
            raise ValueError("Rating must be between 1.0 and 5.0")


@dataclass
class ValidationRequest:
    """Expert validation request."""

    request_id: str
    content_id: str
    content_text: str
    domain: ClinicalDomain
    priority: ValidationPriority
    requester_id: str
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    required_expert_level: ExpertLevel = ExpertLevel.SENIOR
    required_specialties: List[ExpertSpecialty] = field(default_factory=list)
    assigned_experts: Set[str] = field(default_factory=set)
    completed_reviews: Dict[str, str] = field(default_factory=dict)  # expert_id -> assessment_id
    status: WorkflowStatus = WorkflowStatus.CREATED
    metadata: Dict[str, Any] = field(default_factory=dict)
    consensus_threshold: float = 0.8
    minimum_reviewers: int = 2

    def __post_init__(self):
        """Set default deadline if not provided."""
        if self.deadline is None:
            # Set deadline based on priority
            hours_map = {
                ValidationPriority.CRITICAL: 4,
                ValidationPriority.URGENT: 12,
                ValidationPriority.HIGH: 24,
                ValidationPriority.NORMAL: 48,
                ValidationPriority.LOW: 72,
            }
            self.deadline = self.created_at + timedelta(hours=hours_map[self.priority])


@dataclass
class ConsensusResult:
    """Consensus analysis result."""

    request_id: str
    consensus_score: float
    agreement_level: str
    individual_scores: Dict[str, float]
    average_score: float
    score_variance: float
    recommendations: List[str]
    requires_additional_review: bool
    final_decision: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


class ExpertValidationWorkflow:
    """
    Comprehensive expert validation workflow system for managing
    clinical content validation by qualified experts.
    """

    def __init__(self, assessment_framework: ClinicalAccuracyAssessmentFramework):
        """Initialize expert validation workflow."""
        self.assessment_framework = assessment_framework
        self.experts: Dict[str, ExpertProfile] = {}
        self.validation_requests: Dict[str, ValidationRequest] = {}
        self.consensus_results: Dict[str, ConsensusResult] = {}
        self.notification_callbacks: List[Callable] = []
        self.config = self._load_default_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default workflow configuration."""
        return {
            "auto_assignment": True,
            "consensus_threshold": 0.8,
            "minimum_reviewers": 2,
            "maximum_reviewers": 5,
            "escalation_threshold": 0.6,  # If consensus below this, escalate
            "reminder_intervals": [24, 12, 4],  # Hours before deadline
            "expert_selection_algorithm": "weighted_random",  # or "round_robin", "expertise_match"
            "notification_enabled": True,
            "email_config": {"smtp_server": "localhost", "smtp_port": 587, "use_tls": True},
        }

    def register_expert(self, expert: ExpertProfile) -> None:
        """Register a new expert in the system."""
        self.experts[expert.expert_id] = expert
        logger.info(f"Registered expert: {expert.name} ({expert.expert_id})")

    def get_expert(self, expert_id: str) -> Optional[ExpertProfile]:
        """Get expert profile by ID."""
        return self.experts.get(expert_id)

    def get_available_experts(
        self,
        domain: ClinicalDomain,
        required_level: ExpertLevel = ExpertLevel.SENIOR,
        required_specialties: Optional[List[ExpertSpecialty]] = None,
    ) -> List[ExpertProfile]:
        """Get available experts matching criteria."""
        available_experts = []

        for expert in self.experts.values():
            if not expert.is_active:
                continue

            # Check workload
            if expert.current_workload >= expert.max_concurrent_reviews:
                continue

            # Check domain expertise
            if domain not in expert.domains:
                continue

            # Check experience level
            level_hierarchy = {
                ExpertLevel.JUNIOR: 1,
                ExpertLevel.SENIOR: 2,
                ExpertLevel.PRINCIPAL: 3,
                ExpertLevel.DISTINGUISHED: 4,
            }
            if level_hierarchy[expert.level] < level_hierarchy[required_level]:
                continue

            # Check specialties if required
            if required_specialties:
                if not any(specialty in expert.specialties for specialty in required_specialties):
                    continue

            available_experts.append(expert)

        return available_experts

    def create_validation_request(
        self,
        content_id: str,
        content_text: str,
        domain: ClinicalDomain,
        requester_id: str,
        priority: ValidationPriority = ValidationPriority.NORMAL,
        required_expert_level: ExpertLevel = ExpertLevel.SENIOR,
        required_specialties: Optional[List[ExpertSpecialty]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new validation request."""
        request_id = f"validation_{uuid.uuid4().hex[:8]}"

        request = ValidationRequest(
            request_id=request_id,
            content_id=content_id,
            content_text=content_text,
            domain=domain,
            priority=priority,
            requester_id=requester_id,
            required_expert_level=required_expert_level,
            required_specialties=required_specialties or [],
            metadata=metadata or {},
        )

        self.validation_requests[request_id] = request
        logger.info(f"Created validation request {request_id} for content {content_id}")

        # Auto-assign experts if enabled
        if self.config["auto_assignment"]:
            self.assign_experts(request_id)

        return request_id

    def assign_experts(self, request_id: str, expert_ids: Optional[List[str]] = None) -> List[str]:
        """Assign experts to validation request."""
        if request_id not in self.validation_requests:
            raise ValueError(f"Validation request {request_id} not found")

        request = self.validation_requests[request_id]

        if expert_ids:
            # Manual assignment
            assigned_experts = []
            for expert_id in expert_ids:
                if expert_id in self.experts and self.experts[expert_id].is_active:
                    request.assigned_experts.add(expert_id)
                    self.experts[expert_id].current_workload += 1
                    assigned_experts.append(expert_id)

        else:
            # Automatic assignment
            available_experts = self.get_available_experts(
                domain=request.domain,
                required_level=request.required_expert_level,
                required_specialties=request.required_specialties,
            )

            # Select experts based on algorithm
            selected_experts = self._select_experts(
                available_experts, request.minimum_reviewers, self.config["maximum_reviewers"]
            )

            assigned_experts = []
            for expert in selected_experts:
                request.assigned_experts.add(expert.expert_id)
                expert.current_workload += 1
                assigned_experts.append(expert.expert_id)

        if assigned_experts:
            request.status = WorkflowStatus.EXPERT_ASSIGNED
            self._notify_experts_assigned(request, assigned_experts)
            logger.info(f"Assigned {len(assigned_experts)} experts to request {request_id}")

        return assigned_experts

    def _select_experts(
        self, available_experts: List[ExpertProfile], min_reviewers: int, max_reviewers: int
    ) -> List[ExpertProfile]:
        """Select experts using configured algorithm."""
        if len(available_experts) < min_reviewers:
            logger.warning(f"Only {len(available_experts)} experts available, need {min_reviewers}")
            return available_experts

        algorithm = self.config["expert_selection_algorithm"]

        if algorithm == "weighted_random":
            # Weight by rating and experience
            import random

            weights = []
            for expert in available_experts:
                weight = expert.rating * (1 + expert.years_experience / 20)
                weights.append(weight)

            # Select based on weights
            selected = []
            remaining = available_experts.copy()
            remaining_weights = weights.copy()

            num_to_select = min(max_reviewers, len(available_experts))
            for _ in range(num_to_select):
                if not remaining:
                    break

                total_weight = sum(remaining_weights)
                if total_weight == 0:
                    selected.extend(remaining[: min_reviewers - len(selected)])
                    break

                # Weighted random selection
                r = random.uniform(0, total_weight)
                cumulative = 0
                for i, weight in enumerate(remaining_weights):
                    cumulative += weight
                    if r <= cumulative:
                        selected.append(remaining[i])
                        remaining.pop(i)
                        remaining_weights.pop(i)
                        break

            return selected

        elif algorithm == "expertise_match":
            # Sort by relevance to domain and specialties
            scored_experts = []
            for expert in available_experts:
                score = expert.rating
                # Bonus for years of experience
                score += min(expert.years_experience / 10, 2.0)
                # Bonus for distinguished level
                if expert.level == ExpertLevel.DISTINGUISHED:
                    score += 1.0
                elif expert.level == ExpertLevel.PRINCIPAL:
                    score += 0.5

                scored_experts.append((score, expert))

            # Sort by score and take top experts
            scored_experts.sort(key=lambda x: x[0], reverse=True)
            num_to_select = min(max_reviewers, len(scored_experts))
            return [expert for _, expert in scored_experts[:num_to_select]]

        else:  # round_robin
            # Simple round-robin selection
            import random

            shuffled = available_experts.copy()
            random.shuffle(shuffled)
            num_to_select = min(max_reviewers, len(shuffled))
            return shuffled[:num_to_select]

    def submit_expert_review(
        self,
        request_id: str,
        expert_id: str,
        individual_scores: Dict[str, float],
        feedback: Dict[str, str],
        overall_recommendation: str,
    ) -> str:
        """Submit expert review for validation request."""
        if request_id not in self.validation_requests:
            raise ValueError(f"Validation request {request_id} not found")

        request = self.validation_requests[request_id]

        if expert_id not in request.assigned_experts:
            raise ValueError(f"Expert {expert_id} not assigned to request {request_id}")

        if expert_id in request.completed_reviews:
            raise ValueError(
                f"Expert {expert_id} already submitted review for request {request_id}"
            )

        # Create assessment using the framework
        assessment_id = self.assessment_framework.create_assessment(
            content_id=request.content_id,
            domain=request.domain,
            assessor_id=expert_id,
            metadata={
                "validation_request_id": request_id,
                "overall_recommendation": overall_recommendation,
            },
        )

        # Conduct assessment
        self.assessment_framework.conduct_assessment(
            assessment_id=assessment_id,
            content=request.content_text,
            assessor_id=expert_id,
            individual_scores=individual_scores,
            feedback=feedback,
        )

        # Record completion
        request.completed_reviews[expert_id] = assessment_id

        # Update expert workload
        if expert_id in self.experts:
            self.experts[expert_id].current_workload = max(
                0, self.experts[expert_id].current_workload - 1
            )
            self.experts[expert_id].total_reviews += 1

        # Update request status
        if len(request.completed_reviews) >= request.minimum_reviewers:
            request.status = WorkflowStatus.AWAITING_CONSENSUS
            # Check for consensus
            self._check_consensus(request_id)
        else:
            request.status = WorkflowStatus.IN_REVIEW

        logger.info(f"Expert {expert_id} submitted review for request {request_id}")
        return assessment_id

    def _check_consensus(self, request_id: str) -> Optional[ConsensusResult]:
        """Check if consensus has been reached among expert reviews."""
        request = self.validation_requests[request_id]

        if len(request.completed_reviews) < request.minimum_reviewers:
            return None

        # Get all assessment results
        assessment_scores = []
        individual_scores = {}

        for expert_id, assessment_id in request.completed_reviews.items():
            assessment = self.assessment_framework.get_assessment(assessment_id)
            if assessment and assessment.status == ValidationStatus.COMPLETED:
                assessment_scores.append(assessment.overall_score)
                individual_scores[expert_id] = assessment.overall_score

        if len(assessment_scores) < request.minimum_reviewers:
            return None

        # Calculate consensus metrics
        import numpy as np

        average_score = np.mean(assessment_scores)
        score_variance = np.var(assessment_scores)

        # Calculate consensus score (inverse of coefficient of variation)
        if average_score > 0:
            cv = np.sqrt(score_variance) / average_score
            consensus_score = max(0, 1 - cv)
        else:
            consensus_score = 0.0

        # Determine agreement level
        if consensus_score >= 0.9:
            agreement_level = "strong"
        elif consensus_score >= 0.7:
            agreement_level = "moderate"
        elif consensus_score >= 0.5:
            agreement_level = "weak"
        else:
            agreement_level = "poor"

        # Generate recommendations
        recommendations = self._generate_consensus_recommendations(
            consensus_score, average_score, score_variance, assessment_scores
        )

        # Determine if additional review is needed
        requires_additional_review = (
            consensus_score < request.consensus_threshold
            or consensus_score < self.config["escalation_threshold"]
        )

        # Create consensus result
        consensus_result = ConsensusResult(
            request_id=request_id,
            consensus_score=consensus_score,
            agreement_level=agreement_level,
            individual_scores=individual_scores,
            average_score=average_score,
            score_variance=score_variance,
            recommendations=recommendations,
            requires_additional_review=requires_additional_review,
        )

        self.consensus_results[request_id] = consensus_result

        # Update request status
        if requires_additional_review:
            request.status = WorkflowStatus.ESCALATED
            self._handle_escalation(request_id)
        else:
            request.status = WorkflowStatus.CONSENSUS_REACHED
            consensus_result.final_decision = (
                "approved" if average_score >= 0.7 else "needs_revision"
            )

        logger.info(
            f"Consensus analysis completed for request {request_id}: {agreement_level} agreement"
        )
        return consensus_result

    def _generate_consensus_recommendations(
        self,
        consensus_score: float,
        average_score: float,
        score_variance: float,
        individual_scores: List[float],
    ) -> List[str]:
        """Generate recommendations based on consensus analysis."""
        recommendations = []

        if consensus_score < 0.5:
            recommendations.append("Poor consensus - consider additional expert review")
        elif consensus_score < 0.7:
            recommendations.append("Moderate consensus - review conflicting assessments")

        if score_variance > 0.1:
            recommendations.append("High score variance - investigate assessment differences")

        if average_score < 0.6:
            recommendations.append("Low average score - content requires significant revision")
        elif average_score < 0.8:
            recommendations.append("Moderate score - content needs improvement")

        # Check for outliers
        import numpy as np

        mean_score = np.mean(individual_scores)
        std_score = np.std(individual_scores)
        outliers = [score for score in individual_scores if abs(score - mean_score) > 2 * std_score]

        if outliers:
            recommendations.append(f"Found {len(outliers)} outlier assessment(s) - review for bias")

        return recommendations

    def _handle_escalation(self, request_id: str) -> None:
        """Handle escalated validation requests."""
        request = self.validation_requests[request_id]

        # Find distinguished experts for escalation
        distinguished_experts = self.get_available_experts(
            domain=request.domain, required_level=ExpertLevel.DISTINGUISHED
        )

        if distinguished_experts:
            # Assign one distinguished expert
            self.assign_experts(request_id, [distinguished_experts[0].expert_id])
            logger.info(f"Escalated request {request_id} to distinguished expert")
        else:
            # No distinguished experts available - mark for manual review
            request.metadata["escalation_reason"] = "No distinguished experts available"
            logger.warning(f"Request {request_id} escalated but no distinguished experts available")

    def get_validation_request(self, request_id: str) -> Optional[ValidationRequest]:
        """Get validation request by ID."""
        return self.validation_requests.get(request_id)

    def get_consensus_result(self, request_id: str) -> Optional[ConsensusResult]:
        """Get consensus result by request ID."""
        return self.consensus_results.get(request_id)

    def get_expert_workload(self, expert_id: str) -> Dict[str, Any]:
        """Get expert's current workload and statistics."""
        expert = self.get_expert(expert_id)
        if not expert:
            return {}

        # Find active assignments
        active_requests = [
            req
            for req in self.validation_requests.values()
            if expert_id in req.assigned_experts
            and req.status
            in [
                WorkflowStatus.EXPERT_ASSIGNED,
                WorkflowStatus.IN_REVIEW,
                WorkflowStatus.AWAITING_CONSENSUS,
            ]
        ]

        # Calculate average review time
        completed_requests = [
            req for req in self.validation_requests.values() if expert_id in req.completed_reviews
        ]

        return {
            "expert_id": expert_id,
            "current_workload": expert.current_workload,
            "max_concurrent_reviews": expert.max_concurrent_reviews,
            "active_requests": len(active_requests),
            "total_completed": len(completed_requests),
            "rating": expert.rating,
            "utilization": expert.current_workload / expert.max_concurrent_reviews,
        }

    def generate_workflow_report(self, start_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate comprehensive workflow performance report."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)

        # Filter requests by date
        recent_requests = [
            req for req in self.validation_requests.values() if req.created_at >= start_date
        ]

        # Calculate metrics
        total_requests = len(recent_requests)
        completed_requests = [
            req for req in recent_requests if req.status == WorkflowStatus.COMPLETED
        ]
        consensus_reached = [
            req for req in recent_requests if req.status == WorkflowStatus.CONSENSUS_REACHED
        ]
        escalated_requests = [
            req for req in recent_requests if req.status == WorkflowStatus.ESCALATED
        ]

        # Calculate average processing time
        processing_times = []
        for req in completed_requests:
            if req.completed_reviews:
                # Find latest completion time
                latest_completion = None
                for assessment_id in req.completed_reviews.values():
                    assessment = self.assessment_framework.get_assessment(assessment_id)
                    if assessment and assessment.completed_at:
                        if latest_completion is None or assessment.completed_at > latest_completion:
                            latest_completion = assessment.completed_at

                if latest_completion:
                    processing_time = (
                        latest_completion - req.created_at
                    ).total_seconds() / 3600  # hours
                    processing_times.append(processing_time)

        # Expert utilization
        expert_stats = {}
        for expert_id, expert in self.experts.items():
            expert_stats[expert_id] = self.get_expert_workload(expert_id)

        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": datetime.now().isoformat(),
            },
            "request_metrics": {
                "total_requests": total_requests,
                "completed_requests": len(completed_requests),
                "consensus_reached": len(consensus_reached),
                "escalated_requests": len(escalated_requests),
                "completion_rate": (
                    len(completed_requests) / total_requests if total_requests > 0 else 0
                ),
                "escalation_rate": (
                    len(escalated_requests) / total_requests if total_requests > 0 else 0
                ),
            },
            "processing_metrics": {
                "average_processing_time_hours": (
                    sum(processing_times) / len(processing_times) if processing_times else 0
                ),
                "median_processing_time_hours": (
                    sorted(processing_times)[len(processing_times) // 2] if processing_times else 0
                ),
                "total_processing_time_hours": sum(processing_times),
            },
            "expert_utilization": expert_stats,
            "domain_breakdown": self._get_domain_breakdown(recent_requests),
            "priority_breakdown": self._get_priority_breakdown(recent_requests),
        }

    def _get_domain_breakdown(self, requests: List[ValidationRequest]) -> Dict[str, int]:
        """Get breakdown of requests by domain."""
        from collections import Counter

        domains = [req.domain.value for req in requests]
        return dict(Counter(domains))

    def _get_priority_breakdown(self, requests: List[ValidationRequest]) -> Dict[str, int]:
        """Get breakdown of requests by priority."""
        from collections import Counter

        priorities = [req.priority.value for req in requests]
        return dict(Counter(priorities))

    def _notify_experts_assigned(self, request: ValidationRequest, expert_ids: List[str]) -> None:
        """Notify experts of new assignment."""
        if not self.config["notification_enabled"]:
            return

        for expert_id in expert_ids:
            expert = self.get_expert(expert_id)
            if expert:
                self._send_notification(
                    expert.email,
                    f"New Validation Assignment: {request.content_id}",
                    f"You have been assigned to review content {request.content_id} in domain {request.domain.value}. "
                    f"Priority: {request.priority.value}. Deadline: {request.deadline.strftime('%Y-%m-%d %H:%M')}",
                )

    def _send_notification(self, email: str, subject: str, message: str) -> None:
        """Send email notification to expert."""
        try:
            # This is a simplified email notification
            # In production, you would use a proper email service
            logger.info(f"Email notification sent to {email}: {subject}")
        except Exception as e:
            logger.error(f"Failed to send notification to {email}: {e}")

    def add_notification_callback(self, callback: Callable) -> None:
        """Add custom notification callback."""
        self.notification_callbacks.append(callback)

    async def process_pending_requests(self) -> None:
        """Process all pending validation requests."""
        pending_requests = [
            req
            for req in self.validation_requests.values()
            if req.status in [WorkflowStatus.CREATED, WorkflowStatus.EXPERT_ASSIGNED]
        ]

        for request in pending_requests:
            if request.status == WorkflowStatus.CREATED:
                self.assign_experts(request.request_id)
            elif request.status == WorkflowStatus.EXPERT_ASSIGNED:
                # Check for overdue assignments
                if datetime.now() > request.deadline:
                    logger.warning(f"Request {request.request_id} is overdue")
                    # Could implement reminder logic here


# Example usage and testing
if __name__ == "__main__":
    # Initialize systems
    assessment_framework = ClinicalAccuracyAssessmentFramework()
    workflow = ExpertValidationWorkflow(assessment_framework)

    # Register sample experts
    expert1 = ExpertProfile(
        expert_id="expert_001",
        name="Dr. Sarah Johnson",
        email="sarah.johnson@example.com",
        level=ExpertLevel.PRINCIPAL,
        specialties=[ExpertSpecialty.CLINICAL_PSYCHOLOGY, ExpertSpecialty.TRAUMA_THERAPY],
        domains=[ClinicalDomain.DSM5_DIAGNOSTIC, ClinicalDomain.THERAPEUTIC_INTERVENTION],
        years_experience=12,
    )

    expert2 = ExpertProfile(
        expert_id="expert_002",
        name="Dr. Michael Chen",
        email="michael.chen@example.com",
        level=ExpertLevel.DISTINGUISHED,
        specialties=[ExpertSpecialty.PSYCHIATRY, ExpertSpecialty.CRISIS_MANAGEMENT],
        domains=[ClinicalDomain.CRISIS_MANAGEMENT, ClinicalDomain.DSM5_DIAGNOSTIC],
        years_experience=18,
    )

    workflow.register_expert(expert1)
    workflow.register_expert(expert2)

    # Create validation request
    request_id = workflow.create_validation_request(
        content_id="sample_therapeutic_response_001",
        content_text="Sample therapeutic response for validation",
        domain=ClinicalDomain.DSM5_DIAGNOSTIC,
        requester_id="system_001",
        priority=ValidationPriority.HIGH,
    )

    print(f"Created validation request: {request_id}")

    # Generate workflow report
    report = workflow.generate_workflow_report()
    print(json.dumps(report, indent=2))
