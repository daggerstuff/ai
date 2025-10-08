"""
Expert Validation Interface and Workflow

This module provides a comprehensive interface for expert validation of clinical
accuracy assessments, including expert assignment, validation workflows, and
consensus evaluation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import json
import uuid
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .clinical_accuracy_validator import ClinicalAccuracyResult, ClinicalAccuracyLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpertSpecialty(Enum):
    """Expert specialty areas"""

    CLINICAL_PSYCHOLOGY = "clinical_psychology"
    PSYCHIATRY = "psychiatry"
    SOCIAL_WORK = "social_work"
    COUNSELING = "counseling"
    NEUROPSYCHOLOGY = "neuropsychology"
    CHILD_PSYCHOLOGY = "child_psychology"
    TRAUMA_THERAPY = "trauma_therapy"
    SUBSTANCE_ABUSE = "substance_abuse"
    CRISIS_INTERVENTION = "crisis_intervention"


class ValidationStatus(Enum):
    """Validation request status"""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ValidationDecision(Enum):
    """Expert validation decision"""

    APPROVE = "approve"
    REJECT = "reject"
    NEEDS_REVISION = "needs_revision"
    ESCALATE = "escalate"


class ValidationPriority(Enum):
    """Validation priority levels"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


@dataclass
class ExpertProfile:
    """Expert profile information"""

    expert_id: str
    name: str
    email: str
    specialties: List[ExpertSpecialty]
    credentials: List[str]
    years_experience: int
    license_number: str
    institution: str
    availability_hours: Dict[str, List[str]] = field(default_factory=dict)  # day -> hours
    max_concurrent_validations: int = 5
    average_response_time_hours: float = 24.0
    validation_count: int = 0
    accuracy_rating: float = 0.0
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_active: Optional[datetime] = None


@dataclass
class ValidationRequest:
    """Expert validation request"""

    request_id: str
    assessment_result: ClinicalAccuracyResult
    priority: ValidationPriority
    required_specialties: List[ExpertSpecialty]
    deadline: datetime
    min_experts: int = 1
    max_experts: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    status: ValidationStatus = ValidationStatus.PENDING
    assigned_experts: List[str] = field(default_factory=list)
    validation_responses: List["ValidationResponse"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResponse:
    """Expert validation response"""

    response_id: str
    request_id: str
    expert_id: str
    decision: ValidationDecision
    confidence_score: float  # 0.0 to 1.0
    clinical_accuracy_score: float  # 0.0 to 1.0
    therapeutic_appropriateness_score: float  # 0.0 to 1.0
    safety_assessment_score: float  # 0.0 to 1.0
    detailed_feedback: str
    specific_concerns: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    time_spent_minutes: int = 0
    submitted_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusResult:
    """Consensus evaluation result"""

    request_id: str
    final_decision: ValidationDecision
    consensus_confidence: float
    expert_agreement_level: float  # 0.0 to 1.0
    aggregated_scores: Dict[str, float]
    consolidated_feedback: str
    action_items: List[str] = field(default_factory=list)
    dissenting_opinions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationMetrics:
    """Validation system metrics"""

    total_requests: int = 0
    pending_requests: int = 0
    completed_requests: int = 0
    average_response_time_hours: float = 0.0
    expert_utilization: Dict[str, float] = field(default_factory=dict)
    consensus_rate: float = 0.0
    approval_rate: float = 0.0
    escalation_rate: float = 0.0
    quality_scores: Dict[str, float] = field(default_factory=dict)


class ExpertValidationInterface:
    """
    Comprehensive expert validation interface and workflow management

    This class manages expert profiles, validation requests, assignment logic,
    consensus evaluation, and workflow automation.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the expert validation interface"""
        self.config = self._load_config(config_path)
        self.experts: Dict[str, ExpertProfile] = {}
        self.validation_requests: Dict[str, ValidationRequest] = {}
        self.validation_responses: Dict[str, ValidationResponse] = {}
        self.consensus_results: Dict[str, ConsensusResult] = {}

        # Load existing data
        self._load_expert_profiles()
        self._load_validation_history()

        # Initialize notification system
        self.notification_callbacks: List[Callable] = []

        logger.info("Expert validation interface initialized")

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration settings"""
        default_config = {
            "default_deadline_hours": 48,
            "urgent_deadline_hours": 4,
            "critical_deadline_hours": 1,
            "min_consensus_agreement": 0.7,
            "max_assignment_attempts": 3,
            "expert_timeout_hours": 72,
            "notification_enabled": True,
            "email_settings": {
                "smtp_server": "localhost",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "from_address": "validation@pixelatedempathy.com",
            },
        }

        if config_path and config_path.exists():
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        return default_config

    def _load_expert_profiles(self) -> None:
        """Load expert profiles from storage"""
        # In production, this would load from a database
        # For now, creating sample expert profiles
        sample_experts = [
            ExpertProfile(
                expert_id="exp_001",
                name="Dr. Sarah Johnson",
                email="sarah.johnson@example.com",
                specialties=[ExpertSpecialty.CLINICAL_PSYCHOLOGY, ExpertSpecialty.TRAUMA_THERAPY],
                credentials=["PhD Clinical Psychology", "Licensed Psychologist"],
                years_experience=15,
                license_number="PSY12345",
                institution="University Medical Center",
                availability_hours={
                    "monday": ["09:00", "17:00"],
                    "tuesday": ["09:00", "17:00"],
                    "wednesday": ["09:00", "17:00"],
                    "thursday": ["09:00", "17:00"],
                    "friday": ["09:00", "15:00"],
                },
                accuracy_rating=0.92,
            ),
            ExpertProfile(
                expert_id="exp_002",
                name="Dr. Michael Chen",
                email="michael.chen@example.com",
                specialties=[ExpertSpecialty.PSYCHIATRY, ExpertSpecialty.CRISIS_INTERVENTION],
                credentials=["MD Psychiatry", "Board Certified Psychiatrist"],
                years_experience=20,
                license_number="MD67890",
                institution="Regional Medical Center",
                accuracy_rating=0.95,
            ),
            ExpertProfile(
                expert_id="exp_003",
                name="Dr. Emily Rodriguez",
                email="emily.rodriguez@example.com",
                specialties=[ExpertSpecialty.SOCIAL_WORK, ExpertSpecialty.SUBSTANCE_ABUSE],
                credentials=["MSW", "LCSW", "Certified Addiction Counselor"],
                years_experience=12,
                license_number="SW11111",
                institution="Community Health Services",
                accuracy_rating=0.88,
            ),
        ]

        for expert in sample_experts:
            self.experts[expert.expert_id] = expert

    def _load_validation_history(self) -> None:
        """Load validation history from storage"""
        # In production, this would load from a database
        pass

    def add_expert(self, expert: ExpertProfile) -> bool:
        """Add a new expert to the system"""
        try:
            if expert.expert_id in self.experts:
                logger.warning(f"Expert {expert.expert_id} already exists")
                return False

            # Validate expert profile
            if not self._validate_expert_profile(expert):
                return False

            self.experts[expert.expert_id] = expert
            logger.info(f"Added expert: {expert.name} ({expert.expert_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to add expert: {e}")
            return False

    def _validate_expert_profile(self, expert: ExpertProfile) -> bool:
        """Validate expert profile completeness"""
        required_fields = [
            expert.name,
            expert.email,
            expert.license_number,
            expert.institution,
            expert.specialties,
            expert.credentials,
        ]

        if not all(required_fields):
            logger.error("Expert profile missing required fields")
            return False

        if expert.years_experience < 0:
            logger.error("Invalid years of experience")
            return False

        if not expert.specialties:
            logger.error("Expert must have at least one specialty")
            return False

        return True

    def update_expert(self, expert_id: str, updates: Dict[str, Any]) -> bool:
        """Update expert profile"""
        try:
            if expert_id not in self.experts:
                logger.error(f"Expert {expert_id} not found")
                return False

            expert = self.experts[expert_id]

            # Update allowed fields
            allowed_updates = [
                "name",
                "email",
                "specialties",
                "credentials",
                "institution",
                "availability_hours",
                "max_concurrent_validations",
                "is_active",
            ]

            for field, value in updates.items():
                if field in allowed_updates:
                    setattr(expert, field, value)

            expert.last_active = datetime.now()
            logger.info(f"Updated expert: {expert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update expert {expert_id}: {e}")
            return False

    def get_expert(self, expert_id: str) -> Optional[ExpertProfile]:
        """Get expert profile by ID"""
        return self.experts.get(expert_id)

    def list_experts(
        self, specialty: Optional[ExpertSpecialty] = None, active_only: bool = True
    ) -> List[ExpertProfile]:
        """List experts with optional filtering"""
        experts = list(self.experts.values())

        if active_only:
            experts = [e for e in experts if e.is_active]

        if specialty:
            experts = [e for e in experts if specialty in e.specialties]

        return experts

    async def create_validation_request(
        self,
        assessment_result: ClinicalAccuracyResult,
        priority: ValidationPriority = ValidationPriority.NORMAL,
        required_specialties: Optional[List[ExpertSpecialty]] = None,
        min_experts: int = 1,
        max_experts: int = 3,
        custom_deadline: Optional[datetime] = None,
    ) -> str:
        """Create a new validation request"""
        try:
            request_id = f"val_{uuid.uuid4().hex[:8]}"

            # Determine deadline based on priority
            if custom_deadline:
                deadline = custom_deadline
            else:
                deadline_hours = {
                    ValidationPriority.LOW: 72,
                    ValidationPriority.NORMAL: self.config["default_deadline_hours"],
                    ValidationPriority.HIGH: 24,
                    ValidationPriority.URGENT: self.config["urgent_deadline_hours"],
                    ValidationPriority.CRITICAL: self.config["critical_deadline_hours"],
                }
                deadline = datetime.now() + timedelta(hours=deadline_hours[priority])

            # Determine required specialties if not provided
            if not required_specialties:
                required_specialties = self._determine_required_specialties(assessment_result)

            # Create validation request
            request = ValidationRequest(
                request_id=request_id,
                assessment_result=assessment_result,
                priority=priority,
                required_specialties=required_specialties,
                min_experts=min_experts,
                max_experts=max_experts,
                deadline=deadline,
            )

            self.validation_requests[request_id] = request

            # Attempt to assign experts
            await self._assign_experts_to_request(request_id)

            logger.info(f"Created validation request: {request_id}")
            return request_id

        except Exception as e:
            logger.error(f"Failed to create validation request: {e}")
            raise

    def _determine_required_specialties(
        self, assessment_result: ClinicalAccuracyResult
    ) -> List[ExpertSpecialty]:
        """Determine required expert specialties based on assessment"""
        specialties = []

        # Check for crisis indicators
        if assessment_result.safety_assessment.overall_risk.value in ["high", "critical"]:
            specialties.append(ExpertSpecialty.CRISIS_INTERVENTION)

        # Check for specific diagnostic areas
        if assessment_result.dsm5_assessment.primary_diagnosis:
            diagnosis = assessment_result.dsm5_assessment.primary_diagnosis.lower()

            if "substance" in diagnosis or "addiction" in diagnosis:
                specialties.append(ExpertSpecialty.SUBSTANCE_ABUSE)
            elif "trauma" in diagnosis or "ptsd" in diagnosis:
                specialties.append(ExpertSpecialty.TRAUMA_THERAPY)
            elif "child" in diagnosis or "adolescent" in diagnosis:
                specialties.append(ExpertSpecialty.CHILD_PSYCHOLOGY)

        # Default to clinical psychology if no specific specialty identified
        if not specialties:
            specialties.append(ExpertSpecialty.CLINICAL_PSYCHOLOGY)

        return specialties

    async def _assign_experts_to_request(self, request_id: str) -> bool:
        """Assign experts to a validation request"""
        try:
            request = self.validation_requests.get(request_id)
            if not request:
                logger.error(f"Validation request {request_id} not found")
                return False

            # Find available experts
            available_experts = self._find_available_experts(request)

            if not available_experts:
                logger.warning(f"No available experts for request {request_id}")
                return False

            # Sort experts by suitability score
            scored_experts = []
            for expert in available_experts:
                score = self._calculate_expert_suitability_score(expert, request)
                scored_experts.append((expert, score))

            scored_experts.sort(key=lambda x: x[1], reverse=True)

            # Assign top experts up to max_experts
            assigned_count = 0
            for expert, score in scored_experts:
                if assigned_count >= request.max_experts:
                    break

                if self._assign_expert_to_request(expert.expert_id, request_id):
                    assigned_count += 1

            # Update request status
            if assigned_count >= request.min_experts:
                request.status = ValidationStatus.ASSIGNED
                await self._notify_experts_of_assignment(request_id)
                logger.info(f"Assigned {assigned_count} experts to request {request_id}")
                return True
            else:
                logger.warning(f"Could not assign minimum experts to request {request_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to assign experts to request {request_id}: {e}")
            return False

    def _find_available_experts(self, request: ValidationRequest) -> List[ExpertProfile]:
        """Find available experts for a validation request"""
        available_experts = []

        for expert in self.experts.values():
            if not expert.is_active:
                continue

            # Check if expert has required specialty
            if not any(
                specialty in expert.specialties for specialty in request.required_specialties
            ):
                continue

            # Check current workload
            current_assignments = self._get_expert_current_assignments(expert.expert_id)
            if len(current_assignments) >= expert.max_concurrent_validations:
                continue

            # Check availability (simplified - would be more complex in production)
            available_experts.append(expert)

        return available_experts

    def _get_expert_current_assignments(self, expert_id: str) -> List[str]:
        """Get current validation assignments for an expert"""
        assignments = []
        for request_id, request in self.validation_requests.items():
            if expert_id in request.assigned_experts and request.status in [
                ValidationStatus.ASSIGNED,
                ValidationStatus.IN_PROGRESS,
            ]:
                assignments.append(request_id)
        return assignments

    def _calculate_expert_suitability_score(
        self, expert: ExpertProfile, request: ValidationRequest
    ) -> float:
        """Calculate expert suitability score for a request"""
        score = 0.0

        # Specialty match score (0-40 points)
        specialty_matches = sum(
            1 for specialty in request.required_specialties if specialty in expert.specialties
        )
        score += (specialty_matches / len(request.required_specialties)) * 40

        # Experience score (0-20 points)
        experience_score = min(expert.years_experience / 20, 1.0) * 20
        score += experience_score

        # Accuracy rating score (0-20 points)
        score += expert.accuracy_rating * 20

        # Availability score (0-10 points)
        current_load = len(self._get_expert_current_assignments(expert.expert_id))
        availability_score = (
            max(
                0,
                (expert.max_concurrent_validations - current_load)
                / expert.max_concurrent_validations,
            )
            * 10
        )
        score += availability_score

        # Response time score (0-10 points)
        response_time_score = max(0, (48 - expert.average_response_time_hours) / 48) * 10
        score += response_time_score

        return score

    def _assign_expert_to_request(self, expert_id: str, request_id: str) -> bool:
        """Assign a specific expert to a validation request"""
        try:
            request = self.validation_requests.get(request_id)
            expert = self.experts.get(expert_id)

            if not request or not expert:
                return False

            if expert_id not in request.assigned_experts:
                request.assigned_experts.append(expert_id)
                expert.validation_count += 1
                expert.last_active = datetime.now()
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to assign expert {expert_id} to request {request_id}: {e}")
            return False

    async def _notify_experts_of_assignment(self, request_id: str) -> None:
        """Notify experts of new validation assignment"""
        try:
            request = self.validation_requests.get(request_id)
            if not request:
                return

            for expert_id in request.assigned_experts:
                expert = self.experts.get(expert_id)
                if expert:
                    await self._send_assignment_notification(expert, request)

        except Exception as e:
            logger.error(f"Failed to notify experts for request {request_id}: {e}")

    async def _send_assignment_notification(
        self, expert: ExpertProfile, request: ValidationRequest
    ) -> None:
        """Send assignment notification to expert"""
        try:
            if not self.config["notification_enabled"]:
                return

            # Create notification message
            subject = f"New Validation Request - Priority: {request.priority.value.upper()}"

            message = f"""
            Dear {expert.name},
            
            You have been assigned a new clinical accuracy validation request.
            
            Request ID: {request.request_id}
            Priority: {request.priority.value.upper()}
            Deadline: {request.deadline.strftime('%Y-%m-%d %H:%M')}
            Required Specialties: {', '.join([s.value for s in request.required_specialties])}
            
            Assessment Summary:
            - Overall Accuracy: {request.assessment_result.overall_accuracy.value}
            - Confidence Score: {request.assessment_result.confidence_score:.2f}
            - Safety Risk: {request.assessment_result.safety_assessment.overall_risk.value}
            
            Please review and provide your validation response by the deadline.
            
            Best regards,
            Pixel Clinical Validation System
            """

            # Send notification (email, in-app, etc.)
            await self._send_email_notification(expert.email, subject, message)

            # Call notification callbacks
            for callback in self.notification_callbacks:
                try:
                    await callback(expert, request, "assignment")
                except Exception as e:
                    logger.error(f"Notification callback failed: {e}")

        except Exception as e:
            logger.error(f"Failed to send assignment notification: {e}")

    async def _send_email_notification(self, to_email: str, subject: str, message: str) -> None:
        """Send email notification"""
        try:
            if not self.config["email_settings"]["username"]:
                logger.debug("Email notifications not configured")
                return

            msg = MIMEMultipart()
            msg["From"] = self.config["email_settings"]["from_address"]
            msg["To"] = to_email
            msg["Subject"] = subject

            msg.attach(MIMEText(message, "plain"))

            server = smtplib.SMTP(
                self.config["email_settings"]["smtp_server"],
                self.config["email_settings"]["smtp_port"],
            )
            server.starttls()
            server.login(
                self.config["email_settings"]["username"], self.config["email_settings"]["password"]
            )

            text = msg.as_string()
            server.sendmail(self.config["email_settings"]["from_address"], to_email, text)
            server.quit()

            logger.info(f"Email notification sent to {to_email}")

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    async def submit_validation_response(
        self,
        request_id: str,
        expert_id: str,
        decision: ValidationDecision,
        confidence_score: float,
        clinical_accuracy_score: float,
        therapeutic_appropriateness_score: float,
        safety_assessment_score: float,
        detailed_feedback: str,
        specific_concerns: Optional[List[str]] = None,
        recommendations: Optional[List[str]] = None,
        time_spent_minutes: int = 0,
    ) -> str:
        """Submit expert validation response"""
        try:
            # Validate inputs
            if not (0.0 <= confidence_score <= 1.0):
                raise ValueError("Confidence score must be between 0.0 and 1.0")

            if not (0.0 <= clinical_accuracy_score <= 1.0):
                raise ValueError("Clinical accuracy score must be between 0.0 and 1.0")

            if not (0.0 <= therapeutic_appropriateness_score <= 1.0):
                raise ValueError("Therapeutic appropriateness score must be between 0.0 and 1.0")

            if not (0.0 <= safety_assessment_score <= 1.0):
                raise ValueError("Safety assessment score must be between 0.0 and 1.0")

            # Check if request exists and expert is assigned
            request = self.validation_requests.get(request_id)
            if not request:
                raise ValueError(f"Validation request {request_id} not found")

            if expert_id not in request.assigned_experts:
                raise ValueError(f"Expert {expert_id} not assigned to request {request_id}")

            # Check if expert already submitted response
            existing_response = next(
                (r for r in request.validation_responses if r.expert_id == expert_id), None
            )
            if existing_response:
                raise ValueError(
                    f"Expert {expert_id} already submitted response for request {request_id}"
                )

            # Create validation response
            response_id = f"resp_{uuid.uuid4().hex[:8]}"
            response = ValidationResponse(
                response_id=response_id,
                request_id=request_id,
                expert_id=expert_id,
                decision=decision,
                confidence_score=confidence_score,
                clinical_accuracy_score=clinical_accuracy_score,
                therapeutic_appropriateness_score=therapeutic_appropriateness_score,
                safety_assessment_score=safety_assessment_score,
                detailed_feedback=detailed_feedback,
                specific_concerns=specific_concerns or [],
                recommendations=recommendations or [],
                time_spent_minutes=time_spent_minutes,
            )

            # Store response
            self.validation_responses[response_id] = response
            request.validation_responses.append(response)

            # Update request status
            if request.status == ValidationStatus.ASSIGNED:
                request.status = ValidationStatus.IN_PROGRESS

            # Check if we have enough responses for consensus
            if len(request.validation_responses) >= request.min_experts:
                await self._evaluate_consensus(request_id)

            logger.info(f"Validation response submitted: {response_id}")
            return response_id

        except Exception as e:
            logger.error(f"Failed to submit validation response: {e}")
            raise

    async def _evaluate_consensus(self, request_id: str) -> Optional[ConsensusResult]:
        """Evaluate consensus from expert responses"""
        try:
            request = self.validation_requests.get(request_id)
            if not request or not request.validation_responses:
                return None

            responses = request.validation_responses

            # Calculate decision consensus
            decisions = [r.decision for r in responses]
            decision_counts = {}
            for decision in decisions:
                decision_counts[decision] = decision_counts.get(decision, 0) + 1

            # Determine final decision
            most_common_decision = max(decision_counts, key=decision_counts.get)
            decision_consensus = decision_counts[most_common_decision] / len(responses)

            # Calculate aggregated scores
            aggregated_scores = {
                "confidence": sum(r.confidence_score for r in responses) / len(responses),
                "clinical_accuracy": sum(r.clinical_accuracy_score for r in responses)
                / len(responses),
                "therapeutic_appropriateness": sum(
                    r.therapeutic_appropriateness_score for r in responses
                )
                / len(responses),
                "safety_assessment": sum(r.safety_assessment_score for r in responses)
                / len(responses),
            }

            # Calculate expert agreement level
            expert_agreement = self._calculate_expert_agreement(responses)

            # Consolidate feedback
            consolidated_feedback = self._consolidate_feedback(responses)

            # Generate action items
            action_items = self._generate_action_items(responses, most_common_decision)

            # Identify dissenting opinions
            dissenting_opinions = []
            for response in responses:
                if response.decision != most_common_decision:
                    dissenting_opinions.append(
                        f"Expert {response.expert_id}: {response.decision.value} - {response.detailed_feedback[:100]}..."
                    )

            # Create consensus result
            consensus_result = ConsensusResult(
                request_id=request_id,
                final_decision=most_common_decision,
                consensus_confidence=decision_consensus,
                expert_agreement_level=expert_agreement,
                aggregated_scores=aggregated_scores,
                consolidated_feedback=consolidated_feedback,
                action_items=action_items,
                dissenting_opinions=dissenting_opinions,
            )

            # Store consensus result
            self.consensus_results[request_id] = consensus_result

            # Update request status
            request.status = ValidationStatus.COMPLETED

            # Notify stakeholders
            await self._notify_consensus_completion(request_id, consensus_result)

            logger.info(
                f"Consensus evaluated for request {request_id}: {most_common_decision.value}"
            )
            return consensus_result

        except Exception as e:
            logger.error(f"Failed to evaluate consensus for request {request_id}: {e}")
            return None

    def _calculate_expert_agreement(self, responses: List[ValidationResponse]) -> float:
        """Calculate expert agreement level"""
        if len(responses) < 2:
            return 1.0

        # Calculate agreement based on score similarity
        scores = []
        for response in responses:
            avg_score = (
                response.confidence_score
                + response.clinical_accuracy_score
                + response.therapeutic_appropriateness_score
                + response.safety_assessment_score
            ) / 4
            scores.append(avg_score)

        # Calculate standard deviation
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = variance**0.5

        # Convert to agreement level (lower std_dev = higher agreement)
        agreement_level = max(0.0, 1.0 - (std_dev * 2))
        return agreement_level

    def _consolidate_feedback(self, responses: List[ValidationResponse]) -> str:
        """Consolidate feedback from multiple experts"""
        feedback_sections = []

        # Common themes
        all_concerns = []
        all_recommendations = []

        for response in responses:
            all_concerns.extend(response.specific_concerns)
            all_recommendations.extend(response.recommendations)

        # Count frequency of concerns and recommendations
        concern_counts = {}
        for concern in all_concerns:
            concern_counts[concern] = concern_counts.get(concern, 0) + 1

        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1

        # Build consolidated feedback
        consolidated = "EXPERT CONSENSUS SUMMARY\n\n"

        if concern_counts:
            consolidated += "PRIMARY CONCERNS:\n"
            sorted_concerns = sorted(concern_counts.items(), key=lambda x: x[1], reverse=True)
            for concern, count in sorted_concerns[:5]:  # Top 5 concerns
                consolidated += (
                    f"- {concern} (mentioned by {count} expert{'s' if count > 1 else ''})\n"
                )
            consolidated += "\n"

        if recommendation_counts:
            consolidated += "KEY RECOMMENDATIONS:\n"
            sorted_recs = sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)
            for rec, count in sorted_recs[:5]:  # Top 5 recommendations
                consolidated += f"- {rec} (suggested by {count} expert{'s' if count > 1 else ''})\n"
            consolidated += "\n"

        # Add individual expert summaries
        consolidated += "INDIVIDUAL EXPERT FEEDBACK:\n"
        for i, response in enumerate(responses, 1):
            expert = self.experts.get(response.expert_id)
            expert_name = expert.name if expert else f"Expert {response.expert_id}"
            consolidated += f"\n{i}. {expert_name} ({response.decision.value}):\n"
            consolidated += f"   {response.detailed_feedback[:200]}{'...' if len(response.detailed_feedback) > 200 else ''}\n"

        return consolidated

    def _generate_action_items(
        self, responses: List[ValidationResponse], final_decision: ValidationDecision
    ) -> List[str]:
        """Generate action items based on expert responses"""
        action_items = []

        if final_decision == ValidationDecision.APPROVE:
            action_items.append("Assessment approved for use in training")
            action_items.append("Monitor for any emerging patterns in similar cases")

        elif final_decision == ValidationDecision.REJECT:
            action_items.append("Assessment rejected - do not use in training")
            action_items.append("Review and address identified issues")
            action_items.append("Consider retraining on similar scenarios")

        elif final_decision == ValidationDecision.NEEDS_REVISION:
            action_items.append("Revise assessment based on expert feedback")
            action_items.append("Resubmit for validation after improvements")

        elif final_decision == ValidationDecision.ESCALATE:
            action_items.append("Escalate to senior clinical review board")
            action_items.append("Conduct additional expert consultation")

        # Add specific action items from expert recommendations
        all_recommendations = []
        for response in responses:
            all_recommendations.extend(response.recommendations)

        # Get most common recommendations
        rec_counts = {}
        for rec in all_recommendations:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1

        sorted_recs = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)
        for rec, count in sorted_recs[:3]:  # Top 3 recommendations
            if count > 1:  # Only include if mentioned by multiple experts
                action_items.append(f"Implement: {rec}")

        return action_items

    async def _notify_consensus_completion(
        self, request_id: str, consensus_result: ConsensusResult
    ) -> None:
        """Notify stakeholders of consensus completion"""
        try:
            # Notify system administrators
            subject = f"Validation Consensus Complete - {request_id}"
            message = f"""
            Validation consensus has been reached for request {request_id}.
            
            Final Decision: {consensus_result.final_decision.value.upper()}
            Consensus Confidence: {consensus_result.consensus_confidence:.2f}
            Expert Agreement: {consensus_result.expert_agreement_level:.2f}
            
            Action Items:
            {chr(10).join(f'- {item}' for item in consensus_result.action_items)}
            
            Full details available in the validation system.
            """

            # Send notifications to configured recipients
            admin_emails = self.config.get("admin_notification_emails", [])
            for email in admin_emails:
                await self._send_email_notification(email, subject, message)

            # Call notification callbacks
            for callback in self.notification_callbacks:
                try:
                    await callback(None, None, "consensus_complete", consensus_result)
                except Exception as e:
                    logger.error(f"Consensus notification callback failed: {e}")

        except Exception as e:
            logger.error(f"Failed to notify consensus completion: {e}")

    def get_validation_request(self, request_id: str) -> Optional[ValidationRequest]:
        """Get validation request by ID"""
        return self.validation_requests.get(request_id)

    def get_validation_response(self, response_id: str) -> Optional[ValidationResponse]:
        """Get validation response by ID"""
        return self.validation_responses.get(response_id)

    def get_consensus_result(self, request_id: str) -> Optional[ConsensusResult]:
        """Get consensus result by request ID"""
        return self.consensus_results.get(request_id)

    def list_validation_requests(
        self,
        status: Optional[ValidationStatus] = None,
        priority: Optional[ValidationPriority] = None,
        expert_id: Optional[str] = None,
    ) -> List[ValidationRequest]:
        """List validation requests with optional filtering"""
        requests = list(self.validation_requests.values())

        if status:
            requests = [r for r in requests if r.status == status]

        if priority:
            requests = [r for r in requests if r.priority == priority]

        if expert_id:
            requests = [r for r in requests if expert_id in r.assigned_experts]

        return requests

    def get_validation_metrics(self) -> ValidationMetrics:
        """Get validation system metrics"""
        requests = list(self.validation_requests.values())

        metrics = ValidationMetrics()
        metrics.total_requests = len(requests)
        metrics.pending_requests = len(
            [r for r in requests if r.status == ValidationStatus.PENDING]
        )
        metrics.completed_requests = len(
            [r for r in requests if r.status == ValidationStatus.COMPLETED]
        )

        # Calculate average response time
        completed_responses = [r for r in self.validation_responses.values()]
        if completed_responses:
            total_time = 0
            for response in completed_responses:
                request = self.validation_requests.get(response.request_id)
                if request:
                    response_time = (
                        response.submitted_at - request.created_at
                    ).total_seconds() / 3600
                    total_time += response_time
            metrics.average_response_time_hours = total_time / len(completed_responses)

        # Calculate expert utilization
        for expert_id, expert in self.experts.items():
            current_assignments = len(self._get_expert_current_assignments(expert_id))
            utilization = current_assignments / expert.max_concurrent_validations
            metrics.expert_utilization[expert_id] = utilization

        # Calculate consensus and approval rates
        consensus_results = list(self.consensus_results.values())
        if consensus_results:
            high_consensus = len([c for c in consensus_results if c.consensus_confidence >= 0.8])
            metrics.consensus_rate = high_consensus / len(consensus_results)

            approvals = len(
                [c for c in consensus_results if c.final_decision == ValidationDecision.APPROVE]
            )
            metrics.approval_rate = approvals / len(consensus_results)

            escalations = len(
                [c for c in consensus_results if c.final_decision == ValidationDecision.ESCALATE]
            )
            metrics.escalation_rate = escalations / len(consensus_results)

        return metrics

    def add_notification_callback(self, callback: Callable) -> None:
        """Add notification callback function"""
        self.notification_callbacks.append(callback)

    def remove_notification_callback(self, callback: Callable) -> None:
        """Remove notification callback function"""
        if callback in self.notification_callbacks:
            self.notification_callbacks.remove(callback)

    def export_validation_data(self, output_path: Path) -> None:
        """Export validation data to file"""
        try:
            export_data = {
                "experts": {
                    expert_id: {
                        "name": expert.name,
                        "specialties": [s.value for s in expert.specialties],
                        "credentials": expert.credentials,
                        "years_experience": expert.years_experience,
                        "validation_count": expert.validation_count,
                        "accuracy_rating": expert.accuracy_rating,
                        "is_active": expert.is_active,
                    }
                    for expert_id, expert in self.experts.items()
                },
                "validation_requests": {
                    request_id: {
                        "priority": request.priority.value,
                        "status": request.status.value,
                        "required_specialties": [s.value for s in request.required_specialties],
                        "assigned_experts": request.assigned_experts,
                        "created_at": request.created_at.isoformat(),
                        "deadline": request.deadline.isoformat(),
                    }
                    for request_id, request in self.validation_requests.items()
                },
                "consensus_results": {
                    request_id: {
                        "final_decision": result.final_decision.value,
                        "consensus_confidence": result.consensus_confidence,
                        "expert_agreement_level": result.expert_agreement_level,
                        "aggregated_scores": result.aggregated_scores,
                        "action_items": result.action_items,
                        "created_at": result.created_at.isoformat(),
                    }
                    for request_id, result in self.consensus_results.items()
                },
                "metrics": self.get_validation_metrics().__dict__,
            }

            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Validation data exported to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export validation data: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":

    async def main():
        # Initialize validation interface
        interface = ExpertValidationInterface()

        # Example: Create a mock assessment result for testing
        from .clinical_accuracy_validator import (
            ClinicalAccuracyResult,
            ClinicalContext,
            DSM5Assessment,
            PDM2Assessment,
            TherapeuticAppropriatenessScore,
            SafetyAssessment,
            ClinicalAccuracyLevel,
            TherapeuticModality,
            SafetyRiskLevel,
        )

        mock_context = ClinicalContext(
            client_presentation="Client with depression and anxiety",
            therapeutic_modality=TherapeuticModality.CBT,
            session_phase="initial",
        )

        mock_assessment = ClinicalAccuracyResult(
            assessment_id="test_001",
            timestamp=datetime.now(),
            clinical_context=mock_context,
            dsm5_assessment=DSM5Assessment(primary_diagnosis="Major Depressive Disorder"),
            pdm2_assessment=PDM2Assessment(),
            therapeutic_appropriateness=TherapeuticAppropriatenessScore(overall_score=0.7),
            safety_assessment=SafetyAssessment(overall_risk=SafetyRiskLevel.MODERATE),
            overall_accuracy=ClinicalAccuracyLevel.ACCEPTABLE,
            confidence_score=0.65,
            expert_validation_needed=True,
            recommendations=["Consider additional assessment"],
            warnings=[],
        )

        # Create validation request
        request_id = await interface.create_validation_request(
            assessment_result=mock_assessment,
            priority=ValidationPriority.HIGH,
            required_specialties=[ExpertSpecialty.CLINICAL_PSYCHOLOGY],
        )

        print(f"Created validation request: {request_id}")

        # List experts
        experts = interface.list_experts(specialty=ExpertSpecialty.CLINICAL_PSYCHOLOGY)
        print(f"Available experts: {len(experts)}")

        # Get metrics
        metrics = interface.get_validation_metrics()
        print(f"Total requests: {metrics.total_requests}")
        print(f"Pending requests: {metrics.pending_requests}")

    # Run example
    asyncio.run(main())
