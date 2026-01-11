"""
Professional Evaluator Recruitment System (Tier 2.3)

Manages recruitment, onboarding, and coordination of licensed mental health 
professionals for AI therapeutic response evaluation.

Key Features:
- Professional credential verification
- Evaluation assignment and scheduling
- Compensation tracking and management
- Quality assurance and reliability monitoring
- Specialized expertise matching
- Cultural diversity and representation

Target: 10-15 licensed professionals across diverse specialties
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProfessionalCredential(Enum):
    """Licensed mental health professional credentials."""
    PHD_PSYCHOLOGY = "phd_psychology"
    PSYD_PSYCHOLOGY = "psyd_psychology" 
    MD_PSYCHIATRY = "md_psychiatry"
    LCSW = "lcsw"  # Licensed Clinical Social Worker
    LMFT = "lmft"  # Licensed Marriage and Family Therapist
    LPC = "lpc"   # Licensed Professional Counselor
    LMHC = "lmhc"  # Licensed Mental Health Counselor
    LPCC = "lpcc"  # Licensed Professional Clinical Counselor
    LCPC = "lcpc"  # Licensed Clinical Professional Counselor


class SpecialtyArea(Enum):
    """Mental health specialty areas."""
    TRAUMA_PTSD = "trauma_ptsd"
    DEPRESSION_MOOD = "depression_mood"
    ANXIETY_DISORDERS = "anxiety_disorders"
    PERSONALITY_DISORDERS = "personality_disorders"
    SUBSTANCE_USE = "substance_use"
    EATING_DISORDERS = "eating_disorders"
    CHILD_ADOLESCENT = "child_adolescent"
    GERIATRIC = "geriatric"
    COUPLES_FAMILY = "couples_family"
    GROUP_THERAPY = "group_therapy"
    CRISIS_INTERVENTION = "crisis_intervention"
    MULTICULTURAL = "multicultural"
    LGBTQ_AFFIRMING = "lgbtq_affirming"
    MILITARY_VETERANS = "military_veterans"
    NEUROPSYCHOLOGY = "neuropsychology"


@dataclass
class RecruitmentTarget:
    """Target profile for professional recruitment."""
    specialty_area: SpecialtyArea
    preferred_credentials: List[ProfessionalCredential]
    minimum_experience_years: int
    cultural_backgrounds: List[str]
    theoretical_orientations: List[str]
    target_count: int
    priority_level: str  # high, medium, low
    recruitment_status: str = "open"  # open, in_progress, filled


@dataclass
class EvaluatorApplication:
    """Application from potential professional evaluator."""
    application_id: str
    applicant_name: str
    email: str
    phone: str
    
    # Professional credentials
    credentials: List[ProfessionalCredential]
    license_numbers: Dict[str, str]  # credential -> license number
    license_states: List[str]
    
    # Experience and specialties
    years_experience: int
    specialties: List[SpecialtyArea]
    theoretical_orientations: List[str]
    practice_settings: List[str]  # private practice, hospital, clinic, etc.
    
    # Cultural and demographic information
    cultural_backgrounds: List[str]
    languages_spoken: List[str]
    age_groups_served: List[str]
    
    # Availability and compensation
    hours_available_monthly: int
    preferred_evaluation_types: List[str]
    compensation_expectations: str
    
    # Professional references
    references: List[Dict[str, str]]
    
    # Application status
    application_status: str = "submitted"  # submitted, under_review, approved, rejected
    review_notes: str = ""
    onboarding_completed: bool = False


@dataclass
class EvaluationAssignment:
    """Assignment of evaluation tasks to professionals."""
    assignment_id: str
    evaluator_id: str
    evaluation_package_id: str
    scenario_ids: List[str]
    assigned_date: str
    due_date: str
    estimated_hours: float
    compensation_amount: float
    specialty_match_score: float
    assignment_status: str = "assigned"  # assigned, in_progress, completed, overdue
    completion_date: Optional[str] = None
    quality_score: Optional[float] = None


class ProfessionalEvaluatorRecruitment:
    """Manages recruitment and coordination of professional evaluators."""
    
    def __init__(self):
        self.recruitment_targets = self._define_recruitment_targets()
        self.applications: Dict[str, EvaluatorApplication] = {}
        self.approved_evaluators: Dict[str, str] = {}  # evaluator_id -> application_id
        self.assignments: Dict[str, EvaluationAssignment] = {}
        self.recruitment_metrics = {
            "applications_received": 0,
            "applications_approved": 0,
            "evaluations_completed": 0,
            "average_quality_score": 0.0
        }
    
    def create_recruitment_campaign(self) -> Dict[str, Any]:
        """Create comprehensive recruitment campaign materials."""
        
        campaign = {
            "campaign_id": self._generate_id(),
            "title": "Licensed Mental Health Professionals Needed: AI Therapeutic Response Evaluation",
            "description": self._create_recruitment_description(),
            "eligibility_criteria": self._create_eligibility_criteria(),
            "compensation_structure": self._create_compensation_structure(),
            "time_commitment": self._create_time_commitment_details(),
            "application_process": self._create_application_process(),
            "contact_information": {
                "research_coordinator": "ai-evaluation-research@pixelated.ai",
                "phone": "1-800-PIXEL-AI",
                "website": "https://pixelated.ai/professional-evaluation"
            },
            "recruitment_channels": self._identify_recruitment_channels()
        }
        
        return campaign
    
    def process_evaluator_application(self, application_data: Dict[str, Any]) -> EvaluatorApplication:
        """Process and validate professional evaluator application."""
        
        application = EvaluatorApplication(
            application_id=self._generate_id(),
            applicant_name=application_data["name"],
            email=application_data["email"],
            phone=application_data.get("phone", ""),
            credentials=[ProfessionalCredential(c) for c in application_data["credentials"]],
            license_numbers=application_data.get("license_numbers", {}),
            license_states=application_data.get("license_states", []),
            years_experience=application_data["years_experience"],
            specialties=[SpecialtyArea(s) for s in application_data["specialties"]],
            theoretical_orientations=application_data.get("theoretical_orientations", []),
            practice_settings=application_data.get("practice_settings", []),
            cultural_backgrounds=application_data.get("cultural_backgrounds", []),
            languages_spoken=application_data.get("languages_spoken", ["English"]),
            age_groups_served=application_data.get("age_groups_served", []),
            hours_available_monthly=application_data.get("hours_available_monthly", 5),
            preferred_evaluation_types=application_data.get("preferred_evaluation_types", []),
            compensation_expectations=application_data.get("compensation_expectations", ""),
            references=application_data.get("references", [])
        )
        
        # Validate application
        validation_result = self._validate_application(application)
        application.application_status = "under_review"
        application.review_notes = validation_result["notes"]
        
        self.applications[application.application_id] = application
        self.recruitment_metrics["applications_received"] += 1
        
        logger.info(f"Processed application from {application.applicant_name}")
        return application
    
    def evaluate_application(self, application_id: str, reviewer_notes: str) -> str:
        """Evaluate and approve/reject professional application."""
        
        if application_id not in self.applications:
            raise ValueError(f"Application {application_id} not found")
        
        application = self.applications[application_id]
        
        # Scoring criteria
        score = self._calculate_application_score(application)
        
        # Decision criteria
        if score >= 75 and self._meets_recruitment_needs(application):
            application.application_status = "approved"
            evaluator_id = self._generate_evaluator_id(application)
            self.approved_evaluators[evaluator_id] = application_id
            self.recruitment_metrics["applications_approved"] += 1
            
            # Send approval and onboarding materials
            self._create_onboarding_materials(application)
            
            logger.info(f"Approved application from {application.applicant_name}")
            return "approved"
        else:
            application.application_status = "rejected"
            application.review_notes += f" | Reviewer: {reviewer_notes}"
            
            logger.info(f"Rejected application from {application.applicant_name}")
            return "rejected"
    
    def assign_evaluations(self, evaluation_package_id: str, scenario_ids: List[str]) -> List[EvaluationAssignment]:
        """Assign evaluation tasks to appropriate professionals."""
        
        assignments = []
        
        # Group scenarios by specialty requirements
        specialty_groups = self._group_scenarios_by_specialty(scenario_ids)
        
        for specialty, scenarios in specialty_groups.items():
            # Find best-matched evaluators for this specialty
            matched_evaluators = self._find_evaluators_by_specialty(specialty)
            
            for evaluator_id in matched_evaluators[:2]:  # Assign to top 2 matches
                assignment = EvaluationAssignment(
                    assignment_id=self._generate_id(),
                    evaluator_id=evaluator_id,
                    evaluation_package_id=evaluation_package_id,
                    scenario_ids=scenarios,
                    assigned_date=self._get_timestamp(),
                    due_date=self._calculate_due_date(len(scenarios)),
                    estimated_hours=len(scenarios) * 0.25,  # 15 minutes per scenario
                    compensation_amount=len(scenarios) * 25.0,  # $25 per scenario
                    specialty_match_score=self._calculate_specialty_match(evaluator_id, specialty)
                )
                
                assignments.append(assignment)
                self.assignments[assignment.assignment_id] = assignment
        
        logger.info(f"Created {len(assignments)} evaluation assignments")
        return assignments
    
    def track_evaluation_progress(self) -> Dict[str, Any]:
        """Track progress of ongoing evaluations."""
        
        progress = {
            "total_assignments": len(self.assignments),
            "completed": len([a for a in self.assignments.values() if a.assignment_status == "completed"]),
            "in_progress": len([a for a in self.assignments.values() if a.assignment_status == "in_progress"]),
            "overdue": len([a for a in self.assignments.values() if a.assignment_status == "overdue"]),
            "completion_rate": 0.0,
            "average_quality": 0.0,
            "evaluator_performance": {}
        }
        
        if progress["total_assignments"] > 0:
            progress["completion_rate"] = progress["completed"] / progress["total_assignments"]
        
        # Calculate average quality score
        completed_assignments = [a for a in self.assignments.values() if a.quality_score is not None]
        if completed_assignments:
            progress["average_quality"] = sum(a.quality_score for a in completed_assignments) / len(completed_assignments)
        
        # Evaluator performance tracking
        for evaluator_id in self.approved_evaluators.keys():
            evaluator_assignments = [a for a in self.assignments.values() if a.evaluator_id == evaluator_id]
            if evaluator_assignments:
                completed = [a for a in evaluator_assignments if a.assignment_status == "completed"]
                progress["evaluator_performance"][evaluator_id] = {
                    "total_assigned": len(evaluator_assignments),
                    "completed": len(completed),
                    "completion_rate": len(completed) / len(evaluator_assignments),
                    "average_quality": sum(a.quality_score for a in completed if a.quality_score) / len(completed) if completed else 0.0
                }
        
        return progress
    
    def generate_recruitment_report(self) -> str:
        """Generate comprehensive recruitment status report."""
        
        progress = self.track_evaluation_progress()
        
        report = f"""
# Professional Evaluator Recruitment Report

## Recruitment Metrics
- **Applications Received**: {self.recruitment_metrics['applications_received']}
- **Applications Approved**: {self.recruitment_metrics['applications_approved']}
- **Approval Rate**: {self.recruitment_metrics['applications_approved']/max(1, self.recruitment_metrics['applications_received']):.1%}
- **Active Evaluators**: {len(self.approved_evaluators)}

## Evaluation Progress
- **Total Assignments**: {progress['total_assignments']}
- **Completion Rate**: {progress['completion_rate']:.1%}
- **Average Quality Score**: {progress['average_quality']:.1f}/10
- **Overdue Assignments**: {progress['overdue']}

## Specialty Coverage
"""
        
        # Add specialty coverage analysis
        specialty_coverage = self._analyze_specialty_coverage()
        for specialty, coverage in specialty_coverage.items():
            report += f"- **{specialty.value.replace('_', ' ').title()}**: {coverage['evaluators']} evaluators\n"
        
        report += """

## Recruitment Needs
"""
        
        # Add recruitment gaps
        gaps = self._identify_recruitment_gaps()
        for gap in gaps:
            report += f"- **{gap['specialty']}**: Need {gap['additional_evaluators']} more evaluators\n"
        
        return report
    
    # Helper methods
    def _define_recruitment_targets(self) -> List[RecruitmentTarget]:
        """Define recruitment targets for professional evaluators."""
        return [
            RecruitmentTarget(
                specialty_area=SpecialtyArea.TRAUMA_PTSD,
                preferred_credentials=[ProfessionalCredential.PHD_PSYCHOLOGY, ProfessionalCredential.LCSW],
                minimum_experience_years=5,
                cultural_backgrounds=["Diverse"],
                theoretical_orientations=["Trauma-Informed Care", "EMDR", "CBT"],
                target_count=3,
                priority_level="high"
            ),
            RecruitmentTarget(
                specialty_area=SpecialtyArea.DEPRESSION_MOOD,
                preferred_credentials=[ProfessionalCredential.PHD_PSYCHOLOGY, ProfessionalCredential.LPC],
                minimum_experience_years=3,
                cultural_backgrounds=["Diverse"],
                theoretical_orientations=["CBT", "Interpersonal Therapy", "Behavioral Activation"],
                target_count=2,
                priority_level="high"
            ),
            RecruitmentTarget(
                specialty_area=SpecialtyArea.CRISIS_INTERVENTION,
                preferred_credentials=[ProfessionalCredential.PHD_PSYCHOLOGY, ProfessionalCredential.LCSW, ProfessionalCredential.MD_PSYCHIATRY],
                minimum_experience_years=7,
                cultural_backgrounds=["Diverse"],
                theoretical_orientations=["Crisis Intervention", "Safety Planning"],
                target_count=2,
                priority_level="high"
            ),
            # Additional targets would be defined here
        ]
    
    def _create_recruitment_description(self) -> str:
        """Create recruitment campaign description."""
        return """
Join a groundbreaking research initiative to evaluate AI-generated therapeutic responses. 

We are seeking licensed mental health professionals to participate in a comprehensive evaluation study of an innovative therapeutic AI system. This system integrates insights from leading experts like Tim Fletcher, Dr. Ramani, and Dr. Gabor Maté to provide empathetic, clinically-informed therapeutic support.

Your expertise will help validate the clinical appropriateness, safety, and effectiveness of AI-generated therapeutic responses across diverse mental health scenarios.

This is an opportunity to be at the forefront of ethical AI development in mental healthcare.
        """
    
    def _create_eligibility_criteria(self) -> List[str]:
        """Create eligibility criteria for evaluators."""
        return [
            "Current license to practice mental health services in the United States",
            "Minimum 3 years of direct clinical experience",
            "Experience with at least one evidence-based therapeutic modality",
            "Willingness to participate in blind evaluation protocols",
            "Commitment to maintaining confidentiality and research ethics",
            "Access to secure internet connection for online evaluation platform"
        ]
    
    def _create_compensation_structure(self) -> Dict[str, str]:
        """Create compensation structure."""
        return {
            "base_rate": "$25 per evaluation scenario (15-20 minutes each)",
            "bonus_rate": "$50 bonus for completing evaluation package within deadline",
            "training_compensation": "$100 for completion of evaluator training session",
            "payment_schedule": "Monthly via direct deposit or check",
            "total_monthly_potential": "$400-800 based on availability and participation"
        }
    
    def _create_time_commitment_details(self) -> Dict[str, str]:
        """Create time commitment details."""
        return {
            "training_session": "2 hours (one-time, online)",
            "monthly_commitment": "3-8 hours based on availability",
            "evaluation_sessions": "15-20 minutes per scenario",
            "flexible_scheduling": "Complete evaluations within 2-week windows",
            "minimum_commitment": "6 months participation preferred"
        }
    
    def _create_application_process(self) -> List[str]:
        """Create application process steps."""
        return [
            "Complete online application with professional credentials",
            "Provide license verification and professional references",
            "Participate in brief video interview (30 minutes)",
            "Complete online training and calibration session",
            "Sign research participation agreement and confidentiality terms",
            "Begin evaluation assignments"
        ]
    
    def _identify_recruitment_channels(self) -> List[str]:
        """Identify recruitment channels."""
        return [
            "American Psychological Association (APA) member communications",
            "National Association of Social Workers (NASW) networks",
            "American Mental Health Counselors Association (AMHCA)",
            "State licensing board communications",
            "Professional training institution partnerships",
            "Mental health conferences and professional development events",
            "Specialized trauma and crisis intervention organizations",
            "Multicultural mental health professional networks"
        ]
    
    def _validate_application(self, application: EvaluatorApplication) -> Dict[str, Any]:
        """Validate professional application."""
        notes = []
        score = 0
        
        # Check minimum experience
        if application.years_experience >= 3:
            score += 20
        else:
            notes.append("Below minimum experience requirement")
        
        # Check credentials
        valid_credentials = [ProfessionalCredential.PHD_PSYCHOLOGY, ProfessionalCredential.LCSW, 
                           ProfessionalCredential.LPC, ProfessionalCredential.LMFT]
        if any(cred in valid_credentials for cred in application.credentials):
            score += 30
        else:
            notes.append("No qualifying credentials found")
        
        # Check specialty alignment
        high_priority_specialties = [SpecialtyArea.TRAUMA_PTSD, SpecialtyArea.CRISIS_INTERVENTION, 
                                   SpecialtyArea.DEPRESSION_MOOD]
        if any(spec in high_priority_specialties for spec in application.specialties):
            score += 25
        
        return {"score": score, "notes": "; ".join(notes)}
    
    def _calculate_application_score(self, application: EvaluatorApplication) -> int:
        """Calculate overall application score."""
        return self._validate_application(application)["score"]
    
    def _meets_recruitment_needs(self, application: EvaluatorApplication) -> bool:
        """Check if application meets current recruitment needs."""
        # Check if we need more evaluators in applicant's specialties
        for specialty in application.specialties:
            target = next((t for t in self.recruitment_targets if t.specialty_area == specialty), None)
            if target and target.recruitment_status == "open":
                current_count = len([aid for aid, app_id in self.approved_evaluators.items() 
                                   if specialty in self.applications[app_id].specialties])
                if current_count < target.target_count:
                    return True
        return False
    
    def _generate_id(self) -> str:
        """Generate unique ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _generate_evaluator_id(self, application: EvaluatorApplication) -> str:
        """Generate evaluator ID from application."""
        return f"eval_{application.application_id[:8]}"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _calculate_due_date(self, num_scenarios: int) -> str:
        """Calculate due date based on number of scenarios."""
        from datetime import datetime, timedelta
        due_date = datetime.now() + timedelta(days=14)  # 2 weeks
        return due_date.isoformat()
    
    def _group_scenarios_by_specialty(self, scenario_ids: List[str]) -> Dict[SpecialtyArea, List[str]]:
        """Group scenarios by required specialty."""
        # Simplified - would analyze scenario content
        return {SpecialtyArea.TRAUMA_PTSD: scenario_ids}
    
    def _find_evaluators_by_specialty(self, specialty: SpecialtyArea) -> List[str]:
        """Find evaluators matching specialty."""
        matched = []
        for evaluator_id, app_id in self.approved_evaluators.items():
            if specialty in self.applications[app_id].specialties:
                matched.append(evaluator_id)
        return matched
    
    def _calculate_specialty_match(self, evaluator_id: str, specialty: SpecialtyArea) -> float:
        """Calculate how well evaluator matches specialty."""
        return 0.9  # Simplified
    
    def _create_onboarding_materials(self, application: EvaluatorApplication) -> Dict[str, Any]:
        """Create onboarding materials for approved evaluator."""
        return {"training_materials": "comprehensive_evaluator_training.pdf"}
    
    def _analyze_specialty_coverage(self) -> Dict[SpecialtyArea, Dict[str, int]]:
        """Analyze specialty coverage."""
        coverage = {}
        for specialty in SpecialtyArea:
            evaluators = self._find_evaluators_by_specialty(specialty)
            coverage[specialty] = {"evaluators": len(evaluators)}
        return coverage
    
    def _identify_recruitment_gaps(self) -> List[Dict[str, Any]]:
        """Identify recruitment gaps."""
        gaps = []
        for target in self.recruitment_targets:
            current_count = len(self._find_evaluators_by_specialty(target.specialty_area))
            if current_count < target.target_count:
                gaps.append({
                    "specialty": target.specialty_area.value,
                    "current_evaluators": current_count,
                    "target_evaluators": target.target_count,
                    "additional_evaluators": target.target_count - current_count
                })
        return gaps


if __name__ == "__main__":
    # Example usage
    recruitment = ProfessionalEvaluatorRecruitment()
    
    # Create recruitment campaign
    campaign = recruitment.create_recruitment_campaign()
    print(f"Created recruitment campaign: {campaign['title']}")
    print(f"Compensation: {campaign['compensation_structure']['base_rate']}")
    
    # Process sample application
    sample_application = {
        "name": "Dr. Jane Smith",
        "email": "jsmith@example.com",
        "credentials": ["phd_psychology"],
        "years_experience": 8,
        "specialties": ["trauma_ptsd", "anxiety_disorders"],
        "theoretical_orientations": ["CBT", "Trauma-Informed Care"],
        "hours_available_monthly": 6
    }
    
    application = recruitment.process_evaluator_application(sample_application)
    print(f"Processed application from {application.applicant_name}")
    
    # Evaluate application
    decision = recruitment.evaluate_application(application.application_id, "Strong candidate")
    print(f"Application decision: {decision}")
    
    print("🏥 Professional evaluator recruitment system ready!")