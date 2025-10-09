"""
Clinical Accuracy Assessment Framework

This module provides a comprehensive framework for evaluating clinical accuracy
against expert standards, including DSM-5/PDM-2 compliance, therapeutic
appropriateness, and safety validation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalAccuracyLevel(Enum):
    """Clinical accuracy assessment levels"""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    CONCERNING = "concerning"
    DANGEROUS = "dangerous"


class TherapeuticModality(Enum):
    """Supported therapeutic modalities"""

    CBT = "cognitive_behavioral_therapy"
    DBT = "dialectical_behavior_therapy"
    PSYCHODYNAMIC = "psychodynamic"
    HUMANISTIC = "humanistic"
    SYSTEMIC = "systemic"
    INTEGRATIVE = "integrative"


class SafetyRiskLevel(Enum):
    """Safety risk assessment levels"""

    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ClinicalContext:
    """Clinical context for assessment"""

    client_presentation: str
    therapeutic_modality: TherapeuticModality
    session_phase: str  # initial, working, termination
    crisis_indicators: List[str] = field(default_factory=list)
    cultural_factors: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)


@dataclass
class DSM5Assessment:
    """DSM-5 diagnostic assessment"""

    primary_diagnosis: Optional[str] = None
    secondary_diagnoses: List[str] = field(default_factory=list)
    diagnostic_confidence: float = 0.0
    criteria_met: List[str] = field(default_factory=list)
    criteria_not_met: List[str] = field(default_factory=list)
    differential_diagnoses: List[str] = field(default_factory=list)
    severity_specifiers: List[str] = field(default_factory=list)


@dataclass
class PDM2Assessment:
    """PDM-2 psychodynamic assessment"""

    personality_patterns: List[str] = field(default_factory=list)
    mental_functioning: Dict[str, float] = field(default_factory=dict)
    symptom_patterns: List[str] = field(default_factory=list)
    subjective_experience: str = ""
    relational_patterns: List[str] = field(default_factory=list)


@dataclass
class TherapeuticAppropriatenessScore:
    """Therapeutic appropriateness assessment"""

    intervention_appropriateness: float = 0.0
    timing_appropriateness: float = 0.0
    cultural_sensitivity: float = 0.0
    ethical_compliance: float = 0.0
    boundary_maintenance: float = 0.0
    overall_score: Union[float, np.floating] = 0.0
    rationale: str = ""


@dataclass
class SafetyAssessment:
    """Safety and risk assessment"""

    suicide_risk: SafetyRiskLevel = SafetyRiskLevel.MINIMAL
    self_harm_risk: SafetyRiskLevel = SafetyRiskLevel.MINIMAL
    violence_risk: SafetyRiskLevel = SafetyRiskLevel.MINIMAL
    substance_abuse_risk: SafetyRiskLevel = SafetyRiskLevel.MINIMAL
    psychosis_risk: SafetyRiskLevel = SafetyRiskLevel.MINIMAL
    overall_risk: SafetyRiskLevel = SafetyRiskLevel.MINIMAL
    immediate_interventions: List[str] = field(default_factory=list)
    safety_plan_needed: bool = False


@dataclass
class ClinicalAccuracyResult:
    """Complete clinical accuracy assessment result"""

    assessment_id: str
    timestamp: datetime
    clinical_context: ClinicalContext
    dsm5_assessment: DSM5Assessment
    pdm2_assessment: PDM2Assessment
    therapeutic_appropriateness: TherapeuticAppropriatenessScore
    safety_assessment: SafetyAssessment
    overall_accuracy: ClinicalAccuracyLevel
    confidence_score: float
    expert_validation_needed: bool
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClinicalAccuracyValidator:
    """
    Comprehensive clinical accuracy assessment framework

    This class provides methods for evaluating clinical accuracy across
    multiple dimensions including diagnostic accuracy, therapeutic
    appropriateness, and safety compliance.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the clinical accuracy validator"""
        self.config = self._load_config(config_path)
        self.dsm5_criteria = self._load_dsm5_criteria()
        self.pdm2_frameworks = self._load_pdm2_frameworks()
        self.safety_protocols = self._load_safety_protocols()
        self.therapeutic_guidelines = self._load_therapeutic_guidelines()

        # Assessment thresholds
        self.accuracy_thresholds = {
            ClinicalAccuracyLevel.EXCELLENT: 0.9,
            ClinicalAccuracyLevel.GOOD: 0.8,
            ClinicalAccuracyLevel.ACCEPTABLE: 0.7,
            ClinicalAccuracyLevel.CONCERNING: 0.5,
            ClinicalAccuracyLevel.DANGEROUS: 0.0,
        }

        logger.info("Clinical accuracy validator initialized")

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration settings"""
        default_config = {
            "dsm5_weight": 0.3,
            "pdm2_weight": 0.2,
            "therapeutic_weight": 0.3,
            "safety_weight": 0.2,
            "expert_validation_threshold": 0.7,
            "safety_escalation_threshold": SafetyRiskLevel.MODERATE,
        }

        if config_path and config_path.exists():
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                default_config |= user_config
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        return default_config

    def _load_dsm5_criteria(self) -> Dict[str, Any]:
        """Load DSM-5 diagnostic criteria"""
        # In production, this would load from a comprehensive DSM-5 database
        return {
            "major_depressive_disorder": {
                "criteria": [
                    "depressed_mood",
                    "anhedonia",
                    "weight_changes",
                    "sleep_disturbance",
                    "psychomotor_changes",
                    "fatigue",
                    "worthlessness_guilt",
                    "concentration_problems",
                    "suicidal_ideation",
                ],
                "minimum_criteria": 5,
                "duration_requirement": "2_weeks",
                "functional_impairment": True,
            },
            "generalized_anxiety_disorder": {
                "criteria": [
                    "excessive_worry",
                    "difficulty_controlling_worry",
                    "restlessness",
                    "fatigue",
                    "concentration_problems",
                    "irritability",
                    "muscle_tension",
                    "sleep_disturbance",
                ],
                "minimum_criteria": 3,
                "duration_requirement": "6_months",
                "functional_impairment": True,
            },
        }

    def _load_pdm2_frameworks(self) -> Dict[str, Any]:
        """Load PDM-2 psychodynamic frameworks"""
        return {
            "personality_patterns": [
                "schizoid",
                "paranoid",
                "psychopathic",
                "narcissistic",
                "sadistic",
                "masochistic",
                "depressive",
                "manic",
                "obsessive_compulsive",
                "hysterical",
                "phobic",
                "counterphobic",
                "dissociative",
            ],
            "mental_functioning_domains": [
                "cognitive_processes",
                "affective_processes",
                "somatic_processes",
                "relational_processes",
                "self_processes",
            ],
        }

    def _load_safety_protocols(self) -> Dict[str, Any]:
        """Load safety assessment protocols"""
        return {
            "suicide_risk_factors": [
                "previous_attempts",
                "suicidal_ideation",
                "plan_specificity",
                "means_availability",
                "hopelessness",
                "social_isolation",
                "substance_abuse",
                "mental_illness",
                "recent_losses",
            ],
            "crisis_interventions": [
                "safety_planning",
                "crisis_hotline",
                "emergency_services",
                "hospitalization",
                "medication_review",
                "support_system_activation",
            ],
        }

    def _load_therapeutic_guidelines(self) -> Dict[str, Any]:
        """Load therapeutic appropriateness guidelines"""
        return {
            "intervention_timing": {
                "initial_phase": ["rapport_building", "assessment", "psychoeducation"],
                "working_phase": ["skill_building", "processing", "insight_development"],
                "termination_phase": ["consolidation", "relapse_prevention", "closure"],
            },
            "cultural_considerations": [
                "language_preferences",
                "religious_beliefs",
                "family_dynamics",
                "socioeconomic_factors",
                "trauma_history",
                "identity_factors",
            ],
        }

    async def assess_clinical_accuracy(
        self,
        response: str,
        clinical_context: ClinicalContext,
        expert_reference: Optional[str] = None,
    ) -> ClinicalAccuracyResult:
        """
        Perform comprehensive clinical accuracy assessment

        Args:
            response: The clinical response to assess
            clinical_context: Clinical context for the assessment
            expert_reference: Optional expert reference for comparison

        Returns:
            Complete clinical accuracy assessment result
        """
        assessment_id = f"ca_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Perform individual assessments
        dsm5_assessment = await self._assess_dsm5_compliance(response, clinical_context)
        pdm2_assessment = await self._assess_pdm2_compliance(response, clinical_context)
        therapeutic_score = await self._assess_therapeutic_appropriateness(
            response, clinical_context
        )
        safety_assessment = await self._assess_safety_compliance(response, clinical_context)

        # Calculate overall accuracy
        overall_accuracy, confidence_score = self._calculate_overall_accuracy(
            dsm5_assessment, pdm2_assessment, therapeutic_score, safety_assessment
        )

        # Determine if expert validation is needed
        expert_validation_needed = confidence_score < self.config[
            "expert_validation_threshold"
        ] or safety_assessment.overall_risk.value in ["high", "critical"]

        # Generate recommendations and warnings
        recommendations = self._generate_recommendations(
            dsm5_assessment, pdm2_assessment, therapeutic_score, safety_assessment
        )
        warnings = self._generate_warnings(safety_assessment, overall_accuracy)

        return ClinicalAccuracyResult(
            assessment_id=assessment_id,
            timestamp=datetime.now(),
            clinical_context=clinical_context,
            dsm5_assessment=dsm5_assessment,
            pdm2_assessment=pdm2_assessment,
            therapeutic_appropriateness=therapeutic_score,
            safety_assessment=safety_assessment,
            overall_accuracy=overall_accuracy,
            confidence_score=confidence_score,
            expert_validation_needed=expert_validation_needed,
            recommendations=recommendations,
            warnings=warnings,
            metadata={
                "expert_reference_provided": expert_reference is not None,
                "assessment_version": "1.0",
            },
        )

    async def _assess_dsm5_compliance(
        self, response: str, context: ClinicalContext
    ) -> DSM5Assessment:
        """Assess DSM-5 diagnostic compliance"""
        # This would use NLP and clinical reasoning to assess diagnostic accuracy
        # For now, implementing a structured assessment framework

        assessment = DSM5Assessment()

        # Analyze response for diagnostic indicators
        response_lower = response.lower()

        # Check for major depressive disorder indicators
        mdd_indicators = [
            "depressed",
            "depression",
            "sad",
            "hopeless",
            "worthless",
            "sleep",
            "appetite",
            "energy",
            "concentration",
            "guilt",
        ]
        mdd_score = len([ind for ind in mdd_indicators if ind in response_lower])

        if mdd_score >= 3:
            assessment.primary_diagnosis = "Major Depressive Disorder"
            assessment.diagnostic_confidence = min(0.9, mdd_score / len(mdd_indicators))
            assessment.criteria_met = [ind for ind in mdd_indicators if ind in response_lower]

        # Check for anxiety disorder indicators
        anxiety_indicators = [
            "anxiety",
            "anxious",
            "worry",
            "nervous",
            "panic",
            "fear",
            "restless",
            "tense",
            "irritable",
        ]
        anxiety_score = len([ind for ind in anxiety_indicators if ind in response_lower])

        if anxiety_score >= 2:
            if assessment.primary_diagnosis:
                assessment.secondary_diagnoses.append("Generalized Anxiety Disorder")
            else:
                assessment.primary_diagnosis = "Generalized Anxiety Disorder"
                assessment.diagnostic_confidence = min(0.9, anxiety_score / len(anxiety_indicators))

        return assessment

    async def _assess_pdm2_compliance(
        self, response: str, context: ClinicalContext
    ) -> PDM2Assessment:
        """Assess PDM-2 psychodynamic compliance"""
        assessment = PDM2Assessment()

        # Analyze for personality patterns
        response_lower = response.lower()

        # Check for narcissistic patterns
        if any(
            word in response_lower for word in ["grandiose", "special", "entitled", "admiration"]
        ):
            assessment.personality_patterns.append("narcissistic")

        # Check for depressive patterns
        if any(
            word in response_lower
            for word in ["worthless", "guilty", "self-critical", "inadequate"]
        ):
            assessment.personality_patterns.append("depressive")

        # Assess mental functioning domains
        assessment.mental_functioning = {
            "cognitive_processes": 0.7,  # Would be calculated based on response analysis
            "affective_processes": 0.8,
            "somatic_processes": 0.6,
            "relational_processes": 0.7,
            "self_processes": 0.5,
        }

        return assessment

    async def _assess_therapeutic_appropriateness(
        self, response: str, context: ClinicalContext
    ) -> TherapeuticAppropriatenessScore:
        """Assess therapeutic appropriateness"""
        score = TherapeuticAppropriatenessScore()

        # Assess intervention appropriateness
        appropriate_interventions = self.therapeutic_guidelines["intervention_timing"].get(
            context.session_phase, []
        )

        response_lower = response.lower()
        intervention_matches = len(
            [
                intervention
                for intervention in appropriate_interventions
                if any(word in response_lower for word in intervention.split("_"))
            ]
        )

        score.intervention_appropriateness = min(
            1.0, intervention_matches / max(1, len(appropriate_interventions))
        )

        # Assess timing appropriateness (simplified)
        score.timing_appropriateness = 0.8  # Would be more sophisticated in production

        # Assess cultural sensitivity
        cultural_indicators = ["culture", "background", "beliefs", "values", "family"]
        cultural_score = len([ind for ind in cultural_indicators if ind in response_lower])
        score.cultural_sensitivity = min(1.0, cultural_score / 3)

        # Assess ethical compliance
        ethical_indicators = ["confidentiality", "boundaries", "consent", "safety"]
        ethical_score = len([ind for ind in ethical_indicators if ind in response_lower])
        score.ethical_compliance = min(1.0, ethical_score / 2)

        # Assess boundary maintenance
        boundary_violations = ["personal", "friendship", "relationship", "outside"]
        violation_score = len([viol for viol in boundary_violations if viol in response_lower])
        score.boundary_maintenance = max(0.0, 1.0 - (violation_score / 2))

        # Calculate overall score
        score.overall_score = np.mean(
            [
                score.intervention_appropriateness,
                score.timing_appropriateness,
                score.cultural_sensitivity,
                score.ethical_compliance,
                score.boundary_maintenance,
            ]
        )

        score.rationale = f"Based on intervention appropriateness ({score.intervention_appropriateness:.2f}), timing ({score.timing_appropriateness:.2f}), cultural sensitivity ({score.cultural_sensitivity:.2f}), ethical compliance ({score.ethical_compliance:.2f}), and boundary maintenance ({score.boundary_maintenance:.2f})"

        return score

    async def _assess_safety_compliance(
        self, response: str, context: ClinicalContext
    ) -> SafetyAssessment:
        """Assess safety and risk compliance"""
        assessment = SafetyAssessment()

        response_lower = response.lower()

        # Assess suicide risk
        suicide_indicators = ["suicide", "kill", "die", "death", "end it", "hopeless"]
        suicide_score = len([ind for ind in suicide_indicators if ind in response_lower])

        if suicide_score >= 3:
            assessment.suicide_risk = SafetyRiskLevel.HIGH
        elif suicide_score >= 2:
            assessment.suicide_risk = SafetyRiskLevel.MODERATE
        elif suicide_score >= 1:
            assessment.suicide_risk = SafetyRiskLevel.LOW

        # Assess self-harm risk
        self_harm_indicators = ["cut", "hurt", "harm", "pain", "punish"]
        self_harm_score = len([ind for ind in self_harm_indicators if ind in response_lower])

        if self_harm_score >= 2:
            assessment.self_harm_risk = SafetyRiskLevel.MODERATE
        elif self_harm_score >= 1:
            assessment.self_harm_risk = SafetyRiskLevel.LOW

        # Assess violence risk
        violence_indicators = ["angry", "rage", "violent", "hurt others", "attack"]
        violence_score = len([ind for ind in violence_indicators if ind in response_lower])

        if violence_score >= 2:
            assessment.violence_risk = SafetyRiskLevel.MODERATE
        elif violence_score >= 1:
            assessment.violence_risk = SafetyRiskLevel.LOW

        # Determine overall risk
        risk_levels = [
            assessment.suicide_risk,
            assessment.self_harm_risk,
            assessment.violence_risk,
            assessment.substance_abuse_risk,
            assessment.psychosis_risk,
        ]

        max_risk = max(
            risk_levels,
            key=lambda x: ["minimal", "low", "moderate", "high", "critical"].index(x.value),
        )
        assessment.overall_risk = max_risk

        # Generate immediate interventions if needed
        if assessment.suicide_risk.value in ["high", "critical"]:
            assessment.immediate_interventions.extend(
                [
                    "Immediate safety assessment",
                    "Crisis intervention protocol",
                    "Emergency contact activation",
                ]
            )
            assessment.safety_plan_needed = True

        return assessment

    def _calculate_overall_accuracy(
        self,
        dsm5: DSM5Assessment,
        pdm2: PDM2Assessment,
        therapeutic: TherapeuticAppropriatenessScore,
        safety: SafetyAssessment,
    ) -> Tuple[ClinicalAccuracyLevel, float]:
        """Calculate overall clinical accuracy"""

        # Calculate component scores
        dsm5_score = dsm5.diagnostic_confidence
        pdm2_score = (
            np.mean(list(pdm2.mental_functioning.values())) if pdm2.mental_functioning else 0.5
        )
        therapeutic_score = therapeutic.overall_score

        # Safety score (inverse of risk)
        safety_risk_values = {
            "minimal": 1.0,
            "low": 0.8,
            "moderate": 0.6,
            "high": 0.3,
            "critical": 0.0,
        }
        safety_score = safety_risk_values.get(safety.overall_risk.value, 0.5)

        # Weighted overall score
        overall_score = (
            dsm5_score * self.config["dsm5_weight"]
            + pdm2_score * self.config["pdm2_weight"]
            + therapeutic_score * self.config["therapeutic_weight"]
            + safety_score * self.config["safety_weight"]
        )

        # Determine accuracy level
        if overall_score >= self.accuracy_thresholds[ClinicalAccuracyLevel.EXCELLENT]:
            accuracy_level = ClinicalAccuracyLevel.EXCELLENT
        elif overall_score >= self.accuracy_thresholds[ClinicalAccuracyLevel.GOOD]:
            accuracy_level = ClinicalAccuracyLevel.GOOD
        elif overall_score >= self.accuracy_thresholds[ClinicalAccuracyLevel.ACCEPTABLE]:
            accuracy_level = ClinicalAccuracyLevel.ACCEPTABLE
        elif overall_score >= self.accuracy_thresholds[ClinicalAccuracyLevel.CONCERNING]:
            accuracy_level = ClinicalAccuracyLevel.CONCERNING
        else:
            accuracy_level = ClinicalAccuracyLevel.DANGEROUS

        return accuracy_level, overall_score

    def _generate_recommendations(
        self,
        dsm5: DSM5Assessment,
        pdm2: PDM2Assessment,
        therapeutic: TherapeuticAppropriatenessScore,
        safety: SafetyAssessment,
    ) -> List[str]:
        """Generate clinical recommendations"""
        recommendations = []

        # DSM-5 recommendations
        if dsm5.diagnostic_confidence < 0.7:
            recommendations.append("Consider additional diagnostic assessment")

        # Therapeutic recommendations
        if therapeutic.overall_score < 0.7:
            recommendations.append("Review therapeutic approach for appropriateness")

        if therapeutic.cultural_sensitivity < 0.6:
            recommendations.append("Increase cultural sensitivity in interventions")

        # Safety recommendations
        if safety.overall_risk.value in ["moderate", "high", "critical"]:
            recommendations.append("Implement enhanced safety monitoring")

        if safety.safety_plan_needed:
            recommendations.append("Develop comprehensive safety plan")

        return recommendations

    def _generate_warnings(
        self, safety: SafetyAssessment, accuracy: ClinicalAccuracyLevel
    ) -> List[str]:
        """Generate clinical warnings"""
        warnings = []

        if safety.overall_risk.value in ["high", "critical"]:
            warnings.append("HIGH RISK: Immediate safety intervention required")

        if accuracy == ClinicalAccuracyLevel.DANGEROUS:
            warnings.append("DANGEROUS: Clinical response poses significant risk")

        if accuracy == ClinicalAccuracyLevel.CONCERNING:
            warnings.append("CONCERNING: Clinical response requires expert review")

        return warnings

    def export_assessment(self, result: ClinicalAccuracyResult, output_path: Path) -> None:
        """Export assessment result to file"""
        try:
            # Convert dataclass to dictionary for JSON serialization
            result_dict = {
                "assessment_id": result.assessment_id,
                "timestamp": result.timestamp.isoformat(),
                "overall_accuracy": result.overall_accuracy.value,
                "confidence_score": result.confidence_score,
                "expert_validation_needed": result.expert_validation_needed,
                "recommendations": result.recommendations,
                "warnings": result.warnings,
                "dsm5_assessment": {
                    "primary_diagnosis": result.dsm5_assessment.primary_diagnosis,
                    "diagnostic_confidence": result.dsm5_assessment.diagnostic_confidence,
                    "criteria_met": result.dsm5_assessment.criteria_met,
                },
                "therapeutic_appropriateness": {
                    "overall_score": result.therapeutic_appropriateness.overall_score,
                    "rationale": result.therapeutic_appropriateness.rationale,
                },
                "safety_assessment": {
                    "overall_risk": result.safety_assessment.overall_risk.value,
                    "immediate_interventions": result.safety_assessment.immediate_interventions,
                    "safety_plan_needed": result.safety_assessment.safety_plan_needed,
                },
            }

            with open(output_path, "w") as f:
                json.dump(result_dict, f, indent=2)

            logger.info(f"Assessment exported to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export assessment: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":

    async def main():
        # Initialize validator
        validator = ClinicalAccuracyValidator()

        # Example clinical context
        context = ClinicalContext(
            client_presentation="Client reports feeling depressed and anxious for the past month",
            therapeutic_modality=TherapeuticModality.CBT,
            session_phase="initial",
            crisis_indicators=["sleep_disturbance", "appetite_changes"],
            cultural_factors=["hispanic_background"],
            contraindications=[],
        )

        # Example response to assess
        response = """
        Based on your presentation, it sounds like you may be experiencing symptoms 
        consistent with depression and anxiety. Let's explore your sleep patterns, 
        appetite changes, and mood fluctuations. I want to ensure your safety first - 
        are you having any thoughts of hurting yourself? We'll work together using 
        cognitive-behavioral techniques to help you develop coping strategies.
        """

        # Perform assessment
        result = await validator.assess_clinical_accuracy(response, context)

        # Print results
        print(f"Assessment ID: {result.assessment_id}")
        print(f"Overall Accuracy: {result.overall_accuracy.value}")
        print(f"Confidence Score: {result.confidence_score:.2f}")
        print(f"Expert Validation Needed: {result.expert_validation_needed}")
        print(f"Recommendations: {result.recommendations}")
        print(f"Warnings: {result.warnings}")

        # Export results
        output_path = Path("clinical_assessment_example.json")
        validator.export_assessment(result, output_path)

    # Run example
    asyncio.run(main())
