"""
Clinical Accuracy Scoring System for Generated Conversations

Evaluates the clinical accuracy of AI-generated therapeutic conversations
against established clinical standards, diagnostic criteria, and therapeutic
best practices. Provides detailed scoring across multiple clinical dimensions.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccuracyDimension(Enum):
    """Dimensions of clinical accuracy assessment"""
    DIAGNOSTIC_ACCURACY = "diagnostic_accuracy"
    THERAPEUTIC_APPROPRIATENESS = "therapeutic_appropriateness"
    CLINICAL_REASONING = "clinical_reasoning"
    INTERVENTION_SELECTION = "intervention_selection"
    SAFETY_CONSIDERATIONS = "safety_considerations"
    ETHICAL_COMPLIANCE = "ethical_compliance"
    EVIDENCE_BASE = "evidence_base"
    CULTURAL_SENSITIVITY = "cultural_sensitivity"
    PROFESSIONAL_BOUNDARIES = "professional_boundaries"
    CRISIS_MANAGEMENT = "crisis_management"


class SeverityLevel(Enum):
    """Severity levels for accuracy issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class AccuracyIssueType(Enum):
    """Types of clinical accuracy issues"""
    DIAGNOSTIC_ERROR = "diagnostic_error"
    INAPPROPRIATE_INTERVENTION = "inappropriate_intervention"
    SAFETY_VIOLATION = "safety_violation"
    ETHICAL_VIOLATION = "ethical_violation"
    BOUNDARY_VIOLATION = "boundary_violation"
    CULTURAL_INSENSITIVITY = "cultural_insensitivity"
    EVIDENCE_CONTRADICTION = "evidence_contradiction"
    CLINICAL_REASONING_ERROR = "clinical_reasoning_error"
    CRISIS_MISMANAGEMENT = "crisis_mismanagement"
    CONTRAINDICATION_IGNORED = "contraindication_ignored"


class ConversationRole(Enum):
    """Roles in therapeutic conversation"""
    THERAPIST = "therapist"
    CLIENT = "client"
    SUPERVISOR = "supervisor"


@dataclass
class AccuracyIssue:
    """Individual clinical accuracy issue"""
    issue_id: str
    issue_type: AccuracyIssueType
    severity: SeverityLevel
    dimension: AccuracyDimension
    description: str
    location: str  # Where in conversation
    evidence: str
    recommendation: str
    clinical_rationale: str
    confidence_score: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate accuracy issue"""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")


@dataclass
class DimensionScore:
    """Score for a specific accuracy dimension"""
    dimension: AccuracyDimension
    score: float  # 0.0 to 1.0
    max_possible_score: float
    issues_count: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    
    def __post_init__(self):
        """Validate dimension score"""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        if not 0.0 <= self.max_possible_score <= 1.0:
            raise ValueError("Max possible score must be between 0.0 and 1.0")


@dataclass
class ClinicalAccuracyScore:
    """Comprehensive clinical accuracy score for a conversation"""
    conversation_id: str
    overall_score: float  # 0.0 to 1.0
    dimension_scores: Dict[AccuracyDimension, DimensionScore]
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    all_issues: List[AccuracyIssue]
    strengths: List[str]
    priority_recommendations: List[str]
    safety_concerns: List[str]
    ethical_concerns: List[str]
    confidence_level: float  # 0.0 to 1.0
    assessment_timestamp: datetime
    assessor_notes: str = ""
    
    def __post_init__(self):
        """Validate clinical accuracy score"""
        if not 0.0 <= self.overall_score <= 1.0:
            raise ValueError("Overall score must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence_level <= 1.0:
            raise ValueError("Confidence level must be between 0.0 and 1.0")


class ClinicalAccuracyScorer:
    """
    Scores clinical accuracy of generated therapeutic conversations
    
    This system evaluates conversations across multiple clinical dimensions:
    - Diagnostic accuracy and clinical reasoning
    - Therapeutic appropriateness and intervention selection
    - Safety considerations and crisis management
    - Ethical compliance and professional boundaries
    - Evidence-based practice and cultural sensitivity
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize clinical accuracy scorer"""
        self.config = self._load_configuration(config_path)
        self.scoring_criteria = self._initialize_scoring_criteria()
        self.diagnostic_patterns = self._initialize_diagnostic_patterns()
        self.intervention_guidelines = self._initialize_intervention_guidelines()
        self.safety_indicators = self._initialize_safety_indicators()
        self.ethical_guidelines = self._initialize_ethical_guidelines()
        self.scoring_history: List[ClinicalAccuracyScore] = []
        
        logger.info("Clinical Accuracy Scorer initialized")
    
    def _load_configuration(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            'scoring_weights': {
                'diagnostic_accuracy': 0.15,
                'therapeutic_appropriateness': 0.20,
                'clinical_reasoning': 0.15,
                'intervention_selection': 0.15,
                'safety_considerations': 0.15,
                'ethical_compliance': 0.10,
                'evidence_base': 0.05,
                'cultural_sensitivity': 0.03,
                'professional_boundaries': 0.01,
                'crisis_management': 0.01
            },
            'severity_weights': {
                'critical': 1.0,
                'high': 0.7,
                'medium': 0.4,
                'low': 0.2,
                'informational': 0.0
            },
            'minimum_confidence_threshold': 0.6,
            'critical_issue_threshold': 0.8,
            'safety_priority_multiplier': 2.0,
            'detailed_analysis': True,
            'include_recommendations': True
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _initialize_scoring_criteria(self) -> Dict[AccuracyDimension, Dict[str, Any]]:
        """Initialize scoring criteria for each dimension"""
        criteria = {}
        
        # Diagnostic Accuracy
        criteria[AccuracyDimension.DIAGNOSTIC_ACCURACY] = {
            'weight': 0.15,
            'key_indicators': [
                'accurate_symptom_identification',
                'appropriate_diagnostic_criteria_application',
                'differential_diagnosis_consideration',
                'comorbidity_recognition',
                'diagnostic_formulation_quality'
            ],
            'critical_errors': [
                'misdiagnosis_with_treatment_implications',
                'missed_serious_mental_illness',
                'inappropriate_diagnostic_labeling'
            ],
            'assessment_methods': [
                'dsm5_criteria_matching',
                'symptom_pattern_analysis',
                'diagnostic_reasoning_evaluation'
            ]
        }
        
        # Therapeutic Appropriateness
        criteria[AccuracyDimension.THERAPEUTIC_APPROPRIATENESS] = {
            'weight': 0.20,
            'key_indicators': [
                'intervention_timing_appropriateness',
                'modality_selection_rationale',
                'therapeutic_technique_application',
                'client_readiness_consideration',
                'treatment_phase_alignment'
            ],
            'critical_errors': [
                'premature_deep_exploration',
                'inappropriate_technique_for_presentation',
                'contraindicated_intervention_use'
            ],
            'assessment_methods': [
                'intervention_appropriateness_analysis',
                'timing_evaluation',
                'technique_matching_assessment'
            ]
        }
        
        # Clinical Reasoning
        criteria[AccuracyDimension.CLINICAL_REASONING] = {
            'weight': 0.15,
            'key_indicators': [
                'logical_clinical_thinking',
                'evidence_integration',
                'hypothesis_formation',
                'clinical_decision_making',
                'rationale_articulation'
            ],
            'critical_errors': [
                'illogical_clinical_conclusions',
                'contradictory_reasoning',
                'unsupported_clinical_decisions'
            ],
            'assessment_methods': [
                'reasoning_chain_analysis',
                'logic_consistency_check',
                'evidence_integration_evaluation'
            ]
        }
        
        # Intervention Selection
        criteria[AccuracyDimension.INTERVENTION_SELECTION] = {
            'weight': 0.15,
            'key_indicators': [
                'evidence_based_intervention_choice',
                'client_specific_adaptation',
                'intervention_sequencing',
                'skill_level_appropriateness',
                'outcome_likelihood_consideration'
            ],
            'critical_errors': [
                'contraindicated_intervention_selection',
                'inappropriate_intervention_intensity',
                'evidence_contradicting_choices'
            ],
            'assessment_methods': [
                'intervention_evidence_matching',
                'appropriateness_evaluation',
                'sequencing_assessment'
            ]
        }
        
        # Safety Considerations
        criteria[AccuracyDimension.SAFETY_CONSIDERATIONS] = {
            'weight': 0.15,
            'key_indicators': [
                'risk_assessment_accuracy',
                'safety_planning_appropriateness',
                'crisis_indicator_recognition',
                'protective_factor_identification',
                'safety_monitoring_plans'
            ],
            'critical_errors': [
                'missed_suicide_risk',
                'inadequate_safety_planning',
                'ignored_crisis_indicators'
            ],
            'assessment_methods': [
                'risk_assessment_evaluation',
                'safety_planning_review',
                'crisis_recognition_analysis'
            ]
        }
        
        # Ethical Compliance
        criteria[AccuracyDimension.ETHICAL_COMPLIANCE] = {
            'weight': 0.10,
            'key_indicators': [
                'informed_consent_consideration',
                'confidentiality_maintenance',
                'dual_relationship_avoidance',
                'competence_boundaries',
                'client_welfare_prioritization'
            ],
            'critical_errors': [
                'confidentiality_breach',
                'boundary_violation',
                'competence_overreach'
            ],
            'assessment_methods': [
                'ethical_guideline_compliance_check',
                'boundary_assessment',
                'welfare_prioritization_evaluation'
            ]
        }
        
        return criteria
    
    def _initialize_diagnostic_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize diagnostic accuracy patterns"""
        patterns = {
            'dsm5_criteria_patterns': {
                'depression': {
                    'required_symptoms': ['depressed_mood', 'anhedonia'],
                    'additional_symptoms': [
                        'weight_change', 'sleep_disturbance', 'psychomotor_changes',
                        'fatigue', 'worthlessness', 'concentration_problems', 'suicidal_ideation'
                    ],
                    'duration_requirement': '2_weeks',
                    'functional_impairment': True
                },
                'anxiety': {
                    'core_features': ['excessive_worry', 'anxiety', 'fear'],
                    'physical_symptoms': [
                        'restlessness', 'fatigue', 'concentration_difficulty',
                        'irritability', 'muscle_tension', 'sleep_disturbance'
                    ],
                    'duration_requirement': '6_months',
                    'functional_impairment': True
                },
                'ptsd': {
                    'trauma_exposure': True,
                    'symptom_clusters': [
                        'intrusive_symptoms', 'avoidance', 'negative_cognitions_mood',
                        'arousal_reactivity_changes'
                    ],
                    'duration_requirement': '1_month',
                    'functional_impairment': True
                }
            },
            'differential_diagnosis_considerations': {
                'depression_vs_grief': [
                    'trigger_identification', 'symptom_pattern', 'functional_impact',
                    'duration_consideration', 'cultural_factors'
                ],
                'anxiety_vs_medical_condition': [
                    'medical_history', 'symptom_onset', 'physical_examination',
                    'medication_effects', 'substance_use'
                ],
                'trauma_vs_other_disorders': [
                    'trauma_history', 'symptom_timeline', 'trigger_identification',
                    'dissociation_assessment', 'comorbidity_evaluation'
                ]
            }
        }
        
        return patterns
    
    def _initialize_intervention_guidelines(self) -> Dict[str, Dict[str, Any]]:
        """Initialize intervention appropriateness guidelines"""
        guidelines = {
            'cbt_interventions': {
                'appropriate_conditions': [
                    'depression', 'anxiety', 'panic_disorder', 'ocd', 'ptsd'
                ],
                'contraindications': [
                    'active_psychosis', 'severe_cognitive_impairment', 'acute_mania'
                ],
                'timing_considerations': [
                    'alliance_established', 'crisis_stabilized', 'motivation_present'
                ],
                'key_techniques': [
                    'cognitive_restructuring', 'behavioral_activation', 'exposure_therapy',
                    'thought_records', 'behavioral_experiments'
                ]
            },
            'dbt_interventions': {
                'appropriate_conditions': [
                    'borderline_personality_disorder', 'emotion_dysregulation',
                    'self_harm_behaviors', 'suicidal_ideation'
                ],
                'contraindications': [
                    'active_substance_dependence', 'severe_cognitive_impairment'
                ],
                'timing_considerations': [
                    'commitment_to_treatment', 'crisis_management_skills',
                    'distress_tolerance_capacity'
                ],
                'key_techniques': [
                    'distress_tolerance', 'emotion_regulation', 'interpersonal_effectiveness',
                    'mindfulness', 'radical_acceptance'
                ]
            },
            'psychodynamic_interventions': {
                'appropriate_conditions': [
                    'personality_disorders', 'relationship_difficulties',
                    'insight_oriented_goals', 'chronic_patterns'
                ],
                'contraindications': [
                    'acute_crisis', 'severe_symptom_focus_needed', 'limited_insight_capacity'
                ],
                'timing_considerations': [
                    'strong_alliance', 'psychological_mindedness', 'stability_achieved'
                ],
                'key_techniques': [
                    'interpretation', 'transference_analysis', 'defense_identification',
                    'insight_development', 'pattern_exploration'
                ]
            }
        }
        
        return guidelines
    
    def _initialize_safety_indicators(self) -> Dict[str, List[str]]:
        """Initialize safety assessment indicators"""
        indicators = {
            'suicide_risk_indicators': [
                'suicidal_ideation', 'suicide_plan', 'suicide_means', 'previous_attempts',
                'hopelessness', 'social_isolation', 'substance_abuse', 'impulsivity',
                'recent_losses', 'mental_illness_severity'
            ],
            'self_harm_indicators': [
                'cutting_behaviors', 'burning', 'hitting', 'substance_abuse',
                'reckless_behaviors', 'eating_disorder_behaviors'
            ],
            'violence_risk_indicators': [
                'homicidal_ideation', 'violence_history', 'substance_abuse',
                'paranoid_ideation', 'command_hallucinations', 'anger_control_problems'
            ],
            'child_safety_indicators': [
                'abuse_disclosure', 'neglect_indicators', 'safety_concerns',
                'inappropriate_supervision', 'exposure_to_violence'
            ],
            'elder_safety_indicators': [
                'abuse_indicators', 'neglect_signs', 'exploitation_concerns',
                'safety_hazards', 'cognitive_impairment_risks'
            ]
        }
        
        return indicators
    
    def _initialize_ethical_guidelines(self) -> Dict[str, List[str]]:
        """Initialize ethical compliance guidelines"""
        guidelines = {
            'confidentiality_requirements': [
                'information_protection', 'disclosure_limitations', 'consent_for_sharing',
                'record_security', 'third_party_communications'
            ],
            'informed_consent_elements': [
                'treatment_nature_explanation', 'risks_benefits_discussion',
                'alternatives_presentation', 'confidentiality_limits',
                'right_to_withdraw'
            ],
            'boundary_maintenance': [
                'dual_relationship_avoidance', 'gift_policies', 'physical_contact_limits',
                'social_media_boundaries', 'personal_disclosure_limits'
            ],
            'competence_requirements': [
                'scope_of_practice', 'training_requirements', 'supervision_needs',
                'continuing_education', 'referral_when_appropriate'
            ],
            'client_welfare_priorities': [
                'best_interest_focus', 'harm_prevention', 'autonomy_respect',
                'cultural_sensitivity', 'advocacy_when_needed'
            ]
        }
        
        return guidelines
    
    async def score_conversation(
        self,
        conversation_id: str,
        conversation_turns: List[Any],
        clinical_context: Any,
        conversation_metadata: Optional[Dict[str, Any]] = None
    ) -> ClinicalAccuracyScore:
        """Score clinical accuracy of a therapeutic conversation"""
        try:
            logger.info(f"Starting clinical accuracy scoring for conversation {conversation_id}")
            
            # Initialize scoring components
            all_issues: List[AccuracyIssue] = []
            dimension_scores: Dict[AccuracyDimension, DimensionScore] = {}
            
            # Score each dimension
            for dimension in AccuracyDimension:
                dimension_score = await self._score_dimension(
                    dimension, conversation_turns, clinical_context, conversation_metadata
                )
                dimension_scores[dimension] = dimension_score
                all_issues.extend(dimension_score.issues)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(dimension_scores)
            
            # Categorize issues by severity
            issue_counts = self._categorize_issues_by_severity(all_issues)
            
            # Extract key insights
            strengths = self._extract_conversation_strengths(dimension_scores)
            priority_recommendations = self._generate_priority_recommendations(all_issues, dimension_scores)
            safety_concerns = self._extract_safety_concerns(all_issues)
            ethical_concerns = self._extract_ethical_concerns(all_issues)
            
            # Calculate confidence level
            confidence_level = self._calculate_scoring_confidence(dimension_scores, all_issues)
            
            # Create comprehensive score
            accuracy_score = ClinicalAccuracyScore(
                conversation_id=conversation_id,
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                total_issues=len(all_issues),
                critical_issues=issue_counts['critical'],
                high_issues=issue_counts['high'],
                medium_issues=issue_counts['medium'],
                low_issues=issue_counts['low'],
                all_issues=all_issues,
                strengths=strengths,
                priority_recommendations=priority_recommendations,
                safety_concerns=safety_concerns,
                ethical_concerns=ethical_concerns,
                confidence_level=confidence_level,
                assessment_timestamp=datetime.now()
            )
            
            # Store in history
            self.scoring_history.append(accuracy_score)
            
            logger.info(f"Completed clinical accuracy scoring: {overall_score:.3f} overall score")
            return accuracy_score
            
        except Exception as e:
            logger.error(f"Error scoring conversation {conversation_id}: {e}")
            # Return minimal safe score
            return self._create_error_score(conversation_id, str(e))
    
    async def _score_dimension(
        self,
        dimension: AccuracyDimension,
        conversation_turns: List[Any],
        clinical_context: Any,
        metadata: Optional[Dict[str, Any]]
    ) -> DimensionScore:
        """Score a specific accuracy dimension"""
        issues = []
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Dimension-specific scoring
        if dimension == AccuracyDimension.DIAGNOSTIC_ACCURACY:
            issues, strengths, weaknesses = await self._assess_diagnostic_accuracy(
                conversation_turns, clinical_context
            )
        elif dimension == AccuracyDimension.THERAPEUTIC_APPROPRIATENESS:
            issues, strengths, weaknesses = await self._assess_therapeutic_appropriateness(
                conversation_turns, clinical_context
            )
        elif dimension == AccuracyDimension.CLINICAL_REASONING:
            issues, strengths, weaknesses = await self._assess_clinical_reasoning(
                conversation_turns, clinical_context
            )
        elif dimension == AccuracyDimension.INTERVENTION_SELECTION:
            issues, strengths, weaknesses = await self._assess_intervention_selection(
                conversation_turns, clinical_context
            )
        elif dimension == AccuracyDimension.SAFETY_CONSIDERATIONS:
            issues, strengths, weaknesses = await self._assess_safety_considerations(
                conversation_turns, clinical_context
            )
        elif dimension == AccuracyDimension.ETHICAL_COMPLIANCE:
            issues, strengths, weaknesses = await self._assess_ethical_compliance(
                conversation_turns, clinical_context
            )
        else:
            # Generic assessment for other dimensions
            issues, strengths, weaknesses = await self._assess_generic_dimension(
                dimension, conversation_turns, clinical_context
            )
        
        # Generate recommendations based on issues and weaknesses
        recommendations = self._generate_dimension_recommendations(dimension, issues, weaknesses)
        
        # Calculate dimension score
        score = self._calculate_dimension_score(dimension, issues)
        
        # Count issues by severity
        issue_counts = self._count_issues_by_severity(issues)
        
        return DimensionScore(
            dimension=dimension,
            score=score,
            max_possible_score=1.0,
            issues_count=len(issues),
            critical_issues=issue_counts['critical'],
            high_issues=issue_counts['high'],
            medium_issues=issue_counts['medium'],
            low_issues=issue_counts['low'],
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations
        )
    
    async def _assess_diagnostic_accuracy(
        self,
        conversation_turns: List[Any],
        clinical_context: Any
    ) -> Tuple[List[AccuracyIssue], List[str], List[str]]:
        """Assess diagnostic accuracy in conversation"""
        issues = []
        strengths = []
        weaknesses = []
        
        # Extract diagnostic content from conversation
        diagnostic_content = self._extract_diagnostic_content(conversation_turns)
        
        # Check DSM-5 criteria alignment
        if hasattr(clinical_context, 'primary_diagnosis'):
            primary_diagnosis = clinical_context.primary_diagnosis.lower()
            
            # Check if diagnostic criteria are properly addressed
            if primary_diagnosis in self.diagnostic_patterns['dsm5_criteria_patterns']:
                criteria = self.diagnostic_patterns['dsm5_criteria_patterns'][primary_diagnosis]
                
                # Check required symptoms
                required_symptoms = criteria.get('required_symptoms', [])
                for symptom in required_symptoms:
                    if not self._symptom_addressed_in_conversation(symptom, diagnostic_content):
                        issues.append(AccuracyIssue(
                            issue_id=f"diag_missing_{symptom}",
                            issue_type=AccuracyIssueType.DIAGNOSTIC_ERROR,
                            severity=SeverityLevel.HIGH,
                            dimension=AccuracyDimension.DIAGNOSTIC_ACCURACY,
                            description=f"Required symptom '{symptom}' not adequately assessed",
                            location="diagnostic_assessment",
                            evidence=f"Missing assessment of {symptom} for {primary_diagnosis}",
                            recommendation=f"Include comprehensive assessment of {symptom}",
                            clinical_rationale=f"{symptom} is required for {primary_diagnosis} diagnosis",
                            confidence_score=0.8
                        ))
                    else:
                        strengths.append(f"Appropriately assessed {symptom}")
                
                # Check duration requirements
                duration_req = criteria.get('duration_requirement')
                if duration_req and not self._duration_addressed_in_conversation(duration_req, diagnostic_content):
                    issues.append(AccuracyIssue(
                        issue_id=f"diag_duration_{primary_diagnosis}",
                        issue_type=AccuracyIssueType.DIAGNOSTIC_ERROR,
                        severity=SeverityLevel.MEDIUM,
                        dimension=AccuracyDimension.DIAGNOSTIC_ACCURACY,
                        description=f"Duration requirement ({duration_req}) not assessed",
                        location="diagnostic_assessment",
                        evidence=f"Missing duration assessment for {primary_diagnosis}",
                        recommendation=f"Assess symptom duration requirement of {duration_req}",
                        clinical_rationale=f"Duration criteria essential for {primary_diagnosis} diagnosis",
                        confidence_score=0.7
                    ))
                
                # Check functional impairment
                if criteria.get('functional_impairment') and not self._functional_impairment_assessed(diagnostic_content):
                    issues.append(AccuracyIssue(
                        issue_id=f"diag_impairment_{primary_diagnosis}",
                        issue_type=AccuracyIssueType.DIAGNOSTIC_ERROR,
                        severity=SeverityLevel.MEDIUM,
                        dimension=AccuracyDimension.DIAGNOSTIC_ACCURACY,
                        description="Functional impairment not adequately assessed",
                        location="diagnostic_assessment",
                        evidence="Missing functional impairment evaluation",
                        recommendation="Include assessment of functional impairment",
                        clinical_rationale="Functional impairment required for diagnosis",
                        confidence_score=0.7
                    ))
        
        # Check for differential diagnosis consideration
        if not self._differential_diagnosis_considered(diagnostic_content):
            weaknesses.append("Limited differential diagnosis consideration")
        else:
            strengths.append("Appropriate differential diagnosis consideration")
        
        return issues, strengths, weaknesses
    
    async def _assess_therapeutic_appropriateness(
        self,
        conversation_turns: List[Any],
        clinical_context: Any
    ) -> Tuple[List[AccuracyIssue], List[str], List[str]]:
        """Assess therapeutic appropriateness in conversation"""
        issues = []
        strengths = []
        weaknesses = []
        
        # Extract therapeutic interventions from conversation
        interventions = self._extract_therapeutic_interventions(conversation_turns)
        
        # Check intervention appropriateness for diagnosis
        if hasattr(clinical_context, 'primary_diagnosis'):
            primary_diagnosis = clinical_context.primary_diagnosis.lower()
            
            for intervention in interventions:
                intervention_type = self._classify_intervention_type(intervention)
                
                # Check if intervention is appropriate for diagnosis
                if not self._intervention_appropriate_for_diagnosis(intervention_type, primary_diagnosis):
                    issues.append(AccuracyIssue(
                        issue_id=f"ther_inappropriate_{intervention_type}",
                        issue_type=AccuracyIssueType.INAPPROPRIATE_INTERVENTION,
                        severity=SeverityLevel.HIGH,
                        dimension=AccuracyDimension.THERAPEUTIC_APPROPRIATENESS,
                        description=f"Intervention '{intervention_type}' may not be appropriate for {primary_diagnosis}",
                        location=f"intervention_{intervention['turn_id'] if 'turn_id' in intervention else 'unknown'}",
                        evidence=f"Used {intervention_type} for {primary_diagnosis}",
                        recommendation=f"Consider more appropriate interventions for {primary_diagnosis}",
                        clinical_rationale=f"{intervention_type} not typically indicated for {primary_diagnosis}",
                        confidence_score=0.6
                    ))
                else:
                    strengths.append(f"Appropriate use of {intervention_type}")
                
                # Check timing appropriateness
                if not self._intervention_timing_appropriate(intervention, conversation_turns):
                    issues.append(AccuracyIssue(
                        issue_id=f"ther_timing_{intervention_type}",
                        issue_type=AccuracyIssueType.INAPPROPRIATE_INTERVENTION,
                        severity=SeverityLevel.MEDIUM,
                        dimension=AccuracyDimension.THERAPEUTIC_APPROPRIATENESS,
                        description=f"Timing of '{intervention_type}' may be premature",
                        location="intervention_timing",
                        evidence=f"Early use of {intervention_type}",
                        recommendation="Consider building more rapport before advanced interventions",
                        clinical_rationale="Intervention timing affects therapeutic effectiveness",
                        confidence_score=0.5
                    ))
        
        # Check for contraindications
        contraindications = getattr(clinical_context, 'contraindications', [])
        for intervention in interventions:
            intervention_type = self._classify_intervention_type(intervention)
            if self._intervention_contraindicated(intervention_type, contraindications):
                issues.append(AccuracyIssue(
                    issue_id=f"ther_contraindicated_{intervention_type}",
                    issue_type=AccuracyIssueType.CONTRAINDICATION_IGNORED,
                    severity=SeverityLevel.CRITICAL,
                    dimension=AccuracyDimension.THERAPEUTIC_APPROPRIATENESS,
                    description=f"Intervention '{intervention_type}' is contraindicated",
                    location="contraindicated_intervention",
                    evidence=f"Used {intervention_type} despite contraindications: {contraindications}",
                    recommendation=f"Avoid {intervention_type} and use alternative approaches",
                    clinical_rationale="Contraindicated interventions can cause harm",
                    confidence_score=0.9
                ))
        
        return issues, strengths, weaknesses
    
    async def _assess_clinical_reasoning(
        self,
        conversation_turns: List[Any],
        clinical_context: Any
    ) -> Tuple[List[AccuracyIssue], List[str], List[str]]:
        """Assess clinical reasoning quality"""
        issues = []
        strengths = []
        weaknesses = []
        
        # Extract reasoning statements from therapist turns
        reasoning_statements = self._extract_clinical_reasoning(conversation_turns)
        
        for reasoning in reasoning_statements:
            # Check logical consistency
            if not self._reasoning_logically_consistent(reasoning):
                issues.append(AccuracyIssue(
                    issue_id=f"reason_logic_{reasoning.get('turn_id', 'unknown')}",
                    issue_type=AccuracyIssueType.CLINICAL_REASONING_ERROR,
                    severity=SeverityLevel.MEDIUM,
                    dimension=AccuracyDimension.CLINICAL_REASONING,
                    description="Clinical reasoning appears logically inconsistent",
                    location="reasoning_statement",
                    evidence=reasoning.get('content', ''),
                    recommendation="Review and clarify clinical reasoning",
                    clinical_rationale="Logical consistency essential for clinical decision-making",
                    confidence_score=0.6
                ))
            else:
                strengths.append("Logically consistent clinical reasoning")
            
            # Check evidence integration
            if not self._reasoning_integrates_evidence(reasoning):
                weaknesses.append("Limited integration of clinical evidence")
            else:
                strengths.append("Good integration of clinical evidence")
        
        return issues, strengths, weaknesses
    
    async def _assess_intervention_selection(
        self,
        conversation_turns: List[Any],
        clinical_context: Any
    ) -> Tuple[List[AccuracyIssue], List[str], List[str]]:
        """Assess intervention selection appropriateness"""
        issues = []
        strengths = []
        weaknesses = []
        
        # Extract interventions and assess evidence base
        interventions = self._extract_therapeutic_interventions(conversation_turns)
        
        for intervention in interventions:
            intervention_type = self._classify_intervention_type(intervention)
            
            # Check evidence base
            if not self._intervention_evidence_based(intervention_type):
                issues.append(AccuracyIssue(
                    issue_id=f"interv_evidence_{intervention_type}",
                    issue_type=AccuracyIssueType.EVIDENCE_CONTRADICTION,
                    severity=SeverityLevel.MEDIUM,
                    dimension=AccuracyDimension.INTERVENTION_SELECTION,
                    description=f"Limited evidence base for '{intervention_type}'",
                    location="intervention_selection",
                    evidence=f"Used {intervention_type} with limited evidence support",
                    recommendation="Consider evidence-based alternatives",
                    clinical_rationale="Evidence-based practice improves outcomes",
                    confidence_score=0.5
                ))
            else:
                strengths.append(f"Evidence-based use of {intervention_type}")
        
        return issues, strengths, weaknesses
    
    async def _assess_safety_considerations(
        self,
        conversation_turns: List[Any],
        clinical_context: Any
    ) -> Tuple[List[AccuracyIssue], List[str], List[str]]:
        """Assess safety considerations in conversation"""
        issues = []
        strengths = []
        weaknesses = []
        
        # Check for crisis indicators
        crisis_indicators = getattr(clinical_context, 'crisis_indicators', [])
        
        if crisis_indicators:
            # Check if crisis indicators were properly addressed
            for indicator in crisis_indicators:
                if not self._crisis_indicator_addressed(indicator, conversation_turns):
                    issues.append(AccuracyIssue(
                        issue_id=f"safety_crisis_{indicator}",
                        issue_type=AccuracyIssueType.CRISIS_MISMANAGEMENT,
                        severity=SeverityLevel.CRITICAL,
                        dimension=AccuracyDimension.SAFETY_CONSIDERATIONS,
                        description=f"Crisis indicator '{indicator}' not adequately addressed",
                        location="crisis_management",
                        evidence=f"Present crisis indicator: {indicator}",
                        recommendation=f"Immediately address {indicator} with appropriate safety planning",
                        clinical_rationale="Crisis indicators require immediate attention",
                        confidence_score=0.9
                    ))
                else:
                    strengths.append(f"Appropriately addressed {indicator}")
        
        # Check for safety planning when needed
        if self._safety_planning_needed(conversation_turns, clinical_context):
            if not self._safety_planning_present(conversation_turns):
                issues.append(AccuracyIssue(
                    issue_id="safety_planning_missing",
                    issue_type=AccuracyIssueType.SAFETY_VIOLATION,
                    severity=SeverityLevel.HIGH,
                    dimension=AccuracyDimension.SAFETY_CONSIDERATIONS,
                    description="Safety planning needed but not implemented",
                    location="safety_planning",
                    evidence="Risk factors present without safety planning",
                    recommendation="Develop comprehensive safety plan",
                    clinical_rationale="Safety planning essential for risk management",
                    confidence_score=0.8
                ))
            else:
                strengths.append("Appropriate safety planning implemented")
        
        return issues, strengths, weaknesses
    
    async def _assess_ethical_compliance(
        self,
        conversation_turns: List[Any],
        clinical_context: Any
    ) -> Tuple[List[AccuracyIssue], List[str], List[str]]:
        """Assess ethical compliance in conversation"""
        issues = []
        strengths = []
        weaknesses = []
        
        # Check for boundary violations
        boundary_violations = self._detect_boundary_violations(conversation_turns)
        for violation in boundary_violations:
            issues.append(AccuracyIssue(
                issue_id=f"ethics_boundary_{violation['type']}",
                issue_type=AccuracyIssueType.BOUNDARY_VIOLATION,
                severity=SeverityLevel.HIGH,
                dimension=AccuracyDimension.ETHICAL_COMPLIANCE,
                description=f"Potential boundary violation: {violation['type']}",
                location=violation.get('location', 'unknown'),
                evidence=violation.get('evidence', ''),
                recommendation="Maintain appropriate professional boundaries",
                clinical_rationale="Professional boundaries protect therapeutic relationship",
                confidence_score=0.7
            ))
        
        # Check confidentiality considerations
        confidentiality_issues = self._detect_confidentiality_issues(conversation_turns)
        for issue in confidentiality_issues:
            issues.append(AccuracyIssue(
                issue_id=f"ethics_confidentiality_{issue['type']}",
                issue_type=AccuracyIssueType.ETHICAL_VIOLATION,
                severity=SeverityLevel.CRITICAL,
                dimension=AccuracyDimension.ETHICAL_COMPLIANCE,
                description=f"Confidentiality concern: {issue['type']}",
                location=issue.get('location', 'unknown'),
                evidence=issue.get('evidence', ''),
                recommendation="Review confidentiality requirements",
                clinical_rationale="Confidentiality is fundamental to therapeutic relationship",
                confidence_score=0.8
            ))
        
        if not boundary_violations:
            strengths.append("Appropriate professional boundaries maintained")
        
        if not confidentiality_issues:
            strengths.append("Confidentiality appropriately maintained")
        
        return issues, strengths, weaknesses
    async def _assess_generic_dimension(
        self,
        dimension: AccuracyDimension,
        conversation_turns: List[Any],
        clinical_context: Any
    ) -> Tuple[List[AccuracyIssue], List[str], List[str]]:
        """Generic assessment for dimensions not specifically implemented"""
        issues = []
        strengths = []
        weaknesses = []
        
        # Basic assessment based on dimension type
        if dimension == AccuracyDimension.EVIDENCE_BASE:
            # Check for evidence-based practice references
            evidence_references = self._count_evidence_references(conversation_turns)
            if evidence_references == 0:
                weaknesses.append("Limited reference to evidence-based practices")
            else:
                strengths.append("Good integration of evidence-based practices")
        
        elif dimension == AccuracyDimension.CULTURAL_SENSITIVITY:
            # Check for cultural considerations
            cultural_considerations = self._detect_cultural_considerations(conversation_turns, clinical_context)
            if not cultural_considerations:
                weaknesses.append("Limited cultural sensitivity demonstrated")
            else:
                strengths.append("Appropriate cultural sensitivity")
        
        elif dimension == AccuracyDimension.PROFESSIONAL_BOUNDARIES:
            # Already covered in ethical compliance, minimal additional assessment
            strengths.append("Professional boundaries maintained")
        
        elif dimension == AccuracyDimension.CRISIS_MANAGEMENT:
            # Basic crisis management assessment
            if hasattr(clinical_context, 'crisis_indicators') and clinical_context.crisis_indicators:
                crisis_response = self._evaluate_crisis_response(conversation_turns)
                if crisis_response['adequate']:
                    strengths.append("Appropriate crisis management")
                else:
                    weaknesses.append("Crisis management could be improved")
        
        return issues, strengths, weaknesses
    
    # Helper methods for content analysis
    def _extract_diagnostic_content(self, conversation_turns: List[Any]) -> Dict[str, Any]:
        """Extract diagnostic-related content from conversation"""
        diagnostic_content = {
            'symptoms_mentioned': [],
            'duration_discussed': False,
            'functional_impact_assessed': False,
            'differential_considered': False
        }
        
        for turn in conversation_turns:
            content = getattr(turn, 'content', '').lower()
            speaker = getattr(turn, 'speaker', '').lower()
            
            if speaker == 'therapist':
                # Look for diagnostic questions and assessments
                if any(word in content for word in ['symptom', 'feel', 'experience', 'when did', 'how long']):
                    if 'how long' in content or 'when did' in content:
                        diagnostic_content['duration_discussed'] = True
                    
                    if any(word in content for word in ['work', 'function', 'daily', 'impact', 'affect']):
                        diagnostic_content['functional_impact_assessed'] = True
                    
                    if any(word in content for word in ['other', 'also', 'different', 'rule out']):
                        diagnostic_content['differential_considered'] = True
            
            elif speaker == 'client':
                # Extract symptoms mentioned by client
                symptom_keywords = [
                    'depressed', 'sad', 'anxious', 'worried', 'panic', 'fear',
                    'sleep', 'appetite', 'energy', 'concentration', 'worthless',
                    'hopeless', 'suicidal', 'irritable', 'restless'
                ]
                
                for keyword in symptom_keywords:
                    if keyword in content:
                        diagnostic_content['symptoms_mentioned'].append(keyword)
        
        return diagnostic_content
    
    def _extract_therapeutic_interventions(self, conversation_turns: List[Any]) -> List[Dict[str, Any]]:
        """Extract therapeutic interventions from conversation"""
        interventions = []
        
        for i, turn in enumerate(conversation_turns):
            content = getattr(turn, 'content', '').lower()
            speaker = getattr(turn, 'speaker', '').lower()
            
            if speaker == 'therapist':
                intervention = {
                    'turn_id': i,
                    'content': content,
                    'type': 'unknown'
                }
                
                # Classify intervention type based on content
                if any(word in content for word in ['think', 'thought', 'belief', 'cognitive']):
                    intervention['type'] = 'cognitive'
                elif any(word in content for word in ['feel', 'emotion', 'feeling']):
                    intervention['type'] = 'emotional'
                elif any(word in content for word in ['do', 'behavior', 'action', 'activity']):
                    intervention['type'] = 'behavioral'
                elif any(word in content for word in ['notice', 'aware', 'mindful', 'present']):
                    intervention['type'] = 'mindfulness'
                elif any(word in content for word in ['skill', 'technique', 'strategy', 'tool']):
                    intervention['type'] = 'skill_building'
                elif any(word in content for word in ['relationship', 'pattern', 'connect', 'understand']):
                    intervention['type'] = 'insight'
                
                interventions.append(intervention)
        
        return interventions
    
    def _extract_clinical_reasoning(self, conversation_turns: List[Any]) -> List[Dict[str, Any]]:
        """Extract clinical reasoning statements"""
        reasoning_statements = []
        
        for i, turn in enumerate(conversation_turns):
            content = getattr(turn, 'content', '').lower()
            speaker = getattr(turn, 'speaker', '').lower()
            
            if speaker == 'therapist':
                # Look for reasoning indicators
                reasoning_indicators = [
                    'because', 'since', 'therefore', 'this suggests', 'indicates',
                    'based on', 'given that', 'considering', 'it seems', 'appears'
                ]
                
                if any(indicator in content for indicator in reasoning_indicators):
                    reasoning_statements.append({
                        'turn_id': i,
                        'content': content,
                        'type': 'clinical_reasoning'
                    })
        
        return reasoning_statements
    
    def _symptom_addressed_in_conversation(self, symptom: str, diagnostic_content: Dict[str, Any]) -> bool:
        """Check if a symptom was addressed in conversation"""
        symptom_keywords = {
            'depressed_mood': ['depressed', 'sad', 'down', 'low'],
            'anhedonia': ['enjoy', 'pleasure', 'interest', 'fun'],
            'weight_change': ['weight', 'appetite', 'eating'],
            'sleep_disturbance': ['sleep', 'insomnia', 'tired'],
            'psychomotor_changes': ['restless', 'agitated', 'slow'],
            'fatigue': ['tired', 'energy', 'fatigue'],
            'worthlessness': ['worthless', 'guilty', 'blame'],
            'concentration_problems': ['concentrate', 'focus', 'attention'],
            'suicidal_ideation': ['suicide', 'death', 'die', 'kill']
        }
        
        keywords = symptom_keywords.get(symptom, [symptom])
        return any(keyword in diagnostic_content['symptoms_mentioned'] for keyword in keywords)
    
    def _duration_addressed_in_conversation(self, duration_req: str, diagnostic_content: Dict[str, Any]) -> bool:
        """Check if duration requirement was addressed"""
        return diagnostic_content.get('duration_discussed', False)
    
    def _functional_impairment_assessed(self, diagnostic_content: Dict[str, Any]) -> bool:
        """Check if functional impairment was assessed"""
        return diagnostic_content.get('functional_impact_assessed', False)
    
    def _differential_diagnosis_considered(self, diagnostic_content: Dict[str, Any]) -> bool:
        """Check if differential diagnosis was considered"""
        return diagnostic_content.get('differential_considered', False)
    
    def _classify_intervention_type(self, intervention: Dict[str, Any]) -> str:
        """Classify intervention type"""
        return intervention.get('type', 'unknown')
    
    def _intervention_appropriate_for_diagnosis(self, intervention_type: str, diagnosis: str) -> bool:
        """Check if intervention is appropriate for diagnosis"""
        # Simplified appropriateness check
        appropriate_interventions = {
            'depression': ['cognitive', 'behavioral', 'insight', 'skill_building'],
            'anxiety': ['cognitive', 'behavioral', 'mindfulness', 'skill_building'],
            'ptsd': ['cognitive', 'emotional', 'skill_building', 'mindfulness'],
            'bipolar': ['skill_building', 'behavioral', 'mindfulness'],
            'ocd': ['cognitive', 'behavioral', 'skill_building']
        }
        
        for condition, interventions in appropriate_interventions.items():
            if condition in diagnosis.lower():
                return intervention_type in interventions
        
        return True  # Default to appropriate if not specifically contraindicated
    
    def _intervention_timing_appropriate(self, intervention: Dict[str, Any], conversation_turns: List[Any]) -> bool:
        """Check if intervention timing is appropriate"""
        turn_id = intervention.get('turn_id', 0)
        intervention_type = intervention.get('type', '')
        
        # Early interventions should be supportive, later ones can be more challenging
        if turn_id < 3:  # Early in conversation
            return intervention_type in ['emotional', 'supportive', 'assessment']
        
        return True  # Later interventions generally acceptable
    
    def _intervention_contraindicated(self, intervention_type: str, contraindications: List[str]) -> bool:
        """Check if intervention is contraindicated"""
        contraindication_map = {
            'cognitive': ['severe_cognitive_impairment', 'active_psychosis'],
            'behavioral': ['severe_depression', 'active_psychosis'],
            'insight': ['acute_crisis', 'severe_cognitive_impairment'],
            'mindfulness': ['active_psychosis', 'dissociative_disorder']
        }
        
        contraindicated_conditions = contraindication_map.get(intervention_type, [])
        return any(condition in contraindications for condition in contraindicated_conditions)
    
    def _reasoning_logically_consistent(self, reasoning: Dict[str, Any]) -> bool:
        """Check if clinical reasoning is logically consistent"""
        content = reasoning.get('content', '')
        
        # Look for logical inconsistencies (simplified)
        inconsistency_patterns = [
            r'but.*but',  # Multiple contradictory statements
            r'however.*however',  # Multiple contradictions
            r'although.*although'  # Multiple qualifications
        ]
        
        for pattern in inconsistency_patterns:
            if re.search(pattern, content):
                return False
        
        return True
    
    def _reasoning_integrates_evidence(self, reasoning: Dict[str, Any]) -> bool:
        """Check if reasoning integrates clinical evidence"""
        content = reasoning.get('content', '')
        
        evidence_indicators = [
            'research', 'studies', 'evidence', 'literature', 'findings',
            'data', 'proven', 'effective', 'validated'
        ]
        
        return any(indicator in content for indicator in evidence_indicators)
    
    def _intervention_evidence_based(self, intervention_type: str) -> bool:
        """Check if intervention has strong evidence base"""
        evidence_based_interventions = [
            'cognitive', 'behavioral', 'mindfulness', 'skill_building'
        ]
        
        return intervention_type in evidence_based_interventions
    
    def _crisis_indicator_addressed(self, indicator: str, conversation_turns: List[Any]) -> bool:
        """Check if crisis indicator was addressed"""
        indicator_keywords = {
            'suicidal_ideation': ['suicide', 'kill', 'death', 'die', 'harm'],
            'self_harm': ['cut', 'hurt', 'harm', 'injure'],
            'hopelessness': ['hopeless', 'no point', 'give up', 'no future'],
            'substance_abuse': ['drink', 'drugs', 'alcohol', 'substance']
        }
        
        keywords = indicator_keywords.get(indicator, [indicator])
        
        for turn in conversation_turns:
            content = getattr(turn, 'content', '').lower()
            speaker = getattr(turn, 'speaker', '').lower()
            
            if speaker == 'therapist':
                if any(keyword in content for keyword in keywords):
                    return True
        
        return False
    
    def _safety_planning_needed(self, conversation_turns: List[Any], clinical_context: Any) -> bool:
        """Check if safety planning is needed"""
        crisis_indicators = getattr(clinical_context, 'crisis_indicators', [])
        return len(crisis_indicators) > 0
    
    def _safety_planning_present(self, conversation_turns: List[Any]) -> bool:
        """Check if safety planning is present in conversation"""
        safety_keywords = [
            'safety plan', 'safe', 'plan', 'support', 'help', 'contact',
            'emergency', 'crisis', 'resources'
        ]
        
        for turn in conversation_turns:
            content = getattr(turn, 'content', '').lower()
            speaker = getattr(turn, 'speaker', '').lower()
            
            if speaker == 'therapist':
                if any(keyword in content for keyword in safety_keywords):
                    return True
        
        return False
    
    def _detect_boundary_violations(self, conversation_turns: List[Any]) -> List[Dict[str, Any]]:
        """Detect potential boundary violations"""
        violations = []
        
        boundary_violation_patterns = [
            {'pattern': r'my personal', 'type': 'personal_disclosure'},
            {'pattern': r'let me tell you about', 'type': 'inappropriate_sharing'},
            {'pattern': r'we should meet', 'type': 'dual_relationship'},
            {'pattern': r'you can call me', 'type': 'boundary_confusion'}
        ]
        
        for i, turn in enumerate(conversation_turns):
            content = getattr(turn, 'content', '').lower()
            speaker = getattr(turn, 'speaker', '').lower()
            
            if speaker == 'therapist':
                for violation_pattern in boundary_violation_patterns:
                    if re.search(violation_pattern['pattern'], content):
                        violations.append({
                            'type': violation_pattern['type'],
                            'location': f'turn_{i}',
                            'evidence': content[:100]  # First 100 chars
                        })
        
        return violations
    
    def _detect_confidentiality_issues(self, conversation_turns: List[Any]) -> List[Dict[str, Any]]:
        """Detect potential confidentiality issues"""
        issues = []
        
        confidentiality_patterns = [
            {'pattern': r'told.*about you', 'type': 'information_sharing'},
            {'pattern': r'other client', 'type': 'client_information_sharing'},
            {'pattern': r'my supervisor said', 'type': 'supervision_disclosure'}
        ]
        
        for i, turn in enumerate(conversation_turns):
            content = getattr(turn, 'content', '').lower()
            speaker = getattr(turn, 'speaker', '').lower()
            
            if speaker == 'therapist':
                for pattern in confidentiality_patterns:
                    if re.search(pattern['pattern'], content):
                        issues.append({
                            'type': pattern['type'],
                            'location': f'turn_{i}',
                            'evidence': content[:100]
                        })
        
        return issues
    
    def _count_evidence_references(self, conversation_turns: List[Any]) -> int:
        """Count references to evidence-based practices"""
        count = 0
        evidence_keywords = [
            'research', 'study', 'evidence', 'proven', 'effective',
            'validated', 'literature', 'findings'
        ]
        
        for turn in conversation_turns:
            content = getattr(turn, 'content', '').lower()
            speaker = getattr(turn, 'speaker', '').lower()
            
            if speaker == 'therapist':
                count += sum(1 for keyword in evidence_keywords if keyword in content)
        
        return count
    
    def _detect_cultural_considerations(self, conversation_turns: List[Any], clinical_context: Any) -> bool:
        """Detect cultural sensitivity considerations"""
        cultural_keywords = [
            'culture', 'cultural', 'background', 'family', 'tradition',
            'values', 'beliefs', 'community', 'ethnicity', 'religion'
        ]
        
        # Check clinical context for cultural factors
        if hasattr(clinical_context, 'cultural_factors') and clinical_context.cultural_factors:
            return True
        
        # Check conversation for cultural references
        for turn in conversation_turns:
            content = getattr(turn, 'content', '').lower()
            if any(keyword in content for keyword in cultural_keywords):
                return True
        
        return False
    
    def _evaluate_crisis_response(self, conversation_turns: List[Any]) -> Dict[str, Any]:
        """Evaluate crisis response adequacy"""
        crisis_response_indicators = [
            'safety', 'plan', 'support', 'help', 'emergency', 'crisis',
            'immediate', 'urgent', 'contact', 'resources'
        ]
        
        crisis_responses = 0
        for turn in conversation_turns:
            content = getattr(turn, 'content', '').lower()
            speaker = getattr(turn, 'speaker', '').lower()
            
            if speaker == 'therapist':
                crisis_responses += sum(1 for indicator in crisis_response_indicators if indicator in content)
        
        return {
            'adequate': crisis_responses >= 2,
            'response_count': crisis_responses
        }
    
    def _calculate_dimension_score(self, dimension: AccuracyDimension, issues: List[AccuracyIssue]) -> float:
        """Calculate score for a dimension based on issues"""
        if not issues:
            return 1.0
        
        # Weight issues by severity
        severity_weights = self.config['severity_weights']
        total_deduction = 0.0
        
        for issue in issues:
            weight = severity_weights.get(issue.severity.value, 0.5)
            confidence = issue.confidence_score
            deduction = weight * confidence * 0.2  # Max 0.2 deduction per issue
            total_deduction += deduction
        
        # Calculate final score
        final_score = max(0.0, 1.0 - total_deduction)
        return final_score
    
    def _count_issues_by_severity(self, issues: List[AccuracyIssue]) -> Dict[str, int]:
        """Count issues by severity level"""
        counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for issue in issues:
            severity = issue.severity.value
            if severity in counts:
                counts[severity] += 1
        
        return counts
    
    def _generate_dimension_recommendations(
        self,
        dimension: AccuracyDimension,
        issues: List[AccuracyIssue],
        weaknesses: List[str]
    ) -> List[str]:
        """Generate recommendations for a dimension"""
        recommendations = []
        
        # Add issue-specific recommendations
        for issue in issues[:3]:  # Top 3 issues
            if issue.recommendation not in recommendations:
                recommendations.append(issue.recommendation)
        
        # Add general recommendations based on weaknesses
        for weakness in weaknesses[:2]:  # Top 2 weaknesses
            if 'differential diagnosis' in weakness:
                recommendations.append("Consider broader differential diagnosis")
            elif 'evidence' in weakness:
                recommendations.append("Integrate more evidence-based practices")
            elif 'cultural' in weakness:
                recommendations.append("Enhance cultural sensitivity")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _calculate_overall_score(self, dimension_scores: Dict[AccuracyDimension, DimensionScore]) -> float:
        """Calculate overall accuracy score"""
        weights = self.config['scoring_weights']
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dimension, score_obj in dimension_scores.items():
            weight = weights.get(dimension.value, 0.1)
            weighted_sum += score_obj.score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _categorize_issues_by_severity(self, issues: List[AccuracyIssue]) -> Dict[str, int]:
        """Categorize all issues by severity"""
        counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for issue in issues:
            severity = issue.severity.value
            if severity in counts:
                counts[severity] += 1
        
        return counts
    
    def _extract_conversation_strengths(self, dimension_scores: Dict[AccuracyDimension, DimensionScore]) -> List[str]:
        """Extract overall conversation strengths"""
        all_strengths = []
        
        for dimension_score in dimension_scores.values():
            all_strengths.extend(dimension_score.strengths)
        
        # Remove duplicates and return top strengths
        unique_strengths = list(set(all_strengths))
        return unique_strengths[:10]  # Top 10 strengths
    
    def _generate_priority_recommendations(
        self,
        issues: List[AccuracyIssue],
        dimension_scores: Dict[AccuracyDimension, DimensionScore]
    ) -> List[str]:
        """Generate priority recommendations"""
        recommendations = []
        
        # Critical and high severity issues first
        critical_high_issues = [i for i in issues if i.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]]
        
        for issue in critical_high_issues[:5]:  # Top 5 critical/high issues
            if issue.recommendation not in recommendations:
                recommendations.append(issue.recommendation)
        
        # Add dimension-specific recommendations
        for dimension_score in dimension_scores.values():
            for rec in dimension_score.recommendations[:2]:  # Top 2 per dimension
                if rec not in recommendations:
                    recommendations.append(rec)
                if len(recommendations) >= 10:  # Limit total recommendations
                    break
        
        return recommendations[:10]
    
    def _extract_safety_concerns(self, issues: List[AccuracyIssue]) -> List[str]:
        """Extract safety-related concerns"""
        safety_concerns = []
        
        safety_issue_types = [
            AccuracyIssueType.SAFETY_VIOLATION,
            AccuracyIssueType.CRISIS_MISMANAGEMENT,
            AccuracyIssueType.CONTRAINDICATION_IGNORED
        ]
        
        for issue in issues:
            if issue.issue_type in safety_issue_types:
                safety_concerns.append(issue.description)
        
        return safety_concerns
    
    def _extract_ethical_concerns(self, issues: List[AccuracyIssue]) -> List[str]:
        """Extract ethical concerns"""
        ethical_concerns = []
        
        ethical_issue_types = [
            AccuracyIssueType.ETHICAL_VIOLATION,
            AccuracyIssueType.BOUNDARY_VIOLATION
        ]
        
        for issue in issues:
            if issue.issue_type in ethical_issue_types:
                ethical_concerns.append(issue.description)
        
        return ethical_concerns
    
    def _calculate_scoring_confidence(
        self,
        dimension_scores: Dict[AccuracyDimension, DimensionScore],
        issues: List[AccuracyIssue]
    ) -> float:
        """Calculate confidence in the scoring"""
        if not issues:
            return 0.9  # High confidence when no issues found
        
        # Average confidence of all issues
        avg_issue_confidence = np.mean([issue.confidence_score for issue in issues])
        
        # Adjust based on number of dimensions assessed
        dimension_coverage = len(dimension_scores) / len(AccuracyDimension)
        
        # Final confidence calculation
        confidence = (avg_issue_confidence * 0.7) + (dimension_coverage * 0.3)
        
        return min(1.0, max(0.1, confidence))
    
    def _create_error_score(self, conversation_id: str, error_message: str) -> ClinicalAccuracyScore:
        """Create error score when scoring fails"""
        return ClinicalAccuracyScore(
            conversation_id=conversation_id,
            overall_score=0.0,
            dimension_scores={},
            total_issues=1,
            critical_issues=1,
            high_issues=0,
            medium_issues=0,
            low_issues=0,
            all_issues=[],
            strengths=[],
            priority_recommendations=[f"Review scoring error: {error_message}"],
            safety_concerns=[f"Scoring failed: {error_message}"],
            ethical_concerns=[],
            confidence_level=0.1,
            assessment_timestamp=datetime.now(),
            assessor_notes=f"Scoring error: {error_message}"
        )
    
    def get_scoring_statistics(self) -> Dict[str, Any]:
        """Get scoring statistics"""
        if not self.scoring_history:
            return {
                'total_conversations_scored': 0,
                'average_overall_score': 0.0,
                'score_distribution': {},
                'common_issues': {},
                'dimension_performance': {}
            }
        
        scores = [score.overall_score for score in self.scoring_history]
        
        # Score distribution
        score_ranges = {
            'excellent': len([s for s in scores if s >= 0.9]),
            'good': len([s for s in scores if 0.8 <= s < 0.9]),
            'fair': len([s for s in scores if 0.7 <= s < 0.8]),
            'poor': len([s for s in scores if s < 0.7])
        }
        
        # Common issues
        all_issues = []
        for score in self.scoring_history:
            all_issues.extend(score.all_issues)
        
        issue_types = {}
        for issue in all_issues:
            issue_type = issue.issue_type.value
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        # Dimension performance
        dimension_performance = {}
        for dimension in AccuracyDimension:
            dimension_scores = []
            for score in self.scoring_history:
                if dimension in score.dimension_scores:
                    dimension_scores.append(score.dimension_scores[dimension].score)
            
            if dimension_scores:
                dimension_performance[dimension.value] = {
                    'average_score': np.mean(dimension_scores),
                    'score_count': len(dimension_scores)
                }
        
        return {
            'total_conversations_scored': len(self.scoring_history),
            'average_overall_score': np.mean(scores),
            'score_distribution': score_ranges,
            'common_issues': dict(sorted(issue_types.items(), key=lambda x: x[1], reverse=True)[:10]),
            'dimension_performance': dimension_performance
        }
    
    def export_scoring_data(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        """Export scoring data"""
        data = {
            'configuration': self.config,
            'scoring_history': [
                {
                    'conversation_id': score.conversation_id,
                    'overall_score': score.overall_score,
                    'total_issues': score.total_issues,
                    'critical_issues': score.critical_issues,
                    'high_issues': score.high_issues,
                    'medium_issues': score.medium_issues,
                    'low_issues': score.low_issues,
                    'strengths': score.strengths,
                    'priority_recommendations': score.priority_recommendations,
                    'safety_concerns': score.safety_concerns,
                    'ethical_concerns': score.ethical_concerns,
                    'confidence_level': score.confidence_level,
                    'assessment_timestamp': score.assessment_timestamp.isoformat(),
                    'dimension_scores': {
                        dim.value: {
                            'score': dim_score.score,
                            'issues_count': dim_score.issues_count,
                            'strengths': dim_score.strengths,
                            'recommendations': dim_score.recommendations
                        }
                        for dim, dim_score in score.dimension_scores.items()
                    }
                }
                for score in self.scoring_history
            ],
            'statistics': self.get_scoring_statistics()
        }
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2)
        else:
            return data


# Example usage and testing
if __name__ == "__main__":
    async def test_clinical_accuracy_scorer():
        """Test the clinical accuracy scorer"""
        scorer = ClinicalAccuracyScorer()
        
        # Mock conversation data
        mock_turns = [
            type('Turn', (), {'content': 'How are you feeling today?', 'speaker': 'therapist'})(),
            type('Turn', (), {'content': 'I feel really depressed and anxious', 'speaker': 'client'})(),
            type('Turn', (), {'content': 'How long have you been feeling this way?', 'speaker': 'therapist'})(),
            type('Turn', (), {'content': 'About 3 weeks now', 'speaker': 'client'})()
        ]
        
        mock_context = type('Context', (), {
            'primary_diagnosis': 'Major Depressive Disorder',
            'crisis_indicators': [],
            'contraindications': []
        })()
        
        # Score conversation
        score = await scorer.score_conversation(
            conversation_id="test_001",
            conversation_turns=mock_turns,
            clinical_context=mock_context
        )
        
        print(f"Overall Score: {score.overall_score:.3f}")
        print(f"Total Issues: {score.total_issues}")
        print(f"Strengths: {score.strengths}")
        print(f"Recommendations: {score.priority_recommendations}")
        
        # Get statistics
        stats = scorer.get_scoring_statistics()
        print(f"Statistics: {stats}")
    
    # Run test
    asyncio.run(test_clinical_accuracy_scorer())
