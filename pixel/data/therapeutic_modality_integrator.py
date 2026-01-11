"""
Therapeutic Modality Integration System

Integrates multiple therapeutic modalities (CBT, DBT, Psychodynamic, Humanistic, etc.)
into conversation generation with seamless switching, technique blending, and
modality-specific intervention selection based on client needs and context.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .dynamic_conversation_generator import ConversationTurn
from .therapeutic_conversation_schema import ClinicalContext, TherapeuticModality
from .therapist_response_generator import InterventionType, TherapistResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModalityIntegrationStrategy(Enum):
    """Strategies for integrating multiple therapeutic modalities"""
    SEQUENTIAL = "sequential"  # Use modalities in sequence
    BLENDED = "blended"  # Blend techniques from multiple modalities
    ADAPTIVE = "adaptive"  # Adapt modality based on client response
    HIERARCHICAL = "hierarchical"  # Primary modality with secondary support
    CONTEXTUAL = "contextual"  # Switch based on conversation context


class ModalityTransitionTrigger(Enum):
    """Triggers for switching between therapeutic modalities"""
    CLIENT_RESISTANCE = "client_resistance"
    LACK_OF_PROGRESS = "lack_of_progress"
    CRISIS_EMERGENCE = "crisis_emergence"
    GOAL_ACHIEVEMENT = "goal_achievement"
    CLIENT_PREFERENCE = "client_preference"
    SYMPTOM_CHANGE = "symptom_change"
    THERAPEUTIC_ALLIANCE = "therapeutic_alliance"


@dataclass
class ModalityProfile:
    """Profile defining characteristics and applications of a therapeutic modality"""
    modality: TherapeuticModality
    core_techniques: List[str]
    primary_interventions: List[InterventionType]
    target_symptoms: List[str]
    contraindications: List[str]
    effectiveness_domains: List[str]
    typical_session_structure: List[str]
    key_concepts: List[str]
    assessment_tools: List[str]
    homework_assignments: List[str]
    therapeutic_relationship_style: str
    change_mechanisms: List[str]


@dataclass
class ModalityIntegrationPlan:
    """Plan for integrating multiple therapeutic modalities"""
    primary_modality: TherapeuticModality
    secondary_modalities: List[TherapeuticModality]
    integration_strategy: ModalityIntegrationStrategy
    transition_triggers: List[ModalityTransitionTrigger]
    blending_ratio: Dict[TherapeuticModality, float]  # 0.0 to 1.0
    session_allocation: Dict[TherapeuticModality, int]  # sessions per modality
    technique_priorities: Dict[str, float]
    contraindication_checks: List[str]
    effectiveness_metrics: List[str]


@dataclass
class ModalityTransition:
    """Record of a modality transition during conversation"""
    transition_id: str
    from_modality: TherapeuticModality
    to_modality: TherapeuticModality
    trigger: ModalityTransitionTrigger
    rationale: str
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    client_response_quality: Optional[float] = None
    effectiveness_score: Optional[float] = None


@dataclass
class IntegratedResponse:
    """Response that integrates multiple therapeutic modalities"""
    primary_response: TherapistResponse
    secondary_techniques: List[Tuple[TherapeuticModality, str]]  # (modality, technique)
    integration_rationale: str
    modality_contributions: Dict[TherapeuticModality, float]
    blended_interventions: List[InterventionType]
    cross_modal_coherence: float
    overall_effectiveness_prediction: float


class TherapeuticModalityIntegrator:
    """
    System for integrating multiple therapeutic modalities in conversation generation
    
    Provides seamless integration of CBT, DBT, Psychodynamic, Humanistic, and other
    therapeutic approaches based on client needs, context, and treatment goals.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the therapeutic modality integrator"""
        self.config = self._load_config(config_path)
        
        # Load modality profiles and integration patterns
        self.modality_profiles = self._load_modality_profiles()
        self.integration_patterns = self._load_integration_patterns()
        self.transition_rules = self._load_transition_rules()
        
        # Integration settings
        self.integration_settings = {
            'max_modalities_per_session': 3,
            'min_coherence_threshold': 0.7,
            'transition_confidence_threshold': 0.8,
            'blending_smoothness_factor': 0.3,
            'effectiveness_weight': 0.4,
            'client_preference_weight': 0.3,
            'clinical_appropriateness_weight': 0.3
        }
        
        # Track integration history
        self.integration_history: List[ModalityTransition] = []
        self.current_integration_plan: Optional[ModalityIntegrationPlan] = None
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration settings"""
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        return {
            'enable_adaptive_integration': True,
            'allow_modality_blending': True,
            'require_clinical_validation': True,
            'max_transition_frequency': 3,  # per session
            'default_integration_strategy': 'adaptive',
            'enable_cross_modal_techniques': True
        }
    
    def _load_modality_profiles(self) -> Dict[TherapeuticModality, ModalityProfile]:
        """Load detailed profiles for each therapeutic modality"""
        return {
            TherapeuticModality.CBT: ModalityProfile(
                modality=TherapeuticModality.CBT,
                core_techniques=[
                    "Cognitive restructuring", "Thought challenging", "Behavioral activation",
                    "Exposure therapy", "Activity scheduling", "Cognitive defusion",
                    "Behavioral experiments", "Relapse prevention"
                ],
                primary_interventions=[
                    InterventionType.COGNITIVE_RESTRUCTURING,
                    InterventionType.BEHAVIORAL_ACTIVATION,
                    InterventionType.PSYCHOEDUCATION,
                    InterventionType.SKILL_BUILDING
                ],
                target_symptoms=[
                    "Depression", "Anxiety", "Panic disorder", "OCD", "PTSD",
                    "Social anxiety", "Specific phobias", "Eating disorders"
                ],
                contraindications=[
                    "Active psychosis", "Severe cognitive impairment",
                    "Acute suicidal crisis without stabilization"
                ],
                effectiveness_domains=[
                    "Symptom reduction", "Behavioral change", "Cognitive flexibility",
                    "Problem-solving skills", "Relapse prevention"
                ],
                typical_session_structure=[
                    "Agenda setting", "Homework review", "Problem identification",
                    "Intervention application", "Homework assignment", "Session summary"
                ],
                key_concepts=[
                    "Cognitive triangle", "Automatic thoughts", "Core beliefs",
                    "Behavioral activation", "Exposure hierarchy", "Cognitive distortions"
                ],
                assessment_tools=[
                    "Thought records", "Mood monitoring", "Activity schedules",
                    "Behavioral experiments", "Exposure logs"
                ],
                homework_assignments=[
                    "Thought challenging worksheets", "Behavioral activation tasks",
                    "Exposure exercises", "Mood and activity monitoring"
                ],
                therapeutic_relationship_style="Collaborative and structured",
                change_mechanisms=[
                    "Cognitive restructuring", "Behavioral activation",
                    "Exposure and habituation", "Skills acquisition"
                ]
            ),
            
            TherapeuticModality.DBT: ModalityProfile(
                modality=TherapeuticModality.DBT,
                core_techniques=[
                    "Mindfulness skills", "Distress tolerance", "Emotion regulation",
                    "Interpersonal effectiveness", "Radical acceptance", "TIPP skills",
                    "Wise mind", "Dialectical thinking"
                ],
                primary_interventions=[
                    InterventionType.SKILL_BUILDING,
                    InterventionType.VALIDATION,
                    InterventionType.CRISIS_INTERVENTION
                ],
                target_symptoms=[
                    "Borderline personality disorder", "Self-harm", "Suicidal ideation",
                    "Emotional dysregulation", "Interpersonal difficulties",
                    "Impulsivity", "Identity disturbance"
                ],
                contraindications=[
                    "Unwillingness to commit to treatment", "Active substance abuse without treatment",
                    "Severe eating disorder requiring medical stabilization"
                ],
                effectiveness_domains=[
                    "Emotional regulation", "Distress tolerance", "Interpersonal skills",
                    "Self-harm reduction", "Crisis management"
                ],
                typical_session_structure=[
                    "Mindfulness practice", "Skills review", "Problem analysis",
                    "Skills coaching", "Homework planning", "Validation and support"
                ],
                key_concepts=[
                    "Wise mind", "Emotional mind", "Reasonable mind", "Dialectical thinking",
                    "Radical acceptance", "Distress tolerance", "Window of tolerance"
                ],
                assessment_tools=[
                    "Diary cards", "Skills practice logs", "Emotion regulation worksheets",
                    "Interpersonal effectiveness tracking"
                ],
                homework_assignments=[
                    "Daily mindfulness practice", "Skills practice exercises",
                    "Diary card completion", "Interpersonal effectiveness practice"
                ],
                therapeutic_relationship_style="Validating and skills-focused",
                change_mechanisms=[
                    "Skills acquisition", "Emotional regulation", "Mindfulness practice",
                    "Dialectical thinking development"
                ]
            ),
            
            TherapeuticModality.PSYCHODYNAMIC: ModalityProfile(
                modality=TherapeuticModality.PSYCHODYNAMIC,
                core_techniques=[
                    "Free association", "Dream analysis", "Transference interpretation",
                    "Defense mechanism analysis", "Insight development",
                    "Working through", "Countertransference awareness"
                ],
                primary_interventions=[
                    InterventionType.INTERPRETATION,
                    InterventionType.EXPLORATION,
                    InterventionType.REFLECTION
                ],
                target_symptoms=[
                    "Personality disorders", "Relationship difficulties", "Identity issues",
                    "Chronic depression", "Anxiety with deep-rooted causes",
                    "Trauma with complex presentations"
                ],
                contraindications=[
                    "Acute crisis requiring immediate intervention",
                    "Severe cognitive impairment", "Active psychosis"
                ],
                effectiveness_domains=[
                    "Insight development", "Relationship patterns", "Emotional awareness",
                    "Self-understanding", "Character change"
                ],
                typical_session_structure=[
                    "Free association", "Exploration of themes", "Interpretation",
                    "Working through", "Integration", "Reflection"
                ],
                key_concepts=[
                    "Unconscious", "Transference", "Countertransference", "Defense mechanisms",
                    "Object relations", "Attachment patterns", "Repetition compulsion"
                ],
                assessment_tools=[
                    "Clinical interviews", "Projective tests", "Relationship pattern analysis",
                    "Defense mechanism assessment"
                ],
                homework_assignments=[
                    "Dream journaling", "Relationship pattern observation",
                    "Emotional awareness exercises", "Free writing"
                ],
                therapeutic_relationship_style="Exploratory and interpretive",
                change_mechanisms=[
                    "Insight development", "Working through", "Transference resolution",
                    "Integration of unconscious material"
                ]
            ),
            
            TherapeuticModality.HUMANISTIC: ModalityProfile(
                modality=TherapeuticModality.HUMANISTIC,
                core_techniques=[
                    "Active listening", "Empathetic reflection", "Unconditional positive regard",
                    "Genuineness", "Here-and-now focus", "Emotional expression",
                    "Self-actualization support", "Values clarification"
                ],
                primary_interventions=[
                    InterventionType.REFLECTION,
                    InterventionType.VALIDATION,
                    InterventionType.EXPLORATION
                ],
                target_symptoms=[
                    "Low self-esteem", "Identity confusion", "Existential concerns",
                    "Relationship difficulties", "Personal growth issues",
                    "Life transitions", "Grief and loss"
                ],
                contraindications=[
                    "Severe mental illness requiring structured treatment",
                    "Active substance abuse", "Acute safety concerns"
                ],
                effectiveness_domains=[
                    "Self-awareness", "Personal growth", "Emotional expression",
                    "Self-acceptance", "Authentic living"
                ],
                typical_session_structure=[
                    "Present moment awareness", "Emotional exploration",
                    "Reflection and validation", "Growth facilitation",
                    "Integration", "Empowerment"
                ],
                key_concepts=[
                    "Self-actualization", "Unconditional positive regard", "Congruence",
                    "Empathy", "Here-and-now", "Phenomenological experience"
                ],
                assessment_tools=[
                    "Self-concept measures", "Values assessment", "Emotional awareness scales",
                    "Personal growth inventories"
                ],
                homework_assignments=[
                    "Self-reflection exercises", "Values exploration", "Emotional journaling",
                    "Authentic expression practice"
                ],
                therapeutic_relationship_style="Empathetic and non-directive",
                change_mechanisms=[
                    "Self-awareness", "Emotional expression", "Self-acceptance",
                    "Personal growth facilitation"
                ]
            ),
            
            TherapeuticModality.SYSTEMIC: ModalityProfile(
                modality=TherapeuticModality.SYSTEMIC,
                core_techniques=[
                    "Family mapping", "Circular questioning", "Reframing",
                    "Boundary work", "Communication patterns analysis",
                    "Structural interventions", "Strategic interventions"
                ],
                primary_interventions=[
                    InterventionType.EXPLORATION,
                    InterventionType.SKILL_BUILDING,
                    InterventionType.ASSESSMENT
                ],
                target_symptoms=[
                    "Family dysfunction", "Relationship conflicts", "Communication problems",
                    "Boundary issues", "Codependency", "Family trauma"
                ],
                contraindications=[
                    "Individual crisis requiring immediate attention",
                    "Domestic violence without safety planning",
                    "Severe individual pathology requiring specialized treatment"
                ],
                effectiveness_domains=[
                    "Family functioning", "Communication skills", "Boundary setting",
                    "Relationship dynamics", "System change"
                ],
                typical_session_structure=[
                    "System assessment", "Pattern identification", "Intervention planning",
                    "Technique application", "System feedback", "Integration"
                ],
                key_concepts=[
                    "Systems theory", "Circular causality", "Homeostasis",
                    "Boundaries", "Triangulation", "Family life cycle"
                ],
                assessment_tools=[
                    "Genograms", "Family assessment scales", "Communication pattern analysis",
                    "Boundary assessment tools"
                ],
                homework_assignments=[
                    "Communication practice", "Boundary setting exercises",
                    "Family meeting facilitation", "Pattern observation"
                ],
                therapeutic_relationship_style="Systems-focused and collaborative",
                change_mechanisms=[
                    "System restructuring", "Communication improvement",
                    "Boundary clarification", "Pattern interruption"
                ]
            )
        }
    
    def _load_integration_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load patterns for integrating different modality combinations"""
        return {
            'cbt_dbt_integration': {
                'primary_modality': TherapeuticModality.CBT,
                'secondary_modality': TherapeuticModality.DBT,
                'integration_points': [
                    'Emotion regulation skills from DBT enhance CBT cognitive work',
                    'CBT thought challenging supports DBT wise mind development',
                    'DBT distress tolerance complements CBT exposure work',
                    'Both modalities emphasize skills-based approach'
                ],
                'blending_techniques': [
                    'Mindful cognitive restructuring',
                    'Distress tolerance during exposure',
                    'Emotion regulation in behavioral activation',
                    'Interpersonal effectiveness in social situations'
                ],
                'contraindications': [
                    'Overwhelming skill complexity for client',
                    'Conflicting therapeutic goals'
                ],
                'effectiveness_indicators': [
                    'Improved emotional regulation',
                    'Better distress tolerance',
                    'Enhanced cognitive flexibility',
                    'Reduced crisis episodes'
                ]
            },
            
            'cbt_psychodynamic_integration': {
                'primary_modality': TherapeuticModality.CBT,
                'secondary_modality': TherapeuticModality.PSYCHODYNAMIC,
                'integration_points': [
                    'CBT addresses surface symptoms while psychodynamic explores roots',
                    'Psychodynamic insights inform CBT intervention selection',
                    'CBT skills support psychodynamic working through',
                    'Both address cognitive and emotional patterns'
                ],
                'blending_techniques': [
                    'Insight-informed cognitive restructuring',
                    'Pattern-aware behavioral experiments',
                    'Transference-conscious skill building',
                    'Defense-aware exposure work'
                ],
                'contraindications': [
                    'Client preference for single approach',
                    'Time constraints limiting depth work'
                ],
                'effectiveness_indicators': [
                    'Deeper understanding of symptom origins',
                    'More sustainable behavioral changes',
                    'Improved self-awareness',
                    'Better relationship patterns'
                ]
            },
            
            'humanistic_cbt_integration': {
                'primary_modality': TherapeuticModality.HUMANISTIC,
                'secondary_modality': TherapeuticModality.CBT,
                'integration_points': [
                    'Humanistic acceptance supports CBT change work',
                    'CBT skills enhance humanistic growth process',
                    'Both emphasize client empowerment',
                    'Validation supports skill acquisition'
                ],
                'blending_techniques': [
                    'Empathetic cognitive restructuring',
                    'Values-based behavioral activation',
                    'Accepting exposure work',
                    'Growth-oriented skill building'
                ],
                'contraindications': [
                    'Client resistance to structured approaches',
                    'Philosophical conflicts between approaches'
                ],
                'effectiveness_indicators': [
                    'Increased self-acceptance',
                    'Authentic behavioral changes',
                    'Enhanced self-awareness',
                    'Sustainable growth'
                ]
            }
        }
    
    def _load_transition_rules(self) -> Dict[ModalityTransitionTrigger, Dict[str, Any]]:
        """Load rules for when and how to transition between modalities"""
        return {
            ModalityTransitionTrigger.CLIENT_RESISTANCE: {
                'detection_criteria': [
                    'Repeated non-compliance with homework',
                    'Verbal expressions of disagreement with approach',
                    'Lack of engagement in primary modality techniques',
                    'Therapeutic alliance strain'
                ],
                'transition_options': {
                    TherapeuticModality.CBT: [TherapeuticModality.HUMANISTIC, TherapeuticModality.PSYCHODYNAMIC],
                    TherapeuticModality.DBT: [TherapeuticModality.HUMANISTIC, TherapeuticModality.CBT],
                    TherapeuticModality.PSYCHODYNAMIC: [TherapeuticModality.HUMANISTIC, TherapeuticModality.CBT]
                },
                'transition_techniques': [
                    'Explore resistance with curiosity',
                    'Validate client experience',
                    'Collaboratively adjust approach',
                    'Introduce alternative techniques gradually'
                ]
            },
            
            ModalityTransitionTrigger.LACK_OF_PROGRESS: {
                'detection_criteria': [
                    'No symptom improvement after 6-8 sessions',
                    'Plateau in therapeutic gains',
                    'Client reports feeling stuck',
                    'Objective measures show no change'
                ],
                'transition_options': {
                    TherapeuticModality.CBT: [TherapeuticModality.DBT, TherapeuticModality.PSYCHODYNAMIC],
                    TherapeuticModality.HUMANISTIC: [TherapeuticModality.CBT, TherapeuticModality.DBT],
                    TherapeuticModality.PSYCHODYNAMIC: [TherapeuticModality.CBT, TherapeuticModality.SYSTEMIC]
                },
                'transition_techniques': [
                    'Assess barriers to progress',
                    'Introduce complementary techniques',
                    'Shift therapeutic focus',
                    'Enhance motivation and engagement'
                ]
            },
            
            ModalityTransitionTrigger.CRISIS_EMERGENCE: {
                'detection_criteria': [
                    'Suicidal ideation emergence',
                    'Self-harm behaviors',
                    'Severe emotional dysregulation',
                    'Acute safety concerns'
                ],
                'transition_options': {
                    TherapeuticModality.CBT: [TherapeuticModality.DBT],
                    TherapeuticModality.HUMANISTIC: [TherapeuticModality.DBT, TherapeuticModality.CBT],
                    TherapeuticModality.PSYCHODYNAMIC: [TherapeuticModality.DBT, TherapeuticModality.CBT]
                },
                'transition_techniques': [
                    'Immediate safety assessment',
                    'Crisis intervention skills',
                    'Distress tolerance techniques',
                    'Safety planning'
                ]
            },
            
            ModalityTransitionTrigger.GOAL_ACHIEVEMENT: {
                'detection_criteria': [
                    'Primary symptoms significantly improved',
                    'Treatment goals largely met',
                    'Client ready for deeper work',
                    'Maintenance phase reached'
                ],
                'transition_options': {
                    TherapeuticModality.CBT: [TherapeuticModality.PSYCHODYNAMIC, TherapeuticModality.HUMANISTIC],
                    TherapeuticModality.DBT: [TherapeuticModality.PSYCHODYNAMIC, TherapeuticModality.HUMANISTIC],
                    TherapeuticModality.HUMANISTIC: [TherapeuticModality.PSYCHODYNAMIC]
                },
                'transition_techniques': [
                    'Consolidate gains',
                    'Explore deeper themes',
                    'Focus on growth and self-actualization',
                    'Prevent relapse'
                ]
            }
        }
    
    async def create_integration_plan(self, clinical_context: ClinicalContext,
                                    treatment_goals: List[str],
                                    client_preferences: Optional[List[str]] = None) -> ModalityIntegrationPlan:
        """Create a comprehensive plan for integrating multiple therapeutic modalities"""
        try:
            # Analyze client presentation and needs
            modality_suitability = await self._assess_modality_suitability(
                clinical_context, treatment_goals
            )
            
            # Select primary and secondary modalities
            primary_modality = self._select_primary_modality(
                modality_suitability, client_preferences
            )
            secondary_modalities = self._select_secondary_modalities(
                primary_modality, modality_suitability, treatment_goals
            )
            
            # Determine integration strategy
            integration_strategy = self._determine_integration_strategy(
                primary_modality, secondary_modalities, clinical_context
            )
            
            # Calculate blending ratios
            blending_ratio = self._calculate_blending_ratios(
                primary_modality, secondary_modalities, integration_strategy
            )
            
            # Identify transition triggers
            transition_triggers = self._identify_transition_triggers(
                clinical_context, treatment_goals
            )
            
            # Create session allocation plan
            session_allocation = self._plan_session_allocation(
                primary_modality, secondary_modalities, integration_strategy
            )
            
            # Prioritize techniques
            technique_priorities = self._prioritize_techniques(
                primary_modality, secondary_modalities, treatment_goals
            )
            
            # Check contraindications
            contraindication_checks = self._check_contraindications(
                [primary_modality] + secondary_modalities, clinical_context
            )
            
            # Define effectiveness metrics
            effectiveness_metrics = self._define_effectiveness_metrics(
                primary_modality, secondary_modalities, treatment_goals
            )
            
            plan = ModalityIntegrationPlan(
                primary_modality=primary_modality,
                secondary_modalities=secondary_modalities,
                integration_strategy=integration_strategy,
                transition_triggers=transition_triggers,
                blending_ratio=blending_ratio,
                session_allocation=session_allocation,
                technique_priorities=technique_priorities,
                contraindication_checks=contraindication_checks,
                effectiveness_metrics=effectiveness_metrics
            )
            
            self.current_integration_plan = plan
            logger.info(f"Created integration plan with primary modality: {primary_modality.value}")
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating integration plan: {e}")
            raise
    
    async def _assess_modality_suitability(self, clinical_context: ClinicalContext,
                                         treatment_goals: List[str]) -> Dict[TherapeuticModality, float]:
        """Assess suitability of each modality for the given context and goals"""
        suitability_scores = {}
        
        for modality, profile in self.modality_profiles.items():
            score = 0.0
            
            # Check symptom match
            presentation_lower = clinical_context.client_presentation.lower()
            symptom_matches = sum(1 for symptom in profile.target_symptoms 
                                if symptom.lower() in presentation_lower)
            score += (symptom_matches / len(profile.target_symptoms)) * 0.4
            
            # Check goal alignment
            goal_matches = 0
            for goal in treatment_goals:
                goal_lower = goal.lower()
                for domain in profile.effectiveness_domains:
                    if any(word in goal_lower for word in domain.lower().split()):
                        goal_matches += 1
                        break
            score += (goal_matches / max(1, len(treatment_goals))) * 0.3
            
            # Check contraindications
            contraindication_penalty = 0
            for contraindication in profile.contraindications:
                if contraindication.lower() in presentation_lower:
                    contraindication_penalty += 0.2
            score -= min(contraindication_penalty, 0.5)
            
            # Consider crisis indicators
            if clinical_context.crisis_indicators:
                if modality == TherapeuticModality.DBT:
                    score += 0.2  # DBT is excellent for crisis
                elif modality == TherapeuticModality.PSYCHODYNAMIC:
                    score -= 0.1  # Less suitable for acute crisis
            
            # Consider cultural factors
            if clinical_context.cultural_factors:
                if modality in [TherapeuticModality.HUMANISTIC, TherapeuticModality.SYSTEMIC]:
                    score += 0.1  # More culturally adaptable
            
            suitability_scores[modality] = max(0.0, min(1.0, score))
        
        return suitability_scores
    
    def _select_primary_modality(self, suitability_scores: Dict[TherapeuticModality, float],
                               client_preferences: Optional[List[str]] = None) -> TherapeuticModality:
        """Select the primary therapeutic modality"""
        # Apply client preferences if provided
        if client_preferences:
            for modality, score in suitability_scores.items():
                modality_name = modality.value.replace('_', ' ')
                if any(pref.lower() in modality_name.lower() for pref in client_preferences):
                    suitability_scores[modality] += 0.2
        
        # Select modality with highest suitability score
        return max(suitability_scores.keys(), key=lambda x: suitability_scores[x])
    
    def _select_secondary_modalities(self, primary_modality: TherapeuticModality,
                                   suitability_scores: Dict[TherapeuticModality, float],
                                   treatment_goals: List[str]) -> List[TherapeuticModality]:
        """Select secondary modalities to complement the primary modality"""
        secondary_modalities = []
        
        # Remove primary modality from consideration
        remaining_scores = {k: v for k, v in suitability_scores.items() if k != primary_modality}
        
        # Sort by suitability score
        sorted_modalities = sorted(remaining_scores.keys(), 
                                 key=lambda x: remaining_scores[x], reverse=True)
        
        # Select top modalities that complement the primary
        for modality in sorted_modalities[:2]:  # Max 2 secondary modalities
            if remaining_scores[modality] > 0.3:  # Minimum threshold
                # Check if modality complements primary
                if self._modalities_complement(primary_modality, modality):
                    secondary_modalities.append(modality)
        
        return secondary_modalities
    
    def _modalities_complement(self, primary: TherapeuticModality, 
                             secondary: TherapeuticModality) -> bool:
        """Check if two modalities complement each other well"""
        complementary_pairs = {
            TherapeuticModality.CBT: [TherapeuticModality.DBT, TherapeuticModality.HUMANISTIC],
            TherapeuticModality.DBT: [TherapeuticModality.CBT, TherapeuticModality.PSYCHODYNAMIC],
            TherapeuticModality.PSYCHODYNAMIC: [TherapeuticModality.HUMANISTIC, TherapeuticModality.CBT],
            TherapeuticModality.HUMANISTIC: [TherapeuticModality.CBT, TherapeuticModality.PSYCHODYNAMIC],
            TherapeuticModality.SYSTEMIC: [TherapeuticModality.HUMANISTIC, TherapeuticModality.CBT]
        }
        
        return secondary in complementary_pairs.get(primary, [])
    
    def _determine_integration_strategy(self, primary_modality: TherapeuticModality,
                                      secondary_modalities: List[TherapeuticModality],
                                      clinical_context: ClinicalContext) -> ModalityIntegrationStrategy:
        """Determine the best integration strategy"""
        # Crisis situations favor adaptive strategy
        if clinical_context.crisis_indicators:
            return ModalityIntegrationStrategy.ADAPTIVE
        
        # Complex presentations favor blended approach
        if len(secondary_modalities) > 1:
            return ModalityIntegrationStrategy.BLENDED
        
        # Single secondary modality can use hierarchical
        if len(secondary_modalities) == 1:
            return ModalityIntegrationStrategy.HIERARCHICAL
        
        # Default to adaptive for flexibility
        return ModalityIntegrationStrategy.ADAPTIVE
    
    def _calculate_blending_ratios(self, primary_modality: TherapeuticModality,
                                 secondary_modalities: List[TherapeuticModality],
                                 strategy: ModalityIntegrationStrategy) -> Dict[TherapeuticModality, float]:
        """Calculate how much each modality should contribute"""
        ratios = {}
        
        if strategy == ModalityIntegrationStrategy.HIERARCHICAL:
            ratios[primary_modality] = 0.7
            for modality in secondary_modalities:
                ratios[modality] = 0.3 / len(secondary_modalities)
        
        elif strategy == ModalityIntegrationStrategy.BLENDED:
            total_modalities = 1 + len(secondary_modalities)
            ratios[primary_modality] = 0.5
            for modality in secondary_modalities:
                ratios[modality] = 0.5 / len(secondary_modalities)
        
        elif strategy == ModalityIntegrationStrategy.ADAPTIVE:
            # Start with hierarchical, adjust based on effectiveness
            ratios[primary_modality] = 0.6
            for modality in secondary_modalities:
                ratios[modality] = 0.4 / len(secondary_modalities)
        
        else:  # SEQUENTIAL or CONTEXTUAL
            ratios[primary_modality] = 1.0
            for modality in secondary_modalities:
                ratios[modality] = 0.0  # Will be adjusted dynamically
        
        return ratios
    
    def _identify_transition_triggers(self, clinical_context: ClinicalContext,
                                    treatment_goals: List[str]) -> List[ModalityTransitionTrigger]:
        """Identify relevant transition triggers for this case"""
        triggers = []
        
        # Always monitor for resistance and progress
        triggers.extend([
            ModalityTransitionTrigger.CLIENT_RESISTANCE,
            ModalityTransitionTrigger.LACK_OF_PROGRESS
        ])
        
        # Crisis-prone clients need crisis monitoring
        if clinical_context.crisis_indicators:
            triggers.append(ModalityTransitionTrigger.CRISIS_EMERGENCE)
        
        # Goal-oriented treatment needs achievement monitoring
        if treatment_goals:
            triggers.append(ModalityTransitionTrigger.GOAL_ACHIEVEMENT)
        
        # Relationship-focused goals need alliance monitoring
        if any('relationship' in goal.lower() for goal in treatment_goals):
            triggers.append(ModalityTransitionTrigger.THERAPEUTIC_ALLIANCE)
        
        return triggers
    
    def _plan_session_allocation(self, primary_modality: TherapeuticModality,
                               secondary_modalities: List[TherapeuticModality],
                               strategy: ModalityIntegrationStrategy) -> Dict[TherapeuticModality, int]:
        """Plan how many sessions to allocate to each modality"""
        allocation = {}
        total_sessions = 20  # Typical treatment length
        
        if strategy == ModalityIntegrationStrategy.SEQUENTIAL:
            # Divide sessions sequentially
            primary_sessions = total_sessions // 2
            allocation[primary_modality] = primary_sessions
            
            remaining_sessions = total_sessions - primary_sessions
            for i, modality in enumerate(secondary_modalities):
                allocation[modality] = remaining_sessions // len(secondary_modalities)
        
        elif strategy == ModalityIntegrationStrategy.HIERARCHICAL:
            # Primary gets most sessions
            allocation[primary_modality] = int(total_sessions * 0.7)
            remaining = total_sessions - allocation[primary_modality]
            
            for modality in secondary_modalities:
                allocation[modality] = remaining // len(secondary_modalities)
        
        else:  # BLENDED, ADAPTIVE, CONTEXTUAL
            # All modalities used throughout
            allocation[primary_modality] = total_sessions
            for modality in secondary_modalities:
                allocation[modality] = total_sessions  # Integrated throughout
        
        return allocation
    
    def _prioritize_techniques(self, primary_modality: TherapeuticModality,
                             secondary_modalities: List[TherapeuticModality],
                             treatment_goals: List[str]) -> Dict[str, float]:
        """Prioritize specific techniques based on modalities and goals"""
        technique_priorities = {}
        
        # Get techniques from primary modality
        primary_profile = self.modality_profiles[primary_modality]
        for technique in primary_profile.core_techniques:
            technique_priorities[technique] = 0.8  # High priority for primary
        
        # Add techniques from secondary modalities
        for modality in secondary_modalities:
            profile = self.modality_profiles[modality]
            for technique in profile.core_techniques:
                if technique not in technique_priorities:
                    technique_priorities[technique] = 0.5  # Medium priority for secondary
        
        # Boost priorities based on treatment goals
        for goal in treatment_goals:
            goal_lower = goal.lower()
            
            # Boost specific techniques based on goals
            if 'anxiety' in goal_lower or 'worry' in goal_lower:
                technique_priorities['Cognitive restructuring'] = technique_priorities.get('Cognitive restructuring', 0) + 0.2
                technique_priorities['Exposure therapy'] = technique_priorities.get('Exposure therapy', 0) + 0.2
            
            if 'depression' in goal_lower or 'mood' in goal_lower:
                technique_priorities['Behavioral activation'] = technique_priorities.get('Behavioral activation', 0) + 0.2
                technique_priorities['Activity scheduling'] = technique_priorities.get('Activity scheduling', 0) + 0.2
            
            if 'emotion' in goal_lower or 'feeling' in goal_lower:
                technique_priorities['Emotion regulation'] = technique_priorities.get('Emotion regulation', 0) + 0.2
                technique_priorities['Mindfulness skills'] = technique_priorities.get('Mindfulness skills', 0) + 0.2
            
            if 'relationship' in goal_lower or 'interpersonal' in goal_lower:
                technique_priorities['Interpersonal effectiveness'] = technique_priorities.get('Interpersonal effectiveness', 0) + 0.2
                technique_priorities['Communication patterns analysis'] = technique_priorities.get('Communication patterns analysis', 0) + 0.2
        
        # Normalize priorities to 0-1 range
        max_priority = max(technique_priorities.values()) if technique_priorities else 1.0
        for technique in technique_priorities:
            technique_priorities[technique] = min(1.0, technique_priorities[technique] / max_priority)
        
        return technique_priorities
    
    def _check_contraindications(self, modalities: List[TherapeuticModality],
                               clinical_context: ClinicalContext) -> List[str]:
        """Check for contraindications across selected modalities"""
        contraindications = []
        presentation_lower = clinical_context.client_presentation.lower()
        
        for modality in modalities:
            profile = self.modality_profiles[modality]
            for contraindication in profile.contraindications:
                if contraindication.lower() in presentation_lower:
                    contraindications.append(
                        f"{modality.value}: {contraindication}"
                    )
        
        return contraindications
    
    def _define_effectiveness_metrics(self, primary_modality: TherapeuticModality,
                                    secondary_modalities: List[TherapeuticModality],
                                    treatment_goals: List[str]) -> List[str]:
        """Define metrics to measure integration effectiveness"""
        metrics = []
        
        # Add metrics from all modalities
        all_modalities = [primary_modality] + secondary_modalities
        for modality in all_modalities:
            profile = self.modality_profiles[modality]
            metrics.extend(profile.effectiveness_domains)
        
        # Add goal-specific metrics
        for goal in treatment_goals:
            goal_lower = goal.lower()
            if 'symptom' in goal_lower:
                metrics.append('Symptom severity reduction')
            if 'function' in goal_lower:
                metrics.append('Functional improvement')
            if 'quality' in goal_lower:
                metrics.append('Quality of life enhancement')
        
        # Remove duplicates and return
        return list(set(metrics))
    
    async def generate_integrated_response(self, client_statement: str,
                                         conversation_history: List[ConversationTurn],
                                         clinical_context: ClinicalContext) -> IntegratedResponse:
        """Generate a response that integrates multiple therapeutic modalities"""
        try:
            if not self.current_integration_plan:
                raise ValueError("No integration plan available. Create plan first.")
            
            plan = self.current_integration_plan
            
            # Assess current context and needs
            context_assessment = await self._assess_current_context(
                client_statement, conversation_history, clinical_context
            )
            
            # Determine active modalities for this response
            active_modalities = self._determine_active_modalities(
                plan, context_assessment
            )
            
            # Generate primary response
            primary_response = await self._generate_primary_response(
                client_statement, plan.primary_modality, clinical_context
            )
            
            # Generate secondary techniques
            secondary_techniques = await self._generate_secondary_techniques(
                client_statement, active_modalities, plan, clinical_context
            )
            
            # Blend responses coherently
            integrated_response = await self._blend_responses(
                primary_response, secondary_techniques, plan
            )
            
            # Calculate integration metrics
            modality_contributions = self._calculate_modality_contributions(
                active_modalities, plan.blending_ratio
            )
            
            cross_modal_coherence = self._assess_cross_modal_coherence(
                integrated_response, active_modalities
            )
            
            effectiveness_prediction = self._predict_effectiveness(
                integrated_response, plan, context_assessment
            )
            
            return IntegratedResponse(
                primary_response=integrated_response,
                secondary_techniques=secondary_techniques,
                integration_rationale=self._generate_integration_rationale(
                    active_modalities, context_assessment
                ),
                modality_contributions=modality_contributions,
                blended_interventions=self._identify_blended_interventions(
                    integrated_response, active_modalities
                ),
                cross_modal_coherence=cross_modal_coherence,
                overall_effectiveness_prediction=effectiveness_prediction
            )
            
        except Exception as e:
            logger.error(f"Error generating integrated response: {e}")
            raise
    
    async def _assess_current_context(self, client_statement: str,
                                    conversation_history: List[ConversationTurn],
                                    clinical_context: ClinicalContext) -> Dict[str, Any]:
        """Assess current conversational context for modality selection"""
        assessment = {
            'emotional_intensity': 0.0,
            'crisis_indicators': [],
            'resistance_level': 0.0,
            'engagement_level': 0.0,
            'therapeutic_alliance': 0.0,
            'progress_indicators': [],
            'modality_preferences': []
        }
        
        statement_lower = client_statement.lower()
        
        # Assess emotional intensity
        high_emotion_words = ['overwhelmed', 'desperate', 'terrible', 'awful', 'intense']
        assessment['emotional_intensity'] = sum(1 for word in high_emotion_words 
                                              if word in statement_lower) / len(high_emotion_words)
        
        # Check for crisis indicators
        crisis_words = ['suicide', 'kill', 'die', 'hurt', 'end it all', 'can\'t take it']
        assessment['crisis_indicators'] = [word for word in crisis_words if word in statement_lower]
        
        # Assess resistance
        resistance_words = ['don\'t want', 'won\'t work', 'tried that', 'doesn\'t help']
        assessment['resistance_level'] = sum(1 for word in resistance_words 
                                           if word in statement_lower) / len(resistance_words)
        
        # Assess engagement
        engagement_words = ['understand', 'makes sense', 'helpful', 'want to try']
        assessment['engagement_level'] = sum(1 for word in engagement_words 
                                           if word in statement_lower) / len(engagement_words)
        
        # Assess therapeutic alliance
        alliance_words = ['trust', 'comfortable', 'safe', 'understand me']
        assessment['therapeutic_alliance'] = sum(1 for word in alliance_words 
                                               if word in statement_lower) / len(alliance_words)
        
        # Look for progress indicators
        progress_words = ['better', 'improving', 'helping', 'working']
        assessment['progress_indicators'] = [word for word in progress_words if word in statement_lower]
        
        return assessment
    
    def _determine_active_modalities(self, plan: ModalityIntegrationPlan,
                                   context_assessment: Dict[str, Any]) -> List[TherapeuticModality]:
        """Determine which modalities should be active for current response"""
        active_modalities = [plan.primary_modality]
        
        # Add secondary modalities based on context
        for modality in plan.secondary_modalities:
            should_activate = False
            
            # Crisis situations activate DBT
            if (context_assessment['crisis_indicators'] and 
                modality == TherapeuticModality.DBT):
                should_activate = True
            
            # High resistance activates humanistic approach
            if (context_assessment['resistance_level'] > 0.5 and 
                modality == TherapeuticModality.HUMANISTIC):
                should_activate = True
            
            # Low alliance activates relationship-focused modalities
            if (context_assessment['therapeutic_alliance'] < 0.3 and 
                modality in [TherapeuticModality.HUMANISTIC, TherapeuticModality.PSYCHODYNAMIC]):
                should_activate = True
            
            # High emotional intensity may need multiple approaches
            if context_assessment['emotional_intensity'] > 0.7:
                should_activate = True
            
            # Blended strategy activates all modalities
            if plan.integration_strategy == ModalityIntegrationStrategy.BLENDED:
                should_activate = True
            
            if should_activate:
                active_modalities.append(modality)
        
        return active_modalities
    
    async def _generate_primary_response(self, client_statement: str,
                                       primary_modality: TherapeuticModality,
                                       clinical_context: ClinicalContext) -> TherapistResponse:
        """Generate primary response using the primary modality"""
        # This would integrate with the existing TherapistResponseGenerator
        # For now, create a basic response structure
        
        profile = self.modality_profiles[primary_modality]
        
        # Select appropriate technique based on modality
        technique = random.choice(profile.core_techniques)
        intervention = random.choice(profile.primary_interventions)
        
        # Generate modality-specific response
        if primary_modality == TherapeuticModality.CBT:
            content = self._generate_cbt_response(client_statement, technique)
        elif primary_modality == TherapeuticModality.DBT:
            content = self._generate_dbt_response(client_statement, technique)
        elif primary_modality == TherapeuticModality.PSYCHODYNAMIC:
            content = self._generate_psychodynamic_response(client_statement, technique)
        elif primary_modality == TherapeuticModality.HUMANISTIC:
            content = self._generate_humanistic_response(client_statement, technique)
        else:
            content = f"I hear what you're saying. Let's explore this together using {technique.lower()}."
        
        return TherapistResponse(
            content=content,
            clinical_rationale=f"Using {primary_modality.value} approach with {technique}",
            therapeutic_technique=technique,
            intervention_type=intervention,
            confidence_score=0.8,
            contraindications=[],
            follow_up_suggestions=[]
        )
    
    def _generate_cbt_response(self, client_statement: str, technique: str) -> str:
        """Generate CBT-specific response"""
        if 'cognitive restructuring' in technique.lower():
            return "What evidence do you have for that thought? Let's examine whether there might be a more balanced way to look at this situation."
        elif 'behavioral activation' in technique.lower():
            return "It sounds like you're feeling stuck. What's one small activity that used to bring you some satisfaction that we could schedule for this week?"
        elif 'exposure' in technique.lower():
            return "I understand this feels scary. Let's break this down into smaller, manageable steps that we can work through gradually."
        else:
            return "Let's look at the connection between your thoughts, feelings, and behaviors in this situation."
    
    def _generate_dbt_response(self, client_statement: str, technique: str) -> str:
        """Generate DBT-specific response"""
        if 'mindfulness' in technique.lower():
            return "I can see you're in a lot of pain right now. Let's take a moment to ground ourselves. Can you notice what you're experiencing in this moment?"
        elif 'distress tolerance' in technique.lower():
            return "This is clearly a very distressing situation. Let's think about some skills that might help you get through this intense moment safely."
        elif 'emotion regulation' in technique.lower():
            return "I hear how intense these emotions are for you. Let's identify what you're feeling and think about what this emotion might be telling you."
        else:
            return "You're doing the best you can in this moment. Let's work together to find some skills that might help."
    
    def _generate_psychodynamic_response(self, client_statement: str, technique: str) -> str:
        """Generate psychodynamic-specific response"""
        if 'interpretation' in technique.lower():
            return "I'm noticing a pattern here. This situation seems to echo some themes we've discussed before. What do you make of that connection?"
        elif 'transference' in technique.lower():
            return "I'm curious about how you're experiencing our relationship right now. Does this remind you of other relationships in your life?"
        elif 'defense' in technique.lower():
            return "I wonder if there might be some feelings underneath that are difficult to access right now. What comes up for you when I say that?"
        else:
            return "Help me understand what this experience is like for you on a deeper level."
    
    def _generate_humanistic_response(self, client_statement: str, technique: str) -> str:
        """Generate humanistic-specific response"""
        if 'empathetic reflection' in technique.lower():
            return "It sounds like you're feeling really overwhelmed and perhaps a bit lost right now. That must be incredibly difficult."
        elif 'unconditional positive regard' in technique.lower():
            return "I want you to know that I see your strength in sharing this with me. You're worthy of care and support, especially during difficult times like this."
        elif 'genuineness' in technique.lower():
            return "I find myself feeling moved by what you've shared. Your experience matters, and I'm honored that you trust me with these feelings."
        else:
            return "I'm here with you in this moment. What feels most important for you to explore right now?"
    
    async def _generate_secondary_techniques(self, client_statement: str,
                                           active_modalities: List[TherapeuticModality],
                                           plan: ModalityIntegrationPlan,
                                           clinical_context: ClinicalContext) -> List[Tuple[TherapeuticModality, str]]:
        """Generate secondary techniques from other active modalities"""
        secondary_techniques = []
        
        for modality in active_modalities[1:]:  # Skip primary modality
            profile = self.modality_profiles[modality]
            
            # Select technique based on current context and priorities
            prioritized_techniques = [
                tech for tech in profile.core_techniques
                if plan.technique_priorities.get(tech, 0) > 0.5
            ]
            
            if prioritized_techniques:
                technique = random.choice(prioritized_techniques)
            else:
                technique = random.choice(profile.core_techniques)
            
            secondary_techniques.append((modality, technique))
        
        return secondary_techniques
    
    async def _blend_responses(self, primary_response: TherapistResponse,
                             secondary_techniques: List[Tuple[TherapeuticModality, str]],
                             plan: ModalityIntegrationPlan) -> TherapistResponse:
        """Blend primary response with secondary techniques"""
        blended_content = primary_response.content
        blended_rationale = primary_response.clinical_rationale
        
        # Add secondary technique elements
        for modality, technique in secondary_techniques:
            contribution_weight = plan.blending_ratio.get(modality, 0.0)
            
            if contribution_weight > 0.3:  # Significant contribution
                if modality == TherapeuticModality.DBT and 'mindfulness' in technique.lower():
                    blended_content += " Let's also take a moment to be mindful of what you're experiencing right now."
                elif modality == TherapeuticModality.HUMANISTIC and 'validation' in technique.lower():
                    blended_content += " I want to acknowledge how difficult this must be for you."
                elif modality == TherapeuticModality.PSYCHODYNAMIC and 'pattern' in technique.lower():
                    blended_content += " I'm also curious if you notice any patterns in how this situation relates to other experiences."
                
                blended_rationale += f" Integrated {technique} from {modality.value}."
        
        # Create blended response
        return TherapistResponse(
            content=blended_content,
            clinical_rationale=blended_rationale,
            therapeutic_technique=f"Integrated approach: {primary_response.therapeutic_technique}",
            intervention_type=primary_response.intervention_type,
            confidence_score=primary_response.confidence_score,
            contraindications=primary_response.contraindications,
            follow_up_suggestions=primary_response.follow_up_suggestions
        )
    
    def _calculate_modality_contributions(self, active_modalities: List[TherapeuticModality],
                                        blending_ratio: Dict[TherapeuticModality, float]) -> Dict[TherapeuticModality, float]:
        """Calculate actual contribution of each modality to the response"""
        contributions = {}
        
        for modality in active_modalities:
            base_contribution = blending_ratio.get(modality, 0.0)
            
            # Adjust based on actual usage
            if modality in active_modalities:
                contributions[modality] = base_contribution
            else:
                contributions[modality] = 0.0
        
        # Normalize to sum to 1.0
        total = sum(contributions.values())
        if total > 0:
            for modality in contributions:
                contributions[modality] /= total
        
        return contributions
    
    def _assess_cross_modal_coherence(self, response: TherapistResponse,
                                    active_modalities: List[TherapeuticModality]) -> float:
        """Assess how coherently different modalities are integrated"""
        if len(active_modalities) <= 1:
            return 1.0  # Perfect coherence with single modality
        
        coherence_score = 0.8  # Start with good baseline
        
        # Check for conflicting approaches
        content_lower = response.content.lower()
        
        # CBT + Psychodynamic can conflict if not well integrated
        if (TherapeuticModality.CBT in active_modalities and 
            TherapeuticModality.PSYCHODYNAMIC in active_modalities):
            if 'evidence' in content_lower and 'unconscious' in content_lower:
                coherence_score -= 0.1  # Potential conflict
        
        # DBT + CBT generally integrate well
        if (TherapeuticModality.DBT in active_modalities and 
            TherapeuticModality.CBT in active_modalities):
            if 'mindful' in content_lower and ('thought' in content_lower or 'behavior' in content_lower):
                coherence_score += 0.1  # Good integration
        
        # Humanistic + any other modality needs careful balance
        if TherapeuticModality.HUMANISTIC in active_modalities:
            if 'technique' in content_lower and 'feel' in content_lower:
                coherence_score += 0.05  # Balanced approach
        
        return max(0.0, min(1.0, coherence_score))
    
    def _predict_effectiveness(self, response: TherapistResponse,
                             plan: ModalityIntegrationPlan,
                             context_assessment: Dict[str, Any]) -> float:
        """Predict effectiveness of the integrated response"""
        effectiveness = 0.7  # Baseline
        
        # Higher effectiveness for well-matched modalities
        if context_assessment['crisis_indicators'] and TherapeuticModality.DBT in plan.secondary_modalities:
            effectiveness += 0.1
        
        if context_assessment['resistance_level'] > 0.5 and TherapeuticModality.HUMANISTIC in plan.secondary_modalities:
            effectiveness += 0.1
        
        # Lower effectiveness for poor therapeutic alliance
        if context_assessment['therapeutic_alliance'] < 0.3:
            effectiveness -= 0.1
        
        # Higher effectiveness for engaged clients
        if context_assessment['engagement_level'] > 0.7:
            effectiveness += 0.1
        
        # Adjust for response quality
        if response.confidence_score > 0.8:
            effectiveness += 0.05
        elif response.confidence_score < 0.6:
            effectiveness -= 0.05
        
        return max(0.0, min(1.0, effectiveness))
    
    def _generate_integration_rationale(self, active_modalities: List[TherapeuticModality],
                                      context_assessment: Dict[str, Any]) -> str:
        """Generate rationale for the modality integration"""
        rationale_parts = []
        
        primary_modality = active_modalities[0]
        rationale_parts.append(f"Primary {primary_modality.value} approach")
        
        if len(active_modalities) > 1:
            secondary_names = [mod.value for mod in active_modalities[1:]]
            rationale_parts.append(f"integrated with {', '.join(secondary_names)}")
        
        # Add context-specific rationale
        if context_assessment['crisis_indicators']:
            rationale_parts.append("to address crisis indicators")
        
        if context_assessment['resistance_level'] > 0.5:
            rationale_parts.append("to work with client resistance")
        
        if context_assessment['emotional_intensity'] > 0.7:
            rationale_parts.append("to manage high emotional intensity")
        
        return " ".join(rationale_parts) + "."
    
    def _identify_blended_interventions(self, response: TherapistResponse,
                                      active_modalities: List[TherapeuticModality]) -> List[InterventionType]:
        """Identify intervention types present in the blended response"""
        interventions = [response.intervention_type]
        
        content_lower = response.content.lower()
        
        # Identify additional interventions based on content
        if 'validate' in content_lower or 'understand' in content_lower:
            interventions.append(InterventionType.VALIDATION)
        
        if 'explore' in content_lower or 'curious' in content_lower:
            interventions.append(InterventionType.EXPLORATION)
        
        if 'skill' in content_lower or 'technique' in content_lower:
            interventions.append(InterventionType.SKILL_BUILDING)
        
        if 'thought' in content_lower and 'evidence' in content_lower:
            interventions.append(InterventionType.COGNITIVE_RESTRUCTURING)
        
        if 'mindful' in content_lower or 'moment' in content_lower:
            interventions.append(InterventionType.SKILL_BUILDING)  # Mindfulness skill
        
        return list(set(interventions))  # Remove duplicates
    
    async def assess_integration_effectiveness(self, conversation_history: List[ConversationTurn],
                                             plan: ModalityIntegrationPlan) -> Dict[str, float]:
        """Assess how effectively modalities are being integrated"""
        if not conversation_history:
            return {'overall_effectiveness': 0.0}
        
        # Analyze therapist turns for modality usage
        therapist_turns = [turn for turn in conversation_history 
                          if turn.speaker.value == 'therapist']
        
        if not therapist_turns:
            return {'overall_effectiveness': 0.0}
        
        # Calculate modality usage distribution
        modality_usage = {modality: 0 for modality in plan.blending_ratio.keys()}
        
        for turn in therapist_turns:
            # Analyze content for modality indicators
            content_lower = turn.content.lower()
            
            if any(word in content_lower for word in ['thought', 'evidence', 'behavior']):
                modality_usage[TherapeuticModality.CBT] += 1
            
            if any(word in content_lower for word in ['mindful', 'skill', 'distress']):
                modality_usage[TherapeuticModality.DBT] += 1
            
            if any(word in content_lower for word in ['pattern', 'unconscious', 'relationship']):
                modality_usage[TherapeuticModality.PSYCHODYNAMIC] += 1
            
            if any(word in content_lower for word in ['feel', 'experience', 'authentic']):
                modality_usage[TherapeuticModality.HUMANISTIC] += 1
        
        # Calculate effectiveness metrics
        total_turns = len(therapist_turns)
        effectiveness_metrics = {}
        
        # Modality balance effectiveness
        expected_usage = {mod: ratio * total_turns 
                         for mod, ratio in plan.blending_ratio.items()}
        
        balance_score = 0.0
        for modality, expected in expected_usage.items():
            actual = modality_usage.get(modality, 0)
            if expected > 0:
                balance_score += min(1.0, actual / expected)
        
        balance_score /= len(expected_usage) if expected_usage else 1
        effectiveness_metrics['modality_balance'] = balance_score
        
        # Integration coherence (based on smooth transitions)
        coherence_score = 0.8  # Baseline
        for i in range(1, len(therapist_turns)):
            prev_content = therapist_turns[i-1].content.lower()
            curr_content = therapist_turns[i].content.lower()
            
            # Check for abrupt modality switches
            prev_modalities = self._identify_content_modalities(prev_content)
            curr_modalities = self._identify_content_modalities(curr_content)
            
            if len(prev_modalities.intersection(curr_modalities)) == 0:
                coherence_score -= 0.05  # Penalty for abrupt switch
        
        effectiveness_metrics['integration_coherence'] = max(0.0, coherence_score)
        
        # Overall effectiveness
        effectiveness_metrics['overall_effectiveness'] = (
            effectiveness_metrics['modality_balance'] * 0.6 +
            effectiveness_metrics['integration_coherence'] * 0.4
        )
        
        return effectiveness_metrics
    
    def _identify_content_modalities(self, content: str) -> Set[TherapeuticModality]:
        """Identify which modalities are present in content"""
        modalities = set()
        content_lower = content.lower()
        
        # CBT indicators
        if any(word in content_lower for word in ['thought', 'evidence', 'behavior', 'cognitive']):
            modalities.add(TherapeuticModality.CBT)
        
        # DBT indicators
        if any(word in content_lower for word in ['mindful', 'skill', 'distress', 'emotion regulation']):
            modalities.add(TherapeuticModality.DBT)
        
        # Psychodynamic indicators
        if any(word in content_lower for word in ['pattern', 'unconscious', 'relationship', 'insight']):
            modalities.add(TherapeuticModality.PSYCHODYNAMIC)
        
        # Humanistic indicators
        if any(word in content_lower for word in ['feel', 'experience', 'authentic', 'growth']):
            modalities.add(TherapeuticModality.HUMANISTIC)
        
        # Systemic indicators
        if any(word in content_lower for word in ['family', 'system', 'boundary', 'communication']):
            modalities.add(TherapeuticModality.SYSTEMIC)
        
        return modalities
    
    async def suggest_modality_transition(self, conversation_history: List[ConversationTurn],
                                        clinical_context: ClinicalContext,
                                        current_plan: ModalityIntegrationPlan) -> Optional[ModalityTransition]:
        """Suggest a modality transition if needed"""
        if not conversation_history:
            return None
        
        # Analyze recent conversation for transition triggers
        recent_turns = conversation_history[-6:]  # Last 3 exchanges
        client_turns = [turn for turn in recent_turns if turn.speaker.value == 'client']
        
        if not client_turns:
            return None
        
        # Check for transition triggers
        for trigger in current_plan.transition_triggers:
            if await self._detect_transition_trigger(trigger, client_turns, clinical_context):
                # Suggest appropriate transition
                transition_options = self.transition_rules[trigger]['transition_options']
                current_primary = current_plan.primary_modality
                
                if current_primary in transition_options:
                    suggested_modality = transition_options[current_primary][0]  # First option
                    
                    return ModalityTransition(
                        transition_id=f"transition_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        from_modality=current_primary,
                        to_modality=suggested_modality,
                        trigger=trigger,
                        rationale=f"Detected {trigger.value} - transitioning to {suggested_modality.value}",
                        confidence_score=0.8
                    )
        
        return None
    
    async def _detect_transition_trigger(self, trigger: ModalityTransitionTrigger,
                                       client_turns: List[ConversationTurn],
                                       clinical_context: ClinicalContext) -> bool:
        """Detect if a specific transition trigger is present"""
        trigger_rules = self.transition_rules[trigger]
        detection_criteria = trigger_rules['detection_criteria']
        
        # Analyze client turns for trigger indicators
        combined_content = " ".join([turn.content.lower() for turn in client_turns])
        
        if trigger == ModalityTransitionTrigger.CLIENT_RESISTANCE:
            resistance_indicators = ['don\'t want', 'won\'t work', 'tried that', 'doesn\'t help']
            return any(indicator in combined_content for indicator in resistance_indicators)
        
        elif trigger == ModalityTransitionTrigger.CRISIS_EMERGENCE:
            crisis_indicators = ['suicide', 'kill', 'die', 'hurt', 'end it all']
            return any(indicator in combined_content for indicator in crisis_indicators)
        
        elif trigger == ModalityTransitionTrigger.LACK_OF_PROGRESS:
            stagnation_indicators = ['stuck', 'not helping', 'same', 'no better']
            return any(indicator in combined_content for indicator in stagnation_indicators)
        
        elif trigger == ModalityTransitionTrigger.GOAL_ACHIEVEMENT:
            achievement_indicators = ['better', 'improved', 'working', 'helping']
            return any(indicator in combined_content for indicator in achievement_indicators)
        
        return False
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get statistics about modality integration"""
        return {
            'supported_modalities': [modality.value for modality in self.modality_profiles.keys()],
            'integration_strategies': [strategy.value for strategy in ModalityIntegrationStrategy],
            'transition_triggers': [trigger.value for trigger in ModalityTransitionTrigger],
            'integration_patterns': list(self.integration_patterns.keys()),
            'current_plan': {
                'primary_modality': self.current_integration_plan.primary_modality.value if self.current_integration_plan else None,
                'secondary_modalities': [mod.value for mod in self.current_integration_plan.secondary_modalities] if self.current_integration_plan else [],
                'integration_strategy': self.current_integration_plan.integration_strategy.value if self.current_integration_plan else None
            } if self.current_integration_plan else None,
            'integration_history': len(self.integration_history),
            'configuration': self.config
        }
