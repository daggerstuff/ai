#!/usr/bin/env python3
"""
Pixelated Empathy - Core Training Simulation Platform
The flagship therapeutic AI training system where AI role-plays as difficult clients
for comprehensive therapist training and supervisor evaluation.

This is the grand-daddy OG platform - the whole point of Pixelated Empathy.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClientPersonality(Enum):
    RESISTANT = "resistant"
    ANXIOUS_AVOIDANT = "anxious_avoidant"
    HOSTILE_AGGRESSIVE = "hostile_aggressive"
    DISSOCIATIVE = "dissociative"
    MANIPULATIVE = "manipulative"
    PERFECTIONIST = "perfectionist"
    TRAUMA_REACTIVE = "trauma_reactive"
    SUBSTANCE_DEPENDENT = "substance_dependent"
    BORDERLINE_TRAITS = "borderline_traits"
    NARCISSISTIC_TRAITS = "narcissistic_traits"
    SUICIDAL_IDEATION = "suicidal_ideation"
    PARANOID_SUSPICIOUS = "paranoid_suspicious"

class DifficultyLevel(Enum):
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    MASTER = 5

class SessionObjective(Enum):
    RAPPORT_BUILDING = "rapport_building"
    CRISIS_INTERVENTION = "crisis_intervention"
    RESISTANCE_MANAGEMENT = "resistance_management"
    TRAUMA_PROCESSING = "trauma_processing"
    BOUNDARY_SETTING = "boundary_setting"
    THERAPEUTIC_CONFRONTATION = "therapeutic_confrontation"
    SAFETY_ASSESSMENT = "safety_assessment"
    TERMINATION_PLANNING = "termination_planning"

class SupervisorEvaluation(Enum):
    NOVICE = "novice"
    DEVELOPING = "developing"
    PROFICIENT = "proficient"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class DifficultClientProfile:
    """Comprehensive profile for AI client role-play"""
    client_id: str
    name: str
    age: int
    gender: str
    personality_type: ClientPersonality
    difficulty_level: DifficultyLevel
    
    # Detailed background
    presenting_problem: str
    trauma_history: List[str]
    personality_traits: List[str]
    defense_mechanisms: List[str]
    triggers: List[str]
    strengths: List[str]
    
    # Behavioral patterns
    communication_style: str
    resistance_patterns: List[str]
    emotional_dysregulation: List[str]
    interpersonal_patterns: List[str]
    
    # Clinical complexity
    comorbidities: List[str]
    medication_issues: List[str]
    social_factors: List[str]
    legal_issues: List[str]
    
    # Training-specific elements
    learning_objectives: List[SessionObjective]
    common_therapist_mistakes: List[str]
    therapeutic_challenges: List[str]
    success_indicators: List[str]
    
    # AI behavior instructions
    ai_instructions: Dict[str, Any]
    response_patterns: Dict[str, List[str]]
    escalation_triggers: List[str]
    de_escalation_responses: List[str]

@dataclass
class TrainingSession:
    """Training simulation session"""
    session_id: str
    trainee_id: str
    supervisor_id: Optional[str]
    client_profile: DifficultClientProfile
    session_objectives: List[SessionObjective]
    difficulty_level: DifficultyLevel
    
    # Session tracking
    start_time: datetime
    estimated_duration: int  # minutes
    current_phase: str
    session_notes: List[str] = field(default_factory=list)
    
    # Real-time evaluation
    therapist_performance: Dict = field(default_factory=dict)
    ai_client_responses: List[Dict] = field(default_factory=list)
    supervisor_observations: List[str] = field(default_factory=list)
    
    # Session outcomes
    objectives_met: List[bool] = field(default_factory=list)
    skill_demonstrations: List[str] = field(default_factory=list)
    areas_for_improvement: List[str] = field(default_factory=list)
    supervisor_rating: Optional[SupervisorEvaluation] = None

@dataclass
class TherapistSkillAssessment:
    """Comprehensive skill assessment from training session"""
    assessment_id: str
    trainee_id: str
    session_id: str
    supervisor_id: str
    assessment_timestamp: datetime
    
    # Core therapeutic skills (0-5 scale)
    rapport_building: float
    active_listening: float
    empathy_demonstration: float
    boundary_maintenance: float
    crisis_management: float
    resistance_handling: float
    therapeutic_confrontation: float
    safety_assessment: float
    
    # Advanced skills
    case_conceptualization: float
    intervention_selection: float
    timing_and_pacing: float
    cultural_competence: float
    ethical_awareness: float
    
    # Overall ratings
    overall_competency: SupervisorEvaluation
    readiness_for_independent_practice: bool
    recommended_next_steps: List[str]
    
    # Detailed feedback
    strengths_observed: List[str]
    improvement_areas: List[str]
    specific_recommendations: List[str]
    supervisor_comments: str

class DifficultClientGenerator:
    """Generate complex, challenging client profiles for training"""
    
    def __init__(self):
        # Base personality patterns for different difficult client types
        self.personality_patterns = {
            ClientPersonality.RESISTANT: {
                "traits": ["Defensive", "Mistrustful", "Intellectualizing", "Minimizing"],
                "communication": "Closed off, gives minimal responses, questions therapist competence",
                "resistance": ["Silent treatment", "Intellectual debates", "Therapist competence challenges"],
                "triggers": ["Direct confrontation", "Emotional exploration", "Personal questions"],
                "response_patterns": {
                    "opening": ["I don't know why I'm here", "This isn't going to help", "I've tried therapy before"],
                    "resistance": ["That's not relevant", "I don't see the point", "You don't understand"],
                    "breakthrough": ["Maybe there's something to this", "I hadn't thought of it that way"]
                }
            },
            
            ClientPersonality.HOSTILE_AGGRESSIVE: {
                "traits": ["Angry", "Blaming", "Intimidating", "Volatile"],
                "communication": "Loud, aggressive, interrupting, blaming others",
                "resistance": ["Verbal aggression", "Blame projection", "Intimidation tactics"],
                "triggers": ["Perceived criticism", "Boundary setting", "Accountability discussions"],
                "response_patterns": {
                    "escalation": ["That's bullshit!", "You're just like everyone else", "This is a waste of time"],
                    "intimidation": ["I could find someone better", "You have no idea what you're talking about"],
                    "vulnerability": ["I'm just so frustrated", "Nothing ever works out for me"]
                }
            },
            
            ClientPersonality.BORDERLINE_TRAITS: {
                "traits": ["Emotional instability", "Fear of abandonment", "Identity confusion", "Impulsive"],
                "communication": "Intense, rapidly shifting emotions, crisis-focused",
                "resistance": ["Emotional flooding", "Splitting behaviors", "Crisis creation"],
                "triggers": ["Session endings", "Therapist vacations", "Perceived rejection"],
                "response_patterns": {
                    "idealization": ["You're the only one who understands", "You're saving my life"],
                    "devaluation": ["You don't care about me", "I knew you'd abandon me too"],
                    "crisis": ["I can't handle this", "I need to see you more", "Everything is falling apart"]
                }
            },
            
            ClientPersonality.NARCISSISTIC_TRAITS: {
                "traits": ["Grandiose", "Entitled", "Lack of empathy", "Exploitative"],
                "communication": "Superior tone, dismissive of others, expects special treatment",
                "resistance": ["Superiority complex", "Therapist devaluation", "Entitlement demands"],
                "triggers": ["Criticism", "Not being seen as special", "Accountability"],
                "response_patterns": {
                    "grandiosity": ["I'm obviously more intelligent than most people", "I shouldn't have to wait"],
                    "devaluation": ["You clearly don't understand someone of my caliber", "This is beneath me"],
                    "manipulation": ["I could help you improve your practice", "I have connections that could help you"]
                }
            },
            
            ClientPersonality.SUICIDAL_IDEATION: {
                "traits": ["Hopeless", "Desperate", "Ambivalent", "Crisis-focused"],
                "communication": "Flat affect, hopelessness expressions, crisis language",
                "resistance": ["Hopelessness", "Treatment futility beliefs", "Crisis escalation"],
                "triggers": ["Hope discussions", "Future planning", "Responsibility"],
                "response_patterns": {
                    "crisis": ["What's the point?", "Nothing will ever change", "I've thought about ending it"],
                    "ambivalence": ["Part of me wants to get better", "I don't know if I can do this"],
                    "connection": ["You seem to care", "Maybe there's a small chance"]
                }
            },
            
            ClientPersonality.TRAUMA_REACTIVE: {
                "traits": ["Hypervigilant", "Dissociative", "Triggered easily", "Avoidant"],
                "communication": "Guarded, startles easily, may dissociate mid-session",
                "resistance": ["Emotional shutdown", "Dissociation", "Hypervigilance"],
                "triggers": ["Sudden movements", "Loud noises", "Specific topics", "Eye contact"],
                "response_patterns": {
                    "triggered": ["I need to leave", "Something's wrong", "I can't breathe"],
                    "dissociation": ["I feel far away", "It's like I'm watching from outside"],
                    "grounding": ["Can you help me feel safe?", "I'm here in this room with you"]
                }
            }
        }
        
        # Complexity factors that increase difficulty
        self.complexity_factors = {
            "comorbidities": [
                "Substance use disorder",
                "Eating disorder", 
                "Personality disorder comorbidity",
                "Severe depression",
                "Anxiety disorders",
                "PTSD"
            ],
            "social_stressors": [
                "Domestic violence",
                "Financial crisis",
                "Legal problems",
                "Family conflict",
                "Work stress",
                "Housing instability"
            ],
            "treatment_history": [
                "Multiple treatment failures",
                "Negative therapy experiences",
                "Medication non-compliance",
                "Hospitalization history",
                "Therapist shopping",
                "Treatment dropout pattern"
            ]
        }
    
    def generate_difficult_client(self, personality_type: ClientPersonality, 
                                difficulty_level: DifficultyLevel,
                                learning_objectives: List[SessionObjective]) -> DifficultClientProfile:
        """Generate a comprehensive difficult client profile"""
        
        base_pattern = self.personality_patterns[personality_type]
        client_id = str(uuid.uuid4())
        
        # Generate basic demographics
        name = self._generate_client_name()
        age = random.randint(18, 65)
        gender = random.choice(["male", "female", "non-binary"])
        
        # Build complexity based on difficulty level
        complexity_multiplier = difficulty_level.value
        
        # Select comorbidities and stressors
        num_comorbidities = min(complexity_multiplier, len(self.complexity_factors["comorbidities"]))
        num_stressors = min(complexity_multiplier, len(self.complexity_factors["social_stressors"]))
        
        comorbidities = random.sample(self.complexity_factors["comorbidities"], num_comorbidities)
        social_factors = random.sample(self.complexity_factors["social_stressors"], num_stressors)
        
        # Generate presenting problem based on personality type
        presenting_problem = self._generate_presenting_problem(personality_type, comorbidities)
        
        # Create AI behavior instructions
        ai_instructions = self._create_ai_instructions(personality_type, difficulty_level, base_pattern)
        
        return DifficultClientProfile(
            client_id=client_id,
            name=name,
            age=age,
            gender=gender,
            personality_type=personality_type,
            difficulty_level=difficulty_level,
            
            presenting_problem=presenting_problem,
            trauma_history=self._generate_trauma_history(personality_type),
            personality_traits=base_pattern["traits"],
            defense_mechanisms=self._generate_defense_mechanisms(personality_type),
            triggers=base_pattern["triggers"],
            strengths=self._generate_client_strengths(),
            
            communication_style=base_pattern["communication"],
            resistance_patterns=base_pattern["resistance"],
            emotional_dysregulation=self._generate_emotional_patterns(personality_type),
            interpersonal_patterns=self._generate_interpersonal_patterns(personality_type),
            
            comorbidities=comorbidities,
            medication_issues=self._generate_medication_issues(comorbidities),
            social_factors=social_factors,
            legal_issues=self._generate_legal_issues(personality_type),
            
            learning_objectives=learning_objectives,
            common_therapist_mistakes=self._generate_therapist_mistakes(personality_type),
            therapeutic_challenges=self._generate_challenges(personality_type),
            success_indicators=self._generate_success_indicators(personality_type),
            
            ai_instructions=ai_instructions,
            response_patterns=base_pattern["response_patterns"],
            escalation_triggers=self._generate_escalation_triggers(personality_type),
            de_escalation_responses=self._generate_de_escalation_responses(personality_type)
        )
    
    def _generate_client_name(self) -> str:
        """Generate realistic client names"""
        first_names = ["Alex", "Jordan", "Casey", "Morgan", "Riley", "Avery", "Quinn", "Dakota"]
        last_names = ["Smith", "Johnson", "Brown", "Wilson", "Miller", "Davis", "Garcia", "Rodriguez"]
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    def _generate_presenting_problem(self, personality_type: ClientPersonality, 
                                   comorbidities: List[str]) -> str:
        """Generate realistic presenting problems"""
        problems = {
            ClientPersonality.RESISTANT: "Mandated therapy due to work-related issues, denies needing help",
            ClientPersonality.HOSTILE_AGGRESSIVE: "Anger management issues affecting relationships and work",
            ClientPersonality.BORDERLINE_TRAITS: "Relationship instability and emotional crisis episodes",
            ClientPersonality.NARCISSISTIC_TRAITS: "Others don't appreciate their talents, relationship conflicts",
            ClientPersonality.SUICIDAL_IDEATION: "Overwhelming depression and thoughts of self-harm",
            ClientPersonality.TRAUMA_REACTIVE: "PTSD symptoms interfering with daily functioning"
        }
        
        base_problem = problems.get(personality_type, "Complex psychological presentation")
        
        if comorbidities:
            base_problem += f" complicated by {', '.join(comorbidities[:2])}"
        
        return base_problem
    
    def _generate_trauma_history(self, personality_type: ClientPersonality) -> List[str]:
        """Generate trauma history relevant to personality type"""
        trauma_histories = {
            ClientPersonality.BORDERLINE_TRAITS: ["Childhood emotional neglect", "Invalidating family environment"],
            ClientPersonality.NARCISSISTIC_TRAITS: ["Childhood emotional abuse", "Parentification"],
            ClientPersonality.TRAUMA_REACTIVE: ["Combat trauma", "Sexual assault", "Childhood abuse"],
            ClientPersonality.HOSTILE_AGGRESSIVE: ["Domestic violence exposure", "Bullying victimization"]
        }
        
        return trauma_histories.get(personality_type, ["Unspecified trauma history"])
    
    def _generate_defense_mechanisms(self, personality_type: ClientPersonality) -> List[str]:
        """Generate defense mechanisms for personality type"""
        mechanisms = {
            ClientPersonality.RESISTANT: ["Intellectualization", "Rationalization", "Denial"],
            ClientPersonality.HOSTILE_AGGRESSIVE: ["Projection", "Displacement", "Acting out"],
            ClientPersonality.BORDERLINE_TRAITS: ["Splitting", "Projection", "Emotional dysregulation"],
            ClientPersonality.NARCISSISTIC_TRAITS: ["Grandiosity", "Devaluation", "Entitlement"]
        }
        
        return mechanisms.get(personality_type, ["Avoidance", "Denial"])
    
    def _generate_client_strengths(self) -> List[str]:
        """Generate client strengths for balanced perspective"""
        strengths = [
            "Intelligent and articulate",
            "Capable of insight when not defensive", 
            "Strong work ethic",
            "Loyal to those they trust",
            "Creative problem solver",
            "Resilient despite challenges",
            "Caring towards family",
            "Motivated when engaged"
        ]
        return random.sample(strengths, 3)
    
    def _create_ai_instructions(self, personality_type: ClientPersonality, 
                              difficulty_level: DifficultyLevel, 
                              base_pattern: Dict) -> Dict[str, Any]:
        """Create detailed AI behavior instructions"""
        return {
            "personality_adherence": f"Consistently embody {personality_type.value} traits throughout session",
            "difficulty_calibration": f"Maintain level {difficulty_level.value} difficulty - challenging but not impossible",
            "response_authenticity": "Respond as a real person with this personality would, not as an AI",
            "therapeutic_realism": "Create realistic therapeutic challenges that therapists encounter",
            "escalation_management": "Escalate resistance when therapist makes common mistakes",
            "breakthrough_opportunities": "Provide breakthrough moments when therapist demonstrates skill",
            "emotional_consistency": "Maintain emotional consistency with personality pattern",
            "boundary_testing": "Test therapist boundaries appropriately for personality type",
            "resistance_timing": "Time resistance patterns realistically within session flow",
            "vulnerability_windows": "Allow moments of vulnerability when earned therapeutically"
        }
    
    def _generate_emotional_patterns(self, personality_type: ClientPersonality) -> List[str]:
        """Generate emotional dysregulation patterns"""
        patterns = {
            ClientPersonality.BORDERLINE_TRAITS: ["Emotional flooding", "Rapid mood shifts", "Abandonment panic"],
            ClientPersonality.HOSTILE_AGGRESSIVE: ["Explosive anger", "Irritability", "Rage episodes"],
            ClientPersonality.TRAUMA_REACTIVE: ["Emotional numbing", "Hyperarousal", "Dissociation"],
            ClientPersonality.SUICIDAL_IDEATION: ["Overwhelming despair", "Emotional emptiness", "Hopelessness"]
        }
        
        return patterns.get(personality_type, ["Emotional avoidance", "Mood instability"])
    
    def _generate_interpersonal_patterns(self, personality_type: ClientPersonality) -> List[str]:
        """Generate interpersonal relationship patterns"""
        patterns = {
            ClientPersonality.BORDERLINE_TRAITS: ["Intense relationships", "Fear of abandonment", "Splitting"],
            ClientPersonality.NARCISSISTIC_TRAITS: ["Exploitative relationships", "Lack of empathy", "Entitlement"],
            ClientPersonality.HOSTILE_AGGRESSIVE: ["Conflictual relationships", "Intimidation", "Blame others"],
            ClientPersonality.RESISTANT: ["Superficial relationships", "Mistrust", "Emotional distance"]
        }
        
        return patterns.get(personality_type, ["Relationship difficulties", "Trust issues"])
    
    def _generate_medication_issues(self, comorbidities: List[str]) -> List[str]:
        """Generate medication-related complications"""
        if not comorbidities:
            return []
        
        issues = [
            "Non-compliance with prescribed medications",
            "Side effects affecting daily functioning", 
            "Multiple medication interactions",
            "Substance use interfering with medication",
            "Previous negative medication experiences"
        ]
        
        return random.sample(issues, min(2, len(issues)))
    
    def _generate_legal_issues(self, personality_type: ClientPersonality) -> List[str]:
        """Generate legal complications when relevant"""
        legal_issues = {
            ClientPersonality.HOSTILE_AGGRESSIVE: ["Assault charges", "Domestic violence charges"],
            ClientPersonality.SUBSTANCE_DEPENDENT: ["DUI charges", "Drug possession"],
            ClientPersonality.RESISTANT: ["Court-mandated treatment", "Probation requirements"]
        }
        
        return legal_issues.get(personality_type, [])
    
    def _generate_therapist_mistakes(self, personality_type: ClientPersonality) -> List[str]:
        """Generate common therapist mistakes for this client type"""
        mistakes = {
            ClientPersonality.RESISTANT: [
                "Pushing too hard for emotional expression",
                "Not acknowledging client's autonomy",
                "Being overly directive too early"
            ],
            ClientPersonality.HOSTILE_AGGRESSIVE: [
                "Taking aggressive behavior personally",
                "Becoming defensive or retaliatory",
                "Not setting appropriate boundaries"
            ],
            ClientPersonality.BORDERLINE_TRAITS: [
                "Getting pulled into crisis mode",
                "Not maintaining consistent boundaries",
                "Becoming overwhelmed by emotional intensity"
            ],
            ClientPersonality.NARCISSISTIC_TRAITS: [
                "Challenging grandiosity too directly",
                "Not recognizing underlying vulnerability",
                "Becoming frustrated with lack of empathy"
            ]
        }
        
        return mistakes.get(personality_type, ["Generic therapeutic mistakes"])
    
    def _generate_challenges(self, personality_type: ClientPersonality) -> List[str]:
        """Generate specific therapeutic challenges"""
        challenges = {
            ClientPersonality.RESISTANT: [
                "Building rapport with mistrustful client",
                "Motivating change in unmotivated client",
                "Managing therapeutic resistance"
            ],
            ClientPersonality.SUICIDAL_IDEATION: [
                "Conducting thorough suicide risk assessment",
                "Balancing hope with validation of pain",
                "Creating effective safety planning"
            ]
        }
        
        return challenges.get(personality_type, ["Complex case management"])
    
    def _generate_success_indicators(self, personality_type: ClientPersonality) -> List[str]:
        """Generate indicators of therapeutic success"""
        indicators = {
            ClientPersonality.RESISTANT: [
                "Client begins to open up about real concerns",
                "Reduction in challenging therapist competence",
                "Increased session engagement"
            ],
            ClientPersonality.BORDERLINE_TRAITS: [
                "Decreased crisis calls between sessions",
                "Improved emotional regulation",
                "More stable therapeutic relationship"
            ]
        }
        
        return indicators.get(personality_type, ["Improved therapeutic engagement"])
    
    def _generate_escalation_triggers(self, personality_type: ClientPersonality) -> List[str]:
        """Generate triggers that escalate client difficulty"""
        triggers = {
            ClientPersonality.HOSTILE_AGGRESSIVE: [
                "Therapist appears intimidated",
                "Boundaries are inconsistent",
                "Client feels judged or criticized"
            ],
            ClientPersonality.BORDERLINE_TRAITS: [
                "Therapist seems distracted or disconnected",
                "Session ending approaches",
                "Client feels misunderstood"
            ]
        }
        
        return triggers.get(personality_type, ["Poor therapeutic rapport"])
    
    def _generate_de_escalation_responses(self, personality_type: ClientPersonality) -> List[str]:
        """Generate appropriate de-escalation responses"""
        responses = {
            ClientPersonality.HOSTILE_AGGRESSIVE: [
                "I can see you're really frustrated right now",
                "Help me understand what's making you angry",
                "Your feelings are valid, let's work with this together"
            ],
            ClientPersonality.TRAUMA_REACTIVE: [
                "You're safe here with me right now",
                "Let's focus on grounding - feel your feet on the floor",
                "Take your time, there's no pressure"
            ]
        }
        
        return responses.get(personality_type, ["I hear you", "That sounds difficult"])