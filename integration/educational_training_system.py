#!/usr/bin/env python3
"""
Educational Training System Implementation
AI-powered training platform for mental health professionals.

This system provides:
- Simulated client interactions for training
- Competency assessment and tracking
- Personalized learning pathways
- Real-time feedback and coaching
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LearnerLevel(Enum):
    STUDENT = "student"
    INTERN = "intern"
    JUNIOR_CLINICIAN = "junior_clinician"
    EXPERIENCED_CLINICIAN = "experienced_clinician"
    SUPERVISOR = "supervisor"

class CompetencyArea(Enum):
    ASSESSMENT = "assessment"
    INTERVENTION = "intervention"
    THERAPEUTIC_RELATIONSHIP = "therapeutic_relationship"
    CRISIS_MANAGEMENT = "crisis_management"
    CULTURAL_COMPETENCE = "cultural_competence"
    ETHICAL_PRACTICE = "ethical_practice"
    DOCUMENTATION = "documentation"
    PROFESSIONAL_DEVELOPMENT = "professional_development"

class ScenarioType(Enum):
    INITIAL_ASSESSMENT = "initial_assessment"
    ONGOING_THERAPY = "ongoing_therapy"
    CRISIS_INTERVENTION = "crisis_intervention"
    TERMINATION = "termination"
    DIFFICULT_CLIENT = "difficult_client"
    ETHICAL_DILEMMA = "ethical_dilemma"
    MULTICULTURAL = "multicultural"
    TRAUMA_FOCUSED = "trauma_focused"

@dataclass
class LearningObjective:
    """Specific learning objective"""
    objective_id: str
    title: str
    description: str
    competency_area: CompetencyArea
    target_level: LearnerLevel
    assessment_criteria: List[str]
    prerequisite_objectives: List[str]

@dataclass
class TrainingScenario:
    """Simulated training scenario"""
    scenario_id: str
    title: str
    scenario_type: ScenarioType
    difficulty_level: int  # 1-5 scale
    learning_objectives: List[str]
    client_background: Dict
    presenting_problem: str
    session_context: Dict
    assessment_rubric: Dict
    expected_duration_minutes: int

@dataclass
class PerformanceAssessment:
    """Assessment of trainee performance"""
    assessment_id: str
    scenario_id: str
    learner_id: str
    competency_scores: Dict[CompetencyArea, float]  # 0-1 scale
    specific_feedback: List[str]
    strengths_identified: List[str]
    areas_for_improvement: List[str]
    overall_score: float
    assessment_timestamp: datetime

class ScenarioGenerator:
    """Generate realistic training scenarios"""
    
    def __init__(self):
        # Sample client profiles for training scenarios
        self.client_profiles = {
            "anxiety_college_student": {
                "age": 20,
                "gender": "female",
                "background": "College junior, first time in therapy",
                "presenting_problem": "Severe test anxiety affecting academic performance",
                "personality": "Perfectionist, people-pleaser, high achiever",
                "cultural_factors": "First-generation college student, family pressure",
                "session_goals": ["Reduce test anxiety", "Develop coping strategies", "Address perfectionism"]
            },
            "depression_middle_aged": {
                "age": 45,
                "gender": "male", 
                "background": "Recently divorced, works in finance",
                "presenting_problem": "Depression following divorce and job stress",
                "personality": "Previously outgoing, now withdrawn and irritable",
                "cultural_factors": "Traditional masculine values, difficulty expressing emotions",
                "session_goals": ["Process divorce grief", "Develop emotional awareness", "Rebuild social connections"]
            },
            "trauma_survivor": {
                "age": 28,
                "gender": "non-binary",
                "background": "Childhood abuse survivor, works as teacher",
                "presenting_problem": "PTSD symptoms interfering with daily functioning",
                "personality": "Hypervigilant, difficulty trusting others",
                "cultural_factors": "LGBTQ+ identity, religious family conflict",
                "session_goals": ["Process traumatic memories", "Develop safety skills", "Rebuild trust capacity"]
            }
        }
        
        # Scenario templates for different training focuses
        self.scenario_templates = {
            ScenarioType.INITIAL_ASSESSMENT: {
                "structure": [
                    "Greeting and rapport building",
                    "Informed consent process", 
                    "Presenting problem exploration",
                    "History gathering",
                    "Mental status exam",
                    "Risk assessment",
                    "Treatment planning discussion"
                ],
                "key_skills": ["Assessment", "Rapport building", "Risk evaluation", "Treatment planning"],
                "common_challenges": ["Client reluctance", "Information gathering", "Risk factors"]
            },
            ScenarioType.CRISIS_INTERVENTION: {
                "structure": [
                    "Crisis assessment",
                    "Safety evaluation", 
                    "De-escalation techniques",
                    "Resource mobilization",
                    "Safety planning",
                    "Follow-up arrangements"
                ],
                "key_skills": ["Crisis assessment", "De-escalation", "Safety planning", "Resource coordination"],
                "common_challenges": ["High emotions", "Safety concerns", "Time pressure"]
            }
        }
    
    def generate_scenario(self, scenario_type: ScenarioType, difficulty: int, 
                         learning_objectives: List[str]) -> TrainingScenario:
        """Generate a training scenario based on parameters"""
        
        # Select appropriate client profile
        profile_key = random.choice(list(self.client_profiles.keys()))
        client_profile = self.client_profiles[profile_key].copy()
        
        # Adjust difficulty based on level
        if difficulty >= 4:
            # Add complicating factors for advanced scenarios
            client_profile["complicating_factors"] = [
                "Comorbid substance use",
                "Family conflict",
                "Financial stressors",
                "Medical complications"
            ]
        
        scenario_id = f"{scenario_type.value}_{difficulty}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return TrainingScenario(
            scenario_id=scenario_id,
            title=f"{scenario_type.value.replace('_', ' ').title()} - Level {difficulty}",
            scenario_type=scenario_type,
            difficulty_level=difficulty,
            learning_objectives=learning_objectives,
            client_background=client_profile,
            presenting_problem=client_profile["presenting_problem"],
            session_context={
                "session_number": 1 if scenario_type == ScenarioType.INITIAL_ASSESSMENT else random.randint(2, 8),
                "setting": "Outpatient therapy office",
                "time_constraints": "50-minute session"
            },
            assessment_rubric=self._create_assessment_rubric(scenario_type, learning_objectives),
            expected_duration_minutes=50
        )