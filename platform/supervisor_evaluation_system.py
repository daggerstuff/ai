#!/usr/bin/env python3
"""
Supervisor Evaluation System
Real-time assessment and feedback system for therapeutic training simulations.

This system provides supervisors with comprehensive tools to evaluate trainee
performance and provide structured feedback during AI client role-play sessions.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import statistics

from pixelated_empathy_core import (
    TrainingSession, SupervisorEvaluation, TherapistSkillAssessment,
    SessionObjective, DifficultyLevel
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompetencyLevel(Enum):
    UNSATISFACTORY = 1
    DEVELOPING = 2
    COMPETENT = 3
    PROFICIENT = 4
    EXEMPLARY = 5

class InterventionTiming(Enum):
    IMMEDIATE = "immediate"
    END_OF_PHASE = "end_of_phase"
    SESSION_BREAK = "session_break"
    POST_SESSION = "post_session"

@dataclass
class RealTimeObservation:
    """Real-time supervisor observation during training"""
    timestamp: datetime
    observation_type: str  # skill_demonstration, mistake, breakthrough, concern
    description: str
    skill_area: str
    competency_level: CompetencyLevel
    intervention_needed: InterventionTiming
    notes: str

@dataclass
class SkillRubric:
    """Detailed rubric for evaluating therapeutic skills"""
    skill_name: str
    competency_levels: Dict[CompetencyLevel, str]
    behavioral_indicators: Dict[CompetencyLevel, List[str]]
    common_mistakes: List[str]
    development_recommendations: List[str]

class SupervisorEvaluationEngine:
    """Core engine for real-time supervisor evaluation and feedback"""
    
    def __init__(self):
        self.active_evaluations = {}
        self.skill_rubrics = self._initialize_skill_rubrics()
        
        # Competency thresholds for different training levels
        self.competency_thresholds = {
            "student": {
                "minimum_passing": CompetencyLevel.DEVELOPING,
                "target_level": CompetencyLevel.COMPETENT
            },
            "intern": {
                "minimum_passing": CompetencyLevel.COMPETENT,
                "target_level": CompetencyLevel.PROFICIENT
            },
            "junior_clinician": {
                "minimum_passing": CompetencyLevel.PROFICIENT,
                "target_level": CompetencyLevel.EXEMPLARY
            }
        }
    
    def _initialize_skill_rubrics(self) -> Dict[str, SkillRubric]:
        """Initialize comprehensive skill evaluation rubrics"""
        
        return {
            "rapport_building": SkillRubric(
                skill_name="Rapport Building",
                competency_levels={
                    CompetencyLevel.UNSATISFACTORY: "Struggles to connect; may be awkward or inappropriate",
                    CompetencyLevel.DEVELOPING: "Shows basic warmth but inconsistent connection",
                    CompetencyLevel.COMPETENT: "Establishes good rapport with most clients",
                    CompetencyLevel.PROFICIENT: "Quickly builds strong therapeutic alliance",
                    CompetencyLevel.EXEMPLARY: "Exceptional ability to connect with difficult clients"
                },
                behavioral_indicators={
                    CompetencyLevel.EXEMPLARY: [
                        "Uses client's name appropriately",
                        "Matches client's communication style",
                        "Shows genuine warmth and interest",
                        "Demonstrates cultural sensitivity",
                        "Adapts approach to client's needs"
                    ],
                    CompetencyLevel.PROFICIENT: [
                        "Consistent eye contact and open posture",
                        "Reflects client's emotions accurately",
                        "Uses appropriate self-disclosure",
                        "Shows empathy and understanding"
                    ],
                    CompetencyLevel.COMPETENT: [
                        "Greets client warmly",
                        "Shows basic listening skills",
                        "Responds to client's emotional state"
                    ]
                },
                common_mistakes=[
                    "Too formal or distant approach",
                    "Inappropriate self-disclosure",
                    "Not matching client's energy level",
                    "Cultural insensitivity"
                ],
                development_recommendations=[
                    "Practice active listening techniques",
                    "Study cultural competency",
                    "Work on nonverbal communication",
                    "Develop empathy skills"
                ]
            ),
            
            "crisis_management": SkillRubric(
                skill_name="Crisis Management",
                competency_levels={
                    CompetencyLevel.UNSATISFACTORY: "Misses crisis indicators; inappropriate responses",
                    CompetencyLevel.DEVELOPING: "Recognizes obvious crises but unsure how to respond",
                    CompetencyLevel.COMPETENT: "Conducts basic crisis assessment and safety planning",
                    CompetencyLevel.PROFICIENT: "Skilled crisis intervention with appropriate resources",
                    CompetencyLevel.EXEMPLARY: "Expert crisis management with complex presentations"
                },
                behavioral_indicators={
                    CompetencyLevel.EXEMPLARY: [
                        "Quickly identifies subtle crisis indicators",
                        "Conducts thorough suicide risk assessment",
                        "Develops comprehensive safety plans",
                        "Coordinates with emergency services when needed",
                        "Maintains therapeutic relationship during crisis"
                    ],
                    CompetencyLevel.PROFICIENT: [
                        "Asks direct questions about suicidal thoughts",
                        "Assesses means and plan",
                        "Creates basic safety plan",
                        "Knows when to involve others"
                    ],
                    CompetencyLevel.COMPETENT: [
                        "Recognizes expressions of hopelessness",
                        "Asks about safety",
                        "Documents risk factors"
                    ]
                },
                common_mistakes=[
                    "Avoiding direct questions about suicide",
                    "Not assessing means and plan",
                    "Failing to create safety plan",
                    "Not involving supervisor when appropriate"
                ],
                development_recommendations=[
                    "Practice suicide risk assessment protocols",
                    "Learn safety planning techniques",
                    "Study crisis intervention models",
                    "Know local emergency resources"
                ]
            )
        }
    
    def start_evaluation_session(self, session_id: str, trainee_id: str, 
                                supervisor_id: str, trainee_level: str) -> Dict:
        """Initialize real-time evaluation for training session"""
        
        evaluation_session = {
            "session_id": session_id,
            "trainee_id": trainee_id,
            "supervisor_id": supervisor_id,
            "trainee_level": trainee_level,
            "start_time": datetime.now(),
            "observations": [],
            "skill_ratings": {},
            "intervention_notes": [],
            "overall_assessment": None,
            "thresholds": self.competency_thresholds.get(trainee_level, self.competency_thresholds["student"])
        }
        
        self.active_evaluations[session_id] = evaluation_session
        logger.info(f"Started evaluation session {session_id} for {trainee_level} trainee")
        
        return evaluation_session
    
    def record_observation(self, session_id: str, skill_area: str, 
                          description: str, competency_level: CompetencyLevel,
                          observation_type: str = "skill_demonstration") -> RealTimeObservation:
        """Record real-time supervisor observation"""
        
        if session_id not in self.active_evaluations:
            raise ValueError(f"Evaluation session {session_id} not found")
        
        # Determine if intervention is needed
        evaluation_session = self.active_evaluations[session_id]
        thresholds = evaluation_session["thresholds"]
        
        intervention_timing = InterventionTiming.POST_SESSION
        if competency_level == CompetencyLevel.UNSATISFACTORY:
            intervention_timing = InterventionTiming.IMMEDIATE
        elif observation_type == "mistake" and competency_level <= CompetencyLevel.DEVELOPING:
            intervention_timing = InterventionTiming.SESSION_BREAK
        
        observation = RealTimeObservation(
            timestamp=datetime.now(),
            observation_type=observation_type,
            description=description,
            skill_area=skill_area,
            competency_level=competency_level,
            intervention_needed=intervention_timing,
            notes=""
        )
        
        evaluation_session["observations"].append(observation)
        
        # Update running skill ratings
        if skill_area not in evaluation_session["skill_ratings"]:
            evaluation_session["skill_ratings"][skill_area] = []
        evaluation_session["skill_ratings"][skill_area].append(competency_level.value)
        
        logger.info(f"Recorded {observation_type} in {skill_area}: {competency_level.name}")
        
        return observation