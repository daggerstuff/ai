#!/usr/bin/env python3
"""
Pixelated Empathy - Main Platform Orchestrator
The grand-daddy OG platform that orchestrates the complete therapeutic training experience.

This is the core of Pixelated Empathy: AI role-playing as difficult clients for comprehensive
therapist training and supervisor evaluation.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid

from pixelated_empathy_core import (
    DifficultClientGenerator, DifficultClientProfile, TrainingSession,
    ClientPersonality, DifficultyLevel, SessionObjective
)
from therapeutic_simulation_engine import TherapeuticSimulationEngine
from supervisor_evaluation_system import SupervisorEvaluationEngine, CompetencyLevel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingProgram:
    """Complete training program configuration"""
    program_id: str
    program_name: str
    target_competency_level: str
    required_scenarios: List[Dict]
    assessment_criteria: Dict
    completion_requirements: Dict

@dataclass
class TraineeProfile:
    """Comprehensive trainee profile"""
    trainee_id: str
    name: str
    level: str  # student, intern, junior_clinician
    specialization: str
    completed_scenarios: List[str]
    skill_assessments: Dict
    current_competencies: Dict
    development_goals: List[str]

class PixelatedEmpathyPlatform:
    """Main platform orchestrating the complete therapeutic training experience"""
    
    def __init__(self, therapeutic_ai_model=None):
        self.client_generator = DifficultClientGenerator()
        self.simulation_engine = TherapeuticSimulationEngine(therapeutic_ai_model)
        self.evaluation_engine = SupervisorEvaluationEngine()
        
        self.active_sessions = {}
        self.trainee_profiles = {}
        self.training_programs = {}
        
        # Initialize default training programs
        self._initialize_training_programs()
        
        logger.info("🎭 Pixelated Empathy Platform initialized")
    
    def _initialize_training_programs(self):
        """Initialize standard training programs"""
        
        # Basic Therapeutic Skills Program
        basic_program = TrainingProgram(
            program_id="basic_therapeutic_skills",
            program_name="Basic Therapeutic Skills",
            target_competency_level="competent",
            required_scenarios=[
                {
                    "personality_type": ClientPersonality.RESISTANT.value,
                    "difficulty": DifficultyLevel.BEGINNER.value,
                    "objectives": [SessionObjective.RAPPORT_BUILDING.value]
                },
                {
                    "personality_type": ClientPersonality.ANXIOUS_AVOIDANT.value,
                    "difficulty": DifficultyLevel.BEGINNER.value,
                    "objectives": [SessionObjective.RAPPORT_BUILDING.value]
                }
            ],
            assessment_criteria={
                "minimum_passing_score": 3.0,
                "required_competencies": ["rapport_building", "active_listening", "empathy"]
            },
            completion_requirements={
                "scenarios_completed": 5,
                "supervisor_approval": True,
                "final_assessment_score": 3.5
            }
        )
        
        # Crisis Intervention Specialist Program
        crisis_program = TrainingProgram(
            program_id="crisis_intervention_specialist",
            program_name="Crisis Intervention Specialist",
            target_competency_level="proficient",
            required_scenarios=[
                {
                    "personality_type": ClientPersonality.SUICIDAL_IDEATION.value,
                    "difficulty": DifficultyLevel.INTERMEDIATE.value,
                    "objectives": [SessionObjective.CRISIS_INTERVENTION.value, SessionObjective.SAFETY_ASSESSMENT.value]
                },
                {
                    "personality_type": ClientPersonality.BORDERLINE_TRAITS.value,
                    "difficulty": DifficultyLevel.ADVANCED.value,
                    "objectives": [SessionObjective.CRISIS_INTERVENTION.value, SessionObjective.BOUNDARY_SETTING.value]
                }
            ],
            assessment_criteria={
                "minimum_passing_score": 4.0,
                "required_competencies": ["crisis_management", "safety_assessment", "boundary_setting"]
            },
            completion_requirements={
                "scenarios_completed": 8,
                "supervisor_approval": True,
                "final_assessment_score": 4.2
            }
        )
        
        self.training_programs = {
            "basic_therapeutic_skills": basic_program,
            "crisis_intervention_specialist": crisis_program
        }
    
    async def create_training_session(self, trainee_id: str, supervisor_id: str,
                                    personality_type: ClientPersonality,
                                    difficulty_level: DifficultyLevel,
                                    learning_objectives: List[SessionObjective],
                                    custom_profile: Dict = None) -> Dict:
        """Create a new therapeutic training session"""
        
        logger.info(f"Creating training session: {personality_type.value} level {difficulty_level.value}")
        
        # Generate or use custom client profile
        if custom_profile:
            client_profile = DifficultClientProfile(**custom_profile)
        else:
            client_profile = self.client_generator.generate_difficult_client(
                personality_type, difficulty_level, learning_objectives
            )
        
        # Start simulation
        training_session = await self.simulation_engine.start_simulation(
            client_profile, trainee_id, supervisor_id
        )
        
        # Start evaluation
        evaluation_session = self.evaluation_engine.start_evaluation_session(
            training_session.session_id, trainee_id, supervisor_id,
            self._get_trainee_level(trainee_id)
        )
        
        # Package session information
        session_info = {
            "session_id": training_session.session_id,
            "client_profile": {
                "name": client_profile.name,
                "personality_type": client_profile.personality_type.value,
                "difficulty_level": client_profile.difficulty_level.value,
                "presenting_problem": client_profile.presenting_problem,
                "key_challenges": client_profile.therapeutic_challenges,
                "learning_objectives": [obj.value for obj in client_profile.learning_objectives]
            },
            "training_info": {
                "trainee_id": trainee_id,
                "supervisor_id": supervisor_id,
                "start_time": training_session.start_time.isoformat(),
                "estimated_duration": training_session.estimated_duration
            },
            "evaluation_framework": {
                "skill_areas": list(self.evaluation_engine.skill_rubrics.keys()),
                "competency_levels": [level.name for level in CompetencyLevel],
                "target_competencies": evaluation_session["thresholds"]
            }
        }
        
        # Store active session
        self.active_sessions[training_session.session_id] = {
            "training_session": training_session,
            "client_profile": client_profile,
            "evaluation_session": evaluation_session,
            "session_log": []
        }
        
        logger.info(f"✅ Training session {training_session.session_id} created successfully")
        
        return session_info
    
    async def process_training_interaction(self, session_id: str, therapist_input: str,
                                         supervisor_observations: List[Dict] = None) -> Dict:
        """Process a single training interaction (therapist input + AI client response)"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Training session {session_id} not found")
        
        session_data = self.active_sessions[session_id]
        
        # Process therapist response through simulation engine
        client_response = await self.simulation_engine.process_therapist_response(
            session_id, therapist_input
        )
        
        # Record supervisor observations if provided
        supervisor_feedback = []
        if supervisor_observations:
            for obs in supervisor_observations:
                observation = self.evaluation_engine.record_observation(
                    session_id=session_id,
                    skill_area=obs["skill_area"],
                    description=obs["description"],
                    competency_level=CompetencyLevel(obs["competency_level"]),
                    observation_type=obs.get("observation_type", "skill_demonstration")
                )
                supervisor_feedback.append(asdict(observation))
        
        # Log interaction
        interaction_log = {
            "timestamp": datetime.now().isoformat(),
            "therapist_input": therapist_input,
            "client_response": client_response.content,
            "client_emotional_state": client_response.emotional_state,
            "resistance_level": client_response.resistance_level,
            "therapeutic_progress": client_response.therapeutic_progress,
            "supervisor_feedback": supervisor_feedback,
            "skill_feedback": client_response.skill_feedback,
            "red_flags": client_response.red_flags,
            "breakthrough_achieved": client_response.breakthrough_achieved
        }
        
        session_data["session_log"].append(interaction_log)
        
        # Prepare response for platform UI
        response = {
            "interaction_id": len(session_data["session_log"]),
            "client_response": {
                "content": client_response.content,
                "emotional_state": client_response.emotional_state,
                "nonverbal_cues": self._generate_nonverbal_cues(client_response),
                "difficulty_adjustment": client_response.next_challenge_level
            },
            "real_time_feedback": {
                "therapeutic_progress": client_response.therapeutic_progress,
                "resistance_level": client_response.resistance_level,
                "supervisor_notes": client_response.supervisor_notes,
                "skill_demonstrations": client_response.skill_feedback,
                "areas_for_improvement": client_response.red_flags
            },
            "session_status": {
                "breakthrough_achieved": client_response.breakthrough_achieved,
                "crisis_indicators": [flag for flag in client_response.red_flags if "crisis" in flag.lower()],
                "intervention_suggestions": self._generate_intervention_suggestions(client_response),
                "session_phase": session_data["training_session"].current_phase
            }
        }
        
        logger.info(f"Processed interaction {interaction_log['timestamp']} for session {session_id}")
        
        return response
    
    def get_supervisor_dashboard(self, session_id: str) -> Dict:
        """Generate comprehensive supervisor dashboard"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Training session {session_id} not found")
        
        session_data = self.active_sessions[session_id]
        evaluation_session = session_data["evaluation_session"]
        
        # Calculate current skill ratings
        skill_summary = {}
        for skill, ratings in evaluation_session["skill_ratings"].items():
            if ratings:
                skill_summary[skill] = {
                    "current_average": statistics.mean(ratings),
                    "trend": "improving" if len(ratings) > 1 and ratings[-1] > ratings[0] else "stable",
                    "total_observations": len(ratings),
                    "target_level": evaluation_session["thresholds"]["target_level"].value
                }
        
        # Generate development recommendations
        recommendations = self._generate_development_recommendations(evaluation_session)
        
        # Session timeline
        timeline = []
        for log_entry in session_data["session_log"]:
            timeline.append({
                "timestamp": log_entry["timestamp"],
                "therapist_action": log_entry["therapist_input"][:100] + "...",
                "client_emotional_state": log_entry["client_emotional_state"],
                "therapeutic_progress": log_entry["therapeutic_progress"],
                "key_feedback": log_entry["supervisor_feedback"][:2] if log_entry["supervisor_feedback"] else []
            })
        
        dashboard = {
            "session_overview": {
                "session_id": session_id,
                "trainee_id": evaluation_session["trainee_id"],
                "trainee_level": evaluation_session["trainee_level"],
                "session_duration": str(datetime.now() - evaluation_session["start_time"]),
                "total_interactions": len(session_data["session_log"])
            },
            "skill_assessment": {
                "current_ratings": skill_summary,
                "overall_competency": self._calculate_overall_competency(evaluation_session),
                "target_thresholds": {
                    "minimum_passing": evaluation_session["thresholds"]["minimum_passing"].value,
                    "target_level": evaluation_session["thresholds"]["target_level"].value
                }
            },
            "real_time_observations": evaluation_session["observations"][-10:],  # Last 10
            "development_recommendations": recommendations,
            "session_timeline": timeline[-20:],  # Last 20 interactions
            "intervention_alerts": self._get_intervention_alerts(evaluation_session)
        }
        
        return dashboard
    
    def complete_training_session(self, session_id: str, supervisor_final_assessment: Dict) -> Dict:
        """Complete training session with final supervisor assessment"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Training session {session_id} not found")
        
        session_data = self.active_sessions[session_id]
        evaluation_session = session_data["evaluation_session"]
        
        # Generate comprehensive final assessment
        final_assessment = self._generate_final_assessment(
            session_data, supervisor_final_assessment
        )
        
        # Update trainee profile
        self._update_trainee_profile(
            evaluation_session["trainee_id"], final_assessment
        )
        
        # Archive session
        self._archive_session(session_id, final_assessment)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        logger.info(f"✅ Training session {session_id} completed")
        
        return final_assessment
    
    def _get_trainee_level(self, trainee_id: str) -> str:
        """Get trainee level from profile"""
        if trainee_id in self.trainee_profiles:
            return self.trainee_profiles[trainee_id].level
        return "student"  # Default
    
    def _generate_nonverbal_cues(self, client_response) -> Dict:
        """Generate nonverbal cues based on client emotional state"""
        cues = {
            "body_language": "Neutral posture",
            "facial_expression": "Neutral expression",
            "voice_tone": "Normal tone",
            "eye_contact": "Appropriate eye contact"
        }
        
        if client_response.resistance_level > 0.7:
            cues.update({
                "body_language": "Crossed arms, leaning away",
                "facial_expression": "Skeptical, frowning",
                "voice_tone": "Defensive, clipped"
            })
        elif client_response.therapeutic_progress > 0.7:
            cues.update({
                "body_language": "Open posture, leaning forward",
                "facial_expression": "Engaged, thoughtful",
                "voice_tone": "Warmer, more open"
            })
        
        return cues
    
    def _generate_intervention_suggestions(self, client_response) -> List[str]:
        """Generate intervention suggestions based on client state"""
        suggestions = []
        
        if client_response.resistance_level > 0.8:
            suggestions.append("Consider reflecting resistance rather than challenging it")
            suggestions.append("Validate client's autonomy and right to feel resistant")
        
        if client_response.breakthrough_achieved:
            suggestions.append("Capitalize on this breakthrough moment")
            suggestions.append("Explore the client's new insight deeper")
        
        if any("crisis" in flag.lower() for flag in client_response.red_flags):
            suggestions.append("⚠️ Immediate crisis assessment needed")
            suggestions.append("Consider safety planning intervention")
        
        return suggestions
    
    def _generate_development_recommendations(self, evaluation_session: Dict) -> List[str]:
        """Generate personalized development recommendations"""
        recommendations = []
        
        # Analyze skill patterns
        for skill, ratings in evaluation_session["skill_ratings"].items():
            if ratings:
                avg_rating = statistics.mean(ratings)
                target = evaluation_session["thresholds"]["target_level"].value
                
                if avg_rating < target:
                    if skill in self.evaluation_engine.skill_rubrics:
                        rubric = self.evaluation_engine.skill_rubrics[skill]
                        recommendations.extend(rubric.development_recommendations[:2])
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _calculate_overall_competency(self, evaluation_session: Dict) -> float:
        """Calculate overall competency score"""
        if not evaluation_session["skill_ratings"]:
            return 0.0
        
        all_ratings = []
        for ratings in evaluation_session["skill_ratings"].values():
            all_ratings.extend(ratings)
        
        return statistics.mean(all_ratings) if all_ratings else 0.0
    
    def _get_intervention_alerts(self, evaluation_session: Dict) -> List[Dict]:
        """Get current intervention alerts"""
        alerts = []
        
        # Check recent observations for immediate interventions
        recent_observations = evaluation_session["observations"][-5:]
        for obs in recent_observations:
            if obs.intervention_needed in ["immediate", "session_break"]:
                alerts.append({
                    "severity": "high" if obs.intervention_needed == "immediate" else "medium",
                    "skill_area": obs.skill_area,
                    "description": obs.description,
                    "recommended_action": f"Intervention needed: {obs.intervention_needed.value}"
                })
        
        return alerts
    
    def _generate_final_assessment(self, session_data: Dict, supervisor_assessment: Dict) -> Dict:
        """Generate comprehensive final assessment"""
        evaluation_session = session_data["evaluation_session"]
        
        # Calculate final skill ratings
        final_skills = {}
        for skill, ratings in evaluation_session["skill_ratings"].items():
            if ratings:
                final_skills[skill] = {
                    "final_rating": statistics.mean(ratings[-3:]) if len(ratings) >= 3 else statistics.mean(ratings),
                    "improvement": ratings[-1] - ratings[0] if len(ratings) > 1 else 0,
                    "consistency": 1.0 - (statistics.stdev(ratings) if len(ratings) > 1 else 0)
                }
        
        return {
            "session_id": session_data["training_session"].session_id,
            "completion_time": datetime.now().isoformat(),
            "final_skill_ratings": final_skills,
            "overall_competency": self._calculate_overall_competency(evaluation_session),
            "supervisor_assessment": supervisor_assessment,
            "session_statistics": {
                "total_interactions": len(session_data["session_log"]),
                "breakthrough_moments": sum(1 for log in session_data["session_log"] if log["breakthrough_achieved"]),
                "crisis_interventions": sum(1 for log in session_data["session_log"] if any("crisis" in flag.lower() for flag in log["red_flags"]))
            },
            "development_plan": self._generate_development_recommendations(evaluation_session)
        }
    
    def _update_trainee_profile(self, trainee_id: str, assessment: Dict):
        """Update trainee profile with assessment results"""
        # Implementation would update trainee's skill assessments and progress
        logger.info(f"Updated trainee profile for {trainee_id}")
    
    def _archive_session(self, session_id: str, final_assessment: Dict):
        """Archive completed session for future reference"""
        # Implementation would save session data to database
        logger.info(f"Archived training session {session_id}")

def main():
    """Demonstrate Pixelated Empathy platform"""
    logger.info("🎭 Pixelated Empathy Platform - Main Training System")
    logger.info("Ready for comprehensive therapeutic training simulations!")

if __name__ == "__main__":
    main()