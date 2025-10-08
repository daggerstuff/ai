#!/usr/bin/env python3
"""
Therapeutic AI Simulation Engine
The core engine that brings difficult clients to life using the trained H100 therapeutic AI.

This system transforms client profiles into realistic, challenging therapeutic interactions
for comprehensive therapist training and evaluation.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
import random

from pixelated_empathy_core import (
    DifficultClientProfile, TrainingSession, ClientPersonality, 
    DifficultyLevel, SessionObjective, SupervisorEvaluation
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SimulationState:
    """Current state of the therapeutic simulation"""
    session_id: str
    current_mood: str
    resistance_level: float  # 0-1 scale
    trust_level: float  # 0-1 scale
    emotional_intensity: float  # 0-1 scale
    session_phase: str  # opening, exploration, working, closing
    breakthrough_opportunity: bool
    crisis_risk_level: float  # 0-1 scale
    last_therapist_intervention: str
    client_response_history: List[str] = field(default_factory=list)
    therapist_skill_demonstrations: List[str] = field(default_factory=list)
    red_flags_triggered: List[str] = field(default_factory=list)

@dataclass
class TherapistResponse:
    """Therapist's response to analyze and respond to"""
    content: str
    timestamp: datetime
    intervention_type: str  # question, reflection, interpretation, etc.
    skill_level: str  # novice, developing, proficient, advanced
    therapeutic_approach: str  # CBT, psychodynamic, etc.

@dataclass
class ClientSimulationResponse:
    """AI client's response with metadata"""
    content: str
    emotional_state: str
    resistance_level: float
    therapeutic_progress: float
    supervisor_notes: List[str]
    skill_feedback: Dict[str, float]
    red_flags: List[str]
    breakthrough_achieved: bool
    next_challenge_level: float

class TherapeuticSimulationEngine:
    """Core engine for realistic therapeutic AI simulation"""
    
    def __init__(self, therapeutic_ai_model=None):
        self.therapeutic_ai = therapeutic_ai_model
        self.active_simulations = {}
        
        # Therapeutic skill assessment patterns
        self.skill_patterns = {
            "rapport_building": [
                r"(i understand|that sounds|tell me more)",
                r"(you seem|it appears|i notice)",
                r"(thank you for sharing|i appreciate)",
            ],
            "active_listening": [
                r"(what i'm hearing|it sounds like|you're saying)",
                r"(help me understand|can you elaborate)",
                r"(correct me if|am i understanding)",
            ],
            "empathy": [
                r"(that must be|i can imagine|that sounds)",
                r"(difficult|challenging|painful|overwhelming)",
                r"(you're feeling|you must feel|that's)",
            ],
            "boundary_setting": [
                r"(our time|session time|we have)",
                r"(not appropriate|boundaries|professional)",
                r"(i can't|unable to|outside my role)",
            ],
            "crisis_assessment": [
                r"(thoughts of|harm|safety|suicide)",
                r"(safe|danger|hurt yourself|plan)",
                r"(emergency|crisis|immediate|support)",
            ],
            "resistance_handling": [
                r"(seems difficult|reluctant|hesitant)",
                r"(part of you|mixed feelings|ambivalent)",
                r"(resistance|pushback|defensive)",
            ]
        }
        
        # Response calibration based on client personality
        self.personality_response_modifiers = {
            ClientPersonality.RESISTANT: {
                "trust_building_time": 0.1,
                "resistance_decay": 0.05,
                "breakthrough_threshold": 0.8,
                "crisis_escalation": 0.1
            },
            ClientPersonality.HOSTILE_AGGRESSIVE: {
                "trust_building_time": 0.2,
                "resistance_decay": 0.03,
                "breakthrough_threshold": 0.7,
                "crisis_escalation": 0.3
            },
            ClientPersonality.BORDERLINE_TRAITS: {
                "trust_building_time": 0.3,
                "resistance_decay": 0.08,
                "breakthrough_threshold": 0.6,
                "crisis_escalation": 0.4
            },
            ClientPersonality.SUICIDAL_IDEATION: {
                "trust_building_time": 0.4,
                "resistance_decay": 0.06,
                "breakthrough_threshold": 0.9,
                "crisis_escalation": 0.8
            }
        }
    
    async def start_simulation(self, client_profile: DifficultClientProfile, 
                             trainee_id: str, supervisor_id: str = None) -> TrainingSession:
        """Initialize a new therapeutic simulation session"""
        
        session_id = f"sim_{trainee_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create initial simulation state
        initial_state = SimulationState(
            session_id=session_id,
            current_mood=self._determine_initial_mood(client_profile),
            resistance_level=self._calculate_initial_resistance(client_profile),
            trust_level=0.1,  # Start with minimal trust
            emotional_intensity=self._calculate_initial_emotional_intensity(client_profile),
            session_phase="opening",
            breakthrough_opportunity=False,
            crisis_risk_level=self._assess_initial_crisis_risk(client_profile),
            last_therapist_intervention="session_start"
        )
        
        # Create training session
        training_session = TrainingSession(
            session_id=session_id,
            trainee_id=trainee_id,
            supervisor_id=supervisor_id,
            client_profile=client_profile,
            session_objectives=client_profile.learning_objectives,
            difficulty_level=client_profile.difficulty_level,
            start_time=datetime.now(),
            estimated_duration=50,  # Standard therapy session
            current_phase="opening"
        )
        
        # Store active simulation
        self.active_simulations[session_id] = {
            "session": training_session,
            "state": initial_state,
            "client_profile": client_profile
        }
        
        logger.info(f"Started simulation {session_id} with {client_profile.personality_type.value} client")
        
        return training_session
    
    async def process_therapist_response(self, session_id: str, 
                                       therapist_input: str,
                                       intervention_type: str = "unknown") -> ClientSimulationResponse:
        """Process therapist input and generate realistic client response"""
        
        if session_id not in self.active_simulations:
            raise ValueError(f"Simulation {session_id} not found")
        
        simulation = self.active_simulations[session_id]
        state = simulation["state"]
        client_profile = simulation["client_profile"]
        
        # Analyze therapist response for skills and quality
        therapist_analysis = self._analyze_therapist_response(therapist_input, client_profile, state)
        
        # Update simulation state based on therapist response
        self._update_simulation_state(state, client_profile, therapist_analysis)
        
        # Generate client response using therapeutic AI
        client_response = await self._generate_client_response(
            therapist_input, client_profile, state, therapist_analysis
        )
        
        # Assess therapeutic progress and provide supervisor feedback
        supervisor_feedback = self._generate_supervisor_feedback(
            therapist_analysis, client_response, client_profile, state
        )
        
        # Update session history
        state.client_response_history.append(client_response.content)
        state.last_therapist_intervention = therapist_input
        
        # Check for session phase transitions
        self._check_phase_transitions(state, client_profile)
        
        return ClientSimulationResponse(
            content=client_response.content,
            emotional_state=state.current_mood,
            resistance_level=state.resistance_level,
            therapeutic_progress=state.trust_level,
            supervisor_notes=supervisor_feedback["notes"],
            skill_feedback=supervisor_feedback["skills"],
            red_flags=state.red_flags_triggered,
            breakthrough_achieved=state.breakthrough_opportunity,
            next_challenge_level=self._calculate_next_challenge_level(state, client_profile)
        )
    
    def _determine_initial_mood(self, client_profile: DifficultClientProfile) -> str:
        """Determine client's initial emotional presentation"""
        mood_map = {
            ClientPersonality.RESISTANT: "defensive_closed",
            ClientPersonality.HOSTILE_AGGRESSIVE: "angry_irritated", 
            ClientPersonality.BORDERLINE_TRAITS: "anxious_unstable",
            ClientPersonality.NARCISSISTIC_TRAITS: "superior_dismissive",
            ClientPersonality.SUICIDAL_IDEATION: "hopeless_flat",
            ClientPersonality.TRAUMA_REACTIVE: "guarded_hypervigilant"
        }
        return mood_map.get(client_profile.personality_type, "neutral_cautious")
    
    def _calculate_initial_resistance(self, client_profile: DifficultClientProfile) -> float:
        """Calculate initial resistance level based on personality and difficulty"""
        base_resistance = {
            ClientPersonality.RESISTANT: 0.9,
            ClientPersonality.HOSTILE_AGGRESSIVE: 0.8,
            ClientPersonality.NARCISSISTIC_TRAITS: 0.7,
            ClientPersonality.BORDERLINE_TRAITS: 0.6,
            ClientPersonality.TRAUMA_REACTIVE: 0.8,
            ClientPersonality.SUICIDAL_IDEATION: 0.5
        }
        
        base = base_resistance.get(client_profile.personality_type, 0.6)
        difficulty_modifier = client_profile.difficulty_level.value * 0.1
        
        return min(1.0, base + difficulty_modifier)
    
    def _calculate_initial_emotional_intensity(self, client_profile: DifficultClientProfile) -> float:
        """Calculate initial emotional intensity"""
        intensity_map = {
            ClientPersonality.HOSTILE_AGGRESSIVE: 0.8,
            ClientPersonality.BORDERLINE_TRAITS: 0.9,
            ClientPersonality.SUICIDAL_IDEATION: 0.7,
            ClientPersonality.TRAUMA_REACTIVE: 0.6,
            ClientPersonality.RESISTANT: 0.3,
            ClientPersonality.NARCISSISTIC_TRAITS: 0.4
        }
        return intensity_map.get(client_profile.personality_type, 0.5)
    
    def _assess_initial_crisis_risk(self, client_profile: DifficultClientProfile) -> float:
        """Assess initial crisis risk level"""
        if client_profile.personality_type == ClientPersonality.SUICIDAL_IDEATION:
            return 0.8
        elif client_profile.personality_type == ClientPersonality.BORDERLINE_TRAITS:
            return 0.6
        elif client_profile.personality_type == ClientPersonality.HOSTILE_AGGRESSIVE:
            return 0.4
        else:
            return 0.2
    
    def _analyze_therapist_response(self, therapist_input: str, 
                                  client_profile: DifficultClientProfile,
                                  state: SimulationState) -> Dict:
        """Analyze therapist response for therapeutic skills and appropriateness"""
        
        analysis = {
            "skills_demonstrated": {},
            "therapeutic_quality": 0.0,
            "appropriateness": 0.0,
            "timing": 0.0,
            "mistakes": [],
            "strengths": [],
            "impact_prediction": {}
        }
        
        input_lower = therapist_input.lower()
        
        # Assess specific therapeutic skills
        for skill, patterns in self.skill_patterns.items():
            skill_score = 0.0
            for pattern in patterns:
                if re.search(pattern, input_lower):
                    skill_score += 0.3
            analysis["skills_demonstrated"][skill] = min(1.0, skill_score)
        
        # Check for common mistakes with this client type
        mistakes = self._identify_therapist_mistakes(therapist_input, client_profile, state)
        analysis["mistakes"] = mistakes
        
        # Assess appropriateness for current session phase and client state
        appropriateness = self._assess_intervention_appropriateness(
            therapist_input, client_profile, state
        )
        analysis["appropriateness"] = appropriateness
        
        # Predict impact on client state
        impact = self._predict_client_response_impact(therapist_input, client_profile, state)
        analysis["impact_prediction"] = impact
        
        # Overall therapeutic quality score
        quality_factors = [
            analysis["appropriateness"],
            sum(analysis["skills_demonstrated"].values()) / len(analysis["skills_demonstrated"]),
            1.0 - (len(mistakes) * 0.2)  # Penalty for mistakes
        ]
        analysis["therapeutic_quality"] = max(0.0, sum(quality_factors) / len(quality_factors))
        
        return analysis
    
    def _identify_therapist_mistakes(self, therapist_input: str, 
                                   client_profile: DifficultClientProfile,
                                   state: SimulationState) -> List[str]:
        """Identify common therapeutic mistakes for this client type"""
        
        mistakes = []
        input_lower = therapist_input.lower()
        
        # Universal mistakes
        if len(therapist_input.split()) > 50:
            mistakes.append("Response too long - may overwhelm client")
        
        if "?" in therapist_input and therapist_input.count("?") > 2:
            mistakes.append("Too many questions at once")
        
        # Personality-specific mistakes
        if client_profile.personality_type == ClientPersonality.RESISTANT:
            if any(word in input_lower for word in ["should", "must", "need to"]):
                mistakes.append("Being overly directive with resistant client")
            if re.search(r"why don't you|you should", input_lower):
                mistakes.append("Pushing solutions too early")
        
        elif client_profile.personality_type == ClientPersonality.HOSTILE_AGGRESSIVE:
            if any(word in input_lower for word in ["calm down", "relax", "don't be"]):
                mistakes.append("Minimizing or dismissing client's anger")
            if re.search(r"that's not|you're wrong", input_lower):
                mistakes.append("Being confrontational with aggressive client")
        
        elif client_profile.personality_type == ClientPersonality.BORDERLINE_TRAITS:
            if state.emotional_intensity > 0.7 and "homework" in input_lower:
                mistakes.append("Giving homework during emotional crisis")
            if "can't see you" in input_lower or "no extra" in input_lower:
                mistakes.append("Rigid boundaries during emotional crisis")
        
        elif client_profile.personality_type == ClientPersonality.SUICIDAL_IDEATION:
            if state.crisis_risk_level > 0.6 and "positive" in input_lower:
                mistakes.append("Toxic positivity during suicidal crisis")
            if not any(word in input_lower for word in ["safe", "safety", "harm", "plan"]):
                if "hurt" in state.client_response_history[-1] if state.client_response_history else False:
                    mistakes.append("Not addressing safety concerns")
        
        return mistakes
    
    def _assess_intervention_appropriateness(self, therapist_input: str,
                                           client_profile: DifficultClientProfile,
                                           state: SimulationState) -> float:
        """Assess how appropriate the intervention is for current situation"""
        
        appropriateness = 0.5  # Base score
        input_lower = therapist_input.lower()
        
        # Phase-appropriate interventions
        if state.session_phase == "opening":
            if any(word in input_lower for word in ["welcome", "glad", "thank you"]):
                appropriateness += 0.2
            if any(word in input_lower for word in ["homework", "change", "goals"]):
                appropriateness -= 0.3  # Too early for goals
        
        elif state.session_phase == "exploration":
            if any(word in input_lower for word in ["tell me", "what", "how", "when"]):
                appropriateness += 0.2
            if any(word in input_lower for word in ["solution", "fix", "should"]):
                appropriateness -= 0.2  # Too early for solutions
        
        # Crisis-appropriate responses
        if state.crisis_risk_level > 0.6:
            if any(word in input_lower for word in ["safe", "safety", "support", "help"]):
                appropriateness += 0.3
            if any(word in input_lower for word in ["homework", "next week", "goals"]):
                appropriateness -= 0.4  # Inappropriate during crisis
        
        # Resistance-appropriate responses
        if state.resistance_level > 0.7:
            if any(word in input_lower for word in ["understand", "seems", "appears"]):
                appropriateness += 0.2
            if any(word in input_lower for word in ["must", "should", "need to"]):
                appropriateness -= 0.3  # Increases resistance
        
        return max(0.0, min(1.0, appropriateness))
    
    def _predict_client_response_impact(self, therapist_input: str,
                                      client_profile: DifficultClientProfile,
                                      state: SimulationState) -> Dict:
        """Predict impact of therapist response on client state"""
        
        impact = {
            "trust_change": 0.0,
            "resistance_change": 0.0,
            "emotional_intensity_change": 0.0,
            "crisis_risk_change": 0.0,
            "breakthrough_probability": 0.0
        }
        
        input_lower = therapist_input.lower()
        
        # Trust impact
        if any(word in input_lower for word in ["understand", "hear", "sounds like"]):
            impact["trust_change"] += 0.1
        if any(word in input_lower for word in ["wrong", "shouldn't", "that's not"]):
            impact["trust_change"] -= 0.2
        
        # Resistance impact
        if any(word in input_lower for word in ["choice", "up to you", "when you're ready"]):
            impact["resistance_change"] -= 0.1  # Reduces resistance
        if any(word in input_lower for word in ["must", "should", "have to"]):
            impact["resistance_change"] += 0.2  # Increases resistance
        
        # Emotional intensity impact
        personality_modifiers = self.personality_response_modifiers.get(
            client_profile.personality_type, {}
        )
        
        if client_profile.personality_type == ClientPersonality.BORDERLINE_TRAITS:
            if "time is up" in input_lower or "end" in input_lower:
                impact["emotional_intensity_change"] += 0.3  # Abandonment trigger
        
        return impact
    
    def _update_simulation_state(self, state: SimulationState,
                               client_profile: DifficultClientProfile,
                               therapist_analysis: Dict):
        """Update simulation state based on therapist intervention"""
        
        impact = therapist_analysis["impact_prediction"]
        
        # Update trust level
        trust_change = impact.get("trust_change", 0.0)
        if therapist_analysis["therapeutic_quality"] > 0.7:
            trust_change += 0.05  # Bonus for high quality interventions
        state.trust_level = max(0.0, min(1.0, state.trust_level + trust_change))
        
        # Update resistance level
        resistance_change = impact.get("resistance_change", 0.0)
        if therapist_analysis["mistakes"]:
            resistance_change += len(therapist_analysis["mistakes"]) * 0.1
        state.resistance_level = max(0.0, min(1.0, state.resistance_level + resistance_change))
        
        # Update emotional intensity
        intensity_change = impact.get("emotional_intensity_change", 0.0)
        state.emotional_intensity = max(0.0, min(1.0, state.emotional_intensity + intensity_change))
        
        # Update crisis risk
        crisis_change = impact.get("crisis_risk_change", 0.0)
        if client_profile.personality_type == ClientPersonality.SUICIDAL_IDEATION:
            if state.trust_level > 0.6 and "safe" in state.last_therapist_intervention.lower():
                crisis_change -= 0.1
        state.crisis_risk_level = max(0.0, min(1.0, state.crisis_risk_level + crisis_change))
        
        # Check for breakthrough opportunities
        breakthrough_threshold = self.personality_response_modifiers.get(
            client_profile.personality_type, {}
        ).get("breakthrough_threshold", 0.7)
        
        if (state.trust_level > breakthrough_threshold and 
            state.resistance_level < 0.4 and
            therapist_analysis["therapeutic_quality"] > 0.8):
            state.breakthrough_opportunity = True
        
        # Track skill demonstrations
        for skill, score in therapist_analysis["skills_demonstrated"].items():
            if score > 0.5:
                state.therapist_skill_demonstrations.append(skill)
        
        # Track red flags
        if therapist_analysis["mistakes"]:
            state.red_flags_triggered.extend(therapist_analysis["mistakes"])
    
    async def _generate_client_response(self, therapist_input: str,
                                      client_profile: DifficultClientProfile,
                                      state: SimulationState,
                                      therapist_analysis: Dict) -> Any:
        """Generate realistic client response using therapeutic AI"""
        
        # Prepare context for therapeutic AI
        ai_context = {
            "client_personality": client_profile.personality_type.value,
            "current_mood": state.current_mood,
            "resistance_level": state.resistance_level,
            "trust_level": state.trust_level,
            "emotional_intensity": state.emotional_intensity,
            "session_phase": state.session_phase,
            "therapist_quality": therapist_analysis["therapeutic_quality"],
            "breakthrough_opportunity": state.breakthrough_opportunity,
            "crisis_risk": state.crisis_risk_level
        }
        
        if self.therapeutic_ai:
            # Use trained H100 model with expert routing
            expert_preference = self._determine_expert_routing(client_profile, state)
            ai_response = await self._call_therapeutic_ai(
                therapist_input, ai_context, expert_preference, client_profile
            )
        else:
            # Fallback to rule-based response generation
            ai_response = self._generate_rule_based_response(
                therapist_input, client_profile, state, therapist_analysis
            )
        
        return type('ClientResponse', (), {'content': ai_response})
    
    def _determine_expert_routing(self, client_profile: DifficultClientProfile,
                                state: SimulationState) -> str:
        """Determine which AI expert to use for response generation"""
        
        if state.crisis_risk_level > 0.6:
            return "empathetic + therapeutic"
        elif state.resistance_level > 0.7:
            return "therapeutic + practical"
        elif state.breakthrough_opportunity:
            return "empathetic + educational"
        else:
            return "therapeutic"
    
    async def _call_therapeutic_ai(self, therapist_input: str, context: Dict,
                                 expert_preference: str,
                                 client_profile: DifficultClientProfile) -> str:
        """Call the trained therapeutic AI model for response generation"""
        
        # Construct prompt for AI model
        ai_prompt = f"""
        You are role-playing as a {client_profile.personality_type.value} client in therapy.
        
        Client Background: {client_profile.presenting_problem}
        Current State: {context['current_mood']}, resistance={context['resistance_level']:.1f}, trust={context['trust_level']:.1f}
        
        Therapist said: "{therapist_input}"
        
        Respond as this client would, maintaining consistency with their personality and current emotional state.
        """
        
        # In production, this would call the actual H100 model
        # For now, return a placeholder that would be replaced with real AI call
        return "AI therapeutic response would be generated here using trained H100 model"
    
    def _generate_rule_based_response(self, therapist_input: str,
                                    client_profile: DifficultClientProfile,
                                    state: SimulationState,
                                    therapist_analysis: Dict) -> str:
        """Generate rule-based client response as fallback"""
        
        # Select appropriate response pattern based on state and personality
        response_patterns = client_profile.response_patterns
        
        if state.breakthrough_opportunity and state.trust_level > 0.7:
            # Breakthrough response
            if "breakthrough" in response_patterns:
                responses = response_patterns["breakthrough"]
            else:
                responses = ["Maybe there's something to this", "I hadn't thought of it that way"]
        
        elif state.resistance_level > 0.7 or therapist_analysis["mistakes"]:
            # Resistant/defensive response
            if "resistance" in response_patterns:
                responses = response_patterns["resistance"]
            else:
                responses = ["I don't see the point", "That doesn't help", "You don't understand"]
        
        elif state.crisis_risk_level > 0.6:
            # Crisis response
            if "crisis" in response_patterns:
                responses = response_patterns["crisis"]
            else:
                responses = ["I can't handle this", "What's the point?", "Nothing helps"]
        
        else:
            # Default response based on personality
            if "opening" in response_patterns:
                responses = response_patterns["opening"]
            else:
                responses = ["I guess so", "Maybe", "I don't know"]
        
        return random.choice(responses)
    
    def _generate_supervisor_feedback(self, therapist_analysis: Dict,
                                    client_response: Any,
                                    client_profile: DifficultClientProfile,
                                    state: SimulationState) -> Dict:
        """Generate real-time supervisor feedback"""
        
        feedback = {
            "notes": [],
            "skills": {},
            "recommendations": []
        }
        
        # Skill-specific feedback
        for skill, score in therapist_analysis["skills_demonstrated"].items():
            feedback["skills"][skill] = score
            if score > 0.7:
                feedback["notes"].append(f"Strong {skill.replace('_', ' ')} demonstrated")
            elif score < 0.3:
                feedback["notes"].append(f"Consider improving {skill.replace('_', ' ')}")
        
        # Mistake feedback
        for mistake in therapist_analysis["mistakes"]:
            feedback["notes"].append(f"⚠️ {mistake}")
        
        # Progress feedback
        if state.breakthrough_opportunity:
            feedback["notes"].append("🎯 Breakthrough opportunity - client showing openness")
        
        if state.trust_level > 0.6:
            feedback["notes"].append("✅ Good therapeutic rapport building")
        
        if state.resistance_level > 0.8:
            feedback["notes"].append("🔄 High resistance - consider different approach")
        
        # Crisis feedback
        if state.crisis_risk_level > 0.6:
            feedback["notes"].append("🚨 Monitor crisis risk - consider safety assessment")
        
        return feedback
    
    def _check_phase_transitions(self, state: SimulationState,
                               client_profile: DifficultClientProfile):
        """Check if session should transition to next phase"""
        
        # Simple phase transition logic based on trust and time
        if state.session_phase == "opening" and state.trust_level > 0.3:
            state.session_phase = "exploration"
        elif state.session_phase == "exploration" and state.trust_level > 0.6:
            state.session_phase = "working"
        elif state.session_phase == "working" and state.trust_level > 0.8:
            state.session_phase = "integration"
    
    def _calculate_next_challenge_level(self, state: SimulationState,
                                      client_profile: DifficultClientProfile) -> float:
        """Calculate appropriate challenge level for next interaction"""
        
        base_challenge = client_profile.difficulty_level.value * 0.2
        
        # Adjust based on therapist performance
        if state.breakthrough_opportunity:
            base_challenge += 0.1  # Increase challenge after breakthrough
        
        if len(state.red_flags_triggered) > 2:
            base_challenge -= 0.1  # Reduce challenge if struggling
        
        if state.trust_level > 0.7:
            base_challenge += 0.05  # Can handle more challenge with good rapport
        
        return max(0.1, min(1.0, base_challenge))

def main():
    """Demonstrate therapeutic simulation engine"""
    logger.info("🎭 Therapeutic AI Simulation Engine Demo")
    
    # This would be used in the full Pixelated Empathy platform
    logger.info("Engine ready for integration with Pixelated Empathy training platform")

if __name__ == "__main__":
    main()