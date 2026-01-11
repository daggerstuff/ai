"""
Therapeutic Conversation Flow Engine (Tier 2.2)

Manages the natural flow and structure of therapeutic conversations,
ensuring appropriate pacing, intervention timing, and session management.

Key Features:
- Opening, exploration, intervention, and closing protocols
- Therapeutic alliance building and maintenance
- Crisis intervention flow management
- Session structure and timing awareness
- Cultural sensitivity and adaptation
- Memory integration across sessions

Input: Conversation history + client context + expert voice patterns
Output: Structured therapeutic conversation with appropriate flow
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionStage(Enum):
    """Stages of a therapeutic session."""
    OPENING = "opening"
    RAPPORT_BUILDING = "rapport_building"  
    EXPLORATION = "exploration"
    INTERVENTION = "intervention"
    INTEGRATION = "integration"
    CLOSING = "closing"
    CRISIS_INTERVENTION = "crisis_intervention"


class ConversationTone(Enum):
    """Different therapeutic conversation tones."""
    WARM_WELCOMING = "warm_welcoming"
    EMPATHETIC_LISTENING = "empathetic_listening"
    GENTLE_EXPLORATION = "gentle_exploration"
    SUPPORTIVE_CHALLENGING = "supportive_challenging"
    CRISIS_CALM = "crisis_calm"
    CLOSURE_HOPEFUL = "closure_hopeful"


class InterventionTiming(Enum):
    """When to introduce therapeutic interventions."""
    TOO_EARLY = "too_early"
    GOOD_TIMING = "good_timing"
    OVERDUE = "overdue"
    CRISIS_IMMEDIATE = "crisis_immediate"


@dataclass
class ConversationMemory:
    """Memory of previous conversation elements."""
    session_number: int
    key_themes: List[str]
    established_rapport_elements: List[str]
    client_preferences: Dict[str, Any]
    therapeutic_goals: List[str]
    progress_markers: List[str]
    crisis_history: List[Dict[str, Any]]
    effective_interventions: List[str]


@dataclass
class ConversationFlow:
    """Current state and flow of the therapeutic conversation."""
    current_stage: SessionStage
    conversation_history: List[Dict[str, str]]  # role, content, timestamp
    alliance_strength: float  # 0.0-1.0
    session_duration: int  # minutes
    energy_level: str  # high, medium, low, depleted
    client_engagement: str  # engaged, hesitant, resistant, overwhelmed
    therapeutic_momentum: str  # building, stable, stagnant, declining


class TherapeuticConversationFlowEngine:
    """Manages the flow and structure of therapeutic conversations."""
    
    def __init__(self):
        self.session_protocols = self._initialize_session_protocols()
        self.timing_engine = InterventionTimingEngine()
        self.alliance_monitor = TherapeuticAllianceMonitor()
        self.cultural_adapter = CulturalSensitivityAdapter()
        
    def determine_conversation_flow(self, 
                                  conversation_state: ConversationFlow,
                                  client_input: str,
                                  conversation_memory: Optional[ConversationMemory] = None) -> Dict[str, Any]:
        """Determine the optimal conversation flow and response approach."""
        
        # Analyze current conversation state
        flow_analysis = self._analyze_conversation_flow(conversation_state, client_input)
        
        # Determine if stage transition is needed
        stage_recommendation = self._evaluate_stage_transition(conversation_state, flow_analysis)
        
        # Assess intervention timing
        intervention_timing = self.timing_engine.assess_intervention_timing(
            conversation_state, flow_analysis, conversation_memory
        )
        
        # Monitor therapeutic alliance
        alliance_assessment = self.alliance_monitor.assess_alliance_strength(
            conversation_state, client_input, conversation_memory
        )
        
        # Generate flow recommendations
        flow_recommendations = self._generate_flow_recommendations(
            stage_recommendation, intervention_timing, alliance_assessment, flow_analysis
        )
        
        return {
            "current_stage": conversation_state.current_stage.value,
            "recommended_stage": stage_recommendation["recommended_stage"].value,
            "stage_transition_needed": stage_recommendation["transition_needed"],
            "intervention_timing": intervention_timing.value,
            "alliance_strength": alliance_assessment["current_strength"],
            "alliance_trend": alliance_assessment["trend"],
            "flow_recommendations": flow_recommendations,
            "response_guidelines": self._generate_response_guidelines(
                stage_recommendation["recommended_stage"], intervention_timing, alliance_assessment
            )
        }
    
    def _initialize_session_protocols(self) -> Dict[SessionStage, Dict[str, Any]]:
        """Initialize protocols for each session stage."""
        return {
            SessionStage.OPENING: {
                "objectives": ["establish_safety", "build_initial_rapport", "assess_immediate_concerns"],
                "typical_duration": 5,  # minutes
                "key_elements": ["greeting", "confidentiality", "session_structure", "immediate_concerns"],
                "conversation_starters": [
                    "How are you feeling today?",
                    "What's been on your mind since we last spoke?",
                    "What would be most helpful for us to focus on today?",
                    "How has your week been?"
                ],
                "red_flags": ["immediate_crisis", "high_agitation", "dissociation_signs"],
                "expert_approaches": {
                    "Tim Fletcher": "gentle_check_in_with_nervous_system_awareness",
                    "Dr. Ramani": "direct_but_warm_assessment_of_safety",
                    "Dr. Gabor Maté": "compassionate_presence_and_attunement"
                }
            },
            
            SessionStage.RAPPORT_BUILDING: {
                "objectives": ["establish_trust", "understand_client_world", "normalize_therapeutic_process"],
                "typical_duration": 10,
                "key_elements": ["active_listening", "validation", "mirroring", "empathy_building"],
                "conversation_techniques": [
                    "reflective_listening", "validation_statements", "empathy_expressions", 
                    "curiosity_about_experience", "normalizing_responses"
                ],
                "alliance_indicators": ["client_opens_up", "shares_vulnerable_content", "asks_questions"],
                "expert_approaches": {
                    "Tim Fletcher": "trauma_informed_validation_and_psychoeducation",
                    "Dr. Ramani": "reality_validation_and_trust_building",
                    "Dr. Gabor Maté": "deep_listening_and_compassionate_presence"
                }
            },
            
            SessionStage.EXPLORATION: {
                "objectives": ["understand_presenting_concerns", "identify_patterns", "assess_coping_resources"],
                "typical_duration": 20,
                "key_elements": ["open_ended_questions", "pattern_identification", "strength_assessment"],
                "exploration_areas": [
                    "current_symptoms", "relationship_patterns", "family_history", 
                    "trauma_history", "coping_strategies", "support_systems"
                ],
                "intervention_readiness_indicators": [
                    "client_insights", "pattern_recognition", "motivation_for_change"
                ],
                "expert_approaches": {
                    "Tim Fletcher": "nervous_system_informed_exploration_of_trauma_responses",
                    "Dr. Ramani": "pattern_identification_in_relationship_dynamics",
                    "Dr. Gabor Maté": "exploration_of_authenticity_and_emotional_truth"
                }
            },
            
            SessionStage.INTERVENTION: {
                "objectives": ["introduce_therapeutic_techniques", "practice_new_skills", "consolidate_learning"],
                "typical_duration": 15,
                "key_elements": ["skill_teaching", "practice_opportunities", "feedback_loops"],
                "intervention_types": [
                    "psychoeducation", "skill_building", "cognitive_restructuring",
                    "grounding_techniques", "boundary_setting", "communication_skills"
                ],
                "readiness_factors": ["alliance_strength", "client_engagement", "crisis_stability"],
                "expert_approaches": {
                    "Tim Fletcher": "nervous_system_regulation_and_trauma_recovery_tools",
                    "Dr. Ramani": "boundary_setting_and_reality_testing_skills",
                    "Dr. Gabor Maté": "self_compassion_and_authenticity_practices"
                }
            },
            
            SessionStage.INTEGRATION: {
                "objectives": ["consolidate_insights", "plan_between_session_practice", "address_concerns"],
                "typical_duration": 7,
                "key_elements": ["summary", "homework_planning", "obstacle_anticipation"],
                "integration_methods": [
                    "insight_summarization", "skill_practice_planning", "support_resource_identification"
                ],
                "expert_approaches": {
                    "Tim Fletcher": "nervous_system_regulation_practice_and_safety_planning",
                    "Dr. Ramani": "boundary_maintenance_and_reality_checking_strategies",
                    "Dr. Gabor Maté": "self_compassion_practice_and_authentic_expression"
                }
            },
            
            SessionStage.CLOSING: {
                "objectives": ["provide_closure", "reinforce_progress", "schedule_follow_up"],
                "typical_duration": 3,
                "key_elements": ["progress_acknowledgment", "hope_instillation", "next_steps"],
                "closing_elements": [
                    "session_summary", "progress_recognition", "between_session_support",
                    "crisis_plan_review", "next_appointment"
                ],
                "expert_approaches": {
                    "Tim Fletcher": "hope_and_nervous_system_healing_possibility",
                    "Dr. Ramani": "empowerment_and_boundary_strength_recognition",
                    "Dr. Gabor Maté": "compassionate_encouragement_and_inner_wisdom_affirmation"
                }
            },
            
            SessionStage.CRISIS_INTERVENTION: {
                "objectives": ["ensure_immediate_safety", "reduce_acute_distress", "develop_safety_plan"],
                "typical_duration": "as_needed",
                "key_elements": ["safety_assessment", "crisis_stabilization", "resource_mobilization"],
                "crisis_protocols": [
                    "immediate_safety_check", "suicide_risk_assessment", "support_system_activation",
                    "professional_referral", "crisis_hotline_connection"
                ],
                "expert_approaches": {
                    "Tim Fletcher": "nervous_system_calming_and_trauma_informed_crisis_care",
                    "Dr. Ramani": "reality_grounding_and_safety_validation",
                    "Dr. Gabor Maté": "compassionate_presence_and_crisis_normalization"
                }
            }
        }
    
    def _analyze_conversation_flow(self, conversation_state: ConversationFlow, client_input: str) -> Dict[str, Any]:
        """Analyze the current conversation flow and client engagement."""
        
        # Analyze client input characteristics
        input_analysis = {
            "length": len(client_input.split()),
            "emotional_intensity": self._assess_emotional_intensity(client_input),
            "vulnerability_level": self._assess_vulnerability_level(client_input),
            "question_present": "?" in client_input,
            "crisis_indicators": self._detect_crisis_language(client_input),
            "engagement_level": self._assess_client_engagement(client_input, conversation_state)
        }
        
        # Assess conversation momentum
        momentum_analysis = {
            "current_momentum": conversation_state.therapeutic_momentum,
            "energy_trend": self._analyze_energy_trend(conversation_state),
            "depth_progression": self._assess_depth_progression(conversation_state),
            "alliance_stability": conversation_state.alliance_strength > 0.6
        }
        
        # Determine conversation needs
        flow_needs = {
            "needs_validation": input_analysis["emotional_intensity"] > 0.7,
            "needs_exploration": input_analysis["vulnerability_level"] > 0.5,
            "needs_intervention": momentum_analysis["depth_progression"] > 0.6,
            "needs_crisis_response": len(input_analysis["crisis_indicators"]) > 0,
            "needs_rapport_building": conversation_state.alliance_strength < 0.5
        }
        
        return {
            "input_analysis": input_analysis,
            "momentum_analysis": momentum_analysis,
            "flow_needs": flow_needs,
            "stage_appropriateness": self._assess_stage_appropriateness(conversation_state, input_analysis)
        }
    
    def _evaluate_stage_transition(self, conversation_state: ConversationFlow, flow_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate whether a stage transition is needed."""
        
        current_stage = conversation_state.current_stage
        recommended_stage = current_stage  # Default to staying in current stage
        transition_needed = False
        
        # Crisis override - always move to crisis intervention if needed
        if flow_analysis["flow_needs"]["needs_crisis_response"]:
            recommended_stage = SessionStage.CRISIS_INTERVENTION
            transition_needed = True
            transition_reason = "crisis_detected"
            return {
                "recommended_stage": recommended_stage,
                "transition_needed": transition_needed,
                "transition_reason": transition_reason,
                "stage_confidence": 1.0  # High confidence in crisis detection
            }
        
        # Normal stage progression logic
        elif current_stage == SessionStage.OPENING:
            if (conversation_state.session_duration > 5 or 
                flow_analysis["flow_needs"]["needs_exploration"]):
                recommended_stage = SessionStage.RAPPORT_BUILDING
                transition_needed = True
                transition_reason = "opening_complete"
        
        elif current_stage == SessionStage.RAPPORT_BUILDING:
            if (conversation_state.alliance_strength > 0.6 and 
                flow_analysis["input_analysis"]["vulnerability_level"] > 0.5):
                recommended_stage = SessionStage.EXPLORATION
                transition_needed = True
                transition_reason = "rapport_established"
        
        elif current_stage == SessionStage.EXPLORATION:
            if (flow_analysis["flow_needs"]["needs_intervention"] and
                conversation_state.alliance_strength > 0.7):
                recommended_stage = SessionStage.INTERVENTION
                transition_needed = True
                transition_reason = "ready_for_intervention"
            elif conversation_state.session_duration > 35:
                recommended_stage = SessionStage.INTEGRATION
                transition_needed = True
                transition_reason = "time_management"
        
        elif current_stage == SessionStage.INTERVENTION:
            if conversation_state.session_duration > 40:
                recommended_stage = SessionStage.INTEGRATION
                transition_needed = True
                transition_reason = "time_management"
        
        elif current_stage == SessionStage.INTEGRATION:
            if conversation_state.session_duration > 47:
                recommended_stage = SessionStage.CLOSING
                transition_needed = True
                transition_reason = "session_ending"
        
        elif current_stage == SessionStage.CRISIS_INTERVENTION:
            if not flow_analysis["flow_needs"]["needs_crisis_response"]:
                recommended_stage = SessionStage.INTEGRATION
                transition_needed = True
                transition_reason = "crisis_stabilized"
        
        return {
            "recommended_stage": recommended_stage,
            "transition_needed": transition_needed,
            "transition_reason": transition_reason if transition_needed else None,
            "stage_confidence": self._calculate_stage_confidence(conversation_state, flow_analysis)
        }
    
    def _generate_flow_recommendations(self, stage_rec: Dict[str, Any], timing: InterventionTiming, alliance: Dict[str, Any], flow: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific flow recommendations."""
        
        recommendations = {
            "conversation_approach": [],
            "intervention_suggestions": [],
            "alliance_building": [],
            "cautions": []
        }
        
        # Stage-specific recommendations
        recommended_stage = stage_rec["recommended_stage"]
        stage_protocol = self.session_protocols[recommended_stage]
        
        recommendations["conversation_approach"] = stage_protocol["key_elements"]
        
        # Alliance-based recommendations
        if alliance["current_strength"] < 0.5:
            recommendations["alliance_building"].extend([
                "increase_validation", "slow_down_exploration", "focus_on_empathy"
            ])
        
        # Timing-based recommendations
        if timing == InterventionTiming.TOO_EARLY:
            recommendations["cautions"].append("avoid_premature_intervention")
        elif timing == InterventionTiming.OVERDUE:
            recommendations["intervention_suggestions"].append("gentle_skill_introduction")
        elif timing == InterventionTiming.CRISIS_IMMEDIATE:
            recommendations["intervention_suggestions"].extend([
                "safety_assessment", "crisis_stabilization", "immediate_support"
            ])
        
        return recommendations
    
    def _generate_response_guidelines(self, stage: SessionStage, timing: InterventionTiming, alliance: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific guidelines for response generation."""
        
        stage_protocol = self.session_protocols[stage]
        
        guidelines = {
            "primary_objectives": stage_protocol["objectives"],
            "conversation_tone": self._select_conversation_tone(stage, alliance),
            "response_length": self._recommend_response_length(stage, timing),
            "question_type": self._recommend_question_type(stage, alliance),
            "intervention_readiness": timing == InterventionTiming.GOOD_TIMING,
            "alliance_focus": alliance["current_strength"] < 0.6,
            "expert_approach_preferences": stage_protocol["expert_approaches"]
        }
        
        return guidelines
    
    # Helper methods
    def _assess_emotional_intensity(self, text: str) -> float:
        """Assess emotional intensity of client input (0.0-1.0)."""
        intensity_indicators = [
            "extremely", "incredibly", "absolutely", "completely", "totally",
            "devastating", "overwhelming", "unbearable", "impossible"
        ]
        text_lower = text.lower()
        intensity_count = sum(1 for indicator in intensity_indicators if indicator in text_lower)
        return min(1.0, intensity_count * 0.3)
    
    def _assess_vulnerability_level(self, text: str) -> float:
        """Assess vulnerability level of client sharing (0.0-1.0)."""
        vulnerability_indicators = [
            "never told anyone", "hard to say", "embarrassed", "ashamed",
            "secret", "scared to admit", "vulnerable", "exposed"
        ]
        text_lower = text.lower()
        vulnerability_count = sum(1 for indicator in vulnerability_indicators if indicator in text_lower)
        return min(1.0, vulnerability_count * 0.4)
    
    def _detect_crisis_language(self, text: str) -> List[str]:
        """Detect crisis language in client input."""
        crisis_patterns = [
            "want to die", "kill myself", "suicide", "end it all", "can't go on",
            "hurt myself", "self harm", "cutting", "overdose", "want to disappear"
        ]
        text_lower = text.lower()
        return [pattern for pattern in crisis_patterns if pattern in text_lower]
    
    def _assess_client_engagement(self, text: str, state: ConversationFlow) -> str:
        """Assess current client engagement level."""
        if len(text.split()) < 5:
            return "minimal"
        elif len(text.split()) > 50:
            return "high"
        else:
            return "moderate"
    
    def _analyze_energy_trend(self, state: ConversationFlow) -> str:
        """Analyze energy trend in conversation."""
        # Simplified - would analyze conversation history
        return "stable"
    
    def _assess_depth_progression(self, state: ConversationFlow) -> float:
        """Assess how deep the conversation has progressed (0.0-1.0)."""
        # Simplified - would analyze vulnerability and insight progression
        return 0.5
    
    def _assess_stage_appropriateness(self, state: ConversationFlow, analysis: Dict[str, Any]) -> float:
        """Assess how appropriate current stage is (0.0-1.0)."""
        return 0.8  # Simplified
    
    def _calculate_stage_confidence(self, state: ConversationFlow, analysis: Dict[str, Any]) -> float:
        """Calculate confidence in stage recommendation (0.0-1.0)."""
        return 0.85  # Simplified
    
    def _select_conversation_tone(self, stage: SessionStage, alliance: Dict[str, Any]) -> ConversationTone:
        """Select appropriate conversation tone."""
        if stage == SessionStage.CRISIS_INTERVENTION:
            return ConversationTone.CRISIS_CALM
        elif stage == SessionStage.OPENING:
            return ConversationTone.WARM_WELCOMING
        elif alliance["current_strength"] < 0.5:
            return ConversationTone.EMPATHETIC_LISTENING
        else:
            return ConversationTone.GENTLE_EXPLORATION
    
    def _recommend_response_length(self, stage: SessionStage, timing: InterventionTiming) -> str:
        """Recommend response length."""
        if stage == SessionStage.CRISIS_INTERVENTION:
            return "brief_and_focused"
        elif timing == InterventionTiming.TOO_EARLY:
            return "short_and_validating"
        else:
            return "moderate"
    
    def _recommend_question_type(self, stage: SessionStage, alliance: Dict[str, Any]) -> str:
        """Recommend type of questions to ask."""
        if stage == SessionStage.EXPLORATION and alliance["current_strength"] > 0.6:
            return "open_ended_exploratory"
        elif alliance["current_strength"] < 0.5:
            return "gentle_clarifying"
        else:
            return "supportive_curious"


# Supporting classes
class InterventionTimingEngine:
    """Assesses optimal timing for therapeutic interventions."""
    
    def assess_intervention_timing(self, conversation_state: ConversationFlow, 
                                 flow_analysis: Dict[str, Any], 
                                 memory: Optional[ConversationMemory]) -> InterventionTiming:
        """Assess timing for interventions."""
        
        if flow_analysis["flow_needs"]["needs_crisis_response"]:
            return InterventionTiming.CRISIS_IMMEDIATE
        
        if conversation_state.alliance_strength < 0.5:
            return InterventionTiming.TOO_EARLY
        
        if (conversation_state.current_stage in [SessionStage.EXPLORATION, SessionStage.INTERVENTION] and
            conversation_state.alliance_strength > 0.7):
            return InterventionTiming.GOOD_TIMING
        
        return InterventionTiming.TOO_EARLY


class TherapeuticAllianceMonitor:
    """Monitors and assesses therapeutic alliance strength."""
    
    def assess_alliance_strength(self, conversation_state: ConversationFlow, 
                               client_input: str, 
                               memory: Optional[ConversationMemory]) -> Dict[str, Any]:
        """Assess current alliance strength and trend."""
        
        # Analyze client input for alliance indicators
        positive_indicators = [
            "appreciate", "helpful", "understand", "comfortable", "safe", "trust"
        ]
        negative_indicators = [
            "don't understand", "not helping", "frustrated", "confused", "defensive"
        ]
        
        text_lower = client_input.lower()
        positive_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in text_lower)
        
        # Simple alliance assessment
        current_strength = conversation_state.alliance_strength
        if positive_count > 0:
            current_strength = min(1.0, current_strength + 0.1)
        if negative_count > 0:
            current_strength = max(0.0, current_strength - 0.15)
        
        return {
            "current_strength": current_strength,
            "trend": "improving" if positive_count > negative_count else "stable",
            "indicators": {
                "positive": positive_count,
                "negative": negative_count
            }
        }


class CulturalSensitivityAdapter:
    """Adapts therapeutic approach for cultural sensitivity."""
    pass


if __name__ == "__main__":
    # Example usage
    flow_engine = TherapeuticConversationFlowEngine()
    
    conversation_state = ConversationFlow(
        current_stage=SessionStage.EXPLORATION,
        conversation_history=[],
        alliance_strength=0.7,
        session_duration=15,
        energy_level="medium",
        client_engagement="engaged",
        therapeutic_momentum="building"
    )
    
    client_input = "I've been having panic attacks every day and I don't know how to make them stop."
    
    flow_analysis = flow_engine.determine_conversation_flow(conversation_state, client_input)
    
    print("Conversation Flow Analysis:")
    print(f"Current stage: {flow_analysis['current_stage']}")
    print(f"Recommended stage: {flow_analysis['recommended_stage']}")
    print(f"Intervention timing: {flow_analysis['intervention_timing']}")
    print(f"Alliance strength: {flow_analysis['alliance_strength']}")
    print(f"Flow recommendations: {flow_analysis['flow_recommendations']}")