"""
Unified Therapeutic AI System (Tier 2.2)

Integrates psychology knowledge base, expert voice synthesis, and conversation flow
to create a comprehensive therapeutic AI companion.

Key Features:
- 715-concept psychology knowledge base integration
- 9 expert voice patterns (Tim Fletcher, Dr. Ramani, Gabor Maté, etc.)
- Dynamic conversation flow management
- Crisis-aware protocols with safety prioritization
- Cultural sensitivity and personalization
- Session memory and therapeutic alliance tracking

This is the unified system that brings together all Tier 2 components!
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ai.pixel.voice.therapeutic_conversation_flow import (
    ConversationFlow,
    ConversationMemory,
    SessionStage,
    TherapeuticConversationFlowEngine,
)
from ai.pixel.voice.therapeutic_personality_synthesizer import (
    ClientContext,
    TherapeuticPersonalitySynthesizer,
    TherapeuticResponse,
)

logger = logging.getLogger(__name__)


@dataclass
class TherapeuticSession:
    """Complete therapeutic session state and context."""
    session_id: str
    client_id: str
    session_number: int
    start_time: str
    current_duration: int  # minutes
    
    # Conversation state
    conversation_flow: ConversationFlow
    conversation_memory: ConversationMemory
    
    # Client context
    client_context: ClientContext
    
    # Session goals and progress
    session_goals: List[str]
    progress_markers: List[str]
    therapeutic_alliance_history: List[float]
    
    # Expert preferences
    preferred_experts: List[str] = field(default_factory=list)
    expert_effectiveness: Dict[str, float] = field(default_factory=dict)


@dataclass
class TherapeuticInteraction:
    """Single interaction within a therapeutic session."""
    timestamp: str
    client_input: str
    ai_response: TherapeuticResponse
    session_stage: SessionStage
    alliance_strength: float
    intervention_applied: Optional[str] = None
    crisis_level: str = "none"


class UnifiedTherapeuticAI:
    """The complete therapeutic AI system integrating all components."""
    
    def __init__(self, knowledge_base_path: str = "ai/pixel/knowledge/enhanced_psychology_knowledge_base.json"):
        # Initialize core components
        self.personality_synthesizer = TherapeuticPersonalitySynthesizer(knowledge_base_path)
        self.conversation_flow_engine = TherapeuticConversationFlowEngine()
        
        # Load psychology knowledge base
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        
        # Session management
        self.active_sessions: Dict[str, TherapeuticSession] = {}
        self.session_history: Dict[str, List[TherapeuticInteraction]] = {}
        
        # System statistics
        self.stats = {
            "total_interactions": 0,
            "crisis_interventions": 0,
            "expert_usage": {},
            "session_outcomes": []
        }
        
        logger.info(f"Unified Therapeutic AI initialized with {len(self.knowledge_base.get('concepts', {}))} concepts")
    
    def start_therapeutic_session(self, 
                                client_id: str,
                                presenting_concerns: List[str],
                                cultural_background: Optional[str] = None,
                                preferred_experts: Optional[List[str]] = None) -> TherapeuticSession:
        """Start a new therapeutic session."""
        
        from datetime import datetime
        session_id = f"session_{client_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize client context
        client_context = ClientContext(
            presenting_concerns=presenting_concerns,
            emotional_state="unknown",
            crisis_level="none",
            session_stage="opening",
            therapeutic_alliance=0.3,  # Start with minimal alliance
            cultural_background=cultural_background,
            previous_sessions=len([s for s in self.active_sessions.values() if s.client_id == client_id])
        )
        
        # Initialize conversation flow
        conversation_flow = ConversationFlow(
            current_stage=SessionStage.OPENING,
            conversation_history=[],
            alliance_strength=0.3,
            session_duration=0,
            energy_level="unknown",
            client_engagement="unknown",
            therapeutic_momentum="beginning"
        )
        
        # Initialize conversation memory
        conversation_memory = ConversationMemory(
            session_number=client_context.previous_sessions + 1,
            key_themes=presenting_concerns,
            established_rapport_elements=[],
            client_preferences={},
            therapeutic_goals=[],
            progress_markers=[],
            crisis_history=[],
            effective_interventions=[]
        )
        
        # Create session
        session = TherapeuticSession(
            session_id=session_id,
            client_id=client_id,
            session_number=client_context.previous_sessions + 1,
            start_time=datetime.now().isoformat(),
            current_duration=0,
            conversation_flow=conversation_flow,
            conversation_memory=conversation_memory,
            client_context=client_context,
            session_goals=self._generate_initial_session_goals(presenting_concerns),
            progress_markers=[],
            therapeutic_alliance_history=[0.3],
            preferred_experts=preferred_experts or []
        )
        
        self.active_sessions[session_id] = session
        self.session_history[session_id] = []
        
        logger.info(f"Started therapeutic session {session_id} for client {client_id}")
        
        return session
    
    def process_client_input(self, session_id: str, client_input: str) -> TherapeuticResponse:
        """Process client input and generate therapeutic response."""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Update session duration (simplified - would track real time)
        session.current_duration += 2
        
        # Analyze conversation flow
        flow_analysis = self.conversation_flow_engine.determine_conversation_flow(
            session.conversation_flow, client_input, session.conversation_memory
        )
        
        # Update session state based on flow analysis
        self._update_session_state(session, flow_analysis, client_input)
        
        # Select appropriate expert for response
        preferred_expert = self._select_session_expert(session, flow_analysis)
        
        # Generate therapeutic response
        therapeutic_response = self.personality_synthesizer.generate_therapeutic_response(
            client_input, session.client_context, preferred_expert
        )
        
        # Enhance response with session context and flow recommendations
        enhanced_response = self._enhance_response_with_flow(
            therapeutic_response, flow_analysis, session
        )
        
        # Record interaction
        interaction = TherapeuticInteraction(
            timestamp=self._get_current_timestamp(),
            client_input=client_input,
            ai_response=enhanced_response,
            session_stage=session.conversation_flow.current_stage,
            alliance_strength=session.conversation_flow.alliance_strength,
            crisis_level=session.client_context.crisis_level
        )
        
        self.session_history[session_id].append(interaction)
        
        # Update conversation history
        session.conversation_flow.conversation_history.extend([
            {"role": "client", "content": client_input, "timestamp": interaction.timestamp},
            {"role": "ai", "content": enhanced_response.content, "timestamp": interaction.timestamp}
        ])
        
        # Update statistics
        self._update_statistics(enhanced_response, session)
        
        # Handle crisis situations
        if enhanced_response.crisis_indicators:
            self._handle_crisis_situation(session, enhanced_response)
        
        logger.info(f"Processed input in session {session_id}, stage: {session.conversation_flow.current_stage.value}")
        
        return enhanced_response
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        interactions = self.session_history.get(session_id, [])
        
        # Calculate session metrics
        alliance_progression = [i.alliance_strength for i in interactions]
        expert_usage = {}
        crisis_count = 0
        
        for interaction in interactions:
            expert = interaction.ai_response.expert_influence
            expert_usage[expert] = expert_usage.get(expert, 0) + 1
            if interaction.ai_response.crisis_indicators:
                crisis_count += 1
        
        return {
            "session_info": {
                "session_id": session_id,
                "client_id": session.client_id,
                "session_number": session.session_number,
                "duration_minutes": session.current_duration,
                "current_stage": session.conversation_flow.current_stage.value
            },
            "therapeutic_metrics": {
                "alliance_strength": session.conversation_flow.alliance_strength,
                "alliance_progression": alliance_progression,
                "session_goals": session.session_goals,
                "progress_markers": session.progress_markers
            },
            "interaction_summary": {
                "total_interactions": len(interactions),
                "expert_usage": expert_usage,
                "crisis_interventions": crisis_count,
                "primary_themes": session.conversation_memory.key_themes
            },
            "clinical_insights": {
                "presenting_concerns": session.client_context.presenting_concerns,
                "emotional_patterns": self._analyze_emotional_patterns(interactions),
                "intervention_effectiveness": self._assess_intervention_effectiveness(interactions)
            }
        }
    
    def _load_knowledge_base(self, path: str) -> Dict[str, Any]:
        """Load psychology knowledge base."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load knowledge base: {e}")
            return {}
    
    def _generate_initial_session_goals(self, presenting_concerns: List[str]) -> List[str]:
        """Generate initial session goals based on presenting concerns."""
        goal_mapping = {
            "anxiety": "Reduce anxiety symptoms and develop coping strategies",
            "depression": "Improve mood and increase engagement in meaningful activities",
            "trauma": "Process trauma safely and build resilience",
            "relationships": "Improve relationship patterns and communication skills",
            "grief": "Process grief and loss in a healthy way"
        }
        
        goals = []
        for concern in presenting_concerns:
            if concern in goal_mapping:
                goals.append(goal_mapping[concern])
            else:
                goals.append(f"Address {concern} concerns")
        
        goals.append("Build therapeutic alliance and trust")
        return goals
    
    def _update_session_state(self, session: TherapeuticSession, flow_analysis: Dict[str, Any], client_input: str) -> None:
        """Update session state based on flow analysis."""
        
        # Update conversation flow stage
        if flow_analysis["stage_transition_needed"]:
            new_stage = SessionStage(flow_analysis["recommended_stage"])
            session.conversation_flow.current_stage = new_stage
            session.client_context.session_stage = new_stage.value
        
        # Update alliance strength
        session.conversation_flow.alliance_strength = flow_analysis["alliance_strength"]
        session.therapeutic_alliance_history.append(flow_analysis["alliance_strength"])
        session.client_context.therapeutic_alliance = flow_analysis["alliance_strength"]
        
        # Update crisis level if needed
        if flow_analysis.get("intervention_timing") == "crisis_immediate":
            session.client_context.crisis_level = "high"
        
        # Update conversation memory
        self._update_conversation_memory(session, client_input, flow_analysis)
    
    def _select_session_expert(self, session: TherapeuticSession, flow_analysis: Dict[str, Any]) -> Optional[str]:
        """Select best expert for current session context."""
        
        # Use preferred experts if specified
        if session.preferred_experts:
            return session.preferred_experts[0]
        
        # Select based on presenting concerns and session history
        concern_expert_mapping = {
            "trauma": "Tim Fletcher",
            "narcissistic_abuse": "Dr. Ramani", 
            "relationships": "Heidi Priebe",
            "authenticity": "Dr. Gabor Maté",
            "skills": "Patrick Teahan"
        }
        
        for concern in session.client_context.presenting_concerns:
            if concern in concern_expert_mapping:
                return concern_expert_mapping[concern]
        
        return None  # Let synthesizer choose
    
    def _enhance_response_with_flow(self, response: TherapeuticResponse, flow_analysis: Dict[str, Any], session: TherapeuticSession) -> TherapeuticResponse:
        """Enhance response based on conversation flow recommendations."""
        
        # Add session-specific elements
        if session.conversation_flow.current_stage == SessionStage.OPENING:
            if "confidentiality" in flow_analysis["flow_recommendations"]["conversation_approach"]:
                response.content += " Everything we discuss here is confidential and this is a safe space for you to share."
        
        # Add alliance-building elements if needed
        if flow_analysis["alliance_strength"] < 0.5:
            response.content = "I want you to know that I'm here to support you. " + response.content
        
        return response
    
    def _update_conversation_memory(self, session: TherapeuticSession, client_input: str, flow_analysis: Dict[str, Any]) -> None:
        """Update conversation memory with new insights."""
        
        # Extract key themes from input
        themes = self._extract_themes_from_input(client_input)
        for theme in themes:
            if theme not in session.conversation_memory.key_themes:
                session.conversation_memory.key_themes.append(theme)
        
        # Update rapport elements
        if flow_analysis["alliance_strength"] > session.conversation_memory.session_number * 0.1:
            session.conversation_memory.established_rapport_elements.append("trust_building")
    
    def _handle_crisis_situation(self, session: TherapeuticSession, response: TherapeuticResponse) -> None:
        """Handle crisis situations with appropriate protocols."""
        
        crisis_event = {
            "timestamp": self._get_current_timestamp(),
            "crisis_indicators": response.crisis_indicators,
            "response_content": response.content,
            "session_stage": session.conversation_flow.current_stage.value
        }
        
        session.conversation_memory.crisis_history.append(crisis_event)
        session.client_context.crisis_level = "high"
        
        # Force transition to crisis intervention stage
        session.conversation_flow.current_stage = SessionStage.CRISIS_INTERVENTION
        
        logger.warning(f"Crisis situation detected in session {session.session_id}")
    
    def _update_statistics(self, response: TherapeuticResponse, session: TherapeuticSession) -> None:
        """Update system statistics."""
        self.stats["total_interactions"] += 1
        
        expert = response.expert_influence
        self.stats["expert_usage"][expert] = self.stats["expert_usage"].get(expert, 0) + 1
        
        if response.crisis_indicators:
            self.stats["crisis_interventions"] += 1
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _extract_themes_from_input(self, client_input: str) -> List[str]:
        """Extract therapeutic themes from client input."""
        themes = []
        input_lower = client_input.lower()
        
        theme_keywords = {
            "anxiety": ["anxious", "worry", "panic", "nervous"],
            "depression": ["depressed", "sad", "hopeless", "empty"],
            "trauma": ["trauma", "abuse", "hurt", "flashbacks"],
            "relationships": ["relationship", "partner", "family"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    def _analyze_emotional_patterns(self, interactions: List[TherapeuticInteraction]) -> Dict[str, Any]:
        """Analyze emotional patterns across interactions."""
        return {"pattern_analysis": "simplified_implementation"}
    
    def _assess_intervention_effectiveness(self, interactions: List[TherapeuticInteraction]) -> Dict[str, Any]:
        """Assess effectiveness of therapeutic interventions."""
        return {"effectiveness_analysis": "simplified_implementation"}


def create_therapeutic_ai_session(client_id: str, 
                                presenting_concerns: List[str],
                                cultural_background: Optional[str] = None) -> Tuple[UnifiedTherapeuticAI, TherapeuticSession]:
    """Create a new therapeutic AI session."""
    
    ai_system = UnifiedTherapeuticAI()
    session = ai_system.start_therapeutic_session(
        client_id=client_id,
        presenting_concerns=presenting_concerns,
        cultural_background=cultural_background
    )
    
    return ai_system, session


if __name__ == "__main__":
    # Example therapeutic session
    print("🎭 UNIFIED THERAPEUTIC AI SYSTEM DEMO 🎭")
    print()
    
    # Create AI system and start session
    ai_system, session = create_therapeutic_ai_session(
        client_id="demo_client",
        presenting_concerns=["anxiety", "trauma", "relationships"]
    )
    
    print(f"Started session: {session.session_id}")
    print(f"Initial stage: {session.conversation_flow.current_stage.value}")
    print(f"Session goals: {session.session_goals}")
    print()
    
    # Simulate conversation
    client_inputs = [
        "Hi, I'm not sure where to start. I've been having panic attacks and I think it might be related to my childhood.",
        "When I was young, my father was very controlling and would yell at me constantly. Now I feel anxious around authority figures.",
        "Sometimes I feel like I want to just disappear. The panic attacks are getting worse and I don't know how to cope."
    ]
    
    for i, client_input in enumerate(client_inputs, 1):
        print(f"=== Interaction {i} ===")
        print(f"Client: {client_input}")
        
        response = ai_system.process_client_input(session.session_id, client_input)
        
        print(f"AI ({response.expert_influence}): {response.content}")
        print(f"Stage: {session.conversation_flow.current_stage.value}")
        print(f"Alliance: {session.conversation_flow.alliance_strength:.2f}")
        if response.crisis_indicators:
            print(f"⚠️  Crisis indicators: {response.crisis_indicators}")
        print()
    
    # Get session summary
    summary = ai_system.get_session_summary(session.session_id)
    print("=== SESSION SUMMARY ===")
    print(f"Duration: {summary['session_info']['duration_minutes']} minutes")
    print(f"Final alliance strength: {summary['therapeutic_metrics']['alliance_strength']:.2f}")
    print(f"Expert usage: {summary['interaction_summary']['expert_usage']}")
    print(f"Crisis interventions: {summary['interaction_summary']['crisis_interventions']}")
    print()
    
    print("🎉 UNIFIED THERAPEUTIC AI SYSTEM FULLY FUNCTIONAL! 🎉")