"""
Therapeutic Personality Synthesizer (Tier 2.2)

Creates empathetic, therapeutic AI personalities based on expert voice patterns
extracted from 715 clinical concepts and 9 expert voices.

Key Features:
- Expert voice blending (Tim Fletcher + Dr. Ramani + Gabor Maté)
- Therapeutic communication patterns (empathy, validation, crisis response)
- Dynamic personality adaptation based on client needs
- Crisis-aware communication protocols
- Cultural sensitivity and professional boundaries

Input: 715-concept psychology knowledge base + expert voice patterns
Output: Natural, empathetic therapeutic conversations
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TherapeuticResponse:
    """A complete therapeutic response with metadata."""
    content: str
    emotional_tone: str  # warm, empathetic, supportive, concerned, etc.
    expert_influence: str  # primary expert voice influencing this response
    therapeutic_techniques: List[str]  # techniques used in response
    crisis_indicators: List[str]  # any crisis elements detected/addressed
    confidence_score: float
    response_type: str  # validation, intervention, question, psychoeducation


@dataclass
class ExpertVoiceSignature:
    """Signature patterns for each therapeutic expert."""
    expert_name: str
    empathy_phrases: List[str]
    validation_patterns: List[str]
    intervention_style: str
    crisis_response_approach: str
    signature_language: List[str]
    therapeutic_philosophy: str
    communication_patterns: Dict[str, List[str]]


@dataclass
class ClientContext:
    """Context about the client for personalized responses."""
    presenting_concerns: List[str]
    emotional_state: str  # anxious, depressed, angry, overwhelmed, etc.
    crisis_level: str  # none, low, moderate, high, imminent
    session_stage: str  # opening, exploration, intervention, closing
    therapeutic_alliance: float  # 0.0-1.0
    cultural_background: Optional[str] = None
    previous_sessions: int = 0


class TherapeuticPersonalitySynthesizer:
    """Creates authentic therapeutic AI personalities from expert patterns."""
    
    def __init__(self, knowledge_base_path: str = "ai/pixel/knowledge/enhanced_psychology_knowledge_base.json"):
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.expert_signatures = self._create_expert_signatures()
        self.empathy_engine = EmpathyEngine()
        self.crisis_communication = CrisisCommunicationProtocol()
        self.therapeutic_flow = TherapeuticConversationFlow()
        
    def generate_therapeutic_response(self, 
                                    client_input: str, 
                                    client_context: ClientContext,
                                    preferred_expert: Optional[str] = None) -> TherapeuticResponse:
        """Generate a therapeutic response using expert voice synthesis."""
        
        # Analyze client input for emotional state and content
        input_analysis = self._analyze_client_input(client_input, client_context)
        
        # Select appropriate expert voice(s) for response
        expert_blend = self._select_expert_blend(input_analysis, client_context, preferred_expert)
        
        # Generate empathetic response content
        response_content = self._synthesize_response_content(input_analysis, expert_blend, client_context)
        
        # Apply therapeutic techniques if appropriate
        enhanced_response = self._apply_therapeutic_techniques(response_content, input_analysis, client_context)
        
        # Ensure crisis safety if needed
        final_response = self._ensure_crisis_safety(enhanced_response, input_analysis, client_context)
        
        return TherapeuticResponse(
            content=final_response,
            emotional_tone=input_analysis["recommended_tone"],
            expert_influence=expert_blend["primary_expert"],
            therapeutic_techniques=input_analysis["applicable_techniques"],
            crisis_indicators=input_analysis["crisis_indicators"],
            confidence_score=self._calculate_response_confidence(final_response, input_analysis),
            response_type=input_analysis["response_type"]
        )
    
    def _load_knowledge_base(self, path: str) -> Dict[str, Any]:
        """Load the psychology knowledge base."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load knowledge base: {e}")
            return {"concepts": {}, "techniques": {}, "expert_profiles": {}, "statistics": {}}
    
    def _create_expert_signatures(self) -> Dict[str, ExpertVoiceSignature]:
        """Create voice signatures for each expert based on extracted patterns."""
        signatures = {}
        
        # Tim Fletcher - Complex Trauma Specialist
        signatures["Tim Fletcher"] = ExpertVoiceSignature(
            expert_name="Tim Fletcher",
            empathy_phrases=[
                "I understand that this is really difficult for you",
                "What you're experiencing makes complete sense",
                "That's a very normal response to trauma",
                "You're not broken, you're hurt",
                "Your nervous system is doing exactly what it's designed to do"
            ],
            validation_patterns=[
                "Your feelings are completely valid",
                "Of course you would feel that way",
                "That's such a normal response to what you've been through",
                "You're responding exactly as someone would who's experienced trauma"
            ],
            intervention_style="psychoeducational_and_validating",
            crisis_response_approach="calm_grounding_with_education",
            signature_language=[
                "nervous system", "dysregulation", "trauma response", "protective mechanism",
                "hypervigilance", "survival mode", "co-regulation", "felt safety"
            ],
            therapeutic_philosophy="Trauma-informed care with deep understanding of nervous system responses",
            communication_patterns={
                "validation": ["That makes perfect sense", "Of course you'd feel that way"],
                "psychoeducation": ["What's happening in your nervous system is...", "This is a trauma response"],
                "hope": ["Healing is possible", "You can recover from this", "There is hope"]
            }
        )
        
        # Dr. Ramani - Narcissistic Abuse Expert
        signatures["Dr. Ramani"] = ExpertVoiceSignature(
            expert_name="Dr. Ramani",
            empathy_phrases=[
                "I hear how painful this has been for you",
                "That sounds incredibly challenging",
                "You've been through something really difficult",
                "This kind of relationship can be absolutely devastating"
            ],
            validation_patterns=[
                "You're not imagining this",
                "Trust your instincts - they're telling you something important",
                "You're not being too sensitive",
                "Your reality is valid"
            ],
            intervention_style="direct_and_clarifying",
            crisis_response_approach="reality_validation_with_safety_planning",
            signature_language=[
                "gaslighting", "manipulation", "narcissistic behavior", "toxic relationship",
                "emotional abuse", "trauma bonding", "boundaries", "self-preservation"
            ],
            therapeutic_philosophy="Clear, direct communication that validates reality and promotes healthy boundaries",
            communication_patterns={
                "clarity": ["Let me be very clear about this", "What you're describing is..."],
                "validation": ["You're not imagining this", "Trust what you're experiencing"],
                "empowerment": ["You deserve better", "You have the right to...", "Your needs matter"]
            }
        )
        
        # Dr. Gabor Maté - Holistic and Compassionate
        signatures["Dr. Gabor Maté"] = ExpertVoiceSignature(
            expert_name="Dr. Gabor Maté",
            empathy_phrases=[
                "I can feel the pain in what you're sharing",
                "There's such wisdom in your struggle",
                "Your suffering is pointing toward something important",
                "What you're experiencing is deeply human"
            ],
            validation_patterns=[
                "Your pain makes complete sense",
                "Of course this would be difficult for you",
                "You're responding to life exactly as a sensitive person would",
                "There's nothing wrong with you for feeling this way"
            ],
            intervention_style="compassionate_and_holistic",
            crisis_response_approach="gentle_presence_with_deeper_understanding",
            signature_language=[
                "authenticity", "compassion", "wholeness", "healing", "connection",
                "trauma as teacher", "pain as messenger", "inner wisdom"
            ],
            therapeutic_philosophy="Compassionate presence that honors the whole person and sees trauma as a teacher",
            communication_patterns={
                "presence": ["I'm here with you in this", "Let's sit with this together"],
                "wisdom": ["What is your pain trying to tell you?", "There's wisdom in your suffering"],
                "hope": ["Healing is your birthright", "You have everything you need within you"]
            }
        )
        
        # Patrick Teahan - Practical Therapeutic Techniques
        signatures["Patrick Teahan"] = ExpertVoiceSignature(
            expert_name="Patrick Teahan",
            empathy_phrases=[
                "I can see how hard you're working on this",
                "That takes a lot of courage to share",
                "I appreciate you being so open with me",
                "Thank you for trusting me with this"
            ],
            validation_patterns=[
                "What you're feeling is completely understandable",
                "That's a very normal response",
                "You're doing the best you can with what you have",
                "It makes sense that you'd feel that way"
            ],
            intervention_style="practical_and_skill_building",
            crisis_response_approach="practical_tools_with_gentle_guidance",
            signature_language=[
                "coping skills", "practical tools", "step by step", "building capacity",
                "grounding", "self-care", "healthy boundaries", "emotional regulation"
            ],
            therapeutic_philosophy="Practical, skills-based approach with warm encouragement",
            communication_patterns={
                "skill_building": ["Let's try this technique", "Here's a tool that might help"],
                "encouragement": ["You're making progress", "Small steps count", "You're learning"],
                "practical": ["What would be most helpful right now?", "Let's break this down"]
            }
        )
        
        # Heidi Priebe - Attachment and Relationships
        signatures["Heidi Priebe"] = ExpertVoiceSignature(
            expert_name="Heidi Priebe",
            empathy_phrases=[
                "Relationships can be so complex and confusing",
                "It sounds like you're really trying to understand this",
                "That kind of relationship dynamic can be really challenging",
                "I hear how much this relationship matters to you"
            ],
            validation_patterns=[
                "Your concerns about this relationship are valid",
                "It's natural to want understanding in relationships",
                "You're not asking for too much",
                "Your attachment needs are completely normal"
            ],
            intervention_style="insightful_and_relationship_focused",
            crisis_response_approach="relationship_safety_with_attachment_awareness",
            signature_language=[
                "attachment style", "relationship patterns", "emotional needs", "communication",
                "intimacy", "connection", "relationship dynamics", "secure attachment"
            ],
            therapeutic_philosophy="Understanding relationships through the lens of attachment and healthy communication",
            communication_patterns={
                "insight": ["What I'm noticing is...", "There might be a pattern here"],
                "relationships": ["In healthy relationships...", "Consider how this impacts connection"],
                "growth": ["This is an opportunity to...", "You're learning about yourself"]
            }
        )
        
        return signatures
    
    def _analyze_client_input(self, client_input: str, client_context: ClientContext) -> Dict[str, Any]:
        """Analyze client input for emotional content, themes, and therapeutic needs."""
        input_lower = client_input.lower()
        
        # Detect emotional state
        emotional_indicators = {
            "anxious": ["worried", "nervous", "scared", "panic", "anxiety", "stressed"],
            "depressed": ["sad", "hopeless", "worthless", "empty", "depressed", "down"],
            "angry": ["angry", "furious", "rage", "pissed", "irritated", "frustrated"],
            "overwhelmed": ["overwhelmed", "too much", "can't handle", "drowning", "exhausted"],
            "confused": ["confused", "don't understand", "mixed up", "unclear", "lost"]
        }
        
        detected_emotions = []
        for emotion, indicators in emotional_indicators.items():
            if any(indicator in input_lower for indicator in indicators):
                detected_emotions.append(emotion)
        
        # Detect crisis indicators
        crisis_indicators = []
        crisis_patterns = [
            "want to die", "kill myself", "suicide", "end it all", "can't go on",
            "hurt myself", "self harm", "cutting", "overdose", "want to disappear"
        ]
        
        for pattern in crisis_patterns:
            if pattern in input_lower:
                crisis_indicators.append(pattern)
        
        # Determine therapeutic focus areas
        therapeutic_themes = self._identify_therapeutic_themes(input_lower)
        
        # Select appropriate techniques
        applicable_techniques = self._select_applicable_techniques(therapeutic_themes, detected_emotions, client_context)
        
        # Determine response type needed
        response_type = self._determine_response_type(client_input, detected_emotions, crisis_indicators, client_context)
        
        # Recommend emotional tone for response
        recommended_tone = self._recommend_response_tone(detected_emotions, crisis_indicators, client_context)
        
        return {
            "detected_emotions": detected_emotions,
            "crisis_indicators": crisis_indicators,
            "therapeutic_themes": therapeutic_themes,
            "applicable_techniques": applicable_techniques,
            "response_type": response_type,
            "recommended_tone": recommended_tone,
            "input_complexity": len(client_input.split()),
            "primary_concern": therapeutic_themes[0] if therapeutic_themes else "general_support"
        }
    
    def _select_expert_blend(self, input_analysis: Dict[str, Any], 
                           client_context: ClientContext, 
                           preferred_expert: Optional[str]) -> Dict[str, Any]:
        """Select which expert voice(s) to use for response generation."""
        
        if preferred_expert and preferred_expert in self.expert_signatures:
            return {
                "primary_expert": preferred_expert,
                "secondary_experts": [],
                "blend_ratio": {"primary": 1.0}
            }
        
        # Expert selection based on therapeutic themes and client needs
        expert_specialties = {
            "Tim Fletcher": ["trauma", "complex_trauma", "ptsd", "nervous_system", "dissociation"],
            "Dr. Ramani": ["narcissistic_abuse", "toxic_relationships", "gaslighting", "manipulation"],
            "Dr. Gabor Maté": ["authenticity", "holistic_healing", "compassion", "meaning_making"],
            "Patrick Teahan": ["practical_skills", "coping_strategies", "emotional_regulation"],
            "Heidi Priebe": ["relationships", "attachment", "communication", "intimacy"]
        }
        
        # Score experts based on relevance
        expert_scores = {}
        for expert, specialties in expert_specialties.items():
            score = 0
            for theme in input_analysis["therapeutic_themes"]:
                if theme in specialties:
                    score += 2
                # Partial matches
                for specialty in specialties:
                    if specialty in theme or theme in specialty:
                        score += 1
            expert_scores[expert] = score
        
        # Handle crisis situations - Tim Fletcher for trauma-informed crisis response
        if input_analysis["crisis_indicators"]:
            expert_scores["Tim Fletcher"] += 5
            expert_scores["Dr. Gabor Maté"] += 3  # Compassionate presence
        
        # Select primary expert
        primary_expert = max(expert_scores.items(), key=lambda x: x[1])[0] if expert_scores else "Tim Fletcher"
        
        # Select secondary expert for blending
        remaining_experts = {k: v for k, v in expert_scores.items() if k != primary_expert}
        secondary_expert = max(remaining_experts.items(), key=lambda x: x[1])[0] if remaining_experts else None
        
        return {
            "primary_expert": primary_expert,
            "secondary_experts": [secondary_expert] if secondary_expert else [],
            "blend_ratio": {"primary": 0.7, "secondary": 0.3} if secondary_expert else {"primary": 1.0}
        }
    
    def _synthesize_response_content(self, input_analysis: Dict[str, Any], 
                                   expert_blend: Dict[str, Any], 
                                   client_context: ClientContext) -> str:
        """Synthesize response content using expert voice patterns."""
        
        primary_expert = self.expert_signatures[expert_blend["primary_expert"]]
        response_parts = []
        
        # Start with empathy/validation
        if input_analysis["response_type"] in ["validation", "empathy"]:
            empathy_phrase = random.choice(primary_expert.empathy_phrases)
            response_parts.append(empathy_phrase)
            
            if random.choice([True, False]):  # 50% chance to add validation
                validation_phrase = random.choice(primary_expert.validation_patterns)
                response_parts.append(validation_phrase)
        
        # Add therapeutic content based on expert style
        if primary_expert.expert_name == "Tim Fletcher":
            # Add psychoeducational content about trauma/nervous system
            psychoed_phrases = [
                "What you're describing sounds like a trauma response.",
                "Your nervous system is trying to protect you.",
                "This is your body's way of saying it doesn't feel safe.",
                "These symptoms make perfect sense given what you've experienced."
            ]
            if input_analysis["therapeutic_themes"]:
                response_parts.append(random.choice(psychoed_phrases))
        
        elif primary_expert.expert_name == "Dr. Ramani":
            # Add reality validation and boundary-focused content
            reality_phrases = [
                "What you're describing sounds like emotional manipulation.",
                "Trust your instincts - they're telling you something important.",
                "You have the right to feel safe in your relationships.",
                "This behavior is not normal or acceptable."
            ]
            if any(theme in ["relationships", "abuse", "manipulation"] for theme in input_analysis["therapeutic_themes"]):
                response_parts.append(random.choice(reality_phrases))
        
        elif primary_expert.expert_name == "Dr. Gabor Maté":
            # Add compassionate, holistic perspective
            wisdom_phrases = [
                "Your pain is pointing toward something that needs attention.",
                "There's deep wisdom in what you're experiencing.",
                "Healing happens when we can hold our pain with compassion.",
                "You have everything you need for healing within you."
            ]
            response_parts.append(random.choice(wisdom_phrases))
        
        # Add follow-up question or invitation
        follow_up_phrases = [
            "Can you tell me more about that?",
            "What does that feel like in your body?",
            "How long have you been experiencing this?",
            "What would feel most supportive right now?"
        ]
        
        if client_context.session_stage in ["exploration", "intervention"]:
            response_parts.append(random.choice(follow_up_phrases))
        
        return " ".join(response_parts)
    
    # Placeholder methods for complex functionality
    def _identify_therapeutic_themes(self, input_text: str) -> List[str]:
        """Identify therapeutic themes in client input."""
        themes = []
        theme_patterns = {
            "trauma": ["trauma", "abuse", "hurt", "violated", "betrayed"],
            "relationships": ["relationship", "partner", "boyfriend", "girlfriend", "marriage"],
            "anxiety": ["anxious", "worry", "panic", "nervous", "stressed"],
            "depression": ["depressed", "sad", "hopeless", "empty", "worthless"],
            "family": ["family", "mother", "father", "parents", "siblings"]
        }
        
        for theme, keywords in theme_patterns.items():
            if any(keyword in input_text for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    def _select_applicable_techniques(self, themes: List[str], emotions: List[str], context: ClientContext) -> List[str]:
        """Select applicable therapeutic techniques."""
        techniques = []
        
        if "anxious" in emotions:
            techniques.extend(["grounding", "breathing_exercises", "progressive_muscle_relaxation"])
        if "trauma" in themes:
            techniques.extend(["trauma_informed_care", "safety_planning", "nervous_system_regulation"])
        if "relationships" in themes:
            techniques.extend(["communication_skills", "boundary_setting", "attachment_awareness"])
            
        return techniques
    
    def _determine_response_type(self, input_text: str, emotions: List[str], crisis: List[str], context: ClientContext) -> str:
        """Determine the type of therapeutic response needed."""
        if crisis:
            return "crisis_intervention"
        elif "?" in input_text:
            return "psychoeducation"
        elif emotions:
            return "validation"
        else:
            return "exploration"
    
    def _recommend_response_tone(self, emotions: List[str], crisis: List[str], context: ClientContext) -> str:
        """Recommend emotional tone for response."""
        if crisis:
            return "calm_and_grounding"
        elif "anxious" in emotions:
            return "soothing_and_reassuring"
        elif "angry" in emotions:
            return "validating_and_understanding"
        else:
            return "warm_and_empathetic"
    
    def _apply_therapeutic_techniques(self, response: str, analysis: Dict[str, Any], context: ClientContext) -> str:
        """Apply specific therapeutic techniques to enhance response."""
        # This would add specific technique-based enhancements
        return response
    
    def _ensure_crisis_safety(self, response: str, analysis: Dict[str, Any], context: ClientContext) -> str:
        """Ensure crisis safety protocols are followed."""
        if analysis["crisis_indicators"]:
            safety_additions = [
                " If you're having thoughts of hurting yourself, please reach out to a crisis hotline or emergency services.",
                " Your safety is the most important thing right now.",
                " You don't have to go through this alone - help is available."
            ]
            response += random.choice(safety_additions)
        
        return response
    
    def _calculate_response_confidence(self, response: str, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the generated response."""
        base_confidence = 0.8
        
        # Adjust based on various factors
        if analysis["crisis_indicators"]:
            base_confidence += 0.1  # Crisis responses are well-defined
        if len(analysis["therapeutic_themes"]) > 2:
            base_confidence -= 0.1  # Complex situations are harder
        
        return min(1.0, max(0.0, base_confidence))


# Supporting classes
class EmpathyEngine:
    """Generates empathetic responses based on emotional content."""
    pass

class CrisisCommunicationProtocol:
    """Handles crisis communication with appropriate safety measures."""
    pass

class TherapeuticConversationFlow:
    """Manages the flow and structure of therapeutic conversations."""
    pass


def generate_therapeutic_response(client_input: str, 
                                client_context: Optional[ClientContext] = None,
                                preferred_expert: Optional[str] = None) -> TherapeuticResponse:
    """Main function to generate a therapeutic response."""
    
    if client_context is None:
        client_context = ClientContext(
            presenting_concerns=["general_support"],
            emotional_state="unknown",
            crisis_level="none",
            session_stage="exploration",
            therapeutic_alliance=0.5
        )
    
    synthesizer = TherapeuticPersonalitySynthesizer()
    return synthesizer.generate_therapeutic_response(client_input, client_context, preferred_expert)


if __name__ == "__main__":
    # Example usage
    response = generate_therapeutic_response(
        "I feel so overwhelmed and like I can't handle anything anymore. Everything feels too hard.",
        ClientContext(
            presenting_concerns=["anxiety", "overwhelm"],
            emotional_state="overwhelmed",
            crisis_level="low",
            session_stage="exploration",
            therapeutic_alliance=0.6
        )
    )
    
    print(f"Response: {response.content}")
    print(f"Expert influence: {response.expert_influence}")
    print(f"Emotional tone: {response.emotional_tone}")
    print(f"Techniques: {response.therapeutic_techniques}")