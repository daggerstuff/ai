"""
Dynamic Conversation Generation System

Generates therapeutic conversations dynamically based on clinical knowledge,
client presentations, therapeutic goals, and contextual factors. Integrates
with psychology knowledge base and clinical validation systems.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .clinical_knowledge_embedder import ClinicalKnowledgeEmbedder
from .clinical_similarity_search import ClinicalSimilaritySearch
from .psychology_knowledge_processor import PsychologyKnowledgeProcessor
from .therapeutic_conversation_schema import (
    ClinicalContext,
    ConversationRole,
    TherapeuticModality,
)
from .therapist_response_generator import (
    InterventionType,
    TherapistResponse,
    TherapistResponseGenerator,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationComplexity(Enum):
    """Complexity levels for generated conversations"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ConversationPhase(Enum):
    """Phases of therapeutic conversation"""
    INITIAL_ASSESSMENT = "initial_assessment"
    RAPPORT_BUILDING = "rapport_building"
    PROBLEM_EXPLORATION = "problem_exploration"
    INTERVENTION_PLANNING = "intervention_planning"
    SKILL_BUILDING = "skill_building"
    PROGRESS_MONITORING = "progress_monitoring"
    TERMINATION_PLANNING = "termination_planning"


class ClientResponseStyle(Enum):
    """Client response styles for conversation generation"""
    COOPERATIVE = "cooperative"
    RESISTANT = "resistant"
    AMBIVALENT = "ambivalent"
    CRISIS = "crisis"
    INTELLECTUALIZING = "intellectualizing"
    EMOTIONAL = "emotional"
    WITHDRAWN = "withdrawn"


@dataclass
class ConversationParameters:
    """Parameters for dynamic conversation generation"""
    therapeutic_modality: TherapeuticModality
    client_presentation: str
    primary_diagnosis: Optional[str] = None
    secondary_diagnoses: List[str] = field(default_factory=list)
    conversation_phase: ConversationPhase = ConversationPhase.PROBLEM_EXPLORATION
    complexity_level: ConversationComplexity = ConversationComplexity.INTERMEDIATE
    client_response_style: ClientResponseStyle = ClientResponseStyle.COOPERATIVE
    session_number: int = 1
    therapeutic_goals: List[str] = field(default_factory=list)
    cultural_factors: List[str] = field(default_factory=list)
    crisis_indicators: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    conversation_length: int = 10  # Number of exchanges
    include_clinical_notes: bool = True
    validate_responses: bool = True


@dataclass
class ConversationTurn:
    """Single turn in a therapeutic conversation"""
    turn_id: str
    speaker: ConversationRole
    content: str
    clinical_rationale: Optional[str] = None
    intervention_type: Optional[InterventionType] = None
    therapeutic_technique: Optional[str] = None
    confidence_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedConversation:
    """Complete generated therapeutic conversation"""
    conversation_id: str
    parameters: ConversationParameters
    turns: List[ConversationTurn]
    clinical_summary: str
    therapeutic_progress: Dict[str, Any]
    quality_metrics: Dict[str, float]
    recommendations: List[str]
    warnings: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DynamicConversationGenerator:
    """
    Dynamic conversation generation system based on clinical knowledge
    
    Generates realistic therapeutic conversations by combining clinical
    knowledge, therapeutic techniques, and contextual factors.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the dynamic conversation generator"""
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.knowledge_processor = PsychologyKnowledgeProcessor()
        self.knowledge_embedder = ClinicalKnowledgeEmbedder()
        self.similarity_search = ClinicalSimilaritySearch()
        self.response_generator = TherapistResponseGenerator()
        
        # Load conversation templates and patterns
        self.conversation_templates = self._load_conversation_templates()
        self.client_response_patterns = self._load_client_response_patterns()
        self.intervention_sequences = self._load_intervention_sequences()
        
        # Generation settings
        self.generation_settings = {
            'max_retries': 3,
            'quality_threshold': 0.7,
            'diversity_factor': 0.3,
            'clinical_accuracy_weight': 0.4,
            'therapeutic_appropriateness_weight': 0.3,
            'naturalness_weight': 0.3
        }
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration settings"""
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        return {
            'enable_clinical_validation': True,
            'require_expert_review': False,
            'max_conversation_length': 50,
            'min_conversation_length': 5,
            'default_complexity': 'intermediate',
            'safety_check_enabled': True
        }
    
    def _load_conversation_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load conversation templates for different scenarios"""
        return {
            'initial_assessment': {
                'opening_prompts': [
                    "What brings you here today?",
                    "How can I help you?",
                    "Tell me about what's been going on for you lately."
                ],
                'follow_up_patterns': [
                    "Can you tell me more about {topic}?",
                    "How long has this been going on?",
                    "What does that feel like for you?"
                ]
            },
            'problem_exploration': {
                'exploration_prompts': [
                    "Help me understand what that experience is like for you.",
                    "What thoughts go through your mind when {situation}?",
                    "How does this affect your daily life?"
                ],
                'clarification_patterns': [
                    "When you say {statement}, what do you mean exactly?",
                    "Can you give me an example of {behavior}?",
                    "What would need to change for you to feel better?"
                ]
            },
            'skill_building': {
                'teaching_prompts': [
                    "Let me share a technique that might be helpful.",
                    "Have you tried {technique} before?",
                    "Let's practice this together."
                ],
                'practice_patterns': [
                    "How would you apply this in {situation}?",
                    "What might get in the way of using this skill?",
                    "Let's role-play this scenario."
                ]
            }
        }
    
    def _load_client_response_patterns(self) -> Dict[ClientResponseStyle, Dict[str, List[str]]]:
        """Load client response patterns for different styles"""
        return {
            ClientResponseStyle.COOPERATIVE: {
                'agreement': [
                    "Yes, that makes sense.",
                    "I can see that.",
                    "That's exactly how I feel."
                ],
                'elaboration': [
                    "Actually, there's more to it...",
                    "Now that you mention it...",
                    "I've been thinking about that too."
                ],
                'questions': [
                    "What do you think I should do?",
                    "How can I change this?",
                    "Is this normal?"
                ]
            },
            ClientResponseStyle.RESISTANT: {
                'disagreement': [
                    "I don't think that's right.",
                    "That doesn't apply to me.",
                    "You don't understand my situation."
                ],
                'deflection': [
                    "But what about...",
                    "That's not the real problem.",
                    "I've tried that before and it didn't work."
                ],
                'minimization': [
                    "It's not that bad.",
                    "I can handle it myself.",
                    "Everyone has problems."
                ]
            },
            ClientResponseStyle.AMBIVALENT: {
                'uncertainty': [
                    "I'm not sure about that.",
                    "Maybe, but...",
                    "I don't know if that would work."
                ],
                'conflicted': [
                    "Part of me wants to, but...",
                    "I see both sides.",
                    "Sometimes I feel one way, sometimes another."
                ],
                'hesitation': [
                    "I guess so...",
                    "I'm not ready for that yet.",
                    "That sounds scary."
                ]
            },
            ClientResponseStyle.CRISIS: {
                'distress': [
                    "I can't take this anymore.",
                    "Everything is falling apart.",
                    "I don't know what to do."
                ],
                'urgency': [
                    "I need help right now.",
                    "Something has to change.",
                    "I'm scared of what might happen."
                ],
                'hopelessness': [
                    "Nothing will ever get better.",
                    "I've tried everything.",
                    "What's the point?"
                ]
            }
        }
    
    def _load_intervention_sequences(self) -> Dict[TherapeuticModality, Dict[str, List[str]]]:
        """Load intervention sequences for different therapeutic modalities"""
        return {
            TherapeuticModality.CBT: {
                'thought_challenging': [
                    "What evidence supports this thought?",
                    "What evidence contradicts it?",
                    "What would you tell a friend in this situation?",
                    "What's a more balanced way to think about this?"
                ],
                'behavioral_activation': [
                    "What activities used to bring you joy?",
                    "Let's schedule one pleasant activity for this week.",
                    "How did that activity make you feel?",
                    "What got in the way of doing more activities?"
                ]
            },
            TherapeuticModality.DBT: {
                'distress_tolerance': [
                    "Let's practice the TIPP skill.",
                    "What would radical acceptance look like here?",
                    "How can you distract yourself in a healthy way?",
                    "What's your distress level right now, 1-10?"
                ],
                'emotion_regulation': [
                    "What emotion are you experiencing?",
                    "Where do you feel that in your body?",
                    "What's the function of this emotion?",
                    "How can you validate this feeling while also coping?"
                ]
            },
            TherapeuticModality.PSYCHODYNAMIC: {
                'interpretation': [
                    "I notice a pattern here...",
                    "This reminds me of what you said about...",
                    "What do you make of this connection?",
                    "How does this relate to your early experiences?"
                ],
                'transference_exploration': [
                    "How are you experiencing our relationship?",
                    "Does this remind you of other relationships?",
                    "What feelings come up for you with me?",
                    "How is this similar to your relationship with...?"
                ]
            }
        }
    
    async def generate_conversation(self, parameters: ConversationParameters) -> GeneratedConversation:
        """Generate a complete therapeutic conversation"""
        try:
            logger.info(f"Generating conversation with parameters: {parameters}")
            
            # Validate parameters
            self._validate_parameters(parameters)
            
            # Retrieve relevant clinical knowledge
            clinical_knowledge = await self._retrieve_clinical_knowledge(parameters)
            
            # Generate conversation turns
            turns = await self._generate_conversation_turns(parameters, clinical_knowledge)
            
            # Validate conversation quality
            if parameters.validate_responses:
                turns = await self._validate_conversation_turns(turns, parameters)
            
            # Generate clinical summary and metrics
            clinical_summary = await self._generate_clinical_summary(turns, parameters)
            quality_metrics = await self._calculate_quality_metrics(turns, parameters)
            therapeutic_progress = await self._assess_therapeutic_progress(turns, parameters)
            
            # Generate recommendations and warnings
            recommendations = await self._generate_recommendations(turns, parameters, quality_metrics)
            warnings = await self._identify_warnings(turns, parameters)
            
            conversation = GeneratedConversation(
                conversation_id=f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
                parameters=parameters,
                turns=turns,
                clinical_summary=clinical_summary,
                therapeutic_progress=therapeutic_progress,
                quality_metrics=quality_metrics,
                recommendations=recommendations,
                warnings=warnings,
                metadata={
                    'clinical_knowledge_used': len(clinical_knowledge),
                    'generation_time': datetime.now().isoformat(),
                    'complexity_achieved': self._assess_complexity_achieved(turns),
                    'modality_adherence': self._assess_modality_adherence(turns, parameters.therapeutic_modality)
                }
            )
            
            logger.info(f"Generated conversation {conversation.conversation_id} with {len(turns)} turns")
            return conversation
            
        except Exception as e:
            logger.error(f"Error generating conversation: {e}")
            raise
    
    def _validate_parameters(self, parameters: ConversationParameters):
        """Validate conversation generation parameters"""
        if parameters.conversation_length < self.config.get('min_conversation_length', 5):
            raise ValueError(f"Conversation length too short: {parameters.conversation_length}")
        
        if parameters.conversation_length > self.config.get('max_conversation_length', 50):
            raise ValueError(f"Conversation length too long: {parameters.conversation_length}")
        
        if not parameters.client_presentation.strip():
            raise ValueError("Client presentation cannot be empty")
        
        if not parameters.therapeutic_goals and parameters.conversation_phase != ConversationPhase.INITIAL_ASSESSMENT:
            logger.warning("No therapeutic goals specified for non-initial conversation")
    
    async def _retrieve_clinical_knowledge(self, parameters: ConversationParameters) -> List[Dict[str, Any]]:
        """Retrieve relevant clinical knowledge for conversation generation"""
        try:
            # Build search query from parameters
            search_terms = [
                parameters.client_presentation,
                parameters.therapeutic_modality.value,
                parameters.conversation_phase.value
            ]
            
            if parameters.primary_diagnosis:
                search_terms.append(parameters.primary_diagnosis)
            
            search_terms.extend(parameters.therapeutic_goals)
            search_terms.extend(parameters.cultural_factors)
            
            query = " ".join(search_terms)
            
            # Search for relevant knowledge
            knowledge_results = await self.similarity_search.search(
                query=query,
                limit=10,
                filters={
                    'modality': parameters.therapeutic_modality.value,
                    'complexity': parameters.complexity_level.value
                }
            )
            
            return knowledge_results
            
        except Exception as e:
            logger.error(f"Error retrieving clinical knowledge: {e}")
            return []
    
    async def _generate_conversation_turns(self, parameters: ConversationParameters, 
                                         clinical_knowledge: List[Dict[str, Any]]) -> List[ConversationTurn]:
        """Generate individual conversation turns"""
        turns = []
        current_speaker = ConversationRole.THERAPIST  # Therapist typically starts
        
        # Generate opening turn
        opening_turn = await self._generate_opening_turn(parameters, clinical_knowledge)
        turns.append(opening_turn)
        current_speaker = ConversationRole.CLIENT
        
        # Generate conversation body
        for turn_num in range(1, parameters.conversation_length):
            if current_speaker == ConversationRole.CLIENT:
                turn = await self._generate_client_turn(
                    turn_num, parameters, turns, clinical_knowledge
                )
            else:
                turn = await self._generate_therapist_turn(
                    turn_num, parameters, turns, clinical_knowledge
                )
            
            turns.append(turn)
            current_speaker = (ConversationRole.CLIENT if current_speaker == ConversationRole.THERAPIST 
                             else ConversationRole.THERAPIST)
        
        return turns
    
    async def _generate_opening_turn(self, parameters: ConversationParameters, 
                                   clinical_knowledge: List[Dict[str, Any]]) -> ConversationTurn:
        """Generate opening therapist turn"""
        phase_templates = self.conversation_templates.get(parameters.conversation_phase.value, {})
        opening_prompts = phase_templates.get('opening_prompts', [
            "What brings you here today?",
            "How can I help you?",
            "Tell me about what's been going on for you lately."
        ])
        
        # Select appropriate opening based on context
        if parameters.crisis_indicators:
            content = "I understand you're going through a difficult time. You're safe here. What's happening that brought you in today?"
            intervention_type = InterventionType.CRISIS_INTERVENTION
        elif parameters.session_number > 1:
            content = "How have things been since we last met?"
            intervention_type = InterventionType.ASSESSMENT
        else:
            content = random.choice(opening_prompts)
            intervention_type = InterventionType.ASSESSMENT
        
        return ConversationTurn(
            turn_id="turn_000",
            speaker=ConversationRole.THERAPIST,
            content=content,
            clinical_rationale="Opening assessment to establish rapport and understand client's current state",
            intervention_type=intervention_type,
            therapeutic_technique="Open-ended questioning",
            confidence_score=0.9,
            metadata={
                'conversation_phase': parameters.conversation_phase.value,
                'session_number': parameters.session_number
            }
        )
    
    async def _generate_client_turn(self, turn_num: int, parameters: ConversationParameters,
                                  previous_turns: List[ConversationTurn], 
                                  clinical_knowledge: List[Dict[str, Any]]) -> ConversationTurn:
        """Generate client response turn"""
        # Get last therapist turn for context
        last_therapist_turn = None
        for turn in reversed(previous_turns):
            if turn.speaker == ConversationRole.THERAPIST:
                last_therapist_turn = turn
                break
        
        # Generate response based on client style and context
        response_patterns = self.client_response_patterns.get(
            parameters.client_response_style, 
            self.client_response_patterns[ClientResponseStyle.COOPERATIVE]
        )
        
        # Select response type based on intervention and client style
        if last_therapist_turn and last_therapist_turn.intervention_type == InterventionType.CRISIS_INTERVENTION:
            if parameters.client_response_style == ClientResponseStyle.CRISIS:
                response_type = 'distress'
            else:
                response_type = 'agreement'
        elif parameters.client_response_style == ClientResponseStyle.RESISTANT:
            response_type = random.choice(['disagreement', 'deflection', 'minimization'])
        elif parameters.client_response_style == ClientResponseStyle.AMBIVALENT:
            response_type = random.choice(['uncertainty', 'conflicted', 'hesitation'])
        else:
            response_type = random.choice(['agreement', 'elaboration', 'questions'])
        
        # Generate contextual response
        base_responses = response_patterns.get(response_type, ["I understand."])
        base_response = random.choice(base_responses)
        
        # Add context-specific content
        contextual_content = await self._add_contextual_content(
            base_response, parameters, last_therapist_turn, clinical_knowledge
        )
        
        return ConversationTurn(
            turn_id=f"turn_{turn_num:03d}",
            speaker=ConversationRole.CLIENT,
            content=contextual_content,
            confidence_score=0.8,
            metadata={
                'response_style': parameters.client_response_style.value,
                'response_type': response_type,
                'contextual_factors': parameters.cultural_factors
            }
        )
    
    async def _generate_therapist_turn(self, turn_num: int, parameters: ConversationParameters,
                                     previous_turns: List[ConversationTurn], 
                                     clinical_knowledge: List[Dict[str, Any]]) -> ConversationTurn:
        """Generate therapist response turn"""
        # Get last client turn for context
        last_client_turn = None
        for turn in reversed(previous_turns):
            if turn.speaker == ConversationRole.CLIENT:
                last_client_turn = turn
                break
        
        if not last_client_turn:
            raise ValueError("No client turn found for therapist response")
        
        # Generate response using therapist response generator
        clinical_context = self._build_clinical_context(parameters, previous_turns)
        
        therapist_response = self.response_generator.generate_response(
            client_statement=last_client_turn.content,
            clinical_context=clinical_context
        )
        
        # Adapt response based on conversation flow and complexity
        adapted_response = await self._adapt_therapist_response(
            therapist_response, parameters, previous_turns, clinical_knowledge
        )
        
        return ConversationTurn(
            turn_id=f"turn_{turn_num:03d}",
            speaker=ConversationRole.THERAPIST,
            content=adapted_response.content,
            clinical_rationale=adapted_response.clinical_rationale,
            intervention_type=adapted_response.intervention_type,
            therapeutic_technique=adapted_response.therapeutic_technique,
            confidence_score=adapted_response.confidence_score,
            metadata={
                'modality': parameters.therapeutic_modality.value,
                'conversation_phase': parameters.conversation_phase.value,
                'contraindications': adapted_response.contraindications,
                'follow_up_suggestions': adapted_response.follow_up_suggestions
            }
        )
    
    def _build_clinical_context(self, parameters: ConversationParameters, 
                               previous_turns: List[ConversationTurn]) -> ClinicalContext:
        """Build clinical context from parameters and conversation history"""
        return ClinicalContext(
            client_presentation=parameters.client_presentation,
            primary_diagnosis=parameters.primary_diagnosis,
            secondary_diagnoses=parameters.secondary_diagnoses,
            therapeutic_goals=parameters.therapeutic_goals,
            cultural_factors=parameters.cultural_factors,
            crisis_indicators=parameters.crisis_indicators,
            contraindications=parameters.contraindications,
            session_number=parameters.session_number,
            conversation_history=[turn.content for turn in previous_turns[-5:]]  # Last 5 turns
        )
    
    async def _adapt_therapist_response(self, response: TherapistResponse, 
                                      parameters: ConversationParameters,
                                      previous_turns: List[ConversationTurn],
                                      clinical_knowledge: List[Dict[str, Any]]) -> TherapistResponse:
        """Adapt therapist response based on conversation context and complexity"""
        # Adjust complexity based on parameters
        if parameters.complexity_level == ConversationComplexity.BASIC:
            response.content = self._simplify_language(response.content)
        elif parameters.complexity_level == ConversationComplexity.EXPERT:
            response.content = self._enhance_clinical_language(response.content, clinical_knowledge)
        
        # Adjust for cultural factors
        if parameters.cultural_factors:
            response.content = await self._culturally_adapt_response(
                response.content, parameters.cultural_factors
            )
        
        # Ensure modality consistency
        response = await self._ensure_modality_consistency(
            response, parameters.therapeutic_modality, previous_turns
        )
        
        return response
    
    def _simplify_language(self, content: str) -> str:
        """Simplify language for basic complexity level"""
        # Replace complex terms with simpler alternatives
        simplifications = {
            'cognitive distortion': 'unhelpful thought pattern',
            'maladaptive': 'unhelpful',
            'psychoeducation': 'learning about',
            'therapeutic alliance': 'our working relationship',
            'intervention': 'technique',
            'symptomatology': 'symptoms'
        }
        
        simplified = content
        for complex_term, simple_term in simplifications.items():
            simplified = simplified.replace(complex_term, simple_term)
        
        return simplified
    
    def _enhance_clinical_language(self, content: str, clinical_knowledge: List[Dict[str, Any]]) -> str:
        """Enhance language with clinical terminology for expert level"""
        # Add clinical precision and terminology
        if clinical_knowledge:
            # Extract relevant clinical terms from knowledge base
            clinical_terms = []
            for knowledge in clinical_knowledge[:3]:  # Use top 3 most relevant
                if 'clinical_terms' in knowledge:
                    clinical_terms.extend(knowledge['clinical_terms'])
            
            # Integrate appropriate clinical language
            if clinical_terms and len(content.split()) < 30:  # Only for shorter responses
                relevant_term = random.choice(clinical_terms)
                if relevant_term.lower() not in content.lower():
                    content += f" This aligns with {relevant_term} principles."
        
        return content
    
    async def _culturally_adapt_response(self, content: str, cultural_factors: List[str]) -> str:
        """Adapt response for cultural sensitivity"""
        adaptations = {
            'hispanic/latino': {
                'family': 'familia',
                'respect': 'respeto',
                'considerations': 'Consider family dynamics and cultural values around mental health.'
            },
            'asian': {
                'considerations': 'Be mindful of concepts of face, family honor, and collective vs. individual focus.'
            },
            'african american': {
                'considerations': 'Consider historical trauma, systemic factors, and strength-based approaches.'
            }
        }
        
        for factor in cultural_factors:
            factor_lower = factor.lower()
            if factor_lower in adaptations:
                adaptation = adaptations[factor_lower]
                if 'considerations' in adaptation and len(content.split()) < 25:
                    content += f" {adaptation['considerations']}"
        
        return content
    
    async def _ensure_modality_consistency(self, response: TherapistResponse, 
                                         modality: TherapeuticModality,
                                         previous_turns: List[ConversationTurn]) -> TherapistResponse:
        """Ensure response is consistent with therapeutic modality"""
        modality_keywords = {
            TherapeuticModality.CBT: ['thoughts', 'feelings', 'behaviors', 'evidence', 'challenge'],
            TherapeuticModality.DBT: ['mindfulness', 'distress tolerance', 'emotion regulation', 'skills'],
            TherapeuticModality.PSYCHODYNAMIC: ['patterns', 'unconscious', 'relationships', 'past', 'insight'],
            TherapeuticModality.HUMANISTIC: ['feelings', 'experience', 'authentic', 'growth', 'potential']
        }
        
        expected_keywords = modality_keywords.get(modality, [])
        content_lower = response.content.lower()
        
        # Check if response contains modality-appropriate language
        keyword_count = sum(1 for keyword in expected_keywords if keyword in content_lower)
        
        if keyword_count == 0 and len(expected_keywords) > 0:
            # Add modality-appropriate language
            random.choice(expected_keywords)
            if modality == TherapeuticModality.CBT:
                response.content += " What thoughts come up for you about this?"
            elif modality == TherapeuticModality.DBT:
                response.content += " Let's think about what skills might be helpful here."
            elif modality == TherapeuticModality.PSYCHODYNAMIC:
                response.content += " I'm curious about the patterns you're noticing."
            elif modality == TherapeuticModality.HUMANISTIC:
                response.content += " How does this feel for you right now?"
        
        return response
    
    async def _add_contextual_content(self, base_response: str, parameters: ConversationParameters,
                                    last_therapist_turn: Optional[ConversationTurn],
                                    clinical_knowledge: List[Dict[str, Any]]) -> str:
        """Add contextual content to client response"""
        # Add presentation-specific content
        if "depression" in parameters.client_presentation.lower():
            if "tired" not in base_response.lower():
                base_response += " I've been feeling so tired lately."
        elif "anxiety" in parameters.client_presentation.lower():
            if "worry" not in base_response.lower():
                base_response += " I can't stop worrying about everything."
        
        # Add crisis-related content if applicable
        if parameters.crisis_indicators and parameters.client_response_style == ClientResponseStyle.CRISIS:
            crisis_content = [
                "I don't know how much longer I can handle this.",
                "Everything feels overwhelming.",
                "I feel like I'm drowning."
            ]
            base_response += f" {random.choice(crisis_content)}"
        
        # Add cultural context if relevant
        if parameters.cultural_factors:
            cultural_additions = {
                'hispanic/latino': "My family doesn't really understand mental health.",
                'asian': "I feel like I'm bringing shame to my family.",
                'african american': "I've always had to be strong for everyone else."
            }
            
            for factor in parameters.cultural_factors:
                if factor.lower() in cultural_additions:
                    if random.random() < 0.3:  # 30% chance to add cultural context
                        base_response += f" {cultural_additions[factor.lower()]}"
                    break
        
        return base_response
    
    async def _validate_conversation_turns(self, turns: List[ConversationTurn], 
                                         parameters: ConversationParameters) -> List[ConversationTurn]:
        """Validate and potentially regenerate conversation turns"""
        validated_turns = []
        
        for i, turn in enumerate(turns):
            # Validate individual turn
            is_valid, issues = await self._validate_turn(turn, parameters, turns[:i])
            
            if is_valid:
                validated_turns.append(turn)
            else:
                logger.warning(f"Turn {turn.turn_id} validation failed: {issues}")
                
                # Attempt to regenerate turn
                if turn.speaker == ConversationRole.THERAPIST:
                    regenerated_turn = await self._regenerate_therapist_turn(
                        turn, parameters, validated_turns, issues
                    )
                else:
                    regenerated_turn = await self._regenerate_client_turn(
                        turn, parameters, validated_turns, issues
                    )
                
                validated_turns.append(regenerated_turn)
        
        return validated_turns
    
    async def _validate_turn(self, turn: ConversationTurn, parameters: ConversationParameters,
                           previous_turns: List[ConversationTurn]) -> Tuple[bool, List[str]]:
        """Validate individual conversation turn"""
        issues = []
        
        # Check content length
        if len(turn.content.strip()) < 5:
            issues.append("Content too short")
        elif len(turn.content.strip()) > 500:
            issues.append("Content too long")
        
        # Check for inappropriate content
        inappropriate_terms = ['suicide', 'kill', 'die', 'hurt'] if not parameters.crisis_indicators else []
        for term in inappropriate_terms:
            if term in turn.content.lower() and turn.speaker == ConversationRole.CLIENT:
                if parameters.client_response_style != ClientResponseStyle.CRISIS:
                    issues.append(f"Inappropriate content: {term}")
        
        # Check therapeutic appropriateness for therapist turns
        if turn.speaker == ConversationRole.THERAPIST:
            if not turn.clinical_rationale:
                issues.append("Missing clinical rationale")
            
            if turn.confidence_score < self.generation_settings['quality_threshold']:
                issues.append("Low confidence score")
        
        # Check conversation flow
        if previous_turns:
            last_turn = previous_turns[-1]
            if last_turn.speaker == turn.speaker:
                issues.append("Speaker continuity error")
        
        return len(issues) == 0, issues
    
    async def _regenerate_therapist_turn(self, original_turn: ConversationTurn,
                                       parameters: ConversationParameters,
                                       previous_turns: List[ConversationTurn],
                                       issues: List[str]) -> ConversationTurn:
        """Regenerate therapist turn to address validation issues"""
        # Get last client turn
        last_client_turn = None
        for turn in reversed(previous_turns):
            if turn.speaker == ConversationRole.CLIENT:
                last_client_turn = turn
                break
        
        if not last_client_turn:
            # Fallback response
            return ConversationTurn(
                turn_id=original_turn.turn_id,
                speaker=ConversationRole.THERAPIST,
                content="I hear you. Can you tell me more about that?",
                clinical_rationale="Reflective listening to encourage elaboration",
                intervention_type=InterventionType.REFLECTION,
                therapeutic_technique="Active listening",
                confidence_score=0.8
            )
        
        # Generate new response with stricter validation
        clinical_context = self._build_clinical_context(parameters, previous_turns)
        
        new_response = self.response_generator.generate_response(
            client_statement=last_client_turn.content,
            clinical_context=clinical_context
        )
        
        return ConversationTurn(
            turn_id=original_turn.turn_id,
            speaker=ConversationRole.THERAPIST,
            content=new_response.content,
            clinical_rationale=new_response.clinical_rationale,
            intervention_type=new_response.intervention_type,
            therapeutic_technique=new_response.therapeutic_technique,
            confidence_score=new_response.confidence_score,
            metadata=original_turn.metadata
        )
    
    async def _regenerate_client_turn(self, original_turn: ConversationTurn,
                                    parameters: ConversationParameters,
                                    previous_turns: List[ConversationTurn],
                                    issues: List[str]) -> ConversationTurn:
        """Regenerate client turn to address validation issues"""
        # Generate safer, more appropriate client response
        safe_responses = {
            ClientResponseStyle.COOPERATIVE: [
                "I understand what you're saying.",
                "That makes sense to me.",
                "I'd like to work on that."
            ],
            ClientResponseStyle.RESISTANT: [
                "I'm not sure about that approach.",
                "I've heard that before.",
                "That seems difficult for me."
            ],
            ClientResponseStyle.AMBIVALENT: [
                "I'm not sure how I feel about that.",
                "Part of me agrees, but part of me doesn't.",
                "I need to think about that more."
            ],
            ClientResponseStyle.CRISIS: [
                "I'm struggling with this.",
                "This is really hard for me.",
                "I need help with this."
            ]
        }
        
        safe_content = random.choice(
            safe_responses.get(parameters.client_response_style, safe_responses[ClientResponseStyle.COOPERATIVE])
        )
        
        return ConversationTurn(
            turn_id=original_turn.turn_id,
            speaker=ConversationRole.CLIENT,
            content=safe_content,
            confidence_score=0.7,
            metadata=original_turn.metadata
        )
    
    async def _generate_clinical_summary(self, turns: List[ConversationTurn], 
                                       parameters: ConversationParameters) -> str:
        """Generate clinical summary of the conversation"""
        # Analyze conversation content
        therapist_interventions = [turn for turn in turns if turn.speaker == ConversationRole.THERAPIST]
        client_responses = [turn for turn in turns if turn.speaker == ConversationRole.CLIENT]
        
        # Count intervention types
        intervention_counts: Dict[str, int] = {}
        for turn in therapist_interventions:
            if turn.intervention_type:
                intervention_counts[turn.intervention_type.value] = intervention_counts.get(turn.intervention_type.value, 0) + 1
        
        # Assess client engagement
        avg_client_response_length = np.mean([len(turn.content.split()) for turn in client_responses])
        
        # Generate summary
        summary_parts = [
            f"Therapeutic conversation using {parameters.therapeutic_modality.value} approach",
            f"Session {parameters.session_number}, {parameters.conversation_phase.value} phase",
            f"Client presentation: {parameters.client_presentation}",
            f"Total exchanges: {len(turns)}",
            f"Primary interventions: {', '.join(intervention_counts.keys())}",
            f"Client engagement level: {'High' if avg_client_response_length > 15 else 'Moderate' if avg_client_response_length > 8 else 'Low'}"
        ]
        
        if parameters.therapeutic_goals:
            summary_parts.append(f"Therapeutic goals addressed: {', '.join(parameters.therapeutic_goals[:3])}")
        
        if parameters.crisis_indicators:
            summary_parts.append(f"Crisis indicators present: {', '.join(parameters.crisis_indicators)}")
        
        return ". ".join(summary_parts) + "."
    
    async def _calculate_quality_metrics(self, turns: List[ConversationTurn], 
                                       parameters: ConversationParameters) -> Dict[str, float]:
        """Calculate quality metrics for the conversation"""
        therapist_turns = [turn for turn in turns if turn.speaker == ConversationRole.THERAPIST]
        client_turns = [turn for turn in turns if turn.speaker == ConversationRole.CLIENT]
        
        # Clinical accuracy (average confidence of therapist turns)
        clinical_accuracy = np.mean([turn.confidence_score for turn in therapist_turns]) if therapist_turns else 0.0
        
        # Therapeutic appropriateness (based on intervention types and rationales)
        appropriate_interventions = sum(1 for turn in therapist_turns 
                                      if turn.clinical_rationale and len(turn.clinical_rationale) > 10)
        therapeutic_appropriateness = appropriate_interventions / len(therapist_turns) if therapist_turns else 0.0
        
        # Conversation flow (consistency and naturalness)
        flow_score = 1.0  # Start with perfect score
        for i in range(1, len(turns)):
            if turns[i].speaker == turns[i-1].speaker:
                flow_score -= 0.1  # Penalize speaker continuity errors
        flow_score = max(0.0, flow_score)
        
        # Engagement level (based on client response lengths and variety)
        if client_turns:
            avg_length = np.mean([len(turn.content.split()) for turn in client_turns])
            length_variety = np.std([len(turn.content.split()) for turn in client_turns])
            engagement = min(1.0, float(avg_length / 20) + float(length_variety / 10))
        else:
            engagement = 0.0
        
        # Modality adherence (check for modality-specific language)
        modality_keywords = {
            TherapeuticModality.CBT: ['thoughts', 'feelings', 'behaviors', 'evidence'],
            TherapeuticModality.DBT: ['mindfulness', 'skills', 'distress', 'emotion'],
            TherapeuticModality.PSYCHODYNAMIC: ['patterns', 'relationships', 'unconscious', 'insight'],
            TherapeuticModality.HUMANISTIC: ['feelings', 'experience', 'authentic', 'growth']
        }
        
        expected_keywords = modality_keywords.get(parameters.therapeutic_modality, [])
        keyword_mentions = 0
        total_content = " ".join([turn.content for turn in therapist_turns]).lower()
        
        for keyword in expected_keywords:
            if keyword in total_content:
                keyword_mentions += 1
        
        modality_adherence = keyword_mentions / len(expected_keywords) if expected_keywords else 1.0
        
        # Overall quality (weighted combination)
        overall_quality = (
            clinical_accuracy * self.generation_settings['clinical_accuracy_weight'] +
            therapeutic_appropriateness * self.generation_settings['therapeutic_appropriateness_weight'] +
            flow_score * self.generation_settings['naturalness_weight']
        )
        
        return {
            'clinical_accuracy': round(float(clinical_accuracy), 3),
            'therapeutic_appropriateness': round(float(therapeutic_appropriateness), 3),
            'conversation_flow': round(float(flow_score), 3),
            'client_engagement': round(float(engagement), 3),
            'modality_adherence': round(float(modality_adherence), 3),
            'overall_quality': round(float(overall_quality), 3)
        }
    
    async def _assess_therapeutic_progress(self, turns: List[ConversationTurn], 
                                         parameters: ConversationParameters) -> Dict[str, Any]:
        """Assess therapeutic progress made during conversation"""
        [turn for turn in turns if turn.speaker == ConversationRole.THERAPIST]
        client_turns = [turn for turn in turns if turn.speaker == ConversationRole.CLIENT]
        
        # Analyze progression through conversation phases
        phase_progression = self._analyze_phase_progression(turns, parameters)
        
        # Assess goal achievement
        goal_progress = {}
        for goal in parameters.therapeutic_goals:
            # Simple keyword matching for goal progress
            goal_mentions = sum(1 for turn in turns if goal.lower() in turn.content.lower())
            goal_progress[goal] = min(1.0, goal_mentions / 3)  # Normalize to 0-1
        
        # Assess client insight development
        insight_indicators = ['understand', 'realize', 'see', 'notice', 'aware', 'insight']
        client_insights = sum(1 for turn in client_turns 
                            for indicator in insight_indicators 
                            if indicator in turn.content.lower())
        
        insight_development = min(1.0, client_insights / len(client_turns)) if client_turns else 0.0
        
        # Assess skill acquisition (for skill-building phases)
        skill_indicators = ['try', 'practice', 'use', 'apply', 'technique', 'skill']
        skill_mentions = sum(1 for turn in client_turns 
                           for indicator in skill_indicators 
                           if indicator in turn.content.lower())
        
        skill_acquisition = min(1.0, skill_mentions / max(1, len(client_turns) // 2)) if client_turns else 0.0
        
        # Assess emotional regulation progress
        emotion_words_negative = ['anxious', 'depressed', 'angry', 'frustrated', 'overwhelmed']
        emotion_words_positive = ['calm', 'better', 'hopeful', 'confident', 'peaceful']
        
        early_turns = client_turns[:len(client_turns)//2] if len(client_turns) > 2 else client_turns
        late_turns = client_turns[len(client_turns)//2:] if len(client_turns) > 2 else []
        
        early_negative = sum(1 for turn in early_turns 
                           for word in emotion_words_negative 
                           if word in turn.content.lower())
        late_positive = sum(1 for turn in late_turns 
                          for word in emotion_words_positive 
                          if word in turn.content.lower())
        
        emotional_progress = (late_positive - early_negative) / max(1, len(client_turns))
        
        return {
            'phase_progression': phase_progression,
            'goal_achievement': goal_progress,
            'insight_development': round(insight_development, 3),
            'skill_acquisition': round(skill_acquisition, 3),
            'emotional_regulation_progress': round(emotional_progress, 3),
            'overall_progress_score': round(
                (insight_development + skill_acquisition + max(0, emotional_progress)) / 3, 3
            )
        }
    
    def _analyze_phase_progression(self, turns: List[ConversationTurn], 
                                 parameters: ConversationParameters) -> Dict[str, Any]:
        """Analyze progression through therapeutic conversation phases"""
        # Define phase indicators
        phase_indicators = {
            'rapport_building': ['comfortable', 'safe', 'trust', 'understand'],
            'problem_exploration': ['tell me', 'describe', 'what', 'how', 'when'],
            'intervention_planning': ['goal', 'plan', 'strategy', 'approach'],
            'skill_building': ['practice', 'try', 'technique', 'skill', 'exercise'],
            'progress_monitoring': ['better', 'worse', 'change', 'progress', 'improvement']
        }
        
        phase_scores = {}
        total_content = " ".join([turn.content for turn in turns]).lower()
        
        for phase, indicators in phase_indicators.items():
            score = sum(1 for indicator in indicators if indicator in total_content)
            phase_scores[phase] = score / len(indicators)  # Normalize
        
        # Determine dominant phase
        dominant_phase = max(phase_scores.keys(), key=lambda x: phase_scores[x])
        
        return {
            'phase_scores': phase_scores,
            'dominant_phase': dominant_phase,
            'phase_consistency': phase_scores.get(parameters.conversation_phase.value, 0.0)
        }
    
    async def _generate_recommendations(self, turns: List[ConversationTurn], 
                                      parameters: ConversationParameters,
                                      quality_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations for conversation improvement"""
        recommendations = []
        
        # Quality-based recommendations
        if quality_metrics['clinical_accuracy'] < 0.7:
            recommendations.append("Improve clinical accuracy by providing more detailed rationales")
        
        if quality_metrics['therapeutic_appropriateness'] < 0.7:
            recommendations.append("Enhance therapeutic appropriateness with evidence-based interventions")
        
        if quality_metrics['conversation_flow'] < 0.8:
            recommendations.append("Improve conversation flow and natural transitions")
        
        if quality_metrics['client_engagement'] < 0.5:
            recommendations.append("Increase client engagement through more open-ended questions")
        
        if quality_metrics['modality_adherence'] < 0.6:
            recommendations.append(f"Better integrate {parameters.therapeutic_modality.value} techniques and language")
        
        # Content-based recommendations
        therapist_turns = [turn for turn in turns if turn.speaker == ConversationRole.THERAPIST]
        
        # Check for intervention variety
        intervention_types = set(turn.intervention_type.value for turn in therapist_turns if turn.intervention_type)
        if len(intervention_types) < 3:
            recommendations.append("Incorporate more diverse therapeutic interventions")
        
        # Check for crisis handling
        if parameters.crisis_indicators and not any('crisis' in turn.content.lower() for turn in therapist_turns):
            recommendations.append("Address crisis indicators more directly in therapeutic responses")
        
        # Check for cultural sensitivity
        if parameters.cultural_factors and not any(
            any(factor.lower() in turn.content.lower() for factor in parameters.cultural_factors)
            for turn in therapist_turns
        ):
            recommendations.append("Incorporate more cultural sensitivity and awareness")
        
        # Length-based recommendations
        if len(turns) < parameters.conversation_length * 0.8:
            recommendations.append("Extend conversation to meet target length for better therapeutic development")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    async def _identify_warnings(self, turns: List[ConversationTurn], 
                               parameters: ConversationParameters) -> List[str]:
        """Identify potential warnings or concerns in the conversation"""
        warnings = []
        
        # Safety warnings
        safety_keywords = ['suicide', 'kill', 'die', 'hurt', 'harm', 'end it all']
        for turn in turns:
            if turn.speaker == ConversationRole.CLIENT:
                for keyword in safety_keywords:
                    if keyword in turn.content.lower():
                        warnings.append(f"Safety concern: Client mentioned '{keyword}' - requires immediate attention")
                        break
        
        # Therapeutic boundary warnings
        boundary_issues = ['personal', 'friend', 'relationship', 'date', 'outside']
        therapist_turns = [turn for turn in turns if turn.speaker == ConversationRole.THERAPIST]
        
        for turn in therapist_turns:
            for issue in boundary_issues:
                if issue in turn.content.lower() and 'boundary' not in turn.content.lower():
                    warnings.append("Potential boundary issue in therapist response")
                    break
        
        # Quality warnings
        low_confidence_turns = [turn for turn in therapist_turns if turn.confidence_score < 0.5]
        if len(low_confidence_turns) > len(therapist_turns) * 0.3:
            warnings.append("High number of low-confidence therapist responses")
        
        # Modality consistency warnings
        if parameters.therapeutic_modality == TherapeuticModality.CBT:
            cbt_keywords = ['thoughts', 'feelings', 'behaviors']
            if not any(keyword in " ".join([turn.content for turn in therapist_turns]).lower() 
                      for keyword in cbt_keywords):
                warnings.append("CBT conversation lacks core CBT elements")
        
        # Crisis handling warnings
        if parameters.crisis_indicators:
            crisis_addressed = any('crisis' in turn.content.lower() or 'safety' in turn.content.lower() 
                                 for turn in therapist_turns)
            if not crisis_addressed:
                warnings.append("Crisis indicators present but not adequately addressed")
        
        return warnings
    
    def _assess_complexity_achieved(self, turns: List[ConversationTurn]) -> str:
        """Assess the complexity level achieved in the conversation"""
        # Analyze language complexity
        therapist_content = " ".join([turn.content for turn in turns if turn.speaker == ConversationRole.THERAPIST])
        
        # Count clinical terms
        clinical_terms = ['intervention', 'therapeutic', 'cognitive', 'behavioral', 'psychodynamic', 
                         'assessment', 'diagnosis', 'symptom', 'treatment']
        clinical_term_count = sum(1 for term in clinical_terms if term in therapist_content.lower())
        
        # Count complex sentence structures
        complex_sentences = therapist_content.count(',') + therapist_content.count(';')
        
        # Assess based on metrics
        if clinical_term_count >= 5 and complex_sentences >= 10:
            return ConversationComplexity.EXPERT.value
        elif clinical_term_count >= 3 and complex_sentences >= 6:
            return ConversationComplexity.ADVANCED.value
        elif clinical_term_count >= 1 and complex_sentences >= 3:
            return ConversationComplexity.INTERMEDIATE.value
        else:
            return ConversationComplexity.BASIC.value
    
    def _assess_modality_adherence(self, turns: List[ConversationTurn], 
                                 modality: TherapeuticModality) -> float:
        """Assess adherence to specified therapeutic modality"""
        therapist_turns = [turn for turn in turns if turn.speaker == ConversationRole.THERAPIST]
        
        if not therapist_turns:
            return 0.0
        
        # Count modality-specific interventions
        modality_interventions = 0
        total_interventions = 0
        
        for turn in therapist_turns:
            if turn.intervention_type:
                total_interventions += 1
                
                # Check if intervention aligns with modality
                if modality == TherapeuticModality.CBT:
                    if turn.intervention_type in [InterventionType.COGNITIVE_RESTRUCTURING, 
                                                InterventionType.BEHAVIORAL_ACTIVATION,
                                                InterventionType.PSYCHOEDUCATION]:
                        modality_interventions += 1
                elif modality == TherapeuticModality.DBT:
                    if turn.intervention_type in [InterventionType.SKILL_BUILDING,
                                                InterventionType.VALIDATION]:
                        modality_interventions += 1
                elif modality == TherapeuticModality.PSYCHODYNAMIC:
                    if turn.intervention_type in [InterventionType.INTERPRETATION,
                                                InterventionType.EXPLORATION]:
                        modality_interventions += 1
                elif modality == TherapeuticModality.HUMANISTIC:
                    if turn.intervention_type in [InterventionType.REFLECTION,
                                                InterventionType.VALIDATION]:
                        modality_interventions += 1
        
        return modality_interventions / total_interventions if total_interventions > 0 else 0.0
    
    async def generate_conversation_batch(self, parameter_list: List[ConversationParameters]) -> List[GeneratedConversation]:
        """Generate multiple conversations in batch"""
        conversations = []
        
        for i, parameters in enumerate(parameter_list):
            try:
                logger.info(f"Generating conversation {i+1}/{len(parameter_list)}")
                conversation = await self.generate_conversation(parameters)
                conversations.append(conversation)
            except Exception as e:
                logger.error(f"Error generating conversation {i+1}: {e}")
                continue
        
        return conversations
    
    async def export_conversation(self, conversation: GeneratedConversation, 
                                format: str = 'json', output_path: Optional[Path] = None) -> Path:
        """Export conversation in specified format"""
        if not output_path:
            output_path = Path(f"conversations/{conversation.conversation_id}.{format}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            conversation_dict = {
                'conversation_id': conversation.conversation_id,
                'parameters': {
                    'therapeutic_modality': conversation.parameters.therapeutic_modality.value,
                    'client_presentation': conversation.parameters.client_presentation,
                    'primary_diagnosis': conversation.parameters.primary_diagnosis,
                    'conversation_phase': conversation.parameters.conversation_phase.value,
                    'complexity_level': conversation.parameters.complexity_level.value,
                    'client_response_style': conversation.parameters.client_response_style.value,
                    'session_number': conversation.parameters.session_number,
                    'therapeutic_goals': conversation.parameters.therapeutic_goals,
                    'cultural_factors': conversation.parameters.cultural_factors,
                    'crisis_indicators': conversation.parameters.crisis_indicators
                },
                'turns': [
                    {
                        'turn_id': turn.turn_id,
                        'speaker': turn.speaker.value,
                        'content': turn.content,
                        'clinical_rationale': turn.clinical_rationale,
                        'intervention_type': turn.intervention_type.value if turn.intervention_type else None,
                        'therapeutic_technique': turn.therapeutic_technique,
                        'confidence_score': turn.confidence_score,
                        'timestamp': turn.timestamp.isoformat(),
                        'metadata': turn.metadata
                    }
                    for turn in conversation.turns
                ],
                'clinical_summary': conversation.clinical_summary,
                'therapeutic_progress': conversation.therapeutic_progress,
                'quality_metrics': conversation.quality_metrics,
                'recommendations': conversation.recommendations,
                'warnings': conversation.warnings,
                'timestamp': conversation.timestamp.isoformat(),
                'metadata': conversation.metadata
            }
            
            with open(output_path, 'w') as f:
                json.dump(conversation_dict, f, indent=2, default=str)
        
        elif format == 'txt':
            with open(output_path, 'w') as f:
                f.write(f"Therapeutic Conversation: {conversation.conversation_id}\n")
                f.write(f"Modality: {conversation.parameters.therapeutic_modality.value}\n")
                f.write(f"Phase: {conversation.parameters.conversation_phase.value}\n")
                f.write(f"Generated: {conversation.timestamp}\n\n")
                
                for turn in conversation.turns:
                    speaker = "Therapist" if turn.speaker == ConversationRole.THERAPIST else "Client"
                    f.write(f"{speaker}: {turn.content}\n")
                    if turn.clinical_rationale:
                        f.write(f"  [Clinical Rationale: {turn.clinical_rationale}]\n")
                    f.write("\n")
                
                f.write(f"\nClinical Summary: {conversation.clinical_summary}\n")
                f.write(f"\nQuality Metrics: {conversation.quality_metrics}\n")
                
                if conversation.recommendations:
                    f.write("\nRecommendations:\n")
                    for rec in conversation.recommendations:
                        f.write(f"- {rec}\n")
                
                if conversation.warnings:
                    f.write("\nWarnings:\n")
                    for warning in conversation.warnings:
                        f.write(f"- {warning}\n")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Conversation exported to {output_path}")
        return output_path
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics about conversation generation"""
        return {
            'supported_modalities': [modality.value for modality in TherapeuticModality],
            'supported_phases': [phase.value for phase in ConversationPhase],
            'complexity_levels': [level.value for level in ConversationComplexity],
            'client_response_styles': [style.value for style in ClientResponseStyle],
            'generation_settings': self.generation_settings,
            'configuration': self.config
        }
