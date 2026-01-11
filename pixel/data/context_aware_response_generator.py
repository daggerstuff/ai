"""
Context-Aware Therapeutic Response Generation System

Generates therapeutic responses that are highly aware of conversational context,
client state, therapeutic relationship, session dynamics, and treatment progress.
Integrates with modality integration system for contextually appropriate responses.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dynamic_conversation_generator import (
    ConversationPhase,
    ConversationRole,
    ConversationTurn,
)
from .therapeutic_conversation_schema import ClinicalContext
from .therapeutic_modality_integrator import TherapeuticModalityIntegrator
from .therapist_response_generator import TherapistResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextualFactor(Enum):
    """Types of contextual factors that influence response generation"""
    EMOTIONAL_STATE = "emotional_state"
    THERAPEUTIC_ALLIANCE = "therapeutic_alliance"
    SESSION_PHASE = "session_phase"
    TREATMENT_PROGRESS = "treatment_progress"
    CRISIS_LEVEL = "crisis_level"
    RESISTANCE_PATTERN = "resistance_pattern"
    ENGAGEMENT_LEVEL = "engagement_level"
    CULTURAL_CONTEXT = "cultural_context"
    TRAUMA_INDICATORS = "trauma_indicators"
    COGNITIVE_CAPACITY = "cognitive_capacity"


class ResponseContextType(Enum):
    """Types of response contexts"""
    OPENING = "opening"
    EXPLORATION = "exploration"
    INTERVENTION = "intervention"
    PROCESSING = "processing"
    TRANSITION = "transition"
    CRISIS_RESPONSE = "crisis_response"
    RESISTANCE_HANDLING = "resistance_handling"
    CLOSURE = "closure"


class ContextualPriority(Enum):
    """Priority levels for contextual factors"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ConversationalContext:
    """Comprehensive conversational context analysis"""
    turn_number: int
    session_number: int
    time_in_session: timedelta
    conversation_phase: ConversationPhase
    emotional_trajectory: List[Tuple[str, float]]  # (emotion, intensity)
    alliance_strength: float
    engagement_level: float
    resistance_indicators: List[str]
    progress_markers: List[str]
    crisis_indicators: List[str]
    cultural_considerations: List[str]
    cognitive_load: float
    therapeutic_momentum: float
    recent_breakthroughs: List[str]
    stuck_patterns: List[str]
    client_preferences: List[str]
    contraindications: List[str]


@dataclass
class ContextualWeight:
    """Weight assigned to different contextual factors"""
    factor: ContextualFactor
    weight: float
    rationale: str
    confidence: float
    temporal_decay: float  # How quickly this factor loses relevance


@dataclass
class ResponseContext:
    """Context for generating a specific response"""
    context_type: ResponseContextType
    primary_factors: List[ContextualFactor]
    contextual_weights: Dict[ContextualFactor, float]
    immediate_needs: List[str]
    therapeutic_opportunities: List[str]
    potential_risks: List[str]
    recommended_techniques: List[str]
    contraindicated_approaches: List[str]
    timing_considerations: List[str]


@dataclass
class ContextualResponse:
    """Response generated with full contextual awareness"""
    response: TherapistResponse
    context_analysis: ConversationalContext
    response_context: ResponseContext
    contextual_adaptations: List[str]
    risk_mitigations: List[str]
    opportunity_capitalizations: List[str]
    follow_up_recommendations: List[str]
    context_monitoring_alerts: List[str]


class ContextAwareResponseGenerator:
    """
    Context-aware therapeutic response generation system
    
    Generates responses that are highly attuned to conversational context,
    client state, therapeutic relationship dynamics, and treatment progress.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the context-aware response generator"""
        self.config = self._load_config(config_path)
        self.modality_integrator = TherapeuticModalityIntegrator()
        
        # Load contextual patterns and rules
        self.contextual_patterns = self._load_contextual_patterns()
        self.response_templates = self._load_response_templates()
        self.adaptation_rules = self._load_adaptation_rules()
        
        # Context analysis settings
        self.context_settings = {
            'emotional_window_size': 5,  # Number of turns to analyze for emotion
            'alliance_decay_rate': 0.1,  # How quickly alliance strength decays
            'momentum_threshold': 0.7,   # Threshold for therapeutic momentum
            'crisis_sensitivity': 0.8,   # Sensitivity to crisis indicators
            'resistance_threshold': 0.6, # Threshold for resistance detection
            'engagement_window': 3,      # Turns to analyze for engagement
            'cultural_weight': 0.3       # Weight for cultural considerations
        }
        
        # Track context history
        self.context_history: List[ConversationalContext] = []
        self.response_history: List[ContextualResponse] = []
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration settings"""
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        return {
            'enable_deep_context_analysis': True,
            'prioritize_safety': True,
            'adapt_to_resistance': True,
            'monitor_alliance': True,
            'track_progress': True,
            'cultural_sensitivity': True,
            'crisis_override': True,
            'max_context_history': 50
        }
    
    def _load_contextual_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load patterns for contextual analysis"""
        return {
            'emotional_escalation': {
                'indicators': ['getting worse', 'more intense', 'overwhelming', 'can\'t handle'],
                'response_adaptations': ['validate_intensity', 'slow_down', 'ground', 'stabilize'],
                'contraindications': ['challenging', 'pushing', 'interpreting'],
                'priority': ContextualPriority.HIGH
            },
            
            'therapeutic_breakthrough': {
                'indicators': ['I see', 'makes sense', 'never thought', 'understand now'],
                'response_adaptations': ['reinforce_insight', 'deepen_exploration', 'connect_patterns'],
                'opportunities': ['expand_understanding', 'apply_insight', 'generalize'],
                'priority': ContextualPriority.HIGH
            },
            
            'resistance_emergence': {
                'indicators': ['but', 'however', 'don\'t think', 'won\'t work', 'tried that'],
                'response_adaptations': ['validate_concern', 'explore_resistance', 'adjust_approach'],
                'contraindications': ['insisting', 'arguing', 'pushing_agenda'],
                'priority': ContextualPriority.MEDIUM
            },
            
            'alliance_strain': {
                'indicators': ['don\'t understand', 'not helping', 'waste of time', 'don\'t get it'],
                'response_adaptations': ['repair_alliance', 'acknowledge_concern', 'adjust_style'],
                'immediate_needs': ['validation', 'understanding', 'collaboration'],
                'priority': ContextualPriority.CRITICAL
            },
            
            'crisis_emergence': {
                'indicators': ['suicide', 'kill', 'die', 'hurt myself', 'end it all', 'can\'t go on'],
                'response_adaptations': ['crisis_protocol', 'safety_assessment', 'immediate_support'],
                'contraindications': ['minimizing', 'rushing', 'normal_therapy'],
                'priority': ContextualPriority.CRITICAL
            },
            
            'cultural_sensitivity_needed': {
                'indicators': ['family says', 'my culture', 'people like me', 'tradition', 'community'],
                'response_adaptations': ['cultural_validation', 'explore_context', 'honor_values'],
                'considerations': ['family_dynamics', 'cultural_norms', 'identity_factors'],
                'priority': ContextualPriority.MEDIUM
            },
            
            'cognitive_overload': {
                'indicators': ['confused', 'too much', 'don\'t follow', 'overwhelming information'],
                'response_adaptations': ['simplify', 'slow_down', 'check_understanding', 'break_down'],
                'contraindications': ['complex_interventions', 'multiple_concepts', 'rapid_pace'],
                'priority': ContextualPriority.HIGH
            },
            
            'engagement_peak': {
                'indicators': ['want to try', 'makes sense', 'helpful', 'ready to work'],
                'response_adaptations': ['capitalize_motivation', 'introduce_techniques', 'set_goals'],
                'opportunities': ['skill_building', 'homework_assignment', 'practice'],
                'priority': ContextualPriority.MEDIUM
            }
        }
    
    def _load_response_templates(self) -> Dict[ResponseContextType, Dict[str, List[str]]]:
        """Load response templates for different contexts"""
        return {
            ResponseContextType.OPENING: {
                'standard': [
                    "How are you feeling today?",
                    "What's been on your mind since we last met?",
                    "How can I best support you in our time together today?"
                ],
                'crisis_aware': [
                    "I'm glad you're here today. How are you doing right now?",
                    "Thank you for coming in. What feels most important to talk about today?",
                    "I want to check in with you - how are you feeling in this moment?"
                ],
                'alliance_repair': [
                    "I've been thinking about our last session. How are you feeling about our work together?",
                    "I want to make sure we're on the same page. How has therapy been feeling for you?",
                    "I'm curious about your experience of our sessions so far."
                ]
            },
            
            ResponseContextType.CRISIS_RESPONSE: {
                'immediate_safety': [
                    "I'm really concerned about what you just shared. Can you help me understand how you're feeling right now?",
                    "Thank you for trusting me with this. Your safety is my priority. Let's talk about what's happening.",
                    "I hear how much pain you're in. You're not alone in this. Let's work together to keep you safe."
                ],
                'assessment': [
                    "Can you tell me more about these thoughts? How often are you having them?",
                    "Help me understand - do you have a plan for how you might hurt yourself?",
                    "What's been keeping you safe so far? What's helped you get through difficult moments?"
                ],
                'support': [
                    "I'm here with you. We're going to work through this together.",
                    "You showed incredible strength by sharing this with me.",
                    "These feelings are temporary, even though they feel overwhelming right now."
                ]
            },
            
            ResponseContextType.RESISTANCE_HANDLING: {
                'validation': [
                    "I can understand why you might feel that way.",
                    "It makes sense that you'd have concerns about this approach.",
                    "Your hesitation tells me something important about your experience."
                ],
                'exploration': [
                    "Help me understand what doesn't feel right about this for you.",
                    "What would need to be different for this to feel more helpful?",
                    "I'm curious about what's behind your concern."
                ],
                'collaboration': [
                    "Let's figure out together what might work better for you.",
                    "What approach would feel more comfortable or helpful?",
                    "How can we adjust this to better fit your needs?"
                ]
            },
            
            ResponseContextType.PROCESSING: {
                'insight_reinforcement': [
                    "That's a really important realization. What does that mean for you?",
                    "I can see this is a significant insight. How does it feel to recognize this?",
                    "This understanding seems meaningful. How might this change things for you?"
                ],
                'pattern_connection': [
                    "I'm noticing a connection between what you just said and what we discussed earlier.",
                    "This seems to relate to the pattern we've been exploring.",
                    "How does this fit with what you've been learning about yourself?"
                ],
                'integration': [
                    "How can you take this understanding with you into your daily life?",
                    "What would it look like to apply this insight outside of our sessions?",
                    "How might this awareness change how you handle similar situations?"
                ]
            }
        }
    
    def _load_adaptation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load rules for adapting responses based on context"""
        return {
            'emotional_intensity_high': {
                'adaptations': [
                    'Use slower pace',
                    'Increase validation',
                    'Simplify language',
                    'Focus on grounding',
                    'Avoid complex interventions'
                ],
                'language_adjustments': {
                    'tone': 'calm_and_steady',
                    'pace': 'slower',
                    'complexity': 'simple',
                    'validation_level': 'high'
                }
            },
            
            'alliance_strength_low': {
                'adaptations': [
                    'Prioritize relationship repair',
                    'Increase collaboration',
                    'Reduce directiveness',
                    'Explore client concerns',
                    'Adjust therapeutic style'
                ],
                'language_adjustments': {
                    'tone': 'collaborative',
                    'directiveness': 'low',
                    'validation_level': 'high',
                    'exploration': 'increased'
                }
            },
            
            'resistance_level_high': {
                'adaptations': [
                    'Validate resistance',
                    'Explore underlying concerns',
                    'Reduce pressure',
                    'Increase client choice',
                    'Adjust expectations'
                ],
                'language_adjustments': {
                    'tone': 'accepting',
                    'pressure': 'minimal',
                    'choice_emphasis': 'high',
                    'pace': 'client_led'
                }
            },
            
            'cognitive_capacity_low': {
                'adaptations': [
                    'Simplify concepts',
                    'Use concrete examples',
                    'Check understanding frequently',
                    'Slow down pace',
                    'Repeat key points'
                ],
                'language_adjustments': {
                    'complexity': 'very_simple',
                    'examples': 'concrete',
                    'pace': 'very_slow',
                    'repetition': 'increased'
                }
            },
            
            'cultural_factors_present': {
                'adaptations': [
                    'Acknowledge cultural context',
                    'Explore cultural values',
                    'Adapt interventions culturally',
                    'Honor family dynamics',
                    'Consider community factors'
                ],
                'language_adjustments': {
                    'cultural_sensitivity': 'high',
                    'family_inclusion': 'considered',
                    'value_alignment': 'prioritized'
                }
            }
        }
    
    async def analyze_conversational_context(self, conversation_history: List[ConversationTurn],
                                           clinical_context: ClinicalContext,
                                           session_info: Dict[str, Any]) -> ConversationalContext:
        """Analyze comprehensive conversational context"""
        try:
            if not conversation_history:
                return self._create_initial_context(clinical_context, session_info)
            
            # Basic context information
            turn_number = len(conversation_history)
            session_number = session_info.get('session_number', 1)
            session_start = session_info.get('session_start', datetime.now())
            time_in_session = datetime.now() - session_start
            
            # Analyze emotional trajectory
            emotional_trajectory = await self._analyze_emotional_trajectory(conversation_history)
            
            # Assess therapeutic alliance
            alliance_strength = await self._assess_alliance_strength(conversation_history)
            
            # Measure engagement level
            engagement_level = await self._measure_engagement_level(conversation_history)
            
            # Detect resistance patterns
            resistance_indicators = await self._detect_resistance_patterns(conversation_history)
            
            # Identify progress markers
            progress_markers = await self._identify_progress_markers(conversation_history)
            
            # Check for crisis indicators
            crisis_indicators = await self._detect_crisis_indicators(conversation_history)
            
            # Analyze cultural considerations
            cultural_considerations = await self._analyze_cultural_factors(
                conversation_history, clinical_context
            )
            
            # Assess cognitive load
            cognitive_load = await self._assess_cognitive_load(conversation_history)
            
            # Calculate therapeutic momentum
            therapeutic_momentum = await self._calculate_therapeutic_momentum(
                conversation_history, progress_markers, resistance_indicators
            )
            
            # Identify recent breakthroughs
            recent_breakthroughs = await self._identify_recent_breakthroughs(conversation_history)
            
            # Detect stuck patterns
            stuck_patterns = await self._detect_stuck_patterns(conversation_history)
            
            # Extract client preferences
            client_preferences = await self._extract_client_preferences(conversation_history)
            
            # Identify contraindications
            contraindications = await self._identify_contraindications(
                conversation_history, clinical_context
            )
            
            # Determine conversation phase
            conversation_phase = await self._determine_conversation_phase(
                conversation_history, session_info
            )
            
            context = ConversationalContext(
                turn_number=turn_number,
                session_number=session_number,
                time_in_session=time_in_session,
                conversation_phase=conversation_phase,
                emotional_trajectory=emotional_trajectory,
                alliance_strength=alliance_strength,
                engagement_level=engagement_level,
                resistance_indicators=resistance_indicators,
                progress_markers=progress_markers,
                crisis_indicators=crisis_indicators,
                cultural_considerations=cultural_considerations,
                cognitive_load=cognitive_load,
                therapeutic_momentum=therapeutic_momentum,
                recent_breakthroughs=recent_breakthroughs,
                stuck_patterns=stuck_patterns,
                client_preferences=client_preferences,
                contraindications=contraindications
            )
            
            # Store context in history
            self.context_history.append(context)
            if len(self.context_history) > self.config.get('max_context_history', 50):
                self.context_history.pop(0)
            
            return context
            
        except Exception as e:
            logger.error(f"Error analyzing conversational context: {e}")
            raise
    
    def _create_initial_context(self, clinical_context: ClinicalContext,
                              session_info: Dict[str, Any]) -> ConversationalContext:
        """Create initial context for first turn"""
        return ConversationalContext(
            turn_number=0,
            session_number=session_info.get('session_number', 1),
            time_in_session=timedelta(0),
            conversation_phase=ConversationPhase.INITIAL_ASSESSMENT,
            emotional_trajectory=[],
            alliance_strength=0.5,  # Neutral starting point
            engagement_level=0.5,   # Neutral starting point
            resistance_indicators=[],
            progress_markers=[],
            crisis_indicators=clinical_context.crisis_indicators,
            cultural_considerations=clinical_context.cultural_factors,
            cognitive_load=0.3,     # Low initial load
            therapeutic_momentum=0.0,
            recent_breakthroughs=[],
            stuck_patterns=[],
            client_preferences=[],
            contraindications=clinical_context.contraindications
        )
    
    async def _analyze_emotional_trajectory(self, conversation_history: List[ConversationTurn]) -> List[Tuple[str, float]]:
        """Analyze emotional trajectory across conversation"""
        trajectory = []
        
        # Analyze recent client turns for emotional content
        client_turns = [turn for turn in conversation_history[-self.context_settings['emotional_window_size']:] 
                       if turn.speaker == ConversationRole.CLIENT]
        
        emotion_indicators = {
            'anxiety': ['worried', 'anxious', 'nervous', 'scared', 'panic'],
            'depression': ['sad', 'hopeless', 'empty', 'worthless', 'depressed'],
            'anger': ['angry', 'furious', 'mad', 'irritated', 'frustrated'],
            'joy': ['happy', 'excited', 'good', 'better', 'positive'],
            'fear': ['afraid', 'terrified', 'scared', 'frightened'],
            'shame': ['ashamed', 'embarrassed', 'guilty', 'humiliated'],
            'overwhelm': ['overwhelmed', 'too much', 'can\'t handle', 'drowning']
        }
        
        for turn in client_turns:
            content_lower = turn.content.lower()
            turn_emotions = []
            
            for emotion, indicators in emotion_indicators.items():
                intensity = sum(1 for indicator in indicators if indicator in content_lower)
                if intensity > 0:
                    # Normalize intensity (0.0 to 1.0)
                    normalized_intensity = min(1.0, intensity / 3.0)
                    turn_emotions.append((emotion, normalized_intensity))
            
            # If no specific emotions detected, infer from general tone
            if not turn_emotions:
                if any(word in content_lower for word in ['fine', 'okay', 'alright']):
                    turn_emotions.append(('neutral', 0.5))
                elif any(word in content_lower for word in ['difficult', 'hard', 'struggle']):
                    turn_emotions.append(('distress', 0.6))
            
            trajectory.extend(turn_emotions)
        
        return trajectory
    
    async def _assess_alliance_strength(self, conversation_history: List[ConversationTurn]) -> float:
        """Assess therapeutic alliance strength"""
        if not conversation_history:
            return 0.5  # Neutral starting point
        
        alliance_indicators = {
            'positive': ['trust', 'comfortable', 'safe', 'understand', 'helpful', 'working'],
            'negative': ['don\'t understand', 'not helping', 'waste of time', 'don\'t get it', 'not working']
        }
        
        recent_turns = conversation_history[-6:]  # Last 3 exchanges
        client_turns = [turn for turn in recent_turns if turn.speaker == ConversationRole.CLIENT]
        
        positive_score = 0
        negative_score = 0
        
        for turn in client_turns:
            content_lower = turn.content.lower()
            
            for indicator in alliance_indicators['positive']:
                if indicator in content_lower:
                    positive_score += 1
            
            for indicator in alliance_indicators['negative']:
                if indicator in content_lower:
                    negative_score += 1
        
        # Calculate alliance strength (0.0 to 1.0)
        if positive_score + negative_score == 0:
            return 0.5  # Neutral if no indicators
        
        alliance_strength = positive_score / (positive_score + negative_score)
        
        # Apply temporal decay from previous assessments
        if hasattr(self, 'previous_alliance'):
            decay_rate = self.context_settings['alliance_decay_rate']
            alliance_strength = (alliance_strength * 0.7) + (self.previous_alliance * 0.3 * (1 - decay_rate))
        
        self.previous_alliance = alliance_strength
        return alliance_strength
    
    async def _measure_engagement_level(self, conversation_history: List[ConversationTurn]) -> float:
        """Measure client engagement level"""
        if not conversation_history:
            return 0.5
        
        recent_turns = conversation_history[-self.context_settings['engagement_window']:]
        client_turns = [turn for turn in recent_turns if turn.speaker == ConversationRole.CLIENT]
        
        if not client_turns:
            return 0.5
        
        engagement_score = 0.0
        
        for turn in client_turns:
            content = turn.content.strip()
            
            # Length indicates engagement
            if len(content.split()) > 20:
                engagement_score += 0.3
            elif len(content.split()) > 10:
                engagement_score += 0.2
            elif len(content.split()) > 3:
                engagement_score += 0.1
            
            # Question asking indicates engagement
            if '?' in content:
                engagement_score += 0.2
            
            # Elaboration indicates engagement
            elaboration_words = ['because', 'actually', 'also', 'and then', 'for example']
            if any(word in content.lower() for word in elaboration_words):
                engagement_score += 0.2
            
            # Emotional expression indicates engagement
            if any(word in content.lower() for word in ['feel', 'think', 'believe', 'experience']):
                engagement_score += 0.1
        
        # Normalize to 0-1 range
        max_possible_score = len(client_turns) * 0.8
        return min(1.0, engagement_score / max_possible_score if max_possible_score > 0 else 0.5)
    
    async def _detect_resistance_patterns(self, conversation_history: List[ConversationTurn]) -> List[str]:
        """Detect patterns of resistance in conversation"""
        resistance_indicators = []
        
        client_turns = [turn for turn in conversation_history[-8:] 
                       if turn.speaker == ConversationRole.CLIENT]
        
        resistance_patterns = {
            'direct_disagreement': ['no', 'don\'t agree', 'that\'s not right', 'wrong'],
            'deflection': ['but', 'however', 'what about', 'yes but'],
            'minimization': ['not that bad', 'it\'s fine', 'doesn\'t matter', 'no big deal'],
            'intellectualization': ['theoretically', 'in general', 'people say', 'I read that'],
            'compliance_without_engagement': ['okay', 'sure', 'I guess', 'whatever'],
            'previous_failure': ['tried that', 'doesn\'t work', 'never works', 'won\'t help'],
            'external_blame': ['it\'s because of', 'they make me', 'not my fault', 'can\'t help it']
        }
        
        for turn in client_turns:
            content_lower = turn.content.lower()
            
            for pattern_name, indicators in resistance_patterns.items():
                if any(indicator in content_lower for indicator in indicators):
                    if pattern_name not in resistance_indicators:
                        resistance_indicators.append(pattern_name)
        
        return resistance_indicators
    
    async def _identify_progress_markers(self, conversation_history: List[ConversationTurn]) -> List[str]:
        """Identify markers of therapeutic progress"""
        progress_markers = []
        
        client_turns = [turn for turn in conversation_history[-10:] 
                       if turn.speaker == ConversationRole.CLIENT]
        
        progress_indicators = {
            'insight_development': ['I see', 'I realize', 'I understand now', 'makes sense', 'I get it'],
            'behavioral_change': ['I tried', 'I did', 'I practiced', 'I used', 'I applied'],
            'emotional_regulation': ['I felt better', 'I managed', 'I coped', 'I handled it'],
            'self_awareness': ['I notice', 'I recognize', 'I\'m aware', 'I see myself'],
            'motivation_increase': ['I want to', 'I\'m ready', 'I\'m willing', 'let\'s try'],
            'symptom_improvement': ['better', 'improving', 'less', 'not as bad', 'easier'],
            'relationship_improvement': ['we talked', 'I communicated', 'I set boundaries', 'I expressed']
        }
        
        for turn in client_turns:
            content_lower = turn.content.lower()
            
            for marker_type, indicators in progress_indicators.items():
                if any(indicator in content_lower for indicator in indicators):
                    if marker_type not in progress_markers:
                        progress_markers.append(marker_type)
        
        return progress_markers
    
    async def _detect_crisis_indicators(self, conversation_history: List[ConversationTurn]) -> List[str]:
        """Detect crisis indicators in conversation"""
        crisis_indicators = []
        
        client_turns = [turn for turn in conversation_history[-5:] 
                       if turn.speaker == ConversationRole.CLIENT]
        
        crisis_patterns = {
            'suicidal_ideation': ['suicide', 'kill myself', 'end my life', 'don\'t want to live'],
            'self_harm': ['hurt myself', 'cut myself', 'harm myself', 'self-harm'],
            'hopelessness': ['no point', 'hopeless', 'nothing matters', 'give up'],
            'overwhelming_distress': ['can\'t take it', 'too much', 'breaking down', 'falling apart'],
            'isolation': ['no one cares', 'all alone', 'nobody understands', 'isolated'],
            'substance_abuse': ['drinking too much', 'using drugs', 'getting high', 'numbing'],
            'psychotic_symptoms': ['hearing voices', 'seeing things', 'not real', 'paranoid']
        }
        
        for turn in client_turns:
            content_lower = turn.content.lower()
            
            for crisis_type, indicators in crisis_patterns.items():
                if any(indicator in content_lower for indicator in indicators):
                    if crisis_type not in crisis_indicators:
                        crisis_indicators.append(crisis_type)
        
        return crisis_indicators
    
    async def _analyze_cultural_factors(self, conversation_history: List[ConversationTurn],
                                      clinical_context: ClinicalContext) -> List[str]:
        """Analyze cultural factors present in conversation"""
        cultural_considerations = list(clinical_context.cultural_factors)
        
        client_turns = [turn for turn in conversation_history[-8:] 
                       if turn.speaker == ConversationRole.CLIENT]
        
        cultural_indicators = {
            'family_dynamics': ['my family', 'family says', 'parents think', 'family expects'],
            'cultural_values': ['in my culture', 'my people', 'tradition', 'cultural'],
            'language_barriers': ['hard to explain', 'don\'t have words', 'in my language'],
            'religious_spiritual': ['pray', 'God', 'faith', 'spiritual', 'religious'],
            'community_factors': ['community', 'neighborhood', 'people like me'],
            'discrimination': ['because I\'m', 'they treat me', 'prejudice', 'discrimination'],
            'acculturation_stress': ['fitting in', 'two worlds', 'torn between', 'identity']
        }
        
        for turn in client_turns:
            content_lower = turn.content.lower()
            
            for factor_type, indicators in cultural_indicators.items():
                if any(indicator in content_lower for indicator in indicators):
                    if factor_type not in cultural_considerations:
                        cultural_considerations.append(factor_type)
        
        return cultural_considerations
    
    async def _assess_cognitive_load(self, conversation_history: List[ConversationTurn]) -> float:
        """Assess client's cognitive load and capacity"""
        if not conversation_history:
            return 0.3  # Low initial load
        
        client_turns = [turn for turn in conversation_history[-5:] 
                       if turn.speaker == ConversationRole.CLIENT]
        
        if not client_turns:
            return 0.3
        
        overload_indicators = ['confused', 'don\'t understand', 'too much', 'overwhelming', 'can\'t follow']
        clarity_indicators = ['I understand', 'makes sense', 'I see', 'clear', 'I get it']
        
        overload_count = 0
        clarity_count = 0
        
        for turn in client_turns:
            content_lower = turn.content.lower()
            
            overload_count += sum(1 for indicator in overload_indicators if indicator in content_lower)
            clarity_count += sum(1 for indicator in clarity_indicators if indicator in content_lower)
        
        # Calculate cognitive load (0.0 = low load, 1.0 = high load)
        if overload_count + clarity_count == 0:
            return 0.5  # Neutral if no indicators
        
        load_ratio = overload_count / (overload_count + clarity_count)
        
        # Also consider response complexity and length
        avg_response_length = np.mean([len(turn.content.split()) for turn in client_turns])
        if avg_response_length < 5:
            load_ratio += 0.2  # Short responses may indicate overload
        
        return min(1.0, load_ratio)
    
    async def _calculate_therapeutic_momentum(self, conversation_history: List[ConversationTurn],
                                           progress_markers: List[str],
                                           resistance_indicators: List[str]) -> float:
        """Calculate therapeutic momentum"""
        if not conversation_history:
            return 0.0
        
        # Base momentum from progress markers
        progress_score = len(progress_markers) * 0.2
        
        # Reduce momentum for resistance
        resistance_penalty = len(resistance_indicators) * 0.1
        
        # Consider conversation flow
        recent_turns = conversation_history[-6:]
        client_turns = [turn for turn in recent_turns if turn.speaker == ConversationRole.CLIENT]
        
        # Momentum indicators in recent turns
        momentum_words = ['better', 'progress', 'working', 'helping', 'improving', 'ready', 'motivated']
        stagnation_words = ['stuck', 'same', 'not working', 'no change', 'frustrated']
        
        momentum_boost = 0
        momentum_drag = 0
        
        for turn in client_turns:
            content_lower = turn.content.lower()
            momentum_boost += sum(1 for word in momentum_words if word in content_lower)
            momentum_drag += sum(1 for word in stagnation_words if word in content_lower)
        
        # Calculate final momentum
        momentum = progress_score - resistance_penalty + (momentum_boost * 0.1) - (momentum_drag * 0.1)
        
        return max(0.0, min(1.0, momentum))
    
    async def _identify_recent_breakthroughs(self, conversation_history: List[ConversationTurn]) -> List[str]:
        """Identify recent therapeutic breakthroughs"""
        breakthroughs = []
        
        client_turns = [turn for turn in conversation_history[-8:] 
                       if turn.speaker == ConversationRole.CLIENT]
        
        breakthrough_patterns = {
            'insight_breakthrough': ['I never realized', 'I see now', 'it all makes sense', 'I understand'],
            'emotional_breakthrough': ['I can feel', 'I\'m allowing myself', 'I\'m not afraid'],
            'behavioral_breakthrough': ['I actually did it', 'I tried something new', 'I stood up for myself'],
            'relational_breakthrough': ['we finally talked', 'I opened up', 'I set a boundary'],
            'self_acceptance': ['I\'m okay with', 'I accept', 'I\'m learning to love myself']
        }
        
        for turn in client_turns:
            content_lower = turn.content.lower()
            
            for breakthrough_type, indicators in breakthrough_patterns.items():
                if any(indicator in content_lower for indicator in indicators):
                    if breakthrough_type not in breakthroughs:
                        breakthroughs.append(breakthrough_type)
        
        return breakthroughs
    
    async def _detect_stuck_patterns(self, conversation_history: List[ConversationTurn]) -> List[str]:
        """Detect patterns where client seems stuck"""
        stuck_patterns = []
        
        # Look for repetitive themes or complaints
        client_turns = [turn for turn in conversation_history[-10:] 
                       if turn.speaker == ConversationRole.CLIENT]
        
        if len(client_turns) < 3:
            return stuck_patterns
        
        # Check for repetitive content
        content_themes = {}
        for turn in client_turns:
            words = turn.content.lower().split()
            key_words = [word for word in words if len(word) > 4 and word not in ['that', 'this', 'with', 'have', 'been']]
            
            for word in key_words:
                content_themes[word] = content_themes.get(word, 0) + 1
        
        # Identify highly repeated themes
        repeated_themes = [theme for theme, count in content_themes.items() if count >= 3]
        
        if repeated_themes:
            stuck_patterns.append('repetitive_content')
        
        # Check for explicit stuck language
        stuck_language = ['stuck', 'same thing', 'nothing changes', 'going in circles', 'not getting anywhere']
        
        for turn in client_turns:
            content_lower = turn.content.lower()
            if any(phrase in content_lower for phrase in stuck_language):
                stuck_patterns.append('explicit_stuckness')
                break
        
        return stuck_patterns
    
    async def _extract_client_preferences(self, conversation_history: List[ConversationTurn]) -> List[str]:
        """Extract client preferences from conversation"""
        preferences = []
        
        client_turns = [turn for turn in conversation_history 
                       if turn.speaker == ConversationRole.CLIENT]
        
        preference_indicators = {
            'direct_approach': ['tell me directly', 'be straight with me', 'just say it'],
            'gentle_approach': ['be gentle', 'go slow', 'take it easy'],
            'practical_focus': ['practical', 'concrete', 'specific', 'actionable'],
            'emotional_focus': ['feelings', 'emotions', 'how I feel', 'emotional'],
            'insight_focus': ['understand why', 'make sense of', 'figure out', 'analyze'],
            'solution_focus': ['what can I do', 'how to fix', 'solutions', 'strategies']
        }
        
        for turn in client_turns:
            content_lower = turn.content.lower()
            
            for preference_type, indicators in preference_indicators.items():
                if any(indicator in content_lower for indicator in indicators):
                    if preference_type not in preferences:
                        preferences.append(preference_type)
        
        return preferences
    
    async def _identify_contraindications(self, conversation_history: List[ConversationTurn],
                                        clinical_context: ClinicalContext) -> List[str]:
        """Identify contraindications based on conversation and context"""
        contraindications = list(clinical_context.contraindications)
        
        # Add contraindications based on conversation patterns
        client_turns = [turn for turn in conversation_history[-5:] 
                       if turn.speaker == ConversationRole.CLIENT]
        
        contraindication_patterns = {
            'avoid_challenging': ['too hard', 'can\'t handle', 'overwhelming', 'too much pressure'],
            'avoid_deep_work': ['not ready', 'too painful', 'don\'t want to go there'],
            'avoid_homework': ['can\'t do homework', 'too busy', 'don\'t have time'],
            'avoid_emotion_focus': ['don\'t want to feel', 'too emotional', 'makes me cry']
        }
        
        for turn in client_turns:
            content_lower = turn.content.lower()
            
            for contraindication, indicators in contraindication_patterns.items():
                if any(indicator in content_lower for indicator in indicators):
                    if contraindication not in contraindications:
                        contraindications.append(contraindication)
        
        return contraindications
    
    async def _determine_conversation_phase(self, conversation_history: List[ConversationTurn],
                                          session_info: Dict[str, Any]) -> ConversationPhase:
        """Determine current conversation phase"""
        turn_count = len(conversation_history)
        session_number = session_info.get('session_number', 1)
        
        # Early session phases
        if session_number == 1:
            if turn_count < 4:
                return ConversationPhase.INITIAL_ASSESSMENT
            elif turn_count < 8:
                return ConversationPhase.RAPPORT_BUILDING
            else:
                return ConversationPhase.PROBLEM_EXPLORATION
        
        # Later session phases
        if turn_count < 3:
            return ConversationPhase.RAPPORT_BUILDING
        elif turn_count < 8:
            return ConversationPhase.PROBLEM_EXPLORATION
        elif turn_count < 15:
            return ConversationPhase.INTERVENTION_PLANNING
        elif turn_count < 25:
            return ConversationPhase.SKILL_BUILDING
        else:
            return ConversationPhase.PROGRESS_MONITORING
    
    async def generate_contextual_response(self, client_statement: str,
                                         conversation_history: List[ConversationTurn],
                                         clinical_context: ClinicalContext,
                                         session_info: Dict[str, Any]) -> ContextualResponse:
        """Generate a response with full contextual awareness"""
        try:
            # Analyze conversational context
            conv_context = await self.analyze_conversational_context(
                conversation_history, clinical_context, session_info
            )
            
            # Determine response context
            response_context = await self._determine_response_context(
                client_statement, conv_context, clinical_context
            )
            
            # Generate base response using modality integration
            if not self.modality_integrator.current_integration_plan:
                await self.modality_integrator.create_integration_plan(
                    clinical_context, clinical_context.therapeutic_goals or []
                )
            
            integrated_response = await self.modality_integrator.generate_integrated_response(
                client_statement, conversation_history, clinical_context
            )
            
            # Apply contextual adaptations
            adapted_response = await self._apply_contextual_adaptations(
                integrated_response.primary_response, response_context, conv_context
            )
            
            # Generate contextual enhancements
            contextual_adaptations = await self._generate_contextual_adaptations(
                response_context, conv_context
            )
            
            # Identify risk mitigations
            risk_mitigations = await self._identify_risk_mitigations(
                response_context, conv_context
            )
            
            # Capitalize on opportunities
            opportunity_capitalizations = await self._capitalize_opportunities(
                response_context, conv_context
            )
            
            # Generate follow-up recommendations
            follow_up_recommendations = await self._generate_follow_up_recommendations(
                response_context, conv_context
            )
            
            # Create monitoring alerts
            context_monitoring_alerts = await self._create_monitoring_alerts(
                response_context, conv_context
            )
            
            contextual_response = ContextualResponse(
                response=adapted_response,
                context_analysis=conv_context,
                response_context=response_context,
                contextual_adaptations=contextual_adaptations,
                risk_mitigations=risk_mitigations,
                opportunity_capitalizations=opportunity_capitalizations,
                follow_up_recommendations=follow_up_recommendations,
                context_monitoring_alerts=context_monitoring_alerts
            )
            
            # Store response in history
            self.response_history.append(contextual_response)
            if len(self.response_history) > self.config.get('max_context_history', 50):
                self.response_history.pop(0)
            
            return contextual_response
            
        except Exception as e:
            logger.error(f"Error generating contextual response: {e}")
            raise
    
    async def _determine_response_context(self, client_statement: str,
                                        conv_context: ConversationalContext,
                                        clinical_context: ClinicalContext) -> ResponseContext:
        """Determine the context for response generation"""
        # Determine context type
        context_type = self._classify_response_context_type(client_statement, conv_context)
        
        # Identify primary contextual factors
        primary_factors = self._identify_primary_factors(conv_context, clinical_context)
        
        # Calculate contextual weights
        contextual_weights = self._calculate_contextual_weights(primary_factors, conv_context)
        
        # Identify immediate needs
        immediate_needs = self._identify_immediate_needs(conv_context, client_statement)
        
        # Identify therapeutic opportunities
        therapeutic_opportunities = self._identify_therapeutic_opportunities(conv_context)
        
        # Assess potential risks
        potential_risks = self._assess_potential_risks(conv_context, client_statement)
        
        # Recommend techniques
        recommended_techniques = self._recommend_techniques(context_type, conv_context)
        
        # Identify contraindicated approaches
        contraindicated_approaches = self._identify_contraindicated_approaches(conv_context)
        
        # Consider timing
        timing_considerations = self._assess_timing_considerations(conv_context)
        
        return ResponseContext(
            context_type=context_type,
            primary_factors=primary_factors,
            contextual_weights=contextual_weights,
            immediate_needs=immediate_needs,
            therapeutic_opportunities=therapeutic_opportunities,
            potential_risks=potential_risks,
            recommended_techniques=recommended_techniques,
            contraindicated_approaches=contraindicated_approaches,
            timing_considerations=timing_considerations
        )
    
    def _classify_response_context_type(self, client_statement: str,
                                      conv_context: ConversationalContext) -> ResponseContextType:
        """Classify the type of response context needed"""
        statement_lower = client_statement.lower()
        
        # Crisis response takes priority
        if conv_context.crisis_indicators or any(word in statement_lower for word in ['suicide', 'kill', 'die', 'hurt']):
            return ResponseContextType.CRISIS_RESPONSE
        
        # Opening context for early turns
        if conv_context.turn_number < 2:
            return ResponseContextType.OPENING
        
        # Resistance handling
        if conv_context.resistance_indicators or any(word in statement_lower for word in ['but', 'won\'t work', 'don\'t want']):
            return ResponseContextType.RESISTANCE_HANDLING
        
        # Processing context for breakthroughs
        if conv_context.recent_breakthroughs or any(word in statement_lower for word in ['I see', 'realize', 'understand']):
            return ResponseContextType.PROCESSING
        
        # Exploration context
        if conv_context.conversation_phase in [ConversationPhase.PROBLEM_EXPLORATION, ConversationPhase.INITIAL_ASSESSMENT]:
            return ResponseContextType.EXPLORATION
        
        # Intervention context
        if conv_context.conversation_phase in [ConversationPhase.INTERVENTION_PLANNING, ConversationPhase.SKILL_BUILDING]:
            return ResponseContextType.INTERVENTION
        
        # Default to exploration
        return ResponseContextType.EXPLORATION
    
    def _identify_primary_factors(self, conv_context: ConversationalContext,
                                clinical_context: ClinicalContext) -> List[ContextualFactor]:
        """Identify primary contextual factors"""
        factors = []
        
        # Crisis level is always primary if present
        if conv_context.crisis_indicators:
            factors.append(ContextualFactor.CRISIS_LEVEL)
        
        # Emotional state if high intensity
        if any(intensity > 0.7 for _, intensity in conv_context.emotional_trajectory):
            factors.append(ContextualFactor.EMOTIONAL_STATE)
        
        # Alliance strength if low
        if conv_context.alliance_strength < 0.4:
            factors.append(ContextualFactor.THERAPEUTIC_ALLIANCE)
        
        # Resistance if present
        if conv_context.resistance_indicators:
            factors.append(ContextualFactor.RESISTANCE_PATTERN)
        
        # Treatment progress if significant
        if conv_context.progress_markers or conv_context.therapeutic_momentum > 0.7:
            factors.append(ContextualFactor.TREATMENT_PROGRESS)
        
        # Cultural context if relevant
        if conv_context.cultural_considerations:
            factors.append(ContextualFactor.CULTURAL_CONTEXT)
        
        # Cognitive capacity if impaired
        if conv_context.cognitive_load > 0.7:
            factors.append(ContextualFactor.COGNITIVE_CAPACITY)
        
        # Engagement level if very low or high
        if conv_context.engagement_level < 0.3 or conv_context.engagement_level > 0.8:
            factors.append(ContextualFactor.ENGAGEMENT_LEVEL)
        
        return factors
    
    def _calculate_contextual_weights(self, primary_factors: List[ContextualFactor],
                                    conv_context: ConversationalContext) -> Dict[ContextualFactor, float]:
        """Calculate weights for contextual factors"""
        weights = {}
        
        # Base weights for all factors
        base_weights = {
            ContextualFactor.CRISIS_LEVEL: 1.0,
            ContextualFactor.EMOTIONAL_STATE: 0.8,
            ContextualFactor.THERAPEUTIC_ALLIANCE: 0.7,
            ContextualFactor.RESISTANCE_PATTERN: 0.6,
            ContextualFactor.TREATMENT_PROGRESS: 0.5,
            ContextualFactor.CULTURAL_CONTEXT: 0.4,
            ContextualFactor.COGNITIVE_CAPACITY: 0.6,
            ContextualFactor.ENGAGEMENT_LEVEL: 0.5
        }
        
        # Adjust weights based on context
        for factor in primary_factors:
            weight = base_weights.get(factor, 0.5)
            
            # Boost weight based on severity/relevance
            if factor == ContextualFactor.CRISIS_LEVEL and conv_context.crisis_indicators:
                weight = 1.0  # Maximum priority
            elif factor == ContextualFactor.EMOTIONAL_STATE:
                max_intensity = max([intensity for _, intensity in conv_context.emotional_trajectory], default=0)
                weight = min(1.0, weight + max_intensity * 0.3)
            elif factor == ContextualFactor.THERAPEUTIC_ALLIANCE and conv_context.alliance_strength < 0.3:
                weight = min(1.0, weight + 0.3)  # Boost for very low alliance
            
            weights[factor] = weight
        
        return weights
    
    def _identify_immediate_needs(self, conv_context: ConversationalContext,
                                client_statement: str) -> List[str]:
        """Identify immediate needs based on context"""
        needs = []
        
        # Safety needs
        if conv_context.crisis_indicators:
            needs.extend(['safety_assessment', 'crisis_intervention', 'immediate_support'])
        
        # Alliance repair needs
        if conv_context.alliance_strength < 0.4:
            needs.extend(['alliance_repair', 'validation', 'collaboration'])
        
        # Emotional regulation needs
        if any(intensity > 0.8 for _, intensity in conv_context.emotional_trajectory):
            needs.extend(['emotional_regulation', 'grounding', 'stabilization'])
        
        # Cognitive support needs
        if conv_context.cognitive_load > 0.7:
            needs.extend(['simplification', 'clarification', 'pacing_adjustment'])
        
        # Engagement needs
        if conv_context.engagement_level < 0.3:
            needs.extend(['motivation_enhancement', 'relevance_connection', 'interest_building'])
        
        return needs
    
    def _identify_therapeutic_opportunities(self, conv_context: ConversationalContext) -> List[str]:
        """Identify therapeutic opportunities in current context"""
        opportunities = []
        
        # Breakthrough opportunities
        if conv_context.recent_breakthroughs:
            opportunities.extend(['deepen_insight', 'reinforce_learning', 'apply_understanding'])
        
        # High engagement opportunities
        if conv_context.engagement_level > 0.7:
            opportunities.extend(['introduce_techniques', 'assign_homework', 'set_goals'])
        
        # Progress momentum opportunities
        if conv_context.therapeutic_momentum > 0.6:
            opportunities.extend(['accelerate_progress', 'expand_skills', 'generalize_learning'])
        
        # Alliance strength opportunities
        if conv_context.alliance_strength > 0.7:
            opportunities.extend(['deeper_exploration', 'challenging_work', 'difficult_topics'])
        
        return opportunities
    
    def _assess_potential_risks(self, conv_context: ConversationalContext,
                              client_statement: str) -> List[str]:
        """Assess potential risks in current context"""
        risks = []
        
        # Crisis risks
        if conv_context.crisis_indicators:
            risks.extend(['suicide_risk', 'self_harm_risk', 'safety_concerns'])
        
        # Alliance risks
        if conv_context.alliance_strength < 0.5:
            risks.extend(['alliance_rupture', 'premature_termination', 'resistance_escalation'])
        
        # Emotional overwhelm risks
        if any(intensity > 0.8 for _, intensity in conv_context.emotional_trajectory):
            risks.extend(['emotional_overwhelm', 'dissociation_risk', 'session_disruption'])
        
        # Cognitive overload risks
        if conv_context.cognitive_load > 0.6:
            risks.extend(['confusion', 'misunderstanding', 'intervention_failure'])
        
        # Stuck pattern risks
        if conv_context.stuck_patterns:
            risks.extend(['therapeutic_stagnation', 'frustration_increase', 'hopelessness'])
        
        return risks
    
    async def _apply_contextual_adaptations(self, base_response: TherapistResponse,
                                          response_context: ResponseContext,
                                          conv_context: ConversationalContext) -> TherapistResponse:
        """Apply contextual adaptations to base response"""
        adapted_content = base_response.content
        adapted_rationale = base_response.clinical_rationale
        
        # Apply adaptations based on primary factors
        for factor in response_context.primary_factors:
            if factor == ContextualFactor.CRISIS_LEVEL:
                adapted_content = self._adapt_for_crisis(adapted_content, conv_context)
            elif factor == ContextualFactor.EMOTIONAL_STATE:
                adapted_content = self._adapt_for_emotional_state(adapted_content, conv_context)
            elif factor == ContextualFactor.THERAPEUTIC_ALLIANCE:
                adapted_content = self._adapt_for_alliance(adapted_content, conv_context)
            elif factor == ContextualFactor.RESISTANCE_PATTERN:
                adapted_content = self._adapt_for_resistance(adapted_content, conv_context)
            elif factor == ContextualFactor.COGNITIVE_CAPACITY:
                adapted_content = self._adapt_for_cognitive_capacity(adapted_content, conv_context)
            elif factor == ContextualFactor.CULTURAL_CONTEXT:
                adapted_content = self._adapt_for_cultural_context(adapted_content, conv_context)
        
        # Update rationale
        adaptations_made = [factor.value for factor in response_context.primary_factors]
        adapted_rationale += f" Contextually adapted for: {', '.join(adaptations_made)}"
        
        return TherapistResponse(
            content=adapted_content,
            clinical_rationale=adapted_rationale,
            therapeutic_technique=base_response.therapeutic_technique,
            intervention_type=base_response.intervention_type,
            confidence_score=base_response.confidence_score,
            contraindications=base_response.contraindications,
            follow_up_suggestions=base_response.follow_up_suggestions
        )
    
    def _adapt_for_crisis(self, content: str, conv_context: ConversationalContext) -> str:
        """Adapt response for crisis context"""
        crisis_prefix = "I'm really concerned about what you've shared. "
        safety_focus = " Your safety is my priority right now."
        
        if not any(word in content.lower() for word in ['safe', 'concern', 'priority']):
            content = crisis_prefix + content + safety_focus
        
        return content
    
    def _adapt_for_emotional_state(self, content: str, conv_context: ConversationalContext) -> str:
        """Adapt response for high emotional intensity"""
        # Add validation and pacing
        if not any(word in content.lower() for word in ['understand', 'hear', 'see']):
            validation = "I can see how intense this is for you. "
            content = validation + content
        
        # Add grounding if very high intensity
        max_intensity = max([intensity for _, intensity in conv_context.emotional_trajectory], default=0)
        if max_intensity > 0.8:
            grounding = " Let's take a moment to ground ourselves here."
            content += grounding
        
        return content
    
    def _adapt_for_alliance(self, content: str, conv_context: ConversationalContext) -> str:
        """Adapt response for low therapeutic alliance"""
        if conv_context.alliance_strength < 0.4:
            collaboration = "I want to make sure we're working together on this. "
            validation = " How does this feel for you?"
            
            content = collaboration + content + validation
        
        return content
    
    def _adapt_for_resistance(self, content: str, conv_context: ConversationalContext) -> str:
        """Adapt response for resistance patterns"""
        if conv_context.resistance_indicators:
            validation = "I can understand why you might feel that way. "
            exploration = " Help me understand what doesn't feel right about this for you."
            
            content = validation + content + exploration
        
        return content
    
    def _adapt_for_cognitive_capacity(self, content: str, conv_context: ConversationalContext) -> str:
        """Adapt response for cognitive capacity"""
        if conv_context.cognitive_load > 0.7:
            # Simplify language
            content = content.replace('intervention', 'approach')
            content = content.replace('therapeutic', 'helpful')
            content = content.replace('cognitive', 'thinking')
            
            # Add understanding check
            content += " Does this make sense to you?"
        
        return content
    
    def _adapt_for_cultural_context(self, content: str, conv_context: ConversationalContext) -> str:
        """Adapt response for cultural context"""
        if 'family_dynamics' in conv_context.cultural_considerations:
            content += " I'm also curious about how your family might view this situation."
        
        if 'cultural_values' in conv_context.cultural_considerations:
            content += " How does this fit with your cultural values and beliefs?"
        
        return content
    
    async def _generate_contextual_adaptations(self, response_context: ResponseContext,
                                             conv_context: ConversationalContext) -> List[str]:
        """Generate list of contextual adaptations made"""
        adaptations = []
        
        for factor in response_context.primary_factors:
            if factor == ContextualFactor.CRISIS_LEVEL:
                adaptations.append("Crisis-focused safety prioritization")
            elif factor == ContextualFactor.EMOTIONAL_STATE:
                adaptations.append("Emotional intensity validation and pacing")
            elif factor == ContextualFactor.THERAPEUTIC_ALLIANCE:
                adaptations.append("Alliance repair and collaboration emphasis")
            elif factor == ContextualFactor.RESISTANCE_PATTERN:
                adaptations.append("Resistance validation and exploration")
            elif factor == ContextualFactor.COGNITIVE_CAPACITY:
                adaptations.append("Language simplification and understanding checks")
            elif factor == ContextualFactor.CULTURAL_CONTEXT:
                adaptations.append("Cultural sensitivity and value acknowledgment")
        
        return adaptations
    
    async def _identify_risk_mitigations(self, response_context: ResponseContext,
                                       conv_context: ConversationalContext) -> List[str]:
        """Identify risk mitigation strategies"""
        mitigations = []
        
        for risk in response_context.potential_risks:
            if 'suicide_risk' in risk:
                mitigations.append("Safety assessment and crisis intervention protocols activated")
            elif 'alliance_rupture' in risk:
                mitigations.append("Collaborative approach and validation emphasized")
            elif 'emotional_overwhelm' in risk:
                mitigations.append("Pacing slowed and grounding techniques available")
            elif 'confusion' in risk:
                mitigations.append("Language simplified and understanding checked")
        
        return mitigations
    
    async def _capitalize_opportunities(self, response_context: ResponseContext,
                                      conv_context: ConversationalContext) -> List[str]:
        """Capitalize on therapeutic opportunities"""
        capitalizations = []
        
        for opportunity in response_context.therapeutic_opportunities:
            if 'deepen_insight' in opportunity:
                capitalizations.append("Breakthrough insight reinforced and expanded")
            elif 'introduce_techniques' in opportunity:
                capitalizations.append("High engagement leveraged for skill introduction")
            elif 'accelerate_progress' in opportunity:
                capitalizations.append("Therapeutic momentum utilized for progress acceleration")
        
        return capitalizations
    
    async def _generate_follow_up_recommendations(self, response_context: ResponseContext,
                                                conv_context: ConversationalContext) -> List[str]:
        """Generate follow-up recommendations"""
        recommendations = []
        
        # Based on context type
        if response_context.context_type == ResponseContextType.CRISIS_RESPONSE:
            recommendations.extend([
                "Continue safety monitoring",
                "Schedule follow-up within 24-48 hours",
                "Provide crisis resources and contacts"
            ])
        elif response_context.context_type == ResponseContextType.PROCESSING:
            recommendations.extend([
                "Explore applications of new insight",
                "Connect insight to behavioral changes",
                "Monitor integration of understanding"
            ])
        
        # Based on primary factors
        if ContextualFactor.THERAPEUTIC_ALLIANCE in response_context.primary_factors:
            recommendations.append("Monitor alliance strength in next session")
        
        if ContextualFactor.TREATMENT_PROGRESS in response_context.primary_factors:
            recommendations.append("Build on current therapeutic momentum")
        
        return recommendations
    
    async def _create_monitoring_alerts(self, response_context: ResponseContext,
                                      conv_context: ConversationalContext) -> List[str]:
        """Create context monitoring alerts"""
        alerts = []
        
        # Crisis monitoring
        if conv_context.crisis_indicators:
            alerts.append("CRITICAL: Continue crisis monitoring - safety assessment required")
        
        # Alliance monitoring
        if conv_context.alliance_strength < 0.3:
            alerts.append("WARNING: Very low alliance strength - prioritize relationship repair")
        
        # Stuck pattern monitoring
        if conv_context.stuck_patterns:
            alerts.append("NOTICE: Stuck patterns detected - consider approach modification")
        
        # Progress monitoring
        if conv_context.therapeutic_momentum < 0.2:
            alerts.append("NOTICE: Low therapeutic momentum - assess barriers to progress")
        
        return alerts
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get statistics about context analysis and response generation"""
        return {
            'context_history_length': len(self.context_history),
            'response_history_length': len(self.response_history),
            'supported_contextual_factors': [factor.value for factor in ContextualFactor],
            'response_context_types': [context_type.value for context_type in ResponseContextType],
            'contextual_patterns': list(self.contextual_patterns.keys()),
            'adaptation_rules': list(self.adaptation_rules.keys()),
            'configuration': self.config,
            'context_settings': self.context_settings
        }
