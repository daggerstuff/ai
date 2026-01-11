"""
Unit tests for Context-Aware Response Generator

Tests the context-aware therapeutic response generation system including
conversational context analysis, contextual adaptations, risk mitigation,
and opportunity capitalization.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional
from unittest.mock import Mock, patch

import pytest
import pytest_asyncio


# Mock the missing imports
@dataclass
class MockClinicalContext:
    client_presentation: str
    primary_diagnosis: str
    secondary_diagnoses: List[str] = None
    therapeutic_goals: List[str] = None
    cultural_factors: List[str] = None
    crisis_indicators: List[str] = None
    contraindications: List[str] = None
    session_number: int = 1
    conversation_history: List = None

class MockTherapeuticModality(Enum):
    CBT = "cbt"
    DBT = "dbt"

class MockConversationRole(Enum):
    THERAPIST = "therapist"
    CLIENT = "client"

class MockConversationPhase(Enum):
    INITIAL_ASSESSMENT = "initial_assessment"
    PROBLEM_EXPLORATION = "problem_exploration"
    INTERVENTION_PLANNING = "intervention_planning"

class MockInterventionType(Enum):
    ASSESSMENT = "assessment"
    EXPLORATION = "exploration"
    VALIDATION = "validation"

@dataclass
class MockConversationTurn:
    turn_id: str
    speaker: MockConversationRole
    content: str
    confidence_score: float
    clinical_rationale: Optional[str] = None
    intervention_type: Optional[MockInterventionType] = None

@dataclass
class MockTherapistResponse:
    content: str
    clinical_rationale: str
    intervention_type: MockInterventionType
    confidence_score: float

# Patch the imports
with patch.dict('sys.modules', {
    'ai.pixel.data.therapeutic_conversation_schema': Mock(),
    'ai.pixel.data.therapist_response_generator': Mock(),
    'ai.pixel.data.dynamic_conversation_generator': Mock(),
    'ai.pixel.data.therapeutic_modality_integrator': Mock()
}):
    # Mock the classes before importing
    import sys
    sys.modules['ai.pixel.data.therapeutic_conversation_schema'].TherapeuticModality = MockTherapeuticModality
    sys.modules['ai.pixel.data.therapeutic_conversation_schema'].ClinicalContext = MockClinicalContext
    sys.modules['ai.pixel.data.therapist_response_generator'].InterventionType = MockInterventionType
    sys.modules['ai.pixel.data.therapist_response_generator'].TherapistResponse = MockTherapistResponse
    sys.modules['ai.pixel.data.dynamic_conversation_generator'].ConversationTurn = MockConversationTurn
    sys.modules['ai.pixel.data.dynamic_conversation_generator'].ConversationRole = MockConversationRole
    sys.modules['ai.pixel.data.dynamic_conversation_generator'].ConversationPhase = MockConversationPhase
    sys.modules['ai.pixel.data.therapeutic_modality_integrator'].TherapeuticModalityIntegrator = Mock()
    
    from .context_aware_response_generator import (
        ContextAwareResponseGenerator,
        ContextualFactor,
        ConversationalContext,
        ResponseContextType,
    )


@pytest.fixture
def sample_clinical_context():
    """Create sample clinical context"""
    return ClinicalContext(
        client_presentation="Client with depression and anxiety seeking help",
        primary_diagnosis="Major Depressive Disorder",
        secondary_diagnoses=["Generalized Anxiety Disorder"],
        therapeutic_goals=["Reduce symptoms", "Improve functioning"],
        cultural_factors=["Hispanic/Latino background"],
        crisis_indicators=[],
        contraindications=[],
        session_number=3,
        conversation_history=[]
    )


@pytest.fixture
def crisis_clinical_context():
    """Create clinical context with crisis indicators"""
    return ClinicalContext(
        client_presentation="Client expressing suicidal thoughts and hopelessness",
        primary_diagnosis="Major Depressive Disorder",
        crisis_indicators=["suicidal ideation", "hopelessness"],
        therapeutic_goals=["Safety planning", "Crisis stabilization"],
        session_number=1,
        conversation_history=[]
    )


@pytest.fixture
def sample_conversation_history():
    """Create sample conversation history"""
    return [
        ConversationTurn(
            turn_id="turn_001",
            speaker=ConversationRole.THERAPIST,
            content="How are you feeling today?",
            clinical_rationale="Opening assessment",
            intervention_type=InterventionType.ASSESSMENT,
            confidence_score=0.8
        ),
        ConversationTurn(
            turn_id="turn_002",
            speaker=ConversationRole.CLIENT,
            content="I've been feeling really overwhelmed and anxious lately.",
            confidence_score=0.8
        ),
        ConversationTurn(
            turn_id="turn_003",
            speaker=ConversationRole.THERAPIST,
            content="That sounds really difficult. Can you tell me more about what's been overwhelming?",
            clinical_rationale="Validation and exploration",
            intervention_type=InterventionType.EXPLORATION,
            confidence_score=0.9
        ),
        ConversationTurn(
            turn_id="turn_004",
            speaker=ConversationRole.CLIENT,
            content="Everything feels like too much. I can't handle work, relationships, nothing is working.",
            confidence_score=0.8
        )
    ]


@pytest.fixture
def crisis_conversation_history():
    """Create conversation history with crisis content"""
    return [
        ConversationTurn(
            turn_id="turn_001",
            speaker=ConversationRole.THERAPIST,
            content="What brings you in today?",
            confidence_score=0.8
        ),
        ConversationTurn(
            turn_id="turn_002",
            speaker=ConversationRole.CLIENT,
            content="I can't take this anymore. I've been thinking about ending my life.",
            confidence_score=0.8
        )
    ]


@pytest.fixture
def session_info():
    """Create sample session information"""
    return {
        'session_number': 3,
        'session_start': datetime.now() - timedelta(minutes=25),
        'session_type': 'individual_therapy'
    }


@pytest_asyncio.fixture
async def generator():
    """Create context-aware response generator instance"""
    with patch('ai.pixel.data.context_aware_response_generator.TherapeuticModalityIntegrator'):
        generator = ContextAwareResponseGenerator()
        return generator


class TestContextAwareResponseGenerator:
    """Test cases for ContextAwareResponseGenerator"""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test generator initialization"""
        with patch('ai.pixel.data.context_aware_response_generator.TherapeuticModalityIntegrator'):
            generator = ContextAwareResponseGenerator()
            
            assert generator.config is not None
            assert generator.modality_integrator is not None
            assert generator.contextual_patterns is not None
            assert generator.response_templates is not None
            assert generator.adaptation_rules is not None
            assert generator.context_settings is not None
            assert generator.context_history == []
            assert generator.response_history == []
    
    @pytest.mark.asyncio
    async def test_emotional_trajectory_analysis(self, generator, sample_conversation_history):
        """Test emotional trajectory analysis"""
        trajectory = await generator._analyze_emotional_trajectory(sample_conversation_history)
        
        assert isinstance(trajectory, list)
        # Should detect overwhelm and anxiety from client statements
        emotions = [emotion for emotion, _ in trajectory]
        assert 'overwhelm' in emotions or 'anxiety' in emotions
        
        # Intensities should be between 0 and 1
        for emotion, intensity in trajectory:
            assert 0.0 <= intensity <= 1.0
    
    @pytest.mark.asyncio
    async def test_alliance_strength_assessment(self, generator, sample_conversation_history):
        """Test therapeutic alliance strength assessment"""
        alliance_strength = await generator._assess_alliance_strength(sample_conversation_history)
        
        assert isinstance(alliance_strength, float)
        assert 0.0 <= alliance_strength <= 1.0
        
        # Should be neutral to positive for cooperative conversation
        assert alliance_strength >= 0.4
    
    @pytest.mark.asyncio
    async def test_alliance_strength_with_negative_indicators(self, generator):
        """Test alliance assessment with negative indicators"""
        negative_history = [
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="This isn't helping at all. I don't think you understand me.",
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_002",
                speaker=ConversationRole.CLIENT,
                content="This feels like a waste of time.",
                confidence_score=0.8
            )
        ]
        
        alliance_strength = await generator._assess_alliance_strength(negative_history)
        
        # Should be low due to negative indicators
        assert alliance_strength < 0.5
    
    @pytest.mark.asyncio
    async def test_engagement_level_measurement(self, generator, sample_conversation_history):
        """Test engagement level measurement"""
        engagement = await generator._measure_engagement_level(sample_conversation_history)
        
        assert isinstance(engagement, float)
        assert 0.0 <= engagement <= 1.0
        
        # Should be moderate to high for detailed responses
        assert engagement >= 0.4
    
    @pytest.mark.asyncio
    async def test_resistance_pattern_detection(self, generator):
        """Test resistance pattern detection"""
        resistant_history = [
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="I don't agree with that approach. I've tried that before and it doesn't work.",
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_002",
                speaker=ConversationRole.CLIENT,
                content="But that's not the real problem. What about my situation is different.",
                confidence_score=0.8
            )
        ]
        
        resistance_indicators = await generator._detect_resistance_patterns(resistant_history)
        
        assert isinstance(resistance_indicators, list)
        assert len(resistance_indicators) > 0
        assert 'direct_disagreement' in resistance_indicators
        assert 'deflection' in resistance_indicators
        assert 'previous_failure' in resistance_indicators
    
    @pytest.mark.asyncio
    async def test_progress_marker_identification(self, generator):
        """Test progress marker identification"""
        progress_history = [
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="I see what you mean now. I realize I've been doing this pattern for years.",
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_002",
                speaker=ConversationRole.CLIENT,
                content="I tried the technique you suggested and it actually helped.",
                confidence_score=0.8
            )
        ]
        
        progress_markers = await generator._identify_progress_markers(progress_history)
        
        assert isinstance(progress_markers, list)
        assert len(progress_markers) > 0
        assert 'insight_development' in progress_markers
        assert 'behavioral_change' in progress_markers
    
    @pytest.mark.asyncio
    async def test_crisis_indicator_detection(self, generator, crisis_conversation_history):
        """Test crisis indicator detection"""
        crisis_indicators = await generator._detect_crisis_indicators(crisis_conversation_history)
        
        assert isinstance(crisis_indicators, list)
        assert len(crisis_indicators) > 0
        assert 'suicidal_ideation' in crisis_indicators
    
    @pytest.mark.asyncio
    async def test_cultural_factor_analysis(self, generator, sample_clinical_context):
        """Test cultural factor analysis"""
        cultural_history = [
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="My family doesn't understand mental health. In my culture, we don't talk about these things.",
                confidence_score=0.8
            )
        ]
        
        cultural_factors = await generator._analyze_cultural_factors(cultural_history, sample_clinical_context)
        
        assert isinstance(cultural_factors, list)
        assert 'Hispanic/Latino background' in cultural_factors  # From clinical context
        assert 'family_dynamics' in cultural_factors  # From conversation
        assert 'cultural_values' in cultural_factors  # From conversation
    
    @pytest.mark.asyncio
    async def test_cognitive_load_assessment(self, generator):
        """Test cognitive load assessment"""
        # High cognitive load conversation
        overload_history = [
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="I'm confused. This is too much information. I don't understand what you're saying.",
                confidence_score=0.8
            )
        ]
        
        cognitive_load = await generator._assess_cognitive_load(overload_history)
        
        assert isinstance(cognitive_load, float)
        assert 0.0 <= cognitive_load <= 1.0
        assert cognitive_load > 0.5  # Should be high due to confusion indicators
        
        # Low cognitive load conversation
        clear_history = [
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="I understand what you're saying. That makes perfect sense to me.",
                confidence_score=0.8
            )
        ]
        
        cognitive_load = await generator._assess_cognitive_load(clear_history)
        assert cognitive_load < 0.5  # Should be low due to clarity indicators
    
    @pytest.mark.asyncio
    async def test_therapeutic_momentum_calculation(self, generator):
        """Test therapeutic momentum calculation"""
        progress_markers = ['insight_development', 'behavioral_change']
        resistance_indicators = []
        
        momentum_history = [
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="Things are getting better. I feel like I'm making progress.",
                confidence_score=0.8
            )
        ]
        
        momentum = await generator._calculate_therapeutic_momentum(
            momentum_history, progress_markers, resistance_indicators
        )
        
        assert isinstance(momentum, float)
        assert 0.0 <= momentum <= 1.0
        assert momentum > 0.5  # Should be high with progress markers and positive language
    
    @pytest.mark.asyncio
    async def test_breakthrough_identification(self, generator):
        """Test recent breakthrough identification"""
        breakthrough_history = [
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="I never realized this before, but I see now how my childhood affects my relationships.",
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_002",
                speaker=ConversationRole.CLIENT,
                content="I actually stood up for myself at work yesterday. I set a boundary.",
                confidence_score=0.8
            )
        ]
        
        breakthroughs = await generator._identify_recent_breakthroughs(breakthrough_history)
        
        assert isinstance(breakthroughs, list)
        assert len(breakthroughs) > 0
        assert 'insight_breakthrough' in breakthroughs
        assert 'behavioral_breakthrough' in breakthroughs
    
    @pytest.mark.asyncio
    async def test_stuck_pattern_detection(self, generator):
        """Test stuck pattern detection"""
        stuck_history = [
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="I feel stuck. Nothing is changing and I keep going in circles.",
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_002",
                speaker=ConversationRole.CLIENT,
                content="It's the same problem over and over. I'm not getting anywhere.",
                confidence_score=0.8
            )
        ]
        
        stuck_patterns = await generator._detect_stuck_patterns(stuck_history)
        
        assert isinstance(stuck_patterns, list)
        assert len(stuck_patterns) > 0
        assert 'explicit_stuckness' in stuck_patterns
    
    @pytest.mark.asyncio
    async def test_client_preference_extraction(self, generator):
        """Test client preference extraction"""
        preference_history = [
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="I prefer when you're direct with me. Just tell me what I need to do.",
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_002",
                speaker=ConversationRole.CLIENT,
                content="I want practical solutions, not just talking about feelings.",
                confidence_score=0.8
            )
        ]
        
        preferences = await generator._extract_client_preferences(preference_history)
        
        assert isinstance(preferences, list)
        assert 'direct_approach' in preferences
        assert 'practical_focus' in preferences
    
    @pytest.mark.asyncio
    async def test_conversation_phase_determination(self, generator, session_info):
        """Test conversation phase determination"""
        # Early session, few turns
        early_history = [
            ConversationTurn(turn_id="turn_001", speaker=ConversationRole.THERAPIST, content="Hello", confidence_score=0.8),
            ConversationTurn(turn_id="turn_002", speaker=ConversationRole.CLIENT, content="Hi", confidence_score=0.8)
        ]
        
        phase = await generator._determine_conversation_phase(early_history, {'session_number': 1})
        assert phase == ConversationPhase.INITIAL_ASSESSMENT
        
        # Later in session
        later_history = early_history + [
            ConversationTurn(turn_id=f"turn_{i:03d}", speaker=ConversationRole.CLIENT, content="Response", confidence_score=0.8)
            for i in range(3, 10)
        ]
        
        phase = await generator._determine_conversation_phase(later_history, {'session_number': 1})
        assert phase == ConversationPhase.PROBLEM_EXPLORATION
    
    @pytest.mark.asyncio
    async def test_conversational_context_analysis(self, generator, sample_conversation_history, 
                                                  sample_clinical_context, session_info):
        """Test comprehensive conversational context analysis"""
        context = await generator.analyze_conversational_context(
            sample_conversation_history, sample_clinical_context, session_info
        )
        
        assert isinstance(context, ConversationalContext)
        assert context.turn_number == len(sample_conversation_history)
        assert context.session_number == session_info['session_number']
        assert isinstance(context.time_in_session, timedelta)
        assert isinstance(context.conversation_phase, ConversationPhase)
        assert isinstance(context.emotional_trajectory, list)
        assert isinstance(context.alliance_strength, float)
        assert isinstance(context.engagement_level, float)
        assert isinstance(context.resistance_indicators, list)
        assert isinstance(context.progress_markers, list)
        assert isinstance(context.crisis_indicators, list)
        assert isinstance(context.cultural_considerations, list)
        assert isinstance(context.cognitive_load, float)
        assert isinstance(context.therapeutic_momentum, float)
        
        # Context should be stored in history
        assert len(generator.context_history) == 1
        assert generator.context_history[0] == context
    
    @pytest.mark.asyncio
    async def test_response_context_type_classification(self, generator, sample_conversation_history):
        """Test response context type classification"""
        # Create sample conversational context
        conv_context = ConversationalContext(
            turn_number=4,
            session_number=1,
            time_in_session=timedelta(minutes=20),
            conversation_phase=ConversationPhase.PROBLEM_EXPLORATION,
            emotional_trajectory=[('anxiety', 0.7)],
            alliance_strength=0.6,
            engagement_level=0.7,
            resistance_indicators=[],
            progress_markers=[],
            crisis_indicators=[],
            cultural_considerations=[],
            cognitive_load=0.4,
            therapeutic_momentum=0.5,
            recent_breakthroughs=[],
            stuck_patterns=[],
            client_preferences=[],
            contraindications=[]
        )
        
        # Test crisis response
        crisis_context = conv_context
        crisis_context.crisis_indicators = ['suicidal_ideation']
        
        context_type = generator._classify_response_context_type("I want to die", crisis_context)
        assert context_type == ResponseContextType.CRISIS_RESPONSE
        
        # Test resistance handling
        context_type = generator._classify_response_context_type("That won't work for me", conv_context)
        assert context_type == ResponseContextType.RESISTANCE_HANDLING
        
        # Test processing context
        breakthrough_context = conv_context
        breakthrough_context.recent_breakthroughs = ['insight_breakthrough']
        
        context_type = generator._classify_response_context_type("I see what you mean", breakthrough_context)
        assert context_type == ResponseContextType.PROCESSING
    
    @pytest.mark.asyncio
    async def test_primary_factor_identification(self, generator, sample_clinical_context):
        """Test primary contextual factor identification"""
        # Create context with various factors
        conv_context = ConversationalContext(
            turn_number=5,
            session_number=2,
            time_in_session=timedelta(minutes=30),
            conversation_phase=ConversationPhase.PROBLEM_EXPLORATION,
            emotional_trajectory=[('anxiety', 0.9)],  # High intensity
            alliance_strength=0.3,  # Low alliance
            engagement_level=0.7,
            resistance_indicators=['direct_disagreement'],
            progress_markers=[],
            crisis_indicators=['suicidal_ideation'],
            cultural_considerations=['family_dynamics'],
            cognitive_load=0.8,  # High load
            therapeutic_momentum=0.2,
            recent_breakthroughs=[],
            stuck_patterns=[],
            client_preferences=[],
            contraindications=[]
        )
        
        primary_factors = generator._identify_primary_factors(conv_context, sample_clinical_context)
        
        assert isinstance(primary_factors, list)
        assert ContextualFactor.CRISIS_LEVEL in primary_factors  # Crisis indicators present
        assert ContextualFactor.EMOTIONAL_STATE in primary_factors  # High emotional intensity
        assert ContextualFactor.THERAPEUTIC_ALLIANCE in primary_factors  # Low alliance
        assert ContextualFactor.RESISTANCE_PATTERN in primary_factors  # Resistance present
        assert ContextualFactor.CULTURAL_CONTEXT in primary_factors  # Cultural factors present
        assert ContextualFactor.COGNITIVE_CAPACITY in primary_factors  # High cognitive load
    
    @pytest.mark.asyncio
    async def test_contextual_weight_calculation(self, generator):
        """Test contextual weight calculation"""
        primary_factors = [
            ContextualFactor.CRISIS_LEVEL,
            ContextualFactor.EMOTIONAL_STATE,
            ContextualFactor.THERAPEUTIC_ALLIANCE
        ]
        
        conv_context = ConversationalContext(
            turn_number=3,
            session_number=1,
            time_in_session=timedelta(minutes=15),
            conversation_phase=ConversationPhase.PROBLEM_EXPLORATION,
            emotional_trajectory=[('anxiety', 0.8)],
            alliance_strength=0.2,  # Very low
            engagement_level=0.5,
            resistance_indicators=[],
            progress_markers=[],
            crisis_indicators=['suicidal_ideation'],
            cultural_considerations=[],
            cognitive_load=0.5,
            therapeutic_momentum=0.3,
            recent_breakthroughs=[],
            stuck_patterns=[],
            client_preferences=[],
            contraindications=[]
        )
        
        weights = generator._calculate_contextual_weights(primary_factors, conv_context)
        
        assert isinstance(weights, dict)
        assert ContextualFactor.CRISIS_LEVEL in weights
        assert ContextualFactor.EMOTIONAL_STATE in weights
        assert ContextualFactor.THERAPEUTIC_ALLIANCE in weights
        
        # Crisis should have maximum weight
        assert weights[ContextualFactor.CRISIS_LEVEL] == 1.0
        
        # All weights should be between 0 and 1
        for factor, weight in weights.items():
            assert 0.0 <= weight <= 1.0
    
    @pytest.mark.asyncio
    async def test_crisis_adaptation(self, generator):
        """Test crisis-specific response adaptation"""
        conv_context = ConversationalContext(
            turn_number=2,
            session_number=1,
            time_in_session=timedelta(minutes=10),
            conversation_phase=ConversationPhase.INITIAL_ASSESSMENT,
            emotional_trajectory=[],
            alliance_strength=0.5,
            engagement_level=0.5,
            resistance_indicators=[],
            progress_markers=[],
            crisis_indicators=['suicidal_ideation'],
            cultural_considerations=[],
            cognitive_load=0.5,
            therapeutic_momentum=0.0,
            recent_breakthroughs=[],
            stuck_patterns=[],
            client_preferences=[],
            contraindications=[]
        )
        
        original_content = "How are you feeling about that?"
        adapted_content = generator._adapt_for_crisis(original_content, conv_context)
        
        assert len(adapted_content) > len(original_content)
        assert 'concern' in adapted_content.lower()
        assert 'safety' in adapted_content.lower()
        assert 'priority' in adapted_content.lower()
    
    @pytest.mark.asyncio
    async def test_emotional_state_adaptation(self, generator):
        """Test emotional state adaptation"""
        conv_context = ConversationalContext(
            turn_number=3,
            session_number=1,
            time_in_session=timedelta(minutes=15),
            conversation_phase=ConversationPhase.PROBLEM_EXPLORATION,
            emotional_trajectory=[('overwhelm', 0.9)],  # Very high intensity
            alliance_strength=0.6,
            engagement_level=0.5,
            resistance_indicators=[],
            progress_markers=[],
            crisis_indicators=[],
            cultural_considerations=[],
            cognitive_load=0.5,
            therapeutic_momentum=0.3,
            recent_breakthroughs=[],
            stuck_patterns=[],
            client_preferences=[],
            contraindications=[]
        )
        
        original_content = "Let's explore this further."
        adapted_content = generator._adapt_for_emotional_state(original_content, conv_context)
        
        assert len(adapted_content) > len(original_content)
        assert 'intense' in adapted_content.lower() or 'see' in adapted_content.lower()
        assert 'ground' in adapted_content.lower()  # Should add grounding for high intensity
    
    @pytest.mark.asyncio
    async def test_cognitive_capacity_adaptation(self, generator):
        """Test cognitive capacity adaptation"""
        conv_context = ConversationalContext(
            turn_number=4,
            session_number=2,
            time_in_session=timedelta(minutes=20),
            conversation_phase=ConversationPhase.INTERVENTION_PLANNING,
            emotional_trajectory=[],
            alliance_strength=0.6,
            engagement_level=0.5,
            resistance_indicators=[],
            progress_markers=[],
            crisis_indicators=[],
            cultural_considerations=[],
            cognitive_load=0.8,  # High cognitive load
            therapeutic_momentum=0.4,
            recent_breakthroughs=[],
            stuck_patterns=[],
            client_preferences=[],
            contraindications=[]
        )
        
        original_content = "This therapeutic intervention involves cognitive restructuring techniques."
        adapted_content = generator._adapt_for_cognitive_capacity(original_content, conv_context)
        
        # Should simplify language
        assert 'intervention' not in adapted_content or 'approach' in adapted_content
        assert 'therapeutic' not in adapted_content or 'helpful' in adapted_content
        assert 'cognitive' not in adapted_content or 'thinking' in adapted_content
        assert 'make sense' in adapted_content.lower()  # Should add understanding check
    
    @pytest.mark.asyncio
    async def test_context_statistics(self, generator):
        """Test context statistics retrieval"""
        stats = generator.get_context_statistics()
        
        assert isinstance(stats, dict)
        
        required_keys = [
            'context_history_length', 'response_history_length', 'supported_contextual_factors',
            'response_context_types', 'contextual_patterns', 'adaptation_rules',
            'configuration', 'context_settings'
        ]
        
        for key in required_keys:
            assert key in stats
        
        # Check supported factors
        assert len(stats['supported_contextual_factors']) > 0
        assert 'crisis_level' in stats['supported_contextual_factors']
        assert 'emotional_state' in stats['supported_contextual_factors']
        
        # Check response context types
        assert len(stats['response_context_types']) > 0
        assert 'crisis_response' in stats['response_context_types']
        assert 'exploration' in stats['response_context_types']


if __name__ == "__main__":
    pytest.main([__file__])
