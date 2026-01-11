"""
Unit tests for Conversation Complexity Progression System

Tests the conversation complexity progression system including complexity assessment,
readiness evaluation, progression triggers, and complexity level management.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import pytest
import pytest_asyncio

from .conversation_complexity_progression import (
    ComplexityAssessment,
    ComplexityDimension,
    ComplexityLevel,
    ComplexityProfile,
    ConversationComplexityProgression,
    ProgressionDirection,
    ProgressionTrigger,
)


@dataclass
class MockConversationTurn:
    content: str
    speaker: str = "client"


@dataclass
class MockClinicalContext:
    crisis_indicators: List[str]
    session_number: int
    primary_diagnosis: str = "Depression"


@pytest.fixture
def basic_conversation_history():
    """Create basic conversation history"""
    return [
        MockConversationTurn("I'm feeling okay today"),
        MockConversationTurn("Things have been stable"),
        MockConversationTurn("I understand what you're saying")
    ]


@pytest.fixture
def advanced_conversation_history():
    """Create advanced conversation history"""
    return [
        MockConversationTurn("I feel ready to explore deeper issues"),
        MockConversationTurn("I understand the techniques you taught me"),
        MockConversationTurn("I see the patterns in my relationships now"),
        MockConversationTurn("I realize how my childhood affects my current behavior"),
        MockConversationTurn("I'm ready to be more vulnerable and share personal things")
    ]


@pytest.fixture
def crisis_conversation_history():
    """Create crisis conversation history"""
    return [
        MockConversationTurn("I can't handle this anymore"),
        MockConversationTurn("Everything is overwhelming"),
        MockConversationTurn("I don't know if I can continue")
    ]


@pytest.fixture
def stable_clinical_context():
    """Create stable clinical context"""
    return MockClinicalContext(
        crisis_indicators=[],
        session_number=5
    )


@pytest.fixture
def crisis_clinical_context():
    """Create crisis clinical context"""
    return MockClinicalContext(
        crisis_indicators=["suicidal_ideation", "hopelessness"],
        session_number=2
    )


@pytest.fixture
def session_info():
    """Create session information"""
    return {
        'session_number': 5,
        'session_start': datetime.now() - timedelta(minutes=30),
        'session_type': 'individual_therapy'
    }


@pytest_asyncio.fixture
async def progression_system():
    """Create complexity progression system instance"""
    return ConversationComplexityProgression()


class TestConversationComplexityProgression:
    """Test cases for ConversationComplexityProgression"""
    
    def test_initialization(self, progression_system):
        """Test system initialization"""
        assert progression_system.config is not None
        assert progression_system.complexity_profiles is not None
        assert progression_system.progression_criteria is not None
        assert progression_system.assessment_history == []
        assert progression_system.progression_history == []
        assert progression_system.current_complexity is None
        assert progression_system.session_complexity_tracking == {}
    
    def test_complexity_profiles_initialization(self, progression_system):
        """Test complexity profiles are properly initialized"""
        profiles = progression_system.complexity_profiles
        
        # Check all complexity levels are present
        assert ComplexityLevel.BASIC in profiles
        assert ComplexityLevel.INTERMEDIATE in profiles
        assert ComplexityLevel.ADVANCED in profiles
        assert ComplexityLevel.EXPERT in profiles
        
        # Check basic profile
        basic_profile = profiles[ComplexityLevel.BASIC]
        assert basic_profile.level == ComplexityLevel.BASIC
        assert len(basic_profile.dimensions) == 8
        assert all(0.0 <= score <= 1.0 for score in basic_profile.dimensions.values())
        assert basic_profile.description
        assert basic_profile.prerequisites
        assert basic_profile.indicators
        
        # Check expert profile has higher scores
        expert_profile = profiles[ComplexityLevel.EXPERT]
        assert expert_profile.dimensions[ComplexityDimension.EMOTIONAL_DEPTH] > basic_profile.dimensions[ComplexityDimension.EMOTIONAL_DEPTH]
        assert expert_profile.dimensions[ComplexityDimension.COGNITIVE_LOAD] > basic_profile.dimensions[ComplexityDimension.COGNITIVE_LOAD]
    
    def test_progression_criteria_initialization(self, progression_system):
        """Test progression criteria are properly initialized"""
        criteria = progression_system.progression_criteria
        
        assert len(criteria) > 0
        
        # Check criteria structure
        for criterion in criteria:
            assert isinstance(criterion.trigger, ProgressionTrigger)
            assert 0.0 <= criterion.threshold <= 1.0
            assert 0.0 <= criterion.weight <= 1.0
            assert isinstance(criterion.direction, ProgressionDirection)
            assert isinstance(criterion.conditions, list)
            assert criterion.evaluation_method
    
    @pytest.mark.asyncio
    async def test_emotional_depth_readiness_assessment(self, progression_system, advanced_conversation_history, stable_clinical_context):
        """Test emotional depth readiness assessment"""
        score = await progression_system._assess_emotional_depth_readiness(
            advanced_conversation_history, stable_clinical_context
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Should be higher for advanced conversation
        assert score > 0.5
    
    @pytest.mark.asyncio
    async def test_emotional_depth_with_crisis(self, progression_system, crisis_conversation_history, crisis_clinical_context):
        """Test emotional depth assessment with crisis indicators"""
        score = await progression_system._assess_emotional_depth_readiness(
            crisis_conversation_history, crisis_clinical_context
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Should be lower due to crisis indicators
        assert score < 0.5
    
    @pytest.mark.asyncio
    async def test_cognitive_load_readiness_assessment(self, progression_system, basic_conversation_history, stable_clinical_context):
        """Test cognitive load readiness assessment"""
        score = await progression_system._assess_cognitive_load_readiness(
            basic_conversation_history, stable_clinical_context
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_cognitive_load_with_confusion(self, progression_system, stable_clinical_context):
        """Test cognitive load assessment with confusion indicators"""
        confused_history = [
            MockConversationTurn("I'm confused about what you're saying"),
            MockConversationTurn("This is too complex for me"),
            MockConversationTurn("I don't understand")
        ]
        
        score = await progression_system._assess_cognitive_load_readiness(
            confused_history, stable_clinical_context
        )
        
        # Should be lower due to confusion indicators
        assert score < 0.5
    
    @pytest.mark.asyncio
    async def test_technique_readiness_assessment(self, progression_system, stable_clinical_context):
        """Test therapeutic technique readiness assessment"""
        engaged_history = [
            MockConversationTurn("I practiced the technique you taught me"),
            MockConversationTurn("The strategy really helped"),
            MockConversationTurn("I want to learn more skills")
        ]
        
        score = await progression_system._assess_technique_readiness(
            engaged_history, stable_clinical_context
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Should be higher for technique engagement
        assert score > 0.5
    
    @pytest.mark.asyncio
    async def test_technique_readiness_with_resistance(self, progression_system, stable_clinical_context):
        """Test technique readiness with resistance"""
        resistant_history = [
            MockConversationTurn("That technique won't work for me"),
            MockConversationTurn("I tried that before and it doesn't help"),
            MockConversationTurn("This approach is not for me")
        ]
        
        score = await progression_system._assess_technique_readiness(
            resistant_history, stable_clinical_context
        )
        
        # Should be lower due to resistance
        assert score < 0.5
    
    @pytest.mark.asyncio
    async def test_insight_readiness_assessment(self, progression_system, stable_clinical_context):
        """Test insight readiness assessment"""
        insightful_history = [
            MockConversationTurn("I realize now that I have this pattern"),
            MockConversationTurn("I see the connection between my past and present"),
            MockConversationTurn("Aha! That makes so much sense now")
        ]
        
        score = await progression_system._assess_insight_readiness(
            insightful_history, stable_clinical_context
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Should be higher for insight indicators
        assert score > 0.5
    
    @pytest.mark.asyncio
    async def test_vulnerability_readiness_assessment(self, progression_system, stable_clinical_context):
        """Test vulnerability readiness assessment"""
        vulnerable_history = [
            MockConversationTurn("I feel safe sharing this with you"),
            MockConversationTurn("I trust you enough to be vulnerable"),
            MockConversationTurn("I want to open up about something personal")
        ]
        
        score = await progression_system._assess_vulnerability_readiness(
            vulnerable_history, stable_clinical_context
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Should be higher for vulnerability indicators
        assert score > 0.4
    
    @pytest.mark.asyncio
    async def test_vulnerability_with_crisis(self, progression_system, crisis_clinical_context):
        """Test vulnerability readiness with crisis"""
        history = [MockConversationTurn("I want to share something")]
        
        score = await progression_system._assess_vulnerability_readiness(
            history, crisis_clinical_context
        )
        
        # Should be lower due to crisis indicators
        assert score < 0.5
    
    @pytest.mark.asyncio
    async def test_abstraction_readiness_assessment(self, progression_system, stable_clinical_context):
        """Test abstraction readiness assessment"""
        abstract_history = [
            MockConversationTurn("I understand the concept you're explaining"),
            MockConversationTurn("That's an interesting philosophical idea"),
            MockConversationTurn("The theory makes sense to me")
        ]
        
        score = await progression_system._assess_abstraction_readiness(
            abstract_history, stable_clinical_context
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Should be higher for abstract engagement
        assert score > 0.5
    
    @pytest.mark.asyncio
    async def test_intervention_readiness_assessment(self, progression_system, stable_clinical_context):
        """Test intervention readiness assessment"""
        successful_history = [
            MockConversationTurn("That intervention really helped me"),
            MockConversationTurn("I'm seeing progress from our work"),
            MockConversationTurn("The approach is working well")
        ]
        
        score = await progression_system._assess_intervention_readiness(
            successful_history, stable_clinical_context
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Should be higher for intervention success
        assert score > 0.5
    
    @pytest.mark.asyncio
    async def test_relational_readiness_assessment(self, progression_system, stable_clinical_context):
        """Test relational readiness assessment"""
        relational_history = [
            MockConversationTurn("I feel a good connection with you"),
            MockConversationTurn("Our relationship feels safe"),
            MockConversationTurn("I trust how we work together")
        ]
        
        score = await progression_system._assess_relational_readiness(
            relational_history, stable_clinical_context
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Should be higher for relational indicators
        assert score > 0.5
    
    def test_determine_current_level_by_session(self, progression_system):
        """Test current level determination by session number"""
        # Early sessions
        level = progression_system._determine_current_level([], {'session_number': 2})
        assert level == ComplexityLevel.BASIC
        
        # Mid sessions
        level = progression_system._determine_current_level([], {'session_number': 6})
        assert level == ComplexityLevel.INTERMEDIATE
        
        # Advanced sessions
        level = progression_system._determine_current_level([], {'session_number': 12})
        assert level == ComplexityLevel.ADVANCED
        
        # Expert sessions
        level = progression_system._determine_current_level([], {'session_number': 20})
        assert level == ComplexityLevel.EXPERT
    
    def test_determine_current_level_with_tracking(self, progression_system):
        """Test current level determination with session tracking"""
        # Set tracked complexity
        progression_system.session_complexity_tracking[5] = ComplexityLevel.ADVANCED
        
        level = progression_system._determine_current_level([], {'session_number': 5})
        assert level == ComplexityLevel.ADVANCED
    
    @pytest.mark.asyncio
    async def test_recommend_complexity_level_basic(self, progression_system, stable_clinical_context):
        """Test complexity level recommendation for basic readiness"""
        low_readiness_scores = {dim: 0.3 for dim in ComplexityDimension}
        
        recommended = await progression_system._recommend_complexity_level(
            low_readiness_scores, ComplexityLevel.BASIC, stable_clinical_context
        )
        
        assert recommended == ComplexityLevel.BASIC
    
    @pytest.mark.asyncio
    async def test_recommend_complexity_level_advanced(self, progression_system, stable_clinical_context):
        """Test complexity level recommendation for advanced readiness"""
        high_readiness_scores = {dim: 0.8 for dim in ComplexityDimension}
        
        recommended = await progression_system._recommend_complexity_level(
            high_readiness_scores, ComplexityLevel.INTERMEDIATE, stable_clinical_context
        )
        
        # Should recommend advanced (one level up from intermediate)
        assert recommended == ComplexityLevel.ADVANCED
    
    @pytest.mark.asyncio
    async def test_recommend_complexity_level_with_crisis(self, progression_system, crisis_clinical_context):
        """Test complexity level recommendation with crisis"""
        high_readiness_scores = {dim: 0.9 for dim in ComplexityDimension}
        
        recommended = await progression_system._recommend_complexity_level(
            high_readiness_scores, ComplexityLevel.ADVANCED, crisis_clinical_context
        )
        
        # Should recommend basic due to crisis
        assert recommended == ComplexityLevel.BASIC
    
    @pytest.mark.asyncio
    async def test_identify_progression_triggers(self, progression_system, stable_clinical_context):
        """Test progression trigger identification"""
        high_readiness_scores = {
            ComplexityDimension.EMOTIONAL_DEPTH: 0.8,
            ComplexityDimension.THERAPEUTIC_TECHNIQUES: 0.8,
            ComplexityDimension.INSIGHT_REQUIREMENTS: 0.8,
            ComplexityDimension.COGNITIVE_LOAD: 0.7,
            ComplexityDimension.VULNERABILITY_LEVEL: 0.7,
            ComplexityDimension.ABSTRACTION_LEVEL: 0.6,
            ComplexityDimension.INTERVENTION_SOPHISTICATION: 0.7,
            ComplexityDimension.RELATIONAL_COMPLEXITY: 0.7
        }
        
        triggers = await progression_system._identify_progression_triggers(
            [], stable_clinical_context, high_readiness_scores
        )
        
        assert isinstance(triggers, list)
        assert ProgressionTrigger.CLIENT_READINESS in triggers
        assert ProgressionTrigger.SKILL_MASTERY in triggers
        assert ProgressionTrigger.INSIGHT_DEVELOPMENT in triggers
    
    @pytest.mark.asyncio
    async def test_identify_progression_triggers_with_crisis(self, progression_system, crisis_clinical_context):
        """Test progression trigger identification with crisis"""
        readiness_scores = {dim: 0.5 for dim in ComplexityDimension}
        
        triggers = await progression_system._identify_progression_triggers(
            [], crisis_clinical_context, readiness_scores
        )
        
        assert ProgressionTrigger.CRISIS_RESOLUTION in triggers
    
    @pytest.mark.asyncio
    async def test_identify_blocking_factors(self, progression_system, stable_clinical_context):
        """Test blocking factor identification"""
        low_readiness_scores = {
            ComplexityDimension.EMOTIONAL_DEPTH: 0.2,
            ComplexityDimension.COGNITIVE_LOAD: 0.1,
            ComplexityDimension.THERAPEUTIC_TECHNIQUES: 0.4,
            ComplexityDimension.INSIGHT_REQUIREMENTS: 0.5,
            ComplexityDimension.VULNERABILITY_LEVEL: 0.3,
            ComplexityDimension.ABSTRACTION_LEVEL: 0.4,
            ComplexityDimension.INTERVENTION_SOPHISTICATION: 0.4,
            ComplexityDimension.RELATIONAL_COMPLEXITY: 0.4
        }
        
        blocking_factors = await progression_system._identify_blocking_factors(
            [], stable_clinical_context, low_readiness_scores
        )
        
        assert isinstance(blocking_factors, list)
        assert len(blocking_factors) > 0
        assert "low_emotional_depth_readiness" in blocking_factors
        assert "cognitive_overload" in blocking_factors
        assert "emotional_instability" in blocking_factors
    
    @pytest.mark.asyncio
    async def test_identify_blocking_factors_with_crisis(self, progression_system, crisis_clinical_context):
        """Test blocking factor identification with crisis"""
        readiness_scores = {dim: 0.5 for dim in ComplexityDimension}
        
        blocking_factors = await progression_system._identify_blocking_factors(
            [], crisis_clinical_context, readiness_scores
        )
        
        assert "crisis_suicidal_ideation" in blocking_factors
        assert "crisis_hopelessness" in blocking_factors
    
    def test_calculate_confidence_score(self, progression_system):
        """Test confidence score calculation"""
        readiness_scores = {dim: 0.6 for dim in ComplexityDimension}
        progression_triggers = [ProgressionTrigger.CLIENT_READINESS]
        blocking_factors = ["low_emotional_depth_readiness"]
        
        confidence = progression_system._calculate_confidence_score(
            readiness_scores, progression_triggers, blocking_factors
        )
        
        assert isinstance(confidence, float)
        assert 0.1 <= confidence <= 1.0
    
    def test_calculate_confidence_score_high_variance(self, progression_system):
        """Test confidence score with high variance in readiness"""
        # High variance in scores
        readiness_scores = {
            ComplexityDimension.EMOTIONAL_DEPTH: 0.9,
            ComplexityDimension.COGNITIVE_LOAD: 0.1,
            ComplexityDimension.THERAPEUTIC_TECHNIQUES: 0.8,
            ComplexityDimension.INSIGHT_REQUIREMENTS: 0.2,
            ComplexityDimension.VULNERABILITY_LEVEL: 0.7,
            ComplexityDimension.ABSTRACTION_LEVEL: 0.3,
            ComplexityDimension.INTERVENTION_SOPHISTICATION: 0.6,
            ComplexityDimension.RELATIONAL_COMPLEXITY: 0.4
        }
        
        confidence = progression_system._calculate_confidence_score(
            readiness_scores, [], []
        )
        
        # Should be lower due to high variance
        assert confidence < 0.7
    
    def test_generate_assessment_reasoning(self, progression_system):
        """Test assessment reasoning generation"""
        readiness_scores = {
            ComplexityDimension.EMOTIONAL_DEPTH: 0.8,
            ComplexityDimension.COGNITIVE_LOAD: 0.3,
            ComplexityDimension.THERAPEUTIC_TECHNIQUES: 0.7,
            ComplexityDimension.INSIGHT_REQUIREMENTS: 0.6,
            ComplexityDimension.VULNERABILITY_LEVEL: 0.5,
            ComplexityDimension.ABSTRACTION_LEVEL: 0.4,
            ComplexityDimension.INTERVENTION_SOPHISTICATION: 0.6,
            ComplexityDimension.RELATIONAL_COMPLEXITY: 0.5
        }
        
        reasoning = progression_system._generate_assessment_reasoning(
            ComplexityLevel.BASIC,
            ComplexityLevel.INTERMEDIATE,
            readiness_scores,
            [ProgressionTrigger.CLIENT_READINESS],
            ["cognitive_overload"]
        )
        
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert "basic" in reasoning.lower()
        assert "intermediate" in reasoning.lower()
        assert "emotional_depth" in reasoning.lower()  # High score
        assert "cognitive_load" in reasoning.lower()   # Low score
        assert "client readiness" in reasoning.lower()  # Trigger
        assert "cognitive overload" in reasoning.lower()  # Blocking factor
    
    @pytest.mark.asyncio
    async def test_assess_complexity_readiness_comprehensive(self, progression_system, advanced_conversation_history, stable_clinical_context, session_info):
        """Test comprehensive complexity readiness assessment"""
        assessment = await progression_system.assess_complexity_readiness(
            advanced_conversation_history, stable_clinical_context, session_info
        )
        
        assert isinstance(assessment, ComplexityAssessment)
        assert isinstance(assessment.current_level, ComplexityLevel)
        assert isinstance(assessment.recommended_level, ComplexityLevel)
        assert isinstance(assessment.readiness_scores, dict)
        assert len(assessment.readiness_scores) == 8  # All dimensions
        assert isinstance(assessment.progression_triggers, list)
        assert isinstance(assessment.blocking_factors, list)
        assert 0.0 <= assessment.confidence_score <= 1.0
        assert isinstance(assessment.assessment_timestamp, datetime)
        assert isinstance(assessment.reasoning, str)
        
        # Assessment should be stored in history
        assert len(progression_system.assessment_history) == 1
        assert progression_system.assessment_history[0] == assessment
    
    @pytest.mark.asyncio
    async def test_assess_complexity_readiness_with_crisis(self, progression_system, crisis_conversation_history, crisis_clinical_context, session_info):
        """Test complexity readiness assessment with crisis"""
        assessment = await progression_system.assess_complexity_readiness(
            crisis_conversation_history, crisis_clinical_context, session_info
        )
        
        # Should recommend basic level due to crisis
        assert assessment.recommended_level == ComplexityLevel.BASIC
        assert ProgressionTrigger.CRISIS_RESOLUTION in assessment.progression_triggers
        assert any("crisis" in factor for factor in assessment.blocking_factors)
    
    @pytest.mark.asyncio
    async def test_apply_complexity_progression_with_change(self, progression_system):
        """Test applying complexity progression with level change"""
        assessment = ComplexityAssessment(
            current_level=ComplexityLevel.BASIC,
            recommended_level=ComplexityLevel.INTERMEDIATE,
            readiness_scores={dim: 0.6 for dim in ComplexityDimension},
            progression_triggers=[ProgressionTrigger.CLIENT_READINESS],
            blocking_factors=[],
            confidence_score=0.8,
            assessment_timestamp=datetime.now(),
            reasoning="Client ready for progression"
        )
        
        success = await progression_system.apply_complexity_progression(
            assessment, session_number=5, therapist_notes="Good progress"
        )
        
        assert success is True
        assert len(progression_system.progression_history) == 1
        assert progression_system.session_complexity_tracking[5] == ComplexityLevel.INTERMEDIATE
        assert progression_system.current_complexity == ComplexityLevel.INTERMEDIATE
        
        # Check progression history
        progression = progression_system.progression_history[0]
        assert progression.session_number == 5
        assert progression.from_level == ComplexityLevel.BASIC
        assert progression.to_level == ComplexityLevel.INTERMEDIATE
        assert progression.therapist_notes == "Good progress"
    
    @pytest.mark.asyncio
    async def test_apply_complexity_progression_no_change(self, progression_system):
        """Test applying complexity progression with no level change"""
        assessment = ComplexityAssessment(
            current_level=ComplexityLevel.INTERMEDIATE,
            recommended_level=ComplexityLevel.INTERMEDIATE,
            readiness_scores={dim: 0.5 for dim in ComplexityDimension},
            progression_triggers=[],
            blocking_factors=[],
            confidence_score=0.7,
            assessment_timestamp=datetime.now(),
            reasoning="Maintain current level"
        )
        
        success = await progression_system.apply_complexity_progression(
            assessment, session_number=6
        )
        
        assert success is False
        assert len(progression_system.progression_history) == 0
        assert progression_system.session_complexity_tracking[6] == ComplexityLevel.INTERMEDIATE
        assert progression_system.current_complexity == ComplexityLevel.INTERMEDIATE
    
    def test_get_complexity_profile(self, progression_system):
        """Test getting complexity profile"""
        profile = progression_system.get_complexity_profile(ComplexityLevel.ADVANCED)
        
        assert isinstance(profile, ComplexityProfile)
        assert profile.level == ComplexityLevel.ADVANCED
        assert len(profile.dimensions) == 8
        assert profile.description
        assert profile.prerequisites
        assert profile.indicators
        assert profile.contraindications
    
    def test_get_progression_statistics_empty(self, progression_system):
        """Test getting progression statistics with no history"""
        stats = progression_system.get_progression_statistics()
        
        assert isinstance(stats, dict)
        assert stats['total_assessments'] == 0
        assert stats['total_progressions'] == 0
        assert stats['current_complexity'] is None
        assert stats['complexity_distribution'] == {}
        assert stats['progression_triggers'] == {}
        assert stats['blocking_factors'] == {}
        assert stats['average_confidence'] == 0.0
    
    @pytest.mark.asyncio
    async def test_get_progression_statistics_with_data(self, progression_system, advanced_conversation_history, stable_clinical_context, session_info):
        """Test getting progression statistics with assessment data"""
        # Create some assessment history
        await progression_system.assess_complexity_readiness(
            advanced_conversation_history, stable_clinical_context, session_info
        )
        
        stats = progression_system.get_progression_statistics()
        
        assert stats['total_assessments'] == 1
        assert stats['complexity_distribution']
        assert stats['average_confidence'] > 0.0
    
    def test_export_progression_data_json(self, progression_system):
        """Test exporting progression data as JSON"""
        json_data = progression_system.export_progression_data(format='json')
        
        assert isinstance(json_data, str)
        # Should be valid JSON
        import json
        parsed_data = json.loads(json_data)
        assert 'configuration' in parsed_data
        assert 'complexity_profiles' in parsed_data
        assert 'assessment_history' in parsed_data
        assert 'progression_history' in parsed_data
        assert 'statistics' in parsed_data
    
    def test_export_progression_data_dict(self, progression_system):
        """Test exporting progression data as dictionary"""
        dict_data = progression_system.export_progression_data(format='dict')
        
        assert isinstance(dict_data, dict)
        assert 'configuration' in dict_data
        assert 'complexity_profiles' in dict_data
        assert 'assessment_history' in dict_data
        assert 'progression_history' in dict_data
        assert 'statistics' in dict_data
        
        # Check complexity profiles structure
        profiles = dict_data['complexity_profiles']
        assert 'basic' in profiles
        assert 'intermediate' in profiles
        assert 'advanced' in profiles
        assert 'expert' in profiles
        
        for level_name, profile_data in profiles.items():
            assert 'dimensions' in profile_data
            assert 'description' in profile_data
            assert 'prerequisites' in profile_data
            assert 'indicators' in profile_data
            assert 'contraindications' in profile_data


if __name__ == "__main__":
    pytest.main([__file__])
