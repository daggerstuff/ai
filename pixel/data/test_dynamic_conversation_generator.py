"""
Unit tests for Dynamic Conversation Generator

Tests the dynamic conversation generation system including conversation
parameters, turn generation, validation, quality assessment, and export functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from .dynamic_conversation_generator import (
    ClientResponseStyle,
    ConversationComplexity,
    ConversationParameters,
    ConversationPhase,
    ConversationTurn,
    DynamicConversationGenerator,
    GeneratedConversation,
)
from .therapeutic_conversation_schema import ConversationRole, TherapeuticModality
from .therapist_response_generator import InterventionType


@pytest.fixture
def basic_conversation_parameters():
    """Create basic conversation parameters for testing"""
    return ConversationParameters(
        therapeutic_modality=TherapeuticModality.CBT,
        client_presentation="Client presents with symptoms of depression and anxiety",
        primary_diagnosis="Major Depressive Disorder",
        conversation_phase=ConversationPhase.PROBLEM_EXPLORATION,
        complexity_level=ConversationComplexity.INTERMEDIATE,
        client_response_style=ClientResponseStyle.COOPERATIVE,
        session_number=3,
        therapeutic_goals=["Reduce depressive symptoms", "Improve coping skills"],
        cultural_factors=["Hispanic/Latino background"],
        conversation_length=8,
        include_clinical_notes=True,
        validate_responses=True
    )


@pytest.fixture
def crisis_conversation_parameters():
    """Create conversation parameters with crisis indicators"""
    return ConversationParameters(
        therapeutic_modality=TherapeuticModality.CBT,
        client_presentation="Client expressing suicidal ideation and hopelessness",
        primary_diagnosis="Major Depressive Disorder",
        conversation_phase=ConversationPhase.INITIAL_ASSESSMENT,
        complexity_level=ConversationComplexity.ADVANCED,
        client_response_style=ClientResponseStyle.CRISIS,
        session_number=1,
        therapeutic_goals=["Safety planning", "Crisis stabilization"],
        crisis_indicators=["suicidal ideation", "hopelessness"],
        conversation_length=6,
        validate_responses=True
    )


@pytest.fixture
def resistant_client_parameters():
    """Create parameters for resistant client conversation"""
    return ConversationParameters(
        therapeutic_modality=TherapeuticModality.PSYCHODYNAMIC,
        client_presentation="Client resistant to therapy, defensive about problems",
        conversation_phase=ConversationPhase.RAPPORT_BUILDING,
        complexity_level=ConversationComplexity.ADVANCED,
        client_response_style=ClientResponseStyle.RESISTANT,
        session_number=2,
        therapeutic_goals=["Build therapeutic alliance", "Explore resistance"],
        conversation_length=10,
        validate_responses=True
    )


@pytest.fixture
async def generator():
    """Create dynamic conversation generator instance"""
    with patch('ai.pixel.data.dynamic_conversation_generator.PsychologyKnowledgeProcessor'):
        with patch('ai.pixel.data.dynamic_conversation_generator.ClinicalKnowledgeEmbedder'):
            with patch('ai.pixel.data.dynamic_conversation_generator.ClinicalSimilaritySearch'):
                with patch('ai.pixel.data.dynamic_conversation_generator.TherapistResponseGenerator'):
                    generator = DynamicConversationGenerator()
                    return generator


class TestDynamicConversationGenerator:
    """Test cases for DynamicConversationGenerator"""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test generator initialization"""
        with patch('ai.pixel.data.dynamic_conversation_generator.PsychologyKnowledgeProcessor'):
            with patch('ai.pixel.data.dynamic_conversation_generator.ClinicalKnowledgeEmbedder'):
                with patch('ai.pixel.data.dynamic_conversation_generator.ClinicalSimilaritySearch'):
                    with patch('ai.pixel.data.dynamic_conversation_generator.TherapistResponseGenerator'):
                        generator = DynamicConversationGenerator()
                        
                        assert generator.config is not None
                        assert generator.conversation_templates is not None
                        assert generator.client_response_patterns is not None
                        assert generator.intervention_sequences is not None
                        assert generator.generation_settings is not None
    
    @pytest.mark.asyncio
    async def test_parameter_validation_valid(self, generator, basic_conversation_parameters):
        """Test parameter validation with valid parameters"""
        # Should not raise exception
        generator._validate_parameters(basic_conversation_parameters)
    
    @pytest.mark.asyncio
    async def test_parameter_validation_short_conversation(self, generator):
        """Test parameter validation with too short conversation"""
        params = ConversationParameters(
            therapeutic_modality=TherapeuticModality.CBT,
            client_presentation="Test",
            conversation_length=2  # Too short
        )
        
        with pytest.raises(ValueError, match="Conversation length too short"):
            generator._validate_parameters(params)
    
    @pytest.mark.asyncio
    async def test_parameter_validation_long_conversation(self, generator):
        """Test parameter validation with too long conversation"""
        params = ConversationParameters(
            therapeutic_modality=TherapeuticModality.CBT,
            client_presentation="Test",
            conversation_length=100  # Too long
        )
        
        with pytest.raises(ValueError, match="Conversation length too long"):
            generator._validate_parameters(params)
    
    @pytest.mark.asyncio
    async def test_parameter_validation_empty_presentation(self, generator):
        """Test parameter validation with empty client presentation"""
        params = ConversationParameters(
            therapeutic_modality=TherapeuticModality.CBT,
            client_presentation="",  # Empty
            conversation_length=10
        )
        
        with pytest.raises(ValueError, match="Client presentation cannot be empty"):
            generator._validate_parameters(params)
    
    @pytest.mark.asyncio
    async def test_retrieve_clinical_knowledge(self, generator, basic_conversation_parameters):
        """Test clinical knowledge retrieval"""
        # Mock similarity search
        mock_results = [
            {'content': 'CBT techniques for depression', 'relevance': 0.9},
            {'content': 'Cultural considerations in therapy', 'relevance': 0.8}
        ]
        
        generator.similarity_search.search_clinical_knowledge = AsyncMock(return_value=mock_results)
        
        knowledge = await generator._retrieve_clinical_knowledge(basic_conversation_parameters)
        
        assert len(knowledge) == 2
        assert knowledge[0]['content'] == 'CBT techniques for depression'
        generator.similarity_search.search_clinical_knowledge.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_opening_turn_normal(self, generator, basic_conversation_parameters):
        """Test opening turn generation for normal session"""
        opening_turn = await generator._generate_opening_turn(basic_conversation_parameters, [])
        
        assert opening_turn.speaker == ConversationRole.THERAPIST
        assert opening_turn.turn_id == "turn_000"
        assert len(opening_turn.content) > 0
        assert opening_turn.clinical_rationale is not None
        assert opening_turn.intervention_type == InterventionType.ASSESSMENT
        assert opening_turn.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_generate_opening_turn_crisis(self, generator, crisis_conversation_parameters):
        """Test opening turn generation for crisis session"""
        opening_turn = await generator._generate_opening_turn(crisis_conversation_parameters, [])
        
        assert opening_turn.speaker == ConversationRole.THERAPIST
        assert opening_turn.intervention_type == InterventionType.CRISIS_INTERVENTION
        assert "safe" in opening_turn.content.lower()
        assert opening_turn.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_generate_opening_turn_follow_up_session(self, generator, basic_conversation_parameters):
        """Test opening turn generation for follow-up session"""
        basic_conversation_parameters.session_number = 5
        
        opening_turn = await generator._generate_opening_turn(basic_conversation_parameters, [])
        
        assert opening_turn.speaker == ConversationRole.THERAPIST
        assert "since we last met" in opening_turn.content.lower() or "how have things been" in opening_turn.content.lower()
    
    @pytest.mark.asyncio
    async def test_language_simplification(self, generator):
        """Test language simplification for basic complexity"""
        complex_text = "This cognitive distortion represents a maladaptive pattern requiring psychoeducation intervention."
        simplified = generator._simplify_language(complex_text)
        
        assert "unhelpful thought pattern" in simplified
        assert "unhelpful" in simplified
        assert "learning about" in simplified
        assert "technique" in simplified
    
    @pytest.mark.asyncio
    async def test_clinical_language_enhancement(self, generator):
        """Test clinical language enhancement for expert level"""
        basic_text = "That's a good observation."
        clinical_knowledge = [
            {'clinical_terms': ['cognitive-behavioral principles', 'therapeutic alliance']}
        ]
        
        enhanced = generator._enhance_clinical_language(basic_text, clinical_knowledge)
        
        # Should contain original text plus clinical enhancement
        assert basic_text in enhanced
        assert len(enhanced) > len(basic_text)
    
    @pytest.mark.asyncio
    async def test_cultural_adaptation_hispanic(self, generator):
        """Test cultural adaptation for Hispanic/Latino factors"""
        content = "Let's talk about your family relationships."
        cultural_factors = ["Hispanic/Latino background"]
        
        adapted = await generator._culturally_adapt_response(content, cultural_factors)
        
        # Should contain original content
        assert content in adapted
        # May contain additional cultural considerations
        assert len(adapted) >= len(content)
    
    @pytest.mark.asyncio
    async def test_modality_consistency_cbt(self, generator, basic_conversation_parameters):
        """Test modality consistency enforcement for CBT"""
        from .therapist_response_generator import TherapistResponse
        
        response = TherapistResponse(
            content="That's interesting.",
            clinical_rationale="Basic response",
            therapeutic_technique="Reflection",
            intervention_type=InterventionType.REFLECTION,
            confidence_score=0.8,
            contraindications=[],
            follow_up_suggestions=[]
        )
        
        consistent_response = await generator._ensure_modality_consistency(
            response, TherapeuticModality.CBT, []
        )
        
        # Should contain CBT-specific language
        content_lower = consistent_response.content.lower()
        cbt_keywords = ['thoughts', 'feelings', 'behaviors', 'evidence']
        assert any(keyword in content_lower for keyword in cbt_keywords)
    
    @pytest.mark.asyncio
    async def test_modality_consistency_dbt(self, generator):
        """Test modality consistency enforcement for DBT"""
        from .therapist_response_generator import TherapistResponse
        
        response = TherapistResponse(
            content="I understand.",
            clinical_rationale="Basic response",
            therapeutic_technique="Validation",
            intervention_type=InterventionType.VALIDATION,
            confidence_score=0.8,
            contraindications=[],
            follow_up_suggestions=[]
        )
        
        consistent_response = await generator._ensure_modality_consistency(
            response, TherapeuticModality.DBT, []
        )
        
        # Should contain DBT-specific language
        content_lower = consistent_response.content.lower()
        assert "skills" in content_lower
    
    @pytest.mark.asyncio
    async def test_contextual_content_depression(self, generator, basic_conversation_parameters):
        """Test contextual content addition for depression presentation"""
        base_response = "I understand."
        
        contextual = await generator._add_contextual_content(
            base_response, basic_conversation_parameters, None, []
        )
        
        # Should contain depression-related content
        assert "tired" in contextual.lower()
        assert len(contextual) > len(base_response)
    
    @pytest.mark.asyncio
    async def test_contextual_content_anxiety(self, generator):
        """Test contextual content addition for anxiety presentation"""
        params = ConversationParameters(
            therapeutic_modality=TherapeuticModality.CBT,
            client_presentation="Client presents with severe anxiety and panic attacks",
            conversation_length=8
        )
        
        base_response = "Yes, that's right."
        
        contextual = await generator._add_contextual_content(
            base_response, params, None, []
        )
        
        # Should contain anxiety-related content
        assert "worry" in contextual.lower()
    
    @pytest.mark.asyncio
    async def test_contextual_content_crisis(self, generator, crisis_conversation_parameters):
        """Test contextual content addition for crisis presentation"""
        base_response = "I agree."
        
        contextual = await generator._add_contextual_content(
            base_response, crisis_conversation_parameters, None, []
        )
        
        # Should contain crisis-related content
        crisis_indicators = ["overwhelming", "handle", "drowning"]
        assert any(indicator in contextual.lower() for indicator in crisis_indicators)
    
    @pytest.mark.asyncio
    async def test_contextual_content_cultural(self, generator):
        """Test contextual content addition with cultural factors"""
        params = ConversationParameters(
            therapeutic_modality=TherapeuticModality.CBT,
            client_presentation="Test presentation",
            cultural_factors=["Hispanic/Latino background"],
            conversation_length=8
        )
        
        base_response = "I see."
        
        # Mock random to ensure cultural content is added
        with patch('random.random', return_value=0.1):  # Below 0.3 threshold
            contextual = await generator._add_contextual_content(
                base_response, params, None, []
            )
        
        # Should contain cultural context
        assert "family" in contextual.lower()
    
    @pytest.mark.asyncio
    async def test_turn_validation_valid(self, generator, basic_conversation_parameters):
        """Test turn validation with valid turn"""
        turn = ConversationTurn(
            turn_id="turn_001",
            speaker=ConversationRole.THERAPIST,
            content="How are you feeling today?",
            clinical_rationale="Assessment of current emotional state",
            intervention_type=InterventionType.ASSESSMENT,
            confidence_score=0.8
        )
        
        is_valid, issues = await generator._validate_turn(turn, basic_conversation_parameters, [])
        
        assert is_valid
        assert len(issues) == 0
    
    @pytest.mark.asyncio
    async def test_turn_validation_too_short(self, generator, basic_conversation_parameters):
        """Test turn validation with too short content"""
        turn = ConversationTurn(
            turn_id="turn_001",
            speaker=ConversationRole.CLIENT,
            content="Yes",  # Too short
            confidence_score=0.8
        )
        
        is_valid, issues = await generator._validate_turn(turn, basic_conversation_parameters, [])
        
        assert not is_valid
        assert "Content too short" in issues
    
    @pytest.mark.asyncio
    async def test_turn_validation_too_long(self, generator, basic_conversation_parameters):
        """Test turn validation with too long content"""
        turn = ConversationTurn(
            turn_id="turn_001",
            speaker=ConversationRole.CLIENT,
            content="A" * 600,  # Too long
            confidence_score=0.8
        )
        
        is_valid, issues = await generator._validate_turn(turn, basic_conversation_parameters, [])
        
        assert not is_valid
        assert "Content too long" in issues
    
    @pytest.mark.asyncio
    async def test_turn_validation_inappropriate_content(self, generator, basic_conversation_parameters):
        """Test turn validation with inappropriate content"""
        turn = ConversationTurn(
            turn_id="turn_001",
            speaker=ConversationRole.CLIENT,
            content="I want to kill myself",  # Inappropriate for non-crisis
            confidence_score=0.8
        )
        
        is_valid, issues = await generator._validate_turn(turn, basic_conversation_parameters, [])
        
        assert not is_valid
        assert any("Inappropriate content" in issue for issue in issues)
    
    @pytest.mark.asyncio
    async def test_turn_validation_missing_rationale(self, generator, basic_conversation_parameters):
        """Test turn validation with missing clinical rationale"""
        turn = ConversationTurn(
            turn_id="turn_001",
            speaker=ConversationRole.THERAPIST,
            content="How are you feeling?",
            clinical_rationale=None,  # Missing rationale
            confidence_score=0.8
        )
        
        is_valid, issues = await generator._validate_turn(turn, basic_conversation_parameters, [])
        
        assert not is_valid
        assert "Missing clinical rationale" in issues
    
    @pytest.mark.asyncio
    async def test_turn_validation_low_confidence(self, generator, basic_conversation_parameters):
        """Test turn validation with low confidence score"""
        turn = ConversationTurn(
            turn_id="turn_001",
            speaker=ConversationRole.THERAPIST,
            content="How are you feeling?",
            clinical_rationale="Assessment question",
            confidence_score=0.3  # Low confidence
        )
        
        is_valid, issues = await generator._validate_turn(turn, basic_conversation_parameters, [])
        
        assert not is_valid
        assert "Low confidence score" in issues
    
    @pytest.mark.asyncio
    async def test_turn_validation_speaker_continuity_error(self, generator, basic_conversation_parameters):
        """Test turn validation with speaker continuity error"""
        previous_turn = ConversationTurn(
            turn_id="turn_000",
            speaker=ConversationRole.THERAPIST,
            content="Previous therapist turn",
            confidence_score=0.8
        )
        
        turn = ConversationTurn(
            turn_id="turn_001",
            speaker=ConversationRole.THERAPIST,  # Same speaker as previous
            content="Another therapist turn",
            clinical_rationale="Assessment",
            confidence_score=0.8
        )
        
        is_valid, issues = await generator._validate_turn(turn, basic_conversation_parameters, [previous_turn])
        
        assert not is_valid
        assert "Speaker continuity error" in issues
    
    @pytest.mark.asyncio
    async def test_regenerate_therapist_turn(self, generator, basic_conversation_parameters):
        """Test therapist turn regeneration"""
        original_turn = ConversationTurn(
            turn_id="turn_001",
            speaker=ConversationRole.THERAPIST,
            content="Invalid content",
            confidence_score=0.3
        )
        
        client_turn = ConversationTurn(
            turn_id="turn_000",
            speaker=ConversationRole.CLIENT,
            content="I'm feeling sad today",
            confidence_score=0.8
        )
        
        # Mock response generator
        from .therapist_response_generator import TherapistResponse
        mock_response = TherapistResponse(
            content="I hear that you're feeling sad. Can you tell me more about what's contributing to these feelings?",
            clinical_rationale="Empathetic reflection with open-ended exploration",
            therapeutic_technique="Reflective listening",
            intervention_type=InterventionType.EXPLORATION,
            confidence_score=0.9,
            contraindications=[],
            follow_up_suggestions=[]
        )
        
        generator.response_generator.generate_response = AsyncMock(return_value=mock_response)
        
        regenerated = await generator._regenerate_therapist_turn(
            original_turn, basic_conversation_parameters, [client_turn], ["Low confidence"]
        )
        
        assert regenerated.turn_id == original_turn.turn_id
        assert regenerated.speaker == ConversationRole.THERAPIST
        assert regenerated.confidence_score > original_turn.confidence_score
        assert regenerated.clinical_rationale is not None
    
    @pytest.mark.asyncio
    async def test_regenerate_client_turn(self, generator, basic_conversation_parameters):
        """Test client turn regeneration"""
        original_turn = ConversationTurn(
            turn_id="turn_001",
            speaker=ConversationRole.CLIENT,
            content="Inappropriate content",
            confidence_score=0.8
        )
        
        regenerated = await generator._regenerate_client_turn(
            original_turn, basic_conversation_parameters, [], ["Inappropriate content"]
        )
        
        assert regenerated.turn_id == original_turn.turn_id
        assert regenerated.speaker == ConversationRole.CLIENT
        assert len(regenerated.content) > 0
        assert "inappropriate" not in regenerated.content.lower()
    
    @pytest.mark.asyncio
    async def test_clinical_summary_generation(self, generator, basic_conversation_parameters):
        """Test clinical summary generation"""
        turns = [
            ConversationTurn(
                turn_id="turn_000",
                speaker=ConversationRole.THERAPIST,
                content="How are you feeling?",
                intervention_type=InterventionType.ASSESSMENT,
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="I'm feeling very depressed and anxious lately.",
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_002",
                speaker=ConversationRole.THERAPIST,
                content="Let's explore those feelings together.",
                intervention_type=InterventionType.EXPLORATION,
                confidence_score=0.9
            )
        ]
        
        summary = await generator._generate_clinical_summary(turns, basic_conversation_parameters)
        
        assert "CBT approach" in summary
        assert "Session 3" in summary
        assert "problem_exploration phase" in summary
        assert "Major Depressive Disorder" in summary
        assert "Total exchanges: 3" in summary
        assert "assessment" in summary.lower()
    
    @pytest.mark.asyncio
    async def test_quality_metrics_calculation(self, generator, basic_conversation_parameters):
        """Test quality metrics calculation"""
        turns = [
            ConversationTurn(
                turn_id="turn_000",
                speaker=ConversationRole.THERAPIST,
                content="How are your thoughts and feelings today?",
                clinical_rationale="CBT-focused assessment of cognitive and emotional state",
                intervention_type=InterventionType.ASSESSMENT,
                confidence_score=0.9
            ),
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="I've been having really negative thoughts about myself and feeling overwhelmed.",
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_002",
                speaker=ConversationRole.THERAPIST,
                content="Let's examine the evidence for and against those thoughts.",
                clinical_rationale="Cognitive restructuring technique to challenge negative thoughts",
                intervention_type=InterventionType.COGNITIVE_RESTRUCTURING,
                confidence_score=0.85
            )
        ]
        
        metrics = await generator._calculate_quality_metrics(turns, basic_conversation_parameters)
        
        assert 'clinical_accuracy' in metrics
        assert 'therapeutic_appropriateness' in metrics
        assert 'conversation_flow' in metrics
        assert 'client_engagement' in metrics
        assert 'modality_adherence' in metrics
        assert 'overall_quality' in metrics
        
        # Check reasonable values
        assert 0 <= metrics['clinical_accuracy'] <= 1
        assert 0 <= metrics['therapeutic_appropriateness'] <= 1
        assert 0 <= metrics['conversation_flow'] <= 1
        assert 0 <= metrics['overall_quality'] <= 1
        
        # Should have good modality adherence for CBT
        assert metrics['modality_adherence'] > 0.5
    
    @pytest.mark.asyncio
    async def test_therapeutic_progress_assessment(self, generator, basic_conversation_parameters):
        """Test therapeutic progress assessment"""
        turns = [
            ConversationTurn(
                turn_id="turn_000",
                speaker=ConversationRole.THERAPIST,
                content="Let's work on reducing your depressive symptoms.",
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="I understand that I need to challenge my negative thoughts.",
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_002",
                speaker=ConversationRole.THERAPIST,
                content="Let's practice this coping skill together.",
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_003",
                speaker=ConversationRole.CLIENT,
                content="I'll try to use this technique when I feel anxious.",
                confidence_score=0.8
            )
        ]
        
        progress = await generator._assess_therapeutic_progress(turns, basic_conversation_parameters)
        
        assert 'phase_progression' in progress
        assert 'goal_achievement' in progress
        assert 'insight_development' in progress
        assert 'skill_acquisition' in progress
        assert 'emotional_regulation_progress' in progress
        assert 'overall_progress_score' in progress
        
        # Should show some insight development
        assert progress['insight_development'] > 0
        
        # Should show skill acquisition
        assert progress['skill_acquisition'] > 0
        
        # Should have goal progress for specified goals
        assert 'Reduce depressive symptoms' in progress['goal_achievement']
    
    @pytest.mark.asyncio
    async def test_recommendations_generation(self, generator, basic_conversation_parameters):
        """Test recommendations generation"""
        # Create turns with some quality issues
        turns = [
            ConversationTurn(
                turn_id="turn_000",
                speaker=ConversationRole.THERAPIST,
                content="Okay.",  # Short, low quality
                clinical_rationale="Brief response",
                confidence_score=0.5  # Low confidence
            ),
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="Yes.",  # Short response
                confidence_score=0.8
            )
        ]
        
        quality_metrics = {
            'clinical_accuracy': 0.6,  # Below threshold
            'therapeutic_appropriateness': 0.65,  # Below threshold
            'conversation_flow': 0.9,
            'client_engagement': 0.4,  # Low engagement
            'modality_adherence': 0.5,  # Low adherence
            'overall_quality': 0.6
        }
        
        recommendations = await generator._generate_recommendations(
            turns, basic_conversation_parameters, quality_metrics
        )
        
        assert len(recommendations) > 0
        assert any("clinical accuracy" in rec.lower() for rec in recommendations)
        assert any("therapeutic appropriateness" in rec.lower() for rec in recommendations)
        assert any("engagement" in rec.lower() for rec in recommendations)
        assert any("cbt" in rec.lower() for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_warnings_identification_safety(self, generator, basic_conversation_parameters):
        """Test warnings identification for safety concerns"""
        turns = [
            ConversationTurn(
                turn_id="turn_000",
                speaker=ConversationRole.THERAPIST,
                content="How are you feeling?",
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="I want to kill myself.",  # Safety concern
                confidence_score=0.8
            )
        ]
        
        warnings = await generator._identify_warnings(turns, basic_conversation_parameters)
        
        assert len(warnings) > 0
        assert any("Safety concern" in warning for warning in warnings)
        assert any("kill" in warning for warning in warnings)
    
    @pytest.mark.asyncio
    async def test_warnings_identification_boundary_issues(self, generator, basic_conversation_parameters):
        """Test warnings identification for boundary issues"""
        turns = [
            ConversationTurn(
                turn_id="turn_000",
                speaker=ConversationRole.THERAPIST,
                content="We could be friends outside of therapy.",  # Boundary issue
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="That sounds nice.",
                confidence_score=0.8
            )
        ]
        
        warnings = await generator._identify_warnings(turns, basic_conversation_parameters)
        
        assert len(warnings) > 0
        assert any("boundary issue" in warning.lower() for warning in warnings)
    
    @pytest.mark.asyncio
    async def test_warnings_identification_low_confidence(self, generator, basic_conversation_parameters):
        """Test warnings identification for low confidence responses"""
        turns = [
            ConversationTurn(
                turn_id="turn_000",
                speaker=ConversationRole.THERAPIST,
                content="I'm not sure what to say.",
                clinical_rationale="Uncertain response",
                confidence_score=0.3  # Low confidence
            ),
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.THERAPIST,
                content="Maybe we should try something.",
                clinical_rationale="Another uncertain response",
                confidence_score=0.4  # Low confidence
            )
        ]
        
        warnings = await generator._identify_warnings(turns, basic_conversation_parameters)
        
        assert len(warnings) > 0
        assert any("low-confidence" in warning.lower() for warning in warnings)
    
    @pytest.mark.asyncio
    async def test_warnings_identification_crisis_not_addressed(self, generator, crisis_conversation_parameters):
        """Test warnings identification when crisis indicators not addressed"""
        turns = [
            ConversationTurn(
                turn_id="turn_000",
                speaker=ConversationRole.THERAPIST,
                content="How was your week?",  # Not addressing crisis
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="Terrible, I want to die.",
                confidence_score=0.8
            )
        ]
        
        warnings = await generator._identify_warnings(turns, crisis_conversation_parameters)
        
        assert len(warnings) > 0
        assert any("crisis indicators" in warning.lower() and "not adequately addressed" in warning.lower() 
                  for warning in warnings)
    
    @pytest.mark.asyncio
    async def test_complexity_assessment_basic(self, generator):
        """Test complexity assessment for basic level"""
        turns = [
            ConversationTurn(
                turn_id="turn_000",
                speaker=ConversationRole.THERAPIST,
                content="How are you?",  # Simple language
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.THERAPIST,
                content="That's good.",  # Simple language
                confidence_score=0.8
            )
        ]
        
        complexity = generator._assess_complexity_achieved(turns)
        assert complexity == ConversationComplexity.BASIC.value
    
    @pytest.mark.asyncio
    async def test_complexity_assessment_expert(self, generator):
        """Test complexity assessment for expert level"""
        turns = [
            ConversationTurn(
                turn_id="turn_000",
                speaker=ConversationRole.THERAPIST,
                content="This therapeutic intervention addresses your cognitive distortions through systematic behavioral assessment, while considering psychodynamic factors and diagnostic criteria.",
                confidence_score=0.8
            )
        ]
        
        complexity = generator._assess_complexity_achieved(turns)
        assert complexity == ConversationComplexity.EXPERT.value
    
    @pytest.mark.asyncio
    async def test_modality_adherence_assessment_cbt(self, generator):
        """Test modality adherence assessment for CBT"""
        turns = [
            ConversationTurn(
                turn_id="turn_000",
                speaker=ConversationRole.THERAPIST,
                content="Let's examine your thoughts and behaviors.",
                intervention_type=InterventionType.COGNITIVE_RESTRUCTURING,
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.THERAPIST,
                content="What evidence supports this thought?",
                intervention_type=InterventionType.COGNITIVE_RESTRUCTURING,
                confidence_score=0.8
            )
        ]
        
        adherence = generator._assess_modality_adherence(turns, TherapeuticModality.CBT)
        assert adherence == 1.0  # Perfect CBT adherence
    
    @pytest.mark.asyncio
    async def test_modality_adherence_assessment_mixed(self, generator):
        """Test modality adherence assessment with mixed interventions"""
        turns = [
            ConversationTurn(
                turn_id="turn_000",
                speaker=ConversationRole.THERAPIST,
                content="Let's examine your thoughts.",
                intervention_type=InterventionType.COGNITIVE_RESTRUCTURING,  # CBT
                confidence_score=0.8
            ),
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.THERAPIST,
                content="I notice a pattern here.",
                intervention_type=InterventionType.INTERPRETATION,  # Psychodynamic
                confidence_score=0.8
            )
        ]
        
        adherence = generator._assess_modality_adherence(turns, TherapeuticModality.CBT)
        assert adherence == 0.5  # 50% CBT adherence
    
    @pytest.mark.asyncio
    async def test_export_conversation_json(self, generator, basic_conversation_parameters):
        """Test conversation export in JSON format"""
        # Create a simple conversation
        conversation = GeneratedConversation(
            conversation_id="test_conv_001",
            parameters=basic_conversation_parameters,
            turns=[
                ConversationTurn(
                    turn_id="turn_000",
                    speaker=ConversationRole.THERAPIST,
                    content="How are you feeling today?",
                    clinical_rationale="Assessment of current state",
                    intervention_type=InterventionType.ASSESSMENT,
                    confidence_score=0.8
                )
            ],
            clinical_summary="Test conversation summary",
            therapeutic_progress={'overall_progress_score': 0.7},
            quality_metrics={'overall_quality': 0.8},
            recommendations=["Continue CBT approach"],
            warnings=[]
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = await generator.export_conversation(
                conversation, format='json', output_path=Path(temp_dir) / "test_conv.json"
            )
            
            assert output_path.exists()
            assert output_path.suffix == '.json'
            
            # Verify content
            with open(output_path, 'r') as f:
                exported_data = json.load(f)
            
            assert exported_data['conversation_id'] == "test_conv_001"
            assert len(exported_data['turns']) == 1
            assert exported_data['turns'][0]['speaker'] == 'therapist'
            assert exported_data['clinical_summary'] == "Test conversation summary"
    
    @pytest.mark.asyncio
    async def test_export_conversation_txt(self, generator, basic_conversation_parameters):
        """Test conversation export in TXT format"""
        conversation = GeneratedConversation(
            conversation_id="test_conv_002",
            parameters=basic_conversation_parameters,
            turns=[
                ConversationTurn(
                    turn_id="turn_000",
                    speaker=ConversationRole.THERAPIST,
                    content="How are you feeling?",
                    clinical_rationale="Assessment question",
                    confidence_score=0.8
                ),
                ConversationTurn(
                    turn_id="turn_001",
                    speaker=ConversationRole.CLIENT,
                    content="I'm feeling sad.",
                    confidence_score=0.8
                )
            ],
            clinical_summary="Test summary",
            therapeutic_progress={},
            quality_metrics={'overall_quality': 0.8},
            recommendations=["Continue therapy"],
            warnings=["Monitor mood"]
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = await generator.export_conversation(
                conversation, format='txt', output_path=Path(temp_dir) / "test_conv.txt"
            )
            
            assert output_path.exists()
            assert output_path.suffix == '.txt'
            
            # Verify content
            with open(output_path, 'r') as f:
                content = f.read()
            
            assert "test_conv_002" in content
            assert "Therapist: How are you feeling?" in content
            assert "Client: I'm feeling sad." in content
            assert "Clinical Rationale: Assessment question" in content
            assert "Test summary" in content
            assert "Continue therapy" in content
            assert "Monitor mood" in content
    
    @pytest.mark.asyncio
    async def test_export_conversation_unsupported_format(self, generator, basic_conversation_parameters):
        """Test conversation export with unsupported format"""
        conversation = GeneratedConversation(
            conversation_id="test_conv_003",
            parameters=basic_conversation_parameters,
            turns=[],
            clinical_summary="",
            therapeutic_progress={},
            quality_metrics={},
            recommendations=[],
            warnings=[]
        )
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            await generator.export_conversation(conversation, format='unsupported')
    
    @pytest.mark.asyncio
    async def test_generation_statistics(self, generator):
        """Test generation statistics retrieval"""
        stats = generator.get_generation_statistics()
        
        assert 'supported_modalities' in stats
        assert 'supported_phases' in stats
        assert 'complexity_levels' in stats
        assert 'client_response_styles' in stats
        assert 'generation_settings' in stats
        assert 'configuration' in stats
        
        # Check that all expected modalities are present
        assert 'cognitive_behavioral_therapy' in stats['supported_modalities']
        assert 'dialectical_behavior_therapy' in stats['supported_modalities']
        assert 'psychodynamic' in stats['supported_modalities']
        
        # Check complexity levels
        assert 'basic' in stats['complexity_levels']
        assert 'expert' in stats['complexity_levels']


if __name__ == "__main__":
    pytest.main([__file__])
