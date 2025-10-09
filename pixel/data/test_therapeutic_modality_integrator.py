"""
Unit tests for Therapeutic Modality Integrator

Tests the integration of multiple therapeutic modalities including CBT, DBT,
Psychodynamic, Humanistic, and Systemic approaches with seamless switching,
technique blending, and effectiveness assessment.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from .therapeutic_modality_integrator import (
    TherapeuticModalityIntegrator, ModalityIntegrationPlan, ModalityProfile,
    ModalityIntegrationStrategy, ModalityTransitionTrigger, ModalityTransition,
    IntegratedResponse
)
from .therapeutic_conversation_schema import TherapeuticModality, ClinicalContext
from .therapist_response_generator import InterventionType, TherapistResponse
from .dynamic_conversation_generator import ConversationTurn, ConversationRole


@pytest.fixture
def sample_clinical_context():
    """Create sample clinical context"""
    return ClinicalContext(
        client_presentation="Client presents with depression, anxiety, and relationship difficulties",
        primary_diagnosis="Major Depressive Disorder",
        secondary_diagnoses=["Generalized Anxiety Disorder"],
        therapeutic_goals=["Reduce depressive symptoms", "Improve relationships", "Develop coping skills"],
        cultural_factors=["Hispanic/Latino background"],
        crisis_indicators=[],
        contraindications=[],
        session_number=5,
        conversation_history=[]
    )


@pytest.fixture
def crisis_clinical_context():
    """Create clinical context with crisis indicators"""
    return ClinicalContext(
        client_presentation="Client expressing suicidal ideation and severe emotional dysregulation",
        primary_diagnosis="Borderline Personality Disorder",
        therapeutic_goals=["Safety planning", "Emotional regulation", "Crisis management"],
        crisis_indicators=["suicidal ideation", "self-harm history"],
        session_number=2,
        conversation_history=[]
    )


@pytest.fixture
def sample_conversation_history():
    """Create sample conversation history"""
    return [
        ConversationTurn(
            turn_id="turn_001",
            speaker=ConversationRole.THERAPIST,
            content="How have you been feeling since our last session?",
            clinical_rationale="Assessment of current state",
            intervention_type=InterventionType.ASSESSMENT,
            confidence_score=0.8
        ),
        ConversationTurn(
            turn_id="turn_002",
            speaker=ConversationRole.CLIENT,
            content="I've been feeling really depressed and my thoughts keep spiraling.",
            confidence_score=0.8
        ),
        ConversationTurn(
            turn_id="turn_003",
            speaker=ConversationRole.THERAPIST,
            content="Let's examine those thoughts and see if we can find a more balanced perspective.",
            clinical_rationale="Cognitive restructuring intervention",
            intervention_type=InterventionType.COGNITIVE_RESTRUCTURING,
            confidence_score=0.9
        )
    ]


@pytest_asyncio.fixture
async def integrator():
    """Create therapeutic modality integrator instance"""
    integrator = TherapeuticModalityIntegrator()
    return integrator


class TestTherapeuticModalityIntegrator:
    """Test cases for TherapeuticModalityIntegrator"""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test integrator initialization"""
        integrator = TherapeuticModalityIntegrator()
        
        assert integrator.config is not None
        assert integrator.modality_profiles is not None
        assert integrator.integration_patterns is not None
        assert integrator.transition_rules is not None
        assert integrator.integration_settings is not None
        
        # Check that all expected modalities are loaded
        expected_modalities = [
            TherapeuticModality.CBT,
            TherapeuticModality.DBT,
            TherapeuticModality.PSYCHODYNAMIC,
            TherapeuticModality.HUMANISTIC,
            TherapeuticModality.SYSTEMIC
        ]
        
        for modality in expected_modalities:
            assert modality in integrator.modality_profiles
            profile = integrator.modality_profiles[modality]
            assert isinstance(profile, ModalityProfile)
            assert len(profile.core_techniques) > 0
            assert len(profile.primary_interventions) > 0
    
    @pytest.mark.asyncio
    async def test_modality_suitability_assessment(self, integrator, sample_clinical_context):
        """Test assessment of modality suitability"""
        treatment_goals = ["Reduce depression", "Improve coping skills"]
        
        suitability_scores = await integrator._assess_modality_suitability(
            sample_clinical_context, treatment_goals
        )
        
        assert isinstance(suitability_scores, dict)
        assert len(suitability_scores) == len(integrator.modality_profiles)
        
        # All scores should be between 0 and 1
        for modality, score in suitability_scores.items():
            assert 0.0 <= score <= 1.0
        
        # CBT should score well for depression
        assert suitability_scores[TherapeuticModality.CBT] > 0.3
    
    @pytest.mark.asyncio
    async def test_modality_suitability_with_crisis(self, integrator, crisis_clinical_context):
        """Test modality suitability assessment with crisis indicators"""
        treatment_goals = ["Safety planning", "Crisis management"]
        
        suitability_scores = await integrator._assess_modality_suitability(
            crisis_clinical_context, treatment_goals
        )
        
        # DBT should score highest for crisis situations
        assert suitability_scores[TherapeuticModality.DBT] > suitability_scores[TherapeuticModality.PSYCHODYNAMIC]
    
    @pytest.mark.asyncio
    async def test_primary_modality_selection(self, integrator):
        """Test primary modality selection"""
        suitability_scores = {
            TherapeuticModality.CBT: 0.8,
            TherapeuticModality.DBT: 0.6,
            TherapeuticModality.PSYCHODYNAMIC: 0.4,
            TherapeuticModality.HUMANISTIC: 0.5,
            TherapeuticModality.SYSTEMIC: 0.3
        }
        
        primary = integrator._select_primary_modality(suitability_scores)
        assert primary == TherapeuticModality.CBT
    
    @pytest.mark.asyncio
    async def test_primary_modality_selection_with_preferences(self, integrator):
        """Test primary modality selection with client preferences"""
        suitability_scores = {
            TherapeuticModality.CBT: 0.8,
            TherapeuticModality.DBT: 0.6,
            TherapeuticModality.PSYCHODYNAMIC: 0.4,
            TherapeuticModality.HUMANISTIC: 0.5,
            TherapeuticModality.SYSTEMIC: 0.3
        }
        
        client_preferences = ["humanistic", "person-centered"]
        
        primary = integrator._select_primary_modality(suitability_scores, client_preferences)
        # Humanistic should get a boost from preferences
        assert primary in [TherapeuticModality.CBT, TherapeuticModality.HUMANISTIC]
    
    @pytest.mark.asyncio
    async def test_secondary_modality_selection(self, integrator):
        """Test secondary modality selection"""
        primary_modality = TherapeuticModality.CBT
        suitability_scores = {
            TherapeuticModality.CBT: 0.8,
            TherapeuticModality.DBT: 0.7,
            TherapeuticModality.PSYCHODYNAMIC: 0.4,
            TherapeuticModality.HUMANISTIC: 0.6,
            TherapeuticModality.SYSTEMIC: 0.3
        }
        treatment_goals = ["Reduce anxiety", "Improve relationships"]
        
        secondary = integrator._select_secondary_modalities(
            primary_modality, suitability_scores, treatment_goals
        )
        
        assert isinstance(secondary, list)
        assert len(secondary) <= 2
        assert primary_modality not in secondary
        
        # Should include complementary modalities
        for modality in secondary:
            assert integrator._modalities_complement(primary_modality, modality)
    
    @pytest.mark.asyncio
    async def test_modality_complementarity(self, integrator):
        """Test modality complementarity assessment"""
        # Test known complementary pairs
        assert integrator._modalities_complement(TherapeuticModality.CBT, TherapeuticModality.DBT)
        assert integrator._modalities_complement(TherapeuticModality.CBT, TherapeuticModality.HUMANISTIC)
        assert integrator._modalities_complement(TherapeuticModality.PSYCHODYNAMIC, TherapeuticModality.HUMANISTIC)
        
        # Test non-complementary pairs (should still work but not be optimal)
        result = integrator._modalities_complement(TherapeuticModality.SYSTEMIC, TherapeuticModality.DBT)
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_integration_strategy_determination(self, integrator, sample_clinical_context):
        """Test integration strategy determination"""
        # Single secondary modality should favor hierarchical
        strategy = integrator._determine_integration_strategy(
            TherapeuticModality.CBT, [TherapeuticModality.DBT], sample_clinical_context
        )
        assert strategy == ModalityIntegrationStrategy.HIERARCHICAL
        
        # Multiple secondary modalities should favor blended
        strategy = integrator._determine_integration_strategy(
            TherapeuticModality.CBT, 
            [TherapeuticModality.DBT, TherapeuticModality.HUMANISTIC], 
            sample_clinical_context
        )
        assert strategy == ModalityIntegrationStrategy.BLENDED
        
        # Crisis context should favor adaptive
        crisis_context = ClinicalContext(
            client_presentation="Crisis situation",
            crisis_indicators=["suicidal ideation"]
        )
        strategy = integrator._determine_integration_strategy(
            TherapeuticModality.DBT, [TherapeuticModality.CBT], crisis_context
        )
        assert strategy == ModalityIntegrationStrategy.ADAPTIVE
    
    @pytest.mark.asyncio
    async def test_blending_ratio_calculation(self, integrator):
        """Test blending ratio calculation"""
        primary = TherapeuticModality.CBT
        secondary = [TherapeuticModality.DBT]
        
        # Test hierarchical strategy
        ratios = integrator._calculate_blending_ratios(
            primary, secondary, ModalityIntegrationStrategy.HIERARCHICAL
        )
        
        assert isinstance(ratios, dict)
        assert primary in ratios
        assert secondary[0] in ratios
        assert ratios[primary] > ratios[secondary[0]]  # Primary should dominate
        assert abs(sum(ratios.values()) - 1.0) < 0.01  # Should sum to ~1.0
        
        # Test blended strategy
        ratios = integrator._calculate_blending_ratios(
            primary, secondary, ModalityIntegrationStrategy.BLENDED
        )
        
        assert ratios[primary] == 0.5
        assert ratios[secondary[0]] == 0.5
    
    @pytest.mark.asyncio
    async def test_transition_trigger_identification(self, integrator, sample_clinical_context):
        """Test identification of transition triggers"""
        treatment_goals = ["Reduce depression", "Improve relationships"]
        
        triggers = integrator._identify_transition_triggers(sample_clinical_context, treatment_goals)
        
        assert isinstance(triggers, list)
        assert len(triggers) > 0
        
        # Should always include basic triggers
        assert ModalityTransitionTrigger.CLIENT_RESISTANCE in triggers
        assert ModalityTransitionTrigger.LACK_OF_PROGRESS in triggers
        
        # Should include goal achievement for goal-oriented treatment
        assert ModalityTransitionTrigger.GOAL_ACHIEVEMENT in triggers
    
    @pytest.mark.asyncio
    async def test_transition_trigger_identification_with_crisis(self, integrator, crisis_clinical_context):
        """Test transition trigger identification with crisis indicators"""
        treatment_goals = ["Safety planning"]
        
        triggers = integrator._identify_transition_triggers(crisis_clinical_context, treatment_goals)
        
        # Should include crisis emergence trigger
        assert ModalityTransitionTrigger.CRISIS_EMERGENCE in triggers
    
    @pytest.mark.asyncio
    async def test_technique_prioritization(self, integrator):
        """Test technique prioritization"""
        primary = TherapeuticModality.CBT
        secondary = [TherapeuticModality.DBT]
        treatment_goals = ["Reduce anxiety", "Improve emotion regulation"]
        
        priorities = integrator._prioritize_techniques(primary, secondary, treatment_goals)
        
        assert isinstance(priorities, dict)
        assert len(priorities) > 0
        
        # All priorities should be between 0 and 1
        for technique, priority in priorities.items():
            assert 0.0 <= priority <= 1.0
        
        # Anxiety-related techniques should have higher priority
        if 'Cognitive restructuring' in priorities:
            assert priorities['Cognitive restructuring'] > 0.5
        
        if 'Emotion regulation' in priorities:
            assert priorities['Emotion regulation'] > 0.5
    
    @pytest.mark.asyncio
    async def test_contraindication_checking(self, integrator):
        """Test contraindication checking"""
        modalities = [TherapeuticModality.CBT, TherapeuticModality.PSYCHODYNAMIC]
        
        # Context with contraindications
        context_with_contraindications = ClinicalContext(
            client_presentation="Client with active psychosis and severe cognitive impairment"
        )
        
        contraindications = integrator._check_contraindications(
            modalities, context_with_contraindications
        )
        
        assert isinstance(contraindications, list)
        # Should find contraindications for psychosis
        assert len(contraindications) > 0
        
        # Context without contraindications
        clean_context = ClinicalContext(
            client_presentation="Client with mild depression and good cognitive functioning"
        )
        
        contraindications = integrator._check_contraindications(modalities, clean_context)
        assert len(contraindications) == 0
    
    @pytest.mark.asyncio
    async def test_effectiveness_metrics_definition(self, integrator):
        """Test effectiveness metrics definition"""
        primary = TherapeuticModality.CBT
        secondary = [TherapeuticModality.DBT]
        treatment_goals = ["Reduce symptoms", "Improve functioning"]
        
        metrics = integrator._define_effectiveness_metrics(primary, secondary, treatment_goals)
        
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        
        # Should include modality-specific metrics
        cbt_profile = integrator.modality_profiles[TherapeuticModality.CBT]
        dbt_profile = integrator.modality_profiles[TherapeuticModality.DBT]
        
        # Some CBT or DBT effectiveness domains should be included
        modality_domains = cbt_profile.effectiveness_domains + dbt_profile.effectiveness_domains
        assert any(domain in metrics for domain in modality_domains)
        
        # Should include goal-specific metrics
        assert any('symptom' in metric.lower() for metric in metrics)
        assert any('function' in metric.lower() for metric in metrics)
    
    @pytest.mark.asyncio
    async def test_create_integration_plan(self, integrator, sample_clinical_context):
        """Test creation of comprehensive integration plan"""
        treatment_goals = ["Reduce depression", "Improve coping skills", "Better relationships"]
        client_preferences = ["cognitive behavioral"]
        
        plan = await integrator.create_integration_plan(
            sample_clinical_context, treatment_goals, client_preferences
        )
        
        assert isinstance(plan, ModalityIntegrationPlan)
        assert plan.primary_modality is not None
        assert isinstance(plan.secondary_modalities, list)
        assert isinstance(plan.integration_strategy, ModalityIntegrationStrategy)
        assert isinstance(plan.transition_triggers, list)
        assert isinstance(plan.blending_ratio, dict)
        assert isinstance(plan.session_allocation, dict)
        assert isinstance(plan.technique_priorities, dict)
        assert isinstance(plan.contraindication_checks, list)
        assert isinstance(plan.effectiveness_metrics, list)
        
        # Plan should be stored in integrator
        assert integrator.current_integration_plan == plan
        
        # Primary modality should be CBT due to client preference
        assert plan.primary_modality == TherapeuticModality.CBT
    
    @pytest.mark.asyncio
    async def test_create_integration_plan_crisis(self, integrator, crisis_clinical_context):
        """Test integration plan creation for crisis situation"""
        treatment_goals = ["Safety planning", "Crisis management", "Emotional regulation"]
        
        plan = await integrator.create_integration_plan(crisis_clinical_context, treatment_goals)
        
        # Should prioritize DBT for crisis situations
        assert plan.primary_modality == TherapeuticModality.DBT or TherapeuticModality.DBT in plan.secondary_modalities
        
        # Should include crisis emergence trigger
        assert ModalityTransitionTrigger.CRISIS_EMERGENCE in plan.transition_triggers
        
        # Should use adaptive strategy for crisis flexibility
        assert plan.integration_strategy == ModalityIntegrationStrategy.ADAPTIVE
    
    @pytest.mark.asyncio
    async def test_current_context_assessment(self, integrator, sample_conversation_history):
        """Test assessment of current conversational context"""
        client_statement = "I'm feeling overwhelmed and nothing seems to help anymore."
        clinical_context = ClinicalContext(client_presentation="Depression and anxiety")
        
        assessment = await integrator._assess_current_context(
            client_statement, sample_conversation_history, clinical_context
        )
        
        assert isinstance(assessment, dict)
        
        # Check required assessment components
        required_keys = [
            'emotional_intensity', 'crisis_indicators', 'resistance_level',
            'engagement_level', 'therapeutic_alliance', 'progress_indicators',
            'modality_preferences'
        ]
        
        for key in required_keys:
            assert key in assessment
        
        # Should detect high emotional intensity from "overwhelmed"
        assert assessment['emotional_intensity'] > 0.0
        
        # Should detect resistance from "nothing seems to help"
        assert assessment['resistance_level'] > 0.0
    
    @pytest.mark.asyncio
    async def test_current_context_assessment_crisis(self, integrator):
        """Test context assessment with crisis indicators"""
        client_statement = "I can't take this anymore. I just want to end it all."
        
        assessment = await integrator._assess_current_context(
            client_statement, [], ClinicalContext(client_presentation="Crisis")
        )
        
        # Should detect crisis indicators
        assert len(assessment['crisis_indicators']) > 0
        assert 'end it all' in assessment['crisis_indicators'] or 'can\'t take it' in assessment['crisis_indicators']
        
        # Should show high emotional intensity
        assert assessment['emotional_intensity'] > 0.5
    
    @pytest.mark.asyncio
    async def test_active_modality_determination(self, integrator, sample_clinical_context):
        """Test determination of active modalities"""
        # Create integration plan
        plan = ModalityIntegrationPlan(
            primary_modality=TherapeuticModality.CBT,
            secondary_modalities=[TherapeuticModality.DBT, TherapeuticModality.HUMANISTIC],
            integration_strategy=ModalityIntegrationStrategy.ADAPTIVE,
            transition_triggers=[],
            blending_ratio={},
            session_allocation={},
            technique_priorities={},
            contraindication_checks=[],
            effectiveness_metrics=[]
        )
        
        # Test with crisis context
        crisis_assessment = {
            'emotional_intensity': 0.8,
            'crisis_indicators': ['suicide'],
            'resistance_level': 0.2,
            'engagement_level': 0.6,
            'therapeutic_alliance': 0.7,
            'progress_indicators': [],
            'modality_preferences': []
        }
        
        active_modalities = integrator._determine_active_modalities(plan, crisis_assessment)
        
        assert TherapeuticModality.CBT in active_modalities  # Primary always included
        assert TherapeuticModality.DBT in active_modalities  # Should activate for crisis
        
        # Test with resistance context
        resistance_assessment = {
            'emotional_intensity': 0.4,
            'crisis_indicators': [],
            'resistance_level': 0.8,
            'engagement_level': 0.3,
            'therapeutic_alliance': 0.2,
            'progress_indicators': [],
            'modality_preferences': []
        }
        
        active_modalities = integrator._determine_active_modalities(plan, resistance_assessment)
        
        assert TherapeuticModality.CBT in active_modalities
        assert TherapeuticModality.HUMANISTIC in active_modalities  # Should activate for resistance
    
    @pytest.mark.asyncio
    async def test_cbt_response_generation(self, integrator):
        """Test CBT-specific response generation"""
        client_statement = "I keep thinking that I'm a failure."
        
        # Test cognitive restructuring
        response = integrator._generate_cbt_response(client_statement, "Cognitive restructuring")
        assert "evidence" in response.lower()
        assert "thought" in response.lower()
        
        # Test behavioral activation
        response = integrator._generate_cbt_response(client_statement, "Behavioral activation")
        assert "activity" in response.lower()
        
        # Test exposure
        response = integrator._generate_cbt_response(client_statement, "Exposure therapy")
        assert "step" in response.lower() or "gradual" in response.lower()
    
    @pytest.mark.asyncio
    async def test_dbt_response_generation(self, integrator):
        """Test DBT-specific response generation"""
        client_statement = "I'm in so much emotional pain right now."
        
        # Test mindfulness
        response = integrator._generate_dbt_response(client_statement, "Mindfulness skills")
        assert "moment" in response.lower()
        assert "experiencing" in response.lower()
        
        # Test distress tolerance
        response = integrator._generate_dbt_response(client_statement, "Distress tolerance")
        assert "skill" in response.lower()
        assert "distress" in response.lower()
        
        # Test emotion regulation
        response = integrator._generate_dbt_response(client_statement, "Emotion regulation")
        assert "emotion" in response.lower()
        assert "feeling" in response.lower()
    
    @pytest.mark.asyncio
    async def test_psychodynamic_response_generation(self, integrator):
        """Test psychodynamic-specific response generation"""
        client_statement = "This always happens to me in relationships."
        
        # Test interpretation
        response = integrator._generate_psychodynamic_response(client_statement, "Interpretation")
        assert "pattern" in response.lower()
        
        # Test transference exploration
        response = integrator._generate_psychodynamic_response(client_statement, "Transference interpretation")
        assert "relationship" in response.lower()
        assert "remind" in response.lower()
        
        # Test defense analysis
        response = integrator._generate_psychodynamic_response(client_statement, "Defense mechanism analysis")
        assert "feeling" in response.lower()
        assert "underneath" in response.lower()
    
    @pytest.mark.asyncio
    async def test_humanistic_response_generation(self, integrator):
        """Test humanistic-specific response generation"""
        client_statement = "I don't know who I am anymore."
        
        # Test empathetic reflection
        response = integrator._generate_humanistic_response(client_statement, "Empathetic reflection")
        assert "sound" in response.lower() or "feel" in response.lower()
        
        # Test unconditional positive regard
        response = integrator._generate_humanistic_response(client_statement, "Unconditional positive regard")
        assert "worthy" in response.lower() or "strength" in response.lower()
        
        # Test genuineness
        response = integrator._generate_humanistic_response(client_statement, "Genuineness")
        assert "moved" in response.lower() or "honored" in response.lower()
    
    @pytest.mark.asyncio
    async def test_response_blending(self, integrator):
        """Test blending of primary and secondary responses"""
        # Create primary response
        primary_response = TherapistResponse(
            content="Let's examine the evidence for that thought.",
            clinical_rationale="CBT cognitive restructuring",
            therapeutic_technique="Cognitive restructuring",
            intervention_type=InterventionType.COGNITIVE_RESTRUCTURING,
            confidence_score=0.8,
            contraindications=[],
            follow_up_suggestions=[]
        )
        
        # Create secondary techniques
        secondary_techniques = [
            (TherapeuticModality.DBT, "Mindfulness skills"),
            (TherapeuticModality.HUMANISTIC, "Validation")
        ]
        
        # Create integration plan
        plan = ModalityIntegrationPlan(
            primary_modality=TherapeuticModality.CBT,
            secondary_modalities=[TherapeuticModality.DBT, TherapeuticModality.HUMANISTIC],
            integration_strategy=ModalityIntegrationStrategy.BLENDED,
            transition_triggers=[],
            blending_ratio={
                TherapeuticModality.CBT: 0.5,
                TherapeuticModality.DBT: 0.3,
                TherapeuticModality.HUMANISTIC: 0.2
            },
            session_allocation={},
            technique_priorities={},
            contraindication_checks=[],
            effectiveness_metrics=[]
        )
        
        blended_response = await integrator._blend_responses(
            primary_response, secondary_techniques, plan
        )
        
        assert isinstance(blended_response, TherapistResponse)
        assert len(blended_response.content) > len(primary_response.content)
        assert "examine the evidence" in blended_response.content  # Original content preserved
        assert "mindful" in blended_response.content or "difficult" in blended_response.content  # Secondary content added
    
    @pytest.mark.asyncio
    async def test_modality_contribution_calculation(self, integrator):
        """Test calculation of modality contributions"""
        active_modalities = [TherapeuticModality.CBT, TherapeuticModality.DBT]
        blending_ratio = {
            TherapeuticModality.CBT: 0.7,
            TherapeuticModality.DBT: 0.3
        }
        
        contributions = integrator._calculate_modality_contributions(active_modalities, blending_ratio)
        
        assert isinstance(contributions, dict)
        assert TherapeuticModality.CBT in contributions
        assert TherapeuticModality.DBT in contributions
        
        # Should sum to 1.0
        assert abs(sum(contributions.values()) - 1.0) < 0.01
        
        # CBT should have higher contribution
        assert contributions[TherapeuticModality.CBT] > contributions[TherapeuticModality.DBT]
    
    @pytest.mark.asyncio
    async def test_cross_modal_coherence_assessment(self, integrator):
        """Test assessment of cross-modal coherence"""
        # Test single modality (should be perfect coherence)
        single_modality_response = TherapistResponse(
            content="Let's examine your thoughts about this situation.",
            clinical_rationale="CBT approach",
            therapeutic_technique="Cognitive restructuring",
            intervention_type=InterventionType.COGNITIVE_RESTRUCTURING,
            confidence_score=0.8,
            contraindications=[],
            follow_up_suggestions=[]
        )
        
        coherence = integrator._assess_cross_modal_coherence(
            single_modality_response, [TherapeuticModality.CBT]
        )
        assert coherence == 1.0
        
        # Test well-integrated response
        integrated_response = TherapistResponse(
            content="Let's mindfully examine the evidence for your thoughts and feelings.",
            clinical_rationale="CBT with DBT integration",
            therapeutic_technique="Integrated approach",
            intervention_type=InterventionType.COGNITIVE_RESTRUCTURING,
            confidence_score=0.8,
            contraindications=[],
            follow_up_suggestions=[]
        )
        
        coherence = integrator._assess_cross_modal_coherence(
            integrated_response, [TherapeuticModality.CBT, TherapeuticModality.DBT]
        )
        assert coherence > 0.8  # Should be high for good integration
    
    @pytest.mark.asyncio
    async def test_effectiveness_prediction(self, integrator):
        """Test prediction of response effectiveness"""
        response = TherapistResponse(
            content="Let's work together on this.",
            clinical_rationale="Collaborative approach",
            therapeutic_technique="Integration",
            intervention_type=InterventionType.EXPLORATION,
            confidence_score=0.9,
            contraindications=[],
            follow_up_suggestions=[]
        )
        
        plan = ModalityIntegrationPlan(
            primary_modality=TherapeuticModality.CBT,
            secondary_modalities=[TherapeuticModality.DBT],
            integration_strategy=ModalityIntegrationStrategy.ADAPTIVE,
            transition_triggers=[],
            blending_ratio={},
            session_allocation={},
            technique_priorities={},
            contraindication_checks=[],
            effectiveness_metrics=[]
        )
        
        # Test with crisis context and appropriate modality
        crisis_context = {
            'emotional_intensity': 0.8,
            'crisis_indicators': ['suicide'],
            'resistance_level': 0.2,
            'engagement_level': 0.7,
            'therapeutic_alliance': 0.8,
            'progress_indicators': [],
            'modality_preferences': []
        }
        
        effectiveness = integrator._predict_effectiveness(response, plan, crisis_context)
        
        assert 0.0 <= effectiveness <= 1.0
        assert effectiveness > 0.5  # Should be reasonably effective
    
    @pytest.mark.asyncio
    async def test_integration_rationale_generation(self, integrator):
        """Test generation of integration rationale"""
        active_modalities = [TherapeuticModality.CBT, TherapeuticModality.DBT]
        context_assessment = {
            'emotional_intensity': 0.8,
            'crisis_indicators': ['overwhelmed'],
            'resistance_level': 0.3,
            'engagement_level': 0.7,
            'therapeutic_alliance': 0.8,
            'progress_indicators': [],
            'modality_preferences': []
        }
        
        rationale = integrator._generate_integration_rationale(active_modalities, context_assessment)
        
        assert isinstance(rationale, str)
        assert len(rationale) > 0
        assert "CBT" in rationale or "cognitive_behavioral_therapy" in rationale
        assert "DBT" in rationale or "dialectical_behavior_therapy" in rationale
    
    @pytest.mark.asyncio
    async def test_blended_intervention_identification(self, integrator):
        """Test identification of blended interventions"""
        response = TherapistResponse(
            content="Let's validate your feelings while also exploring the evidence for your thoughts.",
            clinical_rationale="Integrated approach",
            therapeutic_technique="Blended",
            intervention_type=InterventionType.COGNITIVE_RESTRUCTURING,
            confidence_score=0.8,
            contraindications=[],
            follow_up_suggestions=[]
        )
        
        active_modalities = [TherapeuticModality.CBT, TherapeuticModality.HUMANISTIC]
        
        interventions = integrator._identify_blended_interventions(response, active_modalities)
        
        assert isinstance(interventions, list)
        assert InterventionType.COGNITIVE_RESTRUCTURING in interventions
        assert InterventionType.VALIDATION in interventions  # Should detect validation
        assert InterventionType.EXPLORATION in interventions  # Should detect exploration
    
    @pytest.mark.asyncio
    async def test_content_modality_identification(self, integrator):
        """Test identification of modalities in content"""
        # CBT content
        cbt_content = "Let's examine the evidence for your thoughts and change your behavior."
        modalities = integrator._identify_content_modalities(cbt_content)
        assert TherapeuticModality.CBT in modalities
        
        # DBT content
        dbt_content = "Let's practice mindfulness skills and emotion regulation techniques."
        modalities = integrator._identify_content_modalities(dbt_content)
        assert TherapeuticModality.DBT in modalities
        
        # Psychodynamic content
        psychodynamic_content = "I notice a pattern in your relationships and unconscious motivations."
        modalities = integrator._identify_content_modalities(psychodynamic_content)
        assert TherapeuticModality.PSYCHODYNAMIC in modalities
        
        # Humanistic content
        humanistic_content = "I want to understand your authentic experience and feelings."
        modalities = integrator._identify_content_modalities(humanistic_content)
        assert TherapeuticModality.HUMANISTIC in modalities
        
        # Mixed content
        mixed_content = "Let's mindfully examine your thoughts and validate your feelings."
        modalities = integrator._identify_content_modalities(mixed_content)
        assert len(modalities) > 1
        assert TherapeuticModality.DBT in modalities  # mindfully
        assert TherapeuticModality.CBT in modalities  # thoughts
    
    @pytest.mark.asyncio
    async def test_transition_trigger_detection(self, integrator):
        """Test detection of transition triggers"""
        # Test resistance detection
        client_turns = [
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="I don't want to do that homework. It won't work for me.",
                confidence_score=0.8
            )
        ]
        
        is_resistance = await integrator._detect_transition_trigger(
            ModalityTransitionTrigger.CLIENT_RESISTANCE, client_turns, ClinicalContext()
        )
        assert is_resistance
        
        # Test crisis detection
        crisis_turns = [
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="I can't take this anymore. I want to die.",
                confidence_score=0.8
            )
        ]
        
        is_crisis = await integrator._detect_transition_trigger(
            ModalityTransitionTrigger.CRISIS_EMERGENCE, crisis_turns, ClinicalContext()
        )
        assert is_crisis
        
        # Test progress detection
        progress_turns = [
            ConversationTurn(
                turn_id="turn_001",
                speaker=ConversationRole.CLIENT,
                content="I'm feeling much better and the techniques are really helping.",
                confidence_score=0.8
            )
        ]
        
        is_progress = await integrator._detect_transition_trigger(
            ModalityTransitionTrigger.GOAL_ACHIEVEMENT, progress_turns, ClinicalContext()
        )
        assert is_progress
    
    @pytest.mark.asyncio
    async def test_integration_statistics(self, integrator):
        """Test retrieval of integration statistics"""
        stats = integrator.get_integration_statistics()
        
        assert isinstance(stats, dict)
        
        # Check required statistics
        required_keys = [
            'supported_modalities', 'integration_strategies', 'transition_triggers',
            'integration_patterns', 'current_plan', 'integration_history', 'configuration'
        ]
        
        for key in required_keys:
            assert key in stats
        
        # Check supported modalities
        assert len(stats['supported_modalities']) >= 5
        assert 'cognitive_behavioral_therapy' in stats['supported_modalities']
        assert 'dialectical_behavior_therapy' in stats['supported_modalities']
        
        # Check integration strategies
        assert len(stats['integration_strategies']) > 0
        assert 'adaptive' in stats['integration_strategies']
        assert 'blended' in stats['integration_strategies']
        
        # Check transition triggers
        assert len(stats['transition_triggers']) > 0
        assert 'client_resistance' in stats['transition_triggers']
        assert 'crisis_emergence' in stats['transition_triggers']


if __name__ == "__main__":
    pytest.main([__file__])
