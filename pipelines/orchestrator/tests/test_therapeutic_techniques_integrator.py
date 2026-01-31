"""
Tests for Therapeutic Techniques and Intervention Strategies Integration System

This module provides comprehensive tests for the therapeutic techniques integrator,
covering intervention protocol selection, conversation generation, and export functionality.
"""

import json
import os
import tempfile

import pytest

from .client_scenario_generator import (
    ClientScenario,
    ClinicalFormulation,
    ScenarioType,
    SeverityLevel,
)
from .therapeutic_response_generator import TherapeuticTechnique
from .therapeutic_techniques_integrator import (
    ConversationIntervention,
    EvidenceLevel,
    InterventionProtocol,
    InterventionSelection,
    InterventionStrategy,
    SpecializedFramework,
    TherapeuticTechniquesIntegrator,
)


class TestTherapeuticTechniquesIntegrator:
    """Test suite for TherapeuticTechniquesIntegrator."""

    @pytest.fixture
    def integrator(self):
        """Create integrator instance for testing."""
        return TherapeuticTechniquesIntegrator()

    @pytest.fixture
    def sample_client_scenario(self):
        """Create sample client scenario for testing."""
        clinical_formulation = ClinicalFormulation(
            dsm5_considerations=["Major Depressive Disorder", "Generalized Anxiety Disorder"],
            attachment_style="secure",
            defense_mechanisms=["intellectualization"],
            psychodynamic_themes=["abandonment fears"]
        )

        return ClientScenario(
            id="test_scenario_001",
            scenario_type=ScenarioType.THERAPEUTIC_SESSION,
            severity_level=SeverityLevel.MODERATE,
            presenting_problem="Depression and anxiety affecting work performance",
            clinical_formulation=clinical_formulation,
            demographics={"age": 32, "gender": "female"},
            session_context="Third therapy session"
        )

    @pytest.fixture
    def crisis_client_scenario(self):
        """Create crisis client scenario for testing."""
        clinical_formulation = ClinicalFormulation(
            dsm5_considerations=["Major Depressive Disorder with suicidal ideation"],
            attachment_style="anxious",
            defense_mechanisms=["denial"],
            psychodynamic_themes=["hopelessness"]
        )

        return ClientScenario(
            id="crisis_scenario_001",
            scenario_type=ScenarioType.CRISIS_INTERVENTION,
            severity_level=SeverityLevel.SEVERE,
            presenting_problem="Suicidal ideation with plan",
            clinical_formulation=clinical_formulation,
            demographics={"age": 28, "gender": "male"},
            session_context="Emergency session"
        )

    def test_initialization(self, integrator):
        """Test integrator initialization."""
        assert integrator is not None
        assert hasattr(integrator, "intervention_protocols")
        assert hasattr(integrator, "framework_mappings")
        assert hasattr(integrator, "evidence_base")

        # Check that protocols are loaded
        assert len(integrator.intervention_protocols) > 0
        assert InterventionStrategy.CRISIS_STABILIZATION in integrator.intervention_protocols
        assert InterventionStrategy.TRAUMA_PROCESSING in integrator.intervention_protocols
        assert InterventionStrategy.BEHAVIORAL_MODIFICATION in integrator.intervention_protocols

    def test_intervention_protocol_structure(self, integrator):
        """Test intervention protocol data structure."""
        crisis_protocols = integrator.intervention_protocols[InterventionStrategy.CRISIS_STABILIZATION]
        assert len(crisis_protocols) > 0

        protocol = crisis_protocols[0]
        assert hasattr(protocol, "id")
        assert hasattr(protocol, "name")
        assert hasattr(protocol, "strategy")
        assert hasattr(protocol, "evidence_level")
        assert hasattr(protocol, "target_conditions")
        assert hasattr(protocol, "contraindications")
        assert hasattr(protocol, "session_structure")
        assert hasattr(protocol, "key_techniques")
        assert hasattr(protocol, "expected_outcomes")
        assert hasattr(protocol, "duration_sessions")

        # Validate data types
        assert isinstance(protocol.target_conditions, list)
        assert isinstance(protocol.contraindications, list)
        assert isinstance(protocol.session_structure, list)
        assert isinstance(protocol.key_techniques, list)
        assert isinstance(protocol.expected_outcomes, list)
        assert isinstance(protocol.duration_sessions, tuple)
        assert len(protocol.duration_sessions) == 2

    def test_select_intervention_protocol_depression(self, integrator, sample_client_scenario):
        """Test intervention protocol selection for depression."""
        selection = integrator.select_intervention_protocol(sample_client_scenario)

        assert isinstance(selection, InterventionSelection)
        assert selection.protocol is not None
        assert selection.primary_rationale is not None
        assert isinstance(selection.secondary_considerations, list)
        assert isinstance(selection.adaptation_notes, list)
        assert isinstance(selection.monitoring_plan, list)
        assert selection.expected_timeline is not None
        assert isinstance(selection.success_metrics, list)

        # Should select behavioral modification for depression
        assert selection.protocol.strategy == InterventionStrategy.BEHAVIORAL_MODIFICATION

    def test_select_intervention_protocol_crisis(self, integrator, crisis_client_scenario):
        """Test intervention protocol selection for crisis."""
        selection = integrator.select_intervention_protocol(crisis_client_scenario)

        assert isinstance(selection, InterventionSelection)
        assert selection.protocol.strategy == InterventionStrategy.CRISIS_STABILIZATION

        # Crisis protocols should have safety planning
        assert any(
            technique == TherapeuticTechnique.SAFETY_PLANNING
            for technique in selection.protocol.key_techniques
        )

        # Should include crisis-specific monitoring
        assert any(
            "safety" in monitor.lower() or "crisis" in monitor.lower()
            for monitor in selection.monitoring_plan
        )

    def test_analyze_primary_concerns(self, integrator, sample_client_scenario):
        """Test primary concerns analysis."""
        concerns = integrator._analyze_primary_concerns(sample_client_scenario)

        assert isinstance(concerns, list)
        assert "depression" in concerns
        assert "anxiety" in concerns

        # Should not have duplicates
        assert len(concerns) == len(set(concerns))

    def test_identify_contraindications(self, integrator):
        """Test contraindications identification."""
        # Create scenario with psychosis
        clinical_formulation = ClinicalFormulation(
            dsm5_considerations=["Schizophrenia with active psychosis"],
            attachment_style="disorganized",
            defense_mechanisms=["projection"],
            psychodynamic_themes=["reality_testing_issues"]
        )

        scenario = ClientScenario(
            id="psychosis_scenario",
            scenario_type=ScenarioType.INITIAL_ASSESSMENT,
            severity_level=SeverityLevel.SEVERE,
            presenting_problem="Hearing voices and paranoid thoughts",
            clinical_formulation=clinical_formulation,
            demographics={"age": 25, "gender": "non-binary"},
            session_context="First session"
        )

        contraindications = integrator._identify_contraindications(scenario)
        assert "active psychosis" in contraindications

    def test_generate_intervention_conversation(self, integrator, sample_client_scenario):
        """Test intervention conversation generation."""
        selection = integrator.select_intervention_protocol(sample_client_scenario)
        conversation = integrator.generate_intervention_conversation(
            sample_client_scenario, selection, session_phase="working", num_exchanges=6
        )

        assert isinstance(conversation, ConversationIntervention)
        assert conversation.id is not None
        assert conversation.client_scenario == sample_client_scenario
        assert conversation.intervention_selection == selection
        assert len(conversation.conversation_exchanges) == 6
        assert conversation.session_phase == "working"
        assert isinstance(conversation.therapeutic_goals, list)
        assert 0 <= conversation.intervention_fidelity_score <= 1
        assert isinstance(conversation.clinical_notes, list)

        # Check exchange structure
        exchanges = conversation.conversation_exchanges
        assert len(exchanges) == 6

        # Should alternate between therapist and client
        for i, exchange in enumerate(exchanges):
            if i % 2 == 0:
                assert exchange["speaker"] == "therapist"
                assert "technique_used" in exchange
                assert "structure_element" in exchange
            else:
                assert exchange["speaker"] == "client"
                assert "engagement_level" in exchange
                assert "emotional_state" in exchange

    def test_conversation_phases(self, integrator, sample_client_scenario):
        """Test different conversation phases."""
        selection = integrator.select_intervention_protocol(sample_client_scenario)

        # Test opening phase
        opening_conv = integrator.generate_intervention_conversation(
            sample_client_scenario, selection, session_phase="opening", num_exchanges=4
        )
        assert opening_conv.session_phase == "opening"
        assert "rapport" in " ".join(opening_conv.therapeutic_goals).lower()

        # Test working phase
        working_conv = integrator.generate_intervention_conversation(
            sample_client_scenario, selection, session_phase="working", num_exchanges=6
        )
        assert working_conv.session_phase == "working"

        # Test closing phase
        closing_conv = integrator.generate_intervention_conversation(
            sample_client_scenario, selection, session_phase="closing", num_exchanges=4
        )
        assert closing_conv.session_phase == "closing"
        assert "summarize" in " ".join(closing_conv.therapeutic_goals).lower()
        assert closing_conv.homework_assigned is not None
        assert closing_conv.next_session_plan is not None

    def test_intervention_fidelity_calculation(self, integrator, sample_client_scenario):
        """Test intervention fidelity score calculation."""
        selection = integrator.select_intervention_protocol(sample_client_scenario)
        conversation = integrator.generate_intervention_conversation(
            sample_client_scenario, selection, num_exchanges=6
        )

        fidelity_score = conversation.intervention_fidelity_score
        assert 0 <= fidelity_score <= 1
        assert isinstance(fidelity_score, float)

    def test_framework_specific_selection(self, integrator, sample_client_scenario):
        """Test framework-specific protocol selection."""
        # Test with DBT preference
        selection_dbt = integrator.select_intervention_protocol(
            sample_client_scenario, preferred_framework=SpecializedFramework.DBT
        )

        # Should respect framework preference if available
        if selection_dbt.protocol.framework:
            assert selection_dbt.protocol.framework == SpecializedFramework.DBT

    def test_export_intervention_conversations(self, integrator, sample_client_scenario):
        """Test conversation export functionality."""
        selection = integrator.select_intervention_protocol(sample_client_scenario)
        conversations = []

        # Generate multiple conversations
        for _i in range(3):
            conv = integrator.generate_intervention_conversation(
                sample_client_scenario, selection, num_exchanges=4
            )
            conversations.append(conv)

        # Test export
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            export_result = integrator.export_intervention_conversations(conversations, temp_file)

            # Check export result
            assert export_result["exported_conversations"] == 3
            assert export_result["output_file"] == temp_file
            assert "protocols_covered" in export_result
            assert "average_fidelity" in export_result

            # Check exported file
            assert os.path.exists(temp_file)
            with open(temp_file, encoding="utf-8") as f:
                exported_data = json.load(f)

            assert "metadata" in exported_data
            assert "conversations" in exported_data
            assert exported_data["metadata"]["total_conversations"] == 3
            assert len(exported_data["conversations"]) == 3

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_evidence_level_prioritization(self, integrator):
        """Test that higher evidence level protocols are prioritized."""
        # Create mock protocols with different evidence levels
        protocol_strong = InterventionProtocol(
            id="strong_evidence",
            name="Strong Evidence Protocol",
            strategy=InterventionStrategy.COGNITIVE_RESTRUCTURING_PROTOCOL,
            framework=None,
            evidence_level=EvidenceLevel.STRONG,
            target_conditions=["depression"],
            contraindications=[],
            session_structure=["assessment"],
            key_techniques=[TherapeuticTechnique.COGNITIVE_RESTRUCTURING],
            expected_outcomes=["symptom reduction"],
            duration_sessions=(8, 12),
            homework_assignments=["thought record"],
            progress_indicators=["mood improvement"],
            risk_considerations=["monitor for deterioration"]
        )

        protocol_moderate = InterventionProtocol(
            id="moderate_evidence",
            name="Moderate Evidence Protocol",
            strategy=InterventionStrategy.COGNITIVE_RESTRUCTURING_PROTOCOL,
            framework=None,
            evidence_level=EvidenceLevel.MODERATE,
            target_conditions=["depression"],
            contraindications=[],
            session_structure=["assessment"],
            key_techniques=[TherapeuticTechnique.COGNITIVE_RESTRUCTURING],
            expected_outcomes=["symptom reduction"],
            duration_sessions=(8, 12),
            homework_assignments=["thought record"],
            progress_indicators=["mood improvement"],
            risk_considerations=["monitor for deterioration"]
        )

        # Mock scenario
        clinical_formulation = ClinicalFormulation(
            dsm5_considerations=["depression"],
            attachment_style="secure",
            defense_mechanisms=[],
            psychodynamic_themes=[]
        )

        scenario = ClientScenario(
            id="test",
            scenario_type=ScenarioType.THERAPEUTIC_SESSION,
            severity_level=SeverityLevel.MODERATE,
            presenting_problem="depression",
            clinical_formulation=clinical_formulation,
            demographics={},
            session_context="test"
        )

        # Test protocol selection
        selected = integrator._select_best_protocol([protocol_moderate, protocol_strong], scenario)
        assert selected.evidence_level == EvidenceLevel.STRONG


if __name__ == "__main__":
    pytest.main([__file__])
