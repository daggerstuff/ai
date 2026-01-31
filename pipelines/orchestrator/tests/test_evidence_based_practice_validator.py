"""
Tests for Evidence-Based Practice Validation System

This module provides comprehensive tests for the evidence-based practice validator,
covering research validation, practice guideline adherence, outcome measurement,
and treatment fidelity assessment.
"""

import json
import os
import tempfile
from unittest.mock import Mock

import pytest

from .client_scenario_generator import (
    ClientScenario,
    ClinicalFormulation,
    ScenarioType,
    SeverityLevel,
)
from .evidence_based_practice_validator import (
    EvidenceBasedConversation,
    EvidenceBasedPracticeValidator,
    EvidenceLevel,
    OutcomeMeasureType,
    PracticeGuidelineSource,
    TreatmentFidelityDomain,
    ValidationResult,
)


class TestEvidenceBasedPracticeValidator:
    """Test suite for EvidenceBasedPracticeValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance for testing."""
        return EvidenceBasedPracticeValidator()

    @pytest.fixture
    def depression_scenario(self):
        """Create depression client scenario for testing."""
        clinical_formulation = ClinicalFormulation(
            dsm5_considerations=["Major Depressive Disorder"],
            attachment_style="anxious",
            defense_mechanisms=["intellectualization"],
            psychodynamic_themes=["loss_and_grief"]
        )

        return ClientScenario(
            id="depression_client_001",
            scenario_type=ScenarioType.THERAPEUTIC_SESSION,
            severity_level=SeverityLevel.MODERATE,
            presenting_problem="Major depression with anxiety",
            clinical_formulation=clinical_formulation,
            demographics={"age": 32, "gender": "female"},
            session_context="CBT therapy session"
        )

    @pytest.fixture
    def cbt_conversation(self):
        """Create CBT conversation data for testing."""
        return {
            "id": "cbt_conversation_001",
            "conversation_exchanges": [
                {"speaker": "therapist", "content": "Let's explore your thoughts about this situation", "technique": "cognitive_restructuring"},
                {"speaker": "client", "content": "I always think the worst will happen", "engagement_level": "engaged"},
                {"speaker": "therapist", "content": "What evidence do you have for that thought?", "technique": "socratic_questioning"},
                {"speaker": "client", "content": "I guess I don't have much evidence", "engagement_level": "highly_engaged"},
                {"speaker": "therapist", "content": "Let's assess your progress using this questionnaire", "technique": "outcome_monitoring"},
                {"speaker": "client", "content": "I feel like I'm making progress", "engagement_level": "highly_engaged"}
            ]
        }

    def test_initialization(self, validator):
        """Test validator initialization."""
        assert validator is not None
        assert hasattr(validator, "research_database")
        assert hasattr(validator, "practice_guidelines")
        assert hasattr(validator, "outcome_measures")
        assert hasattr(validator, "fidelity_protocols")

        # Check that databases are loaded
        assert len(validator.research_database) > 0
        assert len(validator.practice_guidelines) > 0
        assert len(validator.outcome_measures) > 0
        assert len(validator.fidelity_protocols) > 0

    def test_research_database_structure(self, validator):
        """Test research database structure."""
        database = validator.research_database

        assert "cognitive_behavioral_therapy" in database
        assert "therapeutic_alliance" in database
        assert "trauma_informed_care" in database

        # Check CBT evidence structure
        cbt_evidence = database["cognitive_behavioral_therapy"]
        assert len(cbt_evidence) > 0

        evidence = cbt_evidence[0]
        assert hasattr(evidence, "id")
        assert hasattr(evidence, "title")
        assert hasattr(evidence, "authors")
        assert hasattr(evidence, "journal")
        assert hasattr(evidence, "year")
        assert hasattr(evidence, "evidence_level")
        assert hasattr(evidence, "study_design")
        assert hasattr(evidence, "sample_size")
        assert hasattr(evidence, "population")
        assert hasattr(evidence, "intervention")
        assert hasattr(evidence, "outcomes")
        assert hasattr(evidence, "effect_sizes")
        assert hasattr(evidence, "confidence_intervals")
        assert hasattr(evidence, "limitations")
        assert hasattr(evidence, "clinical_significance")
        assert hasattr(evidence, "replication_status")

        # Validate data types
        assert isinstance(evidence.authors, list)
        assert isinstance(evidence.year, int)
        assert isinstance(evidence.sample_size, int)
        assert isinstance(evidence.outcomes, list)
        assert isinstance(evidence.effect_sizes, dict)
        assert isinstance(evidence.confidence_intervals, dict)
        assert isinstance(evidence.limitations, list)

    def test_practice_guidelines_structure(self, validator):
        """Test practice guidelines structure."""
        guidelines = validator.practice_guidelines

        assert PracticeGuidelineSource.APA_DIVISION_12 in guidelines
        assert PracticeGuidelineSource.SAMHSA in guidelines

        # Check APA guideline structure
        apa_guidelines = guidelines[PracticeGuidelineSource.APA_DIVISION_12]
        assert len(apa_guidelines) > 0

        guideline = apa_guidelines[0]
        assert hasattr(guideline, "id")
        assert hasattr(guideline, "title")
        assert hasattr(guideline, "source")
        assert hasattr(guideline, "version")
        assert hasattr(guideline, "publication_date")
        assert hasattr(guideline, "target_population")
        assert hasattr(guideline, "target_conditions")
        assert hasattr(guideline, "recommended_interventions")
        assert hasattr(guideline, "evidence_strength")
        assert hasattr(guideline, "implementation_requirements")
        assert hasattr(guideline, "contraindications")
        assert hasattr(guideline, "outcome_expectations")
        assert hasattr(guideline, "monitoring_protocols")
        assert hasattr(guideline, "update_schedule")

        # Validate data types
        assert isinstance(guideline.target_conditions, list)
        assert isinstance(guideline.recommended_interventions, list)
        assert isinstance(guideline.implementation_requirements, list)
        assert isinstance(guideline.contraindications, list)
        assert isinstance(guideline.outcome_expectations, list)
        assert isinstance(guideline.monitoring_protocols, list)

    def test_outcome_measures_structure(self, validator):
        """Test outcome measures structure."""
        measures = validator.outcome_measures

        assert OutcomeMeasureType.SYMPTOM_REDUCTION in measures
        assert OutcomeMeasureType.THERAPEUTIC_ALLIANCE in measures
        assert OutcomeMeasureType.FUNCTIONAL_IMPROVEMENT in measures

        # Check symptom reduction measures
        symptom_measures = measures[OutcomeMeasureType.SYMPTOM_REDUCTION]
        assert len(symptom_measures) > 0

        measure = symptom_measures[0]
        assert hasattr(measure, "id")
        assert hasattr(measure, "name")
        assert hasattr(measure, "acronym")
        assert hasattr(measure, "measure_type")
        assert hasattr(measure, "target_construct")
        assert hasattr(measure, "administration_time")
        assert hasattr(measure, "scoring_method")
        assert hasattr(measure, "reliability_coefficient")
        assert hasattr(measure, "validity_evidence")
        assert hasattr(measure, "normative_data")
        assert hasattr(measure, "clinical_cutoffs")
        assert hasattr(measure, "change_indicators")
        assert hasattr(measure, "frequency_of_use")
        assert hasattr(measure, "population_appropriateness")

        # Validate data types
        assert isinstance(measure.administration_time, int)
        assert isinstance(measure.reliability_coefficient, float)
        assert isinstance(measure.validity_evidence, list)
        assert isinstance(measure.normative_data, dict)
        assert isinstance(measure.clinical_cutoffs, dict)
        assert isinstance(measure.change_indicators, dict)
        assert isinstance(measure.population_appropriateness, list)

    def test_fidelity_protocols_structure(self, validator):
        """Test fidelity protocols structure."""
        protocols = validator.fidelity_protocols

        assert TreatmentFidelityDomain.PROTOCOL_ADHERENCE in protocols
        assert TreatmentFidelityDomain.THERAPIST_COMPETENCE in protocols
        assert TreatmentFidelityDomain.INTERVENTION_INTEGRITY in protocols

        # Check protocol adherence structure
        adherence_protocol = protocols[TreatmentFidelityDomain.PROTOCOL_ADHERENCE]
        assert "description" in adherence_protocol
        assert "assessment_criteria" in adherence_protocol
        assert "scoring_method" in adherence_protocol
        assert "benchmarks" in adherence_protocol
        assert "monitoring_frequency" in adherence_protocol

        # Validate data types
        assert isinstance(adherence_protocol["assessment_criteria"], list)
        assert isinstance(adherence_protocol["benchmarks"], dict)

    def test_validate_conversation(self, validator, depression_scenario, cbt_conversation):
        """Test conversation validation."""
        validated_conversation = validator.validate_conversation(
            cbt_conversation, depression_scenario, "cognitive_behavioral"
        )

        assert isinstance(validated_conversation, EvidenceBasedConversation)
        assert validated_conversation.id is not None
        assert validated_conversation.base_conversation == cbt_conversation
        assert validated_conversation.client_scenario == depression_scenario
        assert validated_conversation.therapeutic_approach == "cognitive_behavioral"
        assert len(validated_conversation.conversation_exchanges) == 6
        assert isinstance(validated_conversation.evidence_citations, list)
        assert isinstance(validated_conversation.guideline_adherence, list)
        assert isinstance(validated_conversation.outcome_predictions, dict)
        assert isinstance(validated_conversation.fidelity_markers, list)
        assert isinstance(validated_conversation.validation_result, ValidationResult)
        assert isinstance(validated_conversation.effectiveness_indicators, list)
        assert isinstance(validated_conversation.research_support_summary, str)
        assert isinstance(validated_conversation.clinical_recommendations, list)

    def test_find_relevant_evidence(self, validator, depression_scenario):
        """Test finding relevant research evidence."""
        evidence = validator._find_relevant_evidence("cognitive_behavioral", depression_scenario)

        assert isinstance(evidence, list)
        assert len(evidence) > 0

        # Should include CBT evidence
        cbt_evidence = [e for e in evidence if "cbt" in e.id.lower() or "cognitive" in e.intervention.lower()]
        assert len(cbt_evidence) > 0

        # Should include therapeutic alliance evidence
        alliance_evidence = [e for e in evidence if "alliance" in e.id.lower()]
        assert len(alliance_evidence) > 0

    def test_assess_guideline_adherence(self, validator, depression_scenario, cbt_conversation):
        """Test guideline adherence assessment."""
        exchanges = cbt_conversation["conversation_exchanges"]
        adherent_guidelines = validator._assess_guideline_adherence(
            "cognitive_behavioral", exchanges, depression_scenario
        )

        assert isinstance(adherent_guidelines, list)
        # Should find at least one adherent guideline for CBT
        assert len(adherent_guidelines) > 0

    def test_predict_outcomes(self, validator, depression_scenario, cbt_conversation):
        """Test outcome prediction."""
        evidence = validator._find_relevant_evidence("cognitive_behavioral", depression_scenario)
        exchanges = cbt_conversation["conversation_exchanges"]

        predictions = validator._predict_outcomes(evidence, exchanges, depression_scenario)

        assert isinstance(predictions, dict)
        assert len(predictions) > 0

        # All predictions should be between 0 and 1
        for _outcome, probability in predictions.items():
            assert 0.0 <= probability <= 1.0

    def test_identify_fidelity_markers(self, validator, cbt_conversation):
        """Test fidelity markers identification."""
        exchanges = cbt_conversation["conversation_exchanges"]
        markers = validator._identify_fidelity_markers(exchanges, "cognitive_behavioral")

        assert isinstance(markers, list)
        assert "adequate_session_length" in markers  # 6 exchanges >= 6
        assert "therapeutic_techniques_used" in markers  # techniques present
        assert "outcome_monitoring_present" in markers  # assessment mentioned

    def test_conversation_quality_assessment(self, validator, cbt_conversation):
        """Test conversation quality assessment."""
        exchanges = cbt_conversation["conversation_exchanges"]
        quality = validator._assess_conversation_quality(exchanges)

        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0

        # Should be high quality due to techniques, engagement, and monitoring
        assert quality >= 0.5

    def test_validation_score_calculation(self, validator):
        """Test validation score calculation."""
        # Mock evidence and guidelines
        evidence = [Mock(evidence_level=EvidenceLevel.LEVEL_I)]
        guidelines = [Mock()]
        fidelity_markers = ["adequate_session_length", "therapeutic_techniques_used"]

        score = validator._calculate_validation_score(evidence, guidelines, fidelity_markers)

        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0
        assert score >= 50.0  # Should be good score with strong evidence

    def test_evidence_strength_rating(self, validator):
        """Test evidence strength rating determination."""
        # Test strong evidence
        strong_evidence = [Mock(evidence_level=EvidenceLevel.LEVEL_I)]
        strong_rating = validator._determine_evidence_strength_rating(strong_evidence)
        assert strong_rating == "strong_evidence"

        # Test moderate evidence
        moderate_evidence = [Mock(evidence_level=EvidenceLevel.LEVEL_II)]
        moderate_rating = validator._determine_evidence_strength_rating(moderate_evidence)
        assert moderate_rating == "moderate_to_strong_evidence"

        # Test no evidence
        no_evidence = []
        no_rating = validator._determine_evidence_strength_rating(no_evidence)
        assert no_rating == "insufficient_evidence"

    def test_research_support_summary(self, validator):
        """Test research support summary generation."""
        evidence = validator.research_database["cognitive_behavioral_therapy"]
        summary = validator._create_research_support_summary(evidence)

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "studies" in summary.lower()
        assert "effect sizes" in summary.lower()

    def test_clinical_recommendations_generation(self, validator):
        """Test clinical recommendations generation."""
        # Mock validation result
        validation_result = Mock(validation_score=85.0)
        guidelines = [Mock()]

        recommendations = validator._generate_clinical_recommendations(validation_result, guidelines)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("continue" in rec.lower() for rec in recommendations)  # High score recommendation

    def test_export_validated_conversations(self, validator, depression_scenario, cbt_conversation):
        """Test validated conversations export."""
        # Generate validated conversation
        validated_conv = validator.validate_conversation(
            cbt_conversation, depression_scenario, "cognitive_behavioral"
        )

        conversations = [validated_conv]

        # Test export
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            export_result = validator.export_validated_conversations(conversations, temp_file)

            # Check export result
            assert export_result["exported_conversations"] == 1
            assert export_result["output_file"] == temp_file
            assert "average_validation_score" in export_result
            assert "therapeutic_approaches" in export_result
            assert "guideline_compliance_rate" in export_result

            # Check exported file
            assert os.path.exists(temp_file)
            with open(temp_file, encoding="utf-8") as f:
                exported_data = json.load(f)

            assert "metadata" in exported_data
            assert "conversations" in exported_data
            assert exported_data["metadata"]["total_conversations"] == 1
            assert len(exported_data["conversations"]) == 1

            # Check metadata
            metadata = exported_data["metadata"]
            assert "therapeutic_approaches" in metadata
            assert "average_validation_score" in metadata
            assert "evidence_strength_distribution" in metadata
            assert "guideline_compliance_rate" in metadata

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_different_therapeutic_approaches(self, validator, depression_scenario, cbt_conversation):
        """Test validation of different therapeutic approaches."""
        # Test CBT
        cbt_validated = validator.validate_conversation(
            cbt_conversation, depression_scenario, "cognitive_behavioral"
        )
        assert cbt_validated.therapeutic_approach == "cognitive_behavioral"

        # Test trauma-informed approach
        trauma_validated = validator.validate_conversation(
            cbt_conversation, depression_scenario, "trauma_informed"
        )
        assert trauma_validated.therapeutic_approach == "trauma_informed"

        # Test alliance building
        alliance_validated = validator.validate_conversation(
            cbt_conversation, depression_scenario, "alliance_building"
        )
        assert alliance_validated.therapeutic_approach == "alliance_building"


if __name__ == "__main__":
    pytest.main([__file__])
