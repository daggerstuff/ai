"""
Unit tests for Expert Validation Interface

This module provides comprehensive unit tests for the expert validation
interface, covering expert management, validation workflows, and consensus
evaluation.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock

from .expert_validation_interface import (
    ExpertValidationInterface,
    ExpertProfile,
    ValidationRequest,
    ExpertSpecialty,
    ValidationStatus,
    ValidationPriority,
)

from .clinical_accuracy_validator import (
    ClinicalAccuracyResult,
    ClinicalContext,
    DSM5Assessment,
    PDM2Assessment,
    TherapeuticAppropriatenessScore,
    SafetyAssessment,
    ClinicalAccuracyLevel,
    TherapeuticModality,
    SafetyRiskLevel,
)


class TestExpertValidationInterface:
    """Test suite for ExpertValidationInterface"""

    @pytest.fixture
    def interface(self):
        """Create an interface instance for testing"""
        return ExpertValidationInterface()

    @pytest.fixture
    def sample_expert(self):
        """Create a sample expert profile"""
        return ExpertProfile(
            expert_id="test_expert_001",
            name="Dr. Test Expert",
            email="test@example.com",
            specialties=[ExpertSpecialty.CLINICAL_PSYCHOLOGY],
            credentials=["PhD Clinical Psychology"],
            years_experience=10,
            license_number="TEST123",
            institution="Test University",
            accuracy_rating=0.9,
        )

    @pytest.fixture
    def sample_assessment_result(self):
        """Create a sample assessment result"""
        context = ClinicalContext(
            client_presentation="Test presentation",
            therapeutic_modality=TherapeuticModality.CBT,
            session_phase="initial",
        )

        return ClinicalAccuracyResult(
            assessment_id="test_assessment_001",
            timestamp=datetime.now(),
            clinical_context=context,
            dsm5_assessment=DSM5Assessment(primary_diagnosis="Test Disorder"),
            pdm2_assessment=PDM2Assessment(),
            therapeutic_appropriateness=TherapeuticAppropriatenessScore(overall_score=0.8),
            safety_assessment=SafetyAssessment(overall_risk=SafetyRiskLevel.LOW),
            overall_accuracy=ClinicalAccuracyLevel.GOOD,
            confidence_score=0.85,
            expert_validation_needed=True,
            recommendations=["Test recommendation"],
            warnings=[],
        )

    def test_interface_initialization(self, interface):
        """Test interface initialization"""
        assert interface is not None
        assert interface.config is not None
        assert isinstance(interface.experts, dict)
        assert isinstance(interface.validation_requests, dict)
        assert isinstance(interface.validation_responses, dict)
        assert isinstance(interface.consensus_results, dict)
        assert len(interface.experts) > 0  # Should have sample experts

    def test_config_loading_default(self):
        """Test default configuration loading"""
        interface = ExpertValidationInterface()

        assert interface.config["default_deadline_hours"] == 48
        assert interface.config["urgent_deadline_hours"] == 4
        assert interface.config["critical_deadline_hours"] == 1
        assert interface.config["min_consensus_agreement"] == 0.7
        assert interface.config["notification_enabled"] is True

    def test_config_loading_custom(self):
        """Test custom configuration loading"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            custom_config = {"default_deadline_hours": 24, "urgent_deadline_hours": 2}
            json.dump(custom_config, f)
            config_path = Path(f.name)

        try:
            interface = ExpertValidationInterface(config_path)
            assert interface.config["default_deadline_hours"] == 24
            assert interface.config["urgent_deadline_hours"] == 2
            # Default value
            assert interface.config["critical_deadline_hours"] == 1
        finally:
            config_path.unlink()

    def test_add_expert_success(self, interface, sample_expert):
        """Test successful expert addition"""
        initial_count = len(interface.experts)
        result = interface.add_expert(sample_expert)

        assert result is True
        assert len(interface.experts) == initial_count + 1
        assert interface.experts[sample_expert.expert_id] == sample_expert

    def test_add_expert_duplicate(self, interface, sample_expert):
        """Test adding duplicate expert"""
        interface.add_expert(sample_expert)
        result = interface.add_expert(sample_expert)

        assert result is False

    def test_add_expert_invalid_profile(self, interface):
        """Test adding expert with invalid profile"""
        invalid_expert = ExpertProfile(
            expert_id="invalid",
            name="",  # Missing required field
            email="test@example.com",
            specialties=[],  # Empty specialties
            credentials=[],
            years_experience=-1,  # Invalid experience
            license_number="",
            institution="",
        )

        result = interface.add_expert(invalid_expert)
        assert result is False

    def test_update_expert_success(self, interface, sample_expert):
        """Test successful expert update"""
        interface.add_expert(sample_expert)

        updates = {"name": "Dr. Updated Name", "years_experience": 15, "is_active": False}

        result = interface.update_expert(sample_expert.expert_id, updates)

        assert result is True
        updated_expert = interface.get_expert(sample_expert.expert_id)
        assert updated_expert.name == "Dr. Updated Name"
        assert updated_expert.years_experience == 15
        assert updated_expert.is_active is False

    def test_update_expert_not_found(self, interface):
        """Test updating non-existent expert"""
        result = interface.update_expert("nonexistent", {"name": "Test"})
        assert result is False

    def test_get_expert(self, interface, sample_expert):
        """Test getting expert by ID"""
        interface.add_expert(sample_expert)

        retrieved_expert = interface.get_expert(sample_expert.expert_id)
        assert retrieved_expert == sample_expert

        non_existent = interface.get_expert("nonexistent")
        assert non_existent is None

    def test_list_experts_all(self, interface):
        """Test listing all experts"""
        experts = interface.list_experts()
        assert len(experts) > 0
        assert all(expert.is_active for expert in experts)

    def test_list_experts_by_specialty(self, interface):
        """Test listing experts by specialty"""
        experts = interface.list_experts(specialty=ExpertSpecialty.CLINICAL_PSYCHOLOGY)
        assert len(experts) > 0
        assert all(ExpertSpecialty.CLINICAL_PSYCHOLOGY in expert.specialties for expert in experts)

    def test_list_experts_include_inactive(self, interface, sample_expert):
        """Test listing experts including inactive ones"""
        sample_expert.is_active = False
        interface.add_expert(sample_expert)

        active_experts = interface.list_experts(active_only=True)
        all_experts = interface.list_experts(active_only=False)

        assert len(all_experts) > len(active_experts)

    @pytest.mark.asyncio
    async def test_create_validation_request_success(self, interface, sample_assessment_result):
        """Test successful validation request creation"""
        request_id = await interface.create_validation_request(
            assessment_result=sample_assessment_result,
            priority=ValidationPriority.HIGH,
            required_specialties=[ExpertSpecialty.CLINICAL_PSYCHOLOGY],
        )

        assert request_id is not None
        assert request_id.startswith("val_")

        request = interface.get_validation_request(request_id)
        assert request is not None
        assert request.priority == ValidationPriority.HIGH
        assert ExpertSpecialty.CLINICAL_PSYCHOLOGY in request.required_specialties
        assert request.status in [ValidationStatus.PENDING, ValidationStatus.ASSIGNED]

    @pytest.mark.asyncio
    async def test_create_validation_request_auto_specialty(
        self, interface, sample_assessment_result
    ):
        """Test validation request creation with automatic specialty
        determination"""
        # Modify assessment to trigger specific specialty
        sample_assessment_result.safety_assessment.overall_risk = SafetyRiskLevel.HIGH

        request_id = await interface.create_validation_request(
            assessment_result=sample_assessment_result, priority=ValidationPriority.CRITICAL
        )

        request = interface.get_validation_request(request_id)
        assert ExpertSpecialty.CRISIS_INTERVENTION in request.required_specialties

    @pytest.mark.asyncio
    async def test_create_validation_request_custom_deadline(
        self, interface, sample_assessment_result
    ):
        """Test validation request creation with custom deadline"""
        custom_deadline = datetime.now() + timedelta(hours=12)

        request_id = await interface.create_validation_request(
            assessment_result=sample_assessment_result, custom_deadline=custom_deadline
        )

        request = interface.get_validation_request(request_id)
        # Within 1 minute
        time_diff = abs((request.deadline - custom_deadline).total_seconds())
        assert time_diff < 60

    def test_determine_required_specialties_crisis(self, interface):
        """Test specialty determination for crisis cases"""
        context = ClinicalContext(
            client_presentation="Test",
            therapeutic_modality=TherapeuticModality.CBT,
            session_phase="initial",
        )

        assessment = ClinicalAccuracyResult(
            assessment_id="test",
            timestamp=datetime.now(),
            clinical_context=context,
            dsm5_assessment=DSM5Assessment(),
            pdm2_assessment=PDM2Assessment(),
            therapeutic_appropriateness=TherapeuticAppropriatenessScore(),
            safety_assessment=SafetyAssessment(overall_risk=SafetyRiskLevel.CRITICAL),
            overall_accuracy=ClinicalAccuracyLevel.DANGEROUS,
            confidence_score=0.3,
            expert_validation_needed=True,
            recommendations=[],
            warnings=[],
        )

        specialties = interface._determine_required_specialties(assessment)
        assert ExpertSpecialty.CRISIS_INTERVENTION in specialties

    def test_determine_required_specialties_substance_abuse(self, interface):
        """Test specialty determination for substance abuse cases"""
        context = ClinicalContext(
            client_presentation="Test",
            therapeutic_modality=TherapeuticModality.CBT,
            session_phase="initial",
        )

        assessment = ClinicalAccuracyResult(
            assessment_id="test",
            timestamp=datetime.now(),
            clinical_context=context,
            dsm5_assessment=DSM5Assessment(primary_diagnosis="Substance Use Disorder"),
            pdm2_assessment=PDM2Assessment(),
            therapeutic_appropriateness=TherapeuticAppropriatenessScore(),
            safety_assessment=SafetyAssessment(),
            overall_accuracy=ClinicalAccuracyLevel.GOOD,
            confidence_score=0.8,
            expert_validation_needed=True,
            recommendations=[],
            warnings=[],
        )

        specialties = interface._determine_required_specialties(assessment)
        assert ExpertSpecialty.SUBSTANCE_ABUSE in specialties

    def test_find_available_experts(self, interface, sample_expert):
        """Test finding available experts for a request"""
        interface.add_expert(sample_expert)

        request = ValidationRequest(
            request_id="test_request",
            assessment_result=Mock(),
            priority=ValidationPriority.NORMAL,
            required_specialties=[ExpertSpecialty.CLINICAL_PSYCHOLOGY],
            deadline=datetime.now() + timedelta(hours=24),
        )

        available_experts = interface._find_available_experts(request)
        assert len(available_experts) > 0
        assert sample_expert in available_experts

    def test_calculate_expert_suitability_score(self, interface, sample_expert):
        """Test expert suitability score calculation"""
        interface.add_expert(sample_expert)

        request = ValidationRequest(
            request_id="test_request",
            assessment_result=Mock(),
            priority=ValidationPriority.NORMAL,
            required_specialties=[ExpertSpecialty.CLINICAL_PSYCHOLOGY],
            deadline=datetime.now() + timedelta(hours=24),
        )

        score = interface._calculate_expert_suitability_score(sample_expert, request)
        assert 0.0 <= score <= 100.0
        # Should be high for matching specialty and good ratings
        assert score > 50.0

    def test_assign_expert_to_request(self, interface, sample_expert):
        """Test assigning expert to request"""
        interface.add_expert(sample_expert)

        request = ValidationRequest(
            request_id="test_request",
            assessment_result=Mock(),
            priority=ValidationPriority.NORMAL,
            required_specialties=[ExpertSpecialty.CLINICAL_PSYCHOLOGY],
            deadline=datetime.now() + timedelta(hours=24),
        )
        interface.validation_requests["test_request"] = request

        result = interface._assign_expert_to_request(sample_expert.expert_id, "test_request")

        assert result is True
        assert sample_expert.expert_id in request.assigned_experts
        assert sample_expert.validation_count == 1

    def test_assign_expert_to_request_duplicate(self, interface, sample_expert):
        """Test assigning expert to request twice"""
        interface.add_expert(sample_expert)

        request = ValidationRequest(
            request_id="test_request",
            assessment_result=Mock(),
            priority=ValidationPriority.NORMAL,
            required_specialties=[ExpertSpecialty.CLINICAL_PSYCHOLOGY],
            deadline=datetime.now() + timedelta(hours=24),
        )
        interface.validation_requests["test_request"] = request

        # First assignment
        result1 = interface._assign_expert_to_request(sample_expert.expert_id, "test_request")
        # Second assignment (duplicate)
        result2 = interface._assign_expert_to_request(sample_expert.expert_id, "test_request")

        assert result1 is True
        assert result2 is False
        assert len(request.assigned_experts) == 1
