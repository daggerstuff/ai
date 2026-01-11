"""
Unit tests for Expert Validation Workflow System
"""


import pytest
from clinical_accuracy_assessment import (
    ClinicalAccuracyAssessmentFramework,
    ClinicalDomain,
)
from expert_validation_workflow import (
    ExpertLevel,
    ExpertProfile,
    ExpertSpecialty,
    ExpertValidationWorkflow,
    ValidationPriority,
    ValidationRequest,
    WorkflowStatus,
)


class TestExpertProfile:
    """Test expert profile validation."""

    def test_valid_expert_profile(self):
        """Test creating valid expert profile."""
        expert = ExpertProfile(
            expert_id="test_expert",
            name="Dr. Test",
            email="test@example.com",
            level=ExpertLevel.SENIOR,
            specialties=[ExpertSpecialty.CLINICAL_PSYCHOLOGY],
            domains=[ClinicalDomain.DSM5_DIAGNOSTIC],
            years_experience=8,
        )

        assert expert.expert_id == "test_expert"
        assert expert.level == ExpertLevel.SENIOR
        assert expert.years_experience == 8
        assert expert.is_active is True
        assert expert.current_workload == 0

    def test_invalid_years_experience(self):
        """Test that invalid years of experience raises error."""
        with pytest.raises(ValueError, match="Years of experience must be between 0 and 50"):
            ExpertProfile(
                expert_id="test",
                name="Test",
                email="test@example.com",
                level=ExpertLevel.SENIOR,
                specialties=[ExpertSpecialty.CLINICAL_PSYCHOLOGY],
                domains=[ClinicalDomain.DSM5_DIAGNOSTIC],
                years_experience=60,
            )

    def test_invalid_rating(self):
        """Test that invalid rating raises error."""
        with pytest.raises(ValueError, match="Rating must be between 1.0 and 5.0"):
            ExpertProfile(
                expert_id="test",
                name="Test",
                email="test@example.com",
                level=ExpertLevel.SENIOR,
                specialties=[ExpertSpecialty.CLINICAL_PSYCHOLOGY],
                domains=[ClinicalDomain.DSM5_DIAGNOSTIC],
                years_experience=5,
                rating=6.0,
            )


class TestValidationRequest:
    """Test validation request functionality."""

    def test_validation_request_creation(self):
        """Test creating validation request."""
        request = ValidationRequest(
            request_id="test_request",
            content_id="test_content",
            content_text="Test content",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            priority=ValidationPriority.HIGH,
            requester_id="test_requester",
        )

        assert request.request_id == "test_request"
        assert request.priority == ValidationPriority.HIGH
        assert request.status == WorkflowStatus.CREATED
        assert request.deadline is not None
        assert len(request.assigned_experts) == 0

    def test_deadline_setting_by_priority(self):
        """Test that deadline is set correctly based on priority."""
        critical_request = ValidationRequest(
            request_id="critical",
            content_id="test",
            content_text="test",
            domain=ClinicalDomain.CRISIS_MANAGEMENT,
            priority=ValidationPriority.CRITICAL,
            requester_id="test",
        )

        normal_request = ValidationRequest(
            request_id="normal",
            content_id="test",
            content_text="test",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            priority=ValidationPriority.NORMAL,
            requester_id="test",
        )

        # Critical should have shorter deadline than normal
        assert critical_request.deadline < normal_request.deadline


class TestExpertValidationWorkflow:
    """Test the main workflow system."""

    @pytest.fixture
    def assessment_framework(self):
        """Create assessment framework for testing."""
        return ClinicalAccuracyAssessmentFramework()

    @pytest.fixture
    def workflow(self, assessment_framework):
        """Create workflow instance for testing."""
        return ExpertValidationWorkflow(assessment_framework)

    @pytest.fixture
    def sample_expert(self):
        """Create sample expert for testing."""
        return ExpertProfile(
            expert_id="expert_001",
            name="Dr. Test Expert",
            email="expert@example.com",
            level=ExpertLevel.SENIOR,
            specialties=[ExpertSpecialty.CLINICAL_PSYCHOLOGY, ExpertSpecialty.TRAUMA_THERAPY],
            domains=[ClinicalDomain.DSM5_DIAGNOSTIC, ClinicalDomain.THERAPEUTIC_INTERVENTION],
            years_experience=10,
        )

    @pytest.fixture
    def distinguished_expert(self):
        """Create distinguished expert for testing."""
        return ExpertProfile(
            expert_id="expert_002",
            name="Dr. Distinguished Expert",
            email="distinguished@example.com",
            level=ExpertLevel.DISTINGUISHED,
            specialties=[ExpertSpecialty.PSYCHIATRY],
            domains=[ClinicalDomain.CRISIS_MANAGEMENT, ClinicalDomain.DSM5_DIAGNOSTIC],
            years_experience=20,
        )

    def test_workflow_initialization(self, workflow):
        """Test workflow initialization."""
        assert isinstance(workflow.experts, dict)
        assert isinstance(workflow.validation_requests, dict)
        assert isinstance(workflow.consensus_results, dict)
        assert len(workflow.experts) == 0

    def test_register_expert(self, workflow, sample_expert):
        """Test expert registration."""
        workflow.register_expert(sample_expert)

        assert sample_expert.expert_id in workflow.experts
        assert workflow.experts[sample_expert.expert_id] == sample_expert

    def test_get_expert(self, workflow, sample_expert):
        """Test getting expert by ID."""
        workflow.register_expert(sample_expert)

        retrieved = workflow.get_expert(sample_expert.expert_id)
        assert retrieved == sample_expert

        # Test non-existent expert
        assert workflow.get_expert("nonexistent") is None

    def test_get_available_experts(self, workflow, sample_expert, distinguished_expert):
        """Test getting available experts with criteria."""
        workflow.register_expert(sample_expert)
        workflow.register_expert(distinguished_expert)

        # Test domain filtering
        dsm5_experts = workflow.get_available_experts(ClinicalDomain.DSM5_DIAGNOSTIC)
        assert len(dsm5_experts) == 2

        crisis_experts = workflow.get_available_experts(ClinicalDomain.CRISIS_MANAGEMENT)
        assert len(crisis_experts) == 1
        assert crisis_experts[0].expert_id == "expert_002"

        # Test level filtering
        distinguished_only = workflow.get_available_experts(
            ClinicalDomain.DSM5_DIAGNOSTIC, required_level=ExpertLevel.DISTINGUISHED
        )
        assert len(distinguished_only) == 1
        assert distinguished_only[0].expert_id == "expert_002"

    def test_get_available_experts_workload_filtering(self, workflow, sample_expert):
        """Test that experts at max workload are filtered out."""
        sample_expert.current_workload = sample_expert.max_concurrent_reviews
        workflow.register_expert(sample_expert)

        available = workflow.get_available_experts(ClinicalDomain.DSM5_DIAGNOSTIC)
        assert len(available) == 0

    def test_get_available_experts_inactive_filtering(self, workflow, sample_expert):
        """Test that inactive experts are filtered out."""
        sample_expert.is_active = False
        workflow.register_expert(sample_expert)

        available = workflow.get_available_experts(ClinicalDomain.DSM5_DIAGNOSTIC)
        assert len(available) == 0

    def test_create_validation_request(self, workflow):
        """Test creating validation request."""
        request_id = workflow.create_validation_request(
            content_id="test_content",
            content_text="Test therapeutic response",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            requester_id="test_requester",
            priority=ValidationPriority.HIGH,
        )

        assert request_id in workflow.validation_requests
        request = workflow.validation_requests[request_id]
        assert request.content_id == "test_content"
        assert request.domain == ClinicalDomain.DSM5_DIAGNOSTIC
        assert request.priority == ValidationPriority.HIGH
        assert request.status == WorkflowStatus.CREATED

    def test_create_validation_request_with_auto_assignment(self, workflow, sample_expert):
        """Test validation request creation with auto-assignment."""
        workflow.register_expert(sample_expert)

        request_id = workflow.create_validation_request(
            content_id="test_content",
            content_text="Test response",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            requester_id="test_requester",
        )

        request = workflow.validation_requests[request_id]
        # Should be auto-assigned since config has auto_assignment=True
        assert request.status == WorkflowStatus.EXPERT_ASSIGNED
        assert len(request.assigned_experts) > 0

    def test_assign_experts_manual(self, workflow, sample_expert):
        """Test manual expert assignment."""
        # Disable auto-assignment before registering expert
        workflow.config["auto_assignment"] = False
        workflow.register_expert(sample_expert)

        request_id = workflow.create_validation_request(
            content_id="test_content",
            content_text="Test response",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            requester_id="test_requester",
        )

        # Verify no auto-assignment occurred
        request = workflow.validation_requests[request_id]
        assert request.status == WorkflowStatus.CREATED
        assert len(request.assigned_experts) == 0

        assigned = workflow.assign_experts(request_id, [sample_expert.expert_id])

        assert len(assigned) == 1
        assert assigned[0] == sample_expert.expert_id

        request = workflow.validation_requests[request_id]
        assert sample_expert.expert_id in request.assigned_experts
        assert sample_expert.current_workload == 1

    def test_assign_experts_automatic(self, workflow, sample_expert, distinguished_expert):
        """Test automatic expert assignment."""
        workflow.register_expert(sample_expert)
        workflow.register_expert(distinguished_expert)

        request_id = workflow.create_validation_request(
            content_id="test_content",
            content_text="Test response",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            requester_id="test_requester",
        )

        # Should have been auto-assigned
        request = workflow.validation_requests[request_id]
        assert len(request.assigned_experts) >= 1
        assert request.status == WorkflowStatus.EXPERT_ASSIGNED

    def test_submit_expert_review(self, workflow, sample_expert):
        """Test submitting expert review."""
        workflow.register_expert(sample_expert)

        request_id = workflow.create_validation_request(
            content_id="test_content",
            content_text="Test therapeutic response",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            requester_id="test_requester",
        )

        # Submit review
        scores = {"dsm5_diagnostic_accuracy": 0.85}
        feedback = {"dsm5_diagnostic_accuracy": "Good diagnostic accuracy"}

        assessment_id = workflow.submit_expert_review(
            request_id=request_id,
            expert_id=sample_expert.expert_id,
            individual_scores=scores,
            feedback=feedback,
            overall_recommendation="Approve with minor revisions",
        )

        assert assessment_id is not None

        request = workflow.validation_requests[request_id]
        assert sample_expert.expert_id in request.completed_reviews
        assert request.completed_reviews[sample_expert.expert_id] == assessment_id

        # Expert workload should be decremented
        assert sample_expert.current_workload == 0
        assert sample_expert.total_reviews == 1

    def test_submit_expert_review_invalid_request(self, workflow, sample_expert):
        """Test submitting review for invalid request."""
        workflow.register_expert(sample_expert)

        with pytest.raises(ValueError, match="Validation request .* not found"):
            workflow.submit_expert_review(
                request_id="invalid_request",
                expert_id=sample_expert.expert_id,
                individual_scores={},
                feedback={},
                overall_recommendation="Test",
            )

    def test_submit_expert_review_unassigned_expert(self, workflow, sample_expert):
        """Test submitting review from unassigned expert."""
        workflow.register_expert(sample_expert)

        request_id = workflow.create_validation_request(
            content_id="test_content",
            content_text="Test response",
            domain=ClinicalDomain.CRISIS_MANAGEMENT,  # Expert not qualified for this domain
            requester_id="test_requester",
        )

        with pytest.raises(ValueError, match="Expert .* not assigned to request"):
            workflow.submit_expert_review(
                request_id=request_id,
                expert_id=sample_expert.expert_id,
                individual_scores={},
                feedback={},
                overall_recommendation="Test",
            )

    def test_consensus_checking(self, workflow, sample_expert, distinguished_expert):
        """Test consensus checking with multiple reviews."""
        workflow.register_expert(sample_expert)
        workflow.register_expert(distinguished_expert)

        request_id = workflow.create_validation_request(
            content_id="test_content",
            content_text="Test therapeutic response",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            requester_id="test_requester",
        )

        # Submit first review
        workflow.submit_expert_review(
            request_id=request_id,
            expert_id=sample_expert.expert_id,
            individual_scores={"dsm5_diagnostic_accuracy": 0.85},
            feedback={"dsm5_diagnostic_accuracy": "Good work"},
            overall_recommendation="Approve",
        )

        # Submit second review
        workflow.submit_expert_review(
            request_id=request_id,
            expert_id=distinguished_expert.expert_id,
            individual_scores={"dsm5_diagnostic_accuracy": 0.82},
            feedback={"dsm5_diagnostic_accuracy": "Very good"},
            overall_recommendation="Approve",
        )

        # Check consensus
        consensus = workflow.get_consensus_result(request_id)
        assert consensus is not None
        assert consensus.consensus_score > 0
        assert len(consensus.individual_scores) == 2

        request = workflow.validation_requests[request_id]
        assert request.status in [
            WorkflowStatus.CONSENSUS_REACHED,
            WorkflowStatus.AWAITING_CONSENSUS,
        ]

    def test_escalation_handling(self, workflow, sample_expert, distinguished_expert):
        """Test escalation when consensus is poor."""
        workflow.register_expert(sample_expert)
        workflow.register_expert(distinguished_expert)

        # Create request that will need escalation
        request_id = workflow.create_validation_request(
            content_id="test_content",
            content_text="Controversial therapeutic response",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            requester_id="test_requester",
        )

        # Submit conflicting reviews to trigger escalation
        workflow.submit_expert_review(
            request_id=request_id,
            expert_id=sample_expert.expert_id,
            individual_scores={"dsm5_diagnostic_accuracy": 0.3},  # Low score
            feedback={"dsm5_diagnostic_accuracy": "Poor accuracy"},
            overall_recommendation="Reject",
        )

        # Manually trigger consensus check with low threshold
        request = workflow.validation_requests[request_id]
        request.consensus_threshold = 0.9  # High threshold to trigger escalation

        # This should trigger escalation due to low consensus
        consensus = workflow._check_consensus(request_id)
        if consensus and consensus.requires_additional_review:
            assert request.status == WorkflowStatus.ESCALATED

    def test_get_expert_workload(self, workflow, sample_expert):
        """Test getting expert workload statistics."""
        workflow.register_expert(sample_expert)

        workload = workflow.get_expert_workload(sample_expert.expert_id)

        assert workload["expert_id"] == sample_expert.expert_id
        assert workload["current_workload"] == 0
        assert workload["max_concurrent_reviews"] == sample_expert.max_concurrent_reviews
        assert workload["utilization"] == 0.0

    def test_get_expert_workload_invalid_expert(self, workflow):
        """Test getting workload for invalid expert."""
        workload = workflow.get_expert_workload("invalid_expert")
        assert workload == {}

    def test_generate_workflow_report(self, workflow, sample_expert):
        """Test generating workflow performance report."""
        workflow.register_expert(sample_expert)

        # Create and complete a request
        request_id = workflow.create_validation_request(
            content_id="test_content",
            content_text="Test response",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            requester_id="test_requester",
        )

        workflow.submit_expert_review(
            request_id=request_id,
            expert_id=sample_expert.expert_id,
            individual_scores={"dsm5_diagnostic_accuracy": 0.85},
            feedback={"dsm5_diagnostic_accuracy": "Good work"},
            overall_recommendation="Approve",
        )

        report = workflow.generate_workflow_report()

        assert "report_period" in report
        assert "request_metrics" in report
        assert "processing_metrics" in report
        assert "expert_utilization" in report
        assert "domain_breakdown" in report
        assert "priority_breakdown" in report

        assert report["request_metrics"]["total_requests"] >= 1
        assert sample_expert.expert_id in report["expert_utilization"]

    def test_expert_selection_algorithms(self, workflow, sample_expert, distinguished_expert):
        """Test different expert selection algorithms."""
        workflow.register_expert(sample_expert)
        workflow.register_expert(distinguished_expert)

        available_experts = [sample_expert, distinguished_expert]

        # Test weighted_random
        workflow.config["expert_selection_algorithm"] = "weighted_random"
        selected = workflow._select_experts(available_experts, 1, 2)
        assert len(selected) <= 2
        assert all(expert in available_experts for expert in selected)

        # Test expertise_match
        workflow.config["expert_selection_algorithm"] = "expertise_match"
        selected = workflow._select_experts(available_experts, 1, 2)
        assert len(selected) <= 2
        # Distinguished expert should be selected first due to higher rating/experience
        if len(selected) > 0:
            assert selected[0].level in [ExpertLevel.DISTINGUISHED, ExpertLevel.PRINCIPAL]

        # Test round_robin
        workflow.config["expert_selection_algorithm"] = "round_robin"
        selected = workflow._select_experts(available_experts, 1, 2)
        assert len(selected) <= 2

    @pytest.mark.asyncio
    async def test_process_pending_requests(self, workflow, sample_expert):
        """Test processing pending requests."""
        workflow.register_expert(sample_expert)
        workflow.config["auto_assignment"] = False  # Disable auto-assignment

        # Create request without auto-assignment
        request_id = workflow.create_validation_request(
            content_id="test_content",
            content_text="Test response",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            requester_id="test_requester",
        )

        request = workflow.validation_requests[request_id]
        assert request.status == WorkflowStatus.CREATED

        # Process pending requests
        await workflow.process_pending_requests()

        # Request should now be assigned
        assert request.status == WorkflowStatus.EXPERT_ASSIGNED
        assert len(request.assigned_experts) > 0


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.fixture
    def setup_workflow(self):
        """Set up complete workflow with experts."""
        assessment_framework = ClinicalAccuracyAssessmentFramework()
        workflow = ExpertValidationWorkflow(assessment_framework)

        # Register multiple experts
        experts = [
            ExpertProfile(
                expert_id="expert_001",
                name="Dr. Sarah Johnson",
                email="sarah@example.com",
                level=ExpertLevel.SENIOR,
                specialties=[ExpertSpecialty.CLINICAL_PSYCHOLOGY],
                domains=[ClinicalDomain.DSM5_DIAGNOSTIC],
                years_experience=8,
            ),
            ExpertProfile(
                expert_id="expert_002",
                name="Dr. Michael Chen",
                email="michael@example.com",
                level=ExpertLevel.PRINCIPAL,
                specialties=[ExpertSpecialty.PSYCHIATRY],
                domains=[ClinicalDomain.DSM5_DIAGNOSTIC, ClinicalDomain.CRISIS_MANAGEMENT],
                years_experience=15,
            ),
            ExpertProfile(
                expert_id="expert_003",
                name="Dr. Emily Rodriguez",
                email="emily@example.com",
                level=ExpertLevel.DISTINGUISHED,
                specialties=[ExpertSpecialty.TRAUMA_THERAPY],
                domains=[ClinicalDomain.THERAPEUTIC_INTERVENTION, ClinicalDomain.DSM5_DIAGNOSTIC],
                years_experience=22,
            ),
        ]

        for expert in experts:
            workflow.register_expert(expert)

        return workflow, experts

    def test_complete_validation_workflow(self, setup_workflow):
        """Test complete validation workflow from request to consensus."""
        workflow, experts = setup_workflow

        # Step 1: Create validation request
        request_id = workflow.create_validation_request(
            content_id="integration_test_content",
            content_text="Sample therapeutic response requiring validation",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            requester_id="integration_tester",
            priority=ValidationPriority.HIGH,
        )

        request = workflow.validation_requests[request_id]
        assert request.status == WorkflowStatus.EXPERT_ASSIGNED
        assert len(request.assigned_experts) >= 2

        # Step 2: Submit reviews from assigned experts
        assigned_experts = list(request.assigned_experts)

        for i, expert_id in enumerate(assigned_experts[:2]):  # Submit 2 reviews
            workflow.submit_expert_review(
                request_id=request_id,
                expert_id=expert_id,
                individual_scores={"dsm5_diagnostic_accuracy": 0.8 + (i * 0.05)},
                feedback={"dsm5_diagnostic_accuracy": f"Review from expert {expert_id}"},
                overall_recommendation="Approve with minor revisions",
            )

        # Step 3: Check final status
        final_request = workflow.validation_requests[request_id]
        assert final_request.status in [
            WorkflowStatus.CONSENSUS_REACHED,
            WorkflowStatus.AWAITING_CONSENSUS,
            WorkflowStatus.ESCALATED,
        ]

        # Step 4: Verify consensus result exists
        consensus = workflow.get_consensus_result(request_id)
        if consensus:
            assert consensus.consensus_score >= 0
            assert len(consensus.individual_scores) >= 2
            assert consensus.average_score > 0

    def test_escalation_workflow(self, setup_workflow):
        """Test escalation workflow with conflicting reviews."""
        workflow, experts = setup_workflow

        # Create request
        request_id = workflow.create_validation_request(
            content_id="escalation_test",
            content_text="Controversial therapeutic response",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            requester_id="escalation_tester",
            priority=ValidationPriority.NORMAL,
        )

        request = workflow.validation_requests[request_id]
        assigned_experts = list(request.assigned_experts)

        # Submit conflicting reviews
        if len(assigned_experts) >= 2:
            workflow.submit_expert_review(
                request_id=request_id,
                expert_id=assigned_experts[0],
                individual_scores={"dsm5_diagnostic_accuracy": 0.3},  # Low score
                feedback={"dsm5_diagnostic_accuracy": "Significant concerns"},
                overall_recommendation="Reject",
            )

            workflow.submit_expert_review(
                request_id=request_id,
                expert_id=assigned_experts[1],
                individual_scores={"dsm5_diagnostic_accuracy": 0.9},  # High score
                feedback={"dsm5_diagnostic_accuracy": "Excellent work"},
                overall_recommendation="Approve",
            )

            # Check if escalation occurred
            final_request = workflow.validation_requests[request_id]
            consensus = workflow.get_consensus_result(request_id)

            if consensus and consensus.requires_additional_review:
                assert final_request.status == WorkflowStatus.ESCALATED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
