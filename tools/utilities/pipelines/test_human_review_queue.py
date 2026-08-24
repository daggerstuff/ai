"""Tests for the human review queue (PIX-250)."""

import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ai.tools.utilities.core.pipelines.human_review_queue import (
    EscalationCriteria,
    HumanReviewQueue,
    ReviewConsistencyGuideline,
    ReviewDecision,
    Reviewer,
    ReviewerRole,
    ReviewFeedbackCollector,
    ReviewStatus,
)


@pytest.fixture
def temp_queue_dir():
    """Create a temporary directory for queue storage."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def queue(temp_queue_dir):
    """Create a review queue with temporary storage."""
    return HumanReviewQueue(data_dir=temp_queue_dir)


@pytest.fixture
def escalation_criteria():
    """Create escalation criteria for testing."""
    return EscalationCriteria()


@pytest.fixture
def consistency_guidelines():
    """Create consistency guidelines for testing."""
    return ReviewConsistencyGuideline()


@pytest.fixture
def feedback_collector(temp_queue_dir):
    """Create feedback collector with temporary storage."""
    return ReviewFeedbackCollector(data_dir=temp_queue_dir)


@pytest.fixture
def sample_report():
    """Sample gate report for testing."""
    return {
        "source_id": "test-source-001",
        "privacy_tier": "high",
        "content_sensitivity": "sensitive",
        "gates": {
            "gate1": {
                "decision": "escalate",
                "reason": "PII tier HIGH requires human review",
            },
            "gate2": {
                "decision": "pass",
                "reason": "no crisis indicators",
            },
        },
        "pii_findings": [
            {"pii_type": "name", "count": 3, "treatment": "scrubbed"},
            {"pii_type": "email", "count": 1, "treatment": "scrubbed"},
        ],
        "crisis_findings": [],
    }


class TestReviewItemCreation:
    """Test ReviewItem creation from gate reports."""

    def test_create_item_from_report(self, queue, sample_report):
        """Should create a review item from a gate report."""
        item = queue.create_item_from_report(
            source_id="test-source-001",
            gate_result=sample_report,
            content_preview="This is a test content preview...",
            content_length=150,
        )

        assert item.item_id.startswith("review-")
        assert item.source_id == "test-source-001"
        assert item.status == ReviewStatus.PENDING
        assert item.content_preview == "This is a test content preview..."
        assert item.content_length == 150
        assert "PII" in item.escalation_reason or "HIGH" in item.escalation_reason

    def test_create_item_extracts_tags(self, queue, sample_report):
        """Should extract tags from gate result."""
        item = queue.create_item_from_report(
            source_id="test-001",
            gate_result=sample_report,
        )

        # Should have tags for privacy tier and sensitivity
        assert any("tier-" in tag for tag in item.tags)
        assert any("sensitivity-" in tag for tag in item.tags)
        assert any("pii-name" in tag or "pii-email" in tag for tag in item.tags)

    def test_create_item_with_priority(self, queue, sample_report):
        """Should respect priority parameter."""
        item = queue.create_item_from_report(
            source_id="test-001",
            gate_result=sample_report,
            priority="urgent",
        )

        assert item.priority == "urgent"


class TestQueueOperations:
    """Test queue enqueue/dequeue operations."""

    def test_enqueue_item(self, queue, sample_report):
        """Should enqueue items."""
        item = queue.create_item_from_report(
            source_id="test-001",
            gate_result=sample_report,
        )

        queue.enqueue(item)

        # Should be retrievable
        retrieved = queue.get_item(item.item_id)
        assert retrieved is not None
        assert retrieved.source_id == "test-001"

    def test_enqueue_duplicate_raises(self, queue, sample_report):
        """Should raise error on duplicate enqueue."""
        item = queue.create_item_from_report(
            source_id="test-001",
            gate_result=sample_report,
        )

        queue.enqueue(item)

        with pytest.raises(ValueError, match="already in queue"):
            queue.enqueue(item)

    def test_dequeue_pending_item(self, queue, sample_report):
        """Should dequeue items in priority order."""
        item1 = queue.create_item_from_report(
            source_id="test-001",
            gate_result=sample_report,
            priority="normal",
        )
        item2 = queue.create_item_from_report(
            source_id="test-002",
            gate_result=sample_report,
            priority="high",
        )

        queue.enqueue(item1)
        queue.enqueue(item2)

        # Should get high priority item first
        next_item = queue.dequeueReusable()
        assert next_item is not None
        assert next_item.priority == "high"

    def test_list_items_with_filters(self, queue, sample_report):
        """Should filter items by status."""
        item1 = queue.create_item_from_report(
            source_id="test-001",
            gate_result=sample_report,
        )
        item2 = queue.create_item_from_report(
            source_id="test-002",
            gate_result=sample_report,
        )

        queue.enqueue(item1)
        queue.enqueue(item2)

        # All items are pending
        pending = queue.list_items(status=ReviewStatus.PENDING)
        assert len(pending) == 2

        # After approving one
        reviewer = Reviewer(id="reviewer-001", role=ReviewerRole.DATA_STEWARD)
        decision = ReviewDecision(
            item_id=item1.item_id,
            reviewer=reviewer,
            decision=ReviewStatus.APPROVED,
            reason="Test approval",
        )
        queue.apply_decision(decision)

        # Now only 1 pending
        pending = queue.list_items(status=ReviewStatus.PENDING)
        assert len(pending) == 1

        approved = queue.list_items(status=ReviewStatus.APPROVED)
        assert len(approved) == 1


class TestReviewDecision:
    """Test review decision application."""

    def test_apply_approval_decision(self, queue, sample_report):
        """Should apply approval decision."""
        item = queue.create_item_from_report(
            source_id="test-001",
            gate_result=sample_report,
        )
        queue.enqueue(item)

        reviewer = Reviewer(id="reviewer-001", role=ReviewerRole.DATA_STEWARD)
        decision = ReviewDecision(
            item_id=item.item_id,
            reviewer=reviewer,
            decision=ReviewStatus.APPROVED,
            reason="PII scrub verified; content appropriate",
        )

        updated = queue.apply_decision(decision)

        assert updated.status == ReviewStatus.APPROVED
        assert updated.reviewer_id == "reviewer-001"
        assert updated.review_decision == "approved"
        assert updated.review_reason == "PII scrub verified; content appropriate"
        assert len(updated.audit_trail) == 2  # Created + decision

    def test_apply_rejection_decision(self, queue, sample_report):
        """Should apply rejection decision."""
        item = queue.create_item_from_report(
            source_id="test-001",
            gate_result=sample_report,
        )
        queue.enqueue(item)

        reviewer = Reviewer(id="reviewer-001", role=ReviewerRole.CLINICAL)
        decision = ReviewDecision(
            item_id=item.item_id,
            reviewer=reviewer,
            decision=ReviewStatus.REJECTED,
            reason="Content contains prohibited material",
        )

        updated = queue.apply_decision(decision)

        assert updated.status == ReviewStatus.REJECTED
        assert updated.review_decision == "rejected"

    def test_cannot_apply_decision_twice(self, queue, sample_report):
        """Should not apply decision to already reviewed item."""
        item = queue.create_item_from_report(
            source_id="test-001",
            gate_result=sample_report,
        )
        queue.enqueue(item)

        reviewer = Reviewer(id="reviewer-001", role=ReviewerRole.DATA_STEWARD)
        decision1 = ReviewDecision(
            item_id=item.item_id,
            reviewer=reviewer,
            decision=ReviewStatus.APPROVED,
            reason="First decision",
        )

        queue.apply_decision(decision1)

        # Second decision should fail
        decision2 = ReviewDecision(
            item_id=item.item_id,
            reviewer=reviewer,
            decision=ReviewStatus.REJECTED,
            reason="Second decision",
        )

        with pytest.raises(ValueError, match="already has status"):
            queue.apply_decision(decision2)

    def test_decision_not_found_item(self, queue):
        """Should raise error for non-existent item."""
        reviewer = Reviewer(id="reviewer-001", role=ReviewerRole.DATA_STEWARD)
        decision = ReviewDecision(
            item_id="review-nonexistent",
            reviewer=reviewer,
            decision=ReviewStatus.APPROVED,
            reason="Test",
        )

        with pytest.raises(ValueError, match="not found in queue"):
            queue.apply_decision(decision)


class TestPersistence:
    """Test queue persistence."""

    def test_items_persist_across_instances(self, temp_queue_dir, sample_report):
        """Should persist items across queue instances."""
        # Create and enqueue in first queue
        queue1 = HumanReviewQueue(data_dir=temp_queue_dir)
        item = queue1.create_item_from_report(
            source_id="test-001",
            gate_result=sample_report,
        )
        queue1.enqueue(item)

        # Create second queue instance
        queue2 = HumanReviewQueue(data_dir=temp_queue_dir)

        # Should load persisted item
        retrieved = queue2.get_item(item.item_id)
        assert retrieved is not None
        assert retrieved.source_id == "test-001"

    def test_decisions_persist(self, temp_queue_dir, sample_report):
        """Should persist decisions to audit log."""
        queue = HumanReviewQueue(data_dir=temp_queue_dir)
        item = queue.create_item_from_report(
            source_id="test-001",
            gate_result=sample_report,
        )
        queue.enqueue(item)

        reviewer = Reviewer(id="reviewer-001", role=ReviewerRole.DATA_STEWARD)
        decision = ReviewDecision(
            item_id=item.item_id,
            reviewer=reviewer,
            decision=ReviewStatus.APPROVED,
            reason="Test approval",
        )
        queue.apply_decision(decision)

        # Reload queue and verify audit trail
        queue2 = HumanReviewQueue(data_dir=temp_queue_dir)
        item2 = queue2.get_item(item.item_id)

        assert item2 is not None
        assert item2.status == ReviewStatus.APPROVED
        assert len(item2.audit_trail) == 2


class TestQueueStats:
    """Test queue statistics."""

    def test_get_stats(self, queue, sample_report):
        """Should calculate queue statistics."""
        # Create items with different statuses
        item1 = queue.create_item_from_report("test-001", sample_report)
        item2 = queue.create_item_from_report("test-002", sample_report)
        item3 = queue.create_item_from_report("test-003", sample_report)

        queue.enqueue(item1)
        queue.enqueue(item2)
        queue.enqueue(item3)

        # Approve one, reject one
        reviewer = Reviewer(id="reviewer-001", role=ReviewerRole.DATA_STEWARD)

        queue.apply_decision(
            ReviewDecision(
                item_id=item1.item_id,
                reviewer=reviewer,
                decision=ReviewStatus.APPROVED,
                reason="Approved",
            )
        )

        queue.apply_decision(
            ReviewDecision(
                item_id=item2.item_id,
                reviewer=reviewer,
                decision=ReviewStatus.REJECTED,
                reason="Rejected",
            )
        )

        stats = queue.get_stats()

        assert stats["total_items"] == 3
        assert stats["pending_count"] == 1
        assert stats["approved_count"] == 1
        assert stats["rejected_count"] == 1


class TestEscalationReasonExtraction:
    """Test escalation reason extraction from gate results."""

    def test_extract_pii_escalation(self, queue):
        """Should extract PII escalation reason."""
        gate_result = {
            "gates": {
                "gate1": {"decision": "escalate", "reason": "PII tier HIGH"},
                "gate2": {"decision": "pass"},
            }
        }

        item = queue.create_item_from_report(
            source_id="test-001",
            gate_result=gate_result,
        )

        assert "PII" in item.escalation_reason or "G1" in item.escalation_reason

    def test_extract_safety_escalation(self, queue):
        """Should extract safety escalation reason."""
        gate_result = {
            "gates": {
                "gate1": {"decision": "pass"},
                "gate2": {"decision": "escalate", "reason": "Crisis score elevated"},
            }
        }

        item = queue.create_item_from_report(
            source_id="test-001",
            gate_result=gate_result,
        )

        assert "G2" in item.escalation_reason or "Crisis" in item.escalation_reason

    def test_extract_license_escalation(self, queue):
        """Should extract license escalation reason."""
        gate_result = {
            "gates": {
                "gate3": {"decision": "escalate", "reason": "Consent required"},
            }
        }

        item = queue.create_item_from_report(
            source_id="test-001",
            gate_result=gate_result,
        )

        assert "G3" in item.escalation_reason or "License" in item.escalation_reason


class TestEscalationCriteria:
    def test_high_privacy_tier_escalates(self, escalation_criteria):
        gate_result = {"privacy_tier": "high"}
        assert escalation_criteria.should_escalate(gate_result) is True

    def test_prohibited_privacy_tier_escalates(self, escalation_criteria):
        gate_result = {"privacy_tier": "prohibited"}
        assert escalation_criteria.should_escalate(gate_result) is True

    def test_none_privacy_tier_no_escalate(self, escalation_criteria):
        gate_result = {"privacy_tier": "none"}
        assert escalation_criteria.should_escalate(gate_result) is False

    def test_restricted_sensitivity_escalates(self, escalation_criteria):
        gate_result = {"content_sensitivity": "restricted"}
        assert escalation_criteria.should_escalate(gate_result) is True

    def test_pii_count_threshold_escalates(self, escalation_criteria):
        gate_result = {"pii_findings": [{"pii_type": "email", "count": 5}]}
        assert escalation_criteria.should_escalate(gate_result) is True

    def test_gate_decision_escalate_triggers(self, escalation_criteria):
        gate_result = {"gates": {"gate1": {"decision": "escalate", "reason": "test"}}}
        assert escalation_criteria.should_escalate(gate_result) is True

    def test_license_requires_review(self, escalation_criteria):
        gate_result = {
            "license_check": {"license_id": "cc-by-nc-4.0", "requires_consent": True, "consent_recorded": False}
        }
        assert escalation_criteria.should_escalate(gate_result) is True

    def test_get_escalation_reasons(self, escalation_criteria):
        gate_result = {"privacy_tier": "high", "content_sensitivity": "restricted"}
        reasons = escalation_criteria.get_escalation_reasons(gate_result)
        assert "privacy_tier=high" in reasons
        assert "content_sensitivity=restricted" in reasons


class TestReviewConsistencyGuideline:
    def test_get_required_reviewer_role_clinical(self, consistency_guidelines):
        gate_result = {"crisis_findings": [{"requires_escalation": True}]}
        role = consistency_guidelines.get_required_reviewer_role(gate_result)
        assert role == ReviewerRole.CLINICAL

    def test_get_required_reviewer_role_privacy(self, consistency_guidelines):
        gate_result = {"privacy_tier": "high"}
        role = consistency_guidelines.get_required_reviewer_role(gate_result)
        assert role == ReviewerRole.PRIVACY

    def test_get_required_reviewer_role_data_steward(self, consistency_guidelines):
        gate_result = {"license_check": {"license_id": "cc-by-nc-4.0"}}
        role = consistency_guidelines.get_required_reviewer_role(gate_result)
        assert role == ReviewerRole.DATA_STEWARD

    def test_get_review_checklist_pii(self, consistency_guidelines):
        gate_result = {"pii_findings": [{"pii_type": "email", "count": 1}]}
        checklist = consistency_guidelines.get_review_checklist(gate_result)
        assert any("email" in item for item in checklist)

    def test_get_review_checklist_consent(self, consistency_guidelines):
        gate_result = {
            "license_check": {"license_id": "cc-by-nc-4.0", "requires_consent": True, "consent_recorded": False}
        }
        checklist = consistency_guidelines.get_review_checklist(gate_result)
        assert any("Consent verification" in item for item in checklist)


class TestReviewFeedbackCollector:
    def test_record_and_retrieve_decision(self, feedback_collector, sample_report):
        decision = ReviewDecision(
            item_id="review-test-001",
            reviewer=Reviewer(id="r1", role=ReviewerRole.DATA_STEWARD),
            decision=ReviewStatus.APPROVED,
            reason="test approval",
        )
        feedback_collector.record_decision(decision)
        feedback = feedback_collector.get_feedback()
        assert feedback.total_reviews == 1
        assert feedback.approval_rate == 1.0

    def test_feedback_approval_rate(self, feedback_collector):
        for i in range(3):
            feedback_collector.record_decision(
                {
                    "item_id": f"item-{i}",
                    "reviewer_id": "r1",
                    "decision": "approved",
                    "reason": "test",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        feedback = feedback_collector.get_feedback()
        assert feedback.approval_rate == 1.0
        assert feedback.total_reviews == 3

    def test_feedback_rejection_rate(self, feedback_collector):
        for i in range(2):
            feedback_collector.record_decision(
                {
                    "item_id": f"item-{i}",
                    "reviewer_id": "r1",
                    "decision": "rejected",
                    "reason": "test",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        feedback = feedback_collector.get_feedback()
        assert feedback.rejection_rate == 1.0

    def test_patterns_identified(self, feedback_collector):
        for i in range(5):
            feedback_collector.record_decision(
                {
                    "item_id": f"item-{i}",
                    "reviewer_id": "r1",
                    "decision": "rejected",
                    "reason": "prohibited content",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        feedback = feedback_collector.get_feedback()
        assert len(feedback.escalation_patterns) > 0
        assert any("High rejection rate" in p for p in feedback.escalation_patterns)


class TestGateToRecordMetadataIntegration:
    """Integration test for gate-to-record metadata flow (PIX-250).

    This test verifies that PrivacyContentReport properly populates
    RecordMetadata fields when flowing through the human review system.
    """

    def test_privacy_report_to_review_item(self, queue, sample_report):
        """Should create review item from privacy report."""
        item = queue.create_item_from_report(
            source_id="test-source-001",
            gate_result=sample_report,
            content_preview="Test content...",
            content_length=150,
        )

        # Verify escalation reason is extracted
        assert "PII" in item.escalation_reason or "G1" in item.escalation_reason

        # Verify tags are extracted
        assert any("tier-" in tag for tag in item.tags)
        assert any("pii-" in tag for tag in item.tags)

    def test_review_item_to_record_metadata(self, queue, sample_report):
        """Should populate RecordMetadata from review item."""
        from ai.training_corpus.rewrite_contracts import RecordMetadata

        # Create review item from report
        item = queue.create_item_from_report(
            source_id="test-source-001",
            gate_result=sample_report,
        )

        # Simulate what assembly pipeline does when creating metadata
        metadata = RecordMetadata(
            artifact_type="dataset_item",
            source_origin=item.source_id,
            needs_human_review=True,
            review_item_id=item.item_id,
            gate4_result=None,  # Not yet reviewed
        )

        # Verify metadata is correctly populated
        assert metadata.needs_human_review is True
        assert metadata.review_item_id is not None
        assert metadata.review_item_id.startswith("review-")
        assert metadata.gate4_result is None

        # Verify gate_result fields are accessible
        assert item.gate_result is not None
        assert item.gate_result.get("privacy_tier") == "high"
        assert item.gate_result.get("content_sensitivity") == "sensitive"

    def test_full_flow_escalation_to_approval(self, temp_queue_dir, sample_report):
        """Test full flow: escalation → review → approval → metadata update."""
        from ai.training_corpus.rewrite_contracts import RecordMetadata

        queue = HumanReviewQueue(data_dir=temp_queue_dir)

        # Step 1: Create and enqueue item from gate report
        item = queue.create_item_from_report(
            source_id="test-source-001",
            gate_result=sample_report,
            content_preview="Test content for review...",
            content_length=200,
        )
        queue.enqueue(item)

        # Verify item is in queue
        assert item.status == ReviewStatus.PENDING
        assert item.gate_result is not None
        assert item.gate_result.get("privacy_tier") == "high"

        # Step 2: Simulate human review (approval)
        reviewer = Reviewer(id="reviewer-001", role=ReviewerRole.PRIVACY)
        decision = ReviewDecision(
            item_id=item.item_id,
            reviewer=reviewer,
            decision=ReviewStatus.APPROVED,
            reason="PII scrub verified; content appropriate for training",
            additional_notes={"verified_by": "privacy-team", "verification_date": "2026-05-10"},
        )
        updated_item = queue.apply_decision(decision)

        # Verify decision is recorded
        assert updated_item.status == ReviewStatus.APPROVED
        assert updated_item.review_decision == "approved"
        assert updated_item.review_reason == decision.reason

        # Step 3: Create RecordMetadata after approval
        metadata = RecordMetadata(
            artifact_type="dataset_item",
            source_origin=updated_item.source_id,
            needs_human_review=True,
            review_item_id=updated_item.item_id,
            gate4_result=updated_item.review_decision,  # Now set to approved
        )

        # Verify metadata reflects approved state
        assert metadata.needs_human_review is True
        assert metadata.review_item_id == updated_item.item_id
        assert metadata.gate4_result == "approved"

        # Step 4: Verify audit trail contains full decision details
        audit_entries = [e for e in updated_item.audit_trail if e.get("event") == "decision"]
        assert len(audit_entries) == 1
        assert audit_entries[0]["reviewer_id"] == "reviewer-001"
        assert audit_entries[0]["reviewer_role"] == "privacy"
        assert audit_entries[0]["decision"] == "approved"
        assert audit_entries[0]["reason"] == decision.reason
        # Verify additional_notes are included in audit trail (PIX-250 audit completeness)
        assert "additional_notes" in audit_entries[0]
        assert audit_entries[0]["additional_notes"] == decision.additional_notes
        assert audit_entries[0]["additional_notes"]["verified_by"] == "privacy-team"
        assert audit_entries[0]["additional_notes"]["verification_date"] == "2026-05-10"

    def test_rejection_flow(self, queue, sample_report):
        """Test rejection flow updates metadata correctly."""
        from ai.training_corpus.rewrite_contracts import RecordMetadata

        # Create and reject item
        item = queue.create_item_from_report(
            source_id="test-source-001",
            gate_result=sample_report,
        )
        queue.enqueue(item)

        reviewer = Reviewer(id="reviewer-002", role=ReviewerRole.CLINICAL)
        decision = ReviewDecision(
            item_id=item.item_id,
            reviewer=reviewer,
            decision=ReviewStatus.REJECTED,
            reason="Content contains prohibited material",
        )
        updated_item = queue.apply_decision(decision)

        # Create metadata for rejected item
        metadata = RecordMetadata(
            artifact_type="dataset_item",
            source_origin=updated_item.source_id,
            needs_human_review=True,
            review_item_id=updated_item.item_id,
            gate4_result=updated_item.review_decision,
        )

        # Verify rejection is reflected
        assert metadata.gate4_result == "rejected"
        assert updated_item.status == ReviewStatus.REJECTED

    def test_gate_result_fields_propagation(self, queue):
        """Verify all gate result fields propagate correctly."""
        gate_result = {
            "privacy_tier": "high",
            "content_sensitivity": "restricted",
            "gates": {
                "gate1": {"decision": "escalate", "reason": "PII detected"},
                "gate2": {"decision": "pass", "reason": "No crisis indicators"},
            },
            "pii_findings": [
                {"pii_type": "email", "count": 2, "treatment": "scrubbed"},
                {"pii_type": "phone", "count": 1, "treatment": "scrubbed"},
            ],
            "crisis_findings": [],
            "license_check": {"license_id": "cc-by-4.0", "requires_consent": False},
        }

        item = queue.create_item_from_report(
            source_id="test-source-001",
            gate_result=gate_result,
        )

        # Verify gate_result is stored
        assert item.gate_result is not None
        assert item.gate_result["privacy_tier"] == "high"
        assert item.gate_result["content_sensitivity"] == "restricted"
        assert len(item.gate_result.get("pii_findings", [])) == 2

        # Verify tags extracted
        assert "tier-high" in item.tags
        assert "pii-email" in item.tags
        assert "pii-phone" in item.tags


class TestRoleValidation:
    """Test optional role validation against ReviewConsistencyGuideline."""

    def test_role_validation_passes_with_correct_role(self, queue, sample_report):
        """Should allow decision when reviewer role matches requirement."""
        # Create item with high privacy tier (requires privacy review)
        item = queue.create_item_from_report(
            source_id="test-source-001",
            gate_result=sample_report,  # Has privacy_tier="high"
        )
        queue.enqueue(item)

        # Privacy reviewer should be allowed
        reviewer = Reviewer(id="privacy-001", role=ReviewerRole.PRIVACY)
        decision = ReviewDecision(
            item_id=item.item_id,
            reviewer=reviewer,
            decision=ReviewStatus.APPROVED,
            reason="Privacy review complete",
        )

        # Validation passes with correct role
        guidelines = ReviewConsistencyGuideline()
        updated = queue.apply_decision(decision, guidelines, validate_role=True)

        assert updated.status == ReviewStatus.APPROVED

    def test_role_validation_rejects_wrong_role(self, queue, sample_report):
        """Should reject decision when reviewer role doesn't match requirement."""
        # Create item with high privacy tier (requires privacy review)
        item = queue.create_item_from_report(
            source_id="test-source-001",
            gate_result=sample_report,  # Has privacy_tier="high"
        )
        queue.enqueue(item)

        # Data steward should NOT be allowed for high privacy tier
        reviewer = Reviewer(id="steward-001", role=ReviewerRole.DATA_STEWARD)
        decision = ReviewDecision(
            item_id=item.item_id,
            reviewer=reviewer,
            decision=ReviewStatus.APPROVED,
            reason="Data steward review",
        )

        # Validation fails with wrong role
        guidelines = ReviewConsistencyGuideline()
        with pytest.raises(ValueError, match="does not match required role"):
            queue.apply_decision(decision, guidelines, validate_role=True)

    def test_role_validation_with_crisis_requires_clinical(self, queue):
        """Crisis findings require clinical reviewer."""
        report_with_crisis = {
            "gates": {},
            "pii_findings": [],
            "crisis_findings": [{"crisis_type": "suicidal", "score": 0.8, "requires_escalation": True}],
            "privacy_tier": "medium",
            "content_sensitivity": "sensitive",
            "retention_policy": "restrict",
            "license_check": {"license_id": "cc-by-4.0", "requires_consent": False},
        }

        item = queue.create_item_from_report(
            source_id="test-source-002",
            gate_result=report_with_crisis,
        )
        queue.enqueue(item)

        # Non-clinical reviewer should be rejected
        reviewer = Reviewer(id="steward-002", role=ReviewerRole.DATA_STEWARD)
        decision = ReviewDecision(
            item_id=item.item_id,
            reviewer=reviewer,
            decision=ReviewStatus.APPROVED,
            reason="Review complete",
        )

        guidelines = ReviewConsistencyGuideline()
        with pytest.raises(ValueError, match="requires clinical"):
            queue.apply_decision(decision, guidelines, validate_role=True)

        # Clinical reviewer should be accepted
        clinical_reviewer = Reviewer(id="clinical-001", role=ReviewerRole.CLINICAL)
        clinical_decision = ReviewDecision(
            item_id=item.item_id,
            reviewer=clinical_reviewer,
            decision=ReviewStatus.APPROVED,
            reason="Clinical review complete",
        )
        updated = queue.apply_decision(clinical_decision, guidelines, validate_role=True)
        assert updated.status == ReviewStatus.APPROVED

    def test_role_validation_disabled_by_default(self, queue, sample_report):
        """Role validation should not run when disabled (backward compatible)."""
        item = queue.create_item_from_report(
            source_id="test-source-001",
            gate_result=sample_report,
        )
        queue.enqueue(item)

        # Any reviewer should work without validation
        reviewer = Reviewer(id="steward-001", role=ReviewerRole.DATA_STEWARD)
        decision = ReviewDecision(
            item_id=item.item_id,
            reviewer=reviewer,
            decision=ReviewStatus.APPROVED,
            reason="Review complete",
        )

        # Should succeed without validation even with wrong role
        updated = queue.apply_decision(decision)  # No guideline, no validation
        assert updated.status == ReviewStatus.APPROVED


class TestDeadLetterMechanism:
    """Test dead-letter handling for malformed JSONL entries."""

    def test_malformed_entry_moved_to_dead_letter(self, tmp_path):
        """Malformed entries should be moved to dead-letter file."""

        # Create a queue with malformed data
        data_dir = tmp_path / "test-queue"
        data_dir.mkdir()
        queue_file = data_dir / "queue.jsonl"

        # Write mixed valid and malformed data
        with open(queue_file, "w") as f:
            # Valid entry
            f.write('{"item_id": "valid-001", "source_id": "test", "audit_trail": []}\n')
            # Malformed entry (invalid JSON)
            f.write('{"item_id": "broken, "source_id": "test"}\n')
            # Another valid entry
            f.write('{"item_id": "valid-002", "source_id": "test", "audit_trail": []}\n')

        # Create queue - this will process the malformed entry
        queue = HumanReviewQueue(data_dir=data_dir)

        # Valid items should be loaded
        assert len(queue._items) == 2
        assert "valid-001" in queue._items or "valid-002" in queue._items

        # Dead-letter file should be created
        dead_letter_file = data_dir / "dead_letter.jsonl"
        assert dead_letter_file.exists()

        # Dead-letter should contain the malformed entry
        with open(dead_letter_file) as f:
            dead_letter_content = f.read()

        assert "broken" in dead_letter_content
        assert "line_number" in dead_letter_content
        assert "error" in dead_letter_content

    def test_dead_letter_contains_metadata(self, tmp_path):
        """Dead-letter entries should include metadata for debugging."""
        data_dir = tmp_path / "test-queue"
        data_dir.mkdir()
        queue_file = data_dir / "queue.jsonl"

        # Write malformed data
        with open(queue_file, "w") as f:
            f.write('{"incomplete": true\n')  # Missing closing brace

        # Create queue
        HumanReviewQueue(data_dir=data_dir)

        # Read dead-letter
        dead_letter_file = data_dir / "dead_letter.jsonl"
        with open(dead_letter_file) as f:
            entry = json.loads(f.read().strip())

        # Should have metadata
        assert "_meta" in entry
        assert "original_file" in entry["_meta"]
        assert "line_number" in entry["_meta"]
        assert "error" in entry["_meta"]
        assert "captured_at" in entry["_meta"]
        assert "raw_line" in entry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
