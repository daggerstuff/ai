"""Human review queue for borderline dataset items (PIX-250).

This module implements the human review lane for items that escalate through
Gates 1-3 of the privacy/content gates system. Items routed here require
explicit human reviewer decisions before they can be promoted to curation.

Review queue architecture
-------------------------
  Escalation   — PrivacyContentGates writes escalated items to the queue
  Persistence  — JSONL files with full audit trail per item
  Review API   — CLI and programmatic interfaces for reviewers
  Audit        — All reviewer decisions logged with timestamp and rationale

Usage
-----
  # Programmatic review
  queue = HumanReviewQueue()
  queue.enqueue(report)  # After ESCALATE from gates

  reviewer = Reviewer(id="reviewer-001", role="clinical")
  decision = reviewer.approve(item_id, reason="PII scrub verified acceptable")
  queue.apply_decision(decision)

  # CLI usage
  # uv run python -m ai.core.pipelines.human_review_queue list
  # uv run python -m ai.core.pipelines.human_review_queue approve ITEM_ID --reason "..."
"""

from __future__ import annotations

import contextlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class ReviewStatus(StrEnum):
    """Status of a review item."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    RETURNED = "returned"  # Sent back for additional information


class ReviewerRole(StrEnum):
    """Reviewer role types."""

    CLINICAL = "clinical"  # Clinical oversight for sensitive content
    PRIVACY = "privacy"  # Privacy/compliance review
    DATA_STEWARD = "data_steward"  # General data quality review
    SUPERVISOR = "supervisor"  # Override authority


@dataclass
class Reviewer:
    """Information about a reviewer."""

    id: str
    role: ReviewerRole
    name: str | None = None
    department: str | None = None


@dataclass
class ReviewItem:
    """An item awaiting human review."""

    item_id: str
    source_id: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Originalgate report data (simplified for storage)
    gate_result: dict[str, Any] | None = None
    escalation_reason: str = ""

    # Content excerpt (truncated for reviewer context)
    content_preview: str | None = None
    content_length: int = 0

    # Review state
    status: ReviewStatus = ReviewStatus.PENDING
    reviewed_at: str | None = None
    reviewer_id: str | None = None
    review_decision: str | None = None
    review_reason: str | None = None

    # Metadata
    priority: str = "normal"  # low, normal, high, urgent
    tags: list[str] = field(default_factory=list)

    # Audit trail
    audit_trail: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "source_id": self.source_id,
            "created_at": self.created_at,
            "gate_result": self.gate_result,
            "escalation_reason": self.escalation_reason,
            "content_preview": self.content_preview,
            "content_length": self.content_length,
            "status": self.status.value,
            "reviewed_at": self.reviewed_at,
            "reviewer_id": self.reviewer_id,
            "review_decision": self.review_decision,
            "review_reason": self.review_reason,
            "priority": self.priority,
            "tags": self.tags,
            "audit_trail": self.audit_trail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewItem:
        return cls(
            item_id=data["item_id"],
            source_id=data["source_id"],
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
            gate_result=data.get("gate_result"),
            escalation_reason=data.get("escalation_reason", ""),
            content_preview=data.get("content_preview"),
            content_length=data.get("content_length", 0),
            status=ReviewStatus(data.get("status", "pending")),
            reviewed_at=data.get("reviewed_at"),
            reviewer_id=data.get("reviewer_id"),
            review_decision=data.get("review_decision"),
            review_reason=data.get("review_reason"),
            priority=data.get("priority", "normal"),
            tags=data.get("tags", []),
            audit_trail=data.get("audit_trail", []),
        )


@dataclass
class ReviewDecision:
    """A reviewer's decision on an item."""

    item_id: str
    reviewer: Reviewer
    decision: ReviewStatus
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    additional_notes: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "reviewer_id": self.reviewer.id,
            "reviewer_role": self.reviewer.role.value,
            "decision": self.decision.value,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "additional_notes": self.additional_notes,
        }


# ---------------------------------------------------------------------------
# Escalation criteria — defines what triggers human review
# ---------------------------------------------------------------------------


@dataclass
class EscalationCriteria:
    """Defines the thresholds and conditions that trigger human review.

    These criteria determine which items from the privacy/content gates
    pipeline are routed to the human review lane rather than being
    auto-processed.

    Usage::

        criteria = EscalationCriteria(
            privacy_tier_escalates={PrivacyTier.HIGH, PrivacyTier.PROHIBITED},
            sensitivity_escalates={ContentSensitivity.RESTRICTED},
            min_pii_count_escalate=5,
            crisis_score_escalate_threshold=0.4,
            license_requires_review={"cc-by-nc-4.0", "cc-by-nc-sa-4.0"},
        )
        if criteria.should_escalate(gate_report):
            queue.enqueue(queue.create_item_from_report(...))
    """

    # Privacy tiers that always escalate
    privacy_tier_escalates: set[str] = field(default_factory=lambda: {"high", "prohibited"})

    # Content sensitivities that escalate
    sensitivity_escalates: set[str] = field(default_factory=lambda: {"restricted", "prohibited"})

    # PII count threshold for escalation (None to disable)
    min_pii_count_escalate: int | None = 5

    # Crisis score threshold for escalation (0.0 to 1.0)
    crisis_score_escalate_threshold: float | None = 0.4

    # Licenses that require review regardless of other factors
    license_requires_review: set[str] = field(default_factory=lambda: {"cc-by-nc-4.0", "cc-by-nc-sa-4.0"})

    # Gate decisions that trigger escalation
    gate_decision_escalates: set[str] = field(default_factory=lambda: {"escalate"})

    def should_escalate(self, gate_result: dict[str, Any]) -> bool:
        """Determine if a gate result should be escalated for review.

        Args:
            gate_result: The PrivacyContentReport.to_dict() result

        Returns:
            True if any escalation criterion is met
        """
        if not gate_result:
            return False

        # Check privacy tier
        privacy_tier = gate_result.get("privacy_tier", "").lower()
        if privacy_tier in self.privacy_tier_escalates:
            return True

        # Check content sensitivity
        sensitivity = gate_result.get("content_sensitivity", "").lower()
        if sensitivity in self.sensitivity_escalates:
            return True

        # Check PII count threshold
        if self.min_pii_count_escalate is not None:
            pii_findings = gate_result.get("pii_findings", [])
            total_pii = sum(f.get("count", 0) for f in pii_findings)
            if total_pii >= self.min_pii_count_escalate:
                return True

        # Check crisis score threshold
        if self.crisis_score_escalate_threshold is not None:
            crisis_findings = gate_result.get("crisis_findings", [])
            for finding in crisis_findings:
                score = finding.get("score", 0.0)
                if score >= self.crisis_score_escalate_threshold:
                    return True

        # Check license
        license_check = gate_result.get("license_check")
        if license_check:
            license_id = license_check.get("license_id", "").lower()
            if license_id in self.license_requires_review:
                return True
            # Also escalate if consent required but not recorded
            if license_check.get("requires_consent") and not license_check.get("consent_recorded"):
                return True

        # Check gate decisions for explicit escalation
        gates = gate_result.get("gates", {})
        for _gate_key, gate_data in gates.items():
            if not gate_data:
                continue
            decision = gate_data.get("decision", "").lower()
            if decision in self.gate_decision_escalates:
                return True

        return False

    def get_escalation_reasons(self, gate_result: dict[str, Any]) -> list[str]:
        """Return a list of all escalation reasons for a gate result.

        Args:
            gate_result: The PrivacyContentReport.to_dict() result

        Returns:
            List of human-readable escalation reasons
        """
        reasons = []

        if not gate_result:
            return reasons

        # Privacy tier
        privacy_tier = gate_result.get("privacy_tier", "").lower()
        if privacy_tier in self.privacy_tier_escalates:
            reasons.append(f"privacy_tier={privacy_tier}")

        # Content sensitivity
        sensitivity = gate_result.get("content_sensitivity", "").lower()
        if sensitivity in self.sensitivity_escalates:
            reasons.append(f"content_sensitivity={sensitivity}")

        # PII count
        if self.min_pii_count_escalate is not None:
            pii_findings = gate_result.get("pii_findings", [])
            total_pii = sum(f.get("count", 0) for f in pii_findings)
            if total_pii >= self.min_pii_count_escalate:
                reasons.append(f"pii_count={total_pii} (threshold={self.min_pii_count_escalate})")

        # Crisis score
        if self.crisis_score_escalate_threshold is not None:
            crisis_findings = gate_result.get("crisis_findings", [])
            for finding in crisis_findings:
                score = finding.get("score", 0.0)
                if score >= self.crisis_score_escalate_threshold:
                    crisis_type = finding.get("crisis_type", "unknown")
                    reasons.append(f"crisis_score={score:.2f} ({crisis_type})")

        # License
        license_check = gate_result.get("license_check")
        if license_check:
            license_id = license_check.get("license_id", "").lower()
            if license_id in self.license_requires_review:
                reasons.append(f"license_requires_review={license_id}")
            if license_check.get("requires_consent") and not license_check.get("consent_recorded"):
                reasons.append("consent_required_but_not_recorded")

        # Gate decisions
        gates = gate_result.get("gates", {})
        for gate_key, gate_data in gates.items():
            if not gate_data:
                continue
            decision = gate_data.get("decision", "").lower()
            if decision in self.gate_decision_escalates:
                reason_text = gate_data.get("reason", "escalated")
                reasons.append(f"{gate_key}: {reason_text}")

        return reasons


# ---------------------------------------------------------------------------
# Review consistency guidelines — ensures similar cases get similar treatment
# ---------------------------------------------------------------------------


@dataclass
class ReviewConsistencyGuideline:
    """Defines rules for consistent review decisions across similar cases.

    These guidelines help ensure that reviewers make consistent decisions
    for cases with similar characteristics, reducing variance and improving
    reliability of the review lane.

    Usage::

        guidelines = ReviewConsistencyGuideline(
            auto_approve_conditions=[
                {
                    "condition": "privacy_tier == 'none'",
                    "requires": ["all_gates_pass"],
                },
            ],
            auto_reject_conditions=[
                {
                    "condition": "content_sensitivity == 'prohibited'",
                    "reason": "Prohibited content cannot be approved",
                },
            ],
            role_criteria={
                ReviewerRole.CLINICAL: ["crisis_findings", "sensitivity_restricted"],
                ReviewerRole.PRIVACY: ["pii_findings", "privacy_tier_high"],
            },
        )
    """

    # Conditions that typically warrant auto-approval (after human review)
    auto_approve_conditions: list[dict[str, Any]] = field(default_factory=list)

    # Conditions that typically warrant rejection
    auto_reject_conditions: list[dict[str, Any]] = field(default_factory=list)

    # Criteria mapping reviewer roles to required expertise areas
    role_criteria: dict[str, list[str]] = field(
        default_factory=lambda: {
            "clinical": ["crisis_findings", "sensitivity_restricted", "sensitivity_sensitive"],
            "privacy": ["pii_findings", "privacy_tier_high", "privacy_tier_medium"],
            "data_steward": ["quality", "completeness", "license"],
            "supervisor": ["override", "escalation", "complex_cases"],
        }
    )

    # Reviewer role requirements by case type
    required_role_by_case_type: dict[str, str] = field(
        default_factory=lambda: {
            "crisis": "clinical",
            "privacy_high": "privacy",
            "license_exception": "data_steward",
            "prohibited": "supervisor",
        }
    )

    def get_suggested_review_action(self, gate_result: dict[str, Any], review_item: ReviewItem) -> str | None:
        """Suggest a review action based on consistent guidelines.

        Args:
            gate_result: The gate report dict
            review_item: The review item being evaluated

        Returns:
            Suggested action ("approve", "reject", "return", "escalate_to_supervisor")
            or None if no clear guideline applies
        """
        # Check auto-reject conditions first (higher priority)
        for condition in self.auto_reject_conditions:
            if self._matches_condition(gate_result, condition):
                return "reject"

        # Check auto-approve conditions
        for condition in self.auto_approve_conditions:
            if self._matches_condition(gate_result, condition):
                return "approve"

        # Check if supervisor escalation is required
        sensitivity = gate_result.get("content_sensitivity", "").lower()
        if sensitivity == "prohibited":
            return "escalate_to_supervisor"

        privacy_tier = gate_result.get("privacy_tier", "").lower()
        if privacy_tier == "prohibited":
            return "escalate_to_supervisor"

        return None

    def get_required_reviewer_role(self, gate_result: dict[str, Any]) -> ReviewerRole:
        """Determine the appropriate reviewer role for a case.

        Args:
            gate_result: The gate report dict

        Returns:
            Recommended ReviewerRole for this case
        """
        # Crisis cases require clinical review
        crisis_findings = gate_result.get("crisis_findings", [])
        if crisis_findings:
            for finding in crisis_findings:
                if finding.get("requires_escalation", False):
                    return ReviewerRole.CLINICAL

        # Check sensitivity
        sensitivity = gate_result.get("content_sensitivity", "").lower()
        if sensitivity in ("restricted", "prohibited"):
            return ReviewerRole.CLINICAL

        # High privacy tier requires privacy review
        privacy_tier = gate_result.get("privacy_tier", "").lower()
        if privacy_tier == "high":
            return ReviewerRole.PRIVACY

        # NC licenses may need data steward
        license_check = gate_result.get("license_check")
        if license_check:
            license_id = license_check.get("license_id", "").lower()
            if "cc-by-nc" in license_id:
                return ReviewerRole.DATA_STEWARD

        # Default to data steward for routine cases
        return ReviewerRole.DATA_STEWARD

    def _matches_condition(self, gate_result: dict[str, Any], condition: dict[str, Any]) -> bool:
        """Check if a gate result matches a condition."""
        # Simple condition matching based on field equality
        for key, expected in condition.items():
            if key == "condition":
                continue  # Skip meta-field
            actual = gate_result.get(key)
            if isinstance(expected, str):
                expected = expected.lower()
            if isinstance(actual, str):
                actual = actual.lower()
            if actual != expected:
                return False
        return True

    def get_review_checklist(self, gate_result: dict[str, Any]) -> list[str]:
        """Generate a reviewer checklist for a case.

        Args:
            gate_result: The gate report dict

        Returns:
            List of items for the reviewer to verify
        """
        checklist = []

        # PII checklist
        pii_findings = gate_result.get("pii_findings", [])
        if pii_findings:
            checklist.append(f"Verify PII scrubbing for: {', '.join(f['pii_type'] for f in pii_findings)}")

        # Crisis checklist
        crisis_findings = gate_result.get("crisis_findings", [])
        for finding in crisis_findings:
            if finding.get("requires_escalation"):
                checklist.append(
                    f"Clinical review required: {finding.get('crisis_type', 'unknown')} (score={finding.get('score', 0):.2f})"
                )

        # License checklist
        license_check = gate_result.get("license_check")
        if license_check:
            if license_check.get("requires_consent") and not license_check.get("consent_recorded"):
                checklist.append("Consent verification required")
            if "cc-by-nc" in license_check.get("license_id", "").lower():
                checklist.append("NC license: verify acceptable use case")

        # Privacy tier checklist
        privacy_tier = gate_result.get("privacy_tier", "").lower()
        if privacy_tier == "high":
            checklist.append("High privacy tier: verify no identifying information remains")

        return checklist


# ---------------------------------------------------------------------------
# Review feedback collector — aggregates decisions for pipeline improvement
# ---------------------------------------------------------------------------


@dataclass
class ReviewFeedback:
    """Aggregated feedback from review decisions."""

    total_reviews: int = 0
    approval_rate: float = 0.0
    rejection_rate: float = 0.0
    return_rate: float = 0.0
    avg_queue_time_hours: float | None = None
    reviews_by_reviewer: dict[str, int] = field(default_factory=dict)
    reviews_by_reason: dict[str, int] = field(default_factory=dict)
    common_approval_reasons: list[str] = field(default_factory=list)
    common_rejection_reasons: list[str] = field(default_factory=list)
    escalation_patterns: list[str] = field(default_factory=list)
    guidelines_feedback: dict[str, Any] = field(default_factory=dict)


class ReviewFeedbackCollector:
    """Collects and aggregates review decisions for pipeline feedback.

    This collector tracks patterns in review decisions to:
    - Identify systematic issues upstream
    - Improve escalation criteria
    - Refine consistency guidelines
    - Feed insights back into the pipeline

    Usage::

        collector = ReviewFeedbackCollector()
        collector.record_decision(review_decision)

        # Generate periodic reports
        feedback = collector.get_feedback()
        if feedback.approval_rate < 0.5:
            # Many rejections suggest upstream filtering is too permissive
            update_gate_thresholds()
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialize the feedback collector.

        Args:
            data_dir: Directory for feedback storage. Defaults to data/review-feedback.
        """
        self.data_dir = data_dir or Path("data/review-feedback")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._feedback_file = self.data_dir / "feedback.jsonl"

        self._decisions: list[dict[str, Any]] = []
        self._load_decisions()

    def _load_decisions(self) -> None:
        """Load previously recorded decisions."""
        if not self._feedback_file.exists():
            return

        with open(self._feedback_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                with contextlib.suppress(json.JSONDecodeError):
                    self._decisions.append(json.loads(line))

    def _persist_decision(self, decision: dict[str, Any]) -> None:
        """Append a decision to persistent storage."""
        with open(self._feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision, ensure_ascii=False) + "\n")

    def record_decision(self, decision: ReviewDecision | dict[str, Any]) -> None:
        """Record a review decision for analysis.

        Args:
            decision: ReviewDecision instance or dict representation
        """
        decision_dict = decision.to_dict() if isinstance(decision, ReviewDecision) else decision

        self._decisions.append(decision_dict)
        self._persist_decision(decision_dict)

    def record_batch(self, decisions: list[ReviewDecision | dict[str, Any]]) -> None:
        """Record multiple review decisions.

        Args:
            decisions: List of ReviewDecision instances or dicts
        """
        for decision in decisions:
            self.record_decision(decision)

    def get_feedback(self, lookback_days: int | None = None) -> ReviewFeedback:
        """Generate aggregated feedback from recorded decisions.

        Args:
            lookback_days: If set, only consider decisions within this many days

        Returns:
            Aggregated feedback statistics
        """
        feedback = ReviewFeedback()

        if not self._decisions:
            return feedback

        # Filter by lookback if specified
        decisions = self._decisions
        if lookback_days is not None:
            from datetime import timedelta

            cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
            decisions = []
            for d in self._decisions:
                try:
                    decision_time = datetime.fromisoformat(d["timestamp"])
                    if decision_time >= cutoff:
                        decisions.append(d)
                except (KeyError, ValueError):
                    pass

        if not decisions:
            return feedback

        feedback.total_reviews = len(decisions)

        # Calculate rates
        approved = sum(1 for d in decisions if d["decision"] == "approved")
        rejected = sum(1 for d in decisions if d["decision"] == "rejected")
        returned = sum(1 for d in decisions if d["decision"] == "returned")

        if decisions:
            feedback.approval_rate = approved / len(decisions)
            feedback.rejection_rate = rejected / len(decisions)
            feedback.return_rate = returned / len(decisions)

        # Aggregate by reviewer
        for d in decisions:
            reviewer_id = d.get("reviewer_id", "unknown")
            feedback.reviews_by_reviewer[reviewer_id] = feedback.reviews_by_reviewer.get(reviewer_id, 0) + 1

        # Aggregate reasons
        reason_approvals: dict[str, int] = {}
        reason_rejections: dict[str, int] = {}
        for d in decisions:
            reason = d.get("reason", "no_reason")
            # Normalize reason for aggregation
            normalized_reason = self._normalize_reason(reason)
            if d["decision"] == "approved":
                reason_approvals[normalized_reason] = reason_approvals.get(normalized_reason, 0) + 1
            elif d["decision"] == "rejected":
                reason_rejections[normalized_reason] = reason_rejections.get(normalized_reason, 0) + 1

        # Top approval reasons
        feedback.common_approval_reasons = [r for r, _ in sorted(reason_approvals.items(), key=lambda x: -x[1])[:5]]
        # Top rejection reasons
        feedback.common_rejection_reasons = [r for r, _ in sorted(reason_rejections.items(), key=lambda x: -x[1])[:5]]

        # Identify escalation patterns
        feedback.escalation_patterns = self._identify_patterns(decisions)

        # Calculate average queue time
        queue_times = []
        for d in decisions:
            # Would need original item data to calculate queue time
            # This is a placeholder for when we track item creation time
            pass
        feedback.avg_queue_time_hours = sum(queue_times) / len(queue_times) if queue_times else None

        # Generate guidelines feedback
        feedback.guidelines_feedback = self._generate_guidelines_feedback(decisions)

        return feedback

    def get_gate_adjustment_recommendations(self) -> dict[str, Any]:
        """Analyze decisions to recommend gate threshold adjustments.

        Returns:
            Recommendations for adjusting escalation criteria
        """
        recommendations = {
            "adjust_privacy_thresholds": False,
            "adjust_crisis_thresholds": False,
            "new_license_review_required": [],
            "removed_license_review_required": [],
        }

        if len(self._decisions) < 10:
            # Not enough data for recommendations
            return recommendations

        # Analyze approval rate by privacy tier
        # This would require correlating decision data with gate results
        # Placeholder implementation
        approval_rate = sum(1 for d in self._decisions if d["decision"] == "approved") / len(self._decisions)

        # If approval rate is very low (< 20%), thresholds may be too permissive
        if approval_rate < 0.2:
            recommendations["adjust_privacy_thresholds"] = True
            recommendations["adjust_crisis_thresholds"] = True

        return recommendations

    def _normalize_reason(self, reason: str) -> str:
        normalized = reason.lower().strip()
        normalized = " ".join(normalized.split())
        if len(normalized) > 100:
            normalized = normalized[:100] + "..."
        return normalized

    def _identify_patterns(self, decisions: list[dict[str, Any]]) -> list[str]:
        """Identify patterns in review decisions."""
        patterns = []

        if not decisions:
            return patterns

        # Pattern: High rejection rate
        rejected = sum(1 for d in decisions if d["decision"] == "rejected")
        rejection_rate = rejected / len(decisions)
        if rejection_rate > 0.5:
            patterns.append(f"High rejection rate: {rejection_rate:.1%}")

        # Pattern: Same reason repeated
        reason_counts: dict[str, int] = {}
        for d in decisions:
            reason = self._normalize_reason(d.get("reason", ""))
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        for reason, count in reason_counts.items():
            if count >= len(decisions) * 0.3 and count >= 3:
                patterns.append(f"Common rejection reason: '{reason}' ({count} times)")

        return patterns

    def _generate_guidelines_feedback(self, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate feedback for improving consistency guidelines."""
        feedback = {
            "auto_approve_patterns": [],
            "auto_reject_patterns": [],
            "reviewer_role_suggestions": {},
        }

        if not decisions:
            return feedback

        # Analyze what conditions lead to approvals
        approved_decisions = [d for d in decisions if d["decision"] == "approved"]
        [d for d in decisions if d["decision"] == "rejected"]

        # Simple pattern detection based on reason keywords
        approval_keywords: dict[str, int] = {}
        for d in approved_decisions:
            reason = d.get("reason", "").lower()
            for keyword in ["verified", "acceptable", "appropriate", "confirmed"]:
                if keyword in reason:
                    approval_keywords[keyword] = approval_keywords.get(keyword, 0) + 1

        if approval_keywords:
            feedback["auto_approve_patterns"] = [
                f"'{kw}' appears in {count} approvals"
                for kw, count in sorted(approval_keywords.items(), key=lambda x: -x[1])[:3]
            ]

        return feedback


class HumanReviewQueue:
    """Manages the human review queue for borderline dataset items.

    The queue persists items in JSONL format, one item per line, allowing
    incremental updates and easy auditing.

    Usage::

        queue = HumanReviewQueue(data_dir=Path("data/review-queue"))

        # Add item from gate report
        item = queue.create_item_from_report(report, content_preview=text[:500])
        queue.enqueue(item)

        # List pending items
        pending = queue.list_items(status=ReviewStatus.PENDING)

        # Process a decision
        reviewer = Reviewer(id="clinical-001", role=ReviewerRole.CLINICAL)
        decision = ReviewDecision(
            item_id=item.item_id,
            reviewer=reviewer,
            decision=ReviewStatus.APPROVED,
            reason="PII scrub verified; content appropriate for training"
        )
        queue.apply_decision(decision)
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        """Initialize the review queue.

        Args:
            data_dir: Directory for queue storage. Defaults to data/review-queue
                     relative to project root.
        """
        self.data_dir = Path(data_dir) if data_dir is not None else Path("data/review-queue")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._queue_file = self.data_dir / "queue.jsonl"
        self._audit_file = self.data_dir / "audit.jsonl"
        self._items: dict[str, ReviewItem] = {}

        self._load_items()

    def _load_items(self) -> None:
        """Load items from persistent storage."""
        if not self._queue_file.exists():
            return

        malformed_count = 0
        with open(self._queue_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    item = ReviewItem.from_dict(data)
                    self._items[item.item_id] = item
                except (json.JSONDecodeError, KeyError) as e:
                    # Track malformed entries for dead-letter handling
                    malformed_count += 1
                    self._handle_malformed_entry(line, line_num, str(e))

        # Report if any entries were moved to dead-letter
        if malformed_count > 0:
            pass

    def _handle_malformed_entry(self, line: str, line_num: int, error: str) -> None:
        """Move malformed entry to dead-letter file for analysis.

        Args:
            line: The malformed line content
            line_num: Line number in the original file
            error: Error message describing the malformation
        """
        dead_letter_file = self.data_dir / "dead_letter.jsonl"

        # Create dead-letter entry with context
        dead_letter_entry = {
            "_meta": {
                "original_file": str(self._queue_file),
                "line_number": line_num,
                "error": error,
                "captured_at": datetime.now(UTC).isoformat(),
            },
            "raw_line": line,
        }

        # Append to dead-letter file
        with open(dead_letter_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(dead_letter_entry, ensure_ascii=False) + "\n")

    def _persist_items(self) -> None:
        """Persist all items to storage."""
        with open(self._queue_file, "w", encoding="utf-8") as f:
            for item in self._items.values():
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    def _persist_audit(self, decision: ReviewDecision) -> None:
        """Append a decision to the audit log."""
        with open(self._audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision.to_dict(), ensure_ascii=False) + "\n")

    def create_item_from_report(
        self,
        source_id: str,
        gate_result: dict[str, Any],
        content_preview: str | None = None,
        content_length: int = 0,
        priority: str = "normal",
    ) -> ReviewItem:
        """Create a review item from a gate report.

        Args:
            source_id: The original source identifier
            gate_result: The PrivacyContentReport.to_dict() result
            content_preview: Truncated content for reviewer context
            content_length: Full content length
            priority: Priority level (low, normal, high, urgent)

        Returns:
            The created ReviewItem
        """
        # Determine escalation reason from gate results
        escalation_reason = self._extract_escalation_reason(gate_result)

        item = ReviewItem(
            item_id=f"review-{uuid.uuid4().hex[:12]}",
            source_id=source_id,
            gate_result=gate_result,
            escalation_reason=escalation_reason,
            content_preview=content_preview,
            content_length=content_length,
            priority=priority,
            tags=self._extract_tags(gate_result),
        )

        # Add initial audit entry
        item.audit_trail.append(
            {
                "event": "created",
                "timestamp": datetime.now(UTC).isoformat(),
                "details": f"Escalated from privacy/content gates: {escalation_reason}",
            }
        )

        return item

    def _extract_escalation_reason(self, gate_result: dict[str, Any]) -> str:
        """Extract human-readable escalation reason from gate result."""
        if not gate_result:
            return "Unknown escalation"

        gates = gate_result.get("gates", {})
        reasons = []

        # Check Gate 1 (PII)
        if (g1 := gates.get("gate1")) and g1.get("decision") == "escalate":
            reasons.append(f"G1 (PII): {g1.get('reason', '')}")

        # Check Gate 2 (Safety)
        if (g2 := gates.get("gate2")) and g2.get("decision") == "escalate":
            reasons.append(f"G2 (Safety): {g2.get('reason', '')}")

        # Check Gate 3 (License)
        if (g3 := gates.get("gate3")) and g3.get("decision") == "escalate":
            reasons.append(f"G3 (License): {g3.get('reason', '')}")

        return "; ".join(reasons) if reasons else "Manual review required"

    def _extract_tags(self, gate_result: dict[str, Any]) -> list[str]:
        """Extract tags from gate result for sorting/filtering."""
        tags = []

        if not gate_result:
            return tags

        # Tag by privacy tier
        if tier := gate_result.get("privacy_tier"):
            tags.append(f"tier-{tier}")

        # Tag by sensitivity
        if (sensitivity := gate_result.get("content_sensitivity")) and sensitivity != "normal":
            tags.append(f"sensitivity-{sensitivity}")

        # Tag by finding type
        if pii_findings := gate_result.get("pii_findings", []):
            for finding in pii_findings:
                pii_type = finding.get("pii_type", "unknown")
                tags.append(f"pii-{pii_type}")

        if crisis_findings := gate_result.get("crisis_findings", []):
            for finding in crisis_findings:
                crisis_type = finding.get("crisis_type", "unknown")
                tags.append(f"crisis-{crisis_type}")

        return list(set(tags))

    def enqueue(self, item: ReviewItem) -> None:
        """Add an item to the review queue."""
        if item.item_id in self._items:
            raise ValueError(f"Item {item.item_id} already in queue")

        self._items[item.item_id] = item
        self._persist_items()

    def dequeueReusable(self) -> ReviewItem | None:
        """Get the next pending item (FIFO order)."""
        pending = [item for item in self._items.values() if item.status == ReviewStatus.PENDING]

        if not pending:
            return None

        # Sort by priority then creation time
        priority_order = {"urgent": 0, "high": 1, "normal": 2, "low": 3}
        pending.sort(key=lambda x: (priority_order.get(x.priority, 2), x.created_at))

        return pending[0]

    def list_items(
        self,
        status: ReviewStatus | None = None,
        reviewer_id: str | None = None,
        priority: str | None = None,
        tags: list[str] | None = None,
    ) -> list[ReviewItem]:
        """List items with optional filters."""
        items = list(self._items.values())

        if status:
            items = [i for i in items if i.status == status]

        if reviewer_id:
            items = [i for i in items if i.reviewer_id == reviewer_id]

        if priority:
            items = [i for i in items if i.priority == priority]

        if tags:
            items = [i for i in items if any(t in i.tags for t in tags)]

        return items

    def get_item(self, item_id: str) -> ReviewItem | None:
        """Get a specific item by ID."""
        return self._items.get(item_id)

    def apply_decision(
        self,
        decision: ReviewDecision,
        consistency_guideline: ReviewConsistencyGuideline | None = None,
        validate_role: bool = False,
    ) -> ReviewItem:
        """Apply a reviewer's decision to an item.

        Args:
            decision: The review decision to apply
            consistency_guideline: Optional guideline for role validation
            validate_role: If True, validate reviewer role against guideline

        Returns:
            The updated ReviewItem

        Raises:
            ValueError: If item not found, invalid decision state transition,
                        or role validation fails
        """
        item = self._items.get(decision.item_id)
        if not item:
            raise ValueError(f"Item {decision.item_id} not found in queue")

        if item.status != ReviewStatus.PENDING:
            raise ValueError(f"Item {decision.item_id} already has status {item.status.value}, cannot apply decision")

        # Optional role validation (PIX-250 role consistency)
        if validate_role and consistency_guideline is not None and item.gate_result:
            required_role = consistency_guideline.get_required_reviewer_role(item.gate_result)
            if decision.reviewer.role != required_role:
                raise ValueError(
                    f"Reviewer role '{decision.reviewer.role.value}' does not match "
                    f"required role '{required_role.value}' for this case. "
                    f"Case requires {required_role.value} review based on gate analysis."
                )

        # Update item state
        item.status = decision.decision
        item.reviewed_at = decision.timestamp
        item.reviewer_id = decision.reviewer.id
        item.review_decision = decision.decision.value
        item.review_reason = decision.reason

        # Add audit trail entry with full decision details
        audit_entry: dict[str, Any] = {
            "event": "decision",
            "timestamp": decision.timestamp,
            "reviewer_id": decision.reviewer.id,
            "reviewer_role": decision.reviewer.role.value,
            "decision": decision.decision.value,
            "reason": decision.reason,
        }
        # Include additional_notes if present (PIX-250 audit completeness)
        if decision.additional_notes:
            audit_entry["additional_notes"] = decision.additional_notes

        item.audit_trail.append(audit_entry)

        # Persist changes
        self._persist_items()
        self._persist_audit(decision)

        return item

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        total = len(self._items)
        by_status = {
            status.value: len([i for i in self._items.values() if i.status == status]) for status in ReviewStatus
        }
        by_priority = {}
        for item in self._items.values():
            by_priority[item.priority] = by_priority.get(item.priority, 0) + 1

        # Calculate average queue time for completed items
        completed = [i for i in self._items.values() if i.reviewed_at]
        avg_queue_time_hours = None
        if completed:
            times = []
            for item in completed:
                created = datetime.fromisoformat(item.created_at)
                if item.reviewed_at:  # Type guard for Pyright
                    reviewed = datetime.fromisoformat(item.reviewed_at)
                    times.append((reviewed - created).total_seconds() / 3600)
            avg_queue_time_hours = sum(times) / len(times) if times else None

        return {
            "total_items": total,
            "by_status": by_status,
            "by_priority": by_priority,
            "pending_count": by_status.get("pending", 0),
            "approved_count": by_status.get("approved", 0),
            "rejected_count": by_status.get("rejected", 0),
            "avg_queue_time_hours": round(avg_queue_time_hours, 2) if avg_queue_time_hours else None,
        }


# ---------------------------------------------------------------------------
# Metrics export — Prometheus-compatible metrics for observability
# ---------------------------------------------------------------------------


@dataclass
class QueueMetrics:
    """Metrics exported from the human review queue."""

    # Counters
    items_enqueued_total: int = 0
    items_processed_total: int = 0
    approvals_total: int = 0
    rejections_total: int = 0
    returned_total: int = 0

    # Gauges
    queue_depth_pending: int = 0
    queue_depth_by_priority: dict[str, int] = field(default_factory=dict)
    queue_depth_by_tag: dict[str, int] = field(default_factory=dict)

    # Histograms (stored as buckets)
    queue_time_seconds_buckets: dict[str, int] = field(
        default_factory=lambda: {
            "1h": 0,
            "4h": 0,
            "12h": 0,
            "24h": 0,
            "72h": 0,
            "infinite": 0,
        }
    )
    queue_time_seconds_sum: float = 0.0
    queue_time_seconds_count: int = 0

    # Observability
    last_export_timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_prometheus_format(self) -> str:
        """Export metrics in Prometheus exposition format."""
        lines = []

        # Helper to add metric
        def add_metric(name: str, value: float, help_text: str, labels: str = ""):
            if labels:
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{{{labels}}} {value}")
            else:
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")

        # Helper for counters
        def add_counter(name: str, value: float, help_text: str):
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        # Counters
        add_counter(
            "human_review_items_enqueued_total", self.items_enqueued_total, "Total items enqueued for human review"
        )
        add_counter(
            "human_review_items_processed_total", self.items_processed_total, "Total items processed (resolved)"
        )
        add_counter("human_review_approvals_total", self.approvals_total, "Total items approved")
        add_counter("human_review_rejections_total", self.rejections_total, "Total items rejected")
        add_counter("human_review_returned_total", self.returned_total, "Total items returned for additional info")

        # Gauges
        add_metric("human_review_queue_depth_pending", self.queue_depth_pending, "Current pending items in queue")

        for priority, count in self.queue_depth_by_priority.items():
            add_metric(
                "human_review_queue_depth_by_priority", count, "Queue depth by priority", f'priority="{priority}"'
            )

        for tag, count in self.queue_depth_by_tag.items():
            # Truncate tag for prometheus label compliance
            safe_tag = tag.replace("-", "_").replace(" ", "_")[:63]
            add_metric("human_review_queue_depth_by_tag", count, "Queue depth by tag", f'tag="{safe_tag}"')

        # Queue time histogram
        for bucket, count in self.queue_time_seconds_buckets.items():
            add_metric(
                "human_review_queue_time_seconds_bucket", count, "Time items spend in queue (hours)", f'le="{bucket}"'
            )
        add_metric("human_review_queue_time_seconds_sum", self.queue_time_seconds_sum, "Sum of queue times in hours")
        add_metric(
            "human_review_queue_time_seconds_count",
            self.queue_time_seconds_count,
            "Count of items with queue time data",
        )

        return "\n".join(lines)


class QueueMetricsExporter:
    """Exports queue metrics in various formats for monitoring."""

    def __init__(self, queue: HumanReviewQueue, feedback_collector: ReviewFeedbackCollector | None = None):
        self.queue = queue
        self.feedback_collector = feedback_collector
        self._export_history: list[QueueMetrics] = []
        self._items_processed_cache: set[str] = set()
        self._items_enqueued_cache: set[str] = set()

    def collect_metrics(self) -> QueueMetrics:
        """Collect current queue metrics."""
        stats = self.queue.get_stats()

        metrics = QueueMetrics()

        # Counters - track new items since last export
        all_items = self.queue.list_items()

        # New enqueued items
        new_enqueued = [i for i in all_items if i.item_id not in self._items_enqueued_cache]
        metrics.items_enqueued_total = len(new_enqueued)
        self._items_enqueued_cache.update(i.item_id for i in all_items)

        # Processed items (approved, rejected, returned)
        new_processed = [
            i
            for i in all_items
            if i.item_id not in self._items_processed_cache
            and i.status in (ReviewStatus.APPROVED, ReviewStatus.REJECTED, ReviewStatus.RETURNED)
        ]
        metrics.items_processed_total = len(new_processed)

        for item in new_processed:
            if item.status == ReviewStatus.APPROVED:
                metrics.approvals_total += 1
            elif item.status == ReviewStatus.REJECTED:
                metrics.rejections_total += 1
            elif item.status == ReviewStatus.RETURNED:
                metrics.returned_total += 1
            self._items_processed_cache.add(item.item_id)

        # Gauges - current state
        metrics.queue_depth_pending = stats.get("pending_count", 0)
        metrics.queue_depth_by_priority = stats.get("by_priority", {})

        # Queue time histogram - calculate from completed items
        completed = [i for i in all_items if i.reviewed_at]
        times_hours = []
        for item in completed:
            if not item.reviewed_at:
                continue
            try:
                created = datetime.fromisoformat(item.created_at)
                reviewed = datetime.fromisoformat(item.reviewed_at)
                hours = (reviewed - created).total_seconds() / 3600
                times_hours.append(hours)
            except ValueError:
                pass

        if times_hours:
            metrics.queue_time_seconds_sum = sum(times_hours)
            metrics.queue_time_seconds_count = len(times_hours)

            # Bucket counts (buckets are in hours)
            buckets = [1, 4, 12, 24, 72]
            for h in times_hours:
                for _i, bucket in enumerate(buckets):
                    if h <= bucket:
                        metrics.queue_time_seconds_buckets[f"{bucket}h"] += 1
                metrics.queue_time_seconds_buckets["infinite"] += 1

        # Feedback stats if available
        if self.feedback_collector:
            self.feedback_collector.get_feedback(lookback_days=7)
            # Could add additional metrics from feedback here

        return metrics

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = self.collect_metrics()
        self._export_history.append(metrics)
        return metrics.to_prometheus_format()

    def export_json(self) -> dict[str, Any]:
        """Export metrics as JSON for dashboard integration."""
        metrics = self.collect_metrics()
        self._export_history.append(metrics)

        return {
            "timestamp": metrics.last_export_timestamp,
            "counters": {
                "items_enqueued_total": metrics.items_enqueued_total,
                "items_processed_total": metrics.items_processed_total,
                "approvals_total": metrics.approvals_total,
                "rejections_total": metrics.rejections_total,
                "returned_total": metrics.returned_total,
            },
            "gauges": {
                "queue_depth_pending": metrics.queue_depth_pending,
                "queue_depth_by_priority": metrics.queue_depth_by_priority,
                "queue_depth_by_tag": metrics.queue_depth_by_tag,
            },
            "histograms": {
                "queue_time_seconds_sum": metrics.queue_time_seconds_sum,
                "queue_time_seconds_count": metrics.queue_time_seconds_count,
                "queue_time_seconds_buckets": metrics.queue_time_seconds_buckets,
            },
        }

    def check_alerts(self, thresholds: dict[str, float] | None = None) -> list[dict[str, Any]]:
        """Check metrics against alert thresholds.

        Args:
            thresholds: Optional custom thresholds. Defaults:
                - queue_depth_pending: 100
                - avg_queue_time_hours: 24
                - rejection_rate: 0.3

        Returns:
            List of triggered alerts
        """
        if thresholds is None:
            thresholds = {
                "queue_depth_pending": 100,
                "avg_queue_time_hours": 24,
                "rejection_rate": 0.3,
            }

        alerts = []
        metrics = self.collect_metrics()

        # Check pending queue depth
        if metrics.queue_depth_pending > thresholds.get("queue_depth_pending", 100):
            alerts.append(
                {
                    "severity": "warning" if metrics.queue_depth_pending < 200 else "critical",
                    "metric": "queue_depth_pending",
                    "value": metrics.queue_depth_pending,
                    "threshold": thresholds["queue_depth_pending"],
                    "message": f"Human review queue depth ({metrics.queue_depth_pending}) exceeds threshold ({thresholds['queue_depth_pending']})",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        # Check average queue time
        if metrics.queue_time_seconds_count > 0:
            avg_hours = metrics.queue_time_seconds_sum / metrics.queue_time_seconds_count
            if avg_hours > thresholds.get("avg_queue_time_hours", 24):
                alerts.append(
                    {
                        "severity": "warning" if avg_hours < 48 else "critical",
                        "metric": "avg_queue_time_hours",
                        "value": avg_hours,
                        "threshold": thresholds["avg_queue_time_hours"],
                        "message": f"Average queue time ({avg_hours:.1f}h) exceeds threshold ({thresholds['avg_queue_time_hours']}h)",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )

        # Check rejection rate
        total_processed = metrics.approvals_total + metrics.rejections_total
        if total_processed > 0:
            rejection_rate = metrics.rejections_total / total_processed
            if rejection_rate > thresholds.get("rejection_rate", 0.3):
                alerts.append(
                    {
                        "severity": "warning",
                        "metric": "rejection_rate",
                        "value": rejection_rate,
                        "threshold": thresholds["rejection_rate"],
                        "message": f"Review rejection rate ({rejection_rate:.1%}) exceeds threshold ({thresholds['rejection_rate']:.1%})",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )

        return alerts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for review queue operations."""
    import argparse

    parser = argparse.ArgumentParser(description="Human Review Queue CLI")
    parser.add_argument("command", choices=["list", "approve", "reject", "stats", "metrics", "alerts"])
    parser.add_argument("--item-id", help="Item ID for approve/reject")
    parser.add_argument("--reason", help="Review reason (required for approve/reject)")
    parser.add_argument("--reviewer-id", default="cli-reviewer", help="Reviewer ID")
    parser.add_argument("--status-filter", help="Filter by status")
    parser.add_argument("--format", choices=["json", "prometheus"], help="Output format for metrics")
    parser.add_argument("--alert-thresholds", help="JSON string for custom alert thresholds")

    args = parser.parse_args()

    queue = HumanReviewQueue()

    if args.command == "list":
        status = None
        if args.status_filter:
            status = ReviewStatus(args.status_filter)

        items = queue.list_items(status=status)

        if not items:
            return

        for item in items[:20]:  # Show first 20
            pass

        if len(items) > 20:
            pass

    elif args.command == "stats":
        stats = queue.get_stats()
        if stats["avg_queue_time_hours"]:
            pass

    elif args.command == "metrics":
        exporter = QueueMetricsExporter(queue)
        format_type = getattr(args, "format", "json") or "json"

        if format_type == "prometheus":
            pass
        else:
            exporter.export_json()

    elif args.command == "alerts":
        exporter = QueueMetricsExporter(queue)
        thresholds = None
        if args.alert_thresholds:
            import json as _json

            try:
                thresholds = _json.loads(args.alert_thresholds)
            except _json.JSONDecodeError:
                return
        alerts = exporter.check_alerts(thresholds)

        if not alerts:
            pass
        else:
            for alert in alerts:
                alert.get("severity", "unknown").upper()

    elif args.command in ("approve", "reject"):
        if not args.item_id:
            return

        if not args.reason:
            return

        item = queue.get_item(args.item_id)
        if not item:
            return

        reviewer = Reviewer(id=args.reviewer_id, role=ReviewerRole.DATA_STEWARD)
        decision = ReviewStatus.APPROVED if args.command == "approve" else ReviewStatus.REJECTED

        review_decision = ReviewDecision(
            item_id=args.item_id,
            reviewer=reviewer,
            decision=decision,
            reason=args.reason,
        )

        queue.apply_decision(review_decision)


if __name__ == "__main__":
    main()
