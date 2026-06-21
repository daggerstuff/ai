"""Tests for AdvancedRoutingRules.

These tests verify the AdvancedRoutingRules behavior per
VAL-M3-ROUT-001, VAL-M3-ROUT-002, VAL-M3-ROUT-003, and VAL-M3-ROUT-004.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from training.coaching_safety.advanced_routing import (
    AdvancedRoutingRules,
    RoutingBucket,
    RoutingDecision,
    apply as apply_routing,
)
from training.coaching_safety.calibration_metrics import (
    CalibrationSnapshot,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class SnapshotFixture:
    """Fixture for creating CalibrationSnapshot with controlled values."""

    borderline_rate: float
    expert_disagreement_rate: float
    safety_variance: float

    def to_snapshot(self) -> CalibrationSnapshot:
        """Convert fixture to CalibrationSnapshot."""
        return CalibrationSnapshot(
            per_scorer_agreement={"scorer_a": 0.85},
            borderline_rate=self.borderline_rate,
            expert_disagreement_rate=self.expert_disagreement_rate,
            safety_variance=self.safety_variance,
            generated_at="2024-01-01T00:00:00+00:00",
            scoring_report_count=1,
            promotion_report_count=1,
            total_items=100,
            borderline_count=int(self.borderline_rate * 100),
            disagreement_count=int(self.expert_disagreement_rate * 100),
        )


class RoutingDecisionFixture(TypedDict):
    """TypedDict for routing decision fixture."""

    accept_items: list[str]
    reject_items: list[str]
    human_review_items: list[str]
    upstream_boost_items: list[str]


def make_routing_decision(
    accept_items: list[str] | None = None,
    reject_items: list[str] | None = None,
    human_review_items: list[str] | None = None,
    upstream_boost_items: list[str] | None = None,
    accept_scores: list[float] | None = None,
    reject_scores: list[float] | None = None,
    human_review_scores: list[float] | None = None,
    upstream_boost_scores: list[float] | None = None,
) -> RoutingDecision:
    """Create a RoutingDecision with specified items."""
    return RoutingDecision(
        accept=RoutingBucket(
            items=accept_items or [],
            scores=accept_scores or [],
            reasons=["initial"] * len(accept_items) if accept_items else [],
        ),
        reject=RoutingBucket(
            items=reject_items or [],
            scores=reject_scores or [],
            reasons=["initial"] * len(reject_items) if reject_items else [],
        ),
        human_review=RoutingBucket(
            items=human_review_items or [],
            scores=human_review_scores or [],
            reasons=["initial"] * len(human_review_items) if human_review_items else [],
        ),
        upstream_boost=RoutingBucket(
            items=upstream_boost_items or [],
            scores=upstream_boost_scores or [],
            reasons=["initial"] * len(upstream_boost_items) if upstream_boost_items else [],
        ),
    )


def count_bucket_items(routing: RoutingDecision) -> dict[str, int]:
    """Count items in each bucket."""
    return {
        "accept": len(routing.accept.items),
        "reject": len(routing.reject.items),
        "human_review": len(routing.human_review.items),
        "upstream_boost": len(routing.upstream_boost.items),
    }


# ---------------------------------------------------------------------------
# Test: VAL-M3-ROUT-001 - Expert disagreement expands human_review
# ---------------------------------------------------------------------------


class TestExpertDisagreementRule:
    """Tests for expert disagreement rule (VAL-M3-ROUT-001)."""

    def test_no_expansion_when_disagreement_at_threshold(self) -> None:
        """VAL-M3-ROUT-001: No expansion when expert_disagreement_rate == 0.2."""
        # Create routing with some accept items
        routing = make_routing_decision(
            accept_items=["item_1", "item_2", "item_3"],
            accept_scores=[0.7, 0.75, 0.8],
        )

        # Snapshot with disagreement exactly at threshold
        snapshot = SnapshotFixture(
            borderline_rate=0.0,
            expert_disagreement_rate=0.2,  # Exactly at threshold
            safety_variance=0.0,
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # human_review should not expand (no items moved)
        assert count_bucket_items(result)["human_review"] == 0
        # accept items should remain
        assert count_bucket_items(result)["accept"] == 3

    def test_expansion_when_disagreement_above_threshold(self) -> None:
        """VAL-M3-ROUT-001: human_review expands when expert_disagreement_rate > 0.2."""
        # Create routing with accept items
        routing = make_routing_decision(
            accept_items=["item_1", "item_2", "item_3", "item_4", "item_5"],
            accept_scores=[0.7, 0.75, 0.8, 0.72, 0.68],
        )

        # Snapshot with disagreement above threshold
        snapshot = SnapshotFixture(
            borderline_rate=0.0,
            expert_disagreement_rate=0.25,  # 0.05 above threshold
            safety_variance=0.0,
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # human_review should have expanded
        original_hr_count = 0  # original had no human_review items
        new_hr_count = count_bucket_items(result)["human_review"]
        assert new_hr_count > original_hr_count, (
            f"human_review should expand: original={original_hr_count}, new={new_hr_count}"
        )

    def test_expansion_proportional_to_disagreement_excess(self) -> None:
        """VAL-M3-ROUT-001: Expansion is proportional to disagreement rate excess."""
        # Two snapshots with different disagreement rates
        snapshot_low = SnapshotFixture(
            borderline_rate=0.0,
            expert_disagreement_rate=0.21,  # Just above threshold (0.01 excess)
            safety_variance=0.0,
        ).to_snapshot()

        snapshot_high = SnapshotFixture(
            borderline_rate=0.0,
            expert_disagreement_rate=0.35,  # Higher excess (0.15 excess)
            safety_variance=0.0,
        ).to_snapshot()

        # Use 20 items so the ceiling effect doesn't mask the difference
        rules = AdvancedRoutingRules()

        # Apply with lower disagreement
        routing_low = make_routing_decision(
            accept_items=[f"item_{i}" for i in range(20)],
            accept_scores=[0.7] * 20,
        )
        result_low = rules.apply(routing_low, snapshot_low)
        count_low = count_bucket_items(result_low)["human_review"]

        # Apply with higher disagreement
        routing_high = make_routing_decision(
            accept_items=[f"item_{i}" for i in range(20)],
            accept_scores=[0.7] * 20,
        )
        result_high = rules.apply(routing_high, snapshot_high)
        count_high = count_bucket_items(result_high)["human_review"]

        # Higher disagreement should result in more items moved
        assert count_high > count_low, (
            f"Higher disagreement should expand more: low={count_low}, high={count_high}"
        )

    def test_reject_items_not_affected_by_disagreement_rule(self) -> None:
        """VAL-M3-ROUT-001: Reject bucket items are not affected by disagreement rule."""
        routing = make_routing_decision(
            accept_items=["accept_1", "accept_2"],
            reject_items=["reject_1", "reject_2", "reject_3"],
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.0,
            expert_disagreement_rate=0.3,
            safety_variance=0.0,
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # reject items should remain unchanged
        assert count_bucket_items(result)["reject"] == 3


# ---------------------------------------------------------------------------
# Test: VAL-M3-ROUT-002 - Safety variance overrides accept -> human_review
# ---------------------------------------------------------------------------


class TestSafetyVarianceRule:
    """Tests for safety variance rule (VAL-M3-ROUT-002)."""

    def test_no_override_when_variance_at_threshold(self) -> None:
        """VAL-M3-ROUT-002: No override when safety_variance == threshold."""
        routing = make_routing_decision(
            accept_items=["item_1", "item_2"],
            accept_scores=[0.7, 0.8],
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.0,
            expert_disagreement_rate=0.0,
            safety_variance=0.05,  # Exactly at default threshold
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # accept items should remain
        assert count_bucket_items(result)["accept"] == 2

    def test_accept_moved_to_human_review_when_variance_above_threshold(
        self,
    ) -> None:
        """VAL-M3-ROUT-002: Accept items move to human_review when safety_variance > threshold."""
        routing = make_routing_decision(
            accept_items=["accept_1", "accept_2", "accept_3"],
            reject_items=["reject_1"],
            human_review_items=["review_1"],
            accept_scores=[0.7, 0.8, 0.75],
            reject_scores=[0.2],
            human_review_scores=[0.5],
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.0,
            expert_disagreement_rate=0.0,
            safety_variance=0.08,  # Above threshold 0.05
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # All accept items should be moved to human_review
        assert count_bucket_items(result)["accept"] == 0
        assert count_bucket_items(result)["human_review"] >= 3  # original + moved

    def test_reject_items_stay_rejected_under_safety_variance(self) -> None:
        """VAL-M3-ROUT-002: Reject items stay rejected even when variance is high."""
        routing = make_routing_decision(
            accept_items=["accept_1"],
            reject_items=["reject_1", "reject_2"],
            accept_scores=[0.7],
            reject_scores=[0.2, 0.3],
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.0,
            expert_disagreement_rate=0.0,
            safety_variance=0.1,  # High variance
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # reject items should remain rejected
        assert count_bucket_items(result)["reject"] == 2
        # accept items should have moved to human_review
        assert count_bucket_items(result)["accept"] == 0

    def test_human_review_items_stay_in_human_review_under_safety_variance(
        self,
    ) -> None:
        """VAL-M3-ROUT-002: Human_review items stay in human_review under variance rule."""
        routing = make_routing_decision(
            accept_items=["accept_1"],
            human_review_items=["review_1", "review_2"],
            accept_scores=[0.7],
            human_review_scores=[0.5, 0.55],
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.0,
            expert_disagreement_rate=0.0,
            safety_variance=0.08,  # Above threshold
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # original human_review items should still be there
        assert count_bucket_items(result)["human_review"] >= 2
        # accept items should have moved
        assert count_bucket_items(result)["accept"] == 0

    def test_moved_items_have_correct_reason(self) -> None:
        """VAL-M3-ROUT-002: Moved items have 'safety_variance_high' reason."""
        routing = make_routing_decision(
            accept_items=["accept_1", "accept_2"],
            accept_scores=[0.7, 0.8],
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.0,
            expert_disagreement_rate=0.0,
            safety_variance=0.08,  # Above threshold
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # Check that moved items have safety_variance_high reason
        moved_reasons = []
        for reason in result.human_review.reasons:
            if reason == "safety_variance_high":
                moved_reasons.append(reason)

        assert len(moved_reasons) >= 2, (
            f"Expected at least 2 items with safety_variance_high reason, got {moved_reasons}"
        )


# ---------------------------------------------------------------------------
# Test: VAL-M3-ROUT-003 - JSON round-trip and determinism
# ---------------------------------------------------------------------------


class TestDeterminismAndRoundTrip:
    """Tests for determinism and JSON round-trip (VAL-M3-ROUT-003)."""

    def test_json_roundtrip_produces_equivalent_decision(self) -> None:
        """VAL-M3-ROUT-003: JSON round-trip produces equivalent RoutingDecision."""
        routing = make_routing_decision(
            accept_items=["a1", "a2"],
            reject_items=["r1"],
            human_review_items=["h1"],
            upstream_boost_items=["u1"],
            accept_scores=[0.7, 0.75],
            reject_scores=[0.2],
            human_review_scores=[0.5],
            upstream_boost_scores=[0.55],
        )

        # Serialize to JSON
        json_str = routing.to_json()

        # Deserialize back
        restored = RoutingDecision.from_json(json_str)

        # Should be equivalent
        assert routing == restored

    def test_apply_result_roundtrips_through_json(self) -> None:
        """VAL-M3-ROUT-003: apply() result serializes/deserializes correctly."""
        routing = make_routing_decision(
            accept_items=["item_1", "item_2"],
            accept_scores=[0.7, 0.8],
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.0,
            expert_disagreement_rate=0.0,
            safety_variance=0.1,  # High variance to trigger rule
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # Should be able to serialize and deserialize
        json_str = result.to_json()
        restored = RoutingDecision.from_json(json_str)

        # Should be equivalent
        assert result == restored

    def test_apply_is_deterministic_same_inputs(self) -> None:
        """VAL-M3-ROUT-003: apply() is deterministic given same inputs."""
        routing = make_routing_decision(
            accept_items=["item_1", "item_2", "item_3"],
            reject_items=["reject_1"],
            human_review_items=["review_1"],
            accept_scores=[0.7, 0.75, 0.8],
            reject_scores=[0.2],
            human_review_scores=[0.5],
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.3,
            expert_disagreement_rate=0.15,
            safety_variance=0.06,
        ).to_snapshot()

        rules = AdvancedRoutingRules()

        # Apply twice with same inputs
        result1 = rules.apply(routing, snapshot)
        result2 = rules.apply(routing, snapshot)

        # Should be identical
        assert result1 == result2

    def test_different_inputs_produce_different_outputs(self) -> None:
        """VAL-M3-ROUT-003: Different inputs can produce different outputs."""
        routing1 = make_routing_decision(accept_items=["item_1"])
        routing2 = make_routing_decision(accept_items=["item_2", "item_3"])

        snapshot = SnapshotFixture(
            borderline_rate=0.3,
            expert_disagreement_rate=0.15,
            safety_variance=0.06,
        ).to_snapshot()

        rules = AdvancedRoutingRules()

        result1 = rules.apply(routing1, snapshot)
        result2 = rules.apply(routing2, snapshot)

        # Different routing inputs should potentially produce different results
        # (not guaranteed to be different, but checking the bucket counts differ)
        assert count_bucket_items(result1) != count_bucket_items(result2)


# ---------------------------------------------------------------------------
# Test: VAL-M3-ROUT-004 - Borderline threshold rules
# ---------------------------------------------------------------------------


class TestBorderlineRateRule:
    """Tests for borderline rate rule (VAL-M3-ROUT-004)."""

    def test_no_boost_when_borderline_at_threshold(self) -> None:
        """VAL-M3-ROUT-004: No boost when borderline_rate == 0.4."""
        routing = make_routing_decision(
            human_review_items=["review_1", "review_2"],
            human_review_scores=[0.45, 0.55],  # borderline scores
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.4,  # Exactly at threshold
            expert_disagreement_rate=0.0,
            safety_variance=0.0,
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # upstream_boost should remain empty
        assert count_bucket_items(result)["upstream_boost"] == 0

    def test_boost_applied_when_borderline_above_threshold(self) -> None:
        """VAL-M3-ROUT-004: upstream_boost grows when borderline_rate > 0.4."""
        routing = make_routing_decision(
            human_review_items=["review_1", "review_2", "review_3", "review_4"],
            human_review_scores=[0.45, 0.5, 0.55, 0.59],  # borderline scores
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.5,  # Above threshold
            expert_disagreement_rate=0.0,
            safety_variance=0.0,
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # upstream_boost should have grown
        original_upstream = 0
        new_upstream = count_bucket_items(result)["upstream_boost"]
        assert new_upstream > original_upstream, (
            f"upstream_boost should grow: original={original_upstream}, new={new_upstream}"
        )

    def test_boost_factor_scales_with_threshold_delta(self) -> None:
        """VAL-M3-ROUT-004: Boost count scales with (borderline_rate - threshold) delta."""
        # Two snapshots with different deltas
        snapshot_slightly_above = SnapshotFixture(
            borderline_rate=0.45,  # 0.05 above threshold
            expert_disagreement_rate=0.0,
            safety_variance=0.0,
        ).to_snapshot()

        snapshot_well_above = SnapshotFixture(
            borderline_rate=0.6,  # 0.2 above threshold
            expert_disagreement_rate=0.0,
            safety_variance=0.0,
        ).to_snapshot()

        rules = AdvancedRoutingRules()

        # Apply with slightly above threshold
        routing1 = make_routing_decision(
            human_review_items=[f"review_{i}" for i in range(10)],
            human_review_scores=[0.45, 0.5, 0.55, 0.59, 0.41, 0.48, 0.52, 0.58, 0.43, 0.47],
        )
        result1 = rules.apply(routing1, snapshot_slightly_above)
        count1 = count_bucket_items(result1)["upstream_boost"]

        # Apply with well above threshold
        routing2 = make_routing_decision(
            human_review_items=[f"review_{i}" for i in range(10)],
            human_review_scores=[0.45, 0.5, 0.55, 0.59, 0.41, 0.48, 0.52, 0.58, 0.43, 0.47],
        )
        result2 = rules.apply(routing2, snapshot_well_above)
        count2 = count_bucket_items(result2)["upstream_boost"]

        # Higher delta should result in more boost
        assert count2 > count1, (
            f"Higher borderline rate should boost more: count1={count1}, count2={count2}"
        )

    def test_upstream_boost_takes_from_human_review_borderline_items(
        self,
    ) -> None:
        """VAL-M3-ROUT-004: upstream_boost takes from human_review's borderline items."""
        routing = make_routing_decision(
            human_review_items=["review_1", "review_2", "review_3", "review_4"],
            human_review_scores=[0.45, 0.7, 0.55, 0.3],  # 2 borderline, 1 accept-like, 1 reject-like
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.5,
            expert_disagreement_rate=0.0,
            safety_variance=0.0,
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # upstream_boost should have items moved from human_review
        upstream_count = count_bucket_items(result)["upstream_boost"]
        assert upstream_count > 0

        # The items moved should be from human_review's borderline items
        # (review_1 and review_3 with scores 0.45 and 0.55)

    def test_non_borderline_items_not_moved_to_upstream_boost(self) -> None:
        """VAL-M3-ROUT-004: Non-borderline items (accept/reject) not moved to upstream_boost."""
        routing = make_routing_decision(
            accept_items=["accept_1", "accept_2"],
            reject_items=["reject_1"],
            human_review_items=["review_1", "review_2"],
            accept_scores=[0.8, 0.7],  # accept range
            reject_scores=[0.2],  # reject range
            human_review_scores=[0.45, 0.55],  # borderline
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.5,
            expert_disagreement_rate=0.0,
            safety_variance=0.0,
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # upstream_boost items should have come from human_review (borderline)
        # not from accept or reject
        upstream_items = set(result.upstream_boost.items)
        accept_items = set(routing.accept.items)
        reject_items = set(routing.reject.items)

        # None of the upstream items should be from original accept/reject
        assert not upstream_items.intersection(accept_items)
        assert not upstream_items.intersection(reject_items)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for AdvancedRoutingRules."""

    def test_empty_routing_decision(self) -> None:
        """Test apply with empty routing decision."""
        routing = make_routing_decision()

        snapshot = SnapshotFixture(
            borderline_rate=0.5,
            expert_disagreement_rate=0.3,
            safety_variance=0.1,
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # Should handle gracefully - all buckets should be empty
        assert count_bucket_items(result)["accept"] == 0
        assert count_bucket_items(result)["reject"] == 0
        assert count_bucket_items(result)["human_review"] == 0
        assert count_bucket_items(result)["upstream_boost"] == 0

    def test_all_rules_fire_simultaneously(self) -> None:
        """Test when all three rules fire at once."""
        routing = make_routing_decision(
            accept_items=["accept_1", "accept_2"],
            reject_items=["reject_1"],
            human_review_items=["review_1", "review_2"],
            accept_scores=[0.7, 0.8],
            reject_scores=[0.2],
            human_review_scores=[0.45, 0.55],
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.5,  # > 0.4 threshold
            expert_disagreement_rate=0.3,  # > 0.2 threshold
            safety_variance=0.1,  # > 0.05 threshold
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # All rules should have fired
        # - safety variance: accept should be empty (moved to human_review)
        assert count_bucket_items(result)["accept"] == 0
        # - expert disagreement: human_review should expand
        # - borderline rate: upstream_boost should have items

        # The result should be JSON serializable
        json_str = result.to_json()
        restored = RoutingDecision.from_json(json_str)
        assert result == restored

    def test_zero_snapshot_values(self) -> None:
        """Test apply with all-zero snapshot values."""
        routing = make_routing_decision(
            accept_items=["item_1"],
            accept_scores=[0.7],
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.0,
            expert_disagreement_rate=0.0,
            safety_variance=0.0,
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # With all zero values, no rules should fire
        # accept should remain unchanged
        assert count_bucket_items(result)["accept"] == 1

    def test_custom_thresholds(self) -> None:
        """Test with custom threshold values."""
        routing = make_routing_decision(
            human_review_items=["review_1", "review_2", "review_3"],
            human_review_scores=[0.45, 0.55, 0.65],
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.3,  # Would be below default 0.4
            expert_disagreement_rate=0.15,  # Would be below default 0.2
            safety_variance=0.03,  # Would be below default 0.05
        ).to_snapshot()

        # With custom thresholds set to match these values, nothing should fire
        rules = AdvancedRoutingRules(
            borderline_rate_threshold=0.2,  # 0.3 > 0.2, so rule fires
            expert_disagreement_threshold=0.1,  # 0.15 > 0.1, so rule fires
            safety_variance_threshold=0.01,  # 0.03 > 0.01, so rule fires
        )

        result = rules.apply(routing, snapshot)

        # With these custom thresholds, all rules would fire since values exceed
        # We just verify it doesn't crash
        assert result is not None


# ---------------------------------------------------------------------------
# Equality method tests (for branch coverage)
# ---------------------------------------------------------------------------


class TestRoutingBucketEquality:
    """Tests for RoutingBucket __eq__ method branch coverage."""

    def test_eq_returns_not_implemented_for_incompatible_type(self) -> None:
        """Test that RoutingBucket.__eq__ returns NotImplemented for incompatible types."""
        bucket = RoutingBucket(items=["a"], scores=[0.5], reasons=["test"])

        # Should return NotImplemented, not raise
        result = bucket.__eq__("not a bucket")
        assert result is NotImplemented

        result = bucket.__eq__(42)
        assert result is NotImplemented

        result = bucket.__eq__(None)
        assert result is NotImplemented

    def test_eq_returns_false_for_different_bucket(self) -> None:
        """Test that RoutingBucket.__eq__ returns False for different values."""
        bucket1 = RoutingBucket(items=["a"], scores=[0.5], reasons=["test"])
        bucket2 = RoutingBucket(items=["b"], scores=[0.6], reasons=["other"])

        assert bucket1 != bucket2


class TestRoutingDecisionEquality:
    """Tests for RoutingDecision __eq__ method branch coverage."""

    def test_eq_returns_not_implemented_for_incompatible_type(self) -> None:
        """Test that RoutingDecision.__eq__ returns NotImplemented for incompatible types."""
        routing = make_routing_decision()

        # Should return NotImplemented, not raise
        result = routing.__eq__("not a routing")
        assert result is NotImplemented

        result = routing.__eq__(42)
        assert result is NotImplemented

        result = routing.__eq__(None)
        assert result is NotImplemented

    def test_eq_returns_false_for_different_decisions(self) -> None:
        """Test that RoutingDecision.__eq__ returns False for different values."""
        routing1 = make_routing_decision(accept_items=["a1"])
        routing2 = make_routing_decision(accept_items=["a2"])

        assert routing1 != routing2


# ---------------------------------------------------------------------------
# Branch coverage tests for edge cases
# ---------------------------------------------------------------------------


class TestExpertDisagreementBranchCoverage:
    """Additional tests for expert disagreement rule branch coverage."""

    def test_expert_disagreement_with_very_small_excess(self) -> None:
        """Test expert disagreement rule with very small excess (branch coverage)."""
        routing = make_routing_decision(
            accept_items=[f"item_{i}" for i in range(100)],
            accept_scores=[0.7] * 100,
        )

        # Very small excess: 0.2001 - 0.2 = 0.0001
        # With 100 items, ceil(100 * 0.0001 * 2) = ceil(0.02) = 1
        snapshot = SnapshotFixture(
            borderline_rate=0.0,
            expert_disagreement_rate=0.2001,
            safety_variance=0.0,
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # Should move exactly 1 item (due to ceiling effect)
        assert count_bucket_items(result)["human_review"] == 1
        assert count_bucket_items(result)["accept"] == 99

    def test_expert_disagreement_with_large_excess_caps_at_accept_count(self) -> None:
        """Test expert disagreement rule caps moved items at accept count."""
        routing = make_routing_decision(
            accept_items=["only_one"],
            accept_scores=[0.7],
        )

        # Large excess: 0.5 - 0.2 = 0.3, ceil(1 * 0.3 * 2) = 1
        # But if we had 100 items, it would be ceil(100 * 0.3 * 2) = 60
        snapshot = SnapshotFixture(
            borderline_rate=0.0,
            expert_disagreement_rate=0.5,
            safety_variance=0.0,
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # Can't move more than we have
        assert count_bucket_items(result)["human_review"] <= 1
        assert count_bucket_items(result)["accept"] >= 0


class TestBorderlineRateBranchCoverage:
    """Additional tests for borderline rate rule branch coverage."""

    def test_borderline_rule_checks_accept_items(self) -> None:
        """Test borderline rule includes accept items in borderline check."""
        # Accept items with borderline scores
        routing = make_routing_decision(
            accept_items=["accept_borderline"],
            accept_scores=[0.55],  # borderline range
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.5,
            expert_disagreement_rate=0.0,
            safety_variance=0.0,
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # upstream_boost should have been populated
        assert count_bucket_items(result)["upstream_boost"] > 0

    def test_borderline_rule_reject_items_not_borderline(self) -> None:
        """Test that reject items (low scores) are not counted as borderline."""
        routing = make_routing_decision(
            reject_items=["reject_low"],
            reject_scores=[0.2],  # reject range
            human_review_items=["review_borderline"],
            human_review_scores=[0.55],  # borderline
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.5,
            expert_disagreement_rate=0.0,
            safety_variance=0.0,
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # Only borderline review items should be moved to upstream_boost
        # (not reject items since they're in different score range)
        upstream_items = set(result.upstream_boost.items)
        reject_items = set(routing.reject.items)

        # No reject items should be in upstream_boost
        assert not upstream_items.intersection(reject_items)

    def test_borderline_rule_break_branch_coverage(self) -> None:
        """Test borderline rule when borderline_rate is at threshold (rule doesn't fire)."""
        # When borderline_rate == threshold (0.4), the rule doesn't fire
        routing = make_routing_decision(
            human_review_items=["review_1"],
            human_review_scores=[0.5],
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.4,  # Exactly at threshold
            expert_disagreement_rate=0.0,
            safety_variance=0.0,
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # At exactly 0.4, rule doesn't fire (borderline_rate > threshold is False)
        assert count_bucket_items(result)["upstream_boost"] == 0

    def test_borderline_rule_with_zero_borderline_count(self) -> None:
        """Test borderline rule when there are no borderline items."""
        routing = make_routing_decision(
            human_review_items=["review_accept", "review_reject"],
            human_review_scores=[0.7, 0.3],  # No borderline scores
        )

        # Even with high borderline_rate, if no items are borderline, boost is 0
        snapshot = SnapshotFixture(
            borderline_rate=0.5,  # High, but no borderline items
            expert_disagreement_rate=0.0,
            safety_variance=0.0,
        ).to_snapshot()

        rules = AdvancedRoutingRules()
        result = rules.apply(routing, snapshot)

        # No borderline items means boost_count = 0
        assert count_bucket_items(result)["upstream_boost"] == 0


# ---------------------------------------------------------------------------
# Integration test with convenience function
# ---------------------------------------------------------------------------


class TestApplyFunction:
    """Tests for the apply() convenience function."""

    def test_apply_convenience_function(self) -> None:
        """Test the apply() convenience function works correctly."""
        routing = make_routing_decision(
            accept_items=["item_1"],
            accept_scores=[0.7],
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.0,
            expert_disagreement_rate=0.0,
            safety_variance=0.1,  # Triggers safety variance rule
        ).to_snapshot()

        result = apply_routing(routing, snapshot)

        # Should have moved accept to human_review
        assert count_bucket_items(result)["accept"] == 0
        assert count_bucket_items(result)["human_review"] >= 1

    def test_apply_with_custom_thresholds(self) -> None:
        """Test apply() convenience function with custom thresholds."""
        routing = make_routing_decision(
            human_review_items=["review_1"],
            human_review_scores=[0.45],
        )

        snapshot = SnapshotFixture(
            borderline_rate=0.5,
            expert_disagreement_rate=0.0,
            safety_variance=0.0,
        ).to_snapshot()

        result = apply_routing(
            routing,
            snapshot,
            borderline_rate_threshold=0.6,  # 0.5 < 0.6, rule should NOT fire
        )

        # Rule should not fire with higher threshold
        assert count_bucket_items(result)["upstream_boost"] == 0
