"""Advanced routing rules for clinical validity enhancement pipeline.

This module implements the AdvancedRoutingRules that consume a RoutingDecision
and a calibration snapshot to apply threshold-based routing adjustments.

Rules applied:
- If expert_disagreement_rate > 0.2: expand human_review bucket
- If safety_variance > threshold: override accept -> human_review
- If borderline_rate > 0.4: boost upstream sample rate

The AdvancedRoutingRules.apply() method is deterministic and produces a
RoutingDecision that round-trips cleanly through JSON serialization.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import TypedDict

from training.coaching_safety.calibration_metrics import CalibrationSnapshot

logger = logging.getLogger("advanced_routing")


# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------

# Expert disagreement threshold: > 0.2 triggers human_review expansion
EXPERT_DISAGREEMENT_THRESHOLD = 0.2

# Borderline rate threshold: > 0.4 triggers upstream boost
BORDERLINE_RATE_THRESHOLD = 0.4

# Default safety variance threshold
DEFAULT_SAFETY_VARIANCE_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class RoutingBucket:
    """A routing bucket containing item IDs and metadata."""

    items: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    # dataclass has eq=False because we define __eq__ below
    # and we don't need these objects to be hashable
    __hash__: None = None

    def to_dict(self) -> dict:
        """Serialize bucket to dictionary."""
        return {
            "items": self.items,
            "scores": self.scores,
            "reasons": self.reasons,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RoutingBucket:
        """Deserialize bucket from dictionary."""
        return cls(
            items=list(data.get("items", [])),
            scores=list(data.get("scores", [])),
            reasons=list(data.get("reasons", [])),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RoutingBucket):
            return NotImplemented
        return (
            self.items == other.items
            and self.scores == other.scores
            and self.reasons == other.reasons
        )


class RoutingBucketDict(TypedDict):
    """TypedDict representation of RoutingBucket for JSON serialization."""

    items: list[str]
    scores: list[float]
    reasons: list[str]


@dataclass(eq=False)
class RoutingDecision:
    """A routing decision containing all routing buckets.

    This class is designed to be fully JSON round-trippable: it serializes
    to JSON and deserializes back to an equivalent RoutingDecision.

    eq=False because we define __eq__ below and don't need these objects
    to be hashable.
    """

    __hash__: None = None

    accept: RoutingBucket = field(default_factory=RoutingBucket)
    reject: RoutingBucket = field(default_factory=RoutingBucket)
    human_review: RoutingBucket = field(default_factory=RoutingBucket)
    upstream_boost: RoutingBucket = field(default_factory=RoutingBucket)

    def to_dict(self) -> dict:
        """Serialize routing decision to dictionary for JSON output."""
        return {
            "accept": self.accept.to_dict(),
            "reject": self.reject.to_dict(),
            "human_review": self.human_review.to_dict(),
            "upstream_boost": self.upstream_boost.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> RoutingDecision:
        """Deserialize routing decision from dictionary."""
        return cls(
            accept=RoutingBucket.from_dict(data.get("accept", {})),
            reject=RoutingBucket.from_dict(data.get("reject", {})),
            human_review=RoutingBucket.from_dict(data.get("human_review", {})),
            upstream_boost=RoutingBucket.from_dict(data.get("upstream_boost", {})),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> RoutingDecision:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RoutingDecision):
            return NotImplemented
        return (
            self.accept == other.accept
            and self.reject == other.reject
            and self.human_review == other.human_review
            and self.upstream_boost == other.upstream_boost
        )


class RoutingDecisionDict(TypedDict):
    """TypedDict representation of RoutingDecision for JSON serialization."""

    accept: RoutingBucketDict
    reject: RoutingBucketDict
    human_review: RoutingBucketDict
    upstream_boost: RoutingBucketDict


# ---------------------------------------------------------------------------
# AdvancedRoutingRules
# ---------------------------------------------------------------------------


class AdvancedRoutingRules:
    """Applies calibration-driven threshold rules to routing decisions.

    Consumes a RoutingDecision (initial routing from scorer) and a
    CalibrationSnapshot (calibration metrics from aggregator) and produces
    a modified RoutingDecision with adjusted buckets based on threshold rules.

    Rules:
    1. expert_disagreement_rate > 0.2: expand human_review bucket
    2. safety_variance > threshold: override accept -> human_review
    3. borderline_rate > 0.4: boost upstream sample rate

    All methods are deterministic and pure functions.
    """

    def __init__(
        self,
        expert_disagreement_threshold: float = EXPERT_DISAGREEMENT_THRESHOLD,
        borderline_rate_threshold: float = BORDERLINE_RATE_THRESHOLD,
        safety_variance_threshold: float = DEFAULT_SAFETY_VARIANCE_THRESHOLD,
    ) -> None:
        """Initialize the AdvancedRoutingRules.

        Args:
            expert_disagreement_threshold: Threshold for expert disagreement
                rate. Default is 0.2.
            borderline_rate_threshold: Threshold for borderline rate.
                Default is 0.4.
            safety_variance_threshold: Threshold for safety variance.
                Default is 0.05.
        """
        self.expert_disagreement_threshold = expert_disagreement_threshold
        self.borderline_rate_threshold = borderline_rate_threshold
        self.safety_variance_threshold = safety_variance_threshold

    def apply(
        self,
        routing: RoutingDecision,
        snapshot: CalibrationSnapshot,
    ) -> RoutingDecision:
        """Apply calibration threshold rules to a routing decision.

        Args:
            routing: The initial routing decision from the scorer.
            snapshot: The calibration snapshot with current metrics.

        Returns:
            A new RoutingDecision with adjusted buckets.
        """
        # Create a fresh copy of the routing decision
        result = RoutingDecision(
            accept=RoutingBucket(
                items=list(routing.accept.items),
                scores=list(routing.accept.scores),
                reasons=list(routing.accept.reasons),
            ),
            reject=RoutingBucket(
                items=list(routing.reject.items),
                scores=list(routing.reject.scores),
                reasons=list(routing.reject.reasons),
            ),
            human_review=RoutingBucket(
                items=list(routing.human_review.items),
                scores=list(routing.human_review.scores),
                reasons=list(routing.human_review.reasons),
            ),
            upstream_boost=RoutingBucket(
                items=list(routing.upstream_boost.items),
                scores=list(routing.upstream_boost.scores),
                reasons=list(routing.upstream_boost.reasons),
            ),
        )

        # Apply rule 1: expert disagreement > threshold expands human_review
        if snapshot.expert_disagreement_rate > self.expert_disagreement_threshold:
            result = self._apply_expert_disagreement_rule(routing, result, snapshot)

        # Apply rule 2: safety variance > threshold overrides accept -> human_review
        if snapshot.safety_variance > self.safety_variance_threshold:
            result = self._apply_safety_variance_rule(result, snapshot)

        # Apply rule 3: borderline rate > threshold boosts upstream
        if snapshot.borderline_rate > self.borderline_rate_threshold:
            result = self._apply_borderline_rate_rule(routing, result, snapshot)

        # Ensure result is JSON round-trippable (convert to dict and back)
        result = self._ensure_json_roundtrip(result)

        logger.info(
            "Applied advanced routing rules: expert_disagreement=%.3f (threshold=%.3f), "
            "safety_variance=%.3f (threshold=%.3f), borderline_rate=%.3f (threshold=%.3f)",
            snapshot.expert_disagreement_rate,
            self.expert_disagreement_threshold,
            snapshot.safety_variance,
            self.safety_variance_threshold,
            snapshot.borderline_rate,
            self.borderline_rate_threshold,
        )

        return result

    def _apply_expert_disagreement_rule(
        self,
        routing: RoutingDecision,
        result: RoutingDecision,
        snapshot: CalibrationSnapshot,
    ) -> RoutingDecision:
        """Apply expert disagreement rule: expand human_review bucket.

        When expert_disagreement_rate > threshold, we expand the human_review
        bucket by moving items from accept and reject that are in borderline
        score range. The number of items to move is proportional to the
        disagreement rate above the threshold.

        Args:
            routing: The original routing decision.
            result: The current result routing decision (modified in place).
            snapshot: The calibration snapshot.

        Returns:
            Modified RoutingDecision with expanded human_review bucket.
        """
        # Calculate expansion factor: how much above threshold
        excess_disagreement = snapshot.expert_disagreement_rate - self.expert_disagreement_threshold

        # Move a portion of accept items to human_review based on excess
        # Expansion factor: ceil(total_accept * excess_disagreement * 2)
        # Using *2 to make the expansion more significant
        accept_to_move = min(
            len(result.accept.items),
            math.ceil(len(routing.accept.items) * excess_disagreement * 2),
        )

        for i in range(accept_to_move):
            item_id = result.accept.items[i]
            score = result.accept.scores[i] if i < len(result.accept.scores) else 0.0
            result.human_review.items.append(item_id)
            result.human_review.scores.append(score)
            result.human_review.reasons.append("expert_disagreement_high")

        # Remove moved items from accept
        result.accept.items = result.accept.items[accept_to_move:]
        result.accept.scores = result.accept.scores[accept_to_move:]
        result.accept.reasons = result.accept.reasons[accept_to_move:]

        return result

    def _apply_safety_variance_rule(
        self,
        result: RoutingDecision,
        _snapshot: CalibrationSnapshot,
    ) -> RoutingDecision:
        """Apply safety variance rule: override accept -> human_review.

        When safety_variance > threshold, all items in the accept bucket
        are moved to human_review for additional safety review.

        Args:
            result: The current result routing decision.
            _snapshot: The calibration snapshot (unused, kept for API consistency).

        Returns:
            Modified RoutingDecision with accept items moved to human_review.
        """
        # Move all accept items to human_review
        for i, item_id in enumerate(result.accept.items):
            score = result.accept.scores[i] if i < len(result.accept.scores) else 0.0
            result.human_review.items.append(item_id)
            result.human_review.scores.append(score)
            result.human_review.reasons.append("safety_variance_high")

        # Clear accept bucket
        result.accept.items = []
        result.accept.scores = []
        result.accept.reasons = []

        return result

    def _apply_borderline_rate_rule(
        self,
        _routing: RoutingDecision,
        result: RoutingDecision,
        snapshot: CalibrationSnapshot,
    ) -> RoutingDecision:
        """Apply borderline rate rule: boost upstream sample rate.

        When borderline_rate > threshold, we boost the upstream_boost bucket
        by moving borderline-scored items from accept and human_review.
        The boost factor is derived from the threshold delta.

        Formula: boost_count = ceil(borderline_count * boost_factor)
        boost_factor = (borderline_rate - threshold) * 2

        Args:
            _routing: The original routing decision (unused, kept for API consistency).
            result: The current result routing decision.
            snapshot: The calibration snapshot.

        Returns:
            Modified RoutingDecision with expanded upstream_boost bucket.
        """
        # Calculate boost factor from threshold delta
        delta = snapshot.borderline_rate - self.borderline_rate_threshold
        boost_factor = delta * 2

        # Borderline items are in the 0.4-0.6 score range
        # Count items in human_review that are borderline
        borderline_items: list[tuple[str, float]] = []  # (item_id, score)

        for i, score in enumerate(result.human_review.scores):
            if 0.4 <= score < 0.6:
                item_id = result.human_review.items[i] if i < len(result.human_review.items) else f"hr_{i}"
                borderline_items.append((item_id, score))

        # Also check accept items
        for i, score in enumerate(result.accept.scores):
            if 0.4 <= score < 0.6:
                item_id = result.accept.items[i] if i < len(result.accept.items) else f"acc_{i}"
                borderline_items.append((item_id, score))

        # Calculate boost count
        borderline_count = len(borderline_items)
        boost_count = math.ceil(borderline_count * boost_factor)

        # Apply boost (move items to upstream_boost)
        items_moved = 0
        for item_id, score in borderline_items:
            if items_moved >= boost_count:
                break

            result.upstream_boost.items.append(item_id)
            result.upstream_boost.scores.append(score)
            result.upstream_boost.reasons.append("borderline_rate_high")

            items_moved += 1

        logger.info(
            "Borderline rate rule: borderline_count=%d, boost_count=%d, "
            "boost_factor=%.3f, delta=%.3f",
            borderline_count,
            boost_count,
            boost_factor,
            delta,
        )

        return result

    def _ensure_json_roundtrip(self, decision: RoutingDecision) -> RoutingDecision:
        """Ensure the routing decision is fully JSON round-trippable.

        Serializes the decision to JSON and deserializes it back to ensure
        that any nested objects are properly converted to JSON-compatible types.

        Args:
            decision: The routing decision to ensure round-trippability.

        Returns:
            A new RoutingDecision that is guaranteed to be JSON round-trippable.
        """
        # Serialize to JSON and back
        json_str = decision.to_json()
        data = json.loads(json_str)
        return RoutingDecision.from_dict(data)


def apply(
    routing: RoutingDecision,
    snapshot: CalibrationSnapshot,
    **kwargs,
) -> RoutingDecision:
    """Convenience function to apply advanced routing rules.

    Args:
        routing: The initial routing decision from the scorer.
        snapshot: The calibration snapshot with current metrics.
        **kwargs: Additional arguments passed to AdvancedRoutingRules constructor.

    Returns:
        A new RoutingDecision with adjusted buckets.
    """
    rules = AdvancedRoutingRules(**kwargs)
    return rules.apply(routing, snapshot)
