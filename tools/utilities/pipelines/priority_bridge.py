"""Priority Sync Bridge for PIX-4747.

Maps the engine's 5-level priority tiers (URGENT/HIGH/MEDIUM/LOW/BACKLOG)
back to standard Atlassian/Linear priority levels (Highest/High/Medium/Low/
Lowest), and provides a consistency audit that verifies calculated
priorities against the priority fields currently recorded on target-system
tickets.

The bridge is intentionally transport-agnostic: it produces mapping
payloads (``item_id`` -> priority label) that callers can push to Atlassian
or Linear, and an audit diff that flags tickets whose recorded priority no
longer matches the calculated one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ai.tools.utilities.pipelines.reprioritization_engine import PriorityTier
from ai.tools.utilities.pipelines.scoring import ScoredTicket

#: Standard Atlassian/Linear priority labels (spec target scale).
HIGHEST = "Highest"
HIGH = "High"
MEDIUM = "Medium"
LOW = "Low"
LOWEST = "Lowest"

#: Internal tier -> external label (spec: sync bridge mapping).
TIER_TO_LABEL: dict[PriorityTier, str] = {
    PriorityTier.URGENT: HIGHEST,
    PriorityTier.HIGH: HIGH,
    PriorityTier.MEDIUM: MEDIUM,
    PriorityTier.LOW: LOW,
    PriorityTier.BACKLOG: LOWEST,
}

#: External label -> internal tier (inverse, for round-tripping).
LABEL_TO_TIER: dict[str, PriorityTier] = {label: tier for tier, label in TIER_TO_LABEL.items()}

#: Tier string (as emitted by scoring.py) -> internal tier.
TIER_STRING_TO_TIER: dict[str, PriorityTier] = {tier.value: tier for tier in PriorityTier}


class PriorityScale(StrEnum):
    """Target-system priority scales the bridge can emit."""

    ATLASSIAN = "atlassian"
    LINEAR = "linear"


@dataclass
class PriorityMapping:
    """One calculated-priority -> external-label assignment."""

    item_id: str
    tier: PriorityTier
    label: str
    score: float
    rank: int
    scale: PriorityScale = PriorityScale.ATLASSIAN
    current_label: str | None = None

    @property
    def needs_update(self) -> bool:
        """True when the recorded label differs from the calculated one."""
        return self.current_label is not None and self.current_label != self.label

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "tier": self.tier.value,
            "label": self.label,
            "score": round(self.score, 6),
            "rank": self.rank,
            "scale": self.scale.value,
            "current_label": self.current_label,
            "needs_update": self.needs_update,
        }


@dataclass
class ConsistencyAuditReport:
    """Result of comparing calculated priorities to recorded ticket fields."""

    total_items: int
    matched: int
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    unrecorded: list[str] = field(default_factory=list)

    @property
    def is_consistent(self) -> bool:
        return len(self.mismatches) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_items": self.total_items,
            "matched": self.matched,
            "mismatch_count": len(self.mismatches),
            "mismatches": self.mismatches,
            "unrecorded_count": len(self.unrecorded),
            "unrecorded": self.unrecorded,
            "is_consistent": self.is_consistent,
        }


class PrioritySyncBridge:
    """Bridges calculated priorities to Atlassian/Linear priority levels."""

    def __init__(self, scale: PriorityScale = PriorityScale.ATLASSIAN) -> None:
        self._scale = scale

    @property
    def scale(self) -> PriorityScale:
        return self._scale

    def tier_to_label(self, tier: PriorityTier) -> str:
        return TIER_TO_LABEL[tier]

    def label_to_tier(self, label: str) -> PriorityTier:
        try:
            return LABEL_TO_TIER[label]
        except KeyError:
            raise ValueError(f"Unknown external priority label: {label!r}") from None

    def build_mapping(
        self,
        ranked_tickets: list[ScoredTicket],
        current_labels: dict[str, str | None] | None = None,
    ) -> list[PriorityMapping]:
        """Build sync payloads for a ranked ticket list.

        ``current_labels`` optionally carries the labels currently recorded
        on the target system (keyed by item id) so each payload can flag
        whether an update is needed.
        """
        labels = current_labels or {}
        mappings: list[PriorityMapping] = []
        for scored in ranked_tickets:
            tier = TIER_STRING_TO_TIER.get(scored.tier, PriorityTier.BACKLOG)
            mappings.append(
                PriorityMapping(
                    item_id=scored.ticket.item_id,
                    tier=tier,
                    label=self.tier_to_label(tier),
                    score=scored.score,
                    rank=scored.rank,
                    scale=self._scale,
                    current_label=labels.get(scored.ticket.item_id),
                )
            )
        return mappings

    def sync_payload(self, mappings: list[PriorityMapping]) -> dict[str, Any]:
        """Transport-agnostic payload describing the labels to apply."""
        return {
            "scale": self._scale.value,
            "updates": [m.to_dict() for m in mappings if m.needs_update or m.current_label is None],
            "unchanged": [m.item_id for m in mappings if m.current_label is not None and not m.needs_update],
        }

    def audit_consistency(
        self,
        ranked_tickets: list[ScoredTicket],
        recorded_priorities: dict[str, str | None],
    ) -> ConsistencyAuditReport:
        """Verify calculated priorities match recorded ticket fields.

        For each ranked ticket the recorded label is converted back to a
        tier; if it differs from the calculated tier the ticket is a
        mismatch. Tickets with no recorded label are reported separately
        (``unrecorded``) and do not count as mismatches.
        """
        mappings = self.build_mapping(ranked_tickets, current_labels=recorded_priorities)
        mismatches: list[dict[str, Any]] = []
        unrecorded: list[str] = []
        matched = 0
        for m in mappings:
            if m.current_label is None:
                unrecorded.append(m.item_id)
                continue
            if m.needs_update:
                mismatches.append(
                    {
                        "item_id": m.item_id,
                        "recorded_label": m.current_label,
                        "recorded_tier": self.label_to_tier(m.current_label).value
                        if m.current_label in LABEL_TO_TIER
                        else m.current_label,
                        "calculated_label": m.label,
                        "calculated_tier": m.tier.value,
                    }
                )
            else:
                matched += 1
        return ConsistencyAuditReport(
            total_items=len(mappings),
            matched=matched,
            mismatches=mismatches,
            unrecorded=unrecorded,
        )


__all__ = [
    "HIGH",
    "HIGHEST",
    "LABEL_TO_TIER",
    "LOW",
    "LOWEST",
    "MEDIUM",
    "TIER_TO_LABEL",
    "ConsistencyAuditReport",
    "PriorityMapping",
    "PriorityScale",
    "PrioritySyncBridge",
]
