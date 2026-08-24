"""PIX-511: Sprint 2 - Memory Ingestion Gating System.

5-gate sequential evaluation pipeline for therapeutic memory ingestion:
  Gate 0 — PII Redaction
  Gate 1 — Crisis Detection
  Gate 2 — Trauma-Trigger Filtering
  Gate 3 — Consent-Gated Retrieval
  Gate 4 — Human Review Queue

Mirrors ai/core/pipelines/privacy_content_gates.py but scoped to per-memory-block
ingestion rather than dataset-level promotion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

# ─── Gate decision enum ────────────────────────────────────────────────────────


class GateDecision(StrEnum):
    PASS = "pass"
    BLOCK = "block"
    ESCALATE = "escalate"


# ─── Shared result types ───────────────────────────────────────────────────────


@dataclass
class GateResult:
    """Outcome of a single gate evaluation."""

    gate: str
    decision: GateDecision
    reason: str
    confidence: float = 0.0
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "decision": self.decision.value,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "details": self.details,
        }


@dataclass
class GatingReport:
    """Full gating report for a single memory block."""

    source_id: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    gate0_pii: GateResult | None = None
    gate1_crisis: GateResult | None = None
    gate2_trauma: GateResult | None = None
    gate3_consent: GateResult | None = None
    gate4_review: GateResult | None = None

    pii_types_found: list[str] = field(default_factory=list)
    crisis_tier: str = "none"
    trauma_indicators: list[str] = field(default_factory=list)
    consent_gate_value: str = "open"
    safety_intent: str | None = None

    @property
    def passed(self) -> bool:
        results = [g for g in self._gate_results if g is not None]
        return all(r.decision in (GateDecision.PASS, GateDecision.ESCALATE) for r in results) and not self.blocked

    @property
    def blocked(self) -> bool:
        results = [g for g in self._gate_results if g is not None]
        return any(r.decision == GateDecision.BLOCK for r in results)

    @property
    def needs_review(self) -> bool:
        results = [g for g in self._gate_results if g is not None]
        return any(r.decision == GateDecision.ESCALATE for r in results)

    @property
    def can_retain(self) -> bool:
        """True when content is safe to retain (passed all gates or escalated for review)."""
        return self.passed

    @property
    def _gate_results(self) -> list[GateResult | None]:
        return [
            self.gate0_pii,
            self.gate1_crisis,
            self.gate2_trauma,
            self.gate3_consent,
            self.gate4_review,
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "content_length": len(self.content),
            "timestamp": self.timestamp,
            "passed": self.passed,
            "blocked": self.blocked,
            "needs_review": self.needs_review,
            "pii_types_found": self.pii_types_found,
            "crisis_tier": self.crisis_tier,
            "trauma_indicators": self.trauma_indicators,
            "consent_gate_value": self.consent_gate_value,
            "safety_intent": self.safety_intent,
            "gates": {
                f"gate{i}": (r.to_dict() if r else None)
                for i, r in enumerate(
                    [self.gate0_pii, self.gate1_crisis, self.gate2_trauma, self.gate3_consent, self.gate4_review]
                )
            },
        }


__all__ = [
    "GateDecision",
    "GateResult",
    "GatingReport",
]
