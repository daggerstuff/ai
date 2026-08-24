"""CaseFlag: a bias or crisis flag raised for a specific clinician.

Each flag represents one observation from the AnalysisOrchestrator that
a clinician's session exhibited bias above a threshold. The JITTriggerEngine
accumulates flags per clinician in a rolling 7-day window and triggers
a JIT training injection when the count reaches the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class FlagType(StrEnum):
    """The kind of observation that produced this flag."""

    BIAS = "bias"
    CRISIS = "crisis"


@dataclass(frozen=True, slots=True)
class CaseFlag:
    """Immutable record of a single bias/crisis flag for a clinician.

    Attributes:
        clinician_id: The clinician this flag belongs to.
        flag_type: Whether this is a bias or crisis flag.
        timestamp: When the flag was raised (UTC).
        session_id: The session that triggered the flag.
        bias_score: The overall bias score from the analysis.
        detected_biases: List of specific bias types detected.
        receipt_root_hash: Merkle root hash of the receipt for this session.
    """

    clinician_id: str
    flag_type: FlagType
    timestamp: datetime
    session_id: str = ""
    bias_score: float = 0.0
    detected_biases: list[str] = field(default_factory=list)
    receipt_root_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for JSON storage."""
        return {
            "clinician_id": self.clinician_id,
            "flag_type": self.flag_type.value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "bias_score": self.bias_score,
            "detected_biases": list(self.detected_biases),
            "receipt_root_hash": self.receipt_root_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseFlag:
        """Deserialize from a dict (e.g. loaded from SQLite/Postgres JSON column)."""
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            clinician_id=data["clinician_id"],
            flag_type=FlagType(data["flag_type"]),
            timestamp=ts if ts.tzinfo else ts.replace(tzinfo=UTC),
            session_id=data.get("session_id", ""),
            bias_score=data.get("bias_score", 0.0),
            detected_biases=data.get("detected_biases", []),
            receipt_root_hash=data.get("receipt_root_hash", ""),
        )
