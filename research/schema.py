"""
PIX-510: Sprint 1 - Memory Schema & Unification
Canonical memory block schema for therapeutic AI system.

Python dataclass mirror of src/types/memory.ts — keep in sync.
JSON Schema: ai/research/schema.json
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

# ─── Enumerations (must match TypeScript counterparts exactly) ────────────────


class PIIStatus(StrEnum):
    ABSENT = "absent"
    REDACTED = "redacted"
    PRESENT = "present"


class ConsentGate(StrEnum):
    OPEN = "open"
    RESTRICTED = "restricted"
    BLOCKED = "blocked"


class ConsolidationPhase(StrEnum):
    RAW = "raw"
    CONSOLIDATED = "consolidated"
    ARCHIVED = "archived"
    LATENT = "latent"
    FORGOTTEN = "forgotten"


# ─── Nested value objects ──────────────────────────────────────────────────────


class MemoryImportance(BaseModel):
    """Importance scoring breakdown — mirrors MemoryImportance in memory.ts"""

    raw: float = Field(ge=0.0, le=1.0, description="Computed composite score")
    recency: float = Field(ge=0.0, le=1.0, description="Exponential decay factor (τ=7d)")
    relevance: float = Field(ge=0.0, le=1.0, description="Cosine similarity to query")
    emotionalWeight: float = Field(ge=1.0, le=5.0, description="Crisis multiplier (1.0=normal, 5.0=crisis)")
    actionability: float = Field(ge=0.0, le=1.0, description="Goal-relevance score")
    reveriePotential: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Potential for subconscious surfacing as reverie"
    )


class MemoryEmotions(BaseModel):
    """Emotional tagging — mirrors MemoryEmotions in memory.ts"""

    valence: float = Field(ge=-1.0, le=1.0, description="Negative to positive affect")
    arousal: float = Field(ge=0.0, le=1.0, description="Calm to intense activation")
    categories: list[str] = Field(
        default_factory=list,
        description="Plutchik wheel: joy, sadness, anger, fear, surprise, disgust, trust, anticipation",
    )


class MemoryGating(BaseModel):
    """Safety and consent gating metadata — mirrors MemoryGating in memory.ts"""

    piiStatus: PIIStatus = PIIStatus.ABSENT
    crisisFlag: bool = False
    traumaIndicators: list[str] = Field(default_factory=list)
    consentGate: ConsentGate = ConsentGate.OPEN


class MemoryConsolidation(BaseModel):
    """Consolidation / lifecycle state — mirrors MemoryConsolidation in memory.ts"""

    phase: ConsolidationPhase = ConsolidationPhase.RAW
    lastProcessed: int = Field(ge=0, description="Unix timestamp ms")
    remCycles: int = Field(ge=0, description="Remaining consolidation cycles")
    schemaReferences: list[str] = Field(default_factory=list, description="Prior schema version pointers")
    reverieEligible: bool = Field(default=False, description="Whether this memory can surface as a reverie")
    reveriePhase: str = Field(
        default="dormant", description="Reverie lifecycle phase: dormant|seeded|surfacing|active|fading"
    )


# ─── Primary entity ────────────────────────────────────────────────────────────


class MemoryBlock(BaseModel):
    """
    Canonical memory block — all fields required for tenant isolation and safety.
    Mirrors MemoryBlock interface in src/types/memory.ts
    """

    id: str
    tenantId: str
    sessionId: str
    content: str
    timestamp: int = Field(ge=0, description="Unix timestamp ms")

    importance: MemoryImportance = Field(
        default_factory=lambda: MemoryImportance(
            raw=0.0, recency=0.0, relevance=0.0, emotionalWeight=1.0, actionability=0.0, reveriePotential=0.0
        )
    )
    emotions: MemoryEmotions = Field(default_factory=lambda: MemoryEmotions(valence=0.0, arousal=0.0, categories=[]))
    gating: MemoryGating = Field(
        default_factory=lambda: MemoryGating(
            piiStatus=PIIStatus.ABSENT,
            crisisFlag=False,
            traumaIndicators=[],
            consentGate=ConsentGate.OPEN,
        )
    )
    consolidation: MemoryConsolidation = Field(
        default_factory=lambda: MemoryConsolidation(
            phase=ConsolidationPhase.RAW,
            lastProcessed=0,
            remCycles=0,
            schemaReferences=[],
            reverieEligible=False,
            reveriePhase="dormant",
        )
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "mem_01HX4...",
                "tenantId": "tenant_abc",
                "sessionId": "sess_xyz",
                "content": "Client expressed anxiety about upcoming session transition",
                "timestamp": 1715600000000,
                "importance": {
                    "raw": 0.72,
                    "recency": 0.85,
                    "relevance": 0.91,
                    "emotionalWeight": 2.0,
                    "actionability": 0.65,
                    "reveriePotential": 0.0,
                },
                "emotions": {
                    "valence": -0.3,
                    "arousal": 0.7,
                    "categories": ["anxiety", "fear"],
                },
                "gating": {
                    "piiStatus": "absent",
                    "crisisFlag": False,
                    "traumaIndicators": [],
                    "consentGate": "open",
                },
                "consolidation": {
                    "phase": "raw",
                    "lastProcessed": 1715600000000,
                    "remCycles": 3,
                    "schemaReferences": [],
                    "reverieEligible": False,
                    "reveriePhase": "dormant",
                },
            }
        }
    }


# ─── Supporting types ─────────────────────────────────────────────────────────


class MemoryRef(BaseModel):
    """Lightweight memory reference for search results and listings."""

    id: str
    tenantId: str
    sessionId: str
    content: str  # Truncated / redacted
    timestamp: int
    importance_raw: float = Field(alias="importance.raw", ge=0.0, le=1.0)
    emotions_valence: float = Field(alias="emotions.valence", ge=-1.0, le=1.0)
    crisisFlag: bool

    model_config = {"populate_by_name": True}


class MemorySearchFilters(BaseModel):
    """Search and query filter parameters."""

    tenantId: str
    sessionId: str | None = None
    minImportance: float | None = Field(default=None, ge=0.0, le=1.0)
    maxImportance: float | None = Field(default=None, ge=0.0, le=1.0)
    emotions: list[str] | None = None
    crisisOnly: bool = False
    dateFrom: int | None = Field(default=None, ge=0)
    dateTo: int | None = Field(default=None, ge=0)
    consolidationPhases: list[ConsolidationPhase] | None = None
    limit: int = Field(default=50, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class MemoryWriteInput(BaseModel):
    """Memory write input — id is server-generated if omitted."""

    tenantId: str
    sessionId: str
    content: str
    emotions: MemoryEmotions | None = None
    gating: MemoryGating | None = None


@dataclass
class ScoringWeights:
    """
    Importance scoring weights — mirrors ScoringWeights in memory.ts.
    Configurable via environment variables.
    """

    alpha: float = 0.25  # recency weight
    beta: float = 0.25  # relevance weight
    gamma: float = 0.30  # emotional weight
    delta: float = 0.20  # actionability weight
    decay_tau_days: float = 7.0  # exponential decay time constant

    @classmethod
    def from_env(cls) -> ScoringWeights:
        import os

        return cls(
            alpha=float(os.environ.get("MEMORY_SCORE_ALPHA", "0.25")),
            beta=float(os.environ.get("MEMORY_SCORE_BETA", "0.25")),
            gamma=float(os.environ.get("MEMORY_SCORE_GAMMA", "0.30")),
            delta=float(os.environ.get("MEMORY_SCORE_DELTA", "0.20")),
            decay_tau_days=float(os.environ.get("MEMORY_DECAY_TAU_DAYS", "7.0")),
        )

    def compute_importance(
        self,
        recency: float,
        relevance: float,
        emotional_weight: float,
        actionability: float,
    ) -> float:
        """Compute composite importance score using weighted formula."""
        raw = (
            self.alpha * recency
            + self.beta * relevance
            + self.gamma * (emotional_weight / 5.0)  # normalise to [0,1]
            + self.delta * actionability
        )
        return round(min(raw, 1.0), 6)

    @staticmethod
    def decay_factor(age_seconds: float, tau_days: float = 7.0) -> float:
        """Exponential decay: e^(-age / tau)."""
        tau_seconds = tau_days * 86400
        return math.exp(-age_seconds / tau_seconds)
