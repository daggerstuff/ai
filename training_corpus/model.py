"""Core data model for the fresh training corpus builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

type CorpusLane = Literal["simulation", "policy", "evaluator", "benchmark"]
type InventoryDecision = Literal["keep", "defer", "reject"]
type RightsStatus = Literal["cleared", "review_required", "restricted", "unknown"]


@dataclass(frozen=True)
class CorpusSource:
    source_id: str
    registry_group: str
    family: str
    stage: str
    locator: Path
    source_type: str
    quality_profile: str | None = None
    focus: str | None = None
    inventory_decision: InventoryDecision = "defer"
    rights_status: RightsStatus = "unknown"
    license_status: str = "unknown"
    provenance_status: str = "unknown"
    benchmark_role: str = "not_eligible"
    allowed_lanes: tuple[CorpusLane, ...] = field(default_factory=tuple)
    default_lane: CorpusLane | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def corpus_id(self) -> str:
        return self.source_id


@dataclass(frozen=True)
class CorpusEntry:
    entry_id: str
    source_id: str
    stage: str
    lane: CorpusLane
    prompt: str
    response: str
    split: str
    source_family: str
    source_type: str
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def corpus_id(self) -> str:
        return self.source_id


@dataclass(frozen=True)
class CorpusManifest:
    name: str
    version: str
    destination: Path
    total_entries: int
    by_split: dict[str, int]
    by_stage: dict[str, int]
    by_corpus: dict[str, int]
    by_lane: dict[str, int]
    by_family: dict[str, int]
