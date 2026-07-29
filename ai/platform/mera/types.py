"""Type definitions for the Mera "Memorize & Rank" clinical-prediction architecture.

These dataclasses mirror the conventions established by ``rare_disease.types``:
frozen dataclasses, an explicit ``from __future__ import annotations`` import,
``Enum`` value tiers, and ``field(default_factory=...)`` for every mutable
default.  Nothing here performs IO; the package is CPU-resolvable in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConceptLevel(Enum):
    """Granularity level of a node in the therapeutic concept hierarchy.

    The Mera paper encodes medical concepts at multiple granularity levels so
    that retrieval can fall back to a coarser parent concept when a specific
    condition has limited training examples (the *Memorize* stage) and the
    *Rank* judge can credit evidence that matches a parent concept.
    """

    CATEGORY = 0  # Level 0 — General categories (Mood Disorders, Anxiety …)
    CONDITION = 1  # Level 1 — Specific conditions (MDD, GAD, Social Anxiety …)
    SUBTYPE = 2  # Level 2 — Subtypes / specifiers (MDD atypical, seasonal …)
    CLUSTER = 3  # Level 3 — Symptom clusters (sleep, appetite, cognitive …)
    SYMPTOM = 4  # Level 4 — Individual symptoms (insomnia, hypersomnia …)


class EvidenceType(Enum):
    """The kind of patient evidence a finding represents."""

    SYMPTOM = "symptom"
    TEST = "test"
    PROGRESSION = "progression"
    DEMOGRAPHIC = "demographic"
    HISTORY = "history"


class NegativeDifficulty(Enum):
    """Difficulty tier for a contrastive negative sample.

    Harder negatives are closer in the hierarchy and therefore more
    informative for the contrastive objective (Mera §3.2).
    """

    EASY = "easy"  # different top-level category
    MEDIUM = "medium"  # same category, different condition
    HARD = "hard"  # same condition, different subtype


@dataclass(frozen=True)
class ConceptNode:
    """A single node in the therapeutic concept hierarchy."""

    node_id: str
    name: str
    level: ConceptLevel
    parent_id: str | None = None
    specificity: float = 1.0  # 0–1 edge weight = diagnostic specificity
    # Free-form descriptors used by the deterministic encoder.
    descriptors: list[str] = field(default_factory=list)
    # Mapping to a clinical condition (only set on CONDITION-level nodes).
    condition_id: str | None = None


@dataclass(frozen=True)
class TherapeuticCondition:
    """A diagnosable clinical condition anchored in the hierarchy.

    A condition sits at ``ConceptLevel.CONDITION`` and carries the profile the
    *Memorize* / *Rank* stages match against — phenotype, typical tests,
    progression pattern — directly mirroring ``RareDisease``.
    """

    condition_id: str
    name: str
    hierarchy_node_id: str  # the CONDITION-level node this condition maps to
    organ_systems: list[str] = field(default_factory=list)
    typical_symptoms: list[str] = field(default_factory=list)
    typical_tests: list[str] = field(default_factory=list)
    typical_progression: str | None = None
    prevalence_tier: str = "common"  # "common" | "uncommon" | "rare"
    # Splitting key used by the zero-shot transfer protocol (Task 6).
    train_split: bool = True


@dataclass(frozen=True)
class ClinicalFinding:
    """One piece of evidence drawn from a patient presentation."""

    finding_id: str
    text: str
    evidence_type: EvidenceType
    weight: float = 1.0
    organ_systems: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PatientPresentation:
    """A patient case presented to the Memorize & Rank pipeline.

    The Mera paper reasons from *presentations* rather than raw text; a
    presentation is the structured set of findings the system retrieves over.
    """

    case_id: str
    findings: list[ClinicalFinding]
    demographics: dict[str, Any] = field(default_factory=dict)
    additional_notes: str = ""
    # Ground-truth label used by evaluation; ``None`` for live inference.
    ground_truth_condition_id: str | None = None


@dataclass(frozen=True)
class RetrievalEvidence:
    """Why a candidate was retrieved — the transparency record."""

    source: str  # "semantic" | "hierarchical" | "keyword"
    score: float
    detail: str = ""


@dataclass(frozen=True)
class EvidenceChainLink:
    """One finding→candidate association in the evidence chain."""

    finding_id: str
    finding_text: str
    contribution: float  # weighted contribution to the candidate's score
    match_dimension: str  # "symptom_match" | "typical_presentation" | …


@dataclass(frozen=True)
class Candidate:
    """A diagnosis candidate produced by the Memorize stage and scored by Rank."""

    condition_id: str
    condition_name: str
    hierarchy_node_id: str
    retrieval_score: float
    evidence_score: float = 0.0
    final_score: float = 0.0
    retrieval_evidence: list[RetrievalEvidence] = field(default_factory=list)
    evidence_chain: list[EvidenceChainLink] = field(default_factory=list)
    hierarchy_path: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(frozen=True)
class MeraResult:
    """Final ranked output of the Memorize & Rank pipeline."""

    case_id: str
    ranked_candidates: list[Candidate]
    total_latency_ms: float
    hierarchy_used: bool
    stages: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemorizeRankConfig:
    """Runtime configuration for the pipeline (mutable — set at build time)."""

    top_k_retrieval: int = 20
    retrieval_alpha: float = 0.5  # semantic weight
    retrieval_beta: float = 0.3  # hierarchical closeness weight
    retrieval_gamma: float = 0.2  # keyword overlap weight
    rank_symptom_weight: float = 0.4
    rank_presentation_weight: float = 0.25
    rank_test_weight: float = 0.2
    rank_progression_weight: float = 0.15
    rank_retrieval_blend: float = 0.6  # blend of retrieval into final score
    prune_floor: float = 0.01
