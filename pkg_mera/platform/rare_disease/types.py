"""Type definitions for the DeepRare multi-agent diagnostic architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OrganSystem(Enum):
    """Human organ systems for disease classification."""

    CARDIOVASCULAR = "cardiovascular"
    RESPIRATORY = "respiratory"
    GASTROINTESTINAL = "gastrointestinal"
    NERVOUS = "nervous"
    ENDOCRINE = "endocrine"
    MUSCULOSKELETAL = "musculoskeletal"
    SKIN = "skin"
    HEMATOLOGIC = "hematologic"
    RENAL = "renal"
    HEPATIC = "hepatic"
    OPHTHALMIC = "ophthalmic"
    OTOLARYNGOLOGIC = "otolaryngologic"
    PSYCHIATRIC = "psychiatric"
    UTEROGENITAL = "uterogenital"
    MULTISYSTEM = "multisystem"


class DiseaseRarity(Enum):
    """Rarity tier based on prevalence."""

    ULTRA_RARE = "ultra_rare"  # < 1 in 50,000
    RARE = "rare"  # 1 in 2,000 - 1 in 50,000
    UNCOMMON = "uncommon"  # 1 in 500 - 1 in 2,000


class SymptomSeverity(Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Symptom:
    """A clinical symptom with structured metadata."""

    name: str
    severity: SymptomSeverity
    onset: str | None = None  # "acute", "chronic", "subacute"
    duration: str | None = None  # e.g. "3 months"
    associated_factors: list[str] = field(default_factory=list)
    organ_systems: list[OrganSystem] = field(default_factory=list)


@dataclass(frozen=True)
class TestFinding:
    """A single lab/imaging/genetic test finding."""

    test_name: str
    value: str
    normal_range: str | None = None
    is_abnormal: bool = False
    organ_systems: list[OrganSystem] = field(default_factory=list)


@dataclass(frozen=True)
class TestResult:
    """A collection of test findings for a single test event."""

    finding_ids: list[str] = field(default_factory=list)
    findings: list[TestFinding] = field(default_factory=list)
    reported_date: str | None = None


@dataclass(frozen=True)
class Evidence:
    """A piece of evidence supporting or opposing a diagnosis."""

    source: str  # "symptom", "lab", "imaging", "genetic", "literature"
    content: str
    weight: float  # 0.0 - 1.0
    disease_id: str | None = None


@dataclass(frozen=True)
class DifferentialEntry:
    """One entry in a ranked differential diagnosis list."""

    disease_id: str
    disease_name: str
    organ_systems: list[OrganSystem]
    rarity: DiseaseRarity
    posterior_probability: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)


@dataclass(frozen=True)
class RareDisease:
    """A rare disease profile in the knowledge base."""

    disease_id: str
    name: str
    organ_systems: list[OrganSystem]
    rarity: DiseaseRarity
    prevalence: str | None = None
    omim_id: str | None = None
    orpha_code: str | None = None
    hpo_terms: list[str] = field(default_factory=list)
    phenotype_profile: dict[str, Any] = field(default_factory=dict)
    diagnostic_criteria: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class KnowledgeMatch:
    """A result from knowledge-base retrieval."""

    disease_id: str
    score: float  # 0.0 - 1.0
    match_type: str  # "semantic", "keyword", "structured"
    source: str  # which retrieval path produced this match


@dataclass(frozen=True)
class PatientCase:
    """A patient case presented to the diagnostic system."""

    case_id: str
    symptoms: list[Symptom]
    test_results: list[TestResult] = field(default_factory=list)
    family_history: list[str] = field(default_factory=list)
    demographics: dict[str, Any] = field(default_factory=dict)
    additional_notes: str = ""


@dataclass(frozen=True)
class Hypothesis:
    """A diagnostic hypothesis with Bayesian state."""

    disease_id: str
    disease_name: str
    prior_probability: float
    posterior_probability: float
    likelihood_ratio: float = 1.0
    evidence_count: int = 0
    last_updated: str | None = None


@dataclass(frozen=True)
class OrchestrationResult:
    """Result produced by the ControllerOrchestrator after convergence."""

    case_id: str
    differential: list[DifferentialEntry]
    convergence_status: str
    iterations: int
    total_latency_ms: float
    sub_agent_results: dict[str, Any] = field(default_factory=dict)
