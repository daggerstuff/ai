"""Pydantic v2 schema models for the DeepRare multi-agent rare disease diagnosis system.

Implements the 3-tier agent architecture from arXiv 2506.20430:
- Central Controller Orchestrator
- Symptom Analyzer / Test Interpreter / Literature Matcher sub-agents
- Knowledge Retrieval Layer with HPO/ORPHA/OMIM/ICD-10/SNOMED ontology mapping

Enterprise enhancements:
- Clinical urgency and triage on PatientCase
- Audit trail with ISO timestamps on Evidence
- Confidence intervals on Hypothesis
- Clinical safety flags on DiagnosisResult
- Contraindications on TestResult
- ICD-10 and SNOMED coding on DiseaseProfile
- Wilson confidence intervals on EvaluationMetrics

All models follow the conventions established in ``patient_psi.schema``:
Pydantic v2 BaseModel, field validators for range constraints, Literal types
for enum-like fields, and aggregate models with add_* helpers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enum-like Literal types
# ---------------------------------------------------------------------------

SymptomCategory = Literal[
    "constitutional",
    "neurological",
    "cardiovascular",
    "respiratory",
    "gastrointestinal",
    "genitourinary",
    "musculoskeletal",
    "dermatological",
    "endocrine",
    "hematological",
    "psychiatric",
    "ophthalmological",
    "otolaryngological",
    "dental",
    "other",
]

OnsetType = Literal[
    "acute",
    "subacute",
    "chronic",
    "episodic",
    "congenital",
    "unknown",
    "infancy",
    "early_childhood",
    "childhood",
    "adult",
    "variable",
    "neonatal",
]
ProgressionType = Literal["improving", "stable", "worsening", "fluctuating", "stepwise", "unknown"]
SeverityLevel = Literal["mild", "moderate", "severe", "life_threatening", "unknown"]

TestType = Literal["laboratory", "imaging", "genetic", "clinical", "pathology", "electrophysiological"]
TestStatus = Literal["normal", "abnormal", "borderline", "pending", "inconclusive"]

HypothesisStatus = Literal["active", "eliminated", "confirmed", "pending_verification"]
AgentName = Literal["symptom_analyzer", "test_interpreter", "literature_matcher", "orchestrator"]

OrganSystem = Literal[
    "neurological",
    "cardiovascular",
    "respiratory",
    "gastrointestinal",
    "genitourinary",
    "musculoskeletal",
    "dermatological",
    "endocrine",
    "hematological",
    "metabolic",
    "immune",
    "multi_system",
    "renal",
    "other",
]

OntologySource = Literal["HPO", "ORPHA", "OMIM", "ICD10", "SNOMED", "UMLS"]

RarityTier = Literal[
    "ultra_rare",  # <1 per 100,000
    "rare",  # 1-9 per 100,000
    "less_common",  # 10-50 per 100,000
    "moderately_rare",  # 50-100 per 100,000
]

PresentationComplexity = Literal["straightforward", "moderate", "complex", "atypical"]

ClinicalUrgency = Literal[
    "routine",  # Standard diagnostic workup
    "urgent",  # Needs evaluation within 24-48h
    "emergent",  # Needs immediate evaluation (< 1h)
    "life_threatening",  # Critical, requires emergency intervention
]

ConfidenceLevel = Literal[
    "very_low",  # < 0.25
    "low",  # 0.25 - 0.50
    "moderate",  # 0.50 - 0.75
    "high",  # 0.75 - 0.90
    "very_high",  # > 0.90
]

EvidenceGrade = Literal[
    "A",  # High-quality evidence (systematic reviews, RCTs)
    "B",  # Moderate-quality (cohort studies, case-control)
    "C",  # Low-quality (case reports, expert opinion)
    "D",  # Very low-quality (anecdotal, insufficient)
]


# ---------------------------------------------------------------------------
# Core Domain Models
# ---------------------------------------------------------------------------


class SymptomProfile(BaseModel):
    """Structured representation of a patient symptom with clinical metadata."""

    name: str = Field(..., min_length=1, max_length=500, description="Symptom name or description")
    category: SymptomCategory = Field(..., description="Body system category")
    onset: OnsetType = Field(default="unknown", description="Onset pattern")
    duration_days: float | None = Field(default=None, ge=0, description="Duration in days if known")
    severity: SeverityLevel = Field(default="unknown", description="Current severity")
    progression: ProgressionType = Field(default="unknown", description="Progression pattern")
    aggravating_factors: list[str] = Field(default_factory=list, description="Factors that worsen symptom")
    relieving_factors: list[str] = Field(default_factory=list, description="Factors that improve symptom")
    associated_symptoms: list[str] = Field(default_factory=list, description="Frequently co-occurring symptoms")
    is_pathognomonic: bool = Field(
        default=False,
        description="True if this symptom is pathognomonic for a specific condition",
    )
    hpo_term: str | None = Field(default=None, description="HPO ontology term ID if mapped")
    notes: str = Field(default="", description="Additional clinical notes")

    @field_validator("duration_days")
    @classmethod
    def _check_duration(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("duration_days must be non-negative")
        return value


class TestResult(BaseModel):
    """Result of a clinical test (lab, imaging, genetic, etc.)."""

    test_name: str = Field(..., min_length=1, max_length=300, description="Name of the test")
    test_type: TestType = Field(..., description="Category of test")
    status: TestStatus = Field(..., description="Result status")
    value: str = Field(default="", description="Test value or finding description")
    reference_range: str = Field(default="", description="Normal reference range")
    unit: str = Field(default="", description="Unit of measurement")
    interpretation: str = Field(default="", description="Clinical interpretation")
    is_abnormal: bool = Field(default=False, description="True if result is outside normal range")
    requested_by: AgentName | None = Field(default=None, description="Which agent requested this test")
    diagnostic_criteria_mapping: list[str] = Field(
        default_factory=list,
        description="Diagnostic criteria this test maps to",
    )
    contraindications: list[str] = Field(
        default_factory=list,
        description="Contraindications for this test (patient factors that preclude testing)",
    )
    sensitivity: float | None = Field(default=None, ge=0.0, le=1.0, description="Test sensitivity if known")
    specificity: float | None = Field(default=None, ge=0.0, le=1.0, description="Test specificity if known")

    @field_validator("test_name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("test_name must not be empty or whitespace-only")
        return value.strip()


class Evidence(BaseModel):
    """A piece of evidence supporting or refuting a diagnostic hypothesis.

    Enterprise fields:
    - ``timestamp``: ISO 8601 UTC timestamp for audit trail
    - ``confidence_level``: Qualitative confidence assessment
    - ``provenance``: Full provenance chain for traceability
    - ``evidence_grade``: GRADE-based evidence quality
    """

    source: AgentName = Field(..., description="Which agent produced this evidence")
    description: str = Field(..., min_length=1, description="What the evidence shows")
    supports: bool = Field(..., description="True if supports the hypothesis, False if refutes")
    weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Evidence weight 0-1")
    test_result_id: str | None = Field(default=None, description="ID of associated test result if any")
    literature_ref: str | None = Field(default=None, description="Literature reference if applicable")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 UTC timestamp when this evidence was collected",
    )
    confidence_level: ConfidenceLevel = Field(
        default="moderate",
        description="Qualitative confidence in this evidence",
    )
    provenance: str = Field(
        default="",
        description="Full provenance: agent → method → source data",
    )
    evidence_grade: EvidenceGrade = Field(
        default="C",
        description="GRADE evidence quality assessment",
    )

    @field_validator("weight")
    @classmethod
    def _check_weight(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("weight must be between 0.0 and 1.0")
        return value

    @field_validator("timestamp")
    @classmethod
    def _check_iso_format(cls, value: str) -> str:
        """Validate ISO 8601 format for audit compliance."""
        if value:
            datetime.fromisoformat(value)
        return value


class Hypothesis(BaseModel):
    """A diagnostic hypothesis for a specific rare disease."""

    disease_name: str = Field(..., min_length=1, max_length=500, description="Name of the disease")
    disease_id: str | None = Field(default=None, description="ORPHA/OMIM identifier if known")
    prior_probability: float = Field(default=0.01, ge=0.0, le=1.0, description="Prior probability before evidence")
    posterior_probability: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Posterior probability after Bayesian updating"
    )
    evidence_list: list[Evidence] = Field(default_factory=list, description="Supporting/refuting evidence")
    status: HypothesisStatus = Field(default="active", description="Current status")
    organ_system: OrganSystem = Field(default="multi_system", description="Primary organ system")
    rarity_tier: RarityTier = Field(default="rare", description="Disease rarity classification")
    matching_symptoms: list[str] = Field(
        default_factory=list,
        description="Symptoms that match this disease profile",
    )
    missing_symptoms: list[str] = Field(
        default_factory=list,
        description="Expected symptoms not yet observed",
    )
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall confidence in this hypothesis")
    confidence_interval_lower: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Lower bound of 95% confidence interval"
    )
    confidence_interval_upper: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Upper bound of 95% confidence interval"
    )
    is_life_threatening: bool = Field(
        default=False,
        description="True if this disease can be life-threatening if untreated",
    )
    iteration_created: int = Field(default=0, ge=0, description="Iteration when this hypothesis was first created")

    @field_validator(
        "prior_probability",
        "posterior_probability",
        "confidence_score",
        "confidence_interval_lower",
        "confidence_interval_upper",
    )
    @classmethod
    def _check_probability(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("probability values must be between 0.0 and 1.0")
        return value

    def add_evidence(self, evidence: Evidence) -> Hypothesis:
        """Append evidence and return self for chaining."""
        self.evidence_list.append(evidence)
        return self

    def supporting_evidence(self) -> list[Evidence]:
        """Return only evidence that supports this hypothesis."""
        return [e for e in self.evidence_list if e.supports]

    def refuting_evidence(self) -> list[Evidence]:
        """Return only evidence that refutes this hypothesis."""
        return [e for e in self.evidence_list if not e.supports]


class RareDiseaseState(BaseModel):
    """Central state maintained by the ControllerOrchestrator.

    Tracks active hypotheses, eliminated conditions, pending inquiries,
    and convergence state across diagnostic iterations.
    """

    iteration: int = Field(default=0, ge=0, description="Current iteration count")
    active_hypotheses: list[Hypothesis] = Field(default_factory=list, description="Currently active hypotheses")
    eliminated_conditions: list[str] = Field(
        default_factory=list, description="Disease names eliminated from consideration"
    )
    pending_inquiries: list[str] = Field(default_factory=list, description="Additional information/tests requested")
    evidence_strength: dict[str, float] = Field(
        default_factory=dict, description="Disease name → aggregate evidence strength"
    )
    top_hypotheses_history: list[list[str]] = Field(
        default_factory=list,
        description="History of top-5 disease name lists for convergence check",
    )
    is_converged: bool = Field(default=False, description="True if convergence criteria met")
    max_iterations: int = Field(default=10, ge=1, le=50, description="Maximum allowed iterations")
    convergence_window: int = Field(default=3, ge=1, le=10, description="Iterations top-5 must remain stable")

    def add_hypothesis(self, hypothesis: Hypothesis) -> RareDiseaseState:
        """Add a new hypothesis to active list."""
        self.active_hypotheses.append(hypothesis)
        return self

    def eliminate(self, disease_name: str) -> RareDiseaseState:
        """Move a hypothesis from active to eliminated."""
        self.active_hypotheses = [h for h in self.active_hypotheses if h.disease_name != disease_name]
        if disease_name not in self.eliminated_conditions:
            self.eliminated_conditions.append(disease_name)
        return self

    def add_inquiry(self, inquiry: str) -> RareDiseaseState:
        """Add a pending inquiry for additional information/tests."""
        if inquiry not in self.pending_inquiries:
            self.pending_inquiries.append(inquiry)
        return self

    def resolve_inquiry(self, inquiry: str) -> RareDiseaseState:
        """Remove a resolved inquiry."""
        self.pending_inquiries = [i for i in self.pending_inquiries if i != inquiry]
        return self

    def record_top_hypotheses(self) -> RareDiseaseState:
        """Record current top-5 hypotheses for convergence tracking."""
        sorted_hyps = sorted(self.active_hypotheses, key=lambda h: h.posterior_probability, reverse=True)
        top5 = [h.disease_name for h in sorted_hyps[:5]]
        self.top_hypotheses_history.append(top5)
        return self

    def check_convergence(self) -> bool:
        """Check if top-5 hypotheses have been stable for ``convergence_window`` iterations."""
        if len(self.top_hypotheses_history) < self.convergence_window:
            return False
        recent = self.top_hypotheses_history[-self.convergence_window :]
        if len(recent) < 2:
            return False
        first = recent[0]
        return all(h == first for h in recent[1:])

    def to_dict(self) -> dict[str, object]:
        """Serialize state to dictionary."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# Sub-Agent Response Models
# ---------------------------------------------------------------------------


class SymptomAnalysisResult(BaseModel):
    """Output from the Symptom Analyzer sub-agent."""

    agent: Literal["symptom_analyzer"] = "symptom_analyzer"
    symptom_disease_matrix: dict[str, float] = Field(
        default_factory=dict,
        description="Disease name → symptom-match probability (0-1)",
    )
    identified_pathognomonic: list[str] = Field(default_factory=list, description="Pathognomonic symptoms identified")
    symptom_clusters: list[list[str]] = Field(default_factory=list, description="Clusters of co-occurring symptoms")
    temporal_progression_analysis: str = Field(default="", description="Analysis of symptom temporal patterns")
    new_hypotheses: list[Hypothesis] = Field(
        default_factory=list, description="New disease hypotheses generated from symptom analysis"
    )
    recommended_inquiries: list[str] = Field(
        default_factory=list, description="Additional patient history questions recommended"
    )
    reasoning: str = Field(default="", description="Agent's reasoning narrative")

    @field_validator("symptom_disease_matrix")
    @classmethod
    def _check_matrix_values(cls, value: dict[str, float]) -> dict[str, float]:
        for disease, prob in value.items():
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"Probability for {disease} must be 0.0-1.0, got {prob}")
        return value


class TestInterpretationResult(BaseModel):
    """Output from the Test Interpreter sub-agent."""

    agent: Literal["test_interpreter"] = "test_interpreter"
    updated_probabilities: dict[str, float] = Field(
        default_factory=dict,
        description="Disease name → Bayesian-updated posterior probability",
    )
    likelihood_ratios: dict[str, float] = Field(
        default_factory=dict, description="Disease name → likelihood ratio from this test"
    )
    diagnostic_criteria_met: dict[str, list[str]] = Field(
        default_factory=dict, description="Disease name → list of criteria now satisfied"
    )
    additional_tests_recommended: list[TestResult] = Field(
        default_factory=list, description="New tests recommended based on current differential"
    )
    eliminated_hypotheses: list[str] = Field(
        default_factory=list, description="Diseases eliminated based on test results"
    )
    reasoning: str = Field(default="", description="Agent's reasoning narrative")

    @field_validator("updated_probabilities", "likelihood_ratios")
    @classmethod
    def _check_prob_dict(cls, value: dict[str, float]) -> dict[str, float]:
        for k, v in value.items():
            if not isinstance(v, (int, float)):
                raise ValueError(f"Value for {k} must be numeric, got {type(v)}")
        return value


class LiteratureMatch(BaseModel):
    """A single literature/case report match."""

    title: str = Field(..., min_length=1, description="Publication or case report title")
    authors: str = Field(default="", description="Author list")
    source: str = Field(default="", description="Journal or database source")
    year: int | None = Field(default=None, ge=1900, le=2100, description="Publication year")
    similarity_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Semantic similarity to patient profile (0-1)"
    )
    matched_disease: str = Field(default="", description="Disease identified in the literature")
    diagnostic_criteria: list[str] = Field(default_factory=list, description="Diagnostic criteria extracted from match")
    treatment_implications: str = Field(default="", description="Treatment implications from the match")
    url: str = Field(default="", description="Link to source if available")
    match_type: Literal["semantic", "keyword", "hybrid"] = Field(
        default="hybrid", description="How this match was found"
    )


class LiteratureSearchResult(BaseModel):
    """Output from the Literature Matcher sub-agent."""

    agent: Literal["literature_matcher"] = "literature_matcher"
    matches: list[LiteratureMatch] = Field(default_factory=list, description="Matching case reports and literature")
    rare_disease_matches: list[LiteratureMatch] = Field(
        default_factory=list, description="Matches specifically for rare diseases"
    )
    diagnostic_criteria_extracted: dict[str, list[str]] = Field(
        default_factory=dict, description="Disease name → extracted diagnostic criteria"
    )
    average_similarity: float = Field(default=0.0, ge=0.0, le=1.0, description="Average similarity across all matches")
    coverage_percentage: float = Field(
        default=0.0, ge=0.0, le=100.0, description="% of rare diseases with at least one match"
    )
    reasoning: str = Field(default="", description="Agent's reasoning narrative")


# ---------------------------------------------------------------------------
# Knowledge Base Models
# ---------------------------------------------------------------------------


class DiagnosticCriterion(BaseModel):
    """A single diagnostic criterion for a disease."""

    name: str = Field(..., min_length=1, description="Criterion name")
    description: str = Field(..., min_length=1, description="What the criterion requires")
    is_required: bool = Field(default=False, description="True if this is a mandatory criterion")
    test_type: TestType | None = Field(default=None, description="Type of test to verify this criterion")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Weight in scoring")


class DiseaseProfile(BaseModel):
    """Structured profile of a rare disease in the knowledge base."""

    name: str = Field(..., min_length=1, max_length=500, description="Disease name")
    orpha_id: str | None = Field(default=None, description="Orphanet identifier")
    omim_id: str | None = Field(default=None, description="OMIM identifier")
    icd10_code: str | None = Field(default=None, description="ICD-10 code")
    snomed_code: str | None = Field(default=None, description="SNOMED CT concept ID")
    mondo_id: str | None = Field(default=None, description="MONDO disease ontology ID")
    umls_cui: str | None = Field(default=None, description="UMLS Concept Unique Identifier")
    hpo_terms: list[str] = Field(default_factory=list, description="Associated HPO terms")
    organ_system: OrganSystem = Field(default="multi_system", description="Primary organ system")
    rarity_tier: RarityTier = Field(default="rare", description="Disease rarity")
    prevalence: float = Field(default=0.0, ge=0.0, description="Prevalence per 100,000 population")
    pathognomonic_symptoms: list[str] = Field(
        default_factory=list, description="Pathognomonic symptoms unique to this disease"
    )
    common_symptoms: list[str] = Field(default_factory=list, description="Frequently occurring symptoms")
    rare_symptoms: list[str] = Field(default_factory=list, description="Less common but possible symptoms")
    diagnostic_criteria: list[DiagnosticCriterion] = Field(
        default_factory=list, description="Formal diagnostic criteria"
    )
    typical_onset: OnsetType = Field(default="unknown", description="Typical onset pattern")
    gene_associations: list[str] = Field(default_factory=list, description="Associated genes if genetic disease")
    treatment_guidelines: str = Field(default="", description="Brief treatment guidelines")
    differential_diagnoses: list[str] = Field(default_factory=list, description="Diseases to differentiate from")


# ---------------------------------------------------------------------------
# Patient Case and Differential Diagnosis
# ---------------------------------------------------------------------------


class PatientCase(BaseModel):
    """Input patient case for the diagnostic pipeline."""

    case_id: str = Field(..., min_length=1, description="Unique case identifier")
    patient_age: int | None = Field(default=None, ge=0, le=150, description="Patient age in years")
    patient_sex: Literal["male", "female", "intersex", "unknown"] = Field(
        default="unknown", description="Biological sex"
    )
    presenting_symptoms: list[SymptomProfile] = Field(default_factory=list, description="Initial presenting symptoms")
    medical_history: list[str] = Field(default_factory=list, description="Relevant medical history items")
    family_history: list[str] = Field(default_factory=list, description="Family history items")
    current_medications: list[str] = Field(default_factory=list, description="Current medications")
    available_tests: list[TestResult] = Field(default_factory=list, description="Already-available test results")
    clinical_notes: str = Field(default="", description="Additional clinical notes")
    presentation_complexity: PresentationComplexity = Field(
        default="moderate", description="Case complexity assessment"
    )
    clinical_urgency: ClinicalUrgency = Field(
        default="routine",
        description="Clinical urgency/triage level",
    )
    referral_required: bool = Field(
        default=False,
        description="True if immediate clinical referral is required",
    )
    consent_given: bool = Field(
        default=True,
        description="True if patient has consented to AI-assisted diagnosis (HIPAA)",
    )
    phi_protected: bool = Field(
        default=True,
        description="True if PHI protections are in place for this case",
    )
    ground_truth_diagnosis: str | None = Field(default=None, description="Correct diagnosis (for evaluation only)")

    @field_validator("patient_age")
    @classmethod
    def _check_age(cls, value: int | None) -> int | None:
        if value is not None and (value < 0 or value > 150):
            raise ValueError("patient_age must be between 0 and 150")
        return value


class RankedDiagnosis(BaseModel):
    """A single ranked entry in the differential diagnosis."""

    rank: int = Field(..., ge=1, description="Rank position (1 = most likely)")
    disease_name: str = Field(..., min_length=1, description="Disease name")
    probability: float = Field(default=0.0, ge=0.0, le=1.0, description="Posterior probability")
    evidence_summary: str = Field(default="", description="Summary of supporting evidence")
    evidence_count: int = Field(default=0, ge=0, description="Number of evidence items")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in this ranking")
    is_pathognomonic_match: bool = Field(default=False, description="True if pathognomonic symptom matched")


class DifferentialDiagnosis(BaseModel):
    """Ranked differential diagnosis list with evidence strength per condition."""

    ranked_list: list[RankedDiagnosis] = Field(
        default_factory=list, description="Diseases ranked by posterior probability"
    )
    eliminated: list[str] = Field(default_factory=list, description="Eliminated diseases with reason")
    total_hypotheses_considered: int = Field(default=0, ge=0, description="Total hypotheses evaluated")
    iterations_used: int = Field(default=0, ge=0, description="Diagnostic iterations performed")
    convergence_achieved: bool = Field(default=False, description="True if convergence criteria were met")
    reasoning_trace: str = Field(default="", description="Narrative trace of diagnostic reasoning")

    def top_n(self, n: int = 5) -> list[RankedDiagnosis]:
        """Return the top-N diagnoses."""
        return self.ranked_list[:n]

    def top_disease(self) -> RankedDiagnosis | None:
        """Return the top-ranked diagnosis or None if empty."""
        return self.ranked_list[0] if self.ranked_list else None

    def to_dict(self) -> dict[str, object]:
        """Serialize to dictionary."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# Evaluation Models
# ---------------------------------------------------------------------------


class EvaluationMetrics(BaseModel):
    """Metrics for evaluating diagnostic performance (DiagnosisArena adapter).

    Enterprise fields:
    - ``recall_at_1_ci`` etc.: Wilson 95% confidence intervals
    - ``safety_violation_count``: Safety rules violated during eval
    - ``cases_requiring_referral``: Cases flagged for clinical referral
    """

    recall_at_1: float = Field(default=0.0, ge=0.0, le=1.0, description="Recall@1 — top-1 accuracy")
    recall_at_5: float = Field(default=0.0, ge=0.0, le=1.0, description="Recall@5 — top-5 accuracy")
    recall_at_10: float = Field(default=0.0, ge=0.0, le=1.0, description="Recall@10 — top-10 accuracy")
    mrr: float = Field(default=0.0, ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    accuracy_by_organ: dict[str, float] = Field(default_factory=dict, description="Accuracy by organ system")
    accuracy_by_rarity: dict[str, float] = Field(default_factory=dict, description="Accuracy by rarity tier")
    accuracy_by_complexity: dict[str, float] = Field(
        default_factory=dict, description="Accuracy by presentation complexity"
    )
    avg_iterations: float = Field(default=0.0, ge=0.0, description="Average iterations to convergence")
    avg_time_seconds: float = Field(default=0.0, ge=0.0, description="Average end-to-end time per case")
    total_cases: int = Field(default=0, ge=0, description="Total cases evaluated")
    correct_cases: int = Field(default=0, ge=0, description="Cases with correct top-1 diagnosis")
    # Wilson 95% confidence intervals
    recall_at_1_ci_lower: float = Field(default=0.0, ge=0.0, le=1.0, description="Wilson CI lower for Recall@1")
    recall_at_1_ci_upper: float = Field(default=0.0, ge=0.0, le=1.0, description="Wilson CI upper for Recall@1")
    recall_at_5_ci_lower: float = Field(default=0.0, ge=0.0, le=1.0, description="Wilson CI lower for Recall@5")
    recall_at_5_ci_upper: float = Field(default=0.0, ge=0.0, le=1.0, description="Wilson CI upper for Recall@5")
    mrr_ci_lower: float = Field(default=0.0, ge=0.0, le=1.0, description="Wilson CI lower for MRR")
    mrr_ci_upper: float = Field(default=0.0, ge=0.0, le=1.0, description="Wilson CI upper for MRR")
    # Safety metrics
    safety_violation_count: int = Field(default=0, ge=0, description="Total safety rule violations across all cases")
    cases_requiring_referral: int = Field(default=0, ge=0, description="Cases flagged for immediate clinical referral")
    cases_with_life_threatening_conditions: int = Field(
        default=0, ge=0, description="Cases involving life-threatening conditions"
    )
    avg_clinical_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Average clinical confidence across all cases"
    )
    # Per-case error analysis
    error_cases: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Details of cases where diagnosis was incorrect",
    )


class DiagnosisResult(BaseModel):
    """Complete result of running the diagnostic pipeline on a patient case.

    Enterprise fields:
    - ``safety_flags``: Clinical safety warnings detected during diagnosis
    - ``audit_trail``: Full provenance chain for clinical review
    - ``safety_violations``: Safety rules that were violated
    - ``clinical_confidence``: Overall confidence in the diagnostic conclusion
    - ``requires_human_review``: True if human clinical review is mandated
    """

    case_id: str = Field(..., min_length=1, description="Case identifier")
    differential: DifferentialDiagnosis = Field(..., description="Ranked differential diagnosis")
    state: RareDiseaseState = Field(..., description="Final diagnostic state")
    iterations: int = Field(default=0, ge=0, description="Iterations performed")
    time_seconds: float = Field(default=0.0, ge=0.0, description="End-to-end time")
    converged: bool = Field(default=False, description="Whether convergence was achieved")
    agent_outputs: dict[str, str] = Field(default_factory=dict, description="Agent name → reasoning trace")
    recommended_next_steps: list[str] = Field(default_factory=list, description="Recommended next clinical steps")
    evaluation: EvaluationMetrics | None = Field(
        default=None, description="Evaluation metrics if ground truth available"
    )
    safety_flags: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Clinical safety warnings detected (red flags, contraindications, etc.)",
    )
    audit_trail: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Full audit trail of clinical decisions for regulatory compliance",
    )
    safety_violations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Safety rules that were violated during diagnosis",
    )
    clinical_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the diagnostic conclusion",
    )
    requires_human_review: bool = Field(
        default=True,
        description="True if human clinical review is mandated (always true for rare disease diagnosis)",
    )
    phi_deidentified: bool = Field(
        default=True,
        description="True if PHI has been de-identified in this result",
    )

    def to_dict(self) -> dict[str, object]:
        """Serialize to dictionary."""
        return self.model_dump()
