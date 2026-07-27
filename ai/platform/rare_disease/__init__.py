"""Multi-agent architecture for rare-disease diagnosis.

This package implements a 3-tier diagnostic agent architecture inspired by the
DeepRare paper (arXiv 2506.20430), which achieves 57.18% Recall@1 across
2,919 rare diseases — a 23.79% absolute improvement over the next-best single-
agent baseline.  The architecture decomposes clinical reasoning into:

* A :class:`ControllerOrchestrator` (central) that owns the differential and
  sequences sub-agent invocations until the diagnostic list converges.
* Three interchangeable sub-agents:
  :class:`~rare_disease.agents.symptom_analyzer.SymptomAnalyzerAgent`,
  :class:`~rare_disease.agents.test_interpreter.TestInterpreterAgent`, and
  :class:`~rare_disease.agents.literature_matcher.LiteratureMatcherAgent`.
* A pluggable :class:`RareDiseaseKnowledgeBase` whose default implementation
  provides hybrid retrieval (semantic + keyword + structured) over an in-memory
  corpus, with hooks for HPO/ORPHA/OMIM ontology loaders.
* A :class:`DifferentialDiagnosisManager` that applies Bayesian posterior
  updates, prunes conditions whose posterior drops below a configurable floor,
  and reports convergence.
* A :class:`DiagnosisArenaAdapter` evaluation harness that emits the metrics
  used by the DeepRare paper (Recall@1, Recall@5, Recall@10, MRR).

The package is purely CPU-resolvable in tests: language-model calls go through
a swappable :class:`~rare_disease.types.LanguageModelCallable`.  Production
deployments wire the real provider; tests use a deterministic stub.
"""

from rare_disease.differential import DifferentialDiagnosisManager
from rare_disease.evaluation import DiagnosisArenaAdapter, DiagnosisArenaMetrics
from rare_disease.knowledge_base import (
    InMemoryRareDiseaseKnowledgeBase,
    RareDiseaseKnowledgeBase,
)
from rare_disease.orchestrator import ControllerOrchestrator, OrchestrationResult
from rare_disease.pipeline import RareDiseasePipeline, build_default_pipeline
from rare_disease.state import ConvergenceStatus, RareDiseaseState
from rare_disease.types import (
    DifferentialEntry,
    DiseaseRarity,
    Evidence,
    Hypothesis,
    KnowledgeMatch,
    OrganSystem,
    PatientCase,
    RareDisease,
    Symptom,
    SymptomSeverity,
    TestFinding,
    TestResult,
)

__all__ = [
    "ConvergenceStatus",
    "ControllerOrchestrator",
    "DiagnosisArenaAdapter",
    "DiagnosisArenaMetrics",
    "DifferentialDiagnosisManager",
    "DifferentialEntry",
    "DiseaseRarity",
    "Evidence",
    "Hypothesis",
    "InMemoryRareDiseaseKnowledgeBase",
    "KnowledgeMatch",
    "OrchestrationResult",
    "OrganSystem",
    "PatientCase",
    "RareDisease",
    "RareDiseaseKnowledgeBase",
    "RareDiseasePipeline",
    "RareDiseaseState",
    "Symptom",
    "SymptomSeverity",
    "TestFinding",
    "TestResult",
    "build_default_pipeline",
]
