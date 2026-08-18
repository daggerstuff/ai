"""DeepRare: Multi-Agent Rare Disease Diagnosis Architecture.

Implements the 3-tier agent system from arXiv 2506.20430 for rare disease
diagnosis with a central controller orchestrating specialised sub-agents
(symptom analysis, test interpretation, literature matching) over a
shared knowledge base with Bayesian differential diagnosis management.

Enterprise features:
- Clinical safety gates with red-flag detection and audit trails
- Environment-based configuration with safety thresholds and feature flags
- Structured JSON logging, metrics collection, and FHIR audit export
- Wilson confidence intervals, per-case error analysis, statistical significance
- Thread-safe knowledge base caching, TF-IDF literature matching, GRADE evidence

Built on the Pixelated Empathy platform, extending the multi-agent
framework alongside the Patient-Ψ cognitive simulation engine (PIX-3906).

Reference: PIX-3907 — [DeepRare] Multi-Agent Rare Disease Diagnosis.
"""

from __future__ import annotations

from .agents import LiteratureMatcher, SymptomAnalyzer, TestInterpreter
from .clinical_safety import (
    AuditAction,
    AuditEntry,
    AuditTrail,
    ClinicalSafetyContext,
    ClinicalSafetyGate,
    RedFlag,
    RedFlagDetector,
    SafetyLevel,
    SafetyViolation,
)
from .config import (
    DeepRareConfig,
    FeatureFlags,
    LoggingConfig,
    PerformanceConfig,
    SafetyThresholds,
)
from .differential import DifferentialDiagnosisManager
from .evaluator import DiagnosisArenaEvaluator
from .knowledge_base import RareDiseaseKnowledgeBase
from .observability import (
    AuditExporter,
    HealthSnapshot,
    MetricsCollector,
    ObservabilityContext,
    StructuredFormatter,
    TraceContext,
    configure_logging,
)
from .orchestrator import ControllerOrchestrator
from .pipeline import PipelineConfig, RareDiseasePipeline
from .schema import (
    DiagnosisResult,
    DifferentialDiagnosis,
    DiseaseProfile,
    EvaluationMetrics,
    Evidence,
    Hypothesis,
    PatientCase,
    RankedDiagnosis,
    RareDiseaseState,
    SymptomAnalysisResult,
    SymptomProfile,
    TestInterpretationResult,
    TestResult,
)

__all__ = [
    # Agents
    "SymptomAnalyzer",
    "TestInterpreter",
    "LiteratureMatcher",
    # Core
    "ControllerOrchestrator",
    "DifferentialDiagnosisManager",
    "RareDiseaseKnowledgeBase",
    "RareDiseasePipeline",
    "PipelineConfig",
    "DiagnosisArenaEvaluator",
    # Schema
    "PatientCase",
    "RareDiseaseState",
    "Hypothesis",
    "Evidence",
    "SymptomProfile",
    "TestResult",
    "DiagnosisResult",
    "DifferentialDiagnosis",
    "RankedDiagnosis",
    "DiseaseProfile",
    "SymptomAnalysisResult",
    "TestInterpretationResult",
    "EvaluationMetrics",
    # Clinical Safety
    "SafetyLevel",
    "AuditAction",
    "SafetyViolation",
    "AuditEntry",
    "RedFlag",
    "RedFlagDetector",
    "ClinicalSafetyGate",
    "AuditTrail",
    "ClinicalSafetyContext",
    # Configuration
    "DeepRareConfig",
    "SafetyThresholds",
    "FeatureFlags",
    "PerformanceConfig",
    "LoggingConfig",
    # Observability
    "ObservabilityContext",
    "MetricsCollector",
    "TraceContext",
    "AuditExporter",
    "HealthSnapshot",
    "StructuredFormatter",
    "configure_logging",
]

__version__ = "1.0.0"
