"""RareDiseasePipeline — main entry point for the DeepRare multi-agent diagnosis system.

Enterprise-grade pipeline wrapping the ControllerOrchestrator with:
- Environment-based configuration via DeepRareConfig
- Error handling and graceful degradation
- Health check endpoint for service monitoring
- Batch evaluation with Wilson confidence intervals
- File I/O utilities for patient case loading
- PHI de-identification tracking
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .config import DeepRareConfig
from .evaluator import DiagnosisArenaEvaluator
from .knowledge_base import RareDiseaseKnowledgeBase
from .orchestrator import ControllerOrchestrator
from .schema import (
    DiagnosisResult,
    EvaluationMetrics,
    PatientCase,
)

logger = logging.getLogger("deep_rare.pipeline")

__all__ = ["RareDiseasePipeline", "PipelineConfig"]

__version__ = "1.0.0"


class PipelineConfig(BaseModel):
    """Configuration for RareDiseasePipeline.

    This is a Pydantic model for easy serialization. For environment-based
    configuration, use DeepRareConfig.from_env() and pass to the pipeline.
    """

    max_iterations: int = Field(default=10, ge=1, le=50)
    convergence_window: int = Field(default=3, ge=1, le=10)
    pruning_threshold: float = Field(default=0.01, ge=0.0, le=0.5)
    enable_evaluation: bool = True
    timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    enable_safety_gates: bool = True
    enable_audit_trail: bool = True
    enable_red_flag_detection: bool = True

    model_config = {"extra": "forbid"}


class RareDiseasePipeline:
    """High-level pipeline for multi-agent rare disease diagnosis.

    The orchestrator internally creates sub-agents and the differential manager.
    This pipeline wraps it with batch evaluation, file I/O utilities, error
    handling, and health checks.

    Args:
        config: Pipeline configuration. If None, defaults are used.
        deep_rare_config: Optional DeepRareConfig for environment-based settings.
            When provided, takes precedence over ``config`` for overlapping fields.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        deep_rare_config: DeepRareConfig | None = None,
    ) -> None:
        # Merge configs: deep_rare_config takes precedence
        if deep_rare_config is not None:
            self.config = PipelineConfig(
                max_iterations=deep_rare_config.performance.max_iterations,
                convergence_window=deep_rare_config.performance.convergence_window,
                pruning_threshold=deep_rare_config.performance.pruning_threshold,
                enable_evaluation=deep_rare_config.features.enable_evaluation,
                timeout_seconds=deep_rare_config.performance.timeout_seconds,
                enable_safety_gates=deep_rare_config.features.enable_safety_gates,
                enable_audit_trail=deep_rare_config.features.enable_audit_trail,
                enable_red_flag_detection=deep_rare_config.features.enable_red_flag_detection,
            )
            self._deep_rare_config = deep_rare_config
        else:
            self.config = config or PipelineConfig()
            self._deep_rare_config = None

        self._kb = RareDiseaseKnowledgeBase()
        self._orchestrator = ControllerOrchestrator(
            kb=self._kb,
            max_iterations=self.config.max_iterations,
            convergence_window=self.config.convergence_window,
            pruning_threshold=self.config.pruning_threshold,
            timeout_seconds=self.config.timeout_seconds,
            enable_safety_gates=self.config.enable_safety_gates,
            enable_audit_trail=self.config.enable_audit_trail,
            enable_red_flag_detection=self.config.enable_red_flag_detection,
        )
        self._evaluator = DiagnosisArenaEvaluator()

        logger.info(
            "pipeline_initialized",
            extra={
                "kb_size": self._kb.disease_count,
                "max_iterations": self.config.max_iterations,
                "safety_gates": self.config.enable_safety_gates,
            },
        )

    def diagnose(self, case: PatientCase) -> DiagnosisResult:
        """Run the full multi-agent diagnosis pipeline on a single patient case.

        Includes error handling and graceful degradation. If the orchestrator
        fails, returns a minimal DiagnosisResult with error information.

        Args:
            case: PatientCase with symptoms, history, and available tests.

        Returns:
            DiagnosisResult with differential, state, safety flags, and audit trail.
        """
        # PHI check
        if not case.consent_given:
            logger.warning("diagnosis_blocked_no_consent", extra={"case_id": case.case_id})
            return self._error_result(case, "Diagnosis blocked: patient consent not given")

        try:
            start = time.time()
            result = self._orchestrator.diagnose(case)
            elapsed = time.time() - start

            if elapsed > self.config.timeout_seconds:
                logger.warning(
                    "diagnosis_exceeded_timeout",
                    extra={"case_id": case.case_id, "elapsed": elapsed, "timeout": self.config.timeout_seconds},
                )

            return result
        except Exception as exc:
            logger.error(
                "diagnosis_failed",
                extra={"case_id": case.case_id, "error": str(exc)},
                exc_info=True,
            )
            return self._error_result(case, f"Diagnosis failed: {exc}")

    def diagnose_batch(self, cases: list[PatientCase]) -> list[DiagnosisResult]:
        """Run the pipeline on multiple patient cases.

        Each case is processed independently. Failed cases return error
        results rather than raising exceptions.

        Args:
            cases: List of patient cases to diagnose.

        Returns:
            List of DiagnosisResult, one per case.
        """
        results: list[DiagnosisResult] = []
        for case in cases:
            try:
                result = self.diagnose(case)
                results.append(result)
            except Exception as exc:
                logger.error("batch_diagnosis_failed", extra={"case_id": case.case_id, "error": str(exc)})
                results.append(self._error_result(case, f"Batch diagnosis failed: {exc}"))
        return results

    def evaluate(self, cases: list[PatientCase]) -> EvaluationMetrics:
        """Evaluate the pipeline on a set of cases with ground truth diagnoses.

        Runs diagnosis on all cases, then computes evaluation metrics including
        Recall@K, MRR, Wilson confidence intervals, and safety violation tracking.

        Args:
            cases: List of patient cases with ground_truth_diagnosis set.

        Returns:
            Aggregated EvaluationMetrics.
        """
        if not cases:
            return self._empty_metrics()

        results = self.diagnose_batch(cases)
        metrics = self._evaluator.evaluate(results, cases)

        logger.info(
            "evaluation_completed",
            extra={
                "total_cases": metrics.total_cases,
                "recall_at_1": metrics.recall_at_1,
                "mrr": metrics.mrr,
            },
        )
        return metrics

    def load_cases_from_file(self, path: str | Path) -> list[PatientCase]:
        """Load patient cases from a JSON or JSONL file.

        Args:
            path: Path to JSON (array) or JSONL (one object per line) file.

        Returns:
            List of validated PatientCase objects.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValidationError: If case data doesn't match PatientCase schema.
            json.JSONDecodeError: If JSON is malformed.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Case file not found: {p}")

        raw = p.read_text(encoding="utf-8")

        if p.suffix == ".jsonl":
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            data = [json.loads(ln) for ln in lines]
        else:
            data = json.loads(raw)

        if not isinstance(data, list):
            raise ValueError(f"Expected array of cases, got {type(data).__name__}")

        cases: list[PatientCase] = []
        for i, item in enumerate(data):
            try:
                cases.append(PatientCase.model_validate(item))
            except ValidationError as exc:
                logger.error("case_validation_failed", extra={"index": i, "errors": exc.errors()})
                raise

        logger.info("cases_loaded", extra={"file": str(p), "count": len(cases)})
        return cases

    def health_check(self) -> dict[str, Any]:
        """Return health status of the pipeline and its components.

        Useful for service monitoring and readiness probes.

        Returns:
            Dict with status, component health, and configuration info.
        """
        kb_stats = self._kb.get_statistics()
        return {
            "status": "healthy",
            "version": __version__,
            "config": self.config.model_dump(),
            "knowledge_base": kb_stats,
            "agents": ["symptom_analyzer", "test_interpreter", "literature_matcher"],
            "safety_enabled": self.config.enable_safety_gates,
            "audit_trail_enabled": self.config.enable_audit_trail,
            "red_flag_detection_enabled": self.config.enable_red_flag_detection,
        }

    def get_info(self) -> dict[str, Any]:
        """Return pipeline configuration and component info (backward-compatible alias)."""
        return self.health_check()

    @staticmethod
    def _error_result(case: PatientCase, message: str) -> DiagnosisResult:
        """Create a DiagnosisResult for a failed diagnosis."""
        from .schema import DifferentialDiagnosis, RareDiseaseState

        return DiagnosisResult(
            case_id=case.case_id,
            differential=DifferentialDiagnosis(
                ranked_list=[],
                eliminated=[],
                total_hypotheses_considered=0,
                iterations_used=0,
                convergence_achieved=False,
                reasoning_trace=message,
            ),
            state=RareDiseaseState(max_iterations=0, convergence_window=0),
            iterations=0,
            time_seconds=0.0,
            converged=False,
            agent_outputs={"error": message},
            recommended_next_steps=[message, "Manual review required."],
            evaluation=None,
            safety_flags=[],
            audit_trail=[],
            safety_violations=[{"type": "diagnosis_error", "message": message}],
            clinical_confidence=0.0,
            requires_human_review=True,
            phi_deidentified=case.phi_protected,
        )

    @staticmethod
    def _empty_metrics() -> EvaluationMetrics:
        """Return empty evaluation metrics for edge cases."""
        return EvaluationMetrics(
            recall_at_1=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            mrr=0.0,
            accuracy_by_organ={},
            accuracy_by_rarity={},
            accuracy_by_complexity={},
            avg_iterations=0.0,
            avg_time_seconds=0.0,
            total_cases=0,
            correct_cases=0,
        )
