"""Inference adapter for the DeepRare rare disease diagnosis pipeline.

Enterprise wrapper around `platform.deep_rare.RareDiseasePipeline` that exposes
a stable inference API for the core/pipelines layer with:

- Input validation with structured error responses
- Health check endpoint for liveness/readiness probes
- Error handling that returns structured error dicts (not exceptions)
- Observability integration (metrics, tracing, audit export)
- PHI de-identification flags propagated from input
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from ai.tools.utilities.platform.deep_rare import DeepRareConfig, PatientCase, PipelineConfig, RareDiseasePipeline
from ai.tools.utilities.platform.deep_rare.observability import ObservabilityContext

__all__ = ["RareDiseaseInferenceService"]

__version__ = "1.0.0"

logger = logging.getLogger("deep_rare.inference")


class RareDiseaseInferenceService:
    """Inference service wrapping the DeepRare multi-agent pipeline.

    Provides a stable interface for the core/pipelines/inference layer
    that other services can call without depending on the full platform
    package internals. Includes error handling, health checks, and
    observability integration.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        deep_rare_config: DeepRareConfig | None = None,
        enable_observability: bool = True,
    ) -> None:
        self._pipeline = RareDiseasePipeline(config, deep_rare_config)
        self._enable_observability = enable_observability
        self._observability = ObservabilityContext(deep_rare_config) if enable_observability else None
        self._started_at = datetime.now(UTC).isoformat()
        logger.info("RareDiseaseInferenceService initialized", extra={"version": __version__})

    # ------------------------------------------------------------------
    # Diagnosis
    # ------------------------------------------------------------------

    def diagnose(self, case_data: dict[str, Any]) -> dict[str, Any]:
        """Diagnose a patient case from raw dict input.

        Args:
            case_data: Dictionary matching PatientCase schema.

        Returns:
            Either {"result": DiagnosisResult.to_dict()} on success,
            or {"error": ..., "details": ...} on failure.
        """
        # Input validation
        validation_error = self._validate_input(case_data)
        if validation_error:
            return validation_error

        try:
            case = PatientCase.model_validate(case_data)
        except ValidationError as exc:
            logger.warning("Input validation failed", extra={"errors": exc.errors()})
            return {
                "error": "validation_error",
                "details": exc.errors(),
                "timestamp": datetime.now(UTC).isoformat(),
            }

        # Tracing
        if self._observability:
            with self._observability.trace(case.case_id) as trace:
                with trace.phase("diagnosis"):
                    result = self._pipeline.diagnose(case)
                self._observability.record_diagnosis(result, case)
        else:
            result = self._pipeline.diagnose(case)

        logger.info(
            "Diagnosis completed",
            extra={
                "case_id": case.case_id,
                "converged": result.converged,
                "iterations": result.iterations,
                "time_seconds": result.time_seconds,
            },
        )

        return {"result": result.to_dict()}

    def diagnose_batch(self, cases_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Diagnose multiple patient cases from raw dict input.

        Args:
            cases_data: List of dictionaries matching PatientCase schema.

        Returns:
            {"results": [...], "errors": [...]} with per-case outcomes.
        """
        if not isinstance(cases_data, list):
            return {
                "error": "invalid_input",
                "details": "cases_data must be a list",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for i, case_data in enumerate(cases_data):
            outcome = self.diagnose(case_data)
            if "error" in outcome:
                errors.append({"index": i, **outcome})
            else:
                results.append(outcome)

        return {"results": results, "errors": errors, "total": len(cases_data)}

    def evaluate(self, cases_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate the pipeline on a batch of cases with ground truth.

        Args:
            cases_data: List of dicts matching PatientCase schema (with ground_truth_diagnosis).

        Returns:
            Either {"metrics": EvaluationMetrics.model_dump()} on success,
            or {"error": ..., "details": ...} on failure.
        """
        if not isinstance(cases_data, list):
            return {
                "error": "invalid_input",
                "details": "cases_data must be a list",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        try:
            cases = [PatientCase.model_validate(d) for d in cases_data]
        except ValidationError as exc:
            return {
                "error": "validation_error",
                "details": exc.errors(),
                "timestamp": datetime.now(UTC).isoformat(),
            }

        try:
            metrics = self._pipeline.evaluate(cases)
            if self._observability:
                self._observability.record_evaluation(metrics)
            return {"metrics": metrics.model_dump()}
        except Exception as exc:
            logger.exception("Evaluation failed")
            return {
                "error": "evaluation_failed",
                "details": str(exc),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Return health status for liveness/readiness probes.

        Returns:
            Health snapshot dict with status, version, KB stats, and metrics.
        """
        pipeline_health = self._pipeline.health_check()
        if self._observability:
            snapshot = self._observability.get_health_snapshot(kb_stats=pipeline_health.get("kb_stats", {}))
            return snapshot.to_dict()
        return {
            "status": pipeline_health.get("status", "unknown"),
            "version": __version__,
            "started_at": self._started_at,
            "pipeline": pipeline_health,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def get_info(self) -> dict[str, Any]:
        """Return service configuration info (alias for health_check)."""
        return self.health_check()

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        """Get collected metrics (if observability enabled)."""
        if not self._observability:
            return {"error": "observability_disabled"}
        return self._observability.metrics.get_all_metrics()

    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus text exposition format."""
        if not self._observability:
            return "# observability disabled\n"
        return self._observability.metrics.export_prometheus()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(case_data: dict[str, Any]) -> dict[str, Any] | None:
        """Pre-validate input before Pydantic parsing.

        Returns error dict if invalid, None if OK.
        """
        if not isinstance(case_data, dict):
            return {
                "error": "invalid_input",
                "details": "case_data must be a dict",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        if not case_data.get("case_id"):
            return {
                "error": "missing_case_id",
                "details": "case_id is required",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        if not case_data.get("presenting_symptoms"):
            return {
                "error": "missing_symptoms",
                "details": "presenting_symptoms is required and must be non-empty",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        # PHI consent check
        if case_data.get("consent_given") is False:
            return {
                "error": "consent_not_given",
                "details": "Patient consent not given — diagnosis blocked per HIPAA",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        return None
