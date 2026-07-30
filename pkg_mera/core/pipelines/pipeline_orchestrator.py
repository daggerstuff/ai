from __future__ import annotations

"""Core pipeline orchestrator for production data processing workflows."""

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class PipelineStageResult:
    """Result for a single stage in an orchestration run."""

    name: str
    status: str
    input_size: int
    output_size: int
    duration_ms: float
    metadata: dict[str, Any]


@dataclass
class PipelineResult:
    """Aggregate result from :class:`PipelineOrchestrator`."""

    success: bool
    input_hash: str
    output_hash: str
    stages: list[PipelineStageResult]
    payload: Any
    started_at: str
    finished_at: str
    errors: list[str]


class PipelineOrchestrator:
    """Production-ready pipeline orchestrator with pluggable processing stages."""

    DEFAULT_STAGES: dict[str, Callable[[Any], Any]] = {}

    def __init__(
        self,
        stages: dict[str, Callable[[Any], Any]] | None = None,
        logger: logging.Logger | None = None,
        metrics_collector: Any = None,
    ) -> None:
        if stages is not None and not isinstance(stages, dict):
            raise TypeError("stages must be a mapping of stage_name -> callable")
        self.stages = dict(stages) if stages else dict(self.DEFAULT_STAGES)
        if not self.stages:
            self.stages = {
                "normalize": self._default_normalize,
                "enrich": self._default_enrich,
                "validate": self._default_validate,
            }
        self.logger = logger or logging.getLogger(__name__)
        self.metrics_collector = metrics_collector

    def process(self, data: Any) -> PipelineResult:
        """Run configured stages and return a pipeline result."""

        if data is None:
            raise ValueError("data must not be None")

        started = datetime.now(tz=UTC)
        started_iso = started.isoformat()
        errors: list[str] = []
        stage_results: list[PipelineStageResult] = []

        current_payload = data
        input_size = self._safe_len(current_payload)

        for stage_name, stage_callable in self.stages.items():
            stage_start = datetime.now(tz=UTC)
            message: str | None = None
            status = "completed"

            try:
                self.logger.info("Running pipeline stage %s", stage_name)
                processed = stage_callable(current_payload)
                current_payload = processed
                output_size = self._safe_len(current_payload)
            except Exception as exc:
                message = f"Stage '{stage_name}' failed: {exc!s}"
                self.logger.exception(message)
                errors.append(message)
                status = "failed"
                output_size = input_size
                # Carry forward with diagnostic envelope so later stages can still run.
                current_payload = {"error": message, "data": current_payload}

            stage_end = datetime.now(tz=UTC)
            duration_ms = (stage_end - stage_start).total_seconds() * 1000
            stage_results.append(
                PipelineStageResult(
                    name=stage_name,
                    status=status,
                    input_size=input_size,
                    output_size=output_size,
                    duration_ms=duration_ms,
                    metadata={"message": message} if message else {},
                )
            )
            if self.metrics_collector is not None:
                self.metrics_collector.record_stage_execution(
                    stage_name=stage_name,
                    duration_ms=duration_ms,
                    input_size=input_size,
                    output_size=output_size,
                    status=status,
                    error=message,
                )
            input_size = output_size

        finished = datetime.now(tz=UTC)
        return PipelineResult(
            success=not errors,
            input_hash=self._hash_payload(data),
            output_hash=self._hash_payload(current_payload),
            stages=stage_results,
            payload=current_payload,
            started_at=started_iso,
            finished_at=finished.isoformat(),
            errors=errors,
        )

    def register_stage(self, name: str, stage_callable: Callable[[Any], Any]) -> None:
        """Register or replace a processing stage."""

        if not callable(stage_callable):
            raise TypeError("stage_callable must be callable")
        self.stages[name] = stage_callable

    def _default_normalize(self, payload: Any) -> Any:
        """Normalize common container formats."""

        if isinstance(payload, dict):
            return dict(sorted(payload.items(), key=lambda item: item[0]))
        if isinstance(payload, (list, tuple)):
            return list(payload)
        if isinstance(payload, (str, bytes, int, float, bool)):
            return payload
        if isinstance(payload, Path):
            return str(payload)
        return payload

    def _default_enrich(self, payload: Any) -> Any:
        """Attach lightweight enrichment metadata for observability."""

        if isinstance(payload, dict):
            enriched = dict(payload)
            enriched.setdefault("pipeline", {})
            enriched["pipeline"].update(
                {
                    "enriched_at": datetime.now(tz=UTC).isoformat(),
                    "element_count": self._safe_len(payload),
                }
            )
            return enriched
        return payload

    def _default_validate(self, payload: Any) -> Any:
        """Validate payload shape and required keys."""

        if payload is None:
            raise ValueError("payload is empty after upstream stages")
        if isinstance(payload, dict) and not payload:
            raise ValueError("payload must contain data")
        if isinstance(payload, list) and not payload:
            raise ValueError("payload list must not be empty")
        return payload

    def _safe_len(self, payload: Any) -> int:
        try:
            return len(payload)
        except TypeError:
            return 0

    def _hash_payload(self, payload: Any) -> str:
        """Compute a stable sha256 for a payload representation."""

        try:
            serializable = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        except TypeError:
            serializable = str(payload).encode("utf-8")
        return hashlib.sha256(serializable).hexdigest()


__all__ = ["PipelineOrchestrator", "PipelineResult", "PipelineStageResult"]
