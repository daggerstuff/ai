"""Enterprise observability for DeepRare multi-agent rare disease diagnosis.

Provides:
- Structured JSON logging configuration w/ correlation IDs
- Metrics collection (counters, histograms, gauges)
- Audit trail serialization for external SIEM/HIPAA compliance systems
- Performance tracing for agent phases
- Health snapshot aggregation

Designed for clinical/production environments where every diagnostic
decision must be traceable, auditable, and exportable to compliance systems.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .clinical_safety import AuditTrail, SafetyViolation
    from .schema import DiagnosisResult, EvaluationMetrics, PatientCase

__all__ = [
    "StructuredFormatter",
    "configure_logging",
    "MetricsCollector",
    "MetricEntry",
    "TraceContext",
    "AuditExporter",
    "HealthSnapshot",
    "ObservabilityContext",
]

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Structured JSON Logging
# ---------------------------------------------------------------------------


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Emits one JSON object per log record, suitable for ingestion by
    ELK, Datadog, CloudWatch, or any log aggregator that parses JSON lines.
    """

    # Fields that should always appear in output
    _RESERVED = frozenset(
        {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "taskName",
            "getMessage",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        """Format record as JSON string."""
        base: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Merge extra fields from record
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                base[key] = value

        # Exception info
        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)

        return json.dumps(base, default=str, ensure_ascii=False)


def configure_logging(
    level: str | int = "INFO",
    *,
    json_output: bool = True,
    stream: Any = None,
    logger_name: str = "deep_rare",
) -> logging.Logger:
    """Configure structured logging for DeepRare.

    Args:
        level: Log level (string or int).
        json_output: If True, emit JSON; if False, use human-readable format.
        stream: Output stream (default: sys.stderr).
        logger_name: Logger name to configure.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    handler = logging.StreamHandler(stream or sys.stderr)
    if json_output:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s:%(lineno)s - %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Metrics Collection
# ---------------------------------------------------------------------------


@dataclass
class MetricEntry:
    """Single metric measurement."""

    name: str
    value: float
    unit: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "tags": self.tags,
            "timestamp": self.timestamp,
        }


class MetricsCollector:
    """Thread-safe metrics collector for diagnosis pipeline.

    Supports counters, gauges, and histograms (as summaries).
    All operations are thread-safe.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._entries: list[MetricEntry] = []
        self._max_entries = 10000

    def increment(self, name: str, value: float = 1.0, **tags: str) -> None:
        """Increment a counter."""
        with self._lock:
            self._counters[name] += value
            self._entries.append(MetricEntry(name=name, value=value, unit="count", tags=tags))
            self._trim()

    def gauge(self, name: str, value: float, **tags: str) -> None:
        """Set a gauge value."""
        with self._lock:
            self._gauges[name] = value
            self._entries.append(MetricEntry(name=name, value=value, unit="gauge", tags=tags))
            self._trim()

    def histogram(self, name: str, value: float, **tags: str) -> None:
        """Record a histogram observation."""
        with self._lock:
            self._histograms[name].append(value)
            self._entries.append(MetricEntry(name=name, value=value, unit="histogram", tags=tags))
            self._trim()

    def timing(self, name: str, seconds: float, **tags: str) -> None:
        """Record a timing observation (alias for histogram with seconds unit)."""
        self.histogram(name, seconds, **tags)

    def get_counter(self, name: str) -> float:
        """Get current counter value."""
        with self._lock:
            return self._counters.get(name, 0.0)

    def get_gauge(self, name: str) -> float | None:
        """Get current gauge value."""
        with self._lock:
            return self._gauges.get(name)

    def get_histogram_stats(self, name: str) -> dict[str, float]:
        """Get histogram statistics (count, mean, median, p95, min, max)."""
        with self._lock:
            values = sorted(self._histograms.get(name, []))
        if not values:
            return {"count": 0, "mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0}
        n = len(values)
        p95_idx = int(n * 0.95)
        return {
            "count": float(n),
            "mean": sum(values) / n,
            "median": values[n // 2],
            "p95": values[min(p95_idx, n - 1)],
            "min": values[0],
            "max": values[-1],
        }

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all metrics as a summary dict."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: self.get_histogram_stats(k) for k in self._histograms},
                "recent_entries": [e.to_dict() for e in self._entries[-100:]],
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._entries.clear()

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            for name, value in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")
            for name, value in sorted(self._gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")
            for name in sorted(self._histograms):
                stats = self.get_histogram_stats(name)
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {stats['count']}")
                lines.append(f"{name}_mean {stats['mean']}")
                lines.append(f"{name}_p95 {stats['p95']}")
        return "\n".join(lines) + "\n"

    def _trim(self) -> None:
        """Trim entries to max size."""
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]


# ---------------------------------------------------------------------------
# Trace Context (per-diagnosis tracing)
# ---------------------------------------------------------------------------


@dataclass
class TraceContext:
    """Tracing context for a single diagnosis run.

    Captures per-phase timings, agent outputs, and a correlation ID
    for linking all log lines and metrics to a specific diagnosis.
    """

    case_id: str
    correlation_id: str = ""
    start_time: float = field(default_factory=time.monotonic)
    phases: dict[str, float] = field(default_factory=dict)
    agent_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))  # type: ignore[assignment]
    errors: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.correlation_id:
            self.correlation_id = f"diag-{self.case_id}-{int(self.start_time * 1000)}"

    @contextmanager
    def phase(self, name: str) -> Any:
        """Context manager for timing a phase."""
        start = time.monotonic()
        yield
        elapsed = time.monotonic() - start
        self.phases[name] = elapsed

    def record_agent_call(self, agent_name: str) -> None:
        """Record that an agent was invoked."""
        self.agent_calls[agent_name] += 1  # type: ignore[index]

    def record_error(self, phase: str, error: str) -> None:
        """Record an error during a phase."""
        self.errors.append({"phase": phase, "error": error, "timestamp": datetime.now(UTC).isoformat()})

    def total_elapsed(self) -> float:
        """Get total elapsed time since trace started."""
        return time.monotonic() - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Serialize trace to dict."""
        return {
            "case_id": self.case_id,
            "correlation_id": self.correlation_id,
            "total_seconds": self.total_elapsed(),
            "phases": dict(self.phases),
            "agent_calls": dict(self.agent_calls),
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Audit Export (HIPAA-compliant serialization)
# ---------------------------------------------------------------------------


class AuditExporter:
    """Serialize audit trail data for external compliance systems.

    Supports export to:
    - JSON (for SIEM ingestion)
    - CSV (for spreadsheet/audit review)
    - FHIR AuditEvent bundle (for healthcare interoperability)
    """

    @staticmethod
    def to_json(audit_trail: AuditTrail, case_id: str | None = None) -> str:
        """Export audit trail as JSON string.

        Args:
            audit_trail: AuditTrail instance from clinical_safety.
            case_id: Optional case ID filter.

        Returns:
            JSON string of audit entries.
        """
        if case_id:
            entries = audit_trail.get_case_trail(case_id)
        else:
            entries = audit_trail.get_entries()
        return json.dumps(
            {
                "export_timestamp": datetime.now(UTC).isoformat(),
                "entry_count": len(entries),
                "entries": [e.model_dump() if hasattr(e, "model_dump") else e for e in entries],
            },
            default=str,
            indent=2,
        )

    @staticmethod
    def to_csv(audit_trail: AuditTrail, case_id: str | None = None) -> str:
        """Export audit trail as CSV string.

        Args:
            audit_trail: AuditTrail instance.
            case_id: Optional case ID filter.

        Returns:
            CSV string with header row.
        """
        entries = audit_trail.get_case_trail(case_id) if case_id else audit_trail.get_entries()
        header = "entry_id,timestamp,action,agent_name,case_id,details"
        lines = [header]
        for e in entries:
            details = json.dumps(e.details, default=str) if hasattr(e, "details") else str(e)
            lines.append(f"{e.entry_id},{e.timestamp},{e.action},{e.agent_name},{e.case_id},{details}")
        return "\n".join(lines)

    @staticmethod
    def to_fhir_audit_events(audit_trail: AuditTrail, case_id: str | None = None) -> list[dict[str, Any]]:
        """Export audit trail as FHIR AuditEvent resources.

        Args:
            audit_trail: AuditTrail instance.
            case_id: Optional case ID filter.

        Returns:
            List of FHIR AuditEvent resource dicts (R4).
        """
        entries = audit_trail.get_case_trail(case_id) if case_id else audit_trail.get_entries()
        events: list[dict[str, Any]] = []
        for e in entries:
            event = {
                "resourceType": "AuditEvent",
                "type": {
                    "system": "http://terminology.hl7.org/CodeSystem/audit-event-type",
                    "code": "query",
                    "display": e.action,
                },
                "recorded": e.timestamp,
                "agent": [
                    {
                        "type": {
                            "coding": [{"system": "deep_rare", "code": e.agent_name}],
                        },
                        "who": {"reference": f"Device/{e.agent_name}"},
                        "requestor": False,
                    }
                ],
                "source": {
                    "observer": {"reference": "Organization/deep-rare"},
                    "type": [
                        {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/security-source-type",
                                    "code": "1",
                                    "display": "User Device",
                                }
                            ]
                        }
                    ],
                },
            }
            if hasattr(e, "details") and e.details:
                event["entity"] = [
                    {
                        "what": {"reference": f"DiagnosticReport/{e.case_id}"},
                        "detail": [
                            {
                                "type": "metadata",
                                "valueString": json.dumps(e.details, default=str),
                            }
                        ],
                    }
                ]
            events.append(event)
        return events


# ---------------------------------------------------------------------------
# Health Snapshot
# ---------------------------------------------------------------------------


@dataclass
class HealthSnapshot:
    """Point-in-time health snapshot for monitoring.

    Captures the state of all DeepRare subsystems at a moment
    for liveness/readiness checks.
    """

    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "healthy"
    version: str = __version__
    kb_disease_count: int = 0
    kb_cache_sizes: dict[str, int] = field(default_factory=dict)
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    active_traces: int = 0
    safety_violation_count: int = 0
    avg_diagnosis_time_seconds: float = 0.0
    last_diagnosis_case_id: str | None = None
    config_environment: str = "production"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "status": self.status,
            "version": self.version,
            "kb_disease_count": self.kb_disease_count,
            "kb_cache_sizes": self.kb_cache_sizes,
            "metrics_summary": self.metrics_summary,
            "active_traces": self.active_traces,
            "safety_violation_count": self.safety_violation_count,
            "avg_diagnosis_time_seconds": self.avg_diagnosis_time_seconds,
            "last_diagnosis_case_id": self.last_diagnosis_case_id,
            "config_environment": self.config_environment,
        }

    def to_json(self) -> str:
        """Export as JSON string (for HTTP health endpoint)."""
        return json.dumps(self.to_dict(), default=str, indent=2)


# ---------------------------------------------------------------------------
# Observability Context (top-level integration)
# ---------------------------------------------------------------------------


class ObservabilityContext:
    """Top-level observability facade integrating logging, metrics, tracing, and audit.

    Usage:
        obs = ObservabilityContext()
        with obs.trace("case-123") as t:
            with t.phase("symptom_analysis"):
                ...
            obs.record_diagnosis(result, case)

    This is the single entry point for all observability concerns.
    """

    def __init__(
        self,
        config: Any = None,
        *,
        logger_name: str = "deep_rare",
        log_level: str | int = "INFO",
        json_logging: bool = True,
    ) -> None:
        self._logger = configure_logging(log_level, json_output=json_logging, logger_name=logger_name)
        self._metrics = MetricsCollector()
        self._traces: dict[str, TraceContext] = {}
        self._lock = Lock()
        self._config = config
        self._violation_count = 0
        self._last_case_id: str | None = None
        self._diagnosis_times: list[float] = []

    @property
    def logger(self) -> logging.Logger:
        """Get the configured logger."""
        return self._logger

    @property
    def metrics(self) -> MetricsCollector:
        """Get the metrics collector."""
        return self._metrics

    @contextmanager
    def trace(self, case_id: str) -> Any:
        """Start a tracing context for a diagnosis run.

        Yields a TraceContext for recording per-phase timings.
        """
        ctx = TraceContext(case_id=case_id)
        with self._lock:
            self._traces[ctx.correlation_id] = ctx
        self._logger.info("Diagnosis trace started", extra={"case_id": case_id, "correlation_id": ctx.correlation_id})
        try:
            yield ctx
        except Exception:
            ctx.record_error("trace", "Exception during diagnosis")
            self._logger.exception(
                "Diagnosis trace failed",
                extra={"case_id": case_id, "correlation_id": ctx.correlation_id},
            )
            raise
        finally:
            elapsed = ctx.total_elapsed()
            self._metrics.timing("diagnosis.total_time_seconds", elapsed, case_id=case_id)
            self._diagnosis_times.append(elapsed)
            if len(self._diagnosis_times) > 1000:
                self._diagnosis_times = self._diagnosis_times[-1000:]
            with self._lock:
                self._traces.pop(ctx.correlation_id, None)
                self._last_case_id = case_id
            self._logger.info(
                "Diagnosis trace completed",
                extra={
                    "case_id": case_id,
                    "correlation_id": ctx.correlation_id,
                    "total_seconds": elapsed,
                    "phases": dict(ctx.phases),
                    "errors": ctx.errors,
                },
            )

    def record_diagnosis(self, result: DiagnosisResult, case: PatientCase) -> None:
        """Record metrics from a completed diagnosis.

        Args:
            result: DiagnosisResult from orchestrator.
            case: Original PatientCase.
        """
        self._metrics.increment("diagnosis.completed")
        self._metrics.gauge("diagnosis.iterations", result.iterations)
        self._metrics.gauge("diagnosis.time_seconds", result.time_seconds)
        self._metrics.gauge("diagnosis.converged", 1.0 if result.converged else 0.0)
        self._metrics.gauge("diagnosis.clinical_confidence", result.clinical_confidence)
        self._metrics.gauge("diagnosis.requires_human_review", 1.0 if result.requires_human_review else 0.0)

        if result.safety_violations:
            self._violation_count += len(result.safety_violations)
            self._metrics.increment("safety.violations", value=float(len(result.safety_violations)))

        if result.differential:
            self._metrics.gauge("diagnosis.hypotheses_count", result.differential.total_hypotheses_considered)
            top = result.differential.top_disease()
            if top:
                self._logger.info(
                    "Top diagnosis",
                    extra={
                        "case_id": case.case_id,
                        "top_disease": top.disease_name,
                        "probability": top.probability,
                        "confidence": top.confidence,
                        "converged": result.converged,
                    },
                )

    def record_evaluation(self, metrics: EvaluationMetrics) -> None:
        """Record metrics from batch evaluation.

        Args:
            metrics: EvaluationMetrics from evaluator.
        """
        self._metrics.gauge("eval.recall_at_1", metrics.recall_at_1)
        self._metrics.gauge("eval.recall_at_5", metrics.recall_at_5)
        self._metrics.gauge("eval.recall_at_10", metrics.recall_at_10)
        self._metrics.gauge("eval.mrr", metrics.mrr)
        self._metrics.gauge("eval.avg_iterations", metrics.avg_iterations)
        self._metrics.gauge("eval.avg_time_seconds", metrics.avg_time_seconds)
        self._metrics.gauge("eval.total_cases", metrics.total_cases)
        self._metrics.gauge("eval.correct_cases", metrics.correct_cases)
        self._metrics.gauge("eval.safety_violation_count", metrics.safety_violation_count)
        self._metrics.gauge("eval.avg_clinical_confidence", metrics.avg_clinical_confidence)

    def record_safety_violation(self, violation: SafetyViolation) -> None:
        """Record a clinical safety violation.

        Args:
            violation: SafetyViolation from clinical_safety.
        """
        self._violation_count += 1
        self._metrics.increment(
            f"safety.violation.{violation.level.value.lower()}",
            tags={"rule": violation.rule_name},
        )
        self._logger.warning(
            "Safety violation",
            extra={
                "violation_id": violation.violation_id,
                "level": violation.level.value,
                "rule": violation.rule_name,
                "description": violation.description,
            },
        )

    def get_health_snapshot(self, kb_stats: dict[str, Any] | None = None) -> HealthSnapshot:
        """Build a health snapshot for monitoring endpoints.

        Args:
            kb_stats: Knowledge base statistics (from RareDiseaseKnowledgeBase.get_statistics()).

        Returns:
            HealthSnapshot instance.
        """
        avg_time = sum(self._diagnosis_times) / len(self._diagnosis_times) if self._diagnosis_times else 0.0
        status = "healthy"
        if self._violation_count > 10:
            status = "degraded"
        if self._violation_count > 50:
            status = "critical"

        env = "production"
        if self._config and hasattr(self._config, "environment"):
            env = self._config.environment

        return HealthSnapshot(
            status=status,
            kb_disease_count=kb_stats.get("disease_count", 0) if kb_stats else 0,
            kb_cache_sizes=kb_stats.get("cache_sizes", {}) if kb_stats else {},
            metrics_summary=self._metrics.get_all_metrics(),
            active_traces=len(self._traces),
            safety_violation_count=self._violation_count,
            avg_diagnosis_time_seconds=avg_time,
            last_diagnosis_case_id=self._last_case_id,
            config_environment=env,
        )

    def export_audit(
        self, audit_trail: AuditTrail, case_id: str | None = None, fmt: str = "json"
    ) -> str | list[dict[str, Any]]:
        """Export audit trail in the specified format.

        Args:
            audit_trail: AuditTrail instance.
            case_id: Optional case ID filter.
            fmt: Export format ("json", "csv", "fhir").

        Returns:
            Exported audit data (string for json/csv, list for fhir).
        """
        if fmt == "json":
            return AuditExporter.to_json(audit_trail, case_id)
        elif fmt == "csv":
            return AuditExporter.to_csv(audit_trail, case_id)
        elif fmt == "fhir":
            return AuditExporter.to_fhir_audit_events(audit_trail, case_id)
        raise ValueError(f"Unknown export format: {fmt}")

    def reset(self) -> None:
        """Reset all observability state (for testing)."""
        self._metrics.reset()
        with self._lock:
            self._traces.clear()
        self._violation_count = 0
        self._last_case_id = None
        self._diagnosis_times.clear()
