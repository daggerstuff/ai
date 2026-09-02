"""
Pipeline Observability for PIX-507.

This module provides comprehensive observability for the modern dataset pipeline,
making pipeline health visible for operational decisions without requiring
manual reconstruction.

Design principles
----------------
* Stage throughput is instrumented automatically when PipelineOrchestrator runs.
* Validation failures are surfaced with gate-level detail for debugging.
* Failure regressions are detected when the same failure appears across multiple runs.
* Health summary is consumable by operators, dashboards, and Prometheus scraping.

Downstream consumers
--------------------
* Operators: GET /api/v1/pipeline/observability/health
* Prometheus: /metrics endpoint with pipeline_* metrics
* Dashboards: PipelineHealthSummary JSON structure
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ai.tools.utilities.pipelines.training_readiness_gates import (
    ReadinessResult,
    ReadinessStatus,
)


class HealthStatus(StrEnum):
    """Overall pipeline health determination."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class FailureSeverity(StrEnum):
    """Severity classification for pipeline failures."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class StageMetric:
    """A single metric point for a pipeline stage."""

    stage_name: str
    timestamp: str
    duration_ms: float
    input_size: int
    output_size: int
    status: str  # "completed" | "failed"
    error: str | None = None


@dataclass
class ReadinessMetric:
    """A readiness validation result recorded for observability."""

    package_id: str
    stage_id: str
    status: str
    passed: bool
    failed_gates: list[str]
    record_count: int
    timestamp: str


@dataclass
class FailureRecord:
    """A failure event recorded for tracking and regression detection."""

    failure_id: str
    timestamp: str
    stage: str | None
    gate: str | None  # Gate name if from readiness validation
    package_id: str | None
    error_message: str
    severity: str
    count: int = 1


@dataclass
class RegressionAlert:
    """Alert when a failure is repeating across multiple runs."""

    failure_pattern: str  # Normalized pattern (stage/gate/error signature)
    occurrences: int
    first_seen: str
    last_seen: str
    affected_packages: list[str]
    severity: str


@dataclass
class ThroughputMetrics:
    """Aggregated throughput metrics for a time window."""

    total_records_in: int
    total_records_out: int
    records_processed: int  # throughput (records_out - records_in via stages)
    average_duration_ms: float
    p95_duration_ms: float
    stage_metrics: dict[str, dict[str, Any]]  # stage_name -> {count, avg_ms, failures}


@dataclass
class ReadinessMetrics:
    """Aggregated readiness metrics for a time window."""

    total_validations: int
    passed: int
    failed: int
    conditionally_ready: int
    by_stage: dict[str, dict[str, int]]  # stage_id -> {passed, failed, total}
    recent_failures: list[dict[str, Any]]  # Last 10 failures


@dataclass
class FailureMetrics:
    """Aggregated failure metrics for a time window."""

    total_failures: int
    by_stage: dict[str, int]
    by_gate: dict[str, int]
    regressions_detected: int
    recent_failures: list[FailureRecord]
    alert_regressions: list[RegressionAlert]


@dataclass
class PipelineHealthSummary:
    """Complete pipeline health state for operational visibility.

    This is the primary output consumed by operators, dashboards, and
    monitoring systems to determine if the pipeline is:
    - HEALTHY: operating normally, all gates passing
    - DEGRADED: some failures or slowdown, but core function intact
    - UNHEALTHY: blocking failures or regressions requiring attention
    """

    status: str
    throughput: ThroughputMetrics
    readiness: ReadinessMetrics
    failures: FailureMetrics
    last_updated: str
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON API responses."""
        return {
            "status": self.status,
            "version": self.version,
            "last_updated": self.last_updated,
            "throughput": {
                "total_records_in": self.throughput.total_records_in,
                "total_records_out": self.throughput.total_records_out,
                "records_processed": self.throughput.records_processed,
                "average_duration_ms": self.throughput.average_duration_ms,
                "p95_duration_ms": self.throughput.p95_duration_ms,
                "stage_metrics": self.throughput.stage_metrics,
            },
            "readiness": {
                "total_validations": self.readiness.total_validations,
                "passed": self.readiness.passed,
                "failed": self.readiness.failed,
                "conditionally_ready": self.readiness.conditionally_ready,
                "by_stage": self.readiness.by_stage,
                "recent_failures": self.readiness.recent_failures,
            },
            "failures": {
                "total_failures": self.failures.total_failures,
                "by_stage": self.failures.by_stage,
                "by_gate": self.failures.by_gate,
                "regressions_detected": self.failures.regressions_detected,
                "recent_failures": [
                    {
                        "failure_id": f.failure_id,
                        "timestamp": f.timestamp,
                        "stage": f.stage,
                        "gate": f.gate,
                        "error_message": f.error_message,
                        "severity": f.severity,
                        "count": f.count,
                    }
                    for f in self.failures.recent_failures
                ],
                "alert_regressions": [
                    {
                        "failure_pattern": r.failure_pattern,
                        "occurrences": r.occurrences,
                        "first_seen": r.first_seen,
                        "last_seen": r.last_seen,
                        "affected_packages": r.affected_packages,
                        "severity": r.severity,
                    }
                    for r in self.failures.alert_regressions
                ],
            },
        }


class PipelineMetricsCollector:
    """Collects and aggregates pipeline metrics for observability.

    Thread-safe collector that maintains rolling windows of stage execution
    metrics, readiness validation results, and failure events.

    Usage::

        metrics = PipelineMetricsCollector()
        metrics.record_stage_execution("normalize", 45.2, 100, 95, "completed")
        metrics.record_readiness_result(readiness_result)
        health = metrics.get_health_summary()
    """

    def __init__(self, max_history: int = 1000) -> None:
        """Initialize the metrics collector.

        Args:
            max_history: Maximum number of entries to keep in rolling windows.
                        Older entries are evicted when limit is reached.
        """
        self._max_history = max_history
        self._lock = threading.Lock()

        # Rolling windows for metrics
        self._stage_metrics: list[StageMetric] = []
        self._readiness_metrics: list[ReadinessMetric] = []
        self._failure_records: list[FailureRecord] = []

        # Counters for quick aggregation
        self._stage_counts: dict[str, int] = defaultdict(int)
        self._stage_durations: dict[str, list[float]] = defaultdict(list)
        self._stage_failures: dict[str, int] = defaultdict(int)

    def record_stage_execution(
        self,
        stage_name: str,
        duration_ms: float,
        input_size: int,
        output_size: int,
        status: str,
        error: str | None = None,
    ) -> None:
        """Record a stage execution for throughput tracking.

        Args:
            stage_name: Name of the pipeline stage
            duration_ms: Execution time in milliseconds
            input_size: Number of records/items entering the stage
            output_size: Number of records/items leaving the stage
            status: "completed" or "failed"
            error: Optional error message if status is "failed"
        """
        metric = StageMetric(
            stage_name=stage_name,
            timestamp=datetime.now(UTC).isoformat(),
            duration_ms=duration_ms,
            input_size=input_size,
            output_size=output_size,
            status=status,
            error=error,
        )

        with self._lock:
            self._stage_metrics.append(metric)
            self._stage_counts[stage_name] += 1
            self._stage_durations[stage_name].append(duration_ms)

            if status == "failed":
                self._stage_failures[stage_name] += 1
                self._record_failure(
                    stage=stage_name,
                    error_message=error or f"Stage {stage_name} failed",
                    severity=FailureSeverity.HIGH,
                )

            # Trim if needed
            if len(self._stage_metrics) > self._max_history:
                self._stage_metrics = self._stage_metrics[-self._max_history :]

    def record_readiness_result(self, result: ReadinessResult) -> None:
        """Record a readiness validation result.

        Args:
            result: ReadinessResult from TrainingReadinessGates.validate_package()
        """
        metric = ReadinessMetric(
            package_id=result.package_id,
            stage_id=result.stage_id,
            status=result.status.value,
            passed=result.passed,
            failed_gates=result.failed_gates,
            record_count=result.record_count,
            timestamp=result.validated_at,
        )

        with self._lock:
            self._readiness_metrics.append(metric)

            if not result.passed:
                for gate_name in result.failed_gates:
                    self._record_failure(
                        stage=result.stage_id,
                        gate=gate_name,
                        package_id=result.package_id,
                        error_message=f"Gate {gate_name} failed for {result.package_id}",
                        severity=FailureSeverity.MEDIUM,
                    )

            # Trim if needed
            if len(self._readiness_metrics) > self._max_history:
                self._readiness_metrics = self._readiness_metrics[-self._max_history :]

    def record_failure(
        self,
        stage: str | None = None,
        gate: str | None = None,
        package_id: str | None = None,
        error_message: str = "",
        severity: str = FailureSeverity.MEDIUM,
    ) -> None:
        """Record a failure event.

        Args:
            stage: Pipeline stage where failure occurred
            gate: Gate name if from readiness validation
            package_id: Package ID if applicable
            error_message: Description of the failure
            severity: FailureSeverity level
        """
        self._record_failure(
            stage=stage,
            gate=gate,
            package_id=package_id,
            error_message=error_message,
            severity=severity,
        )

    def _record_failure(
        self,
        stage: str | None = None,
        gate: str | None = None,
        package_id: str | None = None,
        error_message: str = "",
        severity: str = FailureSeverity.MEDIUM,
    ) -> None:
        """Internal method to record a failure with deduplication."""
        # Create failure pattern for deduplication
        f"{stage or 'unknown'}:{gate or ''}:{error_message[:50]}"

        failure_id = f"fail_{len(self._failure_records) + 1}"

        record = FailureRecord(
            failure_id=failure_id,
            timestamp=datetime.now(UTC).isoformat(),
            stage=stage,
            gate=gate,
            package_id=package_id,
            error_message=error_message,
            severity=severity,
            count=1,
        )

        self._failure_records.append(record)

        # Trim if needed
        if len(self._failure_records) > self._max_history:
            self._failure_records = self._failure_records[-self._max_history :]

    def get_throughput_metrics(self, window_seconds: int | None = None) -> ThroughputMetrics:
        """Get aggregated throughput metrics.

        Args:
            window_seconds: If provided, only consider metrics within this window.
                          None means all metrics.

        Returns:
            ThroughputMetrics with aggregated stage throughput data.
        """
        with self._lock:
            stage_metrics = list(self._stage_metrics)

        if window_seconds:
            cutoff = datetime.now(UTC).timestamp() - window_seconds
            stage_metrics = [m for m in stage_metrics if datetime.fromisoformat(m.timestamp).timestamp() >= cutoff]

        if not stage_metrics:
            return ThroughputMetrics(
                total_records_in=0,
                total_records_out=0,
                records_processed=0,
                average_duration_ms=0.0,
                p95_duration_ms=0.0,
                stage_metrics={},
            )

        total_in = sum(m.input_size for m in stage_metrics)
        total_out = sum(m.output_size for m in stage_metrics)
        all_durations = [m.duration_ms for m in stage_metrics]

        # Aggregate by stage
        stage_agg: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "durations": [], "failures": 0})

        for m in stage_metrics:
            stage_agg[m.stage_name]["count"] += 1
            stage_agg[m.stage_name]["durations"].append(m.duration_ms)
            if m.status == "failed":
                stage_agg[m.stage_name]["failures"] += 1

        stage_metrics_dict = {}
        for stage_name, agg in stage_agg.items():
            durations = agg["durations"]
            stage_metrics_dict[stage_name] = {
                "count": agg["count"],
                "avg_ms": sum(durations) / len(durations) if durations else 0,
                "failures": agg["failures"],
            }

        sorted_durations = sorted(all_durations)
        p95_idx = int(len(sorted_durations) * 0.95)
        p95_duration = sorted_durations[p95_idx] if sorted_durations else 0.0

        return ThroughputMetrics(
            total_records_in=total_in,
            total_records_out=total_out,
            records_processed=total_out - total_in,  # net change through pipeline
            average_duration_ms=sum(all_durations) / len(all_durations) if all_durations else 0.0,
            p95_duration_ms=p95_duration,
            stage_metrics=stage_metrics_dict,
        )

    def get_readiness_metrics(self, window_seconds: int | None = None) -> ReadinessMetrics:
        """Get aggregated readiness validation metrics.

        Args:
            window_seconds: If provided, only consider metrics within this window.

        Returns:
            ReadinessMetrics with aggregated readiness validation data.
        """
        with self._lock:
            readiness_metrics = list(self._readiness_metrics)

        if window_seconds:
            cutoff = datetime.now(UTC).timestamp() - window_seconds
            readiness_metrics = [
                m for m in readiness_metrics if datetime.fromisoformat(m.timestamp).timestamp() >= cutoff
            ]

        if not readiness_metrics:
            return ReadinessMetrics(
                total_validations=0,
                passed=0,
                failed=0,
                conditionally_ready=0,
                by_stage={},
                recent_failures=[],
            )

        total = len(readiness_metrics)
        passed = sum(1 for m in readiness_metrics if m.passed)
        failed = sum(1 for m in readiness_metrics if not m.passed and m.status == ReadinessStatus.NOT_READY.value)
        conditional = sum(1 for m in readiness_metrics if m.status == ReadinessStatus.CONDITIONALLY_READY.value)

        # Aggregate by stage
        by_stage: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "failed": 0, "total": 0})
        for m in readiness_metrics:
            by_stage[m.stage_id]["total"] += 1
            if m.passed:
                by_stage[m.stage_id]["passed"] += 1
            else:
                by_stage[m.stage_id]["failed"] += 1

        # Recent failures (last 10)
        recent_failures = [
            {"package_id": m.package_id, "stage_id": m.stage_id, "failed_gates": m.failed_gates}
            for m in readiness_metrics[-10:]
            if not m.passed
        ]

        return ReadinessMetrics(
            total_validations=total,
            passed=passed,
            failed=failed,
            conditionally_ready=conditional,
            by_stage=dict(by_stage),
            recent_failures=recent_failures,
        )

    def get_failure_metrics(self, window_seconds: int | None = None) -> FailureMetrics:
        """Get aggregated failure metrics including regression detection.

        Args:
            window_seconds: If provided, only consider failures within this window.

        Returns:
            FailureMetrics with aggregated failures and detected regressions.
        """
        with self._lock:
            failures = list(self._failure_records)

        if window_seconds:
            cutoff = datetime.now(UTC).timestamp() - window_seconds
            failures = [f for f in failures if datetime.fromisoformat(f.timestamp).timestamp() >= cutoff]

        if not failures:
            return FailureMetrics(
                total_failures=0,
                by_stage={},
                by_gate={},
                regressions_detected=0,
                recent_failures=[],
                alert_regressions=[],
            )

        # Aggregate by stage and gate
        by_stage: dict[str, int] = defaultdict(int)
        by_gate: dict[str, int] = defaultdict(int)
        for f in failures:
            if f.stage:
                by_stage[f.stage] += 1
            if f.gate:
                by_gate[f.gate] += 1

        # Detect regressions
        regressions = self._detect_regressions(failures)

        return FailureMetrics(
            total_failures=len(failures),
            by_stage=dict(by_stage),
            by_gate=dict(by_gate),
            regressions_detected=len(regressions),
            recent_failures=failures[-20:],  # Last 20
            alert_regressions=regressions,
        )

    def _detect_regressions(self, failures: list[FailureRecord]) -> list[RegressionAlert]:
        """Detect regressions where the same failure pattern repeats."""
        # Group failures by pattern
        pattern_groups: dict[str, list[FailureRecord]] = defaultdict(list)
        for f in failures:
            # Normalize pattern for grouping
            pattern_key = f"{f.stage or ''}:{f.gate or ''}:{f.error_message[:30]}"
            pattern_groups[pattern_key].append(f)

        regressions = []
        for pattern, group in pattern_groups.items():
            if len(group) >= 2:  # Same pattern appearing 2+ times = regression
                timestamps = [datetime.fromisoformat(f.timestamp) for f in group]
                packages = [f.package_id for f in group if f.package_id]

                # Determine severity based on occurrence count
                severity = FailureSeverity.MEDIUM.value
                if len(group) >= 5:
                    severity = FailureSeverity.HIGH.value
                if len(group) >= 10:
                    severity = FailureSeverity.CRITICAL.value

                alert = RegressionAlert(
                    failure_pattern=pattern,
                    occurrences=len(group),
                    first_seen=min(timestamps).isoformat(),
                    last_seen=max(timestamps).isoformat(),
                    affected_packages=packages[:10],  # Limit to 10 for display
                    severity=severity,
                )
                regressions.append(alert)

        return regressions

    def get_health_summary(self) -> PipelineHealthSummary:
        """Get complete pipeline health summary.

        Returns:
            PipelineHealthSummary combining throughput, readiness, and failure metrics
            with an overall health status determination.
        """
        throughput = self.get_throughput_metrics()
        readiness = self.get_readiness_metrics()
        failures = self.get_failure_metrics()

        # Determine health status
        status = self._determine_health_status(throughput, readiness, failures)

        return PipelineHealthSummary(
            status=status.value,
            throughput=throughput,
            readiness=readiness,
            failures=failures,
            last_updated=datetime.now(UTC).isoformat(),
        )

    def _determine_health_status(
        self,
        throughput: ThroughputMetrics,
        readiness: ReadinessMetrics,
        failures: FailureMetrics,
    ) -> HealthStatus:
        """Determine overall pipeline health status."""
        # UNHEALTHY conditions
        if failures.regressions_detected >= 3:
            return HealthStatus.UNHEALTHY
        if failures.total_failures > 20:
            return HealthStatus.UNHEALTHY

        # Check for critical stage failures
        for _stage_name, metrics in throughput.stage_metrics.items():
            if metrics.get("failures", 0) > 5:
                return HealthStatus.UNHEALTHY

        # DEGRADED conditions
        if failures.regressions_detected >= 1:
            return HealthStatus.DEGRADED
        if failures.total_failures > 5:
            return HealthStatus.DEGRADED

        # Check for degraded throughput (many failures but not critical)
        total_executions = sum(m.get("count", 0) for m in throughput.stage_metrics.values())
        if total_executions > 0:
            total_failures = sum(m.get("failures", 0) for m in throughput.stage_metrics.values())
            failure_rate = total_failures / total_executions if total_executions else 0
            if failure_rate > 0.1:  # >10% failure rate
                return HealthStatus.DEGRADED

        # Check readiness failures
        if readiness.total_validations > 0:
            failure_rate = readiness.failed / readiness.total_validations
            if failure_rate > 0.3:  # >30% validation failure
                return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY


# Global metrics collector instance for convenience
_global_metrics: PipelineMetricsCollector | None = None
_global_lock = threading.Lock()


def get_metrics_collector() -> PipelineMetricsCollector:
    """Get the global metrics collector instance (singleton)."""
    global _global_metrics
    with _global_lock:
        if _global_metrics is None:
            _global_metrics = PipelineMetricsCollector()
        return _global_metrics


def get_health_summary() -> PipelineHealthSummary:
    """Convenience function to get pipeline health from global collector."""
    return get_metrics_collector().get_health_summary()


def record_stage_execution(
    stage_name: str,
    duration_ms: float,
    input_size: int,
    output_size: int,
    status: str,
    error: str | None = None,
) -> None:
    """Convenience function to record stage execution to global collector."""
    get_metrics_collector().record_stage_execution(
        stage_name=stage_name,
        duration_ms=duration_ms,
        input_size=input_size,
        output_size=output_size,
        status=status,
        error=error,
    )


def record_readiness_result(result: ReadinessResult) -> None:
    """Convenience function to record readiness result to global collector."""
    get_metrics_collector().record_readiness_result(result)


def record_failure(
    stage: str | None = None,
    gate: str | None = None,
    package_id: str | None = None,
    error_message: str = "",
    severity: str = FailureSeverity.MEDIUM,
) -> None:
    """Convenience function to record failure to global collector."""
    get_metrics_collector().record_failure(
        stage=stage,
        gate=gate,
        package_id=package_id,
        error_message=error_message,
        severity=severity,
    )


# Prometheus metrics export
def get_prometheus_metrics() -> str:
    """Generate Prometheus-format metrics for scraping.

    Returns:
        Multi-line string in Prometheus exposition format.
    """
    collector = get_metrics_collector()
    health = collector.get_health_summary()

    lines = [
        "# HELP pipeline_health_status Pipeline health status (1=healthy, 0.5=degraded, 0=unhealthy)",
        "# TYPE pipeline_health_status gauge",
    ]

    status_value = {"healthy": 1.0, "degraded": 0.5, "unhealthy": 0.0}.get(health.status, 0.0)
    lines.append(f"pipeline_health_status {status_value}")

    # Stage duration metrics
    lines.append("")
    lines.append("# HELP pipeline_stage_duration_ms Stage execution duration in milliseconds")
    lines.append("# TYPE pipeline_stage_duration_ms gauge")
    for stage_name, metrics in health.throughput.stage_metrics.items():
        labels = f'stage="{stage_name}"'
        lines.append(f"pipeline_stage_duration_ms{{{labels}}} {metrics.get('avg_ms', 0):.2f}")

    # Stage records processed
    lines.append("")
    lines.append("# HELP pipeline_stage_records_total Total records processed by stage")
    lines.append("# TYPE pipeline_stage_records_total counter")
    for stage_name, metrics in health.throughput.stage_metrics.items():
        labels = f'stage="{stage_name}"'
        lines.append(f"pipeline_stage_records_total{{{labels}}} {metrics.get('count', 0)}")

    # Stage failures
    lines.append("")
    lines.append("# HELP pipeline_stage_failures_total Stage failures total")
    lines.append("# TYPE pipeline_stage_failures_total counter")
    for stage_name, metrics in health.throughput.stage_metrics.items():
        labels = f'stage="{stage_name}"'
        lines.append(f"pipeline_stage_failures_total{{{labels}}} {metrics.get('failures', 0)}")

    # Readiness metrics
    lines.append("")
    lines.append("# HELP pipeline_readiness_total Readiness validations total")
    lines.append("# TYPE pipeline_readiness_total counter")
    lines.append(f'pipeline_readiness_total{{status="passed"}} {health.readiness.passed}')
    lines.append(f'pipeline_readiness_total{{status="failed"}} {health.readiness.failed}')
    lines.append(f'pipeline_readiness_total{{status="conditionally_ready"}} {health.readiness.conditionally_ready}')

    # Failure metrics
    lines.append("")
    lines.append("# HELP pipeline_failures_total Pipeline failures total")
    lines.append("# TYPE pipeline_failures_total counter")
    lines.append(f"pipeline_failures_total {health.failures.total_failures}")
    lines.append(f"pipeline_regressions_detected {health.failures.regressions_detected}")

    # Last updated
    lines.append("")
    lines.append("# HELP pipeline_health_last_updated Unix timestamp of last health check")
    lines.append("# TYPE pipeline_health_last_updated gauge")
    last_updated_ts = datetime.fromisoformat(health.last_updated).timestamp()
    lines.append(f"pipeline_health_last_updated {last_updated_ts:.0f}")

    return "\n".join(lines)


__all__ = [
    "FailureMetrics",
    "FailureRecord",
    "FailureSeverity",
    "HealthStatus",
    "PipelineHealthSummary",
    "PipelineMetricsCollector",
    "ReadinessMetric",
    "ReadinessMetrics",
    "RegressionAlert",
    "StageMetric",
    "ThroughputMetrics",
    "get_health_summary",
    "get_metrics_collector",
    "get_prometheus_metrics",
    "record_failure",
    "record_readiness_result",
    "record_stage_execution",
]
