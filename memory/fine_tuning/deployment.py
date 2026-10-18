"""Deployment Packaging & Monitoring — Sprint 5, Task 4.

Packages the winning fine-tuning approach for deployment,
creates monitoring dashboards, and provides rollback capability.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ai.memory.schema import MemoryBlock

from .evaluation import EvaluationReport

log = logging.getLogger(__name__)


class DeploymentStatus(StrEnum):
    READY = "ready"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass(frozen=True)
class DeploymentPackage:
    package_id: str
    version: str
    approach: str
    evaluation_report: EvaluationReport
    artifacts: dict[str, str]
    created_at_ms: int


@dataclass(frozen=True)
class MonitoringMetric:
    name: str
    value: float
    timestamp_ms: int
    tags: dict[str, str]


@dataclass
class MonitoringDashboard:
    metrics: list[MonitoringMetric]
    alerts: list[str]
    status: str


@dataclass
class RollbackPlan:
    steps: list[str]
    preconditions: list[str]
    postconditions: list[str]


class DeploymentPackager:
    """Package and deploy the memory system."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self._output_dir = output_dir
        self._counter = 0

    def create_package(
        self,
        approach: str,
        evaluation_report: EvaluationReport,
        artifacts: dict[str, str] | None = None,
    ) -> DeploymentPackage:
        """Create a deployment package."""
        self._counter += 1
        package = DeploymentPackage(
            package_id=f"pkg_{self._counter}_{int(time.time())}",
            version=f"5.{self._counter}.0",
            approach=approach,
            evaluation_report=evaluation_report,
            artifacts=artifacts or {},
            created_at_ms=int(time.time() * 1000),
        )

        if self._output_dir:
            self._save_package(package)

        log.info(
            "Deployment package created: %s (approach=%s, eval=%s)",
            package.package_id,
            approach,
            "PASS" if evaluation_report.overall_pass else "FAIL",
        )
        return package

    def validate_pre_deploy(self, package: DeploymentPackage) -> list[str]:
        """Run pre-deployment validation checks."""
        issues: list[str] = []

        if not package.evaluation_report.overall_pass:
            issues.append("Evaluation did not pass all gates")

        if package.evaluation_report.safety.pii_leak_rate > 0:
            issues.append("PII leak detected — cannot deploy")

        if package.evaluation_report.safety.crisis_sensitivity < 0.98:
            issues.append("Crisis detection sensitivity below 98% threshold")

        if package.evaluation_report.performance.p95_latency_ms >= 500:
            issues.append("P95 latency exceeds 500ms threshold")

        if package.evaluation_report.retrieval.precision_at_k < 0.75:
            issues.append("Retrieval precision below 0.75 threshold")

        return issues

    def create_rollback_plan(self) -> RollbackPlan:
        """Create a rollback plan for deployment."""
        return RollbackPlan(
            steps=[
                "1. Stop traffic to new deployment",
                "2. Switch DNS/load balancer to previous version",
                "3. Verify previous version is serving requests",
                "4. Monitor error rates and latency for 15 minutes",
                "5. Notify team of rollback",
                "6. Document rollback reason and metrics",
            ],
            preconditions=[
                "Previous version artifacts available",
                "Rollback script tested in staging",
                "Team notified of potential rollback",
            ],
            postconditions=[
                "All traffic routed to previous version",
                "Error rates within normal bounds",
                "Rollback documented in incident log",
            ],
        )

    def create_monitoring_dashboard(self, memories: list[MemoryBlock]) -> MonitoringDashboard:
        """Create monitoring dashboard with key metrics."""
        metrics: list[MonitoringMetric] = []
        alerts: list[str] = []
        now = int(time.time() * 1000)

        metrics.append(
            MonitoringMetric(
                name="memory_count",
                value=len(memories),
                timestamp_ms=now,
                tags={"type": "system"},
            )
        )

        crisis_count = sum(1 for m in memories if m.gating.crisisFlag)
        metrics.append(
            MonitoringMetric(
                name="crisis_memory_count",
                value=crisis_count,
                timestamp_ms=now,
                tags={"type": "safety"},
            )
        )

        avg_valence = sum(m.emotions.valence for m in memories) / len(memories) if memories else 0
        metrics.append(
            MonitoringMetric(
                name="avg_valence",
                value=round(avg_valence, 3),
                timestamp_ms=now,
                tags={"type": "emotional"},
            )
        )

        raw_count = sum(1 for m in memories if m.consolidation.phase.value == "raw")
        metrics.append(
            MonitoringMetric(
                name="raw_memory_ratio",
                value=round(raw_count / len(memories), 3) if memories else 0,
                timestamp_ms=now,
                tags={"type": "consolidation"},
            )
        )

        if crisis_count > len(memories) * 0.1:
            alerts.append("High crisis memory ratio (>10%)")
        if avg_valence < -0.3:
            alerts.append("Negative average valence detected")
        if raw_count > len(memories) * 0.5:
            alerts.append("High raw memory ratio — consolidation may be needed")

        return MonitoringDashboard(
            metrics=metrics,
            alerts=alerts,
            status="healthy" if not alerts else "warning",
        )

    def _save_package(self, package: DeploymentPackage) -> None:
        """Save deployment package to disk."""
        if not self._output_dir:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / f"{package.package_id}.json"
        data = {
            "package_id": package.package_id,
            "version": package.version,
            "approach": package.approach,
            "artifacts": package.artifacts,
            "created_at_ms": package.created_at_ms,
            "evaluation": {
                "overall_pass": package.evaluation_report.overall_pass,
                "retrieval": {
                    "precision_at_k": package.evaluation_report.retrieval.precision_at_k,
                    "recall_at_k": package.evaluation_report.retrieval.recall_at_k,
                    "mrr": package.evaluation_report.retrieval.mrr,
                },
                "safety": {
                    "crisis_sensitivity": package.evaluation_report.safety.crisis_sensitivity,
                    "pii_leak_rate": package.evaluation_report.safety.pii_leak_rate,
                },
                "performance": {
                    "p95_latency_ms": package.evaluation_report.performance.p95_latency_ms,
                },
            },
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("Package saved to %s", path)
