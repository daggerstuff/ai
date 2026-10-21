"""Storage Quota Optimizer for CoreWeave Forge & W&B Artifacts.

Prevents unexpected storage overages against the 200 GB quota by
auditing checkpoint retention, safely archiving intermediate versions,
and enforcing automated TTL pruning policies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.forge.config import ForgeConfig

logger = logging.getLogger(__name__)


@dataclass
class ArtifactVersionInfo:
    """Metadata for an artifact version."""

    name: str
    version: str
    size_bytes: int
    is_milestone: bool = False
    is_terminal: bool = False


@dataclass
class QuotaAuditReport:
    """Summary report of storage utilization and pruning recommendations."""

    total_storage_bytes: int
    total_storage_gb: float
    quota_limit_gb: float
    utilization_pct: float
    is_over_warning_threshold: bool
    retained_versions_count: int
    prune_candidates_count: int
    estimated_reclaimable_gb: float
    prune_candidates: list[str] = field(default_factory=list)


class StorageQuotaOptimizer:
    """Analyzes and optimizes CoreWeave Forge artifact storage consumption."""

    def __init__(self, config: ForgeConfig | None = None):
        self.config = config or ForgeConfig()

    def classify_versions(
        self,
        versions: list[dict[str, Any]],
        keep_every_n: int = 10,
        keep_last_n: int = 3,
    ) -> list[ArtifactVersionInfo]:
        """Classify artifact versions into protected milestones vs prune candidates."""
        sorted_versions = sorted(
            versions,
            key=lambda x: int(
                x.get("step", 0)
                or (
                    x.get("version", "v0")
                    .lstrip("v")
                    .split("-")[0]
                    .split(":")[0]
                    if x.get("version", "").lstrip("v").isdigit()
                    else 0
                )
            ),
        )

        classified: list[ArtifactVersionInfo] = []
        total = len(sorted_versions)

        for i, item in enumerate(sorted_versions):
            name = item.get("name", "unknown")
            version = item.get("version", f"v{i}")
            size_bytes = int(item.get("size_bytes", 0))

            is_first = i == 0
            is_last_n = i >= (total - keep_last_n)
            is_milestone = (i % keep_every_n == 0) or is_first or is_last_n

            classified.append(
                ArtifactVersionInfo(
                    name=name,
                    version=version,
                    size_bytes=size_bytes,
                    is_milestone=is_milestone,
                    is_terminal=is_last_n,
                )
            )

        return classified

    def generate_audit_report(
        self,
        artifact_versions: list[dict[str, Any]],
        keep_every_n: int = 10,
        keep_last_n: int = 3,
    ) -> QuotaAuditReport:
        """Audit artifact storage against the quota and produce prune recommendations."""
        classified = self.classify_versions(
            artifact_versions, keep_every_n, keep_last_n
        )

        total_bytes = sum(v.size_bytes for v in classified)
        total_gb = total_bytes / (1024**3)
        quota_gb = self.config.storage_quota_gb
        util_pct = (total_gb / quota_gb) * 100.0 if quota_gb > 0 else 0.0

        prune_candidates = [
            f"{v.name}:{v.version}" for v in classified if not v.is_milestone
        ]
        reclaimable_bytes = sum(
            v.size_bytes for v in classified if not v.is_milestone
        )
        reclaimable_gb = reclaimable_bytes / (1024**3)

        return QuotaAuditReport(
            total_storage_bytes=total_bytes,
            total_storage_gb=round(total_gb, 2),
            quota_limit_gb=quota_gb,
            utilization_pct=round(util_pct, 2),
            is_over_warning_threshold=util_pct
            >= self.config.storage_warn_threshold_pct,
            retained_versions_count=len(classified) - len(prune_candidates),
            prune_candidates_count=len(prune_candidates),
            estimated_reclaimable_gb=round(reclaimable_gb, 2),
            prune_candidates=prune_candidates,
        )

    def write_prune_plan(
        self, report: QuotaAuditReport, output_path: Path
    ) -> None:
        """Save the audit report and pruning plan to disk."""
        data = {
            "quota_limit_gb": report.quota_limit_gb,
            "current_storage_gb": report.total_storage_gb,
            "utilization_pct": report.utilization_pct,
            "reclaimable_gb": report.estimated_reclaimable_gb,
            "prune_candidates_count": report.prune_candidates_count,
            "prune_candidates": report.prune_candidates,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
