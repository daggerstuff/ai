"""Evidence-based reprioritization engine for PIX-536.

This module implements the engine that reprioritizes acquisition and curation
work based on accumulated evaluation evidence.

Integration points
------------------
* Consumes feedback reports from PIX-508 (feedback_loop)
* Consumes pipeline health from PIX-507 (pipeline_observability)
* Produces reprioritized backlog items for Workstreams A, B, C
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ACTION_THRESHOLD = Decimal("0.3")

# Helpers for Decimal-aware formatting in to_dict()
_DECIMAL_ZERO = Decimal("0")


def _decimal_to_float(d: Decimal) -> float:
    """Convert a Decimal to float for JSON serialization."""
    return float(d)


def _float_to_decimal(f: float) -> Decimal:
    """Convert a float to Decimal with reasonable precision."""
    return Decimal(str(f))


@dataclass
class ReprioritizationConfig:
    """Runtime-tunable parameters for PIX-536 reprioritization.

    All threshold values use Decimal for precise comparisons
    and accumulation arithmetic (Gilfoyle review remediation).
    """

    action_threshold: Decimal = DEFAULT_ACTION_THRESHOLD
    churn_prevention_window_days: int = 7
    evidence_decay_rate: Decimal = Decimal("0.05")
    max_tracked_patterns: int = 10_000
    max_evidence_age_days: int = 30
    urgent_threshold: Decimal = Decimal("3.0")
    high_threshold: Decimal = Decimal("2.0")
    medium_threshold: Decimal = Decimal("1.0")
    low_threshold: Decimal = Decimal("0.5")
    reprioritize_score_delta_ratio: Decimal = Decimal("0.2")


class UpstreamDomain(StrEnum):
    ACQUISITION = "acquisition"
    CURATION = "curation"
    PRIVACY = "privacy"
    REVIEW = "review"
    PACKAGING = "packaging"


class InterventionType(StrEnum):
    PRIORITY_CHANGE = "priority_change"
    RULE_UPDATE = "rule_update"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    DATASET_FILTER = "dataset_filter"
    REVIEW_FOCUS = "review_focus"
    SOURCE_INTAKE = "source_intake"
    NORMALIZATION_UPDATE = "normalization_update"
    VALIDATION_GATE_UPDATE = "validation_gate_update"


class EvidenceSeverity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PriorityTier(StrEnum):
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKLOG = "backlog"


@dataclass
class EvidencePoint:
    pattern_id: str
    pattern_type: str
    description: str
    domain: UpstreamDomain
    severity: EvidenceSeverity
    frequency: float
    confidence: float
    root_cause_hypothesis: str
    metrics_impacted: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # NOTE: frequency and confidence remain float because they originate from
    # JSON input and are converted to Decimal in weight calculations.

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "domain": self.domain.value,
            "severity": self.severity.value,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "metrics_impacted": self.metrics_impacted,
            "timestamp": self.timestamp,
        }


@dataclass
class EvidenceAccumulation:
    pattern_id: str
    domain: UpstreamDomain
    description: str
    total_weight: Decimal = Decimal("0")
    last_updated: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    action_threshold: Decimal = DEFAULT_ACTION_THRESHOLD
    evidence_decay_rate: Decimal = Decimal("0.05")
    max_evidence_age_days: int = 30
    evidence_points: list[EvidencePoint] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    is_actionable: bool = False

    def add_evidence(self, point: EvidencePoint) -> None:
        self.evidence_points.append(point)
        self.last_seen = point.timestamp
        if not self.first_seen:
            self.first_seen = point.timestamp
        self._remove_old_evidence()
        self._recalculate_weight()
        self.is_actionable = self.total_weight >= self.action_threshold

    def _remove_old_evidence(self) -> None:
        """Remove evidence points older than max_evidence_age_days."""
        cutoff_date = datetime.now(UTC) - timedelta(days=self.max_evidence_age_days)
        self.evidence_points = [
            point for point in self.evidence_points if datetime.fromisoformat(point.timestamp) >= cutoff_date
        ]

    def _recalculate_weight(self) -> None:
        now = datetime.now(UTC)
        total = Decimal("0")
        for point in self.evidence_points:
            point_time = datetime.fromisoformat(point.timestamp)
            age_days = (now - point_time).total_seconds() / 86400.0
            decay = Decimal(str(math.exp(-float(self.evidence_decay_rate) * age_days)))
            severity_weight = _severity_weight(point.severity)
            point_weight = severity_weight * Decimal(str(point.frequency)) * Decimal(str(point.confidence)) * decay
            total += point_weight
        self.total_weight = total.quantize(Decimal("0.0001"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "domain": self.domain.value,
            "description": self.description,
            "evidence_count": len(self.evidence_points),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "total_weight": _decimal_to_float(self.total_weight),
            "action_threshold": _decimal_to_float(self.action_threshold),
            "is_actionable": self.is_actionable,
            "latest_severity": self.evidence_points[-1].severity.value if self.evidence_points else None,
        }


@dataclass
class BacklogItem:
    item_id: str
    domain: UpstreamDomain
    intervention_type: InterventionType
    title: str
    description: str
    priority_tier: PriorityTier
    priority_score: Decimal
    evidence_pattern_ids: list[str]
    root_cause_hypothesis: str
    validation_criteria: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    previous_priority_tier: PriorityTier | None = None
    reason_for_change: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "domain": self.domain.value,
            "intervention_type": self.intervention_type.value,
            "title": self.title,
            "description": self.description,
            "priority_tier": self.priority_tier.value,
            "priority_score": _decimal_to_float(self.priority_score),
            "evidence_pattern_ids": self.evidence_pattern_ids,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "validation_criteria": self.validation_criteria,
            "created_at": self.created_at,
            "previous_priority_tier": self.previous_priority_tier.value if self.previous_priority_tier else None,
            "reason_for_change": self.reason_for_change,
        }


@dataclass
class PriorityChange:
    item_id: str
    domain: UpstreamDomain
    previous_tier: PriorityTier | None
    new_tier: PriorityTier
    previous_score: Decimal
    new_score: Decimal
    reason: str
    evidence_pattern_ids: list[str]
    changed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "domain": self.domain.value,
            "previous_tier": self.previous_tier.value if self.previous_tier else None,
            "new_tier": self.new_tier.value,
            "previous_score": _decimal_to_float(self.previous_score),
            "new_score": _decimal_to_float(self.new_score),
            "reason": self.reason,
            "evidence_pattern_ids": self.evidence_pattern_ids,
            "changed_at": self.changed_at,
        }


@dataclass
class ReprioritizationReport:
    run_id: str
    timestamp: str
    evidence_sources_consumed: int
    total_evidence_points: int
    actionable_patterns: int
    backlog_items_created: int
    backlog_items_reprioritized: int
    priority_changes: list[PriorityChange]
    new_backlog_items: list[BacklogItem]
    reprioritized_items: list[BacklogItem]
    unchanged_items: list[BacklogItem]
    by_domain: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "evidence_sources_consumed": self.evidence_sources_consumed,
            "total_evidence_points": self.total_evidence_points,
            "actionable_patterns": self.actionable_patterns,
            "backlog_items_created": self.backlog_items_created,
            "backlog_items_reprioritized": self.backlog_items_reprioritized,
            "priority_changes": [c.to_dict() for c in self.priority_changes],
            "new_backlog_items": [i.to_dict() for i in self.new_backlog_items],
            "reprioritized_items": [i.to_dict() for i in self.reprioritized_items],
            "unchanged_items": [i.to_dict() for i in self.unchanged_items],
            "by_domain": self.by_domain,
        }

    def save(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class EvidenceAccumulator:
    def __init__(
        self,
        action_threshold: Decimal = DEFAULT_ACTION_THRESHOLD,
        config: ReprioritizationConfig | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._accumulations: dict[str, EvidenceAccumulation] = {}
        self._config = config or ReprioritizationConfig(action_threshold=action_threshold)

    def ingest_feedback_report(self, report_path: str | Path) -> list[EvidencePoint]:
        path = Path(report_path)
        with open(path) as f:
            report = json.load(f)
        return self.ingest_feedback_dict(report)

    def ingest_feedback_dict(self, report: dict[str, Any]) -> list[EvidencePoint]:
        points = self._parse_feedback_report(report)
        for point in points:
            self.record_evidence(point)
        return points

    def _parse_feedback_report(self, report: dict[str, Any]) -> list[EvidencePoint]:
        evidence_points: list[EvidencePoint] = []
        failure_patterns = report.get("failure_patterns", [])
        upstream_mappings = report.get("upstream_mappings", [])

        mapping_lookup: dict[str, dict[str, Any]] = {}
        for mapping in upstream_mappings:
            fp = mapping.get("failure_pattern", {})
            pattern_id = fp.get("pattern_id", "")
            if pattern_id:
                mapping_lookup[pattern_id] = mapping

        for pattern in failure_patterns:
            pattern_id = pattern.get("pattern_id", "")
            mapping = mapping_lookup.get(pattern_id, {})

            upstream_domain = mapping.get("upstream_domain", "curation")
            try:
                domain = UpstreamDomain(upstream_domain)
            except ValueError:
                logger.warning(
                    "Unknown upstream_domain %r for pattern %s; defaulting to curation",
                    upstream_domain,
                    pattern_id,
                )
                domain = UpstreamDomain.CURATION

            severity_str = pattern.get("severity", "medium")
            try:
                severity = EvidenceSeverity(severity_str)
            except ValueError:
                logger.warning(
                    "Unknown severity %r for pattern %s; defaulting to medium",
                    severity_str,
                    pattern_id,
                )
                severity = EvidenceSeverity.MEDIUM

            confidence = mapping.get("confidence", 0.5)
            root_cause = mapping.get("root_cause_hypothesis", pattern.get("description", ""))

            point = EvidencePoint(
                pattern_id=pattern_id,
                pattern_type=pattern.get("pattern_type", "unknown"),
                description=pattern.get("description", ""),
                domain=domain,
                severity=severity,
                frequency=pattern.get("frequency", 0.0),
                confidence=confidence,
                root_cause_hypothesis=root_cause,
                metrics_impacted=pattern.get("metrics_impacted", []),
            )
            evidence_points.append(point)

        return evidence_points

    def record_evidence(self, point: EvidencePoint) -> EvidenceAccumulation:
        with self._lock:
            if point.pattern_id not in self._accumulations:
                self._accumulations[point.pattern_id] = EvidenceAccumulation(
                    pattern_id=point.pattern_id,
                    domain=point.domain,
                    description=point.description,
                    action_threshold=self._config.action_threshold,
                    evidence_decay_rate=self._config.evidence_decay_rate,
                )
            accumulation = self._accumulations[point.pattern_id]
            accumulation.add_evidence(point)
            self._prune_accumulations_if_needed()
            return accumulation

    def _prune_accumulations_if_needed(self) -> None:
        limit = self._config.max_tracked_patterns
        if len(self._accumulations) <= limit:
            return
        inactive = [
            (pattern_id, accumulation)
            for pattern_id, accumulation in self._accumulations.items()
            if not accumulation.is_actionable
        ]
        inactive.sort(key=lambda item: item[1].last_seen or item[1].first_seen)
        while len(self._accumulations) > limit and inactive:
            pattern_id, _ = inactive.pop(0)
            self._accumulations.pop(pattern_id, None)

    def get_actionable_patterns(self) -> list[EvidenceAccumulation]:
        with self._lock:
            return [a for a in self._accumulations.values() if a.is_actionable]

    def get_accumulation(self, pattern_id: str) -> EvidenceAccumulation | None:
        with self._lock:
            return self._accumulations.get(pattern_id)

    def get_all_accumulations(self) -> dict[str, EvidenceAccumulation]:
        with self._lock:
            return dict(self._accumulations)

    def clear(self) -> None:
        with self._lock:
            self._accumulations.clear()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            actionable = [a for a in self._accumulations.values() if a.is_actionable]
            by_domain: dict[str, int] = {}
            for a in self._accumulations.values():
                domain_key = a.domain.value
                by_domain[domain_key] = by_domain.get(domain_key, 0) + 1
            return {
                "total_patterns": len(self._accumulations),
                "actionable_patterns": len(actionable),
                "total_evidence_points": sum(len(a.evidence_points) for a in self._accumulations.values()),
                "by_domain": by_domain,
            }


class PriorityCalculator:
    def __init__(self, config: ReprioritizationConfig | None = None) -> None:
        self._config = config or ReprioritizationConfig()

    DOMAIN_URGENCY: dict[UpstreamDomain, Decimal] = {
        UpstreamDomain.PRIVACY: Decimal("1.5"),
        UpstreamDomain.ACQUISITION: Decimal("1.2"),
        UpstreamDomain.CURATION: Decimal("1.0"),
        UpstreamDomain.REVIEW: Decimal("1.1"),
        UpstreamDomain.PACKAGING: Decimal("0.9"),
    }

    @property
    def urgent_threshold(self) -> Decimal:
        return self._config.urgent_threshold

    @property
    def high_threshold(self) -> Decimal:
        return self._config.high_threshold

    @property
    def medium_threshold(self) -> Decimal:
        return self._config.medium_threshold

    @property
    def low_threshold(self) -> Decimal:
        return self._config.low_threshold

    def calculate_priority(
        self,
        evidence_weight: Decimal,
        severity: EvidenceSeverity,
        frequency: float,
        domain: UpstreamDomain,
        coverage_gap: Decimal = Decimal("0"),
    ) -> tuple[Decimal, PriorityTier]:
        urgency_score = _severity_weight(severity) * self.DOMAIN_URGENCY.get(domain, Decimal("1.0"))
        evidence_component = evidence_weight * Decimal("0.4")
        urgency_component = urgency_score * Decimal(str(frequency)) * Decimal("0.3")
        coverage_component = coverage_gap * Decimal("0.3")
        priority_score = (evidence_component + urgency_component + coverage_component).quantize(Decimal("0.0001"))
        tier = self._score_to_tier(priority_score)
        return priority_score, tier

    def calculate_intervention_type(
        self,
        domain: UpstreamDomain,
        pattern_type: str,
        severity: EvidenceSeverity,
    ) -> InterventionType:
        pattern_intervention_map: dict[str, InterventionType] = {
            "memory_deficiency": InterventionType.SOURCE_INTAKE,
            "memory_noise": InterventionType.NORMALIZATION_UPDATE,
            "context_alignment": InterventionType.NORMALIZATION_UPDATE,
            "reflection_quality": InterventionType.REVIEW_FOCUS,
            "generation_quality": InterventionType.VALIDATION_GATE_UPDATE,
            "privacy_risk": InterventionType.RULE_UPDATE,
            "quality_degradation": InterventionType.THRESHOLD_ADJUSTMENT,
            "dataset_gap": InterventionType.DATASET_FILTER,
        }
        if domain == UpstreamDomain.PRIVACY:
            return InterventionType.RULE_UPDATE
        if domain == UpstreamDomain.ACQUISITION:
            return InterventionType.SOURCE_INTAKE
        if severity == EvidenceSeverity.CRITICAL and pattern_type not in pattern_intervention_map:
            return InterventionType.THRESHOLD_ADJUSTMENT
        return pattern_intervention_map.get(pattern_type, InterventionType.PRIORITY_CHANGE)

    def _score_to_tier(self, score: Decimal) -> PriorityTier:
        if score >= self.urgent_threshold:
            return PriorityTier.URGENT
        if score >= self.high_threshold:
            return PriorityTier.HIGH
        if score >= self.medium_threshold:
            return PriorityTier.MEDIUM
        if score >= self.low_threshold:
            return PriorityTier.LOW
        return PriorityTier.BACKLOG


class ReprioritizationEngine:
    def __init__(
        self,
        action_threshold: Decimal = DEFAULT_ACTION_THRESHOLD,
        churn_prevention_window_days: int = 7,
        config: ReprioritizationConfig | None = None,
    ) -> None:
        self._config = config or ReprioritizationConfig(
            action_threshold=action_threshold,
            churn_prevention_window_days=churn_prevention_window_days,
        )
        self.accumulator = EvidenceAccumulator(config=self._config)
        self.calculator = PriorityCalculator(config=self._config)
        self._backlog: dict[str, BacklogItem] = {}
        self._priority_changes: list[PriorityChange] = []
        self._churn_window = timedelta(days=self._config.churn_prevention_window_days)
        self._lock = threading.Lock()

    def load_feedback_report(self, report_path: str | Path) -> list[EvidencePoint]:
        path = Path(report_path)
        with open(path) as f:
            report = json.load(f)
        points = self._parse_feedback_report(report)
        for point in points:
            self.accumulator.record_evidence(point)
        return points

    def load_feedback_dict(self, report: dict[str, Any]) -> list[EvidencePoint]:
        points = self._parse_feedback_report(report)
        for point in points:
            self.accumulator.record_evidence(point)
        return points

    def _parse_feedback_report(self, report: dict[str, Any]) -> list[EvidencePoint]:
        return self.accumulator._parse_feedback_report(report)

    def add_existing_backlog(self, items: list[BacklogItem]) -> None:
        with self._lock:
            for item in items:
                self._backlog[item.item_id] = item

    def run_reprioritization(self) -> ReprioritizationReport:
        actionable = self.accumulator.get_actionable_patterns()
        all_accumulations = self.accumulator.get_all_accumulations()

        new_items: list[BacklogItem] = []
        reprioritized: list[BacklogItem] = []
        unchanged: list[BacklogItem] = []
        priority_changes: list[PriorityChange] = []

        for accumulation in actionable:
            latest_point = accumulation.evidence_points[-1]
            score, tier = self.calculator.calculate_priority(
                evidence_weight=accumulation.total_weight,
                severity=latest_point.severity,
                frequency=latest_point.frequency,
                domain=latest_point.domain,
            )
            intervention_type = self.calculator.calculate_intervention_type(
                domain=latest_point.domain,
                pattern_type=latest_point.pattern_type,
                severity=latest_point.severity,
            )
            item_id = _generate_item_id(accumulation.pattern_id, latest_point.domain)
            title = _generate_title(latest_point, accumulation)
            description = _generate_description(latest_point, accumulation)
            validation_criteria = _generate_validation_criteria(latest_point, intervention_type)

            existing = self._backlog.get(item_id)
            if existing:
                if not self._should_reprioritize(existing, tier, score):
                    unchanged.append(existing)
                    continue

                change = PriorityChange(
                    item_id=item_id,
                    domain=existing.domain,
                    previous_tier=existing.priority_tier,
                    new_tier=tier,
                    previous_score=existing.priority_score,
                    new_score=score,
                    reason=_generate_change_reason(existing, tier, score, accumulation),
                    evidence_pattern_ids=[accumulation.pattern_id],
                )
                priority_changes.append(change)
                existing.previous_priority_tier = existing.priority_tier
                existing.priority_tier = tier
                existing.priority_score = score
                existing.reason_for_change = change.reason
                existing.evidence_pattern_ids = list({*existing.evidence_pattern_ids, accumulation.pattern_id})
                reprioritized.append(existing)
            else:
                new_item = BacklogItem(
                    item_id=item_id,
                    domain=latest_point.domain,
                    intervention_type=intervention_type,
                    title=title,
                    description=description,
                    priority_tier=tier,
                    priority_score=score,
                    evidence_pattern_ids=[accumulation.pattern_id],
                    root_cause_hypothesis=latest_point.root_cause_hypothesis,
                    validation_criteria=validation_criteria,
                )
                self._backlog[item_id] = new_item
                new_items.append(new_item)

        with self._lock:
            for item_id, item in self._backlog.items():
                if item not in reprioritized and item not in new_items:
                    unchanged.append(item)

        new_items.sort(key=lambda x: x.priority_score, reverse=True)
        reprioritized.sort(key=lambda x: x.priority_score, reverse=True)
        unchanged.sort(key=lambda x: x.priority_score, reverse=True)

        by_domain = self._build_domain_summary(new_items, reprioritized, unchanged, all_accumulations)

        now = datetime.now(UTC).isoformat()
        self._priority_changes = priority_changes

        return ReprioritizationReport(
            run_id=_generate_run_id(),
            timestamp=now,
            evidence_sources_consumed=len(all_accumulations),
            total_evidence_points=sum(len(a.evidence_points) for a in all_accumulations.values()),
            actionable_patterns=len(actionable),
            backlog_items_created=len(new_items),
            backlog_items_reprioritized=len(reprioritized),
            priority_changes=priority_changes,
            new_backlog_items=new_items,
            reprioritized_items=reprioritized,
            unchanged_items=unchanged,
            by_domain=by_domain,
        )

    def _should_reprioritize(self, existing: BacklogItem, new_tier: PriorityTier, new_score: Decimal) -> bool:
        # Always reprioritize if the tier changes
        if existing.priority_tier != new_tier:
            return True
        # If the existing score is zero, any positive new score triggers reprioritization
        if existing.priority_score == Decimal("0"):
            return new_score > Decimal("0")
        # Calculate relative change in score
        score_change = abs(new_score - existing.priority_score) / existing.priority_score
        # Reprioritize if the relative change exceeds the configured threshold
        return score_change > self._config.reprioritize_score_delta_ratio

    def _build_domain_summary(
        self,
        new_items: list[BacklogItem],
        reprioritized: list[BacklogItem],
        unchanged: list[BacklogItem],
        accumulations: dict[str, EvidenceAccumulation],
    ) -> dict[str, dict[str, Any]]:
        all_items = new_items + reprioritized + unchanged
        by_domain: dict[str, dict[str, Any]] = {}

        for item in all_items:
            domain_key = item.domain.value
            if domain_key not in by_domain:
                by_domain[domain_key] = {
                    "total_items": 0,
                    "new_items": 0,
                    "reprioritized_items": 0,
                    "unchanged_items": 0,
                    "urgent_count": 0,
                    "high_count": 0,
                    "medium_count": 0,
                    "low_count": 0,
                    "backlog_count": 0,
                    "actionable_evidence_count": 0,
                }
            by_domain[domain_key]["total_items"] += 1
            if item in new_items:
                by_domain[domain_key]["new_items"] += 1
            elif item in reprioritized:
                by_domain[domain_key]["reprioritized_items"] += 1
            else:
                by_domain[domain_key]["unchanged_items"] += 1
            tier_key = f"{item.priority_tier.value}_count"
            by_domain[domain_key][tier_key] += 1

        for acc in accumulations.values():
            if acc.domain.value in by_domain and acc.is_actionable:
                by_domain[acc.domain.value]["actionable_evidence_count"] += 1

        return by_domain

    def get_backlog(self) -> list[BacklogItem]:
        items = list(self._backlog.values())
        items.sort(key=lambda x: x.priority_score, reverse=True)
        return items

    def get_backlog_by_domain(self, domain: UpstreamDomain) -> list[BacklogItem]:
        return [item for item in self.get_backlog() if item.domain == domain]

    def get_priority_changes(self) -> list[PriorityChange]:
        return list(self._priority_changes)


def _severity_weight(severity: EvidenceSeverity | str) -> Decimal:
    if isinstance(severity, str):
        try:
            severity = EvidenceSeverity(severity)
        except ValueError:
            return Decimal("1.0")
    weights: dict[EvidenceSeverity, Decimal] = {
        EvidenceSeverity.CRITICAL: Decimal("4.0"),
        EvidenceSeverity.HIGH: Decimal("3.0"),
        EvidenceSeverity.MEDIUM: Decimal("2.0"),
        EvidenceSeverity.LOW: Decimal("1.0"),
    }
    return weights.get(severity, Decimal("1.0"))


def _generate_item_id(pattern_id: str, domain: UpstreamDomain) -> str:
    raw = f"{domain.value}:{pattern_id}"
    return f"reprio-{hashlib.md5(raw.encode()).hexdigest()[:12]}"


def _generate_run_id() -> str:
    now = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return f"run-{now}"


def _generate_title(point: EvidencePoint, accumulation: EvidenceAccumulation) -> str:
    domain_label = point.domain.value.title()
    severity_label = point.severity.value.upper()
    evidence_count = len(accumulation.evidence_points)
    return f"[{severity_label}] {domain_label}: {point.description} ({evidence_count} evidence points)"


def _generate_description(point: EvidencePoint, accumulation: EvidenceAccumulation) -> str:
    lines = [
        f"**Domain**: {point.domain.value}",
        f"**Pattern**: {point.pattern_type}",
        f"**Severity**: {point.severity.value}",
        f"**Frequency**: {point.frequency:.1%}",
        f"**Confidence**: {point.confidence:.2f}",
        f"**Evidence Weight**: {_decimal_to_float(accumulation.total_weight):.4f}",
        f"**Evidence Points**: {len(accumulation.evidence_points)}",
        f"**First Seen**: {accumulation.first_seen}",
        f"**Last Seen**: {accumulation.last_seen}",
        "",
        f"**Root Cause Hypothesis**: {point.root_cause_hypothesis}",
        "",
        f"**Description**: {point.description}",
        "",
        f"**Metrics Impacted**: {', '.join(point.metrics_impacted) if point.metrics_impacted else 'None specified'}",
    ]
    return "\n".join(lines)


def _generate_validation_criteria(point: EvidencePoint, intervention_type: InterventionType) -> list[str]:
    criteria = [
        "Evidence weight exceeds action threshold",
        f"Severity: {point.severity.value}",
        f"Frequency: {point.frequency:.1%}",
    ]
    type_criteria = {
        InterventionType.SOURCE_INTAKE: [
            "New source qualified per acquisition rubric",
            "Pilot acquisition completed with metadata",
        ],
        InterventionType.RULE_UPDATE: [
            "Rule change reviewed and approved",
            "Privacy audit trail updated",
        ],
        InterventionType.THRESHOLD_ADJUSTMENT: [
            "New threshold validated against holdout data",
            "No regression in existing quality metrics",
        ],
        InterventionType.DATASET_FILTER: [
            "Filter criteria defined and tested",
            "Impact on dataset size quantified",
        ],
        InterventionType.REVIEW_FOCUS: [
            "Review criteria updated",
            "Human review queue updated with new focus area",
        ],
        InterventionType.NORMALIZATION_UPDATE: [
            "Normalization rule tested on sample data",
            "Dedup impact assessed",
        ],
        InterventionType.VALIDATION_GATE_UPDATE: [
            "New gate criteria defined",
            "Gate tested against existing packages",
        ],
    }
    criteria.extend(type_criteria.get(intervention_type, ["Intervention validated against evidence"]))
    return criteria


def _generate_change_reason(
    existing: BacklogItem,
    new_tier: PriorityTier,
    new_score: Decimal,
    accumulation: EvidenceAccumulation,
) -> str:
    direction = "increased" if new_score > existing.priority_score else "decreased"
    evidence_count = len(accumulation.evidence_points)
    return (
        f"Priority {direction} from {existing.priority_tier.value} to {new_tier.value} "
        f"(score: {_decimal_to_float(existing.priority_score):.4f} -> {_decimal_to_float(new_score):.4f}) "
        f"based on {evidence_count} accumulated evidence points for pattern {accumulation.pattern_id}"
    )


def create_engine(
    action_threshold: Decimal = DEFAULT_ACTION_THRESHOLD,
    churn_prevention_window_days: int = 7,
    config: ReprioritizationConfig | None = None,
) -> ReprioritizationEngine:
    return ReprioritizationEngine(
        action_threshold=action_threshold,
        churn_prevention_window_days=churn_prevention_window_days,
        config=config,
    )


def run_reprioritization_from_report(
    feedback_report_path: str | Path,
    output_path: str | Path | None = None,
    action_threshold: Decimal = DEFAULT_ACTION_THRESHOLD,
) -> ReprioritizationReport:
    engine = create_engine(action_threshold=action_threshold)
    engine.load_feedback_report(feedback_report_path)
    report = engine.run_reprioritization()
    if output_path:
        report.save(output_path)
    return report


__all__ = [
    "DEFAULT_ACTION_THRESHOLD",
    "BacklogItem",
    "EvidenceAccumulation",
    "EvidenceAccumulator",
    "EvidencePoint",
    "EvidenceSeverity",
    "InterventionType",
    "PriorityCalculator",
    "PriorityChange",
    "PriorityTier",
    "ReprioritizationConfig",
    "ReprioritizationEngine",
    "ReprioritizationReport",
    "UpstreamDomain",
    "create_engine",
    "run_reprioritization_from_report",
]
