"""Steering integration for PIX-537.

Connects evaluation-driven reprioritization output to upstream workstreams
A (acquisition), B (curation), and C (quality handling). Ensures evaluation
findings actively steer source intake, curation rules, and quality handling
rather than living in isolated reports.

Integration points
------------------
* Consumes ReprioritizationReport from PIX-536 (reprioritization_engine)
* Maps backlog items to workstream-specific actions
* Tracks application state: pending, applied, rejected
* Produces SteeringReport for visibility into what changed and why
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from ai.pkg_mera.core.pipelines.reprioritization_engine import (
    DEFAULT_ACTION_THRESHOLD,
    BacklogItem,
    InterventionType,
    PriorityTier,
    ReprioritizationReport,
    UpstreamDomain,
)


class Workstream(StrEnum):
    ACQUISITION = "acquisition"
    CURATION = "curation"
    QUALITY_HANDLING = "quality_handling"


class SteeringActionType(StrEnum):
    UPDATE_ACQUISITION_RUBRIC = "update_acquisition_rubric"
    ADD_SOURCE_PRIORITY = "add_source_priority"
    UPDATE_NORMALIZATION_RULE = "update_normalization_rule"
    UPDATE_DATASET_FILTER = "update_dataset_filter"
    UPDATE_PRIVACY_RULE = "update_privacy_rule"
    UPDATE_REVIEW_FOCUS = "update_review_focus"
    UPDATE_VALIDATION_GATE = "update_validation_gate"
    ADJUST_THRESHOLD = "adjust_threshold"


class ApplicationStatus(StrEnum):
    PENDING = "pending"
    APPLIED = "applied"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


@dataclass
class SteeringAction:
    action_id: str
    workstream: Workstream
    action_type: SteeringActionType
    description: str
    details: dict[str, Any]
    source_item_id: str
    source_pattern_id: str
    evidence_weight: float
    priority_tier: PriorityTier
    status: ApplicationStatus = ApplicationStatus.PENDING
    applied_at: str | None = None
    rejection_reason: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "workstream": self.workstream.value,
            "action_type": self.action_type.value,
            "description": self.description,
            "details": self.details,
            "source_item_id": self.source_item_id,
            "source_pattern_id": self.source_pattern_id,
            "evidence_weight": self.evidence_weight,
            "priority_tier": self.priority_tier.value,
            "status": self.status.value,
            "applied_at": self.applied_at,
            "rejection_reason": self.rejection_reason,
            "created_at": self.created_at,
        }


@dataclass
class WorkstreamState:
    workstream: Workstream
    active_rules: list[dict[str, Any]] = field(default_factory=list)
    pending_actions: list[str] = field(default_factory=list)
    applied_actions: list[str] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "workstream": self.workstream.value,
            "active_rules": self.active_rules,
            "pending_actions": self.pending_actions,
            "applied_actions": self.applied_actions,
            "last_updated": self.last_updated,
        }


@dataclass
class SteeringReport:
    run_id: str
    timestamp: str
    source_report_id: str
    total_actions_generated: int
    actions_by_workstream: dict[str, int]
    actions_by_type: dict[str, int]
    actions_applied: int
    actions_pending: int
    actions_rejected: int
    actions: list[SteeringAction]
    workstream_states: dict[str, WorkstreamState]
    audit_trail: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "source_report_id": self.source_report_id,
            "total_actions_generated": self.total_actions_generated,
            "actions_by_workstream": self.actions_by_workstream,
            "actions_by_type": self.actions_by_type,
            "actions_applied": self.actions_applied,
            "actions_pending": self.actions_pending,
            "actions_rejected": self.actions_rejected,
            "actions": [a.to_dict() for a in self.actions],
            "workstream_states": {k: v.to_dict() for k, v in self.workstream_states.items()},
            "audit_trail": self.audit_trail,
        }

    def save(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def _domain_to_workstream(domain: UpstreamDomain) -> Workstream:
    mapping = {
        UpstreamDomain.ACQUISITION: Workstream.ACQUISITION,
        UpstreamDomain.CURATION: Workstream.CURATION,
        UpstreamDomain.PRIVACY: Workstream.QUALITY_HANDLING,
        UpstreamDomain.REVIEW: Workstream.QUALITY_HANDLING,
        UpstreamDomain.PACKAGING: Workstream.CURATION,
    }
    return mapping.get(domain, Workstream.CURATION)


def _intervention_to_action_type(intervention: InterventionType, domain: UpstreamDomain) -> SteeringActionType:
    mapping = {
        InterventionType.SOURCE_INTAKE: SteeringActionType.ADD_SOURCE_PRIORITY,
        InterventionType.RULE_UPDATE: SteeringActionType.UPDATE_PRIVACY_RULE,
        InterventionType.THRESHOLD_ADJUSTMENT: SteeringActionType.ADJUST_THRESHOLD,
        InterventionType.DATASET_FILTER: SteeringActionType.UPDATE_DATASET_FILTER,
        InterventionType.REVIEW_FOCUS: SteeringActionType.UPDATE_REVIEW_FOCUS,
        InterventionType.NORMALIZATION_UPDATE: SteeringActionType.UPDATE_NORMALIZATION_RULE,
        InterventionType.VALIDATION_GATE_UPDATE: SteeringActionType.UPDATE_VALIDATION_GATE,
        InterventionType.PRIORITY_CHANGE: SteeringActionType.ADD_SOURCE_PRIORITY,
    }
    action_type = mapping.get(intervention, SteeringActionType.ADD_SOURCE_PRIORITY)
    if domain == UpstreamDomain.ACQUISITION:
        return SteeringActionType.ADD_SOURCE_PRIORITY
    if domain == UpstreamDomain.PRIVACY:
        return SteeringActionType.UPDATE_PRIVACY_RULE
    return action_type


def _generate_action_id(item_id: str, workstream: Workstream) -> str:
    return f"steer-{workstream.value[:3]}-{item_id}"


def _generate_action_details(item: BacklogItem, action_type: SteeringActionType) -> dict[str, Any]:
    details = {
        "root_cause": item.root_cause_hypothesis,
        "validation_criteria": item.validation_criteria,
        "evidence_patterns": item.evidence_pattern_ids,
        "priority_score": float(item.priority_score),
    }
    if action_type == SteeringActionType.ADD_SOURCE_PRIORITY:
        details["action"] = "Prioritize acquisition of sources matching identified gap"
        details["domain"] = "acquisition"
    elif action_type == SteeringActionType.UPDATE_NORMALIZATION_RULE:
        details["action"] = "Update normalization rules to address identified pattern"
        details["domain"] = "curation"
    elif action_type == SteeringActionType.UPDATE_PRIVACY_RULE:
        details["action"] = "Update privacy/content handling rules"
        details["domain"] = "quality"
    elif action_type == SteeringActionType.UPDATE_REVIEW_FOCUS:
        details["action"] = "Add review focus area for human review queue"
        details["domain"] = "quality"
    elif action_type == SteeringActionType.ADJUST_THRESHOLD:
        details["action"] = "Adjust quality threshold based on evaluation evidence"
        details["domain"] = "curation"
    elif action_type == SteeringActionType.UPDATE_DATASET_FILTER:
        details["action"] = "Update dataset filter criteria"
        details["domain"] = "curation"
    elif action_type == SteeringActionType.UPDATE_VALIDATION_GATE:
        details["action"] = "Update validation gate criteria"
        details["domain"] = "curation"
    return details


class SteeringIntegration:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._actions: dict[str, SteeringAction] = {}
        self._workstream_states: dict[Workstream, WorkstreamState] = {
            ws: WorkstreamState(workstream=ws) for ws in Workstream
        }
        self._audit_trail: list[dict[str, Any]] = []
        self._handlers: dict[SteeringActionType, Callable] = {}

    def register_handler(self, action_type: SteeringActionType, handler: Callable) -> None:
        self._handlers[action_type] = handler

    def process_report(self, report: ReprioritizationReport) -> SteeringReport:
        actions = self._generate_actions(report)
        applied, pending, rejected = self._execute_actions(actions)

        actions_by_workstream: dict[str, int] = {}
        actions_by_type: dict[str, int] = {}
        for action in actions:
            ws_key = action.workstream.value
            actions_by_workstream[ws_key] = actions_by_workstream.get(ws_key, 0) + 1
            type_key = action.action_type.value
            actions_by_type[type_key] = actions_by_type.get(type_key, 0) + 1

        now = datetime.now(UTC).isoformat()
        steering_report = SteeringReport(
            run_id=f"steer-{report.run_id}",
            timestamp=now,
            source_report_id=report.run_id,
            total_actions_generated=len(actions),
            actions_by_workstream=actions_by_workstream,
            actions_by_type=actions_by_type,
            actions_applied=len(applied),
            actions_pending=len(pending),
            actions_rejected=len(rejected),
            actions=actions,
            workstream_states={ws.value: state for ws, state in self._workstream_states.items()},
            audit_trail=list(self._audit_trail),
        )

        self._audit_trail.append(
            {
                "event": "steering_report_processed",
                "timestamp": now,
                "source_report_id": report.run_id,
                "actions_generated": len(actions),
                "actions_applied": len(applied),
                "actions_pending": len(pending),
                "actions_rejected": len(rejected),
            }
        )

        return steering_report

    def _generate_actions(self, report: ReprioritizationReport) -> list[SteeringAction]:
        actions: list[SteeringAction] = []
        all_items = report.new_backlog_items + report.reprioritized_items

        for item in all_items:
            workstream = _domain_to_workstream(item.domain)
            action_type = _intervention_to_action_type(item.intervention_type, item.domain)
            action_id = _generate_action_id(item.item_id, workstream)

            with self._lock:
                existing = self._actions.get(action_id)
                if existing and existing.status in (
                    ApplicationStatus.APPLIED,
                    ApplicationStatus.PENDING,
                ):
                    self._audit_trail.append(
                        {
                            "event": "action_skipped_idempotent",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "action_id": action_id,
                            "status": existing.status.value,
                        }
                    )
                    continue

            details = _generate_action_details(item, action_type)

            action = SteeringAction(
                action_id=action_id,
                workstream=workstream,
                action_type=action_type,
                description=item.title,
                details=details,
                source_item_id=item.item_id,
                source_pattern_id=item.evidence_pattern_ids[0] if item.evidence_pattern_ids else "",
                evidence_weight=float(item.priority_score),
                priority_tier=item.priority_tier,
            )
            actions.append(action)

            with self._lock:
                self._actions[action_id] = action
                ws_state = self._workstream_states[workstream]
                ws_state.pending_actions.append(action_id)
                ws_state.last_updated = datetime.now(UTC).isoformat()

        return actions

    def _execute_actions(
        self, actions: list[SteeringAction]
    ) -> tuple[list[SteeringAction], list[SteeringAction], list[SteeringAction]]:
        applied: list[SteeringAction] = []
        pending: list[SteeringAction] = []
        rejected: list[SteeringAction] = []

        for action in actions:
            handler = self._handlers.get(action.action_type)
            if handler is None:
                action.status = ApplicationStatus.PENDING
                pending.append(action)
                self._audit_trail.append(
                    {
                        "event": "action_pending_no_handler",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "action_id": action.action_id,
                        "action_type": action.action_type.value,
                    }
                )
                continue

            try:
                result = handler(action)
                if result.get("status") == "applied":
                    action.status = ApplicationStatus.APPLIED
                    action.applied_at = datetime.now(UTC).isoformat()
                    applied.append(action)
                    with self._lock:
                        ws_state = self._workstream_states[action.workstream]
                        if action.action_id in ws_state.pending_actions:
                            ws_state.pending_actions.remove(action.action_id)
                        ws_state.applied_actions.append(action.action_id)
                        ws_state.last_updated = datetime.now(UTC).isoformat()
                    self._audit_trail.append(
                        {
                            "event": "action_applied",
                            "timestamp": action.applied_at,
                            "action_id": action.action_id,
                            "workstream": action.workstream.value,
                        }
                    )
                else:
                    action.status = ApplicationStatus.REJECTED
                    action.rejection_reason = result.get("reason", "Handler rejected")
                    rejected.append(action)
                    self._audit_trail.append(
                        {
                            "event": "action_rejected",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "action_id": action.action_id,
                            "reason": action.rejection_reason,
                        }
                    )
            except Exception as e:
                action.status = ApplicationStatus.PENDING
                pending.append(action)
                self._audit_trail.append(
                    {
                        "event": "action_error",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "action_id": action.action_id,
                        "error": str(e),
                    }
                )

        return applied, pending, rejected

    def get_action(self, action_id: str) -> SteeringAction | None:
        return self._actions.get(action_id)

    def get_actions_by_workstream(self, workstream: Workstream) -> list[SteeringAction]:
        return [a for a in self._actions.values() if a.workstream == workstream]

    def get_actions_by_status(self, status: ApplicationStatus) -> list[SteeringAction]:
        return [a for a in self._actions.values() if a.status == status]

    def get_workstream_state(self, workstream: Workstream) -> WorkstreamState:
        return self._workstream_states[workstream]

    def get_all_actions(self) -> list[SteeringAction]:
        return list(self._actions.values())

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            by_status: dict[str, int] = {}
            by_workstream: dict[str, int] = {}
            for action in self._actions.values():
                status_key = action.status.value
                by_status[status_key] = by_status.get(status_key, 0) + 1
                ws_key = action.workstream.value
                by_workstream[ws_key] = by_workstream.get(ws_key, 0) + 1
            return {
                "total_actions": len(self._actions),
                "by_status": by_status,
                "by_workstream": by_workstream,
                "workstream_states": {ws.value: state.to_dict() for ws, state in self._workstream_states.items()},
            }


def create_steering_integration() -> SteeringIntegration:
    return SteeringIntegration()


def run_steering_from_report(
    feedback_report_path: str | Path,
    steering_output_path: str | Path | None = None,
    action_threshold: Decimal = DEFAULT_ACTION_THRESHOLD,
    handlers: dict[SteeringActionType, Callable] | None = None,
) -> SteeringReport:
    from ai.pkg_mera.core.pipelines.reprioritization_engine import run_reprioritization_from_report

    reprio_report = run_reprioritization_from_report(
        feedback_report_path,
        action_threshold=action_threshold,
    )

    steering = create_steering_integration()
    if handlers:
        for action_type, handler in handlers.items():
            steering.register_handler(action_type, handler)

    report = steering.process_report(reprio_report)

    if steering_output_path:
        report.save(steering_output_path)

    return report


__all__ = [
    "ApplicationStatus",
    "SteeringAction",
    "SteeringActionType",
    "SteeringIntegration",
    "SteeringReport",
    "Workstream",
    "WorkstreamState",
    "create_steering_integration",
    "run_steering_from_report",
]
