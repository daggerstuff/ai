"""FastAPI router for the PIX-511 Gate 4 human review queue."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

from ai.pkg_mera.core.pipelines.human_review_queue import (
    EscalationCriteria,
    HumanReviewQueue,
    ReviewDecision,
    Reviewer,
    ReviewerRole,
    ReviewItem,
    ReviewStatus,
)
from ai.memory.gates import GateDecision, GateResult, GatingReport

__all__ = [
    "AuditEntryResponse",
    "EnqueueRequest",
    "QueueStatsResponse",
    "ReviewDecisionRequest",
    "ReviewItemResponse",
    "create_review_router",
    "get_queue",
]


def get_queue() -> HumanReviewQueue:
    """Return the module-level queue singleton, initializing it lazily."""
    if not hasattr(get_queue, "_instance"):
        get_queue._instance = HumanReviewQueue()
    return get_queue._instance


class ReviewItemResponse(BaseModel):
    """Public API representation of a human review item."""

    item_id: str
    source_id: str
    status: str
    priority: str
    content_preview: str | None = None
    escalation_reason: str
    tags: list[str] = Field(default_factory=list)
    created_at: str


class ReviewDecisionRequest(BaseModel):
    """Reviewer decision payload for a pending queue item."""

    reviewer_id: str
    reviewer_role: str
    decision: str = Field(description='One of "approved", "rejected", or "returned"')
    reason: str
    additional_notes: dict[str, Any] | None = None


class QueueStatsResponse(BaseModel):
    """Aggregate human review queue statistics."""

    total_items: int
    pending_count: int
    approved_count: int
    rejected_count: int
    avg_queue_time_hours: float | None = None


class EnqueueRequest(BaseModel):
    """Payload used to enqueue a Gate 4 human review item."""

    source_id: str
    content_preview: str | None = None
    gating_report: dict[str, Any]
    priority: str = "normal"


class AuditEntryResponse(BaseModel):
    """Audit event exposed by the review queue API."""

    timestamp: str
    event: str
    reviewer_id: str | None = None
    decision: str | None = None
    reason: str | None = None


def _item_to_response(item: ReviewItem) -> ReviewItemResponse:
    return ReviewItemResponse(
        item_id=item.item_id,
        source_id=item.source_id,
        status=item.status.value,
        priority=item.priority,
        content_preview=item.content_preview,
        escalation_reason=item.escalation_reason,
        tags=item.tags,
        created_at=item.created_at,
    )


def _normalize_gating_report(report: GatingReport | dict[str, Any]) -> dict[str, Any]:
    if isinstance(report, dict):
        return dict(report)
    return report.to_dict()


def _gate4_result(reason: str, confidence: float = 1.0) -> dict[str, Any]:
    return GateResult(
        gate="gate4_review",
        decision=GateDecision.ESCALATE,
        reason=reason,
        confidence=confidence,
    ).to_dict()


def _validated_role(role: str) -> ReviewerRole:
    try:
        return ReviewerRole(role)
    except ValueError as exc:
        valid = ", ".join(member.value for member in ReviewerRole)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid reviewer_role '{role}'. Expected one of: {valid}",
        ) from exc


def _validated_decision(decision: str) -> ReviewStatus:
    try:
        status = ReviewStatus(decision)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid decision. Expected one of: approved, rejected, returned",
        ) from exc

    if status == ReviewStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail="Decision cannot be pending. Expected one of: approved, rejected, returned",
        )
    return status


def _get_existing_item(queue: HumanReviewQueue, item_id: str) -> ReviewItem:
    item = queue.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Review item '{item_id}' not found")
    return item


def _audit_entry_to_response(entry: dict[str, Any]) -> AuditEntryResponse:
    return AuditEntryResponse(
        timestamp=str(entry.get("timestamp", "")),
        event=str(entry.get("event", "decision")),
        reviewer_id=entry.get("reviewer_id"),
        decision=entry.get("decision"),
        reason=entry.get("reason") or entry.get("details"),
    )


def _handle_enqueue_item(request: EnqueueRequest) -> dict[str, str]:
    queue = get_queue()
    gate_result = _normalize_gating_report(request.gating_report)
    criteria = EscalationCriteria()
    reasons = criteria.get_escalation_reasons(gate_result)
    if reasons:
        gate_result.setdefault("gates", {})["gate4"] = _gate4_result("; ".join(reasons))

    content_length = gate_result.get("content_length")
    if not isinstance(content_length, int):
        content_length = len(request.content_preview or "")

    item = queue.create_item_from_report(
        source_id=request.source_id,
        gate_result=gate_result,
        content_preview=request.content_preview,
        content_length=content_length,
        priority=request.priority,
    )

    try:
        queue.enqueue(item)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"item_id": item.item_id}


def _handle_list_pending_items(limit: int, priority: str | None) -> list[ReviewItemResponse]:
    queue = get_queue()
    items = queue.list_items(status=ReviewStatus.PENDING, priority=priority)
    items.sort(key=lambda item: item.created_at)
    return [_item_to_response(item) for item in items[:limit]]


def _handle_get_review_item(item_id: str) -> ReviewItemResponse:
    item = _get_existing_item(get_queue(), item_id)
    return _item_to_response(item)


def _handle_apply_review_decision(item_id: str, request: ReviewDecisionRequest) -> ReviewItemResponse:
    queue = get_queue()
    item = _get_existing_item(queue, item_id)
    if item.status != ReviewStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Review item '{item_id}' is {item.status.value}, not pending",
        )

    reviewer = Reviewer(id=request.reviewer_id, role=_validated_role(request.reviewer_role))
    decision = ReviewDecision(
        item_id=item_id,
        reviewer=reviewer,
        decision=_validated_decision(request.decision),
        reason=request.reason,
        additional_notes=request.additional_notes,
    )

    try:
        updated = queue.apply_decision(decision)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _item_to_response(updated)


def _handle_get_review_stats() -> QueueStatsResponse:
    stats = get_queue().get_stats()
    return QueueStatsResponse(
        total_items=int(stats.get("total_items", 0)),
        pending_count=int(stats.get("pending_count", 0)),
        approved_count=int(stats.get("approved_count", 0)),
        rejected_count=int(stats.get("rejected_count", 0)),
        avg_queue_time_hours=stats.get("avg_queue_time_hours"),
    )


def _handle_get_audit_log(limit: int, item_id: str | None) -> list[AuditEntryResponse]:
    queue = get_queue()
    items = [_get_existing_item(queue, item_id)] if item_id else queue.list_items()

    entries: list[dict[str, Any]] = []
    for item in items:
        entries.extend(item.audit_trail)

    entries.sort(key=lambda entry: str(entry.get("timestamp", "")), reverse=True)
    return [_audit_entry_to_response(entry) for entry in entries[:limit]]


def _handle_update_confidence_display(
    item_id: str, confidence_data: Annotated[dict[str, Any], Body(...)]
) -> ReviewItemResponse:
    queue = get_queue()
    item = _get_existing_item(queue, item_id)
    gate_result = item.gate_result or {}
    metadata = gate_result.setdefault("metadata", {})
    metadata["confidence_display"] = confidence_data
    metadata["confidence_updated_at"] = datetime.now(UTC).isoformat()
    item.gate_result = gate_result
    item.audit_trail.append(
        {
            "event": "confidence_updated",
            "timestamp": metadata["confidence_updated_at"],
            "details": "Confidence display data updated",
        }
    )
    queue._persist_items()
    return _item_to_response(item)


def create_review_router() -> APIRouter:
    """Create the Gate 4 human review queue router."""

    router = APIRouter()

    @router.post("/review/enqueue")
    def enqueue_item(request: EnqueueRequest) -> dict[str, str]:
        return _handle_enqueue_item(request)

    @router.get("/review/pending", response_model=list[ReviewItemResponse])
    def list_pending_items(
        limit: int = Query(default=20, ge=1),
        priority: str | None = Query(default=None),
    ) -> list[ReviewItemResponse]:
        return _handle_list_pending_items(limit, priority)

    @router.get("/review/item/{item_id}", response_model=ReviewItemResponse)
    def get_review_item(item_id: str) -> ReviewItemResponse:
        return _handle_get_review_item(item_id)

    @router.post("/review/item/{item_id}/decision", response_model=ReviewItemResponse)
    def apply_review_decision(
        item_id: str,
        request: ReviewDecisionRequest,
    ) -> ReviewItemResponse:
        return _handle_apply_review_decision(item_id, request)

    @router.get("/review/stats", response_model=QueueStatsResponse)
    def get_review_stats() -> QueueStatsResponse:
        return _handle_get_review_stats()

    @router.get("/review/audit", response_model=list[AuditEntryResponse])
    def get_audit_log(
        limit: int = Query(default=50, ge=1),
        item_id: str | None = Query(default=None),
    ) -> list[AuditEntryResponse]:
        return _handle_get_audit_log(limit, item_id)

    @router.post("/review/item/{item_id}/confidence", response_model=ReviewItemResponse)
    def update_confidence_display(
        item_id: str,
        confidence_data: Annotated[dict[str, Any], Body(...)],
    ) -> ReviewItemResponse:
        return _handle_update_confidence_display(item_id, confidence_data)

    return router
