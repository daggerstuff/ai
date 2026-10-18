"""Workflow, approval, notification, and comment API routes.

These operate on the PostgreSQL relational tables managed by
the postgres_repository DAL.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ai.infrastructure.database.dal.postgres_repository import (
    ApprovalRequestRepository,
    ApprovalWorkflowRepository,
    CommentRepository,
    NotificationRepository,
    PgFilter,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _workflow_repo(request: Request) -> ApprovalWorkflowRepository:
    return ApprovalWorkflowRepository(request.app.state.cms_db.postgres.pool)


def _request_repo(request: Request) -> ApprovalRequestRepository:
    return ApprovalRequestRepository(request.app.state.cms_db.postgres.pool)


def _notification_repo(request: Request) -> NotificationRepository:
    return NotificationRepository(request.app.state.cms_db.postgres.pool)


def _comment_repo(request: Request) -> CommentRepository:
    return CommentRepository(request.app.state.cms_db.postgres.pool)


# --- Approval Workflows ---


@router.get("/workflows")
async def list_workflows(request: Request, resource_type: str | None = None) -> dict[str, Any]:
    repo = _workflow_repo(request)
    workflows = repo.find_by_resource_type(resource_type) if resource_type else repo.find_many(order_by="name")
    return {"success": True, "data": workflows}


@router.get("/workflows/{workflow_id}")
async def get_workflow(request: Request, workflow_id: str) -> dict[str, Any]:
    repo = _workflow_repo(request)
    doc = repo.get_by_id(workflow_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return {"success": True, "data": doc}


# --- Approval Requests ---


@router.get("/requests")
async def list_requests(
    request: Request,
    requestor_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
) -> dict[str, Any]:
    repo = _request_repo(request)
    if requestor_id:
        docs = repo.find_pending_by_requestor(requestor_id)
    elif resource_type and resource_id:
        docs = repo.find_by_resource(resource_type, resource_id)
    else:
        docs = repo.find_many(
            filters=[PgFilter("status", "=", "pending")],
            order_by="created_at",
            order_desc=True,
        )
    return {"success": True, "data": docs}


@router.get("/requests/{request_id}")
async def get_request(request: Request, request_id: str) -> dict[str, Any]:
    repo = _request_repo(request)
    doc = repo.get_by_id(request_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Approval request not found")
    return {"success": True, "data": doc}


@router.post("/steps/{step_id}/approve")
async def approve_step(request: Request, step_id: str) -> dict[str, Any]:
    body = await request.json()
    reviewed_by = body.get("reviewed_by", "")
    comments = body.get("comments", "")
    repo = _request_repo(request)
    ok = repo.approve_step(step_id, reviewed_by, comments)
    if not ok:
        raise HTTPException(status_code=404, detail="Step not found")
    return {"success": True}


@router.post("/steps/{step_id}/reject")
async def reject_step(request: Request, step_id: str) -> dict[str, Any]:
    body = await request.json()
    reviewed_by = body.get("reviewed_by", "")
    comments = body.get("comments", "")
    repo = _request_repo(request)
    ok = repo.reject_step(step_id, reviewed_by, comments)
    if not ok:
        raise HTTPException(status_code=404, detail="Step not found")
    return {"success": True}


# --- Notifications ---


@router.get("/notifications/{user_id}")
async def list_notifications(request: Request, user_id: str, limit: int = 50) -> dict[str, Any]:
    repo = _notification_repo(request)
    docs = repo.find_unread(user_id, limit=limit)
    return {"success": True, "data": docs}


@router.post("/notifications")
async def create_notification(request: Request) -> dict[str, Any]:
    body = await request.json()
    repo = _notification_repo(request)
    doc = repo.create(
        user_id=body["user_id"],
        notif_type=body["type"],
        title=body["title"],
        message=body["message"],
        resource_type=body.get("resource_type", ""),
        resource_id=body.get("resource_id", ""),
    )
    return {"success": True, "data": doc}


@router.patch("/notifications/{notification_id}/read")
async def mark_notification_read(request: Request, notification_id: str) -> dict[str, Any]:
    repo = _notification_repo(request)
    ok = repo.mark_read(notification_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"success": True}


@router.post("/notifications/{user_id}/read-all")
async def mark_all_read(request: Request, user_id: str) -> dict[str, Any]:
    repo = _notification_repo(request)
    count = repo.mark_all_read(user_id)
    return {"success": True, "data": {"marked_read": count}}


# --- Comments ---


@router.get("/comments/{document_id}")
async def list_comments(request: Request, document_id: str, limit: int = 100) -> dict[str, Any]:
    repo = _comment_repo(request)
    docs = repo.find_by_document(document_id, limit=limit)
    return {"success": True, "data": docs}


@router.post("/comments/{comment_id}/resolve")
async def resolve_comment(request: Request, comment_id: str) -> dict[str, Any]:
    body = await request.json()
    resolved_by = body.get("resolved_by", "")
    repo = _comment_repo(request)
    ok = repo.resolve(comment_id, resolved_by)
    if not ok:
        raise HTTPException(status_code=404, detail="Comment not found")
    return {"success": True}
