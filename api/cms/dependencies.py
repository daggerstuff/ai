"""Shared FastAPI dependencies for CMS routes."""

from fastapi import Depends, HTTPException, Request

from ai.infrastructure.database.cms_connection_manager import CMSConnectionManager
from ai.infrastructure.database.dal.cached_repository import CMSCacheLayer
from ai.infrastructure.database.dal.postgres_repository import (
    ApprovalRequestRepository,
    ApprovalWorkflowRepository,
    AuditLogRepository,
    CommentRepository,
    DocumentActivityRepository,
    NotificationRepository,
    PermissionRepository,
)
from ai.infrastructure.database.dal.repositories import (
    BusinessDocumentRepository,
    KnowledgeArticleRepository,
    MarketResearchRepository,
    ProjectRepository,
    SalesOpportunityRepository,
    StrategicPlanRepository,
)


def get_cms_manager(request: Request) -> CMSConnectionManager:
    """Get the CMS connection manager from app state."""
    manager = getattr(request.app.state, "cms_db", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="CMS database not initialized")
    return manager


def get_document_repo(manager: CMSConnectionManager = Depends(get_cms_manager)) -> BusinessDocumentRepository:
    return BusinessDocumentRepository(manager.mongo.db)


def get_project_repo(manager: CMSConnectionManager = Depends(get_cms_manager)) -> ProjectRepository:
    return ProjectRepository(manager.mongo.db)


def get_market_research_repo(manager: CMSConnectionManager = Depends(get_cms_manager)) -> MarketResearchRepository:
    return MarketResearchRepository(manager.mongo.db)


def get_strategy_repo(manager: CMSConnectionManager = Depends(get_cms_manager)) -> StrategicPlanRepository:
    return StrategicPlanRepository(manager.mongo.db)


def get_sales_repo(manager: CMSConnectionManager = Depends(get_cms_manager)) -> SalesOpportunityRepository:
    return SalesOpportunityRepository(manager.mongo.db)


def get_knowledge_repo(manager: CMSConnectionManager = Depends(get_cms_manager)) -> KnowledgeArticleRepository:
    return KnowledgeArticleRepository(manager.mongo.db)


def get_cache_layer(manager: CMSConnectionManager = Depends(get_cms_manager)) -> CMSCacheLayer:
    return CMSCacheLayer(manager.redis.client)


def get_audit_repo(manager: CMSConnectionManager = Depends(get_cms_manager)) -> AuditLogRepository:
    return AuditLogRepository(manager.postgres.pool)


def get_notification_repo(manager: CMSConnectionManager = Depends(get_cms_manager)) -> NotificationRepository:
    return NotificationRepository(manager.postgres.pool)


def get_approval_workflow_repo(manager: CMSConnectionManager = Depends(get_cms_manager)) -> ApprovalWorkflowRepository:
    return ApprovalWorkflowRepository(manager.postgres.pool)


def get_approval_request_repo(manager: CMSConnectionManager = Depends(get_cms_manager)) -> ApprovalRequestRepository:
    return ApprovalRequestRepository(manager.postgres.pool)


def get_comment_repo(manager: CMSConnectionManager = Depends(get_cms_manager)) -> CommentRepository:
    return CommentRepository(manager.postgres.pool)


def get_activity_repo(manager: CMSConnectionManager = Depends(get_cms_manager)) -> DocumentActivityRepository:
    return DocumentActivityRepository(manager.postgres.pool)


def get_permission_repo(manager: CMSConnectionManager = Depends(get_cms_manager)) -> PermissionRepository:
    return PermissionRepository(manager.postgres.pool)


# Pagination helper


def pagination_params(skip: int = 0, limit: int = 100) -> dict[str, int]:
    """Common pagination query parameters."""
    return {"skip": max(0, skip), "limit": min(100, max(1, limit))}
