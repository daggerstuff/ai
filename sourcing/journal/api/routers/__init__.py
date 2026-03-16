"""
API routers for the Journal Dataset Research API.

This module provides route modules for all API endpoints.
"""

from fastapi import APIRouter


def create_api_router() -> APIRouter:
    """Create and configure the main API router."""
    # Create main API router
    api_router = APIRouter(prefix="/api/journal-research", tags=["journal-research"])

    # Import and include sub-routers (lazy import to avoid circular dependencies)
    from ai.sourcing.journal.api.routers import (
        sessions,
        discovery,
        evaluation,
        acquisition,
        integration,
        progress,
        reports,
        training,
        training_global,
    )

    # Include routers
    api_router.include_router(sessions.router, tags=["sessions"])
    api_router.include_router(discovery.router, tags=["discovery"])
    api_router.include_router(evaluation.router, tags=["evaluation"])
    api_router.include_router(acquisition.router, tags=["acquisition"])
    api_router.include_router(integration.router, tags=["integration"])
    api_router.include_router(progress.router, tags=["progress"])
    api_router.include_router(reports.router, tags=["reports"])
    api_router.include_router(training.router, tags=["training"])
    api_router.include_router(training_global.router, tags=["training"])

    return api_router


# Create the API router
api_router = create_api_router()

__all__ = ["api_router", "create_api_router"]

