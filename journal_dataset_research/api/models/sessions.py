"""
Pydantic models for session management endpoints.

This module provides request and response models for session management.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from ai.journal_dataset_research.api.models.common import PaginatedResponse


class CreateSessionRequest(BaseModel):
    """Request model for creating a new session."""

    target_sources: List[str] = Field(
        default=["pubmed", "doaj"],
        description="Target sources for discovery",
    )
    search_keywords: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Search keywords organized by category",
    )
    weekly_targets: Dict[str, int] = Field(
        default_factory=dict,
        description="Weekly targets for research metrics",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID (auto-generated if not provided)",
    )


class SessionUpdateRequest(BaseModel):
    """Request model for updating a session."""

    target_sources: Optional[List[str]] = None
    search_keywords: Optional[Dict[str, List[str]]] = None
    weekly_targets: Optional[Dict[str, int]] = None
    current_phase: Optional[str] = Field(
        default=None,
        description="Current phase: discovery, evaluation, acquisition, integration",
    )


class SessionResponse(BaseModel):
    """Response model for session details."""

    session_id: str
    start_date: datetime
    target_sources: List[str]
    search_keywords: Dict[str, List[str]]
    weekly_targets: Dict[str, int]
    current_phase: str
    progress_metrics: Dict[str, int]

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "session_123",
                "start_date": "2025-01-21T00:00:00Z",
                "target_sources": ["pubmed", "doaj"],
                "search_keywords": {
                    "therapeutic": ["therapy", "counseling"],
                    "dataset": ["dataset", "conversation"],
                },
                "weekly_targets": {
                    "sources_identified": 10,
                    "datasets_evaluated": 5,
                },
                "current_phase": "discovery",
                "progress_metrics": {
                    "sources_identified": 5,
                    "datasets_evaluated": 2,
                },
            }
        }
    }


class SessionListResponse(PaginatedResponse[SessionResponse]):
    """Response model for session list."""

    pass

