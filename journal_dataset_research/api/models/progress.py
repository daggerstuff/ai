"""
Pydantic models for progress endpoints.

This module provides request and response models for progress tracking.
"""

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, Field, field_serializer


class ProgressMetricsResponse(BaseModel):
    """Response model for progress metrics."""

    session_id: str
    sources_identified: int = 0
    datasets_evaluated: int = 0
    datasets_acquired: int = 0
    integration_plans_created: int = 0
    last_updated: Optional[datetime] = None

    @field_serializer("last_updated")
    def serialize_datetime(self, value: Optional[datetime]) -> Optional[str]:
        """Serialize datetime to ISO format string."""
        return value.isoformat() if value else None

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "session_123",
                "sources_identified": 10,
                "datasets_evaluated": 5,
                "datasets_acquired": 3,
                "integration_plans_created": 2,
                "last_updated": "2025-01-21T00:00:00Z",
            }
        }
    }


class ProgressResponse(BaseModel):
    """Response model for progress information."""

    session_id: str
    current_phase: str
    progress_metrics: Dict[str, int]
    weekly_targets: Dict[str, int]
    progress_percentage: float = Field(
        default=0.0, ge=0, le=100, description="Overall progress percentage"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "session_123",
                "current_phase": "discovery",
                "progress_metrics": {
                    "sources_identified": 10,
                    "datasets_evaluated": 5,
                },
                "weekly_targets": {
                    "sources_identified": 10,
                    "datasets_evaluated": 5,
                },
                "progress_percentage": 50.0,
            }
        }
    }

