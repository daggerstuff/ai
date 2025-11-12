"""
Pydantic models for integration endpoints.

This module provides request and response models for integration planning.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_serializer

from ai.journal_dataset_research.api.models.common import PaginatedResponse


class IntegrationInitiateRequest(BaseModel):
    """Request model for initiating integration planning."""

    source_ids: Optional[List[str]] = Field(
        default=None,
        description="Source IDs to integrate (all sources if not provided)",
    )
    target_format: str = Field(
        default="chatml",
        description="Target format for integration",
    )


class IntegrationPlanResponse(BaseModel):
    """Response model for integration plan details."""

    plan_id: str
    source_id: str
    complexity: str
    target_format: str
    required_transformations: List[str]
    estimated_effort_hours: int
    schema_mapping: Dict[str, str]
    created_date: datetime

    @field_serializer("created_date")
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime to ISO format string."""
        return value.isoformat()

    model_config = {
        "json_schema_extra": {
            "example": {
                "plan_id": "plan_123",
                "source_id": "source_123",
                "complexity": "medium",
                "target_format": "chatml",
                "required_transformations": ["format_conversion", "field_mapping"],
                "estimated_effort_hours": 4,
                "schema_mapping": {
                    "input": "output",
                },
                "created_date": "2025-01-21T00:00:00Z",
            }
        }
    }


class IntegrationPlanListResponse(PaginatedResponse[IntegrationPlanResponse]):
    """Response model for integration plan list."""

    pass

