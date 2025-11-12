"""
Pydantic models for evaluation endpoints.

This module provides request and response models for dataset evaluation.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_serializer

from ai.journal_dataset_research.api.models.common import PaginatedResponse


class EvaluationInitiateRequest(BaseModel):
    """Request model for initiating evaluation."""

    source_ids: Optional[List[str]] = Field(
        default=None,
        description="Source IDs to evaluate (all sources if not provided)",
    )


class EvaluationUpdateRequest(BaseModel):
    """Request model for updating evaluation scores."""

    therapeutic_relevance: Optional[int] = Field(
        default=None, ge=1, le=10, description="Therapeutic relevance score (1-10)"
    )
    data_structure_quality: Optional[int] = Field(
        default=None, ge=1, le=10, description="Data structure quality score (1-10)"
    )
    training_integration: Optional[int] = Field(
        default=None, ge=1, le=10, description="Training integration score (1-10)"
    )
    ethical_accessibility: Optional[int] = Field(
        default=None, ge=1, le=10, description="Ethical accessibility score (1-10)"
    )
    priority_tier: Optional[str] = Field(
        default=None,
        description="Priority tier: high, medium, low",
    )


class EvaluationResponse(BaseModel):
    """Response model for evaluation details."""

    evaluation_id: str
    source_id: str
    therapeutic_relevance: int
    data_structure_quality: int
    training_integration: int
    ethical_accessibility: int
    overall_score: float
    priority_tier: str
    evaluation_date: datetime
    evaluator: str = "system"

    @field_serializer("evaluation_date")
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime to ISO format string."""
        return value.isoformat()

    model_config = {
        "json_schema_extra": {
            "example": {
                "evaluation_id": "eval_123",
                "source_id": "source_123",
                "therapeutic_relevance": 8,
                "data_structure_quality": 7,
                "training_integration": 6,
                "ethical_accessibility": 9,
                "overall_score": 7.5,
                "priority_tier": "high",
                "evaluation_date": "2025-01-21T00:00:00Z",
                "evaluator": "system",
            }
        }
    }


class EvaluationListResponse(PaginatedResponse[EvaluationResponse]):
    """Response model for evaluation list."""

    pass

