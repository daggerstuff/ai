"""Pydantic schemas for the annotation API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class QueueItemCreate(BaseModel):
    """Schema for creating a new queue item."""

    sample_text: str = Field(..., description="The text of the sample to review")
    original_score: float = Field(..., ge=0.0, le=1.0, description="Original clinical validity score")
    per_dimension_scores: dict[str, float] = Field(
        ..., description="Per-dimension scores as a dictionary"
    )
    routing_reason: str = Field(..., description="Reason why this item needs review")

    @field_validator("per_dimension_scores")
    @classmethod
    def validate_per_dimension_scores(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate that all dimension scores are in range [0, 1]."""
        for dim, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Dimension {dim} score {score} must be between 0.0 and 1.0")
        return v


class QueueItemResponse(BaseModel):
    """Schema for a queue item response."""

    id: int
    sample_text: str
    original_score: float
    per_dimension_scores: dict[str, float]
    routing_reason: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class ReviewCreate(BaseModel):
    """Schema for creating a review."""

    reviewer_score: float = Field(..., ge=0.0, le=1.0, description="Reviewer's score")
    notes: str = Field(..., description="Reviewer's notes")
    reviewer_id: str = Field(..., description="Reviewer identifier")


class ReviewResponse(BaseModel):
    """Schema for a review response."""

    id: int
    item_id: int
    reviewer_score: float
    notes: str
    reviewer_id: str
    created_at: datetime

    class Config:
        from_attributes = True


class QueueStats(BaseModel):
    """Schema for queue statistics."""

    pending: int
    reviewed: int
    total: int


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str = "ok"


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    detail: str
