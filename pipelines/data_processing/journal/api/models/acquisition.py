"""
Pydantic models for acquisition endpoints.

This module provides request and response models for dataset acquisition.
"""

from datetime import datetime

from pydantic import BaseModel, Field, field_serializer

from ai.pipelines.data_processing.journal.api.models.common import PaginatedResponse


class AcquisitionInitiateRequest(BaseModel):
    """Request model for initiating acquisition."""

    source_ids: list[str] | None = Field(
        default=None,
        description="Source IDs to acquire (all sources if not provided)",
    )


class AcquisitionUpdateRequest(BaseModel):
    """Request model for updating acquisition status."""

    status: str = Field(
        description="Acquisition status: pending, approved, in-progress, completed, failed"
    )


class AcquisitionResponse(BaseModel):
    """Response model for acquisition details."""

    acquisition_id: str
    source_id: str
    status: str
    download_progress: float | None = Field(
        default=None, ge=0, le=100, description="Download progress percentage"
    )
    file_path: str | None = None
    file_size: float | None = Field(default=None, description="File size in MB")
    acquired_date: datetime | None = None

    @field_serializer("acquired_date")
    def serialize_datetime(self, value: datetime | None) -> str | None:
        """Serialize datetime to ISO format string."""
        return value.isoformat() if value else None

    model_config = {
        "json_schema_extra": {
            "example": {
                "acquisition_id": "acq_123",
                "source_id": "source_123",
                "status": "completed",
                "download_progress": 100.0,
                "file_path": "/path/to/dataset.json",
                "file_size": 10.5,
                "acquired_date": "2025-01-21T00:00:00Z",
            }
        }
    }


class AcquisitionListResponse(PaginatedResponse[AcquisitionResponse]):
    """Response model for acquisition list."""

    pass
