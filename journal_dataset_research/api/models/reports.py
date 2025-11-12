"""
Pydantic models for report endpoints.

This module provides request and response models for report generation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_serializer

from ai.journal_dataset_research.api.models.common import PaginatedResponse


class ReportGenerateRequest(BaseModel):
    """Request model for generating a report."""

    report_type: str = Field(
        default="session_report",
        description="Report type: session_report, weekly_report, summary_report",
    )
    format: str = Field(
        default="json",
        description="Report format: json, markdown, pdf",
    )
    date_range: Optional[Dict[str, str]] = Field(
        default=None,
        description="Date range for report (start_date, end_date) as ISO strings",
    )


class ReportResponse(BaseModel):
    """Response model for report details."""

    report_id: str
    session_id: str
    report_type: str
    format: str
    generated_date: datetime
    content: Optional[Any] = None
    file_path: Optional[str] = None

    @field_serializer("generated_date")
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime to ISO format string."""
        return value.isoformat()

    model_config = {
        "json_schema_extra": {
            "example": {
                "report_id": "report_123",
                "session_id": "session_123",
                "report_type": "session_report",
                "format": "json",
                "generated_date": "2025-01-21T00:00:00Z",
                "content": None,
                "file_path": "/path/to/report.json",
            }
        }
    }


class ReportListResponse(PaginatedResponse[ReportResponse]):
    """Response model for report list."""

    pass

