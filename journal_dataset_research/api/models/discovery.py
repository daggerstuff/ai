"""
Pydantic models for discovery endpoints.

This module provides request and response models for source discovery.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_serializer

from ai.journal_dataset_research.api.models.common import PaginatedResponse


class DiscoveryInitiateRequest(BaseModel):
    """Request model for initiating discovery."""

    keywords: List[str] = Field(
        default_factory=list,
        description="Search keywords for discovery",
    )
    sources: List[str] = Field(
        default=["pubmed", "doaj"],
        description="Target sources for discovery",
    )


class SourceResponse(BaseModel):
    """Response model for source details."""

    source_id: str
    title: str
    authors: List[str]
    publication_date: datetime
    source_type: str
    url: str
    doi: Optional[str] = None
    abstract: str = ""
    keywords: List[str] = []
    open_access: bool = False
    data_availability: str = "unknown"
    discovery_date: datetime
    discovery_method: str = ""

    @field_serializer("publication_date", "discovery_date")
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime to ISO format string."""
        return value.isoformat()

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_id": "source_123",
                "title": "Mental Health Dataset",
                "authors": ["John Doe", "Jane Smith"],
                "publication_date": "2024-01-01T00:00:00Z",
                "source_type": "journal",
                "url": "https://example.com/dataset",
                "doi": "10.1234/example",
                "abstract": "A dataset of mental health conversations",
                "keywords": ["mental health", "therapy", "dataset"],
                "open_access": True,
                "data_availability": "available",
                "discovery_date": "2025-01-21T00:00:00Z",
                "discovery_method": "pubmed_search",
            }
        }
    }


class DiscoveryResponse(BaseModel):
    """Response model for discovery operation."""

    session_id: str
    sources: List[SourceResponse]
    total_sources: int
    discovery_status: str = "completed"

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "session_123",
                "sources": [],
                "total_sources": 10,
                "discovery_status": "completed",
            }
        }
    }


class SourceListResponse(PaginatedResponse[SourceResponse]):
    """Response model for source list."""

    pass
