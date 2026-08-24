"""
Pydantic models for API requests and responses.

This module provides request and response models for all API endpoints.
"""

from ai.sourcing.journal.api.models.acquisition import (
    AcquisitionInitiateRequest,
    AcquisitionListResponse,
    AcquisitionResponse,
    AcquisitionUpdateRequest,
)
from ai.sourcing.journal.api.models.common import (
    ErrorResponse,
    MessageResponse,
    PaginatedResponse,
    SuccessResponse,
)
from ai.sourcing.journal.api.models.discovery import (
    DiscoveryInitiateRequest,
    DiscoveryResponse,
    SourceListResponse,
    SourceResponse,
)
from ai.sourcing.journal.api.models.evaluation import (
    EvaluationInitiateRequest,
    EvaluationListResponse,
    EvaluationResponse,
    EvaluationUpdateRequest,
)
from ai.sourcing.journal.api.models.integration import (
    IntegrationInitiateRequest,
    IntegrationPlanListResponse,
    IntegrationPlanResponse,
)
from ai.sourcing.journal.api.models.progress import (
    ProgressMetricsResponse,
    ProgressResponse,
)
from ai.sourcing.journal.api.models.reports import (
    ReportGenerateRequest,
    ReportListResponse,
    ReportResponse,
)
from ai.sourcing.journal.api.models.sessions import (
    CreateSessionRequest,
    SessionListResponse,
    SessionResponse,
    SessionUpdateRequest,
)

__all__ = [
    # Common models
    "ErrorResponse",
    "MessageResponse",
    "PaginatedResponse",
    "SuccessResponse",
    # Session models
    "CreateSessionRequest",
    "SessionResponse",
    "SessionUpdateRequest",
    "SessionListResponse",
    # Discovery models
    "DiscoveryInitiateRequest",
    "DiscoveryResponse",
    "SourceResponse",
    "SourceListResponse",
    # Evaluation models
    "EvaluationInitiateRequest",
    "EvaluationResponse",
    "EvaluationListResponse",
    "EvaluationUpdateRequest",
    # Acquisition models
    "AcquisitionInitiateRequest",
    "AcquisitionResponse",
    "AcquisitionListResponse",
    "AcquisitionUpdateRequest",
    # Integration models
    "IntegrationInitiateRequest",
    "IntegrationPlanResponse",
    "IntegrationPlanListResponse",
    # Progress models
    "ProgressResponse",
    "ProgressMetricsResponse",
    # Report models
    "ReportGenerateRequest",
    "ReportResponse",
    "ReportListResponse",
]
