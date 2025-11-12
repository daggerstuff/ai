"""
Report API routes.

This module provides endpoints for report generation.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from ai.journal_dataset_research.api.dependencies import (
    get_command_handler_service,
    get_current_user,
    require_permission_dependency,
)
from ai.journal_dataset_research.api.models.common import PaginationParams
from ai.journal_dataset_research.api.models.reports import (
    ReportGenerateRequest,
    ReportListResponse,
    ReportResponse,
)
from ai.journal_dataset_research.api.services.command_handler_service import (
    CommandHandlerService,
)

router = APIRouter(prefix="/sessions/{session_id}/reports", tags=["reports"])


@router.post("", response_model=ReportResponse)
async def generate_report(
    session_id: str,
    request: ReportGenerateRequest,
    current_user: dict = Depends(require_permission_dependency("reports:generate")),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> ReportResponse:
    """
    Generate a report for a session.

    Requires: reports:generate permission
    """
    try:
        result = service.generate_report(
            session_id=session_id,
            report_type=request.report_type,
            format=request.format,
            date_range=request.date_range,
        )

        return ReportResponse(
            report_id=result["report_id"],
            session_id=result["session_id"],
            report_type=result["report_type"],
            format=result["format"],
            generated_date=result["generated_date"],
            content=result.get("content"),
            file_path=result.get("file_path"),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate report: {str(e)}",
        )


@router.get("", response_model=ReportListResponse)
async def list_reports(
    session_id: str,
    pagination: PaginationParams = Depends(),
    current_user: Optional[dict] = Depends(get_current_user),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> ReportListResponse:
    """
    List reports for a session.

    Requires: reports:read permission
    """
    try:
        reports = service.list_reports(session_id)
        total = len(reports)

        # Paginate
        start = (pagination.page - 1) * pagination.page_size
        end = start + pagination.page_size
        paginated_reports = reports[start:end]

        # Convert to response models
        report_responses = [
            ReportResponse(
                report_id=report["report_id"],
                session_id=report["session_id"],
                report_type=report["report_type"],
                format=report["format"],
                generated_date=report["generated_date"],
                content=report.get("content"),
                file_path=report.get("file_path"),
            )
            for report in paginated_reports
        ]

        return ReportListResponse.create(
            items=report_responses,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list reports: {str(e)}",
        )


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(
    session_id: str,
    report_id: str,
    current_user: Optional[dict] = Depends(get_current_user),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> ReportResponse:
    """
    Get report details.

    Requires: reports:read permission
    """
    try:
        result = service.get_report(session_id, report_id)

        return ReportResponse(
            report_id=result["report_id"],
            session_id=result["session_id"],
            report_type=result["report_type"],
            format=result["format"],
            generated_date=result["generated_date"],
            content=result.get("content"),
            file_path=result.get("file_path"),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get report: {str(e)}",
        )
