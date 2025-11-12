"""
Report generation tools for MCP Server.

This module provides tools for generating and managing reports through the MCP protocol.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ai.journal_dataset_research.api.services.command_handler_service import (
    CommandHandlerService,
)
from ai.journal_dataset_research.mcp.protocol import MCPError, MCPErrorCode
from ai.journal_dataset_research.mcp.tools.base import MCPTool

logger = logging.getLogger(__name__)


class GenerateReportTool(MCPTool):
    """Tool for generating reports for research sessions."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize GenerateReportTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="generate_report",
            description="Generate a report for a research session. This tool creates comprehensive reports including session details, progress metrics, sources, evaluations, acquired datasets, and integration plans. Reports can be generated in different formats (json, markdown) and for different report types (session_report, weekly_report, summary_report).",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "report_type": {
                        "type": "string",
                        "enum": ["session_report", "weekly_report", "summary_report"],
                        "description": "Type of report to generate (default: session_report)",
                        "default": "session_report",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "markdown", "pdf"],
                        "description": "Report format (default: json)",
                        "default": "json",
                    },
                    "date_range": {
                        "type": "object",
                        "description": "Optional date range for report filtering",
                        "properties": {
                            "start_date": {
                                "type": "string",
                                "format": "date-time",
                                "description": "Start date in ISO format",
                            },
                            "end_date": {
                                "type": "string",
                                "format": "date-time",
                                "description": "End date in ISO format",
                            },
                        },
                    },
                },
                "required": ["session_id"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute generate_report tool.

        Args:
            params: Tool parameters

        Returns:
            Report generation result with report content
        """
        try:
            session_id = params.get("session_id")
            report_type = params.get("report_type", "session_report")
            format_type = params.get("format", "json")
            date_range = params.get("date_range")

            # Validate parameters
            if not session_id:
                raise ValueError("session_id is required")
            if report_type not in ["session_report", "weekly_report", "summary_report"]:
                raise ValueError(
                    "report_type must be one of: session_report, weekly_report, summary_report"
                )
            if format_type not in ["json", "markdown", "pdf"]:
                raise ValueError("format must be one of: json, markdown, pdf")

            # Generate report
            result = self.service.generate_report(
                session_id=session_id,
                report_type=report_type,
                format=format_type,
                date_range=date_range,
            )

            # Convert datetime to ISO string for JSON serialization
            generated_date = result.get("generated_date")
            if isinstance(generated_date, datetime):
                generated_date = generated_date.isoformat()

            return {
                "report_id": result.get("report_id"),
                "session_id": result.get("session_id"),
                "report_type": result.get("report_type"),
                "format": result.get("format"),
                "generated_date": generated_date,
                "content": result.get("content"),
                "file_path": result.get("file_path"),
                "status": "completed",
            }
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error generating report: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to generate report: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e


class GetReportTool(MCPTool):
    """Tool for getting a specific report."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize GetReportTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="get_report",
            description="Get details of a specific report by report ID. This tool retrieves a previously generated report including its content, format, and metadata.",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "report_id": {
                        "type": "string",
                        "description": "Report ID (format: report_{session_id}_{timestamp})",
                    },
                },
                "required": ["session_id", "report_id"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute get_report tool.

        Args:
            params: Tool parameters

        Returns:
            Report details
        """
        try:
            session_id = params.get("session_id")
            report_id = params.get("report_id")

            # Validate parameters
            if not session_id:
                raise ValueError("session_id is required")
            if not report_id:
                raise ValueError("report_id is required")

            # Get report
            result = self.service.get_report(session_id, report_id)

            # Convert datetime to ISO string for JSON serialization
            generated_date = result.get("generated_date")
            if isinstance(generated_date, datetime):
                generated_date = generated_date.isoformat()

            return {
                "report_id": result.get("report_id"),
                "session_id": result.get("session_id"),
                "report_type": result.get("report_type"),
                "format": result.get("format"),
                "generated_date": generated_date,
                "content": result.get("content"),
                "file_path": result.get("file_path"),
            }
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error getting report: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to get report: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e


class ListReportsTool(MCPTool):
    """Tool for listing all reports for a session."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize ListReportsTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="list_reports",
            description="List all reports for a research session. Returns a list of report metadata including report IDs, types, formats, and generation dates.",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                },
                "required": ["session_id"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute list_reports tool.

        Args:
            params: Tool parameters

        Returns:
            List of reports with metadata
        """
        try:
            session_id = params.get("session_id")

            # Validate parameters
            if not session_id:
                raise ValueError("session_id is required")

            # List reports
            reports_list = self.service.list_reports(session_id)

            # Convert datetime objects to ISO strings
            reports_data = []
            for report in reports_list:
                report_dict = dict(report)
                generated_date = report_dict.get("generated_date")
                if isinstance(generated_date, datetime):
                    report_dict["generated_date"] = generated_date.isoformat()
                reports_data.append(report_dict)

            return {
                "session_id": session_id,
                "reports": reports_data,
                "count": len(reports_data),
            }
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error listing reports: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to list reports: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e

