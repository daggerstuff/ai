"""
Error handling utilities for MCP Server.

This module provides error handling, classification, and formatting utilities.
"""

import logging
import traceback
from typing import Any, Dict, Optional, Type

from ai.journal_dataset_research.mcp.protocol import (
    JSONRPCErrorCode,
    MCPError,
    MCPErrorCode,
)

logger = logging.getLogger(__name__)


class MCPErrorHandler:
    """Handles MCP errors with classification and formatting."""

    @staticmethod
    def classify_error(error: Exception) -> tuple[int, str, Optional[Any]]:
        """
        Classify error and return error code, message, and optional data.

        Args:
            error: Exception to classify

        Returns:
            Tuple of (error_code, message, optional_data)
        """
        # Handle MCPError directly
        if isinstance(error, MCPError):
            return (error.code, error.message, error.data)

        # Handle common Python exceptions
        error_type = type(error).__name__

        # Validation errors
        if error_type in ("ValueError", "TypeError", "KeyError", "AttributeError"):
            return (
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Validation error: {str(error)}",
                {"error_type": error_type},
            )

        # Timeout errors
        if error_type in ("TimeoutError", "asyncio.TimeoutError"):
            return (
                MCPErrorCode.TOOL_TIMEOUT,
                f"Operation timed out: {str(error)}",
                {"error_type": error_type},
            )

        # Authentication/Authorization errors
        if error_type in ("PermissionError", "UnauthorizedError"):
            return (
                MCPErrorCode.AUTHENTICATION_ERROR,
                f"Authentication error: {str(error)}",
                {"error_type": error_type},
            )

        # Network/Connection errors
        if error_type in ("ConnectionError", "NetworkError", "HTTPError"):
            return (
                JSONRPCErrorCode.INTERNAL_ERROR,
                f"Network error: {str(error)}",
                {"error_type": error_type},
            )

        # Generic internal error
        return (
            JSONRPCErrorCode.INTERNAL_ERROR,
            f"Internal error: {str(error)}",
            {
                "error_type": error_type,
                "traceback": traceback.format_exc(),
            },
        )

    @staticmethod
    def format_error_response(
        error: Exception,
        request_id: Optional[Any] = None,
        include_traceback: bool = False,
    ) -> Dict[str, Any]:
        """
        Format error as MCP error response.

        Args:
            error: Exception to format
            request_id: Optional request ID
            include_traceback: Whether to include traceback in production

        Returns:
            Error response dictionary
        """
        code, message, data = MCPErrorHandler.classify_error(error)

        error_dict: Dict[str, Any] = {
            "code": code,
            "message": message,
        }

        if data:
            # Only include traceback in debug mode or if explicitly requested
            if include_traceback and "traceback" in data:
                error_dict["data"] = data
            elif data and "traceback" not in data:
                error_dict["data"] = data
            elif include_traceback:
                error_dict["data"] = {
                    **data,
                    "traceback": traceback.format_exc(),
                }

        response: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "error": error_dict,
        }

        if request_id is not None:
            response["id"] = request_id

        return response

    @staticmethod
    def log_error(
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        level: int = logging.ERROR,
    ) -> None:
        """
        Log error with context.

        Args:
            error: Exception to log
            context: Optional context information
            level: Logging level
        """
        code, message, data = MCPErrorHandler.classify_error(error)

        log_data = {
            "error_code": code,
            "error_message": message,
            "error_type": type(error).__name__,
        }

        if context:
            log_data.update(context)

        if data:
            log_data["error_data"] = data

        logger.log(
            level,
            f"MCP Error [{code}]: {message}",
            extra=log_data,
        )

        # Log traceback at DEBUG level
        logger.debug("Error traceback:", exc_info=error)

    @staticmethod
    def handle_error(
        error: Exception,
        request_id: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        include_traceback: bool = False,
    ) -> Dict[str, Any]:
        """
        Handle error: log and format response.

        Args:
            error: Exception to handle
            request_id: Optional request ID
            context: Optional context information
            include_traceback: Whether to include traceback

        Returns:
            Error response dictionary
        """
        # Log the error
        MCPErrorHandler.log_error(error, context)

        # Format error response
        return MCPErrorHandler.format_error_response(
            error,
            request_id,
            include_traceback,
        )

