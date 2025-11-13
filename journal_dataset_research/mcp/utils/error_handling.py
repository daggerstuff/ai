"""
Error handling utilities for MCP Server.

This module provides error handling, classification, formatting, and recovery utilities.
"""

import logging
import traceback
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type

from ai.journal_dataset_research.mcp.protocol import (
    JSONRPCErrorCode,
    MCPError,
    MCPErrorCode,
)

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    VALIDATION = "validation"
    EXECUTION = "execution"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NETWORK = "network"
    RESOURCE = "resource"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class MCPErrorHandler:
    """Handles MCP errors with classification, formatting, and recovery."""

    # Error type to category mapping
    ERROR_CATEGORY_MAP: Dict[str, ErrorCategory] = {
        "ValueError": ErrorCategory.VALIDATION,
        "TypeError": ErrorCategory.VALIDATION,
        "KeyError": ErrorCategory.VALIDATION,
        "AttributeError": ErrorCategory.VALIDATION,
        "TimeoutError": ErrorCategory.TIMEOUT,
        "asyncio.TimeoutError": ErrorCategory.TIMEOUT,
        "PermissionError": ErrorCategory.AUTHORIZATION,
        "UnauthorizedError": ErrorCategory.AUTHENTICATION,
        "ConnectionError": ErrorCategory.NETWORK,
        "NetworkError": ErrorCategory.NETWORK,
        "HTTPError": ErrorCategory.NETWORK,
        "FileNotFoundError": ErrorCategory.RESOURCE,
        "IOError": ErrorCategory.RESOURCE,
        "OSError": ErrorCategory.RESOURCE,
    }

    # Error category to severity mapping
    CATEGORY_SEVERITY_MAP: Dict[ErrorCategory, ErrorSeverity] = {
        ErrorCategory.VALIDATION: ErrorSeverity.LOW,
        ErrorCategory.TIMEOUT: ErrorSeverity.MEDIUM,
        ErrorCategory.AUTHENTICATION: ErrorSeverity.HIGH,
        ErrorCategory.AUTHORIZATION: ErrorSeverity.HIGH,
        ErrorCategory.NETWORK: ErrorSeverity.MEDIUM,
        ErrorCategory.RESOURCE: ErrorSeverity.MEDIUM,
        ErrorCategory.EXECUTION: ErrorSeverity.HIGH,
        ErrorCategory.INTERNAL: ErrorSeverity.CRITICAL,
        ErrorCategory.UNKNOWN: ErrorSeverity.MEDIUM,
    }

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

        # Resource errors
        if error_type in ("FileNotFoundError", "IOError", "OSError"):
            return (
                MCPErrorCode.RESOURCE_NOT_FOUND,
                f"Resource error: {str(error)}",
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
    def get_error_category(error: Exception) -> ErrorCategory:
        """
        Get error category for an exception.

        Args:
            error: Exception to categorize

        Returns:
            Error category
        """
        error_type = type(error).__name__
        return MCPErrorHandler.ERROR_CATEGORY_MAP.get(
            error_type, ErrorCategory.UNKNOWN
        )

    @staticmethod
    def get_error_severity(error: Exception) -> ErrorSeverity:
        """
        Get error severity for an exception.

        Args:
            error: Exception to assess

        Returns:
            Error severity
        """
        category = MCPErrorHandler.get_error_category(error)
        return MCPErrorHandler.CATEGORY_SEVERITY_MAP.get(
            category, ErrorSeverity.MEDIUM
        )

    @staticmethod
    def is_recoverable(error: Exception) -> bool:
        """
        Determine if an error is recoverable.

        Args:
            error: Exception to assess

        Returns:
            True if error is recoverable, False otherwise
        """
        category = MCPErrorHandler.get_error_category(error)
        severity = MCPErrorHandler.get_error_severity(error)

        # Recoverable errors: validation, timeout, network (with retries)
        recoverable_categories = {
            ErrorCategory.VALIDATION,
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK,
        }

        # Low and medium severity errors are generally recoverable
        recoverable_severities = {ErrorSeverity.LOW, ErrorSeverity.MEDIUM}

        return category in recoverable_categories or severity in recoverable_severities

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
        level: Optional[int] = None,
    ) -> None:
        """
        Log error with context and severity-based level.

        Args:
            error: Exception to log
            context: Optional context information
            level: Optional logging level (auto-determined if not provided)
        """
        code, message, data = MCPErrorHandler.classify_error(error)
        category = MCPErrorHandler.get_error_category(error)
        severity = MCPErrorHandler.get_error_severity(error)

        # Determine log level based on severity if not provided
        if level is None:
            level_map = {
                ErrorSeverity.LOW: logging.WARNING,
                ErrorSeverity.MEDIUM: logging.ERROR,
                ErrorSeverity.HIGH: logging.ERROR,
                ErrorSeverity.CRITICAL: logging.CRITICAL,
            }
            level = level_map.get(severity, logging.ERROR)

        log_data = {
            "error_code": code,
            "error_message": message,
            "error_type": type(error).__name__,
            "error_category": category.value,
            "error_severity": severity.value,
            "recoverable": MCPErrorHandler.is_recoverable(error),
        }

        if context:
            log_data.update(context)

        if data:
            log_data["error_data"] = data

        logger.log(
            level,
            f"MCP Error [{code}] [{severity.value}]: {message}",
            extra=log_data,
        )

        # Log traceback at DEBUG level for all errors
        logger.debug("Error traceback:", exc_info=error)

        # Log additional context for critical errors
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(
                f"Critical error detected: {error_type} - {message}",
                extra=log_data,
            )

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

    @staticmethod
    def with_error_recovery(
        func: Callable,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        recoverable_errors: Optional[tuple[Type[Exception], ...]] = None,
    ) -> Callable:
        """
        Decorator for automatic error recovery with retries.

        Args:
            func: Function to wrap
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            recoverable_errors: Tuple of exception types that are recoverable
                (defaults to validation, timeout, network errors)

        Returns:
            Wrapped function with error recovery
        """
        import asyncio
        import functools

        if recoverable_errors is None:
            recoverable_errors = (
                ValueError,
                TypeError,
                TimeoutError,
                ConnectionError,
            )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Optional[Exception] = None

            for attempt in range(max_retries):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    # Check if error is recoverable
                    is_recoverable = (
                        isinstance(e, recoverable_errors)
                        or MCPErrorHandler.is_recoverable(e)
                    )

                    if not is_recoverable or attempt == max_retries - 1:
                        # Not recoverable or last attempt
                        raise

                    # Log retry attempt
                    logger.warning(
                        f"Error in {func.__name__} (attempt {attempt + 1}/{max_retries}): {str(e)}. Retrying..."
                    )

                    # Wait before retry
                    if asyncio.iscoroutinefunction(func):
                        await asyncio.sleep(retry_delay * (attempt + 1))
                    else:
                        import time
                        time.sleep(retry_delay * (attempt + 1))

            # Should not reach here, but just in case
            if last_error:
                raise last_error
            raise RuntimeError("Unexpected error in error recovery wrapper")

        return wrapper

