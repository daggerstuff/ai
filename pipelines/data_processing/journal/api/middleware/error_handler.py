"""
Error handling middleware.

This module provides enhanced error handling for API endpoints.
"""

import logging
import traceback
from collections.abc import Callable

from fastapi import Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ai.pipelines.data_processing.journal.api.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Enhanced error handling middleware."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and handle errors."""
        try:
            return await call_next(request)
        except RequestValidationError as e:
            # Handle validation errors
            logger.warning(f"Validation error: {e.errors()}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "detail": "Validation error",
                    "errors": e.errors(),
                    "body": e.body,
                },
            )
        except Exception as e:
            # Handle unexpected errors
            logger.error(
                f"Unhandled exception: {e!s}\n{traceback.format_exc()}",
                exc_info=True,
            )

            # Return error response
            error_detail = {
                "detail": "Internal server error",
                "type": type(e).__name__,
            }

            # Include error message in development mode
            if settings.debug:
                error_detail["message"] = str(e)
                error_detail["traceback"] = traceback.format_exc().split("\n")

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_detail,
            )
