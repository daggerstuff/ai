"""
Authentication middleware for the API server.

This module provides JWT token validation and authentication middleware.
"""

import logging
from typing import Callable

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ai.journal_dataset_research.api.auth.jwt import get_user_from_token
from ai.journal_dataset_research.api.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for JWT token validation."""

    # Public endpoints that don't require authentication
    PUBLIC_ENDPOINTS = [
        "/health",
        "/",
        "/api/docs",
        "/api/redoc",
        "/api/openapi.json",
    ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and validate authentication."""
        # Skip authentication if disabled
        if not settings.auth_enabled:
            # Set default user for development
            request.state.user = {
                "user_id": "dev-user",
                "email": "dev@example.com",
                "role": "admin",
                "permissions": ["*"],
            }
            return await call_next(request)

        # Skip authentication for public endpoints
        if request.url.path in self.PUBLIC_ENDPOINTS:
            return await call_next(request)

        # Skip authentication for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Get authorization header
        authorization = request.headers.get("Authorization")
        if not authorization:
            # Don't raise error here, let endpoints handle authentication
            # Some endpoints may be public or use optional authentication
            return await call_next(request)

        # Validate token format
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Extract token
        token = authorization.split(" ")[1]

        # Validate token and get user
        try:
            user = get_user_from_token(token)
            request.state.user = user
            request.state.token = token
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Continue to next middleware/handler
        response = await call_next(request)
        return response

