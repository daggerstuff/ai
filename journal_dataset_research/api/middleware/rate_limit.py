"""
Rate limiting middleware.

This module provides rate limiting for API endpoints.
"""

import logging
import time
from collections import defaultdict
from typing import Callable, Dict, Tuple

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from ai.journal_dataset_research.api.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using token bucket algorithm."""

    # Rate limit storage: {client_id: (tokens, last_update)}
    rate_limits: Dict[str, Tuple[float, float]] = defaultdict(
        lambda: (settings.rate_limit_per_minute, time.time())
    )

    # Public endpoints that don't require rate limiting
    PUBLIC_ENDPOINTS = [
        "/health",
        "/",
        "/api/docs",
        "/api/redoc",
        "/api/openapi.json",
    ]

    def __init__(self, app):
        """Initialize rate limiting middleware."""
        super().__init__(app)
        self.rate_limit_per_minute = settings.rate_limit_per_minute
        self.rate_limit_per_hour = settings.rate_limit_per_hour

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Try to get user ID from request state (if authenticated)
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user.get('user_id', 'unknown')}"

        # Fall back to IP address
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"

    def _check_rate_limit(self, client_id: str) -> Tuple[bool, Dict[str, str]]:
        """Check if client has exceeded rate limit."""
        if not settings.rate_limit_enabled:
            return True, {}

        now = time.time()
        tokens, last_update = self.rate_limits[client_id]

        # Refill tokens based on time elapsed
        time_elapsed = now - last_update
        tokens_to_add = (time_elapsed / 60.0) * self.rate_limit_per_minute
        tokens = min(tokens + tokens_to_add, self.rate_limit_per_minute)

        # Update last update time
        self.rate_limits[client_id] = (tokens, now)

        # Check if client has tokens available
        if tokens >= 1.0:
            # Consume one token
            self.rate_limits[client_id] = (tokens - 1.0, now)
            return True, {}

        # Rate limit exceeded
        remaining_tokens = max(0, tokens)
        reset_time = 60.0 - (time_elapsed % 60.0)

        headers = {
            "X-RateLimit-Limit": str(self.rate_limit_per_minute),
            "X-RateLimit-Remaining": str(int(remaining_tokens)),
            "X-RateLimit-Reset": str(int(now + reset_time)),
        }

        return False, headers

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and check rate limits."""
        # Skip rate limiting for public endpoints
        if request.url.path in self.PUBLIC_ENDPOINTS:
            return await call_next(request)

        # Skip rate limiting for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Get client identifier
        client_id = self._get_client_id(request)

        # Check rate limit
        allowed, headers = self._check_rate_limit(client_id)

        if not allowed:
            logger.warning(f"Rate limit exceeded for client {client_id}")
            response = Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
            )
            for key, value in headers.items():
                response.headers[key] = value
            return response

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value

        return response

