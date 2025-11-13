"""
Rate limiting utilities for MCP Server.

This module provides rate limiting functionality using a token bucket algorithm
to prevent abuse and ensure fair resource usage.
"""

import asyncio
import time
from collections import defaultdict
from typing import Dict, Optional

from ai.journal_dataset_research.mcp.config import RateLimitConfig
from ai.journal_dataset_research.mcp.protocol import MCPError, MCPErrorCode

# Global rate limiters per identifier (e.g., user_id, IP address)
_rate_limiters: Dict[str, "RateLimiter"] = {}
_rate_limiters_lock = asyncio.Lock()


class RateLimiter:
    """
    Token bucket rate limiter implementation.

    Uses a token bucket algorithm to enforce rate limits:
    - Tokens are added at a constant rate (requests_per_minute / 60)
    - Each request consumes one token
    - Burst size limits the maximum number of tokens in the bucket
    - Separate buckets for per-minute and per-hour limits
    """

    def __init__(self, config: RateLimitConfig) -> None:
        """
        Initialize rate limiter.

        Args:
            config: Rate limiting configuration
        """
        self.config = config

        # Per-minute bucket
        self.minute_tokens: float = float(config.burst_size)
        self.minute_last_refill: float = time.time()
        self.minute_refill_rate: float = config.requests_per_minute / 60.0  # tokens per second

        # Per-hour bucket
        self.hour_tokens: float = float(config.burst_size)
        self.hour_last_refill: float = time.time()
        self.hour_refill_rate: float = config.requests_per_hour / 3600.0  # tokens per second

        # Request counters for statistics
        self.total_requests: int = 0
        self.blocked_requests: int = 0

    def _refill_tokens(self, current_time: float) -> None:
        """
        Refill tokens based on elapsed time.

        Args:
            current_time: Current timestamp
        """
        # Refill minute bucket
        elapsed_minute = current_time - self.minute_last_refill
        if elapsed_minute > 0:
            tokens_to_add = elapsed_minute * self.minute_refill_rate
            self.minute_tokens = min(
                self.config.burst_size,
                self.minute_tokens + tokens_to_add
            )
            self.minute_last_refill = current_time

        # Refill hour bucket
        elapsed_hour = current_time - self.hour_last_refill
        if elapsed_hour > 0:
            tokens_to_add = elapsed_hour * self.hour_refill_rate
            self.hour_tokens = min(
                self.config.burst_size,
                self.hour_tokens + tokens_to_add
            )
            self.hour_last_refill = current_time

    async def check_rate_limit(self) -> bool:
        """
        Check if request is within rate limit.

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        if not self.config.enabled:
            return True

        current_time = time.time()
        self._refill_tokens(current_time)

        # Check both minute and hour limits
        if self.minute_tokens >= 1.0 and self.hour_tokens >= 1.0:
            # Consume tokens
            self.minute_tokens -= 1.0
            self.hour_tokens -= 1.0
            self.total_requests += 1
            return True

        # Rate limit exceeded
        self.blocked_requests += 1
        return False

    def get_retry_after(self) -> int:
        """
        Get retry-after time in seconds.

        Returns:
            Seconds until next token is available
        """
        current_time = time.time()
        self._refill_tokens(current_time)

        # Calculate time until next token for both buckets
        minute_wait = 0
        if self.minute_tokens < 1.0:
            minute_wait = (1.0 - self.minute_tokens) / self.minute_refill_rate

        hour_wait = 0
        if self.hour_tokens < 1.0:
            hour_wait = (1.0 - self.hour_tokens) / self.hour_refill_rate

        # Return the maximum wait time (most restrictive)
        return int(max(minute_wait, hour_wait)) + 1

    def get_stats(self) -> Dict[str, any]:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "minute_tokens": self.minute_tokens,
            "hour_tokens": self.hour_tokens,
            "minute_refill_rate": self.minute_refill_rate,
            "hour_refill_rate": self.hour_refill_rate,
        }


class RateLimitManager:
    """
    Manages rate limiters for multiple identifiers (users, IPs, etc.).
    """

    def __init__(self, config: RateLimitConfig) -> None:
        """
        Initialize rate limit manager.

        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self.limiters: Dict[str, RateLimiter] = {}

    def get_limiter(self, identifier: str) -> RateLimiter:
        """
        Get or create rate limiter for identifier.

        Args:
            identifier: Unique identifier (e.g., user_id, IP address)

        Returns:
            RateLimiter instance for the identifier
        """
        if identifier not in self.limiters:
            self.limiters[identifier] = RateLimiter(self.config)
        return self.limiters[identifier]

    async def check_rate_limit(self, identifier: str) -> bool:
        """
        Check rate limit for identifier.

        Args:
            identifier: Unique identifier

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        limiter = self.get_limiter(identifier)
        return await limiter.check_rate_limit()

    def get_retry_after(self, identifier: str) -> int:
        """
        Get retry-after time for identifier.

        Args:
            identifier: Unique identifier

        Returns:
            Seconds until next token is available
        """
        limiter = self.get_limiter(identifier)
        return limiter.get_retry_after()

    def get_stats(self, identifier: Optional[str] = None) -> Dict[str, any]:
        """
        Get rate limiter statistics.

        Args:
            identifier: Optional identifier to get stats for specific limiter

        Returns:
            Statistics dictionary
        """
        if identifier:
            limiter = self.get_limiter(identifier)
            return limiter.get_stats()

        # Return aggregate stats
        total_requests = sum(limiter.total_requests for limiter in self.limiters.values())
        total_blocked = sum(limiter.blocked_requests for limiter in self.limiters.values())
        return {
            "total_limiters": len(self.limiters),
            "total_requests": total_requests,
            "total_blocked": total_blocked,
        }

    def cleanup_old_limiters(self, max_age_seconds: int = 3600) -> None:
        """
        Clean up old rate limiters that haven't been used recently.

        Args:
            max_age_seconds: Maximum age in seconds before cleanup
        """
        current_time = time.time()
        to_remove = []

        for identifier, limiter in self.limiters.items():
            # Check if limiter hasn't been used recently
            last_activity = max(limiter.minute_last_refill, limiter.hour_last_refill)
            if current_time - last_activity > max_age_seconds:
                to_remove.append(identifier)

        for identifier in to_remove:
            del self.limiters[identifier]


def get_rate_limit_identifier(request: any, user: Optional[Dict[str, any]] = None) -> str:
    """
    Get rate limit identifier from request and user.

    Args:
        request: MCP request
        user: Optional authenticated user information

    Returns:
        Identifier string for rate limiting
    """
    # Prefer user_id if available
    if user and user.get("user_id"):
        return f"user:{user['user_id']}"

    # Fall back to API key identifier if available
    if user and user.get("user_id", "").startswith("api-key-"):
        return user["user_id"]

    # Extract from request if possible
    if hasattr(request, "params") and request.params:
        # Try to get identifier from params
        identifier = request.params.get("client_id") or request.params.get("identifier")
        if identifier:
            return f"client:{identifier}"

    # Default: use anonymous identifier
    return "anonymous"


async def check_rate_limit(
    request: any,
    rate_limit_manager: RateLimitManager,
    user: Optional[Dict[str, any]] = None,
) -> None:
    """
    Check rate limit for request and raise error if exceeded.

    Args:
        request: MCP request
        rate_limit_manager: Rate limit manager instance
        user: Optional authenticated user information

    Raises:
        MCPError: If rate limit is exceeded
    """
    identifier = get_rate_limit_identifier(request, user)
    is_allowed = await rate_limit_manager.check_rate_limit(identifier)

    if not is_allowed:
        retry_after = rate_limit_manager.get_retry_after(identifier)
        raise MCPError(
            MCPErrorCode.RATE_LIMIT_EXCEEDED,
            f"Rate limit exceeded. Please retry after {retry_after} seconds.",
            data={
                "retry_after": retry_after,
                "identifier": identifier,
            },
        )

