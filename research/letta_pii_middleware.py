"""
Letta PII Middleware - Wraps Letta tool execution with Foresight PII filtering.

This middleware intercepts all tool calls in Letta and filters content
through Foresight's PII detection before storage.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PIISeverity(Enum):
    """PII severity levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FilterResult:
    """Result of PII filtering operation."""

    original: str
    filtered: str
    severity: PIISeverity
    redacted_count: int
    should_block: bool


class PIIBlockedException(Exception):
    """Raised when PII filtering blocks a tool call."""


class LettaPIIMiddleware:
    """Middleware that filters Letta tool calls through Foresight's PII detection."""

    def __init__(self, pii_filter: Any, config: dict[str, Any] | None = None):
        """
        Initialize middleware with Foresight's PII filter.

        Args:
            pii_filter: Foresight's PIIFilter instance
            config: Optional configuration dict
        """
        self.pii_filter = pii_filter
        self.config = config or {}
        self.max_redaction_ratio = self.config.get("max_redaction_ratio", 0.5)

    async def filter_tool_call(self, tool_name: str, tool_input: dict[str, Any]) -> FilterResult:
        """
        Filter a tool call through PII detection.

        Args:
            tool_name: Name of the tool being called
            tool_input: Input parameters to the tool

        Returns:
            FilterResult with original, filtered content and severity
        """
        # Convert input to string for filtering
        content = self._serialize_input(tool_input)

        # Run through Foresight's PII filter
        try:
            filtered = self.pii_filter.filter_for_storage(content)
        except Exception as e:
            logger.error(f"PII filtering failed for {tool_name}: {e}")
            # On error, block the operation for safety
            return FilterResult(
                original=content, filtered="", severity=PIISeverity.CRITICAL, redacted_count=0, should_block=True
            )

        # Calculate redaction ratio
        redaction_ratio = self._calculate_redaction_ratio(content, filtered or "")

        # Count redactions (simple character-based estimation)
        redacted_count = len(content) - len(filtered or "") if filtered else 0

        # Determine if should block
        should_block = redaction_ratio > self.max_redaction_ratio

        return FilterResult(
            original=content,
            filtered=filtered or "",
            severity=self._determine_severity(redaction_ratio),
            redacted_count=redacted_count,
            should_block=should_block,
        )

    def wrap_tool(self, tool_func: Callable, tool_name: str) -> Callable:
        """
        Wrap a tool function with PII filtering.

        Args:
            tool_func: Original tool function
            tool_name: Name of the tool

        Returns:
            Wrapped function with PII filtering
        """

        async def wrapped(*args, **kwargs):
            # Extract tool input from args/kwargs
            tool_input = self._extract_tool_input(args, kwargs)

            # Filter through PII detection
            result = await self.filter_tool_call(tool_name, tool_input)

            # Block if severity too high
            if result.should_block:
                raise PIIBlockedException(
                    f"Tool {tool_name} blocked due to PII content (redaction ratio: {result.redacted_count})"
                )

            # Replace filtered content in kwargs
            filtered_kwargs = self._reconstruct_kwargs(kwargs, result.filtered)

            # Call original tool with filtered input
            return await tool_func(*args, **filtered_kwargs)

        return wrapped

    def _serialize_input(self, tool_input: dict[str, Any]) -> str:
        """Serialize tool input to string for filtering."""
        return json.dumps(tool_input, default=str)

    def _calculate_redaction_ratio(self, original: str, filtered: str) -> float:
        """Calculate ratio of redacted content."""
        if not original:
            return 0.0
        if not filtered:
            return 1.0  # All content redacted
        redacted_chars = len(original) - len(filtered)
        return redacted_chars / len(original) if original else 0.0

    def _determine_severity(self, redaction_ratio: float) -> PIISeverity:
        """Determine severity based on redaction ratio."""
        if redaction_ratio == 0:
            return PIISeverity.NONE
        if redaction_ratio < 0.1:
            return PIISeverity.LOW
        if redaction_ratio < 0.3:
            return PIISeverity.MEDIUM
        if redaction_ratio < 0.5:
            return PIISeverity.HIGH
        return PIISeverity.CRITICAL

    def _extract_tool_input(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        """Extract tool input from args and kwargs."""
        # Default to kwargs, or first arg if it's a dict
        if kwargs:
            return kwargs
        if args and isinstance(args[0], dict):
            return args[0]
        return {}

    def _reconstruct_kwargs(self, kwargs: dict, filtered: str) -> dict:
        """Reconstruct kwargs with filtered content."""
        try:
            filtered_dict = json.loads(filtered)
            return {**kwargs, **filtered_dict}
        except (json.JSONDecodeError, TypeError):
            # If we can't parse filtered content, return original kwargs
            logger.warning(f"Could not reconstruct kwargs from filtered content: {filtered[:100]}...")
            return kwargs
