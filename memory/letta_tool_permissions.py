"""
Letta Tool Permission System - Crisis-aware tool execution.

This module implements fine-grained tool permissions for Letta agents,
integrating with Foresight's crisis detection and PII filtering.

Permission levels:
- read-only: Only read operations (Read, Grep, Glob, web_search)
- therapeutic: Read + therapeutic-specific tools (reflect, consolidate)
- full: All tools including write operations
- whisper: Background processing only, no tool execution
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger("letta_permissions")


class PermissionLevel(StrEnum):
    """Tool permission levels."""

    READ_ONLY = "read-only"
    THERAPEUTIC = "therapeutic"
    FULL = "full"
    WHISPER = "whisper"


class CrisisContext(StrEnum):
    """Crisis context levels for permission decisions."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ToolDefinition:
    """Definition of a client-side tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    permission_level: PermissionLevel
    allowed_in_crisis: bool = False
    requires_user_consent: bool = False
    pii_filter_enabled: bool = True
    risk_level: str = "low"  # low, medium, high
    consent_message: str | None = None  # Message for consent request


@dataclass
class PermissionResult:
    """Result of a permission check."""

    allowed: bool
    reason: str
    filtered_params: dict[str, Any] | None = None
    requires_consent: bool = False
    consent_message: str | None = None


class LettaToolRegistry:
    """
    Registry of available tools with their permission configurations.

    This registry implements Letta's canUseTool handler pattern,
    integrating with Foresight's crisis detection system.
    """

    def __init__(self, permission_level: PermissionLevel = PermissionLevel.THERAPEUTIC):
        """
        Initialize tool registry.

        Args:
            permission_level: Current permission level
        """
        self.permission_level = permission_level
        self._tools: dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register default tool set."""
        # Read-only tools (available in all modes except whisper)
        self.register_tool(
            ToolDefinition(
                name="Read",
                description="Read file contents",
                parameters={"file_path": "string"},
                permission_level=PermissionLevel.READ_ONLY,
                allowed_in_crisis=True,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="Grep",
                description="Search file contents",
                parameters={"pattern": "string", "path": "string"},
                permission_level=PermissionLevel.READ_ONLY,
                allowed_in_crisis=True,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="Glob",
                description="Find files by pattern",
                parameters={"pattern": "string"},
                permission_level=PermissionLevel.READ_ONLY,
                allowed_in_crisis=True,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="web_search",
                description="Search the web",
                parameters={"query": "string"},
                permission_level=PermissionLevel.READ_ONLY,
                allowed_in_crisis=False,
                requires_user_consent=True,
                consent_message="Search the web for information?",
            )
        )

        self.register_tool(
            ToolDefinition(
                name="fetch_webpage",
                description="Fetch webpage content",
                parameters={"url": "string"},
                permission_level=PermissionLevel.READ_ONLY,
                allowed_in_crisis=False,
                requires_user_consent=True,
                consent_message="Fetch content from this URL?",
            )
        )

        # Therapeutic-specific tools
        self.register_tool(
            ToolDefinition(
                name="reflect",
                description="Analyze conversation for therapeutic insights",
                parameters={"conversation_id": "string", "focus_areas": "array"},
                permission_level=PermissionLevel.THERAPEUTIC,
                allowed_in_crisis=True,
                pii_filter_enabled=True,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="consolidate",
                description="Consolidate and compress memories",
                parameters={"memory_ids": "array", "strategy": "string"},
                permission_level=PermissionLevel.THERAPEUTIC,
                allowed_in_crisis=False,  # Never auto-consolidate crisis memories
                risk_level="medium",
            )
        )

        self.register_tool(
            ToolDefinition(
                name="retain",
                description="Store a memory",
                parameters={"content": "string", "category": "string", "metadata": "object"},
                permission_level=PermissionLevel.THERAPEUTIC,
                allowed_in_crisis=True,
                pii_filter_enabled=True,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="recall",
                description="Search memories semantically",
                parameters={"query": "string", "user_id": "string", "limit": "integer"},
                permission_level=PermissionLevel.THERAPEUTIC,
                allowed_in_crisis=True,
            )
        )

        # Full access tools
        self.register_tool(
            ToolDefinition(
                name="Bash",
                description="Execute shell command",
                parameters={"command": "string"},
                permission_level=PermissionLevel.FULL,
                allowed_in_crisis=False,
                requires_user_consent=True,
                risk_level="high",
                consent_message="Execute this shell command?",
            )
        )

        self.register_tool(
            ToolDefinition(
                name="Edit",
                description="Edit file contents",
                parameters={"file_path": "string", "old_string": "string", "new_string": "string"},
                permission_level=PermissionLevel.FULL,
                allowed_in_crisis=False,
                requires_user_consent=True,
                risk_level="medium",
                consent_message="Edit this file?",
            )
        )

        self.register_tool(
            ToolDefinition(
                name="Write",
                description="Write file contents",
                parameters={"file_path": "string", "content": "string"},
                permission_level=PermissionLevel.FULL,
                allowed_in_crisis=False,
                requires_user_consent=True,
                risk_level="high",
                consent_message="Write to this file?",
            )
        )

        self.register_tool(
            ToolDefinition(
                name="Task",
                description="Create a task",
                parameters={"description": "string", "prompt": "string"},
                permission_level=PermissionLevel.FULL,
                allowed_in_crisis=False,
                requires_user_consent=True,
                risk_level="medium",
            )
        )

    def register_tool(self, tool: ToolDefinition) -> None:
        """Register a tool definition."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Get tool definition by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_allowed_tools(self) -> list[str]:
        """Get tools allowed at current permission level."""
        allowed = []
        for name, tool in self._tools.items():
            if self._is_tool_allowed_by_level(tool):
                allowed.append(name)
        return allowed

    def _is_tool_allowed_by_level(self, tool: ToolDefinition) -> bool:
        """Check if tool is allowed by permission level."""
        level_order = [
            PermissionLevel.READ_ONLY,
            PermissionLevel.THERAPEUTIC,
            PermissionLevel.FULL,
        ]

        tool_level_idx = level_order.index(tool.permission_level)
        current_level_idx = level_order.index(self.permission_level)

        return current_level_idx >= tool_level_idx


class LettaPermissionHandler:
    """
    Permission handler for Letta tool execution.

    Implements canUseTool with:
    - Permission level checking
    - Crisis context awareness
    - PII filtering for sensitive tools
    - User consent for high-risk operations
    """

    def __init__(
        self,
        registry: LettaToolRegistry,
        pii_filter: Any | None = None,
        crisis_detector: Any | None = None,
    ):
        """
        Initialize permission handler.

        Args:
            registry: Tool registry
            pii_filter: Optional PII filter instance
            crisis_detector: Optional crisis detector instance
        """
        self.registry = registry
        self._pii_filter = pii_filter
        self._crisis_detector = crisis_detector
        self._consent_callback: Callable | None = None

    def set_consent_callback(self, callback: Callable) -> None:
        """
        Set callback for user consent requests.

        Args:
            callback: Async function(user_id, message) -> bool
        """
        self._consent_callback = callback

    async def can_use_tool(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
        _user_id: str,
        context: dict[str, Any] | None = None,
    ) -> PermissionResult:
        """
        Check if a tool can be used.

        This implements Letta's canUseTool handler with Foresight integration.

        Args:
            tool_name: Name of tool to use
            tool_params: Tool parameters
            user_id: User identifier
            context: Optional context (message, session, etc.)

        Returns:
            PermissionResult with allow/block decision
        """
        # Get tool definition
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return PermissionResult(allowed=False, reason=f"Tool '{tool_name}' not registered")

        # Check permission level
        if not self.registry._is_tool_allowed_by_level(tool):
            return PermissionResult(
                allowed=False, reason=f"Tool '{tool_name}' requires {tool.permission_level.value} permission level"
            )

        # Check crisis context
        crisis_context = await self._get_crisis_context(context)
        if crisis_context != CrisisContext.NONE:
            if not tool.allowed_in_crisis:
                return PermissionResult(
                    allowed=False,
                    reason=f"Tool '{tool_name}' not allowed during crisis (context: {crisis_context.value})",
                )

            # Extra caution for high-severity crisis
            if crisis_context in [CrisisContext.HIGH, CrisisContext.CRITICAL]:
                if tool.risk_level in ["medium", "high"]:
                    return PermissionResult(
                        allowed=False, reason=f"Tool '{tool_name}' blocked due to high crisis severity"
                    )

        # Apply PII filtering if enabled
        filtered_params = tool_params.copy()
        if tool.pii_filter_enabled and self._pii_filter:
            filter_result = await self._pii_filter.filter_tool_call(tool_name, tool_params)

            if filter_result.should_block:
                return PermissionResult(allowed=False, reason="Tool parameters blocked due to PII content")

            filtered_params = filter_result.filtered

        # Check if consent is required
        if tool.requires_user_consent:
            if not self._consent_callback:
                logger.warning(f"Consent required for {tool_name} but no callback set")
                return PermissionResult(allowed=False, reason="Consent required but no consent handler configured")

            return PermissionResult(
                allowed=False,  # Will be set by consent callback
                reason="Waiting for user consent",
                filtered_params=filtered_params,
                requires_consent=True,
                consent_message=tool.consent_message,
            )

        return PermissionResult(
            allowed=True,
            reason="Permission granted",
            filtered_params=filtered_params,
        )

    async def _get_crisis_context(self, context: dict[str, Any] | None) -> CrisisContext:
        """
        Determine crisis context from message/context.

        Args:
            context: Context dictionary

        Returns:
            CrisisContext level
        """
        if not self._crisis_detector or not context:
            return CrisisContext.NONE

        message = context.get("message", "")
        if not message:
            return CrisisContext.NONE

        try:
            result = await self._crisis_detector.check_message(message)
            if result:
                # Map crisis result severity to context
                severity_map = {
                    "none": CrisisContext.NONE,
                    "low": CrisisContext.LOW,
                    "medium": CrisisContext.MEDIUM,
                    "high": CrisisContext.HIGH,
                    "critical": CrisisContext.CRITICAL,
                }
                return severity_map.get(
                    result.severity.value if hasattr(result.severity, "value") else result.severity, CrisisContext.NONE
                )
        except Exception as e:
            logger.error(f"Error checking crisis context: {e}")

        return CrisisContext.NONE

    def get_tools_for_permission_level(self, level: PermissionLevel, include_descriptions: bool = False):
        """
        Get tools available at a specific permission level.

        Args:
            level: Permission level
            include_descriptions: If True, return dict with descriptions

        Returns:
            List of tool names or dict of names to descriptions
        """
        tools = []
        level_order = [
            PermissionLevel.READ_ONLY,
            PermissionLevel.THERAPEUTIC,
            PermissionLevel.FULL,
        ]

        level_idx = level_order.index(level)

        for name, tool in self.registry._tools.items():
            tool_idx = level_order.index(tool.permission_level)
            if tool_idx <= level_idx:
                tools.append((name, tool.description))

        if include_descriptions:
            return dict(tools)
        return [t[0] for t in tools]


def create_permission_handler(
    permission_level: str = "therapeutic",
    pii_filter: Any | None = None,
    crisis_detector: Any | None = None,
) -> LettaPermissionHandler:
    """
    Create a permission handler with default configuration.

    Args:
        permission_level: Permission level string
        pii_filter: Optional PII filter
        crisis_detector: Optional crisis detector

    Returns:
        Configured LettaPermissionHandler instance
    """
    level = PermissionLevel(permission_level)
    registry = LettaToolRegistry(permission_level=level)

    return LettaPermissionHandler(
        registry=registry,
        pii_filter=pii_filter,
        crisis_detector=crisis_detector,
    )
