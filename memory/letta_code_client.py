"""
Letta Code SDK Client - Modern agent-based implementation.

This client implements Letta Code SDK patterns:
- Agent-based persistence (vs session-based)
- Multi-conversation support per agent
- Fine-grained tool permissions
- Crisis-aware memory handling
- Integration with Foresight therapeutic memory

Migration from Claude Agent SDK:
- unstable_v2_createSession → createAgent
- unstable_v2_resumeSession(session_id) → resumeSession(agentId)
- session.send/stream() remain similar but agent-anchored
"""

import contextlib
import json
import logging
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

try:
    from letta import LettaClient as SDKClient
except ModuleNotFoundError:
    SDKClient = None

from .letta_crisis_handler import LettaCrisisHandler
from .letta_pii_middleware import LettaPIIMiddleware

logger = logging.getLogger("letta_code_client")

# Configuration
DEFAULT_BASE_URL = "https://api.letta.ai"
CONFIG_DIR = Path.home() / ".letta" / "pixelated-empathy"
CONFIG_FILE = CONFIG_DIR / "agent_config.json"

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1.0
RETRY_BACKOFF = 2.0


class PermissionMode(StrEnum):
    """Tool permission modes for Letta agents."""

    READONLY = "read-only"
    THERAPEUTIC = "therapeutic"
    FULL = "full"
    WHISPER = "whisper"  # Background only, no tool execution


class ModelProvider(StrEnum):
    """Supported model providers."""

    CLAUDE = "claude"
    GPT = "gpt"
    GEMINI = "gemini"
    LOCAL = "local"


@dataclass
class ToolPermission:
    """Tool permission configuration."""

    can_use: bool
    requires_consent: bool = False
    allowed_for_crisis: bool = False
    filters: list[str] = field(default_factory=list)


@dataclass
class LettaCodeConfig:
    """Configuration for Letta Code SDK integration."""

    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    agent_id: str | None = None
    permission_mode: PermissionMode = PermissionMode.THERAPEUTIC
    model_provider: ModelProvider = ModelProvider.CLAUDE
    crisis_detection_enabled: bool = True
    pii_filter_enabled: bool = True
    dual_storage_enabled: bool = True

    # Tool permissions
    tool_permissions: dict[str, ToolPermission] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default tool permissions based on mode."""
        if not self.tool_permissions:
            self.tool_permissions = self._get_default_permissions()

    def _get_default_permissions(self) -> dict[str, ToolPermission]:
        """Get default tool permissions based on mode."""
        if self.permission_mode == PermissionMode.READONLY:
            return {
                "Read": ToolPermission(True),
                "Grep": ToolPermission(True),
                "Glob": ToolPermission(True),
                "web_search": ToolPermission(True, requires_consent=True),
                "fetch_webpage": ToolPermission(True, requires_consent=True),
            }
        if self.permission_mode == PermissionMode.THERAPEUTIC:
            return {
                "Read": ToolPermission(True),
                "Grep": ToolPermission(True),
                "Glob": ToolPermission(True),
                "web_search": ToolPermission(True),
                "fetch_webpage": ToolPermission(True),
                # Therapeutic-specific tools
                "reflect": ToolPermission(True, allowed_for_crisis=True),
                "consolidate": ToolPermission(True, allowed_for_crisis=False),
            }
        if self.permission_mode == PermissionMode.FULL:
            return {
                "Read": ToolPermission(True),
                "Grep": ToolPermission(True),
                "Glob": ToolPermission(True),
                "web_search": ToolPermission(True),
                "fetch_webpage": ToolPermission(True),
                "Bash": ToolPermission(True, requires_consent=True),
                "Edit": ToolPermission(True, requires_consent=True),
                "Write": ToolPermission(True, requires_consent=True),
                "Task": ToolPermission(True, requires_consent=True),
            }
        # WHISPER
        return {}


class LettaCodeClient:
    """
    Modern Letta Code SDK client implementing agent-based persistence.

    Key features:
    - Agent-based memory (persistent across sessions)
    - Multi-conversation support per agent
    - Crisis-aware tool permissions
    - Foresight PII filtering integration
    - Dual-storage backend support

    Usage:
        # Create agent (one-time)
        agent_id = await client.create_agent(system_prompt)

        # Resume agent for conversations
        session = await client.resume_session(agent_id)
        response = await session.send("Hello!")

        # Multi-conversation support
        conv1 = await client.create_conversation(agent_id)
        conv2 = await client.create_conversation(agent_id)
    """

    def __init__(self, config: LettaCodeConfig | None = None):
        """
        Initialize Letta Code client.

        Args:
            config: Optional configuration. Loads from environment if not provided.
        """
        self.config = config or self._load_config()
        self._sdk_client = None
        self._agent = None
        self._initialized = False
        self._conversations: dict[str, Any] = {}

        # Middleware components
        self._pii_filter = None
        self._crisis_detector = None

    def _load_config(self) -> LettaCodeConfig:
        """Load configuration from environment and config file."""
        api_key = os.environ.get("LETTA_API_KEY")
        base_url = os.environ.get("LETTA_BASE_URL", DEFAULT_BASE_URL)
        agent_id = os.environ.get("LETTA_AGENT_ID")
        permission_mode = PermissionMode(os.environ.get("LETTA_PERMISSION_MODE", "therapeutic"))

        # Load from config file if exists
        if not api_key and CONFIG_FILE.exists():
            try:
                config_data = json.loads(CONFIG_FILE.read_text())
                api_key = api_key or config_data.get("api_key")
                agent_id = agent_id or config_data.get("agent_id")
                base_url = base_url or config_data.get("base_url", DEFAULT_BASE_URL)
            except (json.JSONDecodeError, KeyError):
                pass

        return LettaCodeConfig(
            api_key=api_key,
            base_url=base_url,
            agent_id=agent_id,
            permission_mode=permission_mode,
        )

    async def initialize(self) -> None:
        """Initialize the Letta Code SDK client."""
        if self._initialized:
            return

        if not self.config.api_key:
            logger.warning("LETTA_API_KEY not set. Running in memory-only mode.")
            return

        try:
            if SDKClient is None:
                logger.warning("Letta Code SDK not installed; skipping initialization.")
                return
            # Import Letta Code SDK
            self._sdk_client = SDKClient(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )

            # Initialize crisis detection and PII filtering
            self._init_middleware()

            self._initialized = True
            logger.info("Letta Code SDK client initialized")

        except ImportError as e:
            logger.warning(f"Letta Code SDK not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize Letta Code client: {e}")

    def _init_middleware(self) -> None:
        """Initialize PII filter and crisis detector middleware."""
        # Import Foresight components for middleware
        try:
            if self.config.pii_filter_enabled:
                self._pii_filter = LettaPIIMiddleware(
                    None,  # Will be set by bridge
                    {"max_redaction_ratio": 0.5},
                )

            if self.config.crisis_detection_enabled:
                self._crisis_detector = LettaCrisisHandler(None)

        except ImportError:
            logger.warning("Foresight middleware not available")

    async def create_agent(
        self,
        system_prompt: str,
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        """
        Create a new persistent agent.

        This replaces unstable_v2_createSession from Claude SDK.
        The agent persists across sessions and can have multiple conversations.

        Args:
            system_prompt: System prompt for the agent
            name: Optional agent name
            description: Optional agent description

        Returns:
            Agent ID
        """
        if not self._initialized:
            await self.initialize()

        if not self._sdk_client:
            raise RuntimeError("Letta SDK client not initialized")

        try:
            agent = self._sdk_client.create_agent(
                name=name or "pixelated-empathy-agent",
                description=description or "Therapeutic agent with Foresight memory integration",
                system_prompt=system_prompt,
            )

            self._agent = agent
            self._save_agent_config(agent.id)

            logger.info(f"Created Letta agent: {agent.id}")
            return agent.id

        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise

    async def resume_session(self, agent_id: str) -> "LettaSession":
        """
        Resume a session with an existing agent.

        This replaces unstable_v2_resumeSession from Claude SDK.
        The agent's memory blocks are automatically loaded.

        Args:
            agent_id: Agent ID to resume

        Returns:
            LettaSession instance
        """
        if not self._initialized:
            await self.initialize()

        if not self._sdk_client:
            raise RuntimeError("Letta SDK client not initialized")

        # Get agent state
        try:
            agent = self._sdk_client.get_agent(agent_id)
            self._agent = agent

            return LettaSession(
                client=self,
                agent_id=agent_id,
                pii_filter=self._pii_filter,
                crisis_detector=self._crisis_detector,
            )

        except Exception as e:
            logger.error(f"Failed to resume session for agent {agent_id}: {e}")
            raise

    async def create_conversation(self, agent_id: str) -> str:
        """
        Create a new conversation thread for an agent.

        Letta supports multiple concurrent conversations per agent.
        Each conversation has isolated context.

        Args:
            agent_id: Agent ID

        Returns:
            Conversation ID
        """
        if not self._initialized:
            await self.initialize()

        if not self._sdk_client:
            raise RuntimeError("Letta SDK client not initialized")

        try:
            conversation = self._sdk_client.create_conversation(agent_id=agent_id)
            self._conversations[conversation.id] = conversation
            return conversation.id

        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise

    async def get_memory_blocks(self, agent_id: str) -> dict[str, str]:
        """
        Get all memory blocks for an agent.

        Memory blocks persist across sessions and contain:
        - User preferences
        - Therapeutic insights
        - Crisis context (if any)
        - Treatment progress

        Args:
            agent_id: Agent ID

        Returns:
            Dict of block labels to content
        """
        if not self._initialized:
            await self.initialize()

        if not self._sdk_client:
            return {}

        try:
            state = self._sdk_client.get_agent_state(agent_id)
            blocks = {}

            if hasattr(state, "memory_blocks"):
                for block in state.memory_blocks:
                    blocks[block.label] = block.content

            return blocks

        except Exception as e:
            logger.error(f"Failed to get memory blocks: {e}")
            return {}

    async def update_memory_block(
        self,
        agent_id: str,
        label: str,
        content: str,
    ) -> None:
        """
        Update a memory block.

        Content is filtered through PII middleware before storing.

        Args:
            agent_id: Agent ID
            label: Block label
            content: New content
        """
        if not self._initialized:
            await self.initialize()

        if not self._sdk_client:
            return

        # Filter through PII middleware
        if self._pii_filter:
            filter_result = await self._pii_filter.filter_tool_call("update_memory_block", {"content": content})
            if filter_result.should_block:
                logger.warning(f"Memory block update blocked due to PII: {label}")
                return
            content = filter_result.filtered

        try:
            self._sdk_client.update_memory_block(
                agent_id=agent_id,
                label=label,
                content=content,
            )
            logger.info(f"Updated memory block: {label}")

        except Exception as e:
            logger.error(f"Failed to update memory block: {e}")

    async def can_use_tool(
        self,
        _agent_id: str,
        tool_name: str,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """
        Check if a tool can be used based on permissions and crisis state.

        This implements Letta's canUseTool permission handler with
        Foresight's crisis-aware filtering.

        Args:
            agent_id: Agent ID
            tool_name: Tool name
            context: Optional context for permission check

        Returns:
            True if tool can be used, False otherwise
        """
        # Check if tool is in allowed permissions
        if tool_name not in self.config.tool_permissions:
            logger.warning(f"Tool {tool_name} not in permissions")
            return False

        permission = self.config.tool_permissions[tool_name]

        if not permission.can_use:
            return False

        # Check crisis state if detector is available
        if self._crisis_detector and context:
            crisis_result = await self._crisis_detector.check_message(context.get("message", ""))

            # Block tools not allowed during crisis
            if crisis_result and not permission.allowed_for_crisis:
                logger.warning(f"Tool {tool_name} blocked due to crisis state")
                return False

        return True

    def _save_agent_config(self, agent_id: str) -> None:
        """Save agent configuration to file with secure permissions."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        # Set secure permissions
        os.chmod(CONFIG_DIR, stat.S_IRWXU)  # 0o700

        config_data = {}
        if CONFIG_FILE.exists():
            with contextlib.suppress(json.JSONDecodeError):
                config_data = json.loads(CONFIG_FILE.read_text())

        config_data["agent_id"] = agent_id
        config_data["last_updated"] = datetime.now(UTC).isoformat()

        CONFIG_FILE.write_text(json.dumps(config_data, indent=2))
        os.chmod(CONFIG_FILE, stat.S_IRUSR | stat.S_IWUSR)  # 0o600

    async def close(self) -> None:
        """Close the client and clean up resources."""
        self._initialized = False
        self._conversations.clear()


class LettaSession:
    """
    Session wrapper for Letta agent interactions.

    Provides:
    - PII filtering on all messages
    - Crisis detection before processing
    - Integration with Foresight memory
    - Dual-storage support
    """

    def __init__(
        self,
        client: LettaCodeClient,
        agent_id: str,
        pii_filter: Any | None = None,
        crisis_detector: Any | None = None,
    ):
        self.client = client
        self.agent_id = agent_id
        self._pii_filter = pii_filter
        self._crisis_detector = crisis_detector

    async def send(
        self,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Send a message to the agent.

        Message is processed through:
        1. Crisis detection (blocks if critical)
        2. PII filtering (blocks if too much PII)
        3. Letta agent processing
        4. Foresight memory storage

        Args:
            message: User message
            metadata: Optional metadata

        Returns:
            Agent response
        """
        # Crisis detection first (safety first)
        if self._crisis_detector:
            crisis_result = await self._crisis_detector.check_message(message)
            if crisis_result and crisis_result.severity == "critical":
                return self._crisis_response(crisis_result)

        # PII filtering
        filtered_message = message
        if self._pii_filter:
            filter_result = await self._pii_filter.filter_tool_call("send_message", {"content": message})

            if filter_result.should_block:
                logger.warning("Message blocked due to PII content")
                return "I can't process that message as it contains sensitive information."

            filtered_message = filter_result.filtered

        # Send to Letta agent
        if self.client._sdk_client and self.client._agent:
            try:
                response = self.client._sdk_client.send_message(
                    agent_id=self.agent_id,
                    role="user",
                    content=filtered_message,
                    metadata=metadata or {},
                )
                return response.content if hasattr(response, "content") else str(response)
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                return f"Error processing message: {e}"

        return "Letta client not initialized"

    async def stream(
        self,
        message: str,
        callback: Callable[[str], None],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Stream agent response.

        Args:
            message: User message
            callback: Callback for each chunk
            metadata: Optional metadata
        """
        # Apply same filtering as send()
        # Crisis detection
        if self._crisis_detector:
            crisis_result = await self._crisis_detector.check_message(message)
            if crisis_result and crisis_result.severity == "critical":
                callback(self._crisis_response(crisis_result))
                return

        # PII filtering
        filtered_message = message
        if self._pii_filter:
            filter_result = await self._pii_filter.filter_tool_call("stream_message", {"content": message})

            if filter_result.should_block:
                callback("Message blocked due to PII content")
                return

            filtered_message = filter_result.filtered

        # Stream from Letta
        if self.client._sdk_client and self.client._agent:
            try:
                async for chunk in self.client._sdk_client.stream_message(
                    agent_id=self.agent_id,
                    role="user",
                    content=filtered_message,
                    metadata=metadata or {},
                ):
                    if hasattr(chunk, "content"):
                        callback(chunk.content)
            except Exception as e:
                logger.error(f"Failed to stream message: {e}")
                callback(f"Error: {e}")

    def _crisis_response(self, crisis_result: Any) -> str:
        """Generate crisis response."""
        if hasattr(crisis_result, "suggested_action") and crisis_result.suggested_action:
            return f"I'm concerned. {crisis_result.suggested_action}. Please reach out to a professional."
        return "I'm here to help. Consider reaching out to a professional."


# Convenience functions
_client: LettaCodeClient | None = None


def get_client(config: LettaCodeConfig | None = None) -> LettaCodeClient:
    """Get or create global Letta Code client."""
    global _client
    if _client is None:
        _client = LettaCodeClient(config)
    return _client


async def initialize_client(config: LettaCodeConfig | None = None) -> LettaCodeClient:
    """Initialize and return global client."""
    client = get_client(config)
    await client.initialize()
    return client
