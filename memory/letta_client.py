#!/usr/bin/env python3
"""
Letta SDK Client for Pixelated Empathy.

Provides integration with Letta Code SDK for autonomous agent capabilities:
- Background transcript streaming
- Client-side tools (Read, Grep, Glob, web_search)
- Autonomous memory updates
- Multi-project memory sharing
"""
from letta import LettaClient as SDKClient


import asyncio
import json
import logging
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("letta_client")

# Default configuration
DEFAULT_BASE_URL = "https://api.letta.ai"
CONFIG_DIR = Path.home() / ".letta" / "claude-subconscious"
CONFIG_FILE = CONFIG_DIR / "config.json"

# P0 Fix: Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
RETRY_BACKOFF = 2.0


def retry_on_failure(max_retries=MAX_RETRIES, delay=RETRY_DELAY, backoff=RETRY_BACKOFF):
    """
    P0 Fix: Retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Multiplier for delay after each retry
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {current_delay}s...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed: {e}")

            raise last_exception or Exception("Unknown error")
        return wrapper
    return decorator


@dataclass
class LettaConfig:
    """Configuration for Letta SDK integration."""
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    agent_id: str | None = None
    mode: str = "whisper"  # whisper, full, off
    sdk_tools: list[str] | None = None

    def __post_init__(self):
        if self.sdk_tools is None:
            # Default tools based on mode
            if self.mode == "read-only":
                self.sdk_tools = ["Read", "Grep", "Glob", "web_search", "fetch_webpage"]
            elif self.mode == "full":
                self.sdk_tools = ["Read", "Grep", "Glob", "web_search", "fetch_webpage", "Bash", "Edit", "Write", "Task"]
            else:
                self.sdk_tools = []


class LettaClient:
    """
    Letta SDK client for autonomous agent operations.

    Provides:
    - Session management with persistent agent state
    - Tool registration and execution
    - Memory block synchronization
    - Background transcript streaming
    """

    def __init__(self, config: LettaConfig | None = None):
        """
        Initialize Letta client.

        Args:
            config: Optional configuration. If not provided, loads from environment/config file.
        """
        self.config = config or self._load_config()
        self.client = None
        self._agent = None
        self._initialized = False

    def _load_config(self) -> LettaConfig:
        """Load configuration from environment and config file."""
        # Check environment variables first
        api_key = os.environ.get("LETTA_API_KEY")
        base_url = os.environ.get("LETTA_BASE_URL", DEFAULT_BASE_URL)
        agent_id = os.environ.get("LETTA_AGENT_ID")
        mode = os.environ.get("LETTA_MODE", "whisper")

        # Fall back to config file
        if not api_key and CONFIG_FILE.exists():
            try:
                config_data = json.loads(CONFIG_FILE.read_text())
                api_key = api_key or config_data.get("api_key")
                agent_id = agent_id or config_data.get("agent_id")
                base_url = base_url or config_data.get("base_url", DEFAULT_BASE_URL)
                mode = mode or config_data.get("mode", "whisper")
            except (json.JSONDecodeError, KeyError):
                pass

        return LettaConfig(
            api_key=api_key,
            base_url=base_url,
            agent_id=agent_id,
            mode=mode,
        )

    async def initialize(self) -> None:
        """Initialize the Letta client and agent."""
        if self._initialized:
            return

        if not self.config.api_key:
            logger.warning("LETTA_API_KEY not set. Running in memory-only mode.")
            return

        try:
            # Import Letta SDK

            self.client = SDKClient(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )

            # Get or create agent
            if self.config.agent_id:
                self._agent = self.client.get_agent(self.config.agent_id)
            else:
                # Create new agent with default configuration
                self._agent = self.client.create_agent(
                    name="pixelated-empathy-subconscious",
                    description="Background agent for therapeutic session analysis and memory management",
                )
                self._save_agent_id(self._agent.id)

            self._initialized = True
            logger.info(f"Letta client initialized with agent: {self._agent.id}")

        except ImportError as e:
            logger.warning(f"Letta SDK not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize Letta client: {e}")

    def _save_agent_id(self, agent_id: str) -> None:
        """
        Save agent ID to config file.

        P0 Fix: Set secure permissions on config directory (owner only: 0o700)
        """
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        # P0 Fix: Set secure permissions on config directory
        os.chmod(CONFIG_DIR, stat.S_IRWXU)  # 0o700 - owner rwx only

        config_data = {}
        if CONFIG_FILE.exists():
            try:
                config_data = json.loads(CONFIG_FILE.read_text())
            except json.JSONDecodeError:
                pass

        config_data["agent_id"] = agent_id
        config_data["last_updated"] = datetime.now(timezone.utc).isoformat()

        # P0 Fix: Set secure permissions on config file (owner read/write only)
        CONFIG_FILE.write_text(json.dumps(config_data, indent=2))
        os.chmod(CONFIG_FILE, stat.S_IRUSR | stat.S_IWUSR)  # 0o600

    async def stream_transcript(self, messages: list[dict[str, Any]], session_id: str) -> None:
        """
        Stream session transcript to Letta agent.

        P0 Fix: Added timeout and proper error handling

        Args:
            messages: List of message dicts with role/content/timestamp
            session_id: Unique session identifier
        """
        if not self._initialized:
            await self.initialize()

        if not self._agent or not self.client:
            logger.warning("Letta client not initialized, skipping transcript stream")
            return

        try:
            # P0 Fix: Add timeout to prevent hanging
            await asyncio.wait_for(
                self._stream_messages(messages, session_id),
                timeout=60.0  # 60 second timeout for entire transcript
            )
            logger.info(f"Streamed {len(messages)} messages to Letta agent")

        except TimeoutError:
            logger.error(f"Timeout streaming transcript for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to stream transcript: {e}")

    async def _stream_messages(self, messages: list[dict[str, Any]], session_id: str) -> None:
        """Internal method to stream messages (called with timeout)."""
        for message in messages:
            self.client.send_message(
                agent_id=self._agent.id if self._agent else None,
                role=message.get("role", "user"),
                content=message.get("content", ""),
                metadata={
                    "session_id": session_id,
                    "timestamp": message.get("timestamp", datetime.now(timezone.utc).isoformat()),
                },
            )

    async def get_memory_blocks(self) -> dict[str, str]:
        """
        Get current memory blocks from agent.

        Returns:
            Dict of memory block labels to content
        """
        if not self._initialized:
            await self.initialize()

        if not self._agent or not self.client:
            return {}

        try:
            # Get agent state including memory blocks
            state = self.client.get_agent_state(self._agent.id)
            blocks = {}

            # Extract memory blocks from state
            if hasattr(state, "memory_blocks"):
                for block in state.memory_blocks:
                    blocks[block.label] = block.content

            return blocks

        except Exception as e:
            logger.error(f"Failed to get memory blocks: {e}")
            return {}

    async def update_memory_block(self, label: str, content: str) -> None:
        """
        Update a memory block.

        Args:
            label: Block label
            content: New content
        """
        if not self._initialized:
            await self.initialize()

        if not self._agent or not self.client:
            return

        try:
            self.client.update_memory_block(
                agent_id=self._agent.id,
                label=label,
                content=content,
            )
            logger.info(f"Updated memory block: {label}")

        except Exception as e:
            logger.error(f"Failed to update memory block: {e}")

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a client-side tool.

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if not self._initialized:
            await self.initialize()

        if not self._agent or not self.client:
            return None

        # Check if tool is available
        sdk_tools = self.config.sdk_tools or []
        if tool_name not in sdk_tools:
            logger.warning(f"Tool {tool_name} not available in current mode ({self.config.mode})")
            return None

        try:
            return self.client.execute_tool(
                agent_id=self._agent.id,
                tool_name=tool_name,
                arguments=arguments,
            )

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return None

    async def close(self) -> None:
        """Close the client and clean up resources."""
        if self.client:
            # Close any open connections
            pass
        self._initialized = False


# Singleton instance
_client: LettaClient | None = None


def get_client(config: LettaConfig | None = None) -> LettaClient:
    """Get or create the global Letta client instance."""
    global _client
    if _client is None:
        _client = LettaClient(config)
    return _client


async def initialize_client(config: LettaConfig | None = None) -> LettaClient:
    """Initialize and return the global client."""
    client = get_client(config)
    await client.initialize()
    return client
