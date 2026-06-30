"""
Reflection Bootstrap - Auto-start reflection subagent.

This module provides functions to bootstrap and auto-run the reflection subagent
as a background task or on-demand.
"""

import asyncio
import contextlib
import hashlib
import logging
import os
from collections.abc import Callable

from .reflection_factory import create_reflection_subagent

logger = logging.getLogger(__name__)


class ReflectionBootstrap:
    """
    Bootstrap and manage reflection subagent lifecycle.

    Usage:
        bootstrap = ReflectionBootstrap()
        await bootstrap.start()  # Start background monitoring

        # Or run on-demand:
        result = await bootstrap.reflect_now(user_id="user-123")
    """

    def __init__(
        self,
        model: str | None = None,
        step_threshold: int = 10,
    ):
        self.model = model or os.environ.get("SUBCONSCIOUS_MODEL", "qwen/qwen3.5-397b-a17b")
        self.step_threshold = step_threshold
        self._subagent = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_reflection_fingerprint: str | None = None

    async def start(self):
        """Initialize and start reflection subagent."""
        logger.info("Starting reflection bootstrap...")
        self._subagent = await create_reflection_subagent(
            model=self.model,
            step_threshold=self.step_threshold,
        )
        self._running = True
        logger.info("Reflection bootstrap started")
        return self._subagent

    async def stop(self):
        """Stop the reflection bootstrap."""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._subagent is not None and hasattr(self._subagent, "close"):
            await self._subagent.close()
            self._subagent = None
        logger.info("Reflection bootstrap stopped")

    async def reflect_now(
        self,
        conversation_text: str,
        user_id: str,
    ):
        """
        Run reflection on-demand.

        Args:
            conversation_text: Conversation to analyze
            user_id: User identifier

        Returns:
            ReflectionResult with analysis
        """
        if self._subagent is None:
            await self.start()

        assert self._subagent is not None
        self._last_reflection_fingerprint = self._fingerprint_conversation(
            conversation_text=conversation_text,
            user_id=user_id,
        )
        return await self._subagent.analyze_conversation(
            conversation_text=conversation_text,
            user_id=user_id,
        )

    def should_reflect(self) -> bool:
        """Check if reflection should trigger based on step count."""
        if self._subagent is None:
            return False
        return self._subagent.should_reflect()

    async def run_background_loop(
        self,
        get_conversation: Callable,
        interval_seconds: int = 60,
    ):
        """
        Run reflection in background, checking periodically.

        Args:
            get_conversation: Async function to get conversation text
            interval_seconds: How often to check (default 60s)
        """
        if self._subagent is None:
            await self.start()

        logger.info(f"Starting background reflection loop (interval={interval_seconds}s)")

        while self._running:
            try:
                if self.should_reflect():
                    conversation = await get_conversation()
                    if conversation and self._should_reflect_conversation(conversation, "system"):
                        logger.info("Triggering reflection...")
                        result = await self.reflect_now(
                            conversation_text=conversation,
                            user_id="system",
                        )
                        logger.info(f"Reflection complete: crisis={result.crisis_detected}")

                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background reflection error: {e}")
                await asyncio.sleep(interval_seconds)

    @staticmethod
    def _fingerprint_conversation(conversation_text: str, user_id: str) -> str:
        digest = hashlib.sha256()
        digest.update(user_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(conversation_text.encode("utf-8"))
        return digest.hexdigest()

    def _should_reflect_conversation(self, conversation_text: str, user_id: str) -> bool:
        fingerprint = self._fingerprint_conversation(
            conversation_text=conversation_text,
            user_id=user_id,
        )
        return fingerprint != self._last_reflection_fingerprint


async def create_and_start(
    model: str | None = None,
    step_threshold: int = 10,
) -> ReflectionBootstrap:
    """
    Create and start reflection bootstrap.

    Returns:
        Running ReflectionBootstrap instance
    """
    bootstrap = ReflectionBootstrap(
        model=model,
        step_threshold=step_threshold,
    )
    await bootstrap.start()
    return bootstrap
