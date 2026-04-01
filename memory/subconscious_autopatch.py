"""
Subconscious Auto-Patch - Automatic LLM wrapper injection.

This module patches the OpenAI client at import time to automatically
inject subconscious context into ALL LLM calls - no code changes needed.

Usage:
    import ai.memory.subconscious_autopatch  # Import once at startup

That's it. All OpenAI/Anthropic calls now have subconscious context injection.
"""
import asyncio
import functools
import logging
import os
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Global state
_subconscious_active = False
_user_id: Optional[str] = None
_bootstrap = None
_injection_enabled = os.environ.get("SUBCONSCIOUS_ENABLED", "true").lower() == "true"


def is_subconscious_active() -> bool:
    """Check if subconscious injection is active."""
    return _subconscious_active and _injection_enabled


def get_subconscious_context() -> Optional[Any]:
    """Get current subconscious context if active."""
    if not _subconscious_active:
        return None
    return _bootstrap


async def _query_subconscious_async(conversation_context: str, user_id: str) -> str:
    """Query subconscious and return injection string."""
    try:
        from .reflection_bootstrap import ReflectionBootstrap

        bootstrap = await ReflectionBootstrap.create_and_start()
        result = await bootstrap.reflect_now(
            conversation_text=conversation_context,
            user_id=user_id,
        )

        parts = []

        if result.crisis_detected:
            parts.append(f"<crisis_alert>Active: {', '.join(result.crisis_indicators)}</crisis_alert>")

        if result.memories_preserved:
            memories = "\n".join(f"- {m}" for m in result.memories_preserved[:5])
            parts.append(f"<relevant_memories>\n{memories}\n</relevant_memories>")

        if result.recommendations:
            recs = "\n".join(f"- {r}" for r in result.recommendations[:5])
            parts.append(f"<pattern_observations>\n{recs}\n</pattern_observations>")

        if parts:
            return "<subconscious_context>\n" + "\n".join(parts) + "\n</subconscious_context>\n\n"

        await bootstrap.stop()
        return ""
    except Exception as e:
        logger.error(f"Subconscious query failed: {e}")
        return ""


def _wrap_llm_call(original_func, args, kwargs, user_id: str = "pixelated"):
    """Wrap LLM call to inject subconscious context."""
    if not _subconscious_active or not _injection_enabled:
        return original_func(*args, **kwargs)

    # Extract prompt from kwargs
    messages = kwargs.get("messages", [])
    if not messages:
        return original_func(*args, **kwargs)

    # Get last user message
    last_user_msg = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    if not last_user_msg:
        return original_func(*args, **kwargs)

    # Build conversation context from message history
    conversation_context = "\n".join(
        f"{m.get('role', 'unknown')}: {m.get('content', '')}"
        for m in messages[-10:]  # Last 10 messages for context
    )

    # Query subconscious (sync wrapper around async)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    injection = loop.run_until_complete(
        _query_subconscious_async(conversation_context, user_id)
    )

    if injection:
        # Prepend subconscious context to the last user message
        original_content = messages[-1].get("content", "")
        messages[-1]["content"] = injection + original_content if original_content else injection
        kwargs["messages"] = messages

    return original_func(*args, **kwargs)


def patch_openai_chat():
    """Patch OpenAI ChatCompletion to inject subconscious context."""
    try:
        from openai import OpenAI

        original_init = OpenAI.__init__
        original_chat_create = None

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            nonlocal original_chat_create
            if original_chat_create is None:
                original_chat_create = self.chat.completions.create

                @functools.wraps(original_chat_create)
                def wrapped_create(*args, **wrap_kwargs):
                    return _wrap_llm_call(
                        lambda: original_chat_create(*args, **wrap_kwargs),
                        args,
                        wrap_kwargs,
                        _user_id or "pixelated",
                    )

                self.chat.completions.create = wrapped_create

        OpenAI.__init__ = patched_init
        logger.info("Subconscious: OpenAI client patched")
        return True
    except Exception as e:
        logger.error(f"Subconscious: Failed to patch OpenAI: {e}")
        return False


def patch_anthropic_chat():
    """Patch Anthropic Claude to inject subconscious context."""
    try:
        import anthropic

        original_init = anthropic.Anthropic.__init__
        original_message_create = None

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            nonlocal original_message_create
            if original_message_create is None:
                original_message_create = self.messages.create

                @functools.wraps(original_message_create)
                def wrapped_create(*args, **wrap_kwargs):
                    return _wrap_llm_call(
                        lambda: original_message_create(*args, **wrap_kwargs),
                        args,
                        wrap_kwargs,
                        _user_id or "pixelated",
                    )

                self.messages.create = wrapped_create

        anthropic.Anthropic.__init__ = patched_init
        logger.info("Subconscious: Anthropic client patched")
        return True
    except Exception as e:
        logger.error(f"Subconscious: Failed to patch Anthropic: {e}")
        return False


def activate_subconscious(user_id: str = "pixelated", auto_start: bool = True) -> bool:
    """
    Activate subconscious injection for all LLM calls.

    This is called once at application startup - then all LLM calls
    automatically have subconscious context injected.

    Args:
        user_id: User identifier for memory lookup
        auto_start: If True, auto-starts the reflection bootstrap

    Returns:
        True if activation successful
    """
    global _subconscious_active, _user_id, _bootstrap

    if not _injection_enabled:
        logger.info("Subconscious: Disabled via SUBCONSCIOUS_ENABLED=false")
        return False

    try:
        _user_id = user_id

        # Start reflection bootstrap
        if auto_start:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                from .reflection_bootstrap import create_and_start

                _bootstrap = loop.run_until_complete(
                    create_and_start()
                )
                _subconscious_active = True
                logger.info(f"Subconscious: Activated for user '{user_id}'")
            finally:
                loop.close()
        else:
            _subconscious_active = True
            logger.info(f"Subconscious: Activated (manual start) for user '{user_id}'")

        # Patch LLM clients
        patch_openai_chat()
        patch_anthropic_chat()

        return True
    except Exception as e:
        logger.error(f"Subconscious: Activation failed: {e}")
        return False


def deactivate_subconscious():
    """Deactivate subconscious injection."""
    global _subconscious_active, _bootstrap

    _subconscious_active = False
    if _bootstrap:
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_bootstrap.stop())
        except Exception as e:
            logger.error(f"Subconscious: Deactivation error: {e}")

    logger.info("Subconscious: Deactivated")


# Auto-activate on import if enabled
if os.environ.get("SUBCONSCIOUS_AUTOACTIVATION", "true").lower() == "true":
    user_from_env = os.environ.get("SUBCONSCIOUS_USER_ID", "pixelated")
    activate_subconscious(user_id=user_from_env)
    logger.info(f"Subconscious: Auto-activated for user '{user_from_env}'")
