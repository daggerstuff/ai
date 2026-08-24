"""
Reflection Subagent Factory - local shared-memory initialization with Nvidia NIM.

Usage:
    from ai.research.reflection_factory import create_reflection_subagent
    subagent = await create_reflection_subagent()
"""

import asyncio
import logging
import os

from .local_foresight_manager import LocalForesightMemoryManager
from .nvidia_llm_callback import create_nvidia_callback
from .reflection_memory import LocalReflectionMemoryClient
from .reflection_subagent import ReflectionConfig, ReflectionSubagent, ReflectionTrigger

logger = logging.getLogger(__name__)


async def create_reflection_subagent(
    model: str | None = None,
    trigger: ReflectionTrigger = ReflectionTrigger.STEP_COUNT,
    step_threshold: int = 10,
    include_crisis_context: bool = True,
    auto_consolidate: bool = False,
    memory_client: LocalReflectionMemoryClient | None = None,
) -> ReflectionSubagent:
    """
    Create a reflection subagent with Nvidia NIM backend.

    Args:
        model: Nvidia NIM model to use.
        trigger: What triggers reflection (manual, step_count, compaction, crisis, session_end).
        step_threshold: If step_count trigger, reflect every N messages.
        include_crisis_context: Include crisis context in prompts.
        auto_consolidate: Auto-consolidate memories (False = never auto-consolidate crisis).
        memory_client: Optional existing memory client. If None, creates new one.

    Returns:
        Configured ReflectionSubagent instance.

    Example:
        subagent = await create_reflection_subagent(
            trigger=ReflectionTrigger.STEP_COUNT,
            step_threshold=10,
        )

        result = await subagent.analyze_conversation(
            conversation_text="User session transcript",
            user_id="user-123",
        )
    """
    # Resolve model from env if not provided
    resolved_model = model or os.environ.get("SUBCONSCIOUS_MODEL", "qwen/qwen3.5-397b-a17b")

    # Create or use provided memory client
    if memory_client is None:
        memory_client = LocalReflectionMemoryClient(
            LocalForesightMemoryManager(db_path=os.environ.get("FORESIGHT_DB_PATH", "foresight.db"))
        )

    # Create Nvidia NIM callback
    llm_callback = create_nvidia_callback(model=resolved_model)

    # Configure reflection
    config = ReflectionConfig(
        trigger=trigger,
        step_threshold=step_threshold,
        include_crisis_context=include_crisis_context,
        auto_consolidate=auto_consolidate,
        llm_model=resolved_model,
    )

    # Create subagent
    subagent = ReflectionSubagent(
        memory_provider=memory_client,
        config=config,
        llm_callback=llm_callback,
    )

    logger.info(f"Created reflection subagent with model {resolved_model}")

    return subagent


async def main():
    """Example usage."""

    subagent = await create_reflection_subagent(
        trigger=ReflectionTrigger.STEP_COUNT,
        step_threshold=10,
    )

    conversation = """
    User: I've been feeling anxious about my presentation.
    Therapist: What specifically worries you?
    User: Forgetting my lines or being judged.
    """

    result = await subagent.analyze_conversation(
        conversation_text=conversation,
        user_id="user-123",
    )

    if not result.crisis_detected:
        await subagent.consolidate_memories("user-123", result)
    else:
        pass


if __name__ == "__main__":
    asyncio.run(main())
