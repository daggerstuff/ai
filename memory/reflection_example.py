"""
Example: Using Reflection Subagent with Nvidia NIM.

This file demonstrates how to initialize and use the reflection subagent
with Nvidia NIM (qwen/qwen3.5-397b-a17b) and the shared local memory service.
"""
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from ai.memory.local_hindsight_manager import LocalHindsightMemoryManager  # noqa: E402
from ai.memory.reflection_memory import LocalReflectionMemoryClient  # noqa: E402
from ai.memory.reflection_subagent import (  # noqa: E402
    ReflectionSubagent,
    ReflectionConfig,
    ReflectionTrigger,
)
from ai.memory.nvidia_llm_callback import create_nvidia_callback  # noqa: E402


async def main():
    """Example usage of reflection subagent with Nvidia NIM."""

    # Step 1: Create local shared-memory client
    memory_client = LocalReflectionMemoryClient(LocalHindsightMemoryManager())

    # Step 2: Create Nvidia NIM callback
    # Uses NVIDIA_API_KEY from .env automatically
    llm_callback = create_nvidia_callback(
        model="qwen/qwen3.5-397b-a17b",
        base_url="https://integrate.api.nvidia.com/v1",
    )

    # Step 3: Configure reflection subagent
    config = ReflectionConfig(
        trigger=ReflectionTrigger.STEP_COUNT,  # Reflect every N messages
        step_threshold=10,  # Reflect every 10 messages
        include_crisis_context=True,  # Include crisis context in prompts
        auto_consolidate=False,  # Never auto-consolidate crisis content
        max_memories_to_review=50,
        llm_model="qwen/qwen3.5-397b-a17b",
    )

    # Step 4: Create reflection subagent
    subagent = ReflectionSubagent(
        memory_provider=memory_client,
        config=config,
        llm_callback=llm_callback,  # Pass the Nvidia NIM callback
    )

    # Step 5: Analyze a conversation
    conversation = """
    User: I've been feeling really anxious about my presentation next week.
    Therapist: That's a common concern. What specifically worries you about it?
    User: I'm afraid I'll forget my lines or people will judge me.
    Therapist: Let's work on some coping strategies for that.
    """

    result = await subagent.analyze_conversation(
        conversation_text=conversation,
        user_id="user-123",
    )

    print(f"Crisis detected: {result.crisis_detected}")
    print(f"Memories to preserve: {result.memories_preserved}")
    print(f"Memories to consolidate: {result.memories_consolidated}")
    print(f"Requires review: {result.requires_manual_review}")

    # Step 6: Consolidate memories (if no crisis)
    if not result.crisis_detected:
        stats = await subagent.consolidate_memories("user-123", result)
        print(f"Consolidation stats: {stats}")
    else:
        print("Crisis detected - skipping auto-consolidation")


if __name__ == "__main__":
    asyncio.run(main())
