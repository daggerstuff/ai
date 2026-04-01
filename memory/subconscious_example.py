"""
Example: Using SubconsciousLLM wrapper with OpenAI-compatible API.

This shows how to wrap any LLM call with subconscious context injection.
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from ai.memory.subconscious_wrapper import create_subconscious_llm
from ai.memory.reflection_bootstrap import ReflectionBootstrap


async def openai_chat(prompt: str, **kwargs) -> str:
    """Example OpenAI-compatible chat callback."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("NVIDIA_API_KEY"),
        base_url="https://integrate.api.nvidia.com/v1",
    )

    response = client.chat.completions.create(
        model=kwargs.get("model", "qwen/qwen3.5-397b-a17b"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2048,
    )

    return response.choices[0].message.content or ""


async def main():
    # Step 1: Start reflection bootstrap (subconscious backend)
    bootstrap = await ReflectionBootstrap.create_and_start()

    # Step 2: Create subconscious-wrapped LLM
    llm = create_subconscious_llm(
        llm_callback=openai_chat,
        user_id="user-123",
        bootstrap=bootstrap,
        model="qwen/qwen3.5-397b-a17b",
    )

    # Step 3: Call LLM - subconscious context is automatically injected
    conversation_context = """
    User: I've been feeling anxious about work lately.
    Therapist: What specifically triggers the anxiety?
    User: Deadlines and my manager's expectations.
    Therapist: How do you typically respond?
    User: I work late and avoid taking breaks.
    """

    response = await llm.complete(
        prompt="Help the user develop a healthier coping strategy.",
        conversation_context=conversation_context,
    )

    print(f"Response: {response}")

    # The subconscious injected context like:
    # <subconscious_context>
    # <relevant_memories>
    # - User mentioned anxiety 3 times this week
    # - Previous session: user responded well to boundary-setting exercise
    # </relevant_memories>
    # <pattern_observations>
    # - Pattern: work avoidance when stressed
    # - Consider: boundary-setting techniques
    # </pattern_observations>
    # </subconscious_context>

    await bootstrap.stop()


if __name__ == "__main__":
    asyncio.run(main())
