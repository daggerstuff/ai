import asyncio
import logging
import os

from ai.core.memory.mem0_gemini.manager import GeminiMem0Config, GeminiMem0Manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mem0_gemini_test")


async def run_cookook_demo():
    """
    Runs a demonstration following the Mem0 + Gemini cookbook.
    """
    logger.info("Starting Mem0 + Gemini Integration Demo")

    # Get configuration from environment
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

    if not gemini_api_key:
        logger.error("GEMINI_API_KEY is missing from environment")
        return

    # Initialize Manager
    config = GeminiMem0Config(
        gemini_api_key=gemini_api_key, user_id="empathy_gym_sarah"
    )
    manager = GeminiMem0Manager(config)

    # Step 1: Initial conversation
    print("\n--- STAGE 1: Information Gathering ---")
    q1 = "Hi! I'm Sarah. I prefer a clinical but supportive feedback style during my training."
    print(f"SARAH: {q1}")
    r1 = await manager.get_response(q1)
    print(f"PIXEL: {r1['response']}")

    # Step 2: Test recall
    print("\n--- STAGE 2: Memory Recall & Continuity ---")
    q2 = "What do you know about my background and preferred style?"
    print(f"SARAH: {q2}")
    r2 = await manager.get_response(q2)
    print(f"PIXEL: {r2['response']}")
    print(
        f"[Stats: Latency {r2['latency_ms']:.2f}ms, Memories retrieved: {r2['memories_used']}]"
    )

    # Step 3: View stored memories
    print("\n--- STAGE 3: Inspection of Stored Memories ---")
    memories = manager.get_all_memories(user_id="empathy_gym_sarah")
    for i, m in enumerate(memories):
        print(f"{i + 1}. {m['content']}")


if __name__ == "__main__":
    asyncio.run(run_cookook_demo())
