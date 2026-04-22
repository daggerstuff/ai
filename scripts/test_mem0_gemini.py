import asyncio
import logging
import os

from ai.memory.mem0_gemini.manager import GeminiMem0Config, GeminiMem0Manager

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
    config = GeminiMem0Config(gemini_api_key=gemini_api_key, user_id="empathy_gym_sarah")
    manager = GeminiMem0Manager(config)

    # Step 1: Initial conversation
    q1 = "Hi! I'm Sarah. I prefer a clinical but supportive feedback style during my training."
    await manager.get_response(q1)

    # Step 2: Test recall
    q2 = "What do you know about my background and preferred style?"
    await manager.get_response(q2)

    # Step 3: View stored memories
    memories = manager.get_all_memories(user_id="empathy_gym_sarah")
    for _i, _m in enumerate(memories):
        pass


if __name__ == "__main__":
    asyncio.run(run_cookook_demo())
