import sys
import time
import traceback

from ai.annotation.scripts.multi_agent_system import create_multi_agent_system


def test_single_item_nvidia():
    time.time()
    try:
        # Assuming environment variables for NVIDIA are already set in .env
        orchestrator = create_multi_agent_system(model="nvidia/nemotron-3-nano-30b-a3b")
    except Exception:
        sys.exit(1)

    task = {"task_id": "test_nvidia_1", "data": {"text": "I feel really sad and hopeless."}}

    try:
        orchestrator.annotate_with_consensus(task)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    test_single_item_nvidia()
