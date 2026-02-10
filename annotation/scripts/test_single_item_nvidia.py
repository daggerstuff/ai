import json
import time

from ai.annotation.scripts.multi_agent_system import create_multi_agent_system


def test_single_item_nvidia():
    print("test_single_item_nvidia.py starting...")
    start_time = time.time()
    print("Initializing NVIDIA system...")
    try:
        # Assuming environment variables for NVIDIA are already set in .env
        orchestrator = create_multi_agent_system(model="nvidia/nemotron-3-nano-30b-a3b")
        print("System initialized.")
    except Exception as e:
        print(f"FAILED to initialize: {e}")
        import sys

        sys.exit(1)

    task = {"task_id": "test_nvidia_1", "data": {"text": "I feel really sad and hopeless."}}

    print("Running annotation on test item...")
    try:
        result = orchestrator.annotate_with_consensus(task)
        print("Success!")
        print(json.dumps(result, indent=2))
        print(f"Total time: {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"FAILED during annotation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_single_item_nvidia()
