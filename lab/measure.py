import asyncio
import time
import json
import tempfile
import os
import sys

# set up path to resolve 'ai.' imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


async def test_provenance():
    # create dummy provenance.json
    data = {
        "dataset_id": "test_ds",
        "dataset_name": "Test Dataset",
        "source": {
            "source_id": "src-123",
            "source_type": "HUGGINGFACE",
            "source_name": "Test Source",
            "source_url": "https://example.com",
        },
        "license": {"license_type": "MIT"},
        "metadata": {"quality_tier": "GOLD", "data_types": ["text"]},
        "timestamps": {"created_at": "2023-10-27T10:00:00Z"},
    }

    # We will measure blocking vs non-blocking by checking how much other tasks are delayed
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        # Create a huge file
        json.dump([data] * 500000, f)
        temp_file = f.name

    try:

        async def background_task(interval=0.01):
            """Task that runs continuously and measures delays between iterations."""
            max_delay = 0
            start = time.time()
            # loop runs roughly 1s if interval=0.01 and 100 iterations
            for _ in range(500):
                await asyncio.sleep(interval)
                now = time.time()
                delay = (now - start) - interval
                max_delay = max(max_delay, delay)
                start = now
            return max_delay

        async def run_baseline():
            with open(temp_file, "r") as f:
                loaded = json.load(f)
            return len(loaded)

        # Run baseline
        print("Running baseline...")
        bg_task = asyncio.create_task(background_task())
        await asyncio.sleep(0.1)  # Let bg task start
        await run_baseline()
        max_delay_baseline = await bg_task
        print(f"Max event loop delay (Baseline): {max_delay_baseline:.4f}s")

        async def run_optimized_to_thread():
            def load_json():
                with open(temp_file, "r") as f:
                    return json.load(f)

            loaded = await asyncio.to_thread(load_json)
            return len(loaded)

        # Run optimized with just to_thread
        print("Running optimized (to_thread)...")
        bg_task3 = asyncio.create_task(background_task())
        await asyncio.sleep(0.1)  # Let bg task start
        await run_optimized_to_thread()
        max_delay_optimized_thread = await bg_task3
        print(
            f"Max event loop delay (Optimized to_thread): {max_delay_optimized_thread:.4f}s"
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        os.remove(temp_file)


asyncio.run(test_provenance())
