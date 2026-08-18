#!/usr/bin/env python3

"""
Stress Testing for AI Inference
Tests system behavior under extreme load
"""

import argparse
import asyncio
import json
import time

import aiohttp


class StressTest:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.results = []

    async def _make_request(self, session: aiohttp.ClientSession) -> bool:
        try:
            async with session.post(
                self.endpoint, json={"messages": [{"role": "user", "content": "Test"}]}, timeout=10
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def run_batch(self, concurrency: int, duration: int):
        start_time = time.time()
        successful = 0
        failed = 0

        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < duration:
                tasks = [self._make_request(session) for _ in range(concurrency)]
                batch_results = await asyncio.gather(*tasks)
                successful += sum(1 for r in batch_results if r)
                failed += sum(1 for r in batch_results if not r)

        return {"concurrency": concurrency, "successful": successful, "failed": failed, "duration": duration}

    async def ramp_up_test(self):
        for c in [1, 5, 10, 20, 50]:
            res = await self.run_batch(c, 5)
            self.results.append(res)
        return self.results

    async def spike_test(self):
        return await self.run_batch(100, 10)

    async def endurance_test(self):
        return await self.run_batch(10, 60)


async def main():
    parser = argparse.ArgumentParser(description="AI Stress Test")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--test", choices=["ramp-up", "spike", "endurance", "all"], default="all")
    parser.add_argument("--output", default="stress_test_results.json")

    args = parser.parse_args()
    stress_test = StressTest(args.endpoint)
    results = {}

    try:
        if args.test in ["ramp-up", "all"]:
            results["ramp_up"] = await stress_test.ramp_up_test()
        if args.test in ["spike", "all"]:
            results["spike"] = await stress_test.spike_test()
        if args.test in ["endurance", "all"]:
            results["endurance"] = await stress_test.endurance_test()

    except Exception:
        pass
    finally:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
