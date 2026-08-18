#!/usr/bin/env python3
"""
AI Inference Performance Benchmark
Validates <2s response time SLO and other performance metrics
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import aiohttp
import numpy as np
from tqdm import tqdm


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""

    endpoint: str
    num_requests: int = 1000
    concurrency: int = 10
    timeout: float = 10.0
    warmup_requests: int = 10

    test_scenarios: list[str] = None

    def __post_init__(self):
        if self.test_scenarios is None:
            self.test_scenarios = ["simple", "medium", "complex"]


@dataclass
class RequestResult:
    """Individual request result"""

    scenario: str
    response_time: float
    status_code: int
    success: bool
    error: str | None = None
    response_size: int = 0
    timestamp: float = 0


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results"""

    total_requests: int
    successful_requests: int
    failed_requests: int

    # Latency metrics (seconds)
    min_latency: float
    max_latency: float
    mean_latency: float
    median_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float

    # Throughput
    requests_per_second: float
    total_duration: float

    # Success rate
    success_rate: float
    error_rate: float

    # SLO compliance
    slo_2s_compliance: float  # % of requests < 2s
    slo_3s_compliance: float  # % of requests < 3s

    # Errors
    errors: dict[str, int]

    # Scenario breakdown
    scenario_results: dict[str, dict[str, float]]


class InferenceBenchmark:
    """
    Benchmark AI inference performance
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: list[RequestResult] = []

        # Test scenarios with different complexity
        self.scenarios = {
            "simple": {"conversation_context": [], "user_input": "Hello, how are you?"},
            "medium": {
                "conversation_context": [
                    {"role": "user", "content": "I've been feeling anxious lately."},
                    {
                        "role": "assistant",
                        "content": "I understand. Can you tell me more about what's been making you feel anxious?",
                    },
                ],
                "user_input": "It's mostly work-related stress and deadlines.",
            },
            "complex": {
                "conversation_context": [
                    {
                        "role": "user",
                        "content": "I've been struggling with depression for months.",
                    },
                    {
                        "role": "assistant",
                        "content": "I'm sorry to hear that. Depression can be very challenging. Can you describe what you've been experiencing?",
                    },
                    {
                        "role": "user",
                        "content": "I feel tired all the time, have trouble sleeping, and lost interest in things I used to enjoy.",
                    },
                    {
                        "role": "assistant",
                        "content": "Those are common symptoms of depression. Have you been able to talk to anyone about this?",
                    },
                    {
                        "role": "user",
                        "content": "Not really, I feel like nobody understands.",
                    },
                ],
                "user_input": "I'm worried I'll never feel better. What should I do?",
            },
        }

    async def make_request(self, session: aiohttp.ClientSession, scenario: str) -> RequestResult:
        """Make a single inference request"""
        start_time = time.time()

        try:
            payload = self.scenarios[scenario]

            async with session.post(
                f"{self.config.endpoint}/api/v1/inference",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                response_data = await response.text()
                response_time = time.time() - start_time

                return RequestResult(
                    scenario=scenario,
                    response_time=response_time,
                    status_code=response.status,
                    success=response.status == 200,
                    response_size=len(response_data),
                    timestamp=start_time,
                )

        except TimeoutError:
            return RequestResult(
                scenario=scenario,
                response_time=time.time() - start_time,
                status_code=0,
                success=False,
                error="Timeout",
                timestamp=start_time,
            )

        except Exception as e:
            return RequestResult(
                scenario=scenario,
                response_time=time.time() - start_time,
                status_code=0,
                success=False,
                error=str(e),
                timestamp=start_time,
            )

    async def warmup(self, session: aiohttp.ClientSession):
        """Warmup requests to initialize caches"""

        tasks = []
        for i in range(self.config.warmup_requests):
            scenario = self.config.test_scenarios[i % len(self.config.test_scenarios)]
            tasks.append(self.make_request(session, scenario))

        await asyncio.gather(*tasks)

    async def run_benchmark(self):
        """Run the benchmark"""

        async with aiohttp.ClientSession() as session:
            # Warmup
            await self.warmup(session)

            # Benchmark
            start_time = time.time()

            # Create request tasks
            tasks = []
            for i in range(self.config.num_requests):
                scenario = self.config.test_scenarios[i % len(self.config.test_scenarios)]
                tasks.append(self.make_request(session, scenario))

            # Execute with concurrency limit
            semaphore = asyncio.Semaphore(self.config.concurrency)

            async def bounded_request(task):
                async with semaphore:
                    return await task

            # Run with progress bar
            results = []
            for coro in tqdm(
                asyncio.as_completed([bounded_request(task) for task in tasks]),
                total=len(tasks),
                desc="Requests",
            ):
                result = await coro
                results.append(result)

            total_duration = time.time() - start_time

            self.results = results

            # Analyze results
            return self.analyze_results(total_duration)

    def analyze_results(self, total_duration: float) -> BenchmarkResults:
        """Analyze benchmark results"""

        # Filter successful requests
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        if not successful:
            raise ValueError("No successful requests!")

        # Extract latencies
        latencies = [r.response_time for r in successful]
        latencies.sort()

        # Calculate percentiles
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        # SLO compliance
        under_2s = sum(1 for l in latencies if l < 2.0)
        under_3s = sum(1 for l in latencies if l < 3.0)

        slo_2s_compliance = (under_2s / len(latencies)) * 100
        slo_3s_compliance = (under_3s / len(latencies)) * 100

        # Error analysis
        errors = {}
        for r in failed:
            error_key = r.error or f"HTTP {r.status_code}"
            errors[error_key] = errors.get(error_key, 0) + 1

        # Scenario breakdown
        scenario_results = {}
        for scenario in self.config.test_scenarios:
            scenario_requests = [r for r in successful if r.scenario == scenario]
            if scenario_requests:
                scenario_latencies = [r.response_time for r in scenario_requests]
                scenario_results[scenario] = {
                    "count": len(scenario_requests),
                    "mean": statistics.mean(scenario_latencies),
                    "p50": np.percentile(scenario_latencies, 50),
                    "p95": np.percentile(scenario_latencies, 95),
                    "p99": np.percentile(scenario_latencies, 99),
                }

        return BenchmarkResults(
            total_requests=len(self.results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            min_latency=min(latencies),
            max_latency=max(latencies),
            mean_latency=statistics.mean(latencies),
            median_latency=statistics.median(latencies),
            p50_latency=p50,
            p95_latency=p95,
            p99_latency=p99,
            requests_per_second=len(successful) / total_duration,
            total_duration=total_duration,
            success_rate=(len(successful) / len(self.results)) * 100,
            error_rate=(len(failed) / len(self.results)) * 100,
            slo_2s_compliance=slo_2s_compliance,
            slo_3s_compliance=slo_3s_compliance,
            errors=errors,
            scenario_results=scenario_results,
        )

    def print_results(self, results: BenchmarkResults):
        """Print formatted results"""

        # Summary

        # Latency

        # SLO Compliance

        # Scenario breakdown
        if results.scenario_results:
            for _scenario, _metrics in results.scenario_results.items():
                pass

        # Errors
        if results.errors:
            for _error, _count in sorted(results.errors.items(), key=lambda x: x[1], reverse=True):
                pass

        # Overall assessment

        overall_pass = results.success_rate >= 99.0 and results.p95_latency < 2.0 and results.p99_latency < 3.0

        if overall_pass:
            pass
        else:
            if results.success_rate < 99.0:
                pass
            if results.p95_latency >= 2.0:
                pass
            if results.p99_latency >= 3.0:
                pass

    def save_results(self, results: BenchmarkResults, output_file: str):
        """Save results to JSON file"""
        output = {
            "timestamp": datetime.now(UTC).isoformat(),
            "config": asdict(self.config),
            "results": asdict(results),
            "raw_results": [asdict(r) for r in self.results],
        }

        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)


async def main():
    parser = argparse.ArgumentParser(description="AI Inference Performance Benchmark")
    parser.add_argument("--endpoint", default="http://localhost:8000", help="API endpoint URL")
    parser.add_argument("--requests", type=int, default=1000, help="Number of requests to make")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup requests")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["simple", "medium", "complex"],
        help="Test scenarios to run",
    )
    parser.add_argument("--output", default="benchmark_results.json", help="Output file for results")

    args = parser.parse_args()

    # Create config
    config = BenchmarkConfig(
        endpoint=args.endpoint,
        num_requests=args.requests,
        concurrency=args.concurrency,
        timeout=args.timeout,
        warmup_requests=args.warmup,
        test_scenarios=args.scenarios,
    )

    # Run benchmark
    benchmark = InferenceBenchmark(config)

    try:
        results = await benchmark.run_benchmark()
        benchmark.print_results(results)
        benchmark.save_results(results, args.output)

        # Exit with appropriate code
        if results.success_rate >= 99.0 and results.p95_latency < 2.0:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
