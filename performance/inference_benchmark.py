#!/usr/bin/env python3
"""
AI Inference Performance Benchmark
Validates <2s response time SLO and other performance metrics
"""

import asyncio
import time
import statistics
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import aiohttp
import argparse
from tqdm import tqdm
import numpy as np


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    endpoint: str
    num_requests: int = 1000
    concurrency: int = 10
    timeout: float = 10.0
    warmup_requests: int = 10
    test_scenarios: List[str] = None
    
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
    error: Optional[str] = None
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
    errors: Dict[str, int]
    
    # Scenario breakdown
    scenario_results: Dict[str, Dict[str, float]]


class InferenceBenchmark:
    """
    Benchmark AI inference performance
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[RequestResult] = []
        
        # Test scenarios with different complexity
        self.scenarios = {
            "simple": {
                "conversation_context": [],
                "user_input": "Hello, how are you?"
            },
            "medium": {
                "conversation_context": [
                    {"role": "user", "content": "I've been feeling anxious lately."},
                    {"role": "assistant", "content": "I understand. Can you tell me more about what's been making you feel anxious?"}
                ],
                "user_input": "It's mostly work-related stress and deadlines."
            },
            "complex": {
                "conversation_context": [
                    {"role": "user", "content": "I've been struggling with depression for months."},
                    {"role": "assistant", "content": "I'm sorry to hear that. Depression can be very challenging. Can you describe what you've been experiencing?"},
                    {"role": "user", "content": "I feel tired all the time, have trouble sleeping, and lost interest in things I used to enjoy."},
                    {"role": "assistant", "content": "Those are common symptoms of depression. Have you been able to talk to anyone about this?"},
                    {"role": "user", "content": "Not really, I feel like nobody understands."}
                ],
                "user_input": "I'm worried I'll never feel better. What should I do?"
            }
        }
    
    async def make_request(
        self,
        session: aiohttp.ClientSession,
        scenario: str
    ) -> RequestResult:
        """Make a single inference request"""
        start_time = time.time()
        
        try:
            payload = self.scenarios[scenario]
            
            async with session.post(
                f"{self.config.endpoint}/api/v1/inference",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                response_data = await response.text()
                response_time = time.time() - start_time
                
                return RequestResult(
                    scenario=scenario,
                    response_time=response_time,
                    status_code=response.status,
                    success=response.status == 200,
                    response_size=len(response_data),
                    timestamp=start_time
                )
        
        except asyncio.TimeoutError:
            return RequestResult(
                scenario=scenario,
                response_time=time.time() - start_time,
                status_code=0,
                success=False,
                error="Timeout",
                timestamp=start_time
            )
        
        except Exception as e:
            return RequestResult(
                scenario=scenario,
                response_time=time.time() - start_time,
                status_code=0,
                success=False,
                error=str(e),
                timestamp=start_time
            )
    
    async def warmup(self, session: aiohttp.ClientSession):
        """Warmup requests to initialize caches"""
        print(f"🔥 Warming up with {self.config.warmup_requests} requests...")
        
        tasks = []
        for i in range(self.config.warmup_requests):
            scenario = self.config.test_scenarios[i % len(self.config.test_scenarios)]
            tasks.append(self.make_request(session, scenario))
        
        await asyncio.gather(*tasks)
        print("✅ Warmup complete")
    
    async def run_benchmark(self):
        """Run the benchmark"""
        print(f"\n🚀 Starting benchmark")
        print(f"   Endpoint: {self.config.endpoint}")
        print(f"   Requests: {self.config.num_requests}")
        print(f"   Concurrency: {self.config.concurrency}")
        print(f"   Scenarios: {', '.join(self.config.test_scenarios)}")
        print()
        
        async with aiohttp.ClientSession() as session:
            # Warmup
            await self.warmup(session)
            
            # Benchmark
            print(f"\n📊 Running benchmark...")
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
                desc="Requests"
            ):
                result = await coro
                results.append(result)
            
            total_duration = time.time() - start_time
            
            self.results = results
            
            # Analyze results
            return self.analyze_results(total_duration)
    
    def analyze_results(self, total_duration: float) -> BenchmarkResults:
        """Analyze benchmark results"""
        print("\n📈 Analyzing results...")
        
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
                    "p99": np.percentile(scenario_latencies, 99)
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
            scenario_results=scenario_results
        )
    
    def print_results(self, results: BenchmarkResults):
        """Print formatted results"""
        print("\n" + "=" * 70)
        print("📊 BENCHMARK RESULTS")
        print("=" * 70)
        
        # Summary
        print(f"\n📈 Summary:")
        print(f"   Total Requests:      {results.total_requests:,}")
        print(f"   Successful:          {results.successful_requests:,} ({results.success_rate:.2f}%)")
        print(f"   Failed:              {results.failed_requests:,} ({results.error_rate:.2f}%)")
        print(f"   Duration:            {results.total_duration:.2f}s")
        print(f"   Throughput:          {results.requests_per_second:.2f} req/s")
        
        # Latency
        print(f"\n⏱️  Latency (seconds):")
        print(f"   Min:                 {results.min_latency:.3f}s")
        print(f"   Mean:                {results.mean_latency:.3f}s")
        print(f"   Median:              {results.median_latency:.3f}s")
        print(f"   P50:                 {results.p50_latency:.3f}s")
        print(f"   P95:                 {results.p95_latency:.3f}s")
        print(f"   P99:                 {results.p99_latency:.3f}s")
        print(f"   Max:                 {results.max_latency:.3f}s")
        
        # SLO Compliance
        print(f"\n🎯 SLO Compliance:")
        slo_2s_status = "✅ PASS" if results.slo_2s_compliance >= 95 else "❌ FAIL"
        slo_3s_status = "✅ PASS" if results.slo_3s_compliance >= 99 else "❌ FAIL"
        
        print(f"   P95 < 2s:            {results.slo_2s_compliance:.2f}% {slo_2s_status}")
        print(f"   P99 < 3s:            {results.slo_3s_compliance:.2f}% {slo_3s_status}")
        
        # Scenario breakdown
        if results.scenario_results:
            print(f"\n📋 Scenario Breakdown:")
            for scenario, metrics in results.scenario_results.items():
                print(f"\n   {scenario.upper()}:")
                print(f"      Requests:         {metrics['count']:,}")
                print(f"      Mean:             {metrics['mean']:.3f}s")
                print(f"      P50:              {metrics['p50']:.3f}s")
                print(f"      P95:              {metrics['p95']:.3f}s")
                print(f"      P99:              {metrics['p99']:.3f}s")
        
        # Errors
        if results.errors:
            print(f"\n❌ Errors:")
            for error, count in sorted(results.errors.items(), key=lambda x: x[1], reverse=True):
                print(f"   {error}: {count}")
        
        # Overall assessment
        print(f"\n{'=' * 70}")
        
        overall_pass = (
            results.success_rate >= 99.0 and
            results.p95_latency < 2.0 and
            results.p99_latency < 3.0
        )
        
        if overall_pass:
            print("✅ OVERALL: PASS - All SLOs met!")
        else:
            print("❌ OVERALL: FAIL - Some SLOs not met")
            
            if results.success_rate < 99.0:
                print(f"   ⚠️  Success rate {results.success_rate:.2f}% < 99%")
            if results.p95_latency >= 2.0:
                print(f"   ⚠️  P95 latency {results.p95_latency:.3f}s >= 2s")
            if results.p99_latency >= 3.0:
                print(f"   ⚠️  P99 latency {results.p99_latency:.3f}s >= 3s")
        
        print("=" * 70)
    
    def save_results(self, results: BenchmarkResults, output_file: str):
        """Save results to JSON file"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "results": asdict(results),
            "raw_results": [asdict(r) for r in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")


async def main():
    parser = argparse.ArgumentParser(description="AI Inference Performance Benchmark")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="API endpoint URL"
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=1000,
        help="Number of requests to make"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup requests"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["simple", "medium", "complex"],
        help="Test scenarios to run"
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = BenchmarkConfig(
        endpoint=args.endpoint,
        num_requests=args.requests,
        concurrency=args.concurrency,
        timeout=args.timeout,
        warmup_requests=args.warmup,
        test_scenarios=args.scenarios
    )
    
    # Run benchmark
    benchmark = InferenceBenchmark(config)
    
    try:
        results = await benchmark.run_benchmark()
        benchmark.print_results(results)
        benchmark.save_results(results, args.output)
        
        # Exit with appropriate code
        if results.success_rate >= 99.0 and results.p95_latency < 2.0:
            exit(0)
        else:
            exit(1)
    
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
