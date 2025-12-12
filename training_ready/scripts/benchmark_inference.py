#!/usr/bin/env python3
"""
Benchmark Tool for Inference Performance
Validates <2 second latency target
"""

import time
import asyncio
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
import json
from pathlib import Path

from inference_optimizer import OptimizedInferenceEngine, create_optimized_engine


@dataclass
class BenchmarkResult:
    """Results from benchmark run"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency: float
    median_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    cache_hit_rate: float
    meets_sla: bool
    throughput: float  # requests per second


class InferenceBenchmark:
    """Benchmark inference performance"""
    
    def __init__(self, engine: OptimizedInferenceEngine):
        self.engine = engine
        self.test_prompts = self._load_test_prompts()
    
    def _load_test_prompts(self) -> List[str]:
        """Load test prompts"""
        return [
            "I've been feeling really anxious lately.",
            "Can you help me with my depression?",
            "I'm having trouble sleeping at night.",
            "My relationship is falling apart.",
            "I feel overwhelmed with work stress.",
            "I can't stop worrying about everything.",
            "I'm struggling with low self-esteem.",
            "How do I deal with panic attacks?",
            "I feel disconnected from my emotions.",
            "I'm having intrusive thoughts.",
            "My family doesn't understand me.",
            "I feel like I'm not good enough.",
            "How can I manage my anger better?",
            "I'm dealing with grief and loss.",
            "I feel stuck in my life.",
            "I have trouble setting boundaries.",
            "I'm experiencing burnout.",
            "How do I cope with trauma?",
            "I feel lonely and isolated.",
            "I'm struggling with addiction."
        ]
    
    def run_sequential_benchmark(
        self,
        num_requests: int = 100,
        use_cache: bool = True
    ) -> BenchmarkResult:
        """Run sequential benchmark"""
        print(f"\n🏃 Running sequential benchmark ({num_requests} requests)...")
        
        latencies = []
        successful = 0
        failed = 0
        cache_hits = 0
        
        start_time = time.time()
        
        for i in range(num_requests):
            prompt = self.test_prompts[i % len(self.test_prompts)]
            
            try:
                _, metadata = self.engine.generate(prompt, use_cache=use_cache)
                latencies.append(metadata['latency'])
                successful += 1
                
                if metadata['cache_hit']:
                    cache_hits += 1
                
                if (i + 1) % 10 == 0:
                    print(f"   Progress: {i + 1}/{num_requests}")
            
            except Exception as e:
                print(f"   Error on request {i + 1}: {e}")
                failed += 1
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        result = self._calculate_results(
            latencies, successful, failed, cache_hits, total_time
        )
        
        return result
    
    async def run_concurrent_benchmark(
        self,
        num_requests: int = 100,
        concurrency: int = 10,
        use_cache: bool = True
    ) -> BenchmarkResult:
        """Run concurrent benchmark"""
        print(f"\n🏃 Running concurrent benchmark ({num_requests} requests, {concurrency} concurrent)...")
        
        latencies = []
        successful = 0
        failed = 0
        cache_hits = 0
        
        async def make_request(prompt: str):
            nonlocal successful, failed, cache_hits
            try:
                _, metadata = await self.engine.generate_async(prompt, use_cache=use_cache)
                latencies.append(metadata['latency'])
                successful += 1
                
                if metadata['cache_hit']:
                    cache_hits += 1
            
            except Exception as e:
                failed += 1
        
        # Create tasks
        tasks = []
        for i in range(num_requests):
            prompt = self.test_prompts[i % len(self.test_prompts)]
            tasks.append(make_request(prompt))
        
        # Run with concurrency limit
        start_time = time.time()
        
        for i in range(0, len(tasks), concurrency):
            batch = tasks[i:i + concurrency]
            await asyncio.gather(*batch)
            print(f"   Progress: {min(i + concurrency, num_requests)}/{num_requests}")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        result = self._calculate_results(
            latencies, successful, failed, cache_hits, total_time
        )
        
        return result
    
    def _calculate_results(
        self,
        latencies: List[float],
        successful: int,
        failed: int,
        cache_hits: int,
        total_time: float
    ) -> BenchmarkResult:
        """Calculate benchmark results"""
        if not latencies:
            return BenchmarkResult(
                total_requests=successful + failed,
                successful_requests=successful,
                failed_requests=failed,
                avg_latency=0.0,
                median_latency=0.0,
                p50_latency=0.0,
                p95_latency=0.0,
                p99_latency=0.0,
                min_latency=0.0,
                max_latency=0.0,
                cache_hit_rate=0.0,
                meets_sla=False,
                throughput=0.0
            )
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        return BenchmarkResult(
            total_requests=successful + failed,
            successful_requests=successful,
            failed_requests=failed,
            avg_latency=statistics.mean(latencies),
            median_latency=statistics.median(latencies),
            p50_latency=sorted_latencies[int(n * 0.50)],
            p95_latency=sorted_latencies[int(n * 0.95)],
            p99_latency=sorted_latencies[int(n * 0.99)],
            min_latency=min(latencies),
            max_latency=max(latencies),
            cache_hit_rate=cache_hits / successful if successful > 0 else 0.0,
            meets_sla=sorted_latencies[int(n * 0.95)] < 2.0,
            throughput=successful / total_time
        )
    
    def print_results(self, result: BenchmarkResult, title: str = "Benchmark Results"):
        """Print benchmark results"""
        print(f"\n{'=' * 60}")
        print(f"📊 {title}")
        print(f"{'=' * 60}")
        print(f"Total Requests:      {result.total_requests}")
        print(f"Successful:          {result.successful_requests}")
        print(f"Failed:              {result.failed_requests}")
        print(f"")
        print(f"Latency Statistics:")
        print(f"  Average:           {result.avg_latency:.3f}s")
        print(f"  Median:            {result.median_latency:.3f}s")
        print(f"  P50:               {result.p50_latency:.3f}s")
        print(f"  P95:               {result.p95_latency:.3f}s {'✅' if result.p95_latency < 2.0 else '❌'}")
        print(f"  P99:               {result.p99_latency:.3f}s")
        print(f"  Min:               {result.min_latency:.3f}s")
        print(f"  Max:               {result.max_latency:.3f}s")
        print(f"")
        print(f"Cache Hit Rate:      {result.cache_hit_rate * 100:.1f}%")
        print(f"Throughput:          {result.throughput:.2f} req/s")
        print(f"")
        print(f"SLA Status:          {'✅ PASS' if result.meets_sla else '❌ FAIL'} (P95 < 2.0s)")
        print(f"{'=' * 60}")
    
    def save_results(self, result: BenchmarkResult, filepath: str):
        """Save results to file"""
        data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_requests': result.total_requests,
            'successful_requests': result.successful_requests,
            'failed_requests': result.failed_requests,
            'avg_latency': result.avg_latency,
            'median_latency': result.median_latency,
            'p50_latency': result.p50_latency,
            'p95_latency': result.p95_latency,
            'p99_latency': result.p99_latency,
            'min_latency': result.min_latency,
            'max_latency': result.max_latency,
            'cache_hit_rate': result.cache_hit_rate,
            'meets_sla': result.meets_sla,
            'throughput': result.throughput
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")


async def main():
    """Run benchmarks"""
    print("🚀 Inference Performance Benchmark")
    print("=" * 60)
    
    # Create engine
    print("\n📦 Loading model...")
    engine = create_optimized_engine(
        model_path="./therapeutic_moe_model",
        device="cuda",
        enable_cache=True,
        compile_model=True
    )
    
    # Create benchmark
    benchmark = InferenceBenchmark(engine)
    
    # Run sequential benchmark (no cache)
    print("\n" + "=" * 60)
    print("Test 1: Sequential Requests (No Cache)")
    print("=" * 60)
    result_seq_no_cache = benchmark.run_sequential_benchmark(
        num_requests=50,
        use_cache=False
    )
    benchmark.print_results(result_seq_no_cache, "Sequential (No Cache)")
    benchmark.save_results(result_seq_no_cache, "benchmark_sequential_no_cache.json")
    
    # Run sequential benchmark (with cache)
    print("\n" + "=" * 60)
    print("Test 2: Sequential Requests (With Cache)")
    print("=" * 60)
    result_seq_cache = benchmark.run_sequential_benchmark(
        num_requests=50,
        use_cache=True
    )
    benchmark.print_results(result_seq_cache, "Sequential (With Cache)")
    benchmark.save_results(result_seq_cache, "benchmark_sequential_cache.json")
    
    # Run concurrent benchmark
    print("\n" + "=" * 60)
    print("Test 3: Concurrent Requests")
    print("=" * 60)
    result_concurrent = await benchmark.run_concurrent_benchmark(
        num_requests=50,
        concurrency=5,
        use_cache=True
    )
    benchmark.print_results(result_concurrent, "Concurrent (5 concurrent)")
    benchmark.save_results(result_concurrent, "benchmark_concurrent.json")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"Sequential (No Cache) P95: {result_seq_no_cache.p95_latency:.3f}s")
    print(f"Sequential (Cache)    P95: {result_seq_cache.p95_latency:.3f}s")
    print(f"Concurrent (5x)       P95: {result_concurrent.p95_latency:.3f}s")
    print(f"")
    print(f"Cache Improvement: {((result_seq_no_cache.p95_latency - result_seq_cache.p95_latency) / result_seq_no_cache.p95_latency) * 100:.1f}%")
    print(f"")
    
    all_pass = all([
        result_seq_no_cache.meets_sla,
        result_seq_cache.meets_sla,
        result_concurrent.meets_sla
    ])
    
    if all_pass:
        print("✅ All tests PASSED - Meeting <2s SLA!")
    else:
        print("❌ Some tests FAILED - Not meeting <2s SLA")
    
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
