#!/usr/bin/env python3
"""
Stress Testing for AI Inference
Tests system behavior under extreme load
"""

import asyncio
import time
import json
from datetime import datetime
from typing import List, Dict
import aiohttp
import argparse
from tqdm import tqdm


class StressTest:
    """
    Stress test the AI inference service
    """
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.results = []
    
    async def ramp_up_test(
        self,
        start_concurrency: int = 1,
        max_concurrency: int = 200,
        step: int = 10,
        requests_per_step: int = 100
    ):
        """
        Gradually increase load to find breaking point
        """
        print(f"\n🔥 Ramp-Up Stress Test")
        print(f"   Start: {start_concurrency} concurrent")
        print(f"   Max: {max_concurrency} concurrent")
        print(f"   Step: {step}")
        print(f"   Requests per step: {requests_per_step}")
        print()
        
        results = []
        
        for concurrency in range(start_concurrency, max_concurrency + 1, step):
            print(f"\n📊 Testing with {concurrency} concurrent requests...")
            
            start_time = time.time()
            successful = 0
            failed = 0
            latencies = []
            
            async with aiohttp.ClientSession() as session:
                semaphore = asyncio.Semaphore(concurrency)
                
                async def make_request():
                    async with semaphore:
                        req_start = time.time()
                        try:
                            async with session.post(
                                f"{self.endpoint}/api/v1/inference",
                                json={
                                    "conversation_context": [],
                                    "user_input": "Hello"
                                },
                                timeout=aiohttp.ClientTimeout(total=10)
                            ) as response:
                                latency = time.time() - req_start
                                if response.status == 200:
                                    return True, latency
                                return False, latency
                        except Exception:
                            return False, time.time() - req_start
                
                tasks = [make_request() for _ in range(requests_per_step)]
                
                for coro in tqdm(
                    asyncio.as_completed(tasks),
                    total=len(tasks),
                    desc=f"Concurrency {concurrency}"
                ):
                    success, latency = await coro
                    if success:
                        successful += 1
                        latencies.append(latency)
                    else:
                        failed += 1
            
            duration = time.time() - start_time
            
            result = {
                "concurrency": concurrency,
                "successful": successful,
                "failed": failed,
                "success_rate": (successful / requests_per_step) * 100,
                "duration": duration,
                "rps": successful / duration,
                "mean_latency": sum(latencies) / len(latencies) if latencies else 0,
                "max_latency": max(latencies) if latencies else 0
            }
            
            results.append(result)
            
            print(f"   Success Rate: {result['success_rate']:.2f}%")
            print(f"   RPS: {result['rps']:.2f}")
            print(f"   Mean Latency: {result['mean_latency']:.3f}s")
            
            # Stop if success rate drops below 90%
            if result['success_rate'] < 90:
                print(f"\n⚠️  Breaking point reached at {concurrency} concurrent requests")
                break
        
        return results
    
    async def spike_test(
        self,
        baseline_concurrency: int = 10,
        spike_concurrency: int = 200,
        spike_duration: int = 30,
        requests_per_second: int = 50
    ):
        """
        Test sudden spike in traffic
        """
        print(f"\n⚡ Spike Test")
        print(f"   Baseline: {baseline_concurrency} concurrent")
        print(f"   Spike: {spike_concurrency} concurrent")
        print(f"   Spike Duration: {spike_duration}s")
        print()
        
        results = {
            "baseline": [],
            "spike": [],
            "recovery": []
        }
        
        async with aiohttp.ClientSession() as session:
            # Baseline phase
            print("📊 Baseline phase (30s)...")
            baseline_start = time.time()
            while time.time() - baseline_start < 30:
                # Make requests at baseline concurrency
                await asyncio.sleep(1 / requests_per_second)
            
            # Spike phase
            print("⚡ Spike phase...")
            spike_start = time.time()
            while time.time() - spike_start < spike_duration:
                # Make requests at spike concurrency
                await asyncio.sleep(1 / (requests_per_second * 4))
            
            # Recovery phase
            print("🔄 Recovery phase (30s)...")
            recovery_start = time.time()
            while time.time() - recovery_start < 30:
                # Back to baseline
                await asyncio.sleep(1 / requests_per_second)
        
        return results
    
    async def endurance_test(
        self,
        duration_minutes: int = 60,
        concurrency: int = 20,
        requests_per_second: int = 50
    ):
        """
        Test system stability over extended period
        """
        print(f"\n⏱️  Endurance Test")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Concurrency: {concurrency}")
        print(f"   Target RPS: {requests_per_second}")
        print()
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        results = []
        interval_results = []
        
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(concurrency)
            
            async def make_request():
                async with semaphore:
                    req_start = time.time()
                    try:
                        async with session.post(
                            f"{self.endpoint}/api/v1/inference",
                            json={
                                "conversation_context": [],
                                "user_input": "Hello"
                            },
                            timeout=aiohttp.ClientTimeout(total=10)
                        ) as response:
                            latency = time.time() - req_start
                            return response.status == 200, latency
                    except Exception:
                        return False, time.time() - req_start
            
            interval_start = time.time()
            interval_successful = 0
            interval_failed = 0
            interval_latencies = []
            
            while time.time() < end_time:
                success, latency = await make_request()
                
                if success:
                    interval_successful += 1
                    interval_latencies.append(latency)
                else:
                    interval_failed += 1
                
                # Report every minute
                if time.time() - interval_start >= 60:
                    elapsed_minutes = (time.time() - start_time) / 60
                    
                    interval_result = {
                        "minute": int(elapsed_minutes),
                        "successful": interval_successful,
                        "failed": interval_failed,
                        "success_rate": (interval_successful / (interval_successful + interval_failed)) * 100 if (interval_successful + interval_failed) > 0 else 0,
                        "mean_latency": sum(interval_latencies) / len(interval_latencies) if interval_latencies else 0
                    }
                    
                    interval_results.append(interval_result)
                    
                    print(f"   Minute {interval_result['minute']}: "
                          f"Success Rate: {interval_result['success_rate']:.2f}%, "
                          f"Mean Latency: {interval_result['mean_latency']:.3f}s")
                    
                    # Reset interval counters
                    interval_start = time.time()
                    interval_successful = 0
                    interval_failed = 0
                    interval_latencies = []
                
                # Rate limiting
                await asyncio.sleep(1 / requests_per_second)
        
        return interval_results
    
    def save_results(self, results: Dict, output_file: str):
        """Save stress test results"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": self.endpoint,
            "results": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")


async def main():
    parser = argparse.ArgumentParser(description="AI Inference Stress Testing")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="API endpoint URL"
    )
    parser.add_argument(
        "--test",
        choices=["ramp-up", "spike", "endurance", "all"],
        default="all",
        help="Test type to run"
    )
    parser.add_argument(
        "--output",
        default="stress_test_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    stress_test = StressTest(args.endpoint)
    results = {}
    
    try:
        if args.test in ["ramp-up", "all"]:
            results["ramp_up"] = await stress_test.ramp_up_test()
        
        if args.test in ["spike", "all"]:
            results["spike"] = await stress_test.spike_test()
        
        if args.test in ["enduranc