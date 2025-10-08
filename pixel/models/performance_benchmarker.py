"""
performance_benchmarker.py

Comprehensive Performance Benchmarking System for Pixel Model Components.
Measures latency, throughput, and memory usage across EQ heads, persona classification, clinical heads, and empathy measurement.
"""

import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional


class PerformanceBenchmarker:
    """
    Performance benchmarking for Pixel model components.

    Features:
    - Latency measurement for model components and forward passes.
    - Throughput calculation for batch processing.
    - Memory usage profiling using tracemalloc.
    """

    def __init__(self, model: Any):
        """
        Initialize the benchmarker with a model instance.

        Args:
            model: The Pixel model or component to benchmark.
        """
        self.model = model

    def measure_latency(
        self, input_data: Any, component: Optional[Callable] = None, runs: int = 10
    ) -> float:
        """
        Measures average latency (in seconds) for a model component or forward pass.

        Args:
            input_data: Input data for the model/component.
            component: Optional specific component (callable) to benchmark. If None, uses model's forward.
            runs: Number of runs to average.

        Returns:
            Average latency in seconds.
        """
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            if component:
                component(input_data)
            else:
                self.model(input_data)
            end = time.perf_counter()
            times.append(end - start)
        return sum(times) / len(times)

    def measure_throughput(
        self, batch_data: List[Any], component: Optional[Callable] = None
    ) -> float:
        """
        Measures throughput (samples per second) for batch processing.

        Args:
            batch_data: List of input samples.
            component: Optional specific component (callable) to benchmark.

        Returns:
            Throughput in samples per second.
        """
        start = time.perf_counter()
        if component:
            for sample in batch_data:
                component(sample)
        else:
            for sample in batch_data:
                self.model(sample)
        end = time.perf_counter()
        elapsed = end - start
        return len(batch_data) / elapsed if elapsed > 0 else 0.0

    def profile_memory(
        self, input_data: Any, component: Optional[Callable] = None
    ) -> Dict[str, int]:
        """
        Profiles memory usage (in bytes) for a model component or forward pass.

        Args:
            input_data: Input data for the model/component.
            component: Optional specific component (callable) to profile.

        Returns:
            Dict with peak and current memory usage in bytes.
        """
        tracemalloc.start()
        if component:
            component(input_data)
        else:
            self.model(input_data)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {"current_bytes": current, "peak_bytes": peak}
