"""
Standardization performance optimization with batch processing.
Provides advanced optimization techniques for high-performance data standardization.
"""

import asyncio
import multiprocessing as mp
import pickle
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from data_standardizer import DataStandardizer, StandardizationResult
from logger import get_logger


@dataclass
class OptimizationConfig:
    """Configuration for standardization optimization."""
    batch_size: int = 1000
    max_workers: int = mp.cpu_count()
    use_multiprocessing: bool = True
    enable_caching: bool = True
    cache_size: int = 10000
    memory_limit_mb: int = 1024
    enable_streaming: bool = True
    chunk_size: int = 100
    prefetch_batches: int = 2


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    total_items: int = 0
    processed_items: int = 0
    processing_time: float = 0.0
    throughput: float = 0.0  # items per second
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    batch_times: list[float] = field(default_factory=list)
    worker_utilization: dict[int, float] = field(default_factory=dict)


class StandardizationOptimizer:
    """
    High-performance standardization optimizer with batch processing.

    Features:
    - Multi-threaded and multi-process batch processing
    - Intelligent caching with LRU eviction
    - Memory-aware processing with streaming
    - Dynamic batch size optimization
    - Performance monitoring and tuning
    - Async processing support
    """

    def __init__(
        self,
        standardizer: DataStandardizer,
        config: OptimizationConfig | None = None
    ):
        """
        Initialize StandardizationOptimizer.

        Args:
            standardizer: DataStandardizer instance to optimize
            config: Optimization configuration
        """
        self.standardizer = standardizer
        self.config = config or OptimizationConfig()
        self.logger = get_logger(__name__)

        # Performance tracking
        self.metrics = PerformanceMetrics()

        # Caching
        self.cache: dict[str, StandardizationResult] = {}
        self.cache_access_times: dict[str, float] = {}

        # Optimization state
        self.optimal_batch_size = self.config.batch_size
        self.performance_history: list[tuple[int, float]] = []  # (batch_size, throughput)

        self.logger.info(f"StandardizationOptimizer initialized with {self.config.max_workers} workers")

    def optimize_batch_processing(
        self,
        data_items: list[Any],
        format_hint: str | None = None,
        source: str | None = None
    ) -> list[StandardizationResult]:
        """
        Optimize batch processing with dynamic performance tuning.

        Args:
            data_items: List of data items to process
            format_hint: Optional format hint for all items
            source: Source identifier

        Returns:
            List of StandardizationResult objects
        """
        start_time = time.time()
        self.metrics.total_items = len(data_items)

        # Determine optimal processing strategy
        if len(data_items) < 100:
            # Small batch - use threading
            results = self._process_with_threading(data_items, format_hint, source)
        elif self.config.use_multiprocessing and len(data_items) > 1000:
            # Large batch - use multiprocessing
            results = self._process_with_multiprocessing(data_items, format_hint, source)
        else:
            # Medium batch - use optimized threading
            results = self._process_with_optimized_threading(data_items, format_hint, source)

        # Update metrics
        processing_time = time.time() - start_time
        self.metrics.processing_time = processing_time
        self.metrics.processed_items = len([r for r in results if r.success])
        self.metrics.throughput = len(results) / processing_time if processing_time > 0 else 0

        # Update optimal batch size based on performance
        self._update_optimal_batch_size(len(data_items), self.metrics.throughput)

        self.logger.info(f"Processed {len(results)} items in {processing_time:.2f}s "
                        f"(throughput: {self.metrics.throughput:.1f} items/s)")

        return results

    def stream_processing(
        self,
        data_iterator: Iterator[Any],
        format_hint: str | None = None,
        source: str | None = None
    ) -> Iterator[StandardizationResult]:
        """
        Stream processing for large datasets with memory optimization.

        Args:
            data_iterator: Iterator over data items
            format_hint: Optional format hint
            source: Source identifier

        Yields:
            StandardizationResult objects
        """
        batch = []
        batch_count = 0

        for item in data_iterator:
            batch.append(item)

            if len(batch) >= self.optimal_batch_size:
                # Process batch
                results = self._process_batch_optimized(batch, format_hint, source)

                for result in results:
                    yield result

                # Clear batch and check memory
                batch.clear()
                batch_count += 1

                if batch_count % 10 == 0:
                    self._check_memory_usage()

        # Process remaining items
        if batch:
            results = self._process_batch_optimized(batch, format_hint, source)
            for result in results:
                yield result

    async def async_batch_processing(
        self,
        data_items: list[Any],
        format_hint: str | None = None,
        source: str | None = None
    ) -> list[StandardizationResult]:
        """
        Asynchronous batch processing for I/O bound operations.

        Args:
            data_items: List of data items to process
            format_hint: Optional format hint
            source: Source identifier

        Returns:
            List of StandardizationResult objects
        """
        # Split into chunks for async processing
        chunk_size = self.config.chunk_size
        chunks = [data_items[i:i + chunk_size] for i in range(0, len(data_items), chunk_size)]

        # Process chunks asynchronously
        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(
                self._async_process_chunk(chunk, format_hint, source)
            )
            tasks.append(task)

        # Wait for all tasks to complete
        chunk_results = await asyncio.gather(*tasks)

        # Flatten results
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)

        return results

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.metrics

    def optimize_configuration(self, sample_data: list[Any]) -> OptimizationConfig:
        """
        Optimize configuration based on sample data analysis.

        Args:
            sample_data: Sample data for analysis

        Returns:
            Optimized configuration
        """
        # Analyze data characteristics
        data_size = len(sample_data)
        avg_item_size = sum(len(str(item)) for item in sample_data) / data_size if data_size > 0 else 0

        # Optimize batch size based on data characteristics
        if avg_item_size < 1000:  # Small items
            optimal_batch_size = min(2000, data_size)
        elif avg_item_size < 10000:  # Medium items
            optimal_batch_size = min(1000, data_size)
        else:  # Large items
            optimal_batch_size = min(500, data_size)

        # Optimize worker count based on CPU and data size
        if data_size < 100:
            optimal_workers = min(2, mp.cpu_count())
        elif data_size < 1000:
            optimal_workers = min(4, mp.cpu_count())
        else:
            optimal_workers = mp.cpu_count()

        # Create optimized configuration
        optimized_config = OptimizationConfig(
            batch_size=optimal_batch_size,
            max_workers=optimal_workers,
            use_multiprocessing=data_size > 1000,
            enable_caching=True,
            cache_size=min(10000, data_size),
            memory_limit_mb=1024,
            enable_streaming=data_size > 5000,
            chunk_size=min(100, optimal_batch_size // 10),
            prefetch_batches=2
        )

        self.logger.info(f"Optimized configuration: batch_size={optimal_batch_size}, "
                        f"workers={optimal_workers}, multiprocessing={optimized_config.use_multiprocessing}")

        return optimized_config

    def clear_cache(self) -> None:
        """Clear the standardization cache."""
        self.cache.clear()
        self.cache_access_times.clear()
        self.logger.info("Cache cleared")

    # Private methods

    def _process_with_threading(
        self,
        data_items: list[Any],
        format_hint: str | None,
        source: str | None
    ) -> list[StandardizationResult]:
        """Process data using threading."""
        results = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(
                    self._process_single_with_cache,
                    item,
                    format_hint,
                    source,
                    i
                ): i
                for i, item in enumerate(data_items)
            }

            # Collect results in order
            results = [None] * len(data_items)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()

        return results

    def _process_with_multiprocessing(
        self,
        data_items: list[Any],
        format_hint: str | None,
        source: str | None
    ) -> list[StandardizationResult]:
        """Process data using multiprocessing."""
        # Split data into chunks for multiprocessing
        chunk_size = max(1, len(data_items) // self.config.max_workers)
        chunks = [data_items[i:i + chunk_size] for i in range(0, len(data_items), chunk_size)]

        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            process_func = partial(
                _process_chunk_worker,
                format_hint=format_hint,
                source=source
            )

            chunk_results = list(executor.map(process_func, chunks))

        # Flatten results
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)

        return results

    def _process_with_optimized_threading(
        self,
        data_items: list[Any],
        format_hint: str | None,
        source: str | None
    ) -> list[StandardizationResult]:
        """Process data with optimized threading strategy."""
        # Use dynamic batch sizing
        batch_size = self.optimal_batch_size
        batches = [data_items[i:i + batch_size] for i in range(0, len(data_items), batch_size)]

        results = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Process batches
            batch_futures = [
                executor.submit(self._process_batch_optimized, batch, format_hint, source)
                for batch in batches
            ]

            for future in as_completed(batch_futures):
                batch_results = future.result()
                results.extend(batch_results)

        return results

    def _process_batch_optimized(
        self,
        batch: list[Any],
        format_hint: str | None,
        source: str | None
    ) -> list[StandardizationResult]:
        """Process a batch with optimization."""
        start_time = time.time()
        results = []

        for i, item in enumerate(batch):
            result = self._process_single_with_cache(item, format_hint, source, i)
            results.append(result)

        # Track batch performance
        batch_time = time.time() - start_time
        self.metrics.batch_times.append(batch_time)

        return results

    def _process_single_with_cache(
        self,
        item: Any,
        format_hint: str | None,
        source: str | None,
        index: int
    ) -> StandardizationResult:
        """Process single item with caching."""
        if not self.config.enable_caching:
            return self.standardizer.standardize_single(item, format_hint, source, f"{source}_{index}")

        # Generate cache key
        cache_key = self._generate_cache_key(item, format_hint)

        # Check cache
        if cache_key in self.cache:
            self.cache_access_times[cache_key] = time.time()
            self.metrics.cache_hit_rate = len(self.cache_access_times) / max(1, self.metrics.processed_items)
            return self.cache[cache_key]

        # Process item
        result = self.standardizer.standardize_single(item, format_hint, source, f"{source}_{index}")

        # Cache result if successful
        if result.success and len(self.cache) < self.config.cache_size:
            self.cache[cache_key] = result
            self.cache_access_times[cache_key] = time.time()
        elif len(self.cache) >= self.config.cache_size:
            # Evict least recently used item
            self._evict_lru_cache_item()
            self.cache[cache_key] = result
            self.cache_access_times[cache_key] = time.time()

        return result

    async def _async_process_chunk(
        self,
        chunk: list[Any],
        format_hint: str | None,
        source: str | None
    ) -> list[StandardizationResult]:
        """Asynchronously process a chunk of data."""
        loop = asyncio.get_event_loop()

        # Run in thread pool to avoid blocking
        with ThreadPoolExecutor(max_workers=2) as executor:
            future = loop.run_in_executor(
                executor,
                self._process_batch_optimized,
                chunk,
                format_hint,
                source
            )
            return await future

    def _generate_cache_key(self, item: Any, format_hint: str | None) -> str:
        """Generate cache key for an item."""
        try:
            # Use pickle to create a hash of the item
            item_bytes = pickle.dumps(item)
            item_hash = hash(item_bytes)
            return f"{format_hint}_{item_hash}"
        except Exception:
            # Fallback to string representation
            return f"{format_hint}_{hash(str(item))}"

    def _evict_lru_cache_item(self) -> None:
        """Evict least recently used cache item."""
        if not self.cache_access_times:
            return

        # Find least recently used item
        lru_key = min(self.cache_access_times.keys(), key=lambda k: self.cache_access_times[k])

        # Remove from cache
        del self.cache[lru_key]
        del self.cache_access_times[lru_key]

    def _update_optimal_batch_size(self, batch_size: int, throughput: float) -> None:
        """Update optimal batch size based on performance."""
        self.performance_history.append((batch_size, throughput))

        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]

        # Find optimal batch size from history
        if len(self.performance_history) >= 3:
            best_batch_size, best_throughput = max(self.performance_history, key=lambda x: x[1])

            # Update optimal batch size if significantly better
            if best_throughput > throughput * 1.1:
                self.optimal_batch_size = best_batch_size
                self.logger.info(f"Updated optimal batch size to {best_batch_size}")

    def _check_memory_usage(self) -> None:
        """Check and manage memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.metrics.memory_usage_mb = memory_mb

            if memory_mb > self.config.memory_limit_mb:
                self.logger.warning(f"Memory usage ({memory_mb:.1f}MB) exceeds limit ({self.config.memory_limit_mb}MB)")
                # Clear cache to free memory
                self.clear_cache()
        except ImportError:
            # psutil not available
            pass


def _process_chunk_worker(
    chunk: list[Any],
    format_hint: str | None = None,
    source: str | None = None
) -> list[StandardizationResult]:
    """Worker function for multiprocessing."""
    # Create a new standardizer instance for this process
    from data_standardizer import DataStandardizer
    standardizer = DataStandardizer()

    results = []
    for i, item in enumerate(chunk):
        try:
            result = standardizer.standardize_single(item, format_hint, source, f"{source}_{i}")
            results.append(result)
        except Exception as e:
            # Create error result
            from data_standardizer import StandardizationResult
            result = StandardizationResult(
                success=False,
                error=str(e),
                source_format=format_hint or "unknown"
            )
            results.append(result)

    return results
