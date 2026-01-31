"""
Performance Optimizer for Dataset Pipeline

Implements caching, concurrency, and performance optimizations for dataset
processing operations with intelligent resource management.
"""

import asyncio
import concurrent.futures
import functools
import hashlib
import json
import os
import pickle
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration."""

    enabled: bool = True
    max_size: int = 1000
    ttl_seconds: int = 3600  # 1 hour
    cache_dir: str = "./cache"
    compression: bool = True
    memory_cache: bool = True
    disk_cache: bool = True


@dataclass
class ConcurrencyConfig:
    """Concurrency configuration."""

    max_workers: int = None  # None = auto-detect
    chunk_size: int = 100
    use_async: bool = True
    thread_pool_size: int = None
    process_pool_size: int = None
    batch_processing: bool = True


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""

    total_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_processing_time: float = 0.0
    average_operation_time: float = 0.0
    concurrent_operations: int = 0
    peak_memory_usage: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class MemoryCache:
    """Thread-safe in-memory cache with TTL support."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: dict[str, dict[str, Any]] = {}
        self.access_times: dict[str, datetime] = {}
        self.lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """Get item from cache."""
        with self.lock:
            if key not in self.cache:
                return None

            # Check TTL
            if self._is_expired(key):
                self._remove(key)
                return None

            # Update access time
            self.access_times[key] = datetime.now()
            return self.cache[key]["value"]

    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        with self.lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()

            self.cache[key] = {"value": value, "created": datetime.now()}
            self.access_times[key] = datetime.now()

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.cache:
            return True

        created = self.cache[key]["created"]
        return datetime.now() - created > timedelta(seconds=self.ttl_seconds)

    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.access_times:
            return

        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove(lru_key)

    def _remove(self, key: str) -> None:
        """Remove item from cache."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)


class DiskCache:
    """Persistent disk-based cache."""

    def __init__(self, cache_dir: str = "./cache", compression: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.compression = compression
        self.lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Get item from disk cache."""
        cache_file = self._get_cache_file(key)

        if not cache_file.exists():
            return None

        try:
            with self.lock, open(cache_file, "rb") as f:
                if self.compression:
                    import gzip

                    with gzip.open(f, "rb") as gz:
                        return pickle.load(gz)
                else:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error reading from disk cache: {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        """Set item in disk cache."""
        cache_file = self._get_cache_file(key)

        try:
            with self.lock, open(cache_file, "wb") as f:
                if self.compression:
                    import gzip

                    with gzip.open(f, "wb") as gz:
                        pickle.dump(value, gz)
                else:
                    pickle.dump(value, f)
        except Exception as e:
            logger.error(f"Error writing to disk cache: {e}")

    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key."""
        # Hash the key to create a safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def clear(self) -> None:
        """Clear all disk cache files."""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
        except Exception as e:
            logger.error(f"Error clearing disk cache: {e}")


class PerformanceOptimizer:
    """Main performance optimization system."""

    def __init__(
        self,
        cache_config: CacheConfig | None = None,
        concurrency_config: ConcurrencyConfig | None = None,
    ):
        self.logger = get_logger(__name__)

        # Configuration
        self.cache_config = cache_config or CacheConfig()
        self.concurrency_config = concurrency_config or ConcurrencyConfig()

        # Auto-detect worker counts if not specified
        if self.concurrency_config.max_workers is None:
            self.concurrency_config.max_workers = min(32, (os.cpu_count() or 1) + 4)
        if self.concurrency_config.thread_pool_size is None:
            self.concurrency_config.thread_pool_size = min(
                16, (os.cpu_count() or 1) * 2
            )
        if self.concurrency_config.process_pool_size is None:
            self.concurrency_config.process_pool_size = os.cpu_count() or 1

        # Cache systems
        self.memory_cache = (
            MemoryCache(
                max_size=self.cache_config.max_size,
                ttl_seconds=self.cache_config.ttl_seconds,
            )
            if self.cache_config.memory_cache
            else None
        )

        self.disk_cache = (
            DiskCache(
                cache_dir=self.cache_config.cache_dir,
                compression=self.cache_config.compression,
            )
            if self.cache_config.disk_cache
            else None
        )

        # Thread pools
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.concurrency_config.thread_pool_size
        )
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.concurrency_config.process_pool_size
        )

        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.metrics_lock = threading.Lock()

        logger.info("PerformanceOptimizer initialized")

    def cached(self, ttl: int | None = None, key_func: Callable | None = None):
        """Decorator for caching function results."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.cache_config.enabled:
                    return func(*args, **kwargs)

                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func.__name__, args, kwargs)

                # Try to get from cache
                start_time = time.time()
                cached_result = self._get_from_cache(cache_key)

                if cached_result is not None:
                    self._update_metrics(
                        cache_hit=True, processing_time=time.time() - start_time
                    )
                    return cached_result

                # Execute function
                result = func(*args, **kwargs)
                processing_time = time.time() - start_time

                # Store in cache
                self._set_in_cache(cache_key, result)
                self._update_metrics(cache_hit=False, processing_time=processing_time)

                return result

            return wrapper

        return decorator

    def parallel_map(
        self,
        func: Callable,
        items: list[Any],
        use_processes: bool = False,
        chunk_size: int | None = None,
    ) -> list[Any]:
        """Execute function in parallel across items."""
        if not items:
            return []

        chunk_size = chunk_size or self.concurrency_config.chunk_size

        start_time = time.time()

        try:
            if use_processes:
                # Use process pool for CPU-intensive tasks
                with self.process_pool as executor:
                    if len(items) > chunk_size:
                        # Process in chunks
                        chunks = [
                            items[i : i + chunk_size]
                            for i in range(0, len(items), chunk_size)
                        ]
                        chunk_results = list(
                            executor.map(
                                self._process_chunk, [(func, chunk) for chunk in chunks]
                            )
                        )
                        results = [
                            item
                            for chunk_result in chunk_results
                            for item in chunk_result
                        ]
                    else:
                        results = list(executor.map(func, items))
            else:
                # Use thread pool for I/O-intensive tasks
                with self.thread_pool as executor:
                    results = list(executor.map(func, items))

            processing_time = time.time() - start_time
            self._update_metrics(processing_time=processing_time, operations=len(items))

            return results

        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            # Fallback to sequential processing
            return [func(item) for item in items]

    async def async_map(
        self, func: Callable, items: list[Any], semaphore_limit: int = 10
    ) -> list[Any]:
        """Execute function asynchronously across items."""
        if not items:
            return []

        semaphore = asyncio.Semaphore(semaphore_limit)

        async def bounded_func(item):
            async with semaphore:
                if asyncio.iscoroutinefunction(func):
                    return await func(item)
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self.thread_pool, func, item)

        start_time = time.time()

        try:
            results = await asyncio.gather(*[bounded_func(item) for item in items])
            processing_time = time.time() - start_time
            self._update_metrics(processing_time=processing_time, operations=len(items))
            return results

        except Exception as e:
            logger.error(f"Async processing failed: {e}")
            raise

    def batch_process(
        self, func: Callable, items: list[Any], batch_size: int | None = None
    ) -> list[Any]:
        """Process items in batches for memory efficiency."""
        batch_size = batch_size or self.concurrency_config.chunk_size
        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            batch_results = self.parallel_map(func, batch)
            results.extend(batch_results)

            # Optional: garbage collection between batches
            import gc

            gc.collect()

        return results

    def _process_chunk(self, args: tuple) -> list[Any]:
        """Process a chunk of items (for multiprocessing)."""
        func, chunk = args
        return [func(item) for item in chunk]

    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments."""
        # Create a deterministic string representation
        key_data = {
            "func": func_name,
            "args": str(args),
            "kwargs": str(sorted(kwargs.items())),
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Any | None:
        """Get item from cache (memory first, then disk)."""
        # Try memory cache first
        if self.memory_cache:
            result = self.memory_cache.get(key)
            if result is not None:
                return result

        # Try disk cache
        if self.disk_cache:
            result = self.disk_cache.get(key)
            if result is not None:
                # Store in memory cache for faster access
                if self.memory_cache:
                    self.memory_cache.set(key, result)
                return result

        return None

    def _set_in_cache(self, key: str, value: Any) -> None:
        """Set item in cache (both memory and disk if enabled)."""
        if self.memory_cache:
            self.memory_cache.set(key, value)

        if self.disk_cache:
            self.disk_cache.set(key, value)

    def _update_metrics(
        self, cache_hit: bool = False, processing_time: float = 0.0, operations: int = 1
    ) -> None:
        """Update performance metrics."""
        with self.metrics_lock:
            self.metrics.total_operations += operations
            if cache_hit:
                self.metrics.cache_hits += 1
            else:
                self.metrics.cache_misses += 1

            self.metrics.total_processing_time += processing_time
            if self.metrics.total_operations > 0:
                self.metrics.average_operation_time = (
                    self.metrics.total_processing_time / self.metrics.total_operations
                )

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self.metrics_lock:
            return PerformanceMetrics(
                total_operations=self.metrics.total_operations,
                cache_hits=self.metrics.cache_hits,
                cache_misses=self.metrics.cache_misses,
                total_processing_time=self.metrics.total_processing_time,
                average_operation_time=self.metrics.average_operation_time,
                concurrent_operations=self.metrics.concurrent_operations,
                peak_memory_usage=self.metrics.peak_memory_usage,
                timestamp=datetime.now(),
            )

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats = {}

        if self.memory_cache:
            stats["memory_cache"] = {
                "size": self.memory_cache.size(),
                "max_size": self.memory_cache.max_size,
                "hit_rate": self.metrics.cache_hits
                / max(self.metrics.total_operations, 1),
            }

        if self.disk_cache:
            cache_files = list(self.disk_cache.cache_dir.glob("*.cache"))
            total_size = sum(f.stat().st_size for f in cache_files)
            stats["disk_cache"] = {
                "files": len(cache_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            }

        return stats

    def clear_cache(self) -> None:
        """Clear all caches."""
        if self.memory_cache:
            self.memory_cache.clear()

        if self.disk_cache:
            self.disk_cache.clear()

        logger.info("All caches cleared")

    def shutdown(self) -> None:
        """Shutdown optimizer and clean up resources."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        logger.info("PerformanceOptimizer shutdown complete")


# Utility functions for common optimization patterns
def optimize_dataset_processing(func: Callable) -> Callable:
    """Decorator to optimize dataset processing functions."""
    optimizer = PerformanceOptimizer()

    @optimizer.cached(ttl=3600)  # Cache for 1 hour
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def parallel_dataset_map(
    items: list[Any], func: Callable, max_workers: int | None = None
) -> list[Any]:
    """Utility function for parallel dataset processing."""
    config = ConcurrencyConfig(max_workers=max_workers or os.cpu_count())
    optimizer = PerformanceOptimizer(concurrency_config=config)

    try:
        return optimizer.parallel_map(func, items)
    finally:
        optimizer.shutdown()


# Example usage
if __name__ == "__main__":
    # Example optimization usage
    optimizer = PerformanceOptimizer()

    # Example cached function
    @optimizer.cached(ttl=1800)  # 30 minutes
    def expensive_computation(data):
        time.sleep(0.1)  # Simulate expensive operation
        return len(str(data)) * 2

    # Test caching
    test_data = ["test1", "test2", "test3"] * 100

    # First run (cache miss)
    start = time.time()
    results1 = [expensive_computation(item) for item in test_data]
    time1 = time.time() - start

    # Second run (cache hit)
    start = time.time()
    results2 = [expensive_computation(item) for item in test_data]
    time2 = time.time() - start


    # Test parallel processing
    start = time.time()
    parallel_results = optimizer.parallel_map(expensive_computation, test_data[:10])
    parallel_time = time.time() - start


    # Show metrics
    metrics = optimizer.get_metrics()

    # Show cache stats
    cache_stats = optimizer.get_cache_stats()

    optimizer.shutdown()
