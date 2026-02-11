"""
Simple Caching and Concurrency System

Simplified caching system without Redis dependency for local caching and concurrency.
"""

import asyncio
import functools
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


@dataclass
class CacheStats:
    """Cache statistics tracking."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    last_cleanup: datetime = field(default_factory=datetime.now)


class LocalCache:
    """Local in-memory cache with TTL support."""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: dict[str, dict[str, Any]] = {}
        self.access_times: dict[str, datetime] = {}
        self.stats = CacheStats()
        self.lock = threading.RLock()
        self.logger = get_logger(__name__)

    def get(self, key: str) -> Any | None:
        """Get item from cache."""
        with self.lock:
            if key not in self.cache:
                self.stats.misses += 1
                return None

            # Check TTL
            if self._is_expired(key):
                self._remove(key)
                self.stats.misses += 1
                return None

            # Update access time
            self.access_times[key] = datetime.now()
            self.stats.hits += 1
            return self.cache[key]["value"]

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set item in cache."""
        with self.lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()

            self.cache[key] = {
                "value": value,
                "created": datetime.now(),
                "ttl": ttl or self.ttl,
            }
            self.access_times[key] = datetime.now()
            self.stats.size = len(self.cache)

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.cache:
            return True

        entry = self.cache[key]
        age = datetime.now() - entry["created"]
        return age.total_seconds() > entry["ttl"]

    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.access_times:
            return

        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove(lru_key)
        self.stats.evictions += 1

    def _remove(self, key: str) -> None:
        """Remove item from cache."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.stats.size = len(self.cache)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.stats = CacheStats()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self.lock:
            self.stats.size = len(self.cache)
            return self.stats


class ConcurrencyManager:
    """Concurrency management for dataset processing."""

    def __init__(self, max_threads: int | None = None, max_processes: int | None = None):
        self.max_threads = max_threads or min(32, (os.cpu_count() or 1) + 4)
        self.max_processes = max_processes or (os.cpu_count() or 1)

        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_threads)
        self.process_executor = ProcessPoolExecutor(max_workers=self.max_processes)

        self.logger = get_logger(__name__)
        logger.info(
            f"ConcurrencyManager initialized: {self.max_threads} threads, {self.max_processes} processes"
        )

    async def process_batch_async(
        self,
        items: list[Any],
        processor: Callable,
        batch_size: int = 100,
        rate_limit: int = 10,
    ) -> list[Any]:
        """Process items in batches with rate limiting."""
        semaphore = asyncio.Semaphore(rate_limit)

        async def process_item(item):
            async with semaphore:
                if asyncio.iscoroutinefunction(processor):
                    return await processor(item)
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.thread_executor, processor, item
                )

        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[process_item(item) for item in batch]
            )
            results.extend(batch_results)

            # Yield control between batches
            await asyncio.sleep(0)

        return results

    def process_parallel(
        self,
        items: list[Any],
        processor: Callable,
        use_processes: bool = False,
        chunk_size: int | None = None,
    ) -> list[Any]:
        """Process items in parallel."""
        if not items:
            return []

        chunk_size = chunk_size or max(1, len(items) // (self.max_threads * 2))
        executor = self.process_executor if use_processes else self.thread_executor

        # Submit tasks
        futures = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i : i + chunk_size]
            future = executor.submit(self._process_chunk, processor, chunk)
            futures.append(future)

        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception as e:
                logger.error(f"Parallel processing error: {e}")

        return results

    def _process_chunk(self, processor: Callable, chunk: list[Any]) -> list[Any]:
        """Process chunk of items."""
        return [processor(item) for item in chunk]

    def shutdown(self) -> None:
        """Shutdown all executors."""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        logger.info("ConcurrencyManager shutdown complete")


class SimpleCachingSystem:
    """Main simple caching and concurrency system."""

    def __init__(
        self,
        cache_config: dict | None = None,
        concurrency_config: dict | None = None,
    ):
        self.logger = get_logger(__name__)

        # Initialize components
        cache_config = cache_config or {}
        self.cache = LocalCache(
            max_size=cache_config.get("max_size", 1000),
            ttl=cache_config.get("ttl", 3600),
        )

        concurrency_config = concurrency_config or {}
        self.concurrency = ConcurrencyManager(
            max_threads=concurrency_config.get("max_threads"),
            max_processes=concurrency_config.get("max_processes"),
        )

        logger.info("SimpleCachingSystem initialized")

    def cached_processor(
        self, cache_key_func: Callable | None = None, ttl: int | None = None
    ):
        """Decorator for caching processor results."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if cache_key_func:
                    cache_key = cache_key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func.__name__, args, kwargs)

                # Try cache first
                result = self.cache.get(cache_key)
                if result is not None:
                    return result

                # Execute function
                result = func(*args, **kwargs)

                # Cache result
                self.cache.set(cache_key, result, ttl)

                return result

            return wrapper

        return decorator

    def process_dataset_parallel(
        self,
        items: list[Any],
        processor: Callable,
        use_cache: bool = True,
        use_processes: bool = False,
        batch_size: int | None = None,
    ) -> list[Any]:
        """Process dataset with caching and parallelization."""
        cached_processor = self.cached_processor()(processor) if use_cache else processor

        return self.concurrency.process_parallel(
            items, cached_processor, use_processes, batch_size
        )

    async def process_dataset_async(
        self,
        items: list[Any],
        processor: Callable,
        use_cache: bool = True,
        rate_limit: int = 10,
        batch_size: int = 100,
    ) -> list[Any]:
        """Process dataset asynchronously with caching."""
        cached_processor = self.cached_processor()(processor) if use_cache else processor

        return await self.concurrency.process_batch_async(
            items, cached_processor, batch_size, rate_limit
        )

    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function and arguments."""
        key_data = {
            "func": func_name,
            "args": str(args),
            "kwargs": str(sorted(kwargs.items())),
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        cache_stats = self.cache.get_stats()

        return {
            "cache": {
                "hits": cache_stats.hits,
                "misses": cache_stats.misses,
                "hit_rate": cache_stats.hits
                / max(cache_stats.hits + cache_stats.misses, 1),
                "size": cache_stats.size,
                "evictions": cache_stats.evictions,
            },
            "concurrency": {
                "max_threads": self.concurrency.max_threads,
                "max_processes": self.concurrency.max_processes,
            },
        }

    def clear_all_caches(self) -> None:
        """Clear all caches."""
        self.cache.clear()
        logger.info("All caches cleared")

    def shutdown(self) -> None:
        """Shutdown the caching system."""
        self.concurrency.shutdown()
        logger.info("SimpleCachingSystem shutdown complete")


# Example usage
if __name__ == "__main__":
    # Initialize system
    system = SimpleCachingSystem()

    # Example cached function
    @system.cached_processor(ttl=1800)
    def process_item(item):
        # Simulate processing
        time.sleep(0.01)
        return f"processed_{item}"

    # Test data
    test_items = list(range(50))

    # Test parallel processing with caching
    start_time = time.time()
    results1 = system.process_dataset_parallel(test_items, process_item)
    time1 = time.time() - start_time

    # Test again (should be faster due to caching)
    start_time = time.time()
    results2 = system.process_dataset_parallel(test_items, process_item)
    time2 = time.time() - start_time


    # Show performance stats
    stats = system.get_performance_stats()

    system.shutdown()
