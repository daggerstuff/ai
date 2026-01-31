"""
Advanced Caching and Concurrency System

Enhanced caching system with distributed cache support, intelligent prefetching,
and advanced concurrency patterns for dataset processing.
"""

import asyncio
import hashlib
import json
import pickle
import threading
import time
import weakref
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import redis
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class CacheStats:
    """Cache statistics tracking."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage: int = 0
    last_cleanup: datetime = field(default_factory=datetime.now)


class DistributedCache:
    """Redis-based distributed cache for multi-process scenarios."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "dataset_cache:",
        ttl: int = 3600,
    ):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.ttl = ttl
        self.logger = get_logger(__name__)

        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            self.redis_client.ping()
            self.available = True
            logger.info("Redis distributed cache connected")
        except Exception as e:
            logger.warning(f"Redis not available, using local cache only: {e}")
            self.redis_client = None
            self.available = False

    def get(self, key: str) -> Any | None:
        """Get item from distributed cache."""
        if not self.available:
            return None

        try:
            cache_key = f"{self.key_prefix}{key}"
            data = self.redis_client.get(cache_key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")

        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set item in distributed cache."""
        if not self.available:
            return False

        try:
            cache_key = f"{self.key_prefix}{key}"
            data = pickle.dumps(value)
            ttl = ttl or self.ttl
            return self.redis_client.setex(cache_key, ttl, data)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete item from distributed cache."""
        if not self.available:
            return False

        try:
            cache_key = f"{self.key_prefix}{key}"
            return bool(self.redis_client.delete(cache_key))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache entries with our prefix."""
        if not self.available:
            return False

        try:
            pattern = f"{self.key_prefix}*"
            keys = self.redis_client.keys(pattern)
            if keys:
                return bool(self.redis_client.delete(*keys))
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False


class IntelligentCache:
    """Intelligent cache with prefetching and adaptive sizing."""

    def __init__(
        self, max_size: int = 1000, ttl: int = 3600, prefetch_enabled: bool = True
    ):
        self.max_size = max_size
        self.ttl = ttl
        self.prefetch_enabled = prefetch_enabled

        # Cache storage
        self.cache: dict[str, dict[str, Any]] = {}
        self.access_times: dict[str, datetime] = {}
        self.access_counts: dict[str, int] = {}
        self.prefetch_queue: list[str] = []

        # Statistics
        self.stats = CacheStats()

        # Threading
        self.lock = threading.RLock()
        self.prefetch_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="cache_prefetch"
        )

        # Distributed cache
        self.distributed_cache = DistributedCache()

        logger.info("IntelligentCache initialized")

    def get(self, key: str, loader: Callable | None = None) -> Any | None:
        """Get item with optional loader function."""
        with self.lock:
            # Check local cache first
            if key in self.cache and not self._is_expired(key):
                self._update_access_stats(key)
                self.stats.hits += 1

                # Schedule prefetch of related items
                if self.prefetch_enabled:
                    self._schedule_prefetch(key)

                return self.cache[key]["value"]

            # Check distributed cache
            if self.distributed_cache.available:
                distributed_value = self.distributed_cache.get(key)
                if distributed_value is not None:
                    self._store_local(key, distributed_value)
                    self.stats.hits += 1
                    return distributed_value

            # Cache miss - use loader if provided
            self.stats.misses += 1

            if loader:
                try:
                    value = loader()
                    self.set(key, value)
                    return value
                except Exception as e:
                    logger.error(f"Cache loader failed for key {key}: {e}")

            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set item in cache."""
        with self.lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_items()

            # Store locally
            self._store_local(key, value, ttl)

            # Store in distributed cache
            if self.distributed_cache.available:
                self.distributed_cache.set(key, value, ttl or self.ttl)

    def _store_local(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store item in local cache."""
        self.cache[key] = {
            "value": value,
            "created": datetime.now(),
            "ttl": ttl or self.ttl,
        }
        self.access_times[key] = datetime.now()
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        self.stats.size = len(self.cache)

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.cache:
            return True

        entry = self.cache[key]
        age = datetime.now() - entry["created"]
        return age.total_seconds() > entry["ttl"]

    def _update_access_stats(self, key: str) -> None:
        """Update access statistics."""
        self.access_times[key] = datetime.now()
        self.access_counts[key] = self.access_counts.get(key, 0) + 1

    def _evict_items(self) -> None:
        """Evict items using LRU + frequency strategy."""
        if not self.cache:
            return

        # Calculate eviction scores (lower = more likely to evict)
        scores = {}
        now = datetime.now()

        for key in self.cache:
            if self._is_expired(key):
                scores[key] = -1  # Expired items get evicted first
            else:
                # Combine recency and frequency
                last_access = self.access_times.get(key, now)
                recency_score = (now - last_access).total_seconds()
                frequency_score = 1.0 / max(self.access_counts.get(key, 1), 1)
                scores[key] = recency_score * frequency_score

        # Evict lowest scoring items (25% of cache)
        items_to_evict = max(1, len(self.cache) // 4)
        evict_keys = sorted(scores.keys(), key=lambda k: scores[k])[:items_to_evict]

        for key in evict_keys:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
            self.access_counts.pop(key, None)
            self.stats.evictions += 1

        self.stats.size = len(self.cache)

    def _schedule_prefetch(self, accessed_key: str) -> None:
        """Schedule prefetching of related items."""
        if not self.prefetch_enabled:
            return

        # Simple prefetch strategy: prefetch keys with similar patterns
        related_keys = self._find_related_keys(accessed_key)

        for key in related_keys[:3]:  # Limit prefetch to 3 items
            if key not in self.cache and key not in self.prefetch_queue:
                self.prefetch_queue.append(key)
                self.prefetch_executor.submit(self._prefetch_item, key)

    def _find_related_keys(self, key: str) -> list[str]:
        """Find keys related to the accessed key."""
        # Simple pattern matching - in practice, this could be more sophisticated
        related = []
        key_parts = key.split("_")

        for cache_key in self.cache:
            if cache_key != key:
                cache_parts = cache_key.split("_")
                # Check for common prefixes or patterns
                if any(part in cache_parts for part in key_parts):
                    related.append(cache_key)

        return related

    def _prefetch_item(self, key: str) -> None:
        """Prefetch a single item (placeholder - would need actual loader)."""
        try:
            # In practice, this would call the appropriate loader function
            # For now, just remove from prefetch queue
            if key in self.prefetch_queue:
                self.prefetch_queue.remove(key)
        except Exception as e:
            logger.error(f"Prefetch failed for key {key}: {e}")

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self.lock:
            self.stats.size = len(self.cache)
            return self.stats

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.prefetch_queue.clear()
            self.stats = CacheStats()

            if self.distributed_cache.available:
                self.distributed_cache.clear()


class ConcurrencyManager:
    """Advanced concurrency management for dataset processing."""

    def __init__(self, max_threads: int | None = None, max_processes: int | None = None):
        self.max_threads = max_threads or min(32, (os.cpu_count() or 1) + 4)
        self.max_processes = max_processes or (os.cpu_count() or 1)

        # Executors
        self.thread_executor = ThreadPoolExecutor(
            max_workers=self.max_threads, thread_name_prefix="dataset_thread"
        )
        self.process_executor = ProcessPoolExecutor(max_workers=self.max_processes)

        # Rate limiting
        self.rate_limiters: dict[str, asyncio.Semaphore] = {}

        # Task tracking
        self.active_tasks: weakref.WeakSet = weakref.WeakSet()

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

            # Optional: yield control between batches
            await asyncio.sleep(0)

        return results

    def process_parallel(
        self,
        items: list[Any],
        processor: Callable,
        use_processes: bool = False,
        chunk_size: int | None = None,
    ) -> list[Any]:
        """Process items in parallel with intelligent chunking."""
        if not items:
            return []

        chunk_size = chunk_size or max(1, len(items) // (self.max_threads * 2))
        executor = self.process_executor if use_processes else self.thread_executor

        # Submit tasks
        futures = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i : i + chunk_size]
            if use_processes:
                future = executor.submit(
                    self._process_chunk_multiprocess, processor, chunk
                )
            else:
                future = executor.submit(
                    self._process_chunk_multithread, processor, chunk
                )
            futures.append(future)
            self.active_tasks.add(future)

        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception as e:
                logger.error(f"Parallel processing error: {e}")

        return results

    def _process_chunk_multithread(
        self, processor: Callable, chunk: list[Any]
    ) -> list[Any]:
        """Process chunk in thread pool."""
        return [processor(item) for item in chunk]

    def _process_chunk_multiprocess(
        self, processor: Callable, chunk: list[Any]
    ) -> list[Any]:
        """Process chunk in process pool."""
        # For multiprocessing, processor must be pickleable
        return [processor(item) for item in chunk]

    def get_rate_limiter(self, name: str, limit: int) -> asyncio.Semaphore:
        """Get or create a rate limiter."""
        if name not in self.rate_limiters:
            self.rate_limiters[name] = asyncio.Semaphore(limit)
        return self.rate_limiters[name]

    def shutdown(self) -> None:
        """Shutdown all executors."""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        logger.info("ConcurrencyManager shutdown complete")


class AdvancedCachingSystem:
    """Main advanced caching and concurrency system."""

    def __init__(
        self,
        cache_config: dict | None = None,
        concurrency_config: dict | None = None,
    ):
        self.logger = get_logger(__name__)

        # Initialize components
        cache_config = cache_config or {}
        self.cache = IntelligentCache(
            max_size=cache_config.get("max_size", 1000),
            ttl=cache_config.get("ttl", 3600),
            prefetch_enabled=cache_config.get("prefetch_enabled", True),
        )

        concurrency_config = concurrency_config or {}
        self.concurrency = ConcurrencyManager(
            max_threads=concurrency_config.get("max_threads"),
            max_processes=concurrency_config.get("max_processes"),
        )

        logger.info("AdvancedCachingSystem initialized")

    def cached_processor(
        self, cache_key_func: Callable | None = None, ttl: int | None = None
    ):
        """Decorator for caching processor results."""

        def decorator(func: Callable) -> Callable:
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
        if use_cache:
            # Wrap processor with caching
            cached_processor = self.cached_processor()(processor)
        else:
            cached_processor = processor

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
        """Get comprehensive performance statistics."""
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
                "active_tasks": len(self.concurrency.active_tasks),
            },
            "distributed_cache": {"available": self.cache.distributed_cache.available},
        }

    def clear_all_caches(self) -> None:
        """Clear all caches."""
        self.cache.clear()
        logger.info("All caches cleared")

    def shutdown(self) -> None:
        """Shutdown the caching system."""
        self.concurrency.shutdown()
        logger.info("AdvancedCachingSystem shutdown complete")


# Example usage
if __name__ == "__main__":
    import os

    # Initialize system
    system = AdvancedCachingSystem()

    # Example cached function
    @system.cached_processor(ttl=1800)
    def process_item(item):
        # Simulate processing
        time.sleep(0.01)
        return f"processed_{item}"

    # Test data
    test_items = list(range(100))

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
