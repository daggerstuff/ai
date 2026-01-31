"""Ingestion queue and backpressure handling module.

Implements a Redis-backed queue for ingestion with backpressure handling.
Uses asyncio and async queues when Redis is not available.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, AsyncIterator
from enum import Enum

try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    import queue as sync_queue


class QueueType(Enum):
    """Type of queue to use."""
    REDIS = "redis"
    INTERNAL_ASYNC = "internal_async"


@dataclass
class QueueItem:
    """Item in the ingestion queue."""
    id: str
    payload: Any
    metadata: Dict[str, Any]
    priority: int = 0  # Higher number = higher priority
    created_at: float = 0.0
    source_connector: str = ""

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_json(self) -> str:
        """Convert to JSON string."""
        d = asdict(self)
        # Convert non-serializable types
        d['payload'] = str(d['payload']) if d['payload'] is not None else None
        return json.dumps(d)

    @classmethod
    def from_json(cls, json_str: str) -> QueueItem:
        """Create from JSON string."""
        d = json.loads(json_str)
        return cls(**d)


class IngestionQueue:
    """Queue for handling ingestion with backpressure."""

    def __init__(
        self,
        queue_type: QueueType = QueueType.INTERNAL_ASYNC,
        redis_url: Optional[str] = None,
        queue_name: str = "ingestion_queue",
        max_size: int = 10000,
        batch_size: int = 10
    ):
        self.queue_type = queue_type
        self.redis_url = redis_url
        self.queue_name = queue_name
        self.max_size = max_size
        self.batch_size = batch_size
        self.redis_client = None
        self.internal_queue = None
        
        if queue_type == QueueType.REDIS and HAS_REDIS:
            self.redis_client = redis.from_url(redis_url or "redis://localhost:6379")
        else:
            # Use internal asyncio queue
            self.internal_queue = asyncio.Queue(maxsize=max_size)

    async def connect(self) -> None:
        """Connect to the queue system."""
        if self.queue_type == QueueType.REDIS and self.redis_client:
            try:
                await self.redis_client.ping()
            except Exception as e:
                raise ConnectionError(f"Could not connect to Redis: {e}")

    async def enqueue(self, item: QueueItem, timeout: float = 5.0) -> bool:
        """Add an item to the queue with timeout and backpressure handling."""
        if self.queue_type == QueueType.REDIS and self.redis_client:
            # Use Redis with priority queue pattern (sorted set)
            try:
                # Check queue size before adding
                queue_size = await self.redis_client.zcard(self.queue_name)
                if queue_size >= self.max_size:
                    return False  # Backpressure - queue is full
                
                # Use priority as score (higher priority = lower score in Redis)
                priority_score = -item.priority  # Invert so higher priority = earlier processing
                await self.redis_client.zadd(
                    self.queue_name,
                    {item.to_json(): priority_score}
                )
                return True
            except Exception:
                return False
        else:
            # Internal asyncio queue
            try:
                await asyncio.wait_for(self.internal_queue.put(item), timeout=timeout)
                return True
            except asyncio.TimeoutError:
                return False

    async def dequeue_batch(self, timeout: float = 5.0) -> list[QueueItem]:
        """Dequeue a batch of items."""
        items = []
        
        if self.queue_type == QueueType.REDIS and self.redis_client:
            # Get and remove a batch of items from Redis sorted set
            try:
                # Get items from sorted set (by score = priority)
                raw_items = await self.redis_client.zrange(
                    self.queue_name, 0, self.batch_size - 1, withscores=True
                )
                
                if not raw_items:
                    return []
                
                # Convert to QueueItem and remove from queue
                items_to_remove = []
                for raw_json, score in raw_items:
                    try:
                        item = QueueItem.from_json(raw_json.decode('utf-8'))
                        items.append(item)
                        items_to_remove.append(raw_json)
                    except json.JSONDecodeError:
                        continue  # Skip malformed items
                
                # Remove processed items from Redis
                if items_to_remove:
                    await self.redis_client.zrem(self.queue_name, *items_to_remove)
                
                return items
            except Exception:
                return []
        else:
            # Internal queue - get available items up to batch size
            try:
                for _ in range(self.batch_size):
                    try:
                        item = await asyncio.wait_for(
                            self.internal_queue.get(), timeout=0.1
                        )
                        items.append(item)
                    except asyncio.TimeoutError:
                        break  # No more items available
            except Exception:
                pass
            
            return items

    async def get_queue_size(self) -> int:
        """Get current queue size."""
        if self.queue_type == QueueType.REDIS and self.redis_client:
            try:
                return await self.redis_client.zcard(self.queue_name)
            except Exception:
                return 0
        else:
            return self.internal_queue.qsize()

    async def close(self) -> None:
        """Close the queue connection."""
        if self.redis_client:
            await self.redis_client.close()
            await self.redis_client.connection_pool.disconnect()


class BackpressureMonitor:
    """Monitor queue for backpressure and emit alerts."""

    def __init__(self, queue: IngestionQueue, threshold: float = 0.8):
        self.queue = queue
        self.threshold = threshold  # 80% of max_size
        self.alert_callback = None

    def set_alert_callback(self, callback):
        """Set callback for backpressure alerts."""
        self.alert_callback = callback

    async def check_backpressure(self) -> bool:
        """Check if queue is approaching capacity."""
        size = await self.queue.get_queue_size()
        is_backpressured = size >= (self.queue.max_size * self.threshold)
        
        if is_backpressured and self.alert_callback:
            await self.alert_callback(
                f"Backpressure alert: Queue at {size}/{self.queue.max_size} ({size/self.queue.max_size:.1%})"
            )
        
        return is_backpressured


# Global queue instance
_ingestion_queue: Optional[IngestionQueue] = None
_backpressure_monitor: Optional[BackpressureMonitor] = None


async def get_ingestion_queue(
    queue_type: QueueType = QueueType.INTERNAL_ASYNC,
    redis_url: Optional[str] = None,
    max_size: int = 10000
) -> IngestionQueue:
    """Get or create global ingestion queue."""
    global _ingestion_queue
    if _ingestion_queue is None:
        _ingestion_queue = IngestionQueue(
            queue_type=queue_type,
            redis_url=redis_url,
            max_size=max_size
        )
        await _ingestion_queue.connect()
    return _ingestion_queue


async def get_backpressure_monitor() -> BackpressureMonitor:
    """Get or create backpressure monitor."""
    global _backpressure_monitor
    if _backpressure_monitor is None:
        queue = await get_ingestion_queue()
        _backpressure_monitor = BackpressureMonitor(queue)
    return _backpressure_monitor


async def enqueue_item(item: QueueItem) -> bool:
    """Convenience function to add item to global queue."""
    queue = await get_ingestion_queue()
    return await queue.enqueue(item)


async def dequeue_batch() -> list[QueueItem]:
    """Convenience function to get batch from global queue."""
    queue = await get_ingestion_queue()
    return await queue.dequeue_batch()


__all__ = [
    'QueueType',
    'QueueItem',
    'IngestionQueue',
    'BackpressureMonitor',
    'get_ingestion_queue',
    'get_backpressure_monitor',
    'enqueue_item',
    'dequeue_batch'
]