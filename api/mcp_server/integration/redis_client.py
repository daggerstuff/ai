import json
import logging
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as redis


class MCPRedisClient:
    """
    Redis client for MCP server event bus and caching.
    Uses redis-py's asyncio support.
    """

    def __init__(self, url: str, db: int = 0, password: Optional[str] = None):
        self.url = url
        self.db = db
        self.password = password
        self.logger = logging.getLogger(__name__)
        self.redis = redis.from_url(
            url, db=db, password=password, decode_responses=True
        )
        self.logger.info(f"MCP Redis client connected to {url} (DB {db})")

    async def get(self, key: str) -> Optional[str]:
        """Get a value from cache."""
        return await self.redis.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set a value in cache with optional expiry."""
        return await self.redis.set(key, value, ex=ex)

    async def publish(
        self, channel: str, message: Union[str, Dict[str, Any], List[Any]]
    ) -> int:
        """Publish a message to an event bus channel with auto-serialization."""
        if isinstance(message, (dict, list)):
            message = json.dumps(message)
        return await self.redis.publish(channel, str(message))

    async def close(self):
        """Close the Redis connection."""
        await self.redis.close()
