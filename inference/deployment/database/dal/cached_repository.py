"""Cache-aside layer for CMS DAL operations.

Wraps MongoDB and PostgreSQL repositories with Redis caching
using key patterns from cms_redis_config.py. Handles read-through
caching, write-through invalidation, and TTL management.
"""

import contextlib
import json
import logging
from typing import Any

import redis

from ai.inference.deployment.database.database.cms_redis_config import (
    CMSRedisKeys,
    get_invalidation_keys,
    get_ttl,
)

logger = logging.getLogger(__name__)


def _str_get(client: redis.Redis, key: str) -> str | None:
    """Type-narrowed sync Redis GET — returns str | None instead of ResponseT."""
    raw = client.get(key)
    if raw is None or not isinstance(raw, str):
        return None
    return raw


def _int_incr(client: redis.Redis, key: str) -> int:
    """Type-narrowed sync Redis INCR — returns int instead of ResponseT."""
    result = client.incr(key)
    return result if isinstance(result, int) else 0


def _bool_exists(client: redis.Redis, key: str) -> bool:
    """Type-narrowed sync Redis EXISTS — returns bool instead of ResponseT."""
    result = client.exists(key)
    return bool(result) if isinstance(result, int) else False


def _list_lpop(client: redis.Redis, key: str, count: int) -> list[str]:
    """Type-narrowed sync Redis LPOP — returns list[str] instead of ResponseT."""
    result = client.lpop(key, count)
    if result is None:
        return []
    if isinstance(result, list):
        return [item for item in result if isinstance(item, str)]
    if isinstance(result, str):
        return [result]
    return []


class CMSCacheLayer:
    """Redis-backed cache for CMS read operations.

    Provides get-or-set patterns for single documents and list queries,
    with automatic invalidation on writes.
    """

    def __init__(self, redis_client: redis.Redis):
        self._redis = redis_client

    # ------------------------------------------------------------------
    # DOCUMENT CACHE
    # ------------------------------------------------------------------

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Get a document from cache. Returns None on miss."""
        key = CMSRedisKeys.document_cache(document_id)
        raw = _str_get(self._redis, key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupt cache entry at %s, deleting", key)
            self._redis.delete(key)
            return None

    def set_document(self, document_id: str, data: dict[str, Any]) -> None:
        """Cache a document with the configured TTL."""
        key = CMSRedisKeys.document_cache(document_id)
        ttl = get_ttl(key)
        self._redis.set(key, json.dumps(data, default=str), ex=ttl or None)

    def invalidate_document(self, document_id: str) -> None:
        """Remove a document from cache and related stat caches."""
        keys = get_invalidation_keys("document_updated", documentId=document_id, period="daily")
        if keys:
            self._redis.delete(*keys)

    def delete_document_cache(self, document_id: str) -> None:
        """Remove a document from cache after deletion."""
        keys = get_invalidation_keys("document_deleted", documentId=document_id, period="daily")
        if keys:
            self._redis.delete(*keys)

    # ------------------------------------------------------------------
    # QUERY CACHE (list/search results)
    # ------------------------------------------------------------------

    def get_query(self, cache_key: str) -> list[dict[str, Any]] | None:
        """Get a cached query result list."""
        raw = _str_get(self._redis, cache_key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            self._redis.delete(cache_key)
            return None

    def set_query(self, cache_key: str, results: list[dict[str, Any]], ttl: int = 300) -> None:
        """Cache a query result list with a short TTL (5 min default)."""
        self._redis.set(cache_key, json.dumps(results, default=str), ex=ttl)

    # ------------------------------------------------------------------
    # STATS CACHE
    # ------------------------------------------------------------------

    def get_stats(self, type_key: str, period: str) -> dict[str, Any] | None:
        """Get cached statistics."""
        key = CMSRedisKeys.stats(type_key, period)
        raw = _str_get(self._redis, key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            self._redis.delete(key)
            return None

    def set_stats(self, type_key: str, period: str, data: dict[str, Any]) -> None:
        """Cache statistics with the configured TTL."""
        key = CMSRedisKeys.stats(type_key, period)
        ttl = get_ttl(key)
        self._redis.set(key, json.dumps(data, default=str), ex=ttl or None)

    # ------------------------------------------------------------------
    # FEATURE FLAGS
    # ------------------------------------------------------------------

    def get_feature_flag(self, feature_name: str) -> bool | None:
        """Get a feature flag value. Returns None if not set."""
        key = CMSRedisKeys.feature_flag(feature_name)
        raw = _str_get(self._redis, key)
        if raw is None:
            return None
        return raw.lower() in ("true", "1", "yes")

    def set_feature_flag(self, feature_name: str, enabled: bool) -> None:
        """Set a feature flag."""
        key = CMSRedisKeys.feature_flag(feature_name)
        self._redis.set(key, str(enabled).lower())

    # ------------------------------------------------------------------
    # RATE LIMITING
    # ------------------------------------------------------------------

    def check_rate_limit(self, user_id: str, action: str, limit: int, window: int = 60) -> bool:
        """Check if a rate-limited action is allowed. Returns True if allowed."""
        key = CMSRedisKeys.rate_limit(user_id, action)
        current = _int_incr(self._redis, key)
        if current == 1:
            self._redis.expire(key, window)
        return current <= limit

    # ------------------------------------------------------------------
    # DOCUMENT LOCKING (collaborative editing)
    # ------------------------------------------------------------------

    def acquire_lock(self, document_id: str, user_id: str, ttl: int = 1800) -> bool:
        """Try to acquire an edit lock. Returns True if acquired."""
        key = CMSRedisKeys.document_lock(document_id, user_id)
        return bool(self._redis.set(key, "1", nx=True, ex=ttl))

    def release_lock(self, document_id: str, user_id: str) -> None:
        """Release an edit lock."""
        key = CMSRedisKeys.document_lock(document_id, user_id)
        self._redis.delete(key)

    def is_locked(self, document_id: str, user_id: str) -> bool:
        """Check if a document lock exists for a user."""
        key = CMSRedisKeys.document_lock(document_id, user_id)
        return _bool_exists(self._redis, key)

    # ------------------------------------------------------------------
    # NOTIFICATION QUEUE
    # ------------------------------------------------------------------

    def push_notification(self, user_id: str, notification: dict[str, Any]) -> None:
        """Push a notification to a user's queue."""
        key = CMSRedisKeys.notification_queue(user_id)
        self._redis.rpush(key, json.dumps(notification, default=str))

    def pop_notifications(self, user_id: str, count: int = 10) -> list[dict[str, Any]]:
        """Pop up to `count` notifications from a user's queue."""
        key = CMSRedisKeys.notification_queue(user_id)
        raw_items = _list_lpop(self._redis, key, count)
        if not raw_items:
            return []
        results: list[dict[str, Any]] = []
        for item in raw_items:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                results.append(json.loads(item))
        return results

    # ------------------------------------------------------------------
    # BULK INVALIDATION
    # ------------------------------------------------------------------

    def invalidate_pattern(self, pattern: str) -> int:
        """Delete all keys matching a Redis pattern. Returns count deleted."""
        keys = list(self._redis.scan_iter(match=pattern))
        if not keys:
            return 0
        result = self._redis.delete(*keys)
        return result if isinstance(result, int) else 0

    def flush_cms_cache(self) -> None:
        """Flush all CMS-related cache keys (doc:*, stats:*, collab:*)."""
        for prefix in ("doc:*", "stats:*", "collab:*"):
            self.invalidate_pattern(prefix)
        logger.info("CMS cache flushed")
