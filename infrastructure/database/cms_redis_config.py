"""CMS Business Strategy Redis key structure and configuration.

Defines key patterns, TTLs, and helper functions for Redis operations
used by the CMS business strategy system. Designed to work alongside
the existing RedisClient from techdeck_integration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RedisNamespace(Enum):
    DOC = "doc"
    SESSION = "session"
    COLLAB = "collab"
    FEATURE = "feature"
    RATELIMIT = "ratelimit"
    STATS = "stats"
    NOTIFICATION = "notifications"
    LOCK = "lock"


@dataclass(frozen=True)
class RedisKeyDefinition:
    pattern: str
    ttl_seconds: int
    description: str


# ============================================================================
# KEY DEFINITIONS
# ============================================================================

KEY_DEFINITIONS: dict[str, RedisKeyDefinition] = {
    "document_cache": RedisKeyDefinition(
        pattern="doc:{documentId}",
        ttl_seconds=3600,
        description="Cached document JSON for fast reads",
    ),
    "user_session": RedisKeyDefinition(
        pattern="session:{sessionId}",
        ttl_seconds=86400,
        description="User session data (24h TTL)",
    ),
    "collab_users": RedisKeyDefinition(
        pattern="collab:{documentId}:users",
        ttl_seconds=0,
        description="Set of user IDs actively editing a document (no TTL, cleanup on disconnect)",
    ),
    "collab_cursors": RedisKeyDefinition(
        pattern="collab:{documentId}:cursors",
        ttl_seconds=0,
        description="Cursor positions per user for real-time collaboration",
    ),
    "collab_changes": RedisKeyDefinition(
        pattern="collab:{documentId}:changes",
        ttl_seconds=0,
        description="Pending changes queue for collaborative editing",
    ),
    "feature_flag": RedisKeyDefinition(
        pattern="feature:{featureName}",
        ttl_seconds=0,
        description="Feature flag boolean value (no TTL, manual toggle)",
    ),
    "rate_limit": RedisKeyDefinition(
        pattern="ratelimit:{userId}:{action}",
        ttl_seconds=60,
        description="Rate limit counter per user per action (1-minute window)",
    ),
    "stats_analytics": RedisKeyDefinition(
        pattern="stats:{type}:{period}",
        ttl_seconds=3600,
        description="Aggregated analytics/metrics cache (1h TTL)",
    ),
    "notification_queue": RedisKeyDefinition(
        pattern="notifications:{userId}",
        ttl_seconds=0,
        description="Queue of pending notifications for a user",
    ),
    "document_lock": RedisKeyDefinition(
        pattern="lock:{documentId}:{userId}",
        ttl_seconds=1800,
        description="Mutex lock for concurrent document edits (30min TTL as safety net)",
    ),
}


# ============================================================================
# DEFAULT TTL CONFIGURATION
# ============================================================================

DEFAULT_TTLS: dict[RedisNamespace, int] = {
    RedisNamespace.DOC: 3600,
    RedisNamespace.SESSION: 86400,
    RedisNamespace.COLLAB: 0,
    RedisNamespace.FEATURE: 0,
    RedisNamespace.RATELIMIT: 60,
    RedisNamespace.STATS: 3600,
    RedisNamespace.NOTIFICATION: 0,
    RedisNamespace.LOCK: 1800,
}


# ============================================================================
# KEY BUILDER
# ============================================================================


class CMSRedisKeys:
    """Type-safe Redis key builder for CMS operations.

    Produces keys matching the defined patterns, ensuring consistent
    namespace usage and correct TTL application.
    """

    @staticmethod
    def document_cache(document_id: str) -> str:
        return f"doc:{document_id}"

    @staticmethod
    def user_session(session_id: str) -> str:
        return f"session:{session_id}"

    @staticmethod
    def collab_users(document_id: str) -> str:
        return f"collab:{document_id}:users"

    @staticmethod
    def collab_cursors(document_id: str) -> str:
        return f"collab:{document_id}:cursors"

    @staticmethod
    def collab_changes(document_id: str) -> str:
        return f"collab:{document_id}:changes"

    @staticmethod
    def feature_flag(feature_name: str) -> str:
        return f"feature:{feature_name}"

    @staticmethod
    def rate_limit(user_id: str, action: str) -> str:
        return f"ratelimit:{user_id}:{action}"

    @staticmethod
    def stats(type_key: str, period: str) -> str:
        return f"stats:{type_key}:{period}"

    @staticmethod
    def notification_queue(user_id: str) -> str:
        return f"notifications:{user_id}"

    @staticmethod
    def document_lock(document_id: str, user_id: str) -> str:
        return f"lock:{document_id}:{user_id}"


# ============================================================================
# TTL LOOKUP
# ============================================================================


def get_ttl(key: str) -> int:
    """Derive TTL from a constructed key by matching its namespace prefix.

    Returns 0 for keys that should not expire (e.g., collab, feature flags).
    """
    namespace_prefix = key.split(":", maxsplit=1)[0]
    try:
        ns = RedisNamespace(namespace_prefix)
    except ValueError:
        return 0
    return DEFAULT_TTLS.get(ns, 0)


# ============================================================================
# COLLABORATION HELPERS
# ============================================================================


@dataclass
class CollaborationState:
    """Snapshot of real-time collaboration state for a document."""

    document_id: str
    active_users: list[str] = field(default_factory=list)
    cursor_positions: dict[str, Any] = field(default_factory=dict)
    pending_changes_count: int = 0


# ============================================================================
# RATE LIMIT CONFIGURATION
# ============================================================================


@dataclass(frozen=True)
class RateLimitRule:
    """Rate limit configuration for a specific action type."""

    action: str
    max_requests: int
    window_seconds: int


RATE_LIMIT_RULES: dict[str, RateLimitRule] = {
    "document_read": RateLimitRule("document_read", 100, 60),
    "document_write": RateLimitRule("document_write", 30, 60),
    "document_export": RateLimitRule("document_export", 10, 60),
    "search": RateLimitRule("search", 50, 60),
    "api_general": RateLimitRule("api_general", 200, 60),
}


# ============================================================================
# CACHE INVALIDATION PATTERNS
# ============================================================================

CACHE_INVALIDATION_PATTERNS: dict[str, list[str]] = {
    "document_updated": [
        "doc:{documentId}",
        "stats:documents:{period}",
    ],
    "document_deleted": [
        "doc:{documentId}",
        "stats:documents:{period}",
    ],
    "project_updated": [
        "stats:projects:{period}",
    ],
    "strategy_updated": [
        "stats:strategies:{period}",
    ],
}


def get_invalidation_keys(event: str, **kwargs: str) -> list[str]:
    """Resolve cache invalidation keys for a given event.

    Replaces placeholders in patterns with provided keyword arguments.
    """
    patterns = CACHE_INVALIDATION_PATTERNS.get(event, [])
    resolved: list[str] = []
    for pattern in patterns:
        key = pattern
        for placeholder, value in kwargs.items():
            key = key.replace(f"{{{placeholder}}}", value)
        resolved.append(key)
    return resolved
