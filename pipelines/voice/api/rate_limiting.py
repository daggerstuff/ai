"""
Rate limiting and YouTube API compliance for Pixel Voice pipeline.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum

import aioredis
from fastapi import HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger(__name__)


class RateLimitType(Enum):
    """Types of rate limits."""

    API_ENDPOINT = "api_endpoint"
    YOUTUBE_DOWNLOAD = "youtube_download"
    PIPELINE_JOB = "pipeline_job"
    USER_QUOTA = "user_quota"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_allowance: int = 10
    cooldown_period: int = 300  # seconds

    # YouTube-specific limits
    youtube_downloads_per_hour: int = 100
    youtube_downloads_per_day: int = 1000
    youtube_concurrent_downloads: int = 5

    # Pipeline-specific limits
    concurrent_jobs_per_user: int = 3
    max_job_duration: int = 3600  # seconds


@dataclass
class UserQuota:
    """User quota tracking."""

    user_id: str
    api_calls_today: int = 0
    youtube_downloads_today: int = 0
    active_jobs: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    is_premium: bool = False

    def reset_daily_quotas(self):
        """Reset daily quotas if needed."""
        now = datetime.now()
        if now.date() > self.last_reset.date():
            self.api_calls_today = 0
            self.youtube_downloads_today = 0
            self.last_reset = now


class YouTubeRateLimiter:
    """YouTube API rate limiter with exponential backoff."""

    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis = redis_client
        self.config = RateLimitConfig()
        self.download_queue = asyncio.Queue(maxsize=100)
        self.active_downloads = 0
        self.download_semaphore = asyncio.Semaphore(self.config.youtube_concurrent_downloads)

    async def can_download(self, user_id: str) -> Tuple[bool, Optional[str]]:
        """Check if user can download from YouTube."""
        if not self.redis:
            return True, None

        # Check hourly limit
        hourly_key = f"youtube_downloads:{user_id}:{datetime.now().hour}"
        hourly_count = await self.redis.get(hourly_key)

        if hourly_count and int(hourly_count) >= self.config.youtube_downloads_per_hour:
            return False, "Hourly YouTube download limit exceeded"

        # Check daily limit
        daily_key = f"youtube_downloads:{user_id}:{datetime.now().date()}"
        daily_count = await self.redis.get(daily_key)

        if daily_count and int(daily_count) >= self.config.youtube_downloads_per_day:
            return False, "Daily YouTube download limit exceeded"

        return True, None

    async def record_download(self, user_id: str):
        """Record a YouTube download."""
        if not self.redis:
            return

        # Increment hourly counter
        hourly_key = f"youtube_downloads:{user_id}:{datetime.now().hour}"
        await self.redis.incr(hourly_key)
        await self.redis.expire(hourly_key, 3600)

        # Increment daily counter
        daily_key = f"youtube_downloads:{user_id}:{datetime.now().date()}"
        await self.redis.incr(daily_key)
        await self.redis.expire(daily_key, 86400)

    async def download_with_backoff(self, url: str, user_id: str, max_retries: int = 3):
        """Download with exponential backoff."""
        async with self.download_semaphore:
            can_download, error_msg = await self.can_download(user_id)
            if not can_download:
                raise HTTPException(status_code=429, detail=error_msg)

            for attempt in range(max_retries):
                try:
                    await self.record_download(user_id)
                    # Actual download logic would go here
                    await asyncio.sleep(0.1)  # Simulate download time
                    return f"Downloaded: {url}"

                except Exception as e:
                    if attempt == max_retries - 1:
                        raise

                    # Exponential backoff
                    wait_time = (2**attempt) + (0.1 * attempt)
                    logger.warning(
                        f"Download attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)


class QuotaManager:
    """Manages user quotas and usage tracking."""

    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis = redis_client
        self.config = RateLimitConfig()
        self.user_quotas: Dict[str, UserQuota] = {}

    async def get_user_quota(self, user_id: str) -> UserQuota:
        """Get user quota from cache or create new."""
        if user_id in self.user_quotas:
            quota = self.user_quotas[user_id]
            quota.reset_daily_quotas()
            return quota

        # Try to load from Redis
        if self.redis:
            quota_data = await self.redis.hgetall(f"user_quota:{user_id}")
            if quota_data:
                quota = UserQuota(
                    user_id=user_id,
                    api_calls_today=int(quota_data.get(b"api_calls_today", 0)),
                    youtube_downloads_today=int(quota_data.get(b"youtube_downloads_today", 0)),
                    active_jobs=int(quota_data.get(b"active_jobs", 0)),
                    is_premium=quota_data.get(b"is_premium", b"false") == b"true",
                )
                quota.reset_daily_quotas()
                self.user_quotas[user_id] = quota
                return quota

        # Create new quota
        quota = UserQuota(user_id=user_id)
        self.user_quotas[user_id] = quota
        await self.save_user_quota(quota)
        return quota

    async def save_user_quota(self, quota: UserQuota):
        """Save user quota to Redis."""
        if not self.redis:
            return

        await self.redis.hset(
            f"user_quota:{quota.user_id}",
            mapping={
                "api_calls_today": quota.api_calls_today,
                "youtube_downloads_today": quota.youtube_downloads_today,
                "active_jobs": quota.active_jobs,
                "last_reset": quota.last_reset.isoformat(),
                "is_premium": str(quota.is_premium).lower(),
            },
        )
        await self.redis.expire(f"user_quota:{quota.user_id}", 86400)

    async def check_quota(self, user_id: str, quota_type: str) -> Tuple[bool, Optional[str]]:
        """Check if user has quota available."""
        quota = await self.get_user_quota(user_id)

        if quota_type == "api_call":
            limit = self.config.requests_per_day * (2 if quota.is_premium else 1)
            if quota.api_calls_today >= limit:
                return False, f"Daily API call limit ({limit}) exceeded"

        elif quota_type == "youtube_download":
            limit = self.config.youtube_downloads_per_day * (2 if quota.is_premium else 1)
            if quota.youtube_downloads_today >= limit:
                return False, f"Daily YouTube download limit ({limit}) exceeded"

        elif quota_type == "concurrent_job":
            limit = self.config.concurrent_jobs_per_user * (2 if quota.is_premium else 1)
            if quota.active_jobs >= limit:
                return False, f"Concurrent job limit ({limit}) exceeded"

        return True, None

    async def consume_quota(self, user_id: str, quota_type: str):
        """Consume user quota."""
        quota = await self.get_user_quota(user_id)

        if quota_type == "api_call":
            quota.api_calls_today += 1
        elif quota_type == "youtube_download":
            quota.youtube_downloads_today += 1
        elif quota_type == "job_start":
            quota.active_jobs += 1
        elif quota_type == "job_end":
            quota.active_jobs = max(0, quota.active_jobs - 1)

        await self.save_user_quota(quota)


# Global instances
redis_client: Optional[aioredis.Redis] = None
youtube_limiter: Optional[YouTubeRateLimiter] = None
quota_manager: Optional[QuotaManager] = None

# FastAPI rate limiter
limiter = Limiter(key_func=get_remote_address)


async def init_rate_limiting(redis_url: Optional[str] = None):
    """Initialize rate limiting components."""
    global redis_client, youtube_limiter, quota_manager

    if redis_url:
        redis_client = aioredis.from_url(redis_url)

    youtube_limiter = YouTubeRateLimiter(redis_client)
    quota_manager = QuotaManager(redis_client)

    logger.info("Rate limiting initialized")


async def cleanup_rate_limiting():
    """Cleanup rate limiting resources."""
    global redis_client

    if redis_client:
        await redis_client.close()

    logger.info("Rate limiting cleanup completed")


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom rate limit exceeded handler."""
    response = _rate_limit_exceeded_handler(request, exc)
    response.headers["X-RateLimit-Limit"] = str(exc.detail)
    response.headers["X-RateLimit-Remaining"] = "0"
    response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
    return response
