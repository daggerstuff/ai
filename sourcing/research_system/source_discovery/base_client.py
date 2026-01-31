"""
Enhanced base client for API interactions with caching and standardized error handling.

Provides common functionality for all repository and journal API clients.
"""

import hashlib
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Optional
from functools import wraps

import requests

from ..config import get_config

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cache entry with TTL support."""
    
    def __init__(self, value: Any, ttl_seconds: int = 300):
        """
        Initialize cache entry.
        
        Args:
            value: Cached value
            ttl_seconds: Time to live in seconds (default 5 minutes)
        """
        self.value = value
        self.expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.expires_at


class RequestCache:
    """Simple in-memory cache with TTL for API requests."""
    
    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        """
        Initialize request cache.
        
        Args:
            default_ttl: Default TTL in seconds
            max_size: Maximum number of cached entries
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        if entry.is_expired():
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return None
        
        # Update access order (LRU)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override
        """
        # Evict oldest entry if cache is full
        if len(self._cache) >= self.max_size and key not in self._cache:
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
        
        ttl_seconds = ttl if ttl is not None else self.default_ttl
        self._cache[key] = CacheEntry(value, ttl_seconds)
        
        if key not in self._access_order:
            self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "default_ttl": self.default_ttl
        }


class APIError(Exception):
    """Base exception for API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[requests.Response] = None):
        """
        Initialize API error.
        
        Args:
            message: Error message
            status_code: HTTP status code
            response: Response object
        """
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class RateLimitError(APIError):
    """Exception for rate limit errors."""
    pass


class BaseAPIClient:
    """Enhanced base class for API clients with caching and standardized error handling."""
    
    def __init__(self, service_name: str, enable_cache: bool = True):
        """
        Initialize base API client.
        
        Args:
            service_name: Name of the service for configuration lookup
            enable_cache: Whether to enable request caching
        """
        self.config = get_config()
        self.service_name = service_name
        self.base_url = self.config.get(f"api_endpoints.{service_name}.base_url")
        self.rate_limit = self.config.get_rate_limit(service_name)
        self.retry_config = self.config.get_retry_config()
        
        self._last_request_time = 0.0
        self._session = requests.Session()
        self._cache = RequestCache() if enable_cache else None
        
        # Statistics
        self._stats = {
            "requests_made": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "rate_limit_waits": 0
        }
    
    def _rate_limit_wait(self) -> None:
        """Enforce rate limiting between requests."""
        if self.rate_limit <= 0:
            return
        
        min_interval = 1.0 / self.rate_limit
        elapsed = time.time() - self._last_request_time
        
        if elapsed < min_interval:
            wait_time = min_interval - elapsed
            time.sleep(wait_time)
            self._stats["rate_limit_waits"] += 1
        
        self._last_request_time = time.time()
    
    def _generate_cache_key(self, endpoint: str, params: Optional[dict[str, str]] = None) -> str:
        """
        Generate cache key for request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
        
        Returns:
            Cache key string
        """
        key_parts = [self.service_name, endpoint]
        
        if params:
            # Sort params for consistent keys
            sorted_params = sorted(params.items())
            params_str = "&".join(f"{k}={v}" for k, v in sorted_params)
            key_parts.append(params_str)
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _make_request(
        self,
        endpoint: str,
        params: Optional[dict[str, str]] = None,
        method: str = "GET",
        use_cache: bool = True,
        cache_ttl: Optional[int] = None,
        attempt: int = 1
    ) -> requests.Response:
        """
        Make HTTP request with retry logic and caching.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            method: HTTP method (GET, POST, etc.)
            use_cache: Whether to use cache for this request
            cache_ttl: Optional cache TTL override
            attempt: Current attempt number
        
        Returns:
            Response object
        
        Raises:
            APIError: If all retry attempts fail
        """
        # Check cache for GET requests
        if method == "GET" and use_cache and self._cache:
            cache_key = self._generate_cache_key(endpoint, params)
            cached_response = self._cache.get(cache_key)
            
            if cached_response is not None:
                self._stats["cache_hits"] += 1
                logger.debug(f"Cache hit for {self.service_name}/{endpoint}")
                return cached_response
            
            self._stats["cache_misses"] += 1
        
        self._rate_limit_wait()
        
        url = f"{self.base_url}{endpoint}"
        timeout = self.retry_config.get("timeout_seconds", 30)
        
        try:
            if method == "GET":
                response = self._session.get(url, params=params, timeout=timeout)
            elif method == "POST":
                response = self._session.post(url, json=params, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            self._stats["requests_made"] += 1
            
            # Cache successful GET responses
            if method == "GET" and use_cache and self._cache:
                cache_key = self._generate_cache_key(endpoint, params)
                self._cache.set(cache_key, response, cache_ttl)
            
            return response
        
        except requests.exceptions.HTTPError as e:
            # Handle rate limiting
            if e.response and e.response.status_code == 429:
                self._stats["errors"] += 1
                raise RateLimitError(
                    f"Rate limit exceeded for {self.service_name}",
                    status_code=429,
                    response=e.response
                )
            
            # Retry on server errors
            if e.response and 500 <= e.response.status_code < 600:
                return self._retry_request(endpoint, params, method, use_cache, cache_ttl, attempt, e)
            
            # Client errors - don't retry
            self._stats["errors"] += 1
            raise APIError(
                f"{self.service_name} request failed: {e}",
                status_code=e.response.status_code if e.response else None,
                response=e.response
            )
        
        except requests.RequestException as e:
            return self._retry_request(endpoint, params, method, use_cache, cache_ttl, attempt, e)
    
    def _retry_request(
        self,
        endpoint: str,
        params: Optional[dict[str, str]],
        method: str,
        use_cache: bool,
        cache_ttl: Optional[int],
        attempt: int,
        error: Exception
    ) -> requests.Response:
        """
        Retry failed request with exponential backoff.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            method: HTTP method
            use_cache: Whether to use cache
            cache_ttl: Cache TTL
            attempt: Current attempt number
            error: Original error
        
        Returns:
            Response object
        
        Raises:
            APIError: If all retry attempts fail
        """
        max_attempts = self.retry_config.get("max_attempts", 3)
        
        if attempt < max_attempts:
            backoff = self.retry_config.get("backoff_factor", 2)
            wait_time = backoff ** (attempt - 1)
            
            logger.warning(
                f"{self.service_name} request failed (attempt {attempt}/{max_attempts}): {error}. "
                f"Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)
            return self._make_request(endpoint, params, method, use_cache, cache_ttl, attempt + 1)
        
        self._stats["errors"] += 1
        logger.error(f"{self.service_name} request failed after {max_attempts} attempts: {error}")
        raise APIError(f"{self.service_name} request failed after {max_attempts} attempts: {error}")
    
    def clear_cache(self) -> None:
        """Clear request cache."""
        if self._cache:
            self._cache.clear()
            logger.info(f"Cleared cache for {self.service_name}")
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get client statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = self._stats.copy()
        
        if self._cache:
            stats["cache"] = self._cache.get_stats()
            
            # Calculate cache hit rate
            total_cache_requests = stats["cache_hits"] + stats["cache_misses"]
            if total_cache_requests > 0:
                stats["cache_hit_rate"] = round(
                    stats["cache_hits"] / total_cache_requests * 100, 2
                )
            else:
                stats["cache_hit_rate"] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "requests_made": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "rate_limit_waits": 0
        }
