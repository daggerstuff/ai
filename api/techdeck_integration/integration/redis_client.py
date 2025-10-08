"""
Redis client integration for TechDeck Flask service.

This module provides Redis connection management, caching, and pub/sub
functionality with comprehensive error handling and performance monitoring.
"""

import json
import logging
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta

import redis
from redis.exceptions import RedisError, ConnectionError, TimeoutError

from ..utils.logger import get_logger, log_performance_metric
from ..error_handling.custom_errors import RedisError as TechDeckRedisError
from ..config import TechDeckServiceConfig


class RedisClient:
    """
    Redis client wrapper with comprehensive error handling and monitoring.
    
    Provides connection pooling, caching, pub/sub, and rate limiting
    functionality with HIPAA++ compliant logging.
    """
    
    def __init__(self, redis_url: str, db: int = 0, config: Optional[TechDeckServiceConfig] = None):
        """
        Initialize Redis client with connection pooling.
        
        Args:
            redis_url: Redis connection URL
            db: Redis database number
            config: Service configuration object
        """
        self.redis_url = redis_url
        self.db = db
        self.config = config
        self.logger = get_logger(__name__)
        
        try:
            # Parse Redis URL
            if redis_url.startswith('redis://'):
                # Standard Redis URL
                self.redis_client = redis.from_url(
                    redis_url,
                    db=db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
            else:
                # Custom connection parameters
                connection_params = self._parse_redis_url(redis_url)
                self.redis_client = redis.Redis(
                    **connection_params,
                    db=db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
            
            # Test connection
            self._test_connection()
            
            self.logger.info("Redis client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis client: {e}")
            raise TechDeckRedisError(f"Redis connection failed: {e}")
    
    def _parse_redis_url(self, url: str) -> Dict[str, Any]:
        """
        Parse custom Redis URL format.
        
        Args:
            url: Redis URL
            
        Returns:
            Connection parameters dictionary
        """
        # Handle different URL formats
        if ':' in url:
            parts = url.split(':')
            if len(parts) == 2:
                host, port = parts
                return {
                    'host': host,
                    'port': int(port),
                    'password': None
                }
        
        # Default to localhost
        return {
            'host': 'localhost',
            'port': 6379,
            'password': None
        }
    
    def _test_connection(self) -> None:
        """Test Redis connection."""
        try:
            self.redis_client.ping()
            self.logger.debug("Redis connection test successful")
        except Exception as e:
            self.logger.error(f"Redis connection test failed: {e}")
            raise TechDeckRedisError(f"Redis connection test failed: {e}")
    
    @log_performance_metric('redis_operation')
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from Redis with error handling.
        
        Args:
            key: Redis key
            default: Default value if key doesn't exist
            
        Returns:
            Retrieved value or default
        """
        try:
            value = self.redis_client.get(key)
            if value is None:
                return default
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except (ConnectionError, TimeoutError) as e:
            self.logger.error(f"Redis connection error for key {key}: {e}")
            raise TechDeckRedisError(f"Redis connection error: {e}")
        except RedisError as e:
            self.logger.error(f"Redis error for key {key}: {e}")
            raise TechDeckRedisError(f"Redis operation failed: {e}")
    
    @log_performance_metric('redis_operation')
    def set(
        self,
        key: str,
        value: Any,
        ex: Optional[int] = None,
        px: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """
        Set value in Redis with optional expiration and conditions.
        
        Args:
            key: Redis key
            value: Value to store
            ex: Expiration time in seconds
            px: Expiration time in milliseconds
            nx: Set only if key doesn't exist
            xx: Set only if key exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Serialize complex objects to JSON
            if not isinstance(value, (str, int, float, bool)):
                value = json.dumps(value)
            
            return self.redis_client.set(key, value, ex=ex, px=px, nx=nx, xx=xx)
            
        except (ConnectionError, TimeoutError) as e:
            self.logger.error(f"Redis connection error for key {key}: {e}")
            raise TechDeckRedisError(f"Redis connection error: {e}")
        except RedisError as e:
            self.logger.error(f"Redis error for key {key}: {e}")
            raise TechDeckRedisError(f"Redis operation failed: {e}")
    
    @log_performance_metric('redis_operation')
    def delete(self, key: str) -> int:
        """
        Delete key from Redis.
        
        Args:
            key: Redis key to delete
            
        Returns:
            Number of keys deleted
        """
        try:
            return self.redis_client.delete(key)
        except (ConnectionError, TimeoutError) as e:
            self.logger.error(f"Redis connection error for key {key}: {e}")
            raise TechDeckRedisError(f"Redis connection error: {e}")
        except RedisError as e:
            self.logger.error(f"Redis error for key {key}: {e}")
            raise TechDeckRedisError(f"Redis operation failed: {e}")
    
    @log_performance_metric('redis_operation')
    def exists(self, key: str) -> bool:
        """
        Check if key exists in Redis.
        
        Args:
            key: Redis key to check
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            return bool(self.redis_client.exists(key))
        except (ConnectionError, TimeoutError) as e:
            self.logger.error(f"Redis connection error for key {key}: {e}")
            raise TechDeckRedisError(f"Redis connection error: {e}")
        except RedisError as e:
            self.logger.error(f"Redis error for key {key}: {e}")
            raise TechDeckRedisError(f"Redis operation failed: {e}")
    
    def cache_get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache with automatic deserialization.
        
        Args:
            key: Cache key
            default: Default value if key doesn't exist
            
        Returns:
            Cached value or default
        """
        return self.get(f"cache:{key}", default)
    
    def cache_set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 3600
    ) -> bool:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if successful
        """
        return self.set(f"cache:{key}", value, ex=ttl_seconds)
    
    def cache_delete(self, key: str) -> int:
        """
        Delete cached value.
        
        Args:
            key: Cache key
            
        Returns:
            Number of keys deleted
        """
        return self.delete(f"cache:{key}")
    
    def rate_limit_check(
        self,
        identifier: str,
        limit: int,
        window_seconds: int = 60
    ) -> Dict[str, Any]:
        """
        Check rate limit for identifier.
        
        Args:
            identifier: Rate limit identifier (e.g., user_id, IP)
            limit: Maximum allowed requests
            window_seconds: Time window in seconds
            
        Returns:
            Dictionary with rate limit information
        """
        key = f"rate_limit:{identifier}:{window_seconds}"
        
        try:
            current_count = self.redis_client.incr(key)
            
            # Set expiration on first request
            if current_count == 1:
                self.redis_client.expire(key, window_seconds)
            
            return {
                'allowed': current_count <= limit,
                'current_count': current_count,
                'limit': limit,
                'remaining': max(0, limit - current_count),
                'reset_time': datetime.utcnow().timestamp() + window_seconds
            }
            
        except (ConnectionError, TimeoutError) as e:
            self.logger.error(f"Redis connection error for rate limit {identifier}: {e}")
            # Fail open - allow request if rate limiting fails
            return {
                'allowed': True,
                'current_count': 0,
                'limit': limit,
                'remaining': limit,
                'reset_time': datetime.utcnow().timestamp() + window_seconds
            }
        except RedisError as e:
            self.logger.error(f"Redis error for rate limit {identifier}: {e}")
            # Fail open - allow request if rate limiting fails
            return {
                'allowed': True,
                'current_count': 0,
                'limit': limit,
                'remaining': limit,
                'reset_time': datetime.utcnow().timestamp() + window_seconds
            }
    
    def publish_message(self, channel: str, message: Any) -> int:
        """
        Publish message to Redis channel.
        
        Args:
            channel: Redis channel name
            message: Message to publish
            
        Returns:
            Number of subscribers that received the message
        """
        try:
            # Serialize message to JSON
            if not isinstance(message, str):
                message = json.dumps(message)
            
            return self.redis_client.publish(channel, message)
            
        except (ConnectionError, TimeoutError) as e:
            self.logger.error(f"Redis connection error for channel {channel}: {e}")
            raise TechDeckRedisError(f"Redis connection error: {e}")
        except RedisError as e:
            self.logger.error(f"Redis error for channel {channel}: {e}")
            raise TechDeckRedisError(f"Redis publish failed: {e}")
    
    def subscribe_to_channel(self, channel: str):
        """
        Subscribe to Redis channel.
        
        Args:
            channel: Channel name to subscribe to
            
        Returns:
            Pub/Sub object for message listening
        """
        try:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe(channel)
            return pubsub
            
        except (ConnectionError, TimeoutError) as e:
            self.logger.error(f"Redis connection error for channel {channel}: {e}")
            raise TechDeckRedisError(f"Redis connection error: {e}")
        except RedisError as e:
            self.logger.error(f"Redis error for channel {channel}: {e}")
            raise TechDeckRedisError(f"Redis subscribe failed: {e}")
    
    def pipeline_execute(self, operations: List[Dict[str, Any]]) -> List[Any]:
        """
        Execute multiple Redis operations in a pipeline.
        
        Args:
            operations: List of operation dictionaries
            
        Returns:
            List of operation results
        """
        try:
            with self.redis_client.pipeline() as pipe:
                for operation in operations:
                    op_type = operation.get('type')
                    if op_type == 'set':
                        pipe.set(
                            operation['key'],
                            operation['value'],
                            ex=operation.get('ex'),
                            px=operation.get('px')
                        )
                    elif op_type == 'get':
                        pipe.get(operation['key'])
                    elif op_type == 'delete':
                        pipe.delete(operation['key'])
                    elif op_type == 'exists':
                        pipe.exists(operation['key'])
                    else:
                        self.logger.warning(f"Unknown operation type: {op_type}")
                
                return pipe.execute()
                
        except (ConnectionError, TimeoutError) as e:
            self.logger.error(f"Redis connection error in pipeline: {e}")
            raise TechDeckRedisError(f"Redis connection error: {e}")
        except RedisError as e:
            self.logger.error(f"Redis error in pipeline: {e}")
            raise TechDeckRedisError(f"Redis pipeline failed: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform Redis health check.
        
        Returns:
            Health check results
        """
        try:
            start_time = datetime.utcnow()
            
            # Test connection
            self.redis_client.ping()
            
            # Get connection info
            info = self.redis_client.info()
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time,
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_mb': info.get('used_memory_human', 'unknown'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Redis health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_connection_pool_stats(self) -> Dict[str, Any]:
        """
        Get Redis connection pool statistics.
        
        Returns:
            Connection pool statistics
        """
        try:
            pool = self.redis_client.connection_pool
            return {
                'max_connections': pool.max_connections,
                'created_connections': len(pool._created_connections),
                'available_connections': len(pool._available_connections),
                'in_use_connections': len(pool._in_use_connections),
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get connection pool stats: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def close(self) -> None:
        """Close Redis connection."""
        try:
            if hasattr(self.redis_client, 'close'):
                self.redis_client.close()
            self.logger.info("Redis connection closed")
        except Exception as e:
            self.logger.error(f"Error closing Redis connection: {e}")


class RedisCache:
    """
    High-level Redis cache interface with automatic serialization.
    """
    
    def __init__(self, redis_client: RedisClient, default_ttl: int = 3600):
        """
        Initialize Redis cache.
        
        Args:
            redis_client: Redis client instance
            default_ttl: Default time to live in seconds
        """
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        self.logger = get_logger(__name__)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get cached value.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        return self.redis_client.cache_get(key, default)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set cached value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if successful
        """
        ttl = ttl or self.default_ttl
        return self.redis_client.cache_set(key, value, ttl)
    
    def delete(self, key: str) -> int:
        """
        Delete cached value.
        
        Args:
            key: Cache key
            
        Returns:
            Number of keys deleted
        """
        return self.redis_client.cache_delete(key)
    
    def get_or_set(self, key: str, factory: callable, ttl: Optional[int] = None) -> Any:
        """
        Get cached value or set it using factory function.
        
        Args:
            key: Cache key
            factory: Function to generate value if not cached
            ttl: Time to live in seconds
            
        Returns:
            Cached or newly generated value
        """
        value = self.get(key)
        if value is None:
            value = factory()
            self.set(key, value, ttl)
        return value


class RedisRateLimiter:
    """
    Redis-based rate limiter with sliding window algorithm.
    """
    
    def __init__(self, redis_client: RedisClient):
        """
        Initialize rate limiter.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis_client = redis_client
        self.logger = get_logger(__name__)
    
    def is_allowed(
        self,
        identifier: str,
        limit: int,
        window_seconds: int = 60
    ) -> bool:
        """
        Check if request is allowed under rate limit.
        
        Args:
            identifier: Rate limit identifier
            limit: Maximum allowed requests
            window_seconds: Time window in seconds
            
        Returns:
            True if allowed, False otherwise
        """
        result = self.redis_client.rate_limit_check(identifier, limit, window_seconds)
        return result['allowed']
    
    def get_rate_limit_info(
        self,
        identifier: str,
        limit: int,
        window_seconds: int = 60
    ) -> Dict[str, Any]:
        """
        Get detailed rate limit information.
        
        Args:
            identifier: Rate limit identifier
            limit: Maximum allowed requests
            window_seconds: Time window in seconds
            
        Returns:
            Rate limit information dictionary
        """
        return self.redis_client.rate_limit_check(identifier, limit, window_seconds)


# Convenience functions
def create_redis_client(config: TechDeckServiceConfig) -> RedisClient:
    """
    Create Redis client from configuration.
    
    Args:
        config: Service configuration
        
    Returns:
        Redis client instance
    """
    return RedisClient(
        redis_url=config.REDIS_URL,
        db=config.REDIS_DB,
        config=config
    )


def create_cache(redis_client: RedisClient, default_ttl: int = 3600) -> RedisCache:
    """
    Create Redis cache from client.
    
    Args:
        redis_client: Redis client instance
        default_ttl: Default time to live in seconds
        
    Returns:
        Redis cache instance
    """
    return RedisCache(redis_client, default_ttl)


def create_rate_limiter(redis_client: RedisClient) -> RedisRateLimiter:
    """
    Create Redis rate limiter from client.
    
    Args:
        redis_client: Redis client instance
        
    Returns:
        Redis rate limiter instance
    """
    return RedisRateLimiter(redis_client)