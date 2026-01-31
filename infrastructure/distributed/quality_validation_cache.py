#!/usr/bin/env python3
"""
Quality Validation Caching System for Pixelated Empathy AI
Implements caching to avoid reprocessing quality validations
"""

import os
import sys
import json
import logging
import hashlib
import pickle
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta

# Redis for distributed caching
try:
    import redis
    import redis.exceptions
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - using local file caching only")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached validation result"""
    task_id: str
    data_hash: str
    result_data: bytes
    created_at: str
    expires_at: str
    hit_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary"""
        return cls(**data)


class QualityValidationCache:
    """Caching system for quality validation results"""
    
    def __init__(self, redis_url: str = None, cache_ttl: int = 86400):  # 24 hours default
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/1')
        self.cache_ttl = cache_ttl
        
        # Initialize Redis connection
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                self.redis_client.ping()
                logger.info("Connected to Redis for quality validation caching")
            except Exception as e:
                logger.warning(f"Could not connect to Redis: {e}")
                self.redis_client = None
        
        # Local cache directory
        self.cache_dir = Path.home() / '.pixelated' / 'quality_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for frequently accessed items
        self.memory_cache: Dict[str, Tuple[bytes, datetime]] = {}
        self.memory_cache_max_size = 1000
        
        logger.info("Quality validation cache initialized")
    
    def _calculate_data_hash(self, data_path: str, metadata: Dict[str, Any]) -> str:
        """Calculate hash of data and metadata for cache key"""
        # Hash the file content
        try:
            with open(data_path, 'rb') as f:
                content_hash = hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Could not hash file {data_path}: {e}")
            content_hash = ""
        
        # Hash the metadata
        metadata_str = json.dumps(metadata, sort_keys=True)
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
        
        # Combine hashes
        combined_hash = hashlib.md5((content_hash + metadata_hash).encode()).hexdigest()
        return combined_hash
    
    def _generate_cache_key(self, data_hash: str, validation_type: str) -> str:
        """Generate cache key"""
        return f"quality_val:{validation_type}:{data_hash}"
    
    def get_cached_result(self, data_path: str, validation_type: str, 
                         metadata: Dict[str, Any]) -> Optional[bytes]:
        """Get cached validation result"""
        data_hash = self._calculate_data_hash(data_path, metadata)
        cache_key = self._generate_cache_key(data_hash, validation_type)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            cached_data, timestamp = self.memory_cache[cache_key]
            if datetime.now() - timestamp < timedelta(hours=1):  # Memory cache for 1 hour
                logger.debug(f"Cache hit (memory): {cache_key}")
                # Update hit count
                self.memory_cache[cache_key] = (cached_data, datetime.now())
                return cached_data
            else:
                # Expired, remove from memory cache
                del self.memory_cache[cache_key]
        
        # Check Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    logger.debug(f"Cache hit (Redis): {cache_key}")
                    # Add to memory cache
                    if len(self.memory_cache) < self.memory_cache_max_size:
                        self.memory_cache[cache_key] = (cached_data, datetime.now())
                    return cached_data
            except Exception as e:
                logger.warning(f"Redis cache get failed: {e}")
        
        # Check file cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                # Check if file is expired
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age < timedelta(seconds=self.cache_ttl):
                    with open(cache_file, 'rb') as f:
                        cached_data = f.read()
                    
                    logger.debug(f"Cache hit (file): {cache_key}")
                    # Add to memory cache
                    if len(self.memory_cache) < self.memory_cache_max_size:
                        self.memory_cache[cache_key] = (cached_data, datetime.now())
                    return cached_data
                else:
                    # Expired, remove file
                    cache_file.unlink()
                    logger.debug(f"Cache expired (file): {cache_key}")
            except Exception as e:
                logger.warning(f"File cache get failed: {e}")
        
        logger.debug(f"Cache miss: {cache_key}")
        return None
    
    def cache_result(self, data_path: str, validation_type: str, metadata: Dict[str, Any], 
                    result_data: bytes) -> bool:
        """Cache validation result"""
        try:
            data_hash = self._calculate_data_hash(data_path, metadata)
            cache_key = self._generate_cache_key(data_hash, validation_type)
            
            # Store in memory cache
            if len(self.memory_cache) < self.memory_cache_max_size:
                self.memory_cache[cache_key] = (result_data, datetime.now())
            
            # Store in Redis cache
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        cache_key, 
                        self.cache_ttl, 
                        result_data
                    )
                    logger.debug(f"Cached result in Redis: {cache_key}")
                except Exception as e:
                    logger.warning(f"Redis cache set failed: {e}")
            
            # Store in file cache
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    f.write(result_data)
                # Set file modification time to control expiration
                os.utime(cache_file, None)
                logger.debug(f"Cached result in file: {cache_key}")
            except Exception as e:
                logger.warning(f"File cache set failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")
            return False
    
    def invalidate_cache(self, data_path: str = None, validation_type: str = None) -> bool:
        """Invalidate cache entries"""
        try:
            if data_path and validation_type:
                # Invalidate specific entry
                data_hash = self._calculate_data_hash(data_path, {})  # This should use the same metadata as caching
                cache_key = self._generate_cache_key(data_hash, validation_type)
                
                # Remove from memory cache
                if cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]
                
                # Remove from Redis cache
                if self.redis_client:
                    try:
                        self.redis_client.delete(cache_key)
                    except Exception as e:
                        logger.warning(f"Redis cache delete failed: {e}")
                
                # Remove from file cache
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                
                logger.info(f"Invalidated cache entry: {cache_key}")
            else:
                # Invalidate all cache
                # Clear memory cache
                self.memory_cache.clear()
                
                # Clear Redis cache
                if self.redis_client:
                    try:
                        # Delete all keys with our prefix
                        keys = self.redis_client.keys("quality_val:*")
                        if keys:
                            self.redis_client.delete(*keys)
                    except Exception as e:
                        logger.warning(f"Redis cache clear failed: {e}")
                
                # Clear file cache
                import shutil
                if self.cache_dir.exists():
                    for cache_file in self.cache_dir.glob("*.pkl"):
                        try:
                            cache_file.unlink()
                        except Exception as e:
                            logger.warning(f"Failed to delete cache file {cache_file}: {e}")
                
                logger.info("Invalidated all cache entries")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")
            return False
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'memory_cache_size': len(self.memory_cache),
            'memory_cache_max_size': self.memory_cache_max_size,
            'file_cache_entries': len(list(self.cache_dir.glob("*.pkl"))),
            'redis_available': self.redis_client is not None
        }
        
        if self.redis_client:
            try:
                stats['redis_cache_entries'] = self.redis_client.dbsize()
            except Exception as e:
                logger.warning(f"Could not get Redis cache size: {e}")
                stats['redis_cache_entries'] = 0
        
        return stats
    
    def cleanup_expired_cache(self) -> int:
        """Clean up expired file cache entries"""
        cleaned_count = 0
        current_time = datetime.now()
        
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                file_age = current_time - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age > timedelta(seconds=self.cache_ttl):
                    cache_file.unlink()
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} expired cache files")
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
        
        return cleaned_count
    
    def close(self):
        """Close cache connections"""
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception as e:
                logger.warning(f"Failed to close Redis connection: {e}")


# Enhanced quality validator with caching
class CachedQualityValidator:
    """Quality validator with caching support"""
    
    def __init__(self, cache: QualityValidationCache = None):
        self.cache = cache or QualityValidationCache()
        logger.info("Cached quality validator initialized")
    
    def validate_with_cache(self, data_path: str, validation_type: str,
                          metadata: Dict[str, Any] = None) -> Tuple[bool, Optional[bytes]]:
        """Validate with caching"""
        metadata = metadata or {}
        
        # Check cache first
        cached_result = self.cache.get_cached_result(data_path, validation_type, metadata)
        if cached_result:
            return True, cached_result
        
        # Cache miss - perform actual validation
        return False, None
    
    def cache_validation_result(self, data_path: str, validation_type: str,
                              metadata: Dict[str, Any], result_data: bytes) -> bool:
        """Cache validation result"""
        return self.cache.cache_result(data_path, validation_type, metadata, result_data)


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quality Validation Caching System")
    parser.add_argument('--redis-url', help="Redis URL for caching")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Cache stats command
    stats_parser = subparsers.add_parser('stats', help='Show cache statistics')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up expired cache entries')
    
    # Invalidate command
    invalidate_parser = subparsers.add_parser('invalidate', help='Invalidate cache entries')
    invalidate_parser.add_argument('--data-path', help='Specific data path to invalidate')
    invalidate_parser.add_argument('--type', help='Specific validation type to invalidate')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create cache
    cache = QualityValidationCache(args.redis_url)
    
    try:
        if args.command == 'stats':
            stats = cache.get_cache_statistics()
            print(json.dumps(stats, indent=2))
        
        elif args.command == 'cleanup':
            cleaned = cache.cleanup_expired_cache()
            print(f"Cleaned up {cleaned} expired cache entries")
        
        elif args.command == 'invalidate':
            if args.data_path and args.type:
                success = cache.invalidate_cache(args.data_path, args.type)
                if success:
                    print(f"Invalidated cache for {args.data_path} ({args.type})")
                else:
                    print("Failed to invalidate cache")
            else:
                success = cache.invalidate_cache()
                if success:
                    print("Invalidated all cache entries")
                else:
                    print("Failed to invalidate cache")
    
    finally:
        cache.close()


if __name__ == '__main__':
    main()