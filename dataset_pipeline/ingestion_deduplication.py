"""Fast bloom filter deduplication for ingestion stage.

Implements a bloom filter for quick duplicate detection at ingest time,
with configurable false positive rate and memory usage.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional
from bitarray import bitarray

try:
    import mmh3
    HAS_MURMUR = True
except ImportError:
    HAS_MURMUR = False  # Fallback to built-in hash


class BloomFilter:
    """Bloom filter for fast duplicate detection."""

    def __init__(self, capacity: int = 1000000, error_rate: float = 0.01):
        """
        Initialize bloom filter with capacity and error rate.
        
        Args:
            capacity: Expected number of items to store
            error_rate: Acceptable false positive rate (0.01 = 1%)
        """
        if error_rate >= 1.0 or error_rate <= 0.0:
            raise ValueError("Error rate must be between 0 and 1")
        
        self.capacity = capacity
        self.error_rate = error_rate
        
        # Calculate optimal size and hash count
        # Formulas from https://en.wikipedia.org/wiki/Bloom_filter
        import math
        self.bit_array_size = int(
            -(capacity * math.log(error_rate)) / (math.log(2) ** 2)
        )
        self.hash_count = int(
            (self.bit_array_size / capacity) * math.log(2)
        )
        
        # Initialize bit array
        self.bit_array = bitarray(self.bit_array_size)
        self.bit_array.setall(0)
        
        self.count = 0  # Track actual count for capacity management

    def _hash(self, item: str, seed: int) -> int:
        """Generate hash for item with given seed."""
        if HAS_MURMUR:
            # Use MurmurHash for better distribution
            return mmh3.hash(item, seed) % self.bit_array_size
        else:
            # Fallback to built-in hash
            return hash(f"{item}{seed}") % self.bit_array_size

    def add(self, item: str) -> bool:
        """Add item to bloom filter and return True if it's likely new."""
        if self.contains(item):
            return False  # Likely duplicate
        
        # Add item to filter
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.bit_array[index] = 1
        
        self.count += 1
        return True  # New item

    def contains(self, item: str) -> bool:
        """Check if item is in bloom filter (may have false positives)."""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if not self.bit_array[index]:
                return False  # Definitely not in set
        return True  # Possibly in set

    def add_hash(self, item_hash: str) -> bool:
        """Add a pre-computed hash to bloom filter."""
        return self.add(item_hash)

    def contains_hash(self, item_hash: str) -> bool:
        """Check if a pre-computed hash exists in bloom filter."""
        return self.contains(item_hash)

    def is_full(self) -> bool:
        """Check if bloom filter is approaching capacity."""
        return self.count >= (self.capacity * 0.9)  # 90% capacity

    def clear(self) -> None:
        """Clear the bloom filter."""
        self.bit_array.setall(0)
        self.count = 0


class ContentHasher:
    """Utility for generating stable content hashes."""
    
    @staticmethod
    def hash_content(content: Any) -> str:
        """Generate stable hash for content."""
        # Convert content to string representation
        if isinstance(content, (str, bytes)):
            content_str = str(content)
        elif isinstance(content, dict):
            # Sort keys for consistent hashing
            import json
            content_str = json.dumps(content, sort_keys=True, default=str)
        elif isinstance(content, (list, tuple)):
            import json
            content_str = json.dumps(list(content), default=str)
        else:
            content_str = str(content)
        
        # Generate SHA-256 hash and return as hex
        return hashlib.sha256(content_str.encode()).hexdigest()


class IngestionDeduplicator:
    """Deduplication system for ingestion stage."""

    def __init__(self, capacity: int = 100000, error_rate: float = 0.01):
        self.bloom_filter = BloomFilter(capacity, error_rate)
        self.content_hasher = ContentHasher()

    def is_duplicate(self, content: Any) -> bool:
        """Check if content is a duplicate."""
        content_hash = self.content_hasher.hash_content(content)
        return self.bloom_filter.contains_hash(content_hash)

    def add_content(self, content: Any) -> bool:
        """Add content and return True if it's likely new."""
        content_hash = self.content_hasher.hash_content(content)
        return self.bloom_filter.add_hash(content_hash)

    def add_and_check(self, content: Any) -> tuple[bool, str]:
        """Add content and return (is_new, content_hash)."""
        content_hash = self.content_hasher.hash_content(content)
        is_new = self.bloom_filter.add_hash(content_hash)
        return is_new, content_hash

    def is_bloom_full(self) -> bool:
        """Check if bloom filter is approaching capacity."""
        return self.bloom_filter.is_full()

    def clear(self) -> None:
        """Clear deduplication data."""
        self.bloom_filter.clear()


# Global deduplicator instance
_ingestion_deduplicator: Optional[IngestionDeduplicator] = None


def get_ingestion_deduplicator(
    capacity: int = 100000,
    error_rate: float = 0.01
) -> IngestionDeduplicator:
    """Get or create global ingestion deduplicator."""
    global _ingestion_deduplicator
    if _ingestion_deduplicator is None:
        _ingestion_deduplicator = IngestionDeduplicator(capacity, error_rate)
    return _ingestion_deduplicator


def is_duplicate(content: Any) -> bool:
    """Check if content is a duplicate using global deduplicator."""
    dedup = get_ingestion_deduplicator()
    return dedup.is_duplicate(content)


def add_content_check_duplicate(content: Any) -> bool:
    """Add content and return True if it's likely new."""
    dedup = get_ingestion_deduplicator()
    return dedup.add_content(content)


__all__ = [
    'BloomFilter',
    'ContentHasher', 
    'IngestionDeduplicator',
    'get_ingestion_deduplicator',
    'is_duplicate',
    'add_content_check_duplicate'
]