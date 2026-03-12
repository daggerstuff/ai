import unittest.mock

#!/usr/bin/env python3
"""
Unit tests for Quality Validation Caching System
"""

import json
import os

# Add parent directory to path for imports
import sys
import tempfile
import unittest
from unittest.mock import MagicMock
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


from quality_validation_cache import CachedQualityValidator, QualityValidationCache


class TestQualityValidationCache(unittest.TestCase):
    """Test cases for QualityValidationCache"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_file = Path(self.temp_dir) / "test_data.json"

        # Create test data
        test_data = {"test": "data", "value": 42}
        with open(self.cache_file, "w") as f:
            json.dump(test_data, f)

        # Create cache instance
        self.cache = QualityValidationCache()
        self.cache.cache_dir = Path(self.temp_dir) / "quality_cache"
        self.cache.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache.cache_dir = Path(self.temp_dir) / "quality_cache"
        self.cache.cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        # Clean up cache
        if hasattr(self, "cache"):
            self.cache.close()
            self.cache.invalidate_cache()  # Clear all cache entries

    def test_calculate_data_hash(self):
        """Test data hash calculation"""
        metadata = {"version": "1.0", "config": {"threshold": 0.8}}
        hash1 = self.cache._calculate_data_hash(str(self.cache_file), metadata)
        hash2 = self.cache._calculate_data_hash(str(self.cache_file), metadata)

        # Same data should produce same hash
        self.assertEqual(hash1, hash2)

        # Different metadata should produce different hash
        metadata2 = {"version": "1.1", "config": {"threshold": 0.9}}
        hash3 = self.cache._calculate_data_hash(str(self.cache_file), metadata2)
        self.assertNotEqual(hash1, hash3)

    def test_generate_cache_key(self):
        """Test cache key generation"""
        data_hash = "abc123"
        validation_type = "conversation"
        key = self.cache._generate_cache_key(data_hash, validation_type)

        self.assertEqual(key, f"quality_val:{validation_type}:{data_hash}")

    def test_cache_result_and_get_cached_result(self):
        """Test caching and retrieval of results"""
        test_result = b"test validation result data"
        metadata = {"version": "1.0"}
        validation_type = "conversation"

        # Cache the result
        success = self.cache.cache_result(
            str(self.cache_file), validation_type, metadata, test_result
        )
        self.assertTrue(success)

        # Retrieve the cached result
        cached_result = self.cache.get_cached_result(
            str(self.cache_file), validation_type, metadata
        )
        self.assertEqual(cached_result, test_result)

    def test_cache_miss(self):
        """Test cache miss behavior"""
        metadata = {"version": "1.0"}
        validation_type = "conversation"

        # Try to get result that doesn't exist
        cached_result = self.cache.get_cached_result(
            str(self.cache_file), validation_type, metadata
        )
        self.assertIsNone(cached_result)





    def test_cache_result_redis_exception(self):
        """Test redis cache exception is handled gracefully"""
        from unittest.mock import MagicMock

        test_result = b"success data"
        metadata = {"version": "1.0"}
        validation_type = "conversation"

        # Setup mock redis client to throw an exception
        self.cache.redis_client = MagicMock()
        self.cache.redis_client.setex.side_effect = Exception("Redis Error")

        # Call cache_result
        success = self.cache.cache_result(
            str(self.cache_file), validation_type, metadata, test_result
        )

        # Should still return True as it's cached in memory and file
        self.assertTrue(success)

    def test_cache_result_file_cache_exception(self):
        """Test file cache exception is handled gracefully"""
        from unittest.mock import patch

        test_result = b"success data"
        metadata = {"version": "1.0"}
        validation_type = "conversation"

        # Mock open to raise an exception
        with patch('builtins.open', side_effect=Exception("File Error")):
            # Note: _calculate_data_hash also uses open, so we need to bypass it or handle it
            # We'll mock _calculate_data_hash to avoid the first open() failure
            with patch.object(self.cache, '_calculate_data_hash', return_value="testhash"):
                success = self.cache.cache_result(
                    str(self.cache_file), validation_type, metadata, test_result
                )

                # Should still return True as it's cached in memory
                self.assertTrue(success)

    def test_cache_result_exception(self):
        """Test cache result handling exception correctly"""
        from unittest.mock import patch

        test_result = b"success data"
        metadata = {"version": "1.0"}
        validation_type = "conversation"

        # Mock _calculate_data_hash to raise an Exception
        with patch.object(self.cache, '_calculate_data_hash', side_effect=Exception("Test Error")):
            success = self.cache.cache_result(
                str(self.cache_file), validation_type, metadata, test_result
            )

            self.assertFalse(success)

    def test_cache_result_success(self):
        """Test successful caching including memory, Redis, and file cache"""
        from unittest.mock import MagicMock

        test_result = b"success data"
        metadata = {"version": "1.0"}
        validation_type = "conversation"

        # Setup mock redis client
        self.cache.redis_client = MagicMock()

        # Call cache_result
        success = self.cache.cache_result(
            str(self.cache_file), validation_type, metadata, test_result
        )

        self.assertTrue(success)

        # Verify memory cache
        data_hash = self.cache._calculate_data_hash(str(self.cache_file), metadata)
        cache_key = self.cache._generate_cache_key(data_hash, validation_type)
        self.assertIn(cache_key, self.cache.memory_cache)
        self.assertEqual(self.cache.memory_cache[cache_key][0], test_result)

        # Verify Redis cache
        self.cache.redis_client.setex.assert_called_once_with(
            cache_key, self.cache.cache_ttl, test_result
        )

        # Verify file cache
        cache_file_path = self.cache.cache_dir / f"{cache_key}.pkl"
        self.assertTrue(cache_file_path.exists())
        with open(cache_file_path, "rb") as f:
            self.assertEqual(f.read(), test_result)

    def test_invalidate_cache(self):
        """Test cache invalidation"""
        test_result = b"test validation result data"
        metadata = {"version": "1.0"}
        validation_type = "conversation"

        # Cache the result
        self.cache.cache_result(
            str(self.cache_file), validation_type, metadata, test_result
        )

        # Verify it's cached
        cached_result = self.cache.get_cached_result(
            str(self.cache_file), validation_type, metadata
        )
        self.assertEqual(cached_result, test_result)

        success = self.cache.invalidate_cache(
            str(self.cache_file), validation_type, metadata
        )
        self.assertTrue(success)

        # Verify it's invalidated
        cached_result_after = self.cache.get_cached_result(
            str(self.cache_file), validation_type, metadata
        )
        self.assertIsNone(cached_result_after)

    def test_invalidate_cache_wrong_metadata(self):
        """Test that invalidating with wrong metadata does not invalidate the original cache"""
        test_result = b"test validation result data"
        metadata = {"version": "1.0"}
        wrong_metadata = {"version": "2.0"}
        validation_type = "conversation"

        # Cache the result
        self.cache.cache_result(
            str(self.cache_file), validation_type, metadata, test_result
        )

        success = self.cache.invalidate_cache(
            str(self.cache_file), validation_type, wrong_metadata
        )
        self.assertTrue(success)

        # Verify it's STILL cached under the correct metadata
        cached_result_after = self.cache.get_cached_result(
            str(self.cache_file), validation_type, metadata
        )
        self.assertEqual(cached_result_after, test_result)

    def test_invalidate_cache_missing_entry(self):
        """Test cache invalidation when the entry does not exist"""
        metadata = {"version": "1.0"}
        validation_type = "conversation"

        success = self.cache.invalidate_cache(
            "nonexistent_file.json", validation_type, metadata
        )
        self.assertTrue(success)

    @unittest.mock.patch("infra.cloud.distributed.quality_validation_cache.redis")
    def test_invalidate_cache_redis(self, mock_redis):
        """Test cache invalidation when Redis is available"""
        test_result = b"test validation result data"
        metadata = {"version": "1.0"}
        validation_type = "conversation"

        # Setup mock redis client
        mock_redis_client = unittest.mock.MagicMock()

        # Create a new cache instance with the mocked redis
        cache = QualityValidationCache()
        cache.redis_client = mock_redis_client

        cache.cache_result(str(self.cache_file), validation_type, metadata, test_result)

        success = cache.invalidate_cache(
            str(self.cache_file), validation_type, metadata
        )
        self.assertTrue(success)

        # Verify redis delete was called
        self.assertTrue(mock_redis_client.delete.called)

    def test_invalidate_all_cache(self):
        """Test invalidation of all cache entries"""
        test_result1 = b"test validation result data 1"
        test_result2 = b"test validation result data 2"
        metadata = {"version": "1.0"}

        # Cache multiple results
        self.cache.cache_result(
            str(self.cache_file), "conversation", metadata, test_result1
        )
        self.cache.cache_result(str(self.cache_file), "text", metadata, test_result2)

        # Verify they're cached
        cached_result1 = self.cache.get_cached_result(
            str(self.cache_file), "conversation", metadata
        )
        cached_result2 = self.cache.get_cached_result(
            str(self.cache_file), "text", metadata
        )
        self.assertEqual(cached_result1, test_result1)
        self.assertEqual(cached_result2, test_result2)

        # Invalidate all cache
        success = self.cache.invalidate_cache()
        self.assertTrue(success)

        # Verify they're no longer cached
        cached_result1 = self.cache.get_cached_result(
            str(self.cache_file), "conversation", metadata
        )
        cached_result2 = self.cache.get_cached_result(
            str(self.cache_file), "text", metadata
        )
        self.assertIsNone(cached_result1)
        self.assertIsNone(cached_result2)

    def test_get_cache_statistics_no_redis(self):
        """Test cache statistics when Redis is not available"""
        # Override cache directory to temp_dir to isolate test file entries
        self.cache.cache_dir = Path(self.temp_dir)
        self.cache.redis_client = None
        self.cache.memory_cache = {"a": "b", "c": "d"}

        # Create some fake cache files
        (self.cache.cache_dir / "fake1.pkl").touch()
        (self.cache.cache_dir / "fake2.pkl").touch()
        (self.cache.cache_dir / "fake3.pkl").touch()

        stats = self.cache.get_cache_statistics()

        self.assertEqual(stats["memory_cache_size"], 2)
        self.assertEqual(stats["memory_cache_max_size"], self.cache.memory_cache_max_size)
        self.assertEqual(stats["file_cache_entries"], 3)
        self.assertEqual(stats["redis_available"], False)
        self.assertNotIn("redis_cache_entries", stats)

    def test_get_cache_statistics_with_redis(self):
        """Test cache statistics when Redis is available"""
        self.cache.cache_dir = Path(self.temp_dir)
        self.cache.redis_client = MagicMock()
        self.cache.redis_client.dbsize.return_value = 10
        self.cache.memory_cache = {"a": "b"}

        # Create some fake cache files
        (self.cache.cache_dir / "fake1.pkl").touch()
        (self.cache.cache_dir / "fake2.pkl").touch()

        stats = self.cache.get_cache_statistics()

        self.assertEqual(stats["memory_cache_size"], 1)
        self.assertEqual(stats["file_cache_entries"], 2)
        self.assertEqual(stats["redis_available"], True)
        self.assertEqual(stats["redis_cache_entries"], 10)

    def test_get_cache_statistics_redis_error(self):
        """Test cache statistics handles Redis errors gracefully"""
        self.cache.cache_dir = Path(self.temp_dir)
        self.cache.redis_client = MagicMock()
        self.cache.redis_client.dbsize.side_effect = Exception("Redis connection error")
        self.cache.memory_cache = {}

        stats = self.cache.get_cache_statistics()

        self.assertEqual(stats["redis_available"], True)
        self.assertEqual(stats["redis_cache_entries"], 0)

    def test_cleanup_expired_cache(self):
        """Test cleanup of expired cache entries"""
        # This test would require mocking file timestamps, so we'll just verify it runs
        cleaned_count = self.cache.cleanup_expired_cache()
        self.assertIsInstance(cleaned_count, int)


class TestCachedQualityValidator(unittest.TestCase):
    """Test cases for CachedQualityValidator"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_file = Path(self.temp_dir) / "test_data.json"

        # Create test data
        test_data = {"test": "data", "value": 42}
        with open(self.cache_file, "w") as f:
            json.dump(test_data, f)

        # Create cache and validator instances
        self.cache = QualityValidationCache()
        self.validator = CachedQualityValidator(self.cache)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        # Clean up cache
        if hasattr(self, "cache"):
            self.cache.close()
            self.cache.invalidate_cache()  # Clear all cache entries

    def test_validate_with_cache_hit(self):
        """Test validation with cache hit"""
        test_result = b"cached validation result"
        metadata = {"version": "1.0"}
        validation_type = "conversation"

        # Cache the result first
        self.cache.cache_result(
            str(self.cache_file), validation_type, metadata, test_result
        )

        # Validate with cache
        cache_hit, cached_result = self.validator.validate_with_cache(
            str(self.cache_file), validation_type, metadata
        )

        self.assertTrue(cache_hit)
        self.assertEqual(cached_result, test_result)

    def test_validate_with_cache_miss(self):
        """Test validation with cache miss"""
        metadata = {"version": "1.0"}
        validation_type = "conversation"

        # Validate with cache (should miss)
        cache_hit, cached_result = self.validator.validate_with_cache(
            str(self.cache_file), validation_type, metadata
        )

        self.assertFalse(cache_hit)
        self.assertIsNone(cached_result)

    def test_cache_validation_result(self):
        """Test caching validation result"""
        test_result = b"validation result to cache"
        metadata = {"version": "1.0"}
        validation_type = "conversation"

        # Cache the result
        success = self.validator.cache_validation_result(
            str(self.cache_file), validation_type, metadata, test_result
        )
        self.assertTrue(success)

        # Verify it was cached
        cached_result = self.cache.get_cached_result(
            str(self.cache_file), validation_type, metadata
        )
        self.assertEqual(cached_result, test_result)


if __name__ == "__main__":
    unittest.main()
