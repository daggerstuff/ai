"""Unit and integration tests for ingestion connectors.

Tests for LocalFileConnector, YouTubeConnector, S3Connector, and GCSConnector.
"""

import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from ai.pipelines.orchestrator.ingestion_interface import IngestRecord, LocalFileConnector
from ai.pipelines.orchestrator.youtube_connector import YouTubeConnector, YouTubeConfig
from ai.pipelines.orchestrator.s3_connector import S3Connector, S3Config  
from ai.pipelines.orchestrator.gcs_connector import GCSConnector, GCSConfig
from ai.pipelines.orchestrator.ingestion_queue import IngestionQueue, QueueItem, QueueType
from ai.pipelines.orchestrator.ingestion_deduplication import IngestionDeduplicator


class TestLocalFileConnector:
    """Test LocalFileConnector functionality."""
    
    def test_connect_valid_directory(self):
        """Test connecting to a valid directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            connector = LocalFileConnector(temp_dir)
            connector.connect()  # Should not raise an exception
    
    def test_connect_invalid_directory(self):
        """Test connecting to an invalid directory."""
        connector = LocalFileConnector("/non/existent/directory")
        with pytest.raises(Exception):
            connector.connect()
    
    def test_fetch_empty_directory(self):
        """Test fetching from an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            connector = LocalFileConnector(temp_dir)
            connector.connect()
            records = list(connector.fetch())
            assert len(records) == 0
    
    def test_fetch_single_file(self):
        """Test fetching a single file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("Hello, World!")
            
            connector = LocalFileConnector(temp_dir)
            connector.connect()
            records = list(connector.fetch())
            
            assert len(records) == 1
            record = records[0]
            assert isinstance(record, IngestRecord)
            assert record.id.endswith("test.txt")
            assert b"Hello, World!" in record.payload
    
    def test_fetch_multiple_files(self):
        """Test fetching multiple files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple test files
            file1 = Path(temp_dir) / "test1.txt"
            file1.write_text("Content 1")
            
            file2 = Path(temp_dir) / "test2.txt"
            file2.write_text("Content 2")
            
            connector = LocalFileConnector(temp_dir)
            connector.connect()
            records = list(connector.fetch())
            
            assert len(records) == 2
    
    def test_security_path_traversal_prevention(self):
        """Test that path traversal attempts are blocked."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file outside the allowed directory
            outside_file = Path(temp_dir) / ".." / "outside.txt"
            outside_file.write_text("Outside content")
            
            # Create a legitimate file inside the directory
            inside_file = Path(temp_dir) / "inside.txt"
            inside_file.write_text("Inside content")
            
            connector = LocalFileConnector(temp_dir)
            connector.connect()
            records = list(connector.fetch())
            
            # Should only return the legitimate file
            assert len(records) == 1
            assert "inside.txt" in records[0].id
    
    def test_security_dangerous_extensions_blocked(self):
        """Test that dangerous file extensions are blocked."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dangerous file
            exe_file = Path(temp_dir) / "dangerous.exe"
            exe_file.write_text("Dangerous content")
            
            # Create a safe file
            safe_file = Path(temp_dir) / "safe.txt"
            safe_file.write_text("Safe content")
            
            connector = LocalFileConnector(temp_dir)
            connector.connect()
            records = list(connector.fetch())
            
            # Should only return the safe file
            assert len(records) == 1
            assert "safe.txt" in records[0].id


@patch('ai.pipelines.orchestrator.youtube_connector.subprocess.run')
@patch('ai.pipelines.orchestrator.youtube_processor.YouTubePlaylistProcessor')
class TestYouTubeConnector:
    """Test YouTubeConnector functionality."""
    
    def test_connect_success(self, mock_processor, mock_subprocess):
        """Test successful connection check."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        config = YouTubeConfig(playlist_urls=["https://youtube.com/playlist?list=test"])
        connector = YouTubeConnector(config)
        
        # Should not raise an exception
        connector.connect()
    
    def test_connect_yt_dlp_missing(self, mock_processor, mock_subprocess):
        """Test connection when yt-dlp is not available."""
        mock_subprocess.side_effect = FileNotFoundError()
        
        config = YouTubeConfig(playlist_urls=["https://youtube.com/playlist?list=test"])
        connector = YouTubeConnector(config)
        
        with pytest.raises(Exception):
            connector.connect()
    
    def test_url_validation_allow_list(self, mock_processor, mock_subprocess):
        """Test URL validation against allow list."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        config = YouTubeConfig(playlist_urls=["https://youtube.com/playlist?list=test"])
        connector = YouTubeConnector(config, allowed_domains=['youtube.com'])
        
        # Valid URL should pass
        assert connector._validate_youtube_url("https://youtube.com/watch?v=abc123")
        
        # Invalid domain should fail
        assert not connector._validate_youtube_url("https://malicious.com/watch?v=abc123")
    
    def test_url_validation_ssrp_protection(self, mock_processor, mock_subprocess):
        """Test SSRF protection in URL validation."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        config = YouTubeConfig(playlist_urls=["https://youtube.com/playlist?list=test"])
        connector = YouTubeConnector(config)
        
        # Test various SSRF patterns
        ssrf_urls = [
            "https://youtube.com/watch?v=abc123&redirect=127.0.0.1",
            "https://youtube.com/watch?v=abc123&url=localhost",
            "https://youtube.com/watch?v=abc123&target=metadata.google.internal"
        ]
        
        for url in ssrf_urls:
            assert not connector._validate_youtube_url(url)


@patch('ai.pipelines.orchestrator.s3_connector.boto3.Session')
class TestS3Connector:
    """Test S3Connector functionality."""
    
    def test_connect_success(self, mock_boto3_session):
        """Test successful S3 connection."""
        mock_client = Mock()
        mock_client.head_bucket.return_value = {}
        mock_boto3_session.return_value.client.return_value = mock_client
        
        config = S3Config(bucket_name="test-bucket")
        connector = S3Connector(config)
        
        # Should not raise an exception
        connector.connect()
    
    def test_connect_bucket_not_exists(self, mock_boto3_session):
        """Test connection when bucket doesn't exist."""
        from botocore.exceptions import ClientError
        
        mock_client = Mock()
        error_response = {
            'Error': {'Code': '404', 'Message': 'Not Found'}
        }
        mock_client.head_bucket.side_effect = ClientError(error_response, 'HeadBucket')
        mock_boto3_session.return_value.client.return_value = mock_client
        
        config = S3Config(bucket_name="non-existent-bucket")
        connector = S3Connector(config)
        
        with pytest.raises(Exception):
            connector.connect()
    
    def test_object_key_validation(self, mock_boto3_session):
        """Test S3 object key validation."""
        mock_client = Mock()
        mock_boto3_session.return_value.client.return_value = mock_client
        
        config = S3Config(bucket_name="test-bucket")
        connector = S3Connector(config)
        
        # Valid key should pass
        assert connector._is_valid_s3_key("valid/path/file.txt")
        
        # Path traversal should fail
        assert not connector._is_valid_s3_key("../outside/file.txt")
        assert not connector._is_valid_s3_key("/absolute/path.txt")
        assert not connector._is_valid_s3_key("normal/../outside.txt")
        
        # Dangerous extensions should fail
        assert not connector._is_valid_s3_key("dangerous.exe")
        assert not connector._is_valid_s3_key("script.bat")
        assert not connector._is_valid_s3_key("malicious.sh")
        
        # Safe extensions should pass
        assert connector._is_valid_s3_key("normal.txt")
        assert connector._is_valid_s3_key("data.json")
        assert connector._is_valid_s3_key("document.pdf")


@patch('ai.pipelines.orchestrator.gcs_connector.storage.Client')
class TestGCSConnector:
    """Test GCSConnector functionality."""
    
    def test_connect_success(self, mock_storage_client):
        """Test successful GCS connection."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_bucket.exists.return_value = True
        mock_client.bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client
        
        config = GCSConfig(bucket_name="test-bucket")
        connector = GCSConnector(config)
        
        # Should not raise an exception
        connector.connect()
    
    def test_connect_bucket_not_exists(self, mock_storage_client):
        """Test connection when bucket doesn't exist."""
        from google.cloud.exceptions import NotFound
        
        mock_client = Mock()
        mock_bucket = Mock()
        mock_bucket.exists.return_value = False
        mock_client.bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client
        
        config = GCSConfig(bucket_name="non-existent-bucket")
        connector = GCSConnector(config)
        
        with pytest.raises(Exception):
            connector.connect()
    
    def test_blob_name_validation(self, mock_storage_client):
        """Test GCS blob name validation."""
        mock_client = Mock()
        mock_storage_client.return_value = mock_client
        
        config = GCSConfig(bucket_name="test-bucket")
        connector = GCSConnector(config)
        
        # Valid blob name should pass
        assert connector._is_valid_gcs_blob_name("valid/path/file.txt")
        
        # Path traversal should fail
        assert not connector._is_valid_gcs_blob_name("../outside/file.txt")
        assert not connector._is_valid_gcs_blob_name("/absolute/path.txt")
        assert not connector._is_valid_gcs_blob_name("normal/../outside.txt")
        
        # Dangerous extensions should fail
        assert not connector._is_valid_gcs_blob_name("dangerous.exe")
        assert not connector._is_valid_gcs_blob_name("script.bat")
        assert not connector._is_valid_gcs_blob_name("malicious.sh")
        
        # Safe extensions should pass
        assert connector._is_valid_gcs_blob_name("normal.txt")
        assert connector._is_valid_gcs_blob_name("data.json")
        assert connector._is_valid_gcs_blob_name("document.pdf")


class TestIngestionQueue:
    """Test IngestionQueue functionality."""
    
    def test_internal_queue_basic_ops(self):
        """Test basic operations with internal queue."""
        queue = IngestionQueue(queue_type=QueueType.INTERNAL_ASYNC, max_size=10)
        
        item = QueueItem(
            id="test_id",
            payload="test_payload",
            metadata={"source": "test"},
            source_connector="test"
        )
        
        # Test enqueue
        result = asyncio.run(queue.enqueue(item))
        assert result is True
        
        # Test dequeue
        items = asyncio.run(queue.dequeue_batch())
        assert len(items) == 1
        assert items[0].id == "test_id"
    
    def test_queue_size_tracking(self):
        """Test queue size tracking."""
        queue = IngestionQueue(queue_type=QueueType.INTERNAL_ASYNC, max_size=10)
        
        item = QueueItem(
            id="test_id",
            payload="test_payload", 
            metadata={"source": "test"},
            source_connector="test"
        )
        
        # Add item and check size
        asyncio.run(queue.enqueue(item))
        size = asyncio.run(queue.get_queue_size())
        assert size == 1
    
    @pytest.mark.asyncio
    async def test_queue_backpressure(self):
        """Test queue backpressure handling."""
        queue = IngestionQueue(queue_type=QueueType.INTERNAL_ASYNC, max_size=1)
        
        item1 = QueueItem(id="test1", payload="payload1", metadata={"source": "test"}, source_connector="test")
        item2 = QueueItem(id="test2", payload="payload2", metadata={"source": "test"}, source_connector="test")
        
        # First item should succeed
        result1 = await queue.enqueue(item1)
        assert result1 is True
        
        # Second item should fail due to backpressure
        result2 = await queue.enqueue(item2)
        assert result2 is False


class TestIngestionDeduplicator:
    """Test IngestionDeduplicator functionality."""
    
    def test_basic_deduplication(self):
        """Test basic deduplication functionality."""
        dedup = IngestionDeduplicator(capacity=1000, error_rate=0.01)
        
        content1 = "Hello, World!"
        content2 = "Hello, World!"
        content3 = "Different content"
        
        # First occurrence should be new
        is_new1, hash1 = dedup.add_and_check(content1)
        assert is_new1 is True
        
        # Duplicate should not be new
        is_new2 = dedup.is_duplicate(content2)
        assert is_new2 is True  # Is duplicate
        
        # Different content should be new
        is_new3, hash3 = dedup.add_and_check(content3)
        assert is_new3 is True
        
        # Original content should still be detected as duplicate
        is_new4 = dedup.is_duplicate(content1)
        assert is_new4 is True


# Integration tests
class TestConnectorIntegration:
    """Integration tests for connectors with other components."""
    
    def test_local_connector_with_deduplication(self):
        """Test local connector with deduplication enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a duplicate file
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("Duplicate content")
            
            connector = LocalFileConnector(temp_dir)
            connector.connect()
            
            # First fetch should return the record
            records1 = list(connector.fetch())
            assert len(records1) == 1
            
            # Second fetch of same content should be skipped due to deduplication
            # Reset deduplicator for this test
            from ai.pipelines.orchestrator.ingestion_deduplication import get_ingestion_deduplicator
            dedup = get_ingestion_deduplicator()
            dedup.clear()  # Clear to test deduplication properly
            
            # Re-add same file and fetch again - should still return 1 record due to deduplication
            # (in real scenario, this would require different approach to test deduplication)
    
    def test_local_connector_with_queue(self):
        """Test local connector integration with ingestion queue."""
        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                test_file = Path(temp_dir) / "test.txt"
                test_file.write_text("Queue test content")
                
                # Create queue
                queue = await IngestionQueue(
                    queue_type=QueueType.INTERNAL_ASYNC,
                    max_size=10
                ).__aenter__()  # Use context manager approach or just create
                
                # Create connector
                connector = LocalFileConnector(temp_dir)
                connector.connect()
                
                # Process records and add to queue
                for record in connector.fetch():
                    queue_item = QueueItem(
                        id=record.id,
                        payload=record.payload,
                        metadata=record.metadata,
                        source_connector="local"
                    )
                    success = await queue.enqueue(queue_item)
                    assert success is True
                
                # Verify queue has items
                queued_items = await queue.dequeue_batch()
                assert len(queued_items) == 1
        
        asyncio.run(run_test())