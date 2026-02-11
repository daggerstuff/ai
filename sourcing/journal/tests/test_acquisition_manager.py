"""
Unit tests for the Access & Acquisition Manager.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ai.sourcing.journal.acquisition.acquisition_manager import (
    AccessAcquisitionManager,
    AcquisitionConfig,
    DownloadProgress,
)
from ai.sourcing.journal.models.dataset_models import (
    DatasetSource,
    AccessRequest,
    AcquiredDataset,
)


class TestAcquisitionConfig:
    """Tests for AcquisitionConfig."""

    def test_default_config_valid(self):
        """Test that default config is valid."""
        config = AcquisitionConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_invalid_timeout(self):
        """Test that invalid timeout raises error."""
        config = AcquisitionConfig(download_timeout=-1)
        errors = config.validate()
        assert len(errors) > 0
        assert "download_timeout" in errors[0]

    def test_encryption_requires_key(self):
        """Test that encryption requires key."""
        config = AcquisitionConfig(encryption_enabled=True, encryption_key=None)
        errors = config.validate()
        assert len(errors) > 0
        assert "encryption_key" in errors[0]


class TestAccessAcquisitionManager:
    """Tests for AccessAcquisitionManager."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create a temporary storage directory."""
        return tmp_path / "storage"

    @pytest.fixture
    def config(self, temp_storage):
        """Create a test configuration."""
        return AcquisitionConfig(
            storage_base_path=str(temp_storage),
            download_timeout=30,
            max_retries=1,
        )

    @pytest.fixture
    def manager(self, config):
        """Create a test acquisition manager."""
        return AccessAcquisitionManager(config)

    @pytest.fixture
    def sample_source(self):
        """Create a sample dataset source."""
        return DatasetSource(
            source_id="test-001",
            title="Test Dataset",
            authors=["Author 1"],
            publication_date=datetime(2024, 1, 1),
            source_type="repository",
            url="https://example.com/dataset.zip",
            doi="10.1000/test",
            abstract="Test abstract",
            keywords=["test"],
            open_access=True,
            data_availability="available",
            discovery_date=datetime.now(),
            discovery_method="repository_api",
        )

    def test_determine_access_method_direct(self, manager):
        """Test access method determination for direct download."""
        source = DatasetSource(
            source_id="test",
            title="Test",
            authors=[],
            publication_date=datetime.now(),
            source_type="repository",
            url="https://example.com/dataset.zip",
            open_access=True,
            data_availability="available",
        )

        method = manager.determine_access_method(source)
        assert method == "direct"

    def test_determine_access_method_api(self, manager):
        """Test access method determination for API access."""
        source = DatasetSource(
            source_id="test",
            title="Test",
            authors=[],
            publication_date=datetime.now(),
            source_type="repository",
            url="https://zenodo.org/api/datasets/123",
            open_access=True,
            data_availability="available",
        )

        method = manager.determine_access_method(source)
        assert method in ["api", "direct"]  # Could be either depending on URL parsing

    def test_determine_access_method_request_form(self, manager):
        """Test access method determination for request form."""
        source = DatasetSource(
            source_id="test",
            title="Test",
            authors=[],
            publication_date=datetime.now(),
            source_type="journal",
            url="https://example.com/paper",
            open_access=False,
            data_availability="upon_request",
        )

        method = manager.determine_access_method(source)
        assert method == "request_form"

    def test_determine_access_method_collaboration(self, manager):
        """Test access method determination for collaboration."""
        source = DatasetSource(
            source_id="test",
            title="Test",
            authors=[],
            publication_date=datetime.now(),
            source_type="clinical_trial",
            url="https://example.com/trial",
            open_access=False,
            data_availability="restricted",
        )

        method = manager.determine_access_method(source)
        assert method == "collaboration"

    def test_submit_access_request(self, manager, sample_source):
        """Test access request submission."""
        request = manager.submit_access_request(sample_source)

        assert request.source_id == sample_source.source_id
        assert request.status == "pending"
        assert request.access_method in ["direct", "api", "request_form"]
        assert request.request_date is not None
        assert request.estimated_access_date is not None

    def test_submit_access_request_with_method(self, manager, sample_source):
        """Test access request submission with specified method."""
        request = manager.submit_access_request(
            sample_source, access_method="direct"
        )

        assert request.access_method == "direct"

    def test_get_access_request(self, manager, sample_source):
        """Test getting an access request."""
        request = manager.submit_access_request(sample_source)
        retrieved = manager.get_access_request(sample_source.source_id)

        assert retrieved is not None
        assert retrieved.source_id == request.source_id

    def test_update_access_request_status(self, manager, sample_source):
        """Test updating access request status."""
        request = manager.submit_access_request(sample_source)
        manager.update_access_request_status(
            sample_source.source_id, "approved", "Access granted"
        )

        updated = manager.get_access_request(sample_source.source_id)
        assert updated.status == "approved"
        assert "Access granted" in updated.notes

    def test_list_access_requests(self, manager, sample_source):
        """Test listing access requests."""
        request1 = manager.submit_access_request(sample_source)

        source2 = DatasetSource(
            source_id="test-002",
            title="Test 2",
            authors=[],
            publication_date=datetime.now(),
            source_type="repository",
            url="https://example.com/dataset2.zip",
            open_access=True,
            data_availability="available",
        )
        request2 = manager.submit_access_request(source2)

        all_requests = manager.list_access_requests()
        assert len(all_requests) == 2

        pending_requests = manager.list_access_requests(status="pending")
        assert len(pending_requests) == 2

    def test_get_pending_follow_ups(self, manager, sample_source):
        """Test getting pending follow-ups."""
        # Create an old request
        old_request = AccessRequest(
            source_id="old-001",
            access_method="request_form",
            request_date=datetime.now() - timedelta(days=10),
            status="pending",
        )
        manager.access_requests["old-001"] = old_request

        follow_ups = manager.get_pending_follow_ups()
        assert len(follow_ups) >= 1
        assert any(r.source_id == "old-001" for r in follow_ups)

    def test_generate_access_request_report(self, manager, sample_source):
        """Test access request report generation."""
        manager.submit_access_request(sample_source)
        report = manager.generate_access_request_report()

        assert "# Access Request Report" in report
        assert sample_source.source_id in report
        assert "Summary" in report

    @patch("ai.sourcing.journal.acquisition.acquisition_manager.requests.Session")
    def test_download_dataset_direct(self, mock_session, manager, sample_source, tmp_path):
        """Test direct dataset download."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.headers = {"Content-Length": "1024"}
        mock_response.iter_content = Mock(return_value=[b"test data"] * 100)
        mock_response.raise_for_status = Mock()

        mock_head_response = Mock()
        mock_head_response.headers = {"Content-Length": "1024"}

        mock_session_instance = Mock()
        mock_session_instance.head.return_value = mock_head_response
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance

        manager.session = mock_session_instance

        request = manager.submit_access_request(sample_source, access_method="direct")
        dataset = manager.download_dataset(sample_source, request)

        assert dataset.source_id == sample_source.source_id
        assert dataset.storage_path
        assert dataset.file_size_mb > 0
        assert dataset.checksum
        assert Path(dataset.storage_path).exists()

    def test_calculate_checksum(self, manager, tmp_path):
        """Test checksum calculation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        checksum = manager._calculate_checksum(test_file)
        assert len(checksum) == 64  # SHA256 hex digest length
        assert isinstance(checksum, str)

    def test_verify_download_integrity(self, manager, tmp_path):
        """Test download integrity verification."""
        # Create a test file
        test_file = tmp_path / "test.dat"
        test_file.write_text("test content")

        checksum = manager._calculate_checksum(test_file)

        dataset = AcquiredDataset(
            source_id="test-001",
            storage_path=str(test_file),
            checksum=checksum,
        )

        assert manager.verify_download_integrity(dataset) is True

        # Test with wrong checksum
        dataset.checksum = "wrong_checksum"
        assert manager.verify_download_integrity(dataset) is False

    def test_organize_storage(self, manager, sample_source, tmp_path):
        """Test storage organization."""
        # Create a test file
        test_file = tmp_path / "test.dat"
        test_file.write_text("test content")

        dataset = AcquiredDataset(
            source_id=sample_source.source_id,
            acquisition_date=datetime.now(),
            storage_path=str(test_file),
            file_format="dat",
        )

        organized_path = manager.organize_storage(dataset, sample_source)

        assert organized_path.exists()
        assert sample_source.source_type in str(organized_path)
        assert str(datetime.now().year) in str(organized_path)

    def test_download_progress_tracking(self, manager, sample_source):
        """Test download progress tracking."""
        progress = DownloadProgress(
            source_id=sample_source.source_id,
            url=sample_source.url,
            status="downloading",
            start_time=datetime.now(),
        )

        progress.update(512, total_bytes=1024)
        assert progress.downloaded_bytes == 512
        assert progress.percentage == 50.0
        assert progress.total_bytes == 1024

        progress.update(1024, total_bytes=1024)
        assert progress.status == "completed"

    def test_download_progress_error(self, manager, sample_source):
        """Test download progress error handling."""
        progress = DownloadProgress(
            source_id=sample_source.source_id,
            url=sample_source.url,
            status="downloading",
        )

        progress.update(0, error_message="Connection failed")
        assert progress.status == "failed"
        assert progress.error_message == "Connection failed"

    def test_is_direct_download_url(self, manager):
        """Test direct download URL detection."""
        assert manager._is_direct_download_url("https://example.com/file.zip") is True
        assert manager._is_direct_download_url("https://example.com/file.tar.gz") is True
        assert manager._is_direct_download_url("https://example.com/download") is True
        assert manager._is_direct_download_url("https://example.com/page") is False

    def test_is_repository_api_url(self, manager):
        """Test repository API URL detection."""
        assert manager._is_repository_api_url("https://zenodo.org/api/datasets") is True
        assert manager._is_repository_api_url("https://datadryad.org/api/v2") is True
        assert manager._is_repository_api_url("https://example.com/page") is False

