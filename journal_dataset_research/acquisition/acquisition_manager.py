"""
Access & Acquisition Manager

Handles dataset access requests, downloads, and secure storage for acquired datasets.
"""

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ai.journal_dataset_research.models.dataset_models import (
    AccessRequest,
    AcquiredDataset,
    DatasetSource,
)

logger = logging.getLogger(__name__)


@dataclass
class AcquisitionConfig:
    """Configuration for the acquisition manager."""

    # Storage configuration
    storage_base_path: str = "data/acquired_datasets"
    encryption_enabled: bool = False
    encryption_key: Optional[str] = None

    # Download configuration
    download_timeout: int = 3600  # 1 hour
    max_retries: int = 3
    retry_backoff_factor: float = 1.0
    chunk_size: int = 8192  # 8KB chunks
    resume_downloads: bool = True

    # API configuration
    api_timeout: int = 30
    rate_limit_delay: float = 1.0  # seconds between requests

    # Access request tracking
    follow_up_reminder_days: int = 7
    max_pending_days: int = 30

    # Repository API endpoints (can be extended)
    repository_apis: Dict[str, str] = field(
        default_factory=lambda: {
            "dryad": "https://datadryad.org/api/v2",
            "zenodo": "https://zenodo.org/api",
            "figshare": "https://api.figshare.com/v2",
        }
    )

    def validate(self) -> List[str]:
        """Validate the configuration and return list of errors."""
        errors = []
        if self.download_timeout <= 0:
            errors.append("download_timeout must be positive")
        if self.max_retries < 0:
            errors.append("max_retries must be non-negative")
        if self.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        if self.encryption_enabled and not self.encryption_key:
            errors.append("encryption_key is required when encryption_enabled is True")
        return errors


@dataclass
class DownloadProgress:
    """Tracks download progress."""

    source_id: str
    url: str
    total_bytes: Optional[int] = None
    downloaded_bytes: int = 0
    percentage: float = 0.0
    status: str = "pending"  # pending, downloading, completed, failed, paused
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    download_speed: float = 0.0  # bytes per second

    def update(
        self,
        downloaded_bytes: int,
        total_bytes: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update download progress."""
        self.downloaded_bytes = downloaded_bytes
        if total_bytes:
            self.total_bytes = total_bytes
            self.percentage = (downloaded_bytes / total_bytes) * 100.0
        else:
            self.percentage = 0.0

        if error_message:
            self.status = "failed"
            self.error_message = error_message
        elif self.total_bytes and self.downloaded_bytes >= self.total_bytes:
            self.status = "completed"
            if self.end_time is None:
                self.end_time = datetime.now()

        # Calculate download speed
        if self.start_time and self.downloaded_bytes > 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if elapsed > 0:
                self.download_speed = self.downloaded_bytes / elapsed


class AccessAcquisitionManager:
    """
    Manages dataset access requests, downloads, and secure storage.

    Handles:
    - Determining appropriate access methods
    - Direct downloads with resume capability
    - API-based dataset retrieval
    - Access request tracking
    - Secure storage and organization
    """

    def __init__(self, config: Optional[AcquisitionConfig] = None):
        """
        Initialize the acquisition manager.

        Args:
            config: Acquisition configuration. If None, uses default config.
        """
        self.config = config or AcquisitionConfig()
        config_errors = self.config.validate()
        if config_errors:
            raise ValueError(f"Invalid acquisition config: {', '.join(config_errors)}")

        # Create storage directory
        self.storage_path = Path(self.config.storage_base_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.storage_path / "datasets").mkdir(exist_ok=True)
        (self.storage_path / "metadata").mkdir(exist_ok=True)
        (self.storage_path / "logs").mkdir(exist_ok=True)

        # Initialize access request tracking
        self.access_requests: Dict[str, AccessRequest] = {}

        # Initialize download progress tracking
        self.download_progress: Dict[str, DownloadProgress] = {}

        # Setup requests session with retry strategy
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()

        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_backoff_factor,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def determine_access_method(self, source: DatasetSource) -> str:
        """
        Determine the appropriate access method for a dataset source.

        Args:
            source: The dataset source to analyze

        Returns:
            Access method: "direct", "api", "request_form", "collaboration", or "registration"
        """
        logger.info(f"Determining access method for {source.source_id}")

        # Check data availability
        if source.data_availability == "available":
            # Check if URL points to a direct download
            if source.url and self._is_direct_download_url(source.url):
                return "direct"

            # Check if URL points to a repository API
            if source.url and self._is_repository_api_url(source.url):
                return "api"

            # Check source type
            if source.source_type == "repository":
                return "api"
            elif source.source_type == "journal":
                # Journals often require registration or have request forms
                if source.open_access:
                    return "direct"
                else:
                    return "registration"

        elif source.data_availability == "upon_request":
            return "request_form"

        elif source.data_availability == "restricted":
            return "collaboration"

        else:  # unknown
            # Try to determine from URL and source type
            if source.url:
                if self._is_direct_download_url(source.url):
                    return "direct"
                elif self._is_repository_api_url(source.url):
                    return "api"
                else:
                    return "request_form"
            else:
                return "request_form"

        return "request_form"

    def _is_direct_download_url(self, url: str) -> bool:
        """Check if URL appears to be a direct download link."""
        direct_download_indicators = [
            ".zip",
            ".tar.gz",
            ".tar",
            ".csv",
            ".json",
            ".xml",
            ".parquet",
            "download",
            "file",
        ]
        url_lower = url.lower()
        return any(indicator in url_lower for indicator in direct_download_indicators)

    def _is_repository_api_url(self, url: str) -> bool:
        """Check if URL points to a repository API."""
        api_indicators = ["/api/", "api.", "datadryad.org", "zenodo.org", "figshare.com"]
        url_lower = url.lower()
        return any(indicator in url_lower for indicator in api_indicators)

    def submit_access_request(
        self,
        source: DatasetSource,
        access_method: Optional[str] = None,
        notes: str = "",
    ) -> AccessRequest:
        """
        Submit an access request for a dataset.

        Args:
            source: The dataset source
            access_method: Optional access method (will be determined if not provided)
            notes: Additional notes for the request

        Returns:
            AccessRequest object
        """
        if access_method is None:
            access_method = self.determine_access_method(source)

        # Check if credentials or institutional affiliation are required
        credentials_required = access_method in ["api", "registration", "collaboration"]
        institutional_affiliation_required = access_method == "collaboration"

        # Estimate access date based on method
        estimated_access_date = None
        if access_method == "direct":
            estimated_access_date = datetime.now() + timedelta(hours=1)
        elif access_method == "api":
            estimated_access_date = datetime.now() + timedelta(hours=24)
        elif access_method == "request_form":
            estimated_access_date = datetime.now() + timedelta(days=7)
        elif access_method == "collaboration":
            estimated_access_date = datetime.now() + timedelta(days=30)
        elif access_method == "registration":
            estimated_access_date = datetime.now() + timedelta(days=2)

        access_request = AccessRequest(
            source_id=source.source_id,
            access_method=access_method,
            request_date=datetime.now(),
            status="pending",
            access_url=source.url,
            credentials_required=credentials_required,
            institutional_affiliation_required=institutional_affiliation_required,
            estimated_access_date=estimated_access_date,
            notes=notes,
        )

        # Validate request
        request_errors = access_request.validate()
        if request_errors:
            raise ValueError(f"Invalid access request: {', '.join(request_errors)}")

        # Store request
        self.access_requests[source.source_id] = access_request

        logger.info(
            f"Access request submitted for {source.source_id}: "
            f"method={access_method}, status={access_request.status}"
        )

        return access_request

    def download_dataset(
        self,
        source: DatasetSource,
        access_request: Optional[AccessRequest] = None,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> AcquiredDataset:
        """
        Download a dataset from a source.

        Args:
            source: The dataset source to download
            access_request: Optional access request (will be created if not provided)
            progress_callback: Optional callback for download progress updates

        Returns:
            AcquiredDataset object

        Raises:
            ValueError: If download fails or source is invalid
            requests.RequestException: If HTTP request fails
        """
        logger.info(f"Downloading dataset: {source.source_id} - {source.title}")

        if not source.url:
            raise ValueError(f"No URL provided for dataset {source.source_id}")

        if access_request is None:
            access_request = self.submit_access_request(source)

        # Check if download is possible
        if access_request.access_method not in ["direct", "api"]:
            raise ValueError(
                f"Download not possible for access method: {access_request.access_method}"
            )

        # Initialize download progress
        progress = DownloadProgress(
            source_id=source.source_id,
            url=source.url,
            status="downloading",
            start_time=datetime.now(),
        )
        self.download_progress[source.source_id] = progress

        try:
            if access_request.access_method == "direct":
                file_path = self._download_direct(
                    source, access_request, progress, progress_callback
                )
            else:  # api
                file_path = self._download_via_api(
                    source, access_request, progress, progress_callback
                )

            # Calculate file size and checksum
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            checksum = self._calculate_checksum(file_path)

            # Create acquired dataset record
            acquired_dataset = AcquiredDataset(
                source_id=source.source_id,
                acquisition_date=datetime.now(),
                storage_path=str(file_path),
                file_format=file_path.suffix[1:] if file_path.suffix else "unknown",
                file_size_mb=file_size_mb,
                license="",  # Will be populated from source metadata
                usage_restrictions=[],
                attribution_required=bool(source.doi),
                checksum=checksum,
            )

            # Update access request status
            access_request.status = "downloaded"

            # Organize storage
            organized_path = self.organize_storage(acquired_dataset, source)
            acquired_dataset.storage_path = str(organized_path)

            # Save metadata
            self._save_dataset_metadata(acquired_dataset, source, access_request)

            logger.info(
                f"Dataset downloaded successfully: {source.source_id}, "
                f"size={file_size_mb:.2f}MB, checksum={checksum[:8]}"
            )

            return acquired_dataset

        except Exception as e:
            progress.status = "failed"
            progress.error_message = str(e)
            access_request.status = "error"
            logger.error(f"Download failed for {source.source_id}: {e}")
            raise

        finally:
            progress.end_time = datetime.now()

    def _download_direct(
        self,
        source: DatasetSource,
        access_request: AccessRequest,
        progress: DownloadProgress,
        progress_callback: Optional[Callable[[DownloadProgress], None]],
    ) -> Path:
        """Download dataset directly from URL."""
        url = access_request.access_url or source.url

        # Check for existing partial download
        temp_path = self.storage_path / "datasets" / f"{source.source_id}.tmp"
        final_path = self.storage_path / "datasets" / f"{source.source_id}"

        # Get file info
        head_response = self.session.head(url, timeout=self.config.api_timeout)
        total_bytes = int(head_response.headers.get("Content-Length", 0))

        progress.total_bytes = total_bytes

        # Resume download if enabled and file exists
        resume_pos = 0
        if self.config.resume_downloads and temp_path.exists():
            resume_pos = temp_path.stat().st_size
            if resume_pos < total_bytes:
                logger.info(f"Resuming download from byte {resume_pos}")

        # Download with progress tracking
        headers = {}
        if resume_pos > 0:
            headers["Range"] = f"bytes={resume_pos}-"

        response = self.session.get(
            url,
            stream=True,
            headers=headers,
            timeout=self.config.download_timeout,
        )
        response.raise_for_status()

        # Update total bytes from response
        if "Content-Length" in response.headers:
            total_bytes = int(response.headers["Content-Length"]) + resume_pos
            progress.total_bytes = total_bytes

        # Download in chunks
        mode = "ab" if resume_pos > 0 else "wb"
        with open(temp_path, mode) as f:
            for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                if chunk:
                    f.write(chunk)
                    progress.update(
                        progress.downloaded_bytes + len(chunk), total_bytes=total_bytes
                    )
                    if progress_callback:
                        progress_callback(progress)

        # Move temp file to final location
        if final_path.exists():
            final_path.unlink()
        shutil.move(temp_path, final_path)

        return final_path

    def _download_via_api(
        self,
        source: DatasetSource,
        access_request: AccessRequest,
        progress: DownloadProgress,
        progress_callback: Optional[Callable[[DownloadProgress], None]],
    ) -> Path:
        """Download dataset via repository API."""
        # This is a simplified implementation
        # In a full implementation, this would:
        # 1. Identify the repository type (Dryad, Zenodo, etc.)
        # 2. Use the appropriate API client
        # 3. Authenticate if needed
        # 4. Retrieve dataset download URL
        # 5. Download the dataset

        # For now, treat API URLs as direct downloads
        # In production, implement repository-specific API clients
        logger.warning(
            f"API download not fully implemented for {source.source_id}, "
            "falling back to direct download"
        )
        return self._download_direct(source, access_request, progress, progress_callback)

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def verify_download_integrity(self, dataset: AcquiredDataset) -> bool:
        """
        Verify the integrity of a downloaded dataset.

        Args:
            dataset: The acquired dataset to verify

        Returns:
            True if integrity is verified, False otherwise
        """
        file_path = Path(dataset.storage_path)
        if not file_path.exists():
            logger.error(f"Dataset file not found: {dataset.storage_path}")
            return False

        calculated_checksum = self._calculate_checksum(file_path)
        if calculated_checksum != dataset.checksum:
            logger.error(
                f"Checksum mismatch for {dataset.source_id}: "
                f"expected={dataset.checksum[:8]}, got={calculated_checksum[:8]}"
            )
            return False

        logger.info(f"Integrity verified for {dataset.source_id}")
        return True

    def organize_storage(
        self, dataset: AcquiredDataset, source: DatasetSource
    ) -> Path:
        """
        Organize storage for an acquired dataset.

        Args:
            dataset: The acquired dataset
            source: The dataset source

        Returns:
            Path to the organized storage location
        """
        # Create organized directory structure: source_type/year/month/dataset_id
        acquisition_date = dataset.acquisition_date
        organized_dir = (
            self.storage_path
            / "datasets"
            / source.source_type
            / str(acquisition_date.year)
            / f"{acquisition_date.month:02d}"
        )
        organized_dir.mkdir(parents=True, exist_ok=True)

        # Move file to organized location
        current_path = Path(dataset.storage_path)
        file_extension = current_path.suffix
        organized_path = organized_dir / f"{source.source_id}{file_extension}"

        if current_path != organized_path:
            if organized_path.exists():
                organized_path.unlink()
            shutil.move(current_path, organized_path)
            logger.info(
                f"Organized storage: {current_path} -> {organized_path}"
            )

        return organized_path

    def _save_dataset_metadata(
        self,
        dataset: AcquiredDataset,
        source: DatasetSource,
        access_request: AccessRequest,
    ) -> None:
        """Save dataset metadata to file."""
        import json

        metadata = {
            "source_id": dataset.source_id,
            "acquisition_date": dataset.acquisition_date.isoformat(),
            "storage_path": dataset.storage_path,
            "file_format": dataset.file_format,
            "file_size_mb": dataset.file_size_mb,
            "checksum": dataset.checksum,
            "source": {
                "title": source.title,
                "authors": source.authors,
                "doi": source.doi,
                "url": source.url,
                "source_type": source.source_type,
            },
            "access_request": {
                "access_method": access_request.access_method,
                "request_date": access_request.request_date.isoformat(),
                "status": access_request.status,
            },
        }

        metadata_path = (
            self.storage_path / "metadata" / f"{dataset.source_id}.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def get_access_request(self, source_id: str) -> Optional[AccessRequest]:
        """Get an access request by source ID."""
        return self.access_requests.get(source_id)

    def update_access_request_status(
        self, source_id: str, status: str, notes: str = ""
    ) -> None:
        """
        Update the status of an access request.

        Args:
            source_id: The source ID
            status: New status
            notes: Additional notes
        """
        if source_id not in self.access_requests:
            raise ValueError(f"Access request not found: {source_id}")

        access_request = self.access_requests[source_id]
        access_request.status = status
        if notes:
            access_request.notes = f"{access_request.notes}\n{notes}".strip()

        logger.info(f"Access request updated: {source_id} -> {status}")

    def get_download_progress(self, source_id: str) -> Optional[DownloadProgress]:
        """Get download progress for a source ID."""
        return self.download_progress.get(source_id)

    def list_access_requests(
        self, status: Optional[str] = None
    ) -> List[AccessRequest]:
        """
        List access requests, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of access requests
        """
        requests_list = list(self.access_requests.values())
        if status:
            requests_list = [r for r in requests_list if r.status == status]
        return requests_list

    def get_pending_follow_ups(self) -> List[AccessRequest]:
        """
        Get access requests that need follow-up.

        Returns:
            List of access requests needing follow-up
        """
        pending_requests = []
        cutoff_date = datetime.now() - timedelta(days=self.config.follow_up_reminder_days)

        for request in self.access_requests.values():
            if request.status == "pending":
                # Check if request is old enough to need follow-up
                if request.request_date < cutoff_date:
                    pending_requests.append(request)

        return pending_requests

    def generate_access_request_report(self) -> str:
        """
        Generate a report of all access requests.

        Returns:
            Markdown-formatted report
        """
        report_lines = [
            "# Access Request Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
        ]

        # Count by status
        status_counts = {}
        for request in self.access_requests.values():
            status_counts[request.status] = status_counts.get(request.status, 0) + 1

        report_lines.append(f"- **Total Requests**: {len(self.access_requests)}")
        for status, count in status_counts.items():
            report_lines.append(f"- **{status.capitalize()}**: {count}")
        report_lines.append("")

        # List requests by status
        for status in ["pending", "approved", "downloaded", "denied", "error"]:
            status_requests = [
                r for r in self.access_requests.values() if r.status == status
            ]
            if status_requests:
                report_lines.extend(
                    [
                        f"## {status.capitalize()} Requests",
                        "",
                    ]
                )
                for request in status_requests:
                    report_lines.extend(
                        [
                            f"### {request.source_id}",
                            f"- **Method**: {request.access_method}",
                            f"- **Request Date**: {request.request_date.strftime('%Y-%m-%d')}",
                            f"- **Status**: {request.status}",
                        ]
                    )
                    if request.estimated_access_date:
                        report_lines.append(
                            f"- **Estimated Access**: {request.estimated_access_date.strftime('%Y-%m-%d')}"
                        )
                    if request.notes:
                        report_lines.append(f"- **Notes**: {request.notes}")
                    report_lines.append("")

        return "\n".join(report_lines)

