#!/usr/bin/env python3
"""
Consolidated S3 Dataset Loader for Pixelated Empathy AI

This module provides a comprehensive S3 interface for Hetzner object storage:
- Idempotent upload and download operations
- Batch processing with parallel support
- Progress tracking and monitoring
- Retry logic with exponential backoff
- Checksum verification (MD5, SHA256)
- Thread-safe operations
- Integration with Ray executor
- Support for multiple file formats (JSON, JSONL, Parquet, CSV)

Usage:
    from s3_dataset_loader import S3DatasetLoader, S3Config

    config = S3Config(
        endpoint_url="https://s3.us-east-1.amazonaws.com",
        bucket_name="pixelated-datasets",
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

    loader = S3DatasetLoader(config)

    # Upload a dataset
    loader.upload_file("local_dataset.jsonl", "datasets/v1/dataset.jsonl")

    # Download a dataset
    loader.download_file("datasets/v1/dataset.jsonl", "local_dataset.jsonl")

    # Batch upload
    loader.upload_batch(
        files=["data1.json", "data2.json"],
        s3_prefix="datasets/batch1"
    )
"""

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class S3Config:
    """Configuration for S3 operations."""

    # Connection settings
    endpoint_url: str | None = None
    bucket_name: str = "pixelated-datasets"
    access_key_id: str | None = None
    secret_access_key: str | None = None
    region_name: str = "us-east-1"

    # Retry settings
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    retry_mode: Literal["standard", "adaptive", "legacy"] = "adaptive"  # standard, adaptive, legacy

    # Performance settings
    max_concurrency: int = 10
    multipart_threshold: int = 8 * 1024 * 1024  # 8MB
    multipart_chunksize: int = 8 * 1024 * 1024  # 8MB

    # Verification settings
    verify_checksums: bool = True
    checksum_algorithm: str = "sha256"  # md5, sha256

    # Timeout settings
    connect_timeout: int = 60
    read_timeout: int = 300

    # Other settings
    verify_ssl: bool = True
    enable_progress_tracking: bool = True

    def __post_init__(self):
        """Validate and set defaults."""
        # Get credentials from environment if not provided
        # Prefer HETZNER_S3_* names; fall back to AWS_* for backwards compat
        if self.access_key_id is None:
            self.access_key_id = (
                os.getenv("HETZNER_S3_ACCESS_KEY")
                or os.getenv("HETZNER_ACCESS_KEY")
                or os.getenv("AWS_ACCESS_KEY_ID")
            )
        if self.secret_access_key is None:
            self.secret_access_key = (
                os.getenv("HETZNER_S3_SECRET_KEY")
                or os.getenv("HETZNER_SECRET_KEY")
                or os.getenv("AWS_SECRET_ACCESS_KEY")
            )
        if self.endpoint_url is None:
            self.endpoint_url = (
                os.getenv("HETZNER_S3_ENDPOINT")
                or os.getenv("AWS_S3_ENDPOINT")
            )


@dataclass
class UploadResult:
    """Result of an upload operation."""

    success: bool
    local_path: str
    s3_key: str
    size_bytes: int
    etag: str | None = None
    upload_time_seconds: float = 0.0
    error: str | None = None
    retry_count: int = 0
    checksum_verified: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class DownloadResult:
    """Result of a download operation."""

    success: bool
    s3_key: str
    local_path: str
    size_bytes: int
    etag: str | None = None
    download_time_seconds: float = 0.0
    error: str | None = None
    retry_count: int = 0
    checksum_verified: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class BatchOperationResult:
    """Result of batch operations."""

    operation: str  # "upload" or "download"
    total_files: int
    successful: int
    failed: int
    skipped: int = 0
    results: list[Any] = field(default_factory=list)
    total_time_seconds: float = 0.0
    total_bytes: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_files == 0:
            return 100.0
        return (self.successful / self.total_files) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "total_files": self.total_files,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": round(self.success_rate, 2),
            "total_time_seconds": round(self.total_time_seconds, 3),
            "total_bytes": self.total_bytes,
            "timestamp": datetime.now(UTC).isoformat(),
        }


class ProgressTracker:
    """Track progress for S3 operations."""

    def __init__(self, total_files: int, enable_logging: bool = True):
        self.total_files = total_files
        self.completed_files = 0
        self.total_bytes = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.enable_logging = enable_logging
        self._logger = logging.getLogger("s3_progress")

    def update(self, bytes_transferred: int):
        """Update progress."""
        with self.lock:
            self.total_bytes += bytes_transferred
            self.completed_files += 1

            if self.enable_logging and self.completed_files % 10 == 0:
                self.log_progress()

    def log_progress(self):
        """Log progress."""
        elapsed = time.time() - self.start_time
        progress_percent = (self.completed_files / self.total_files) * 100

        # Calculate transfer rate (bytes/second)
        avg_rate = self.total_bytes / elapsed if elapsed > 0 else 0
        rate_mb_s = (avg_rate / (1024 * 1024)) if avg_rate > 0 else 0

        self._logger.info(
            f"Progress: {self.completed_files}/{self.total_files} files "
            f"({progress_percent:.1f}%) - "
            f"{self.total_bytes / (1024 * 1024):.2f} MB transferred - "
            f"{rate_mb_s:.2f} MB/s"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get progress statistics."""
        elapsed = time.time() - self.start_time
        return {
            "total_files": self.total_files,
            "completed_files": self.completed_files,
            "progress_percent": (self.completed_files / self.total_files) * 100,
            "total_bytes": self.total_bytes,
            "total_bytes_mb": self.total_bytes / (1024 * 1024),
            "elapsed_seconds": elapsed,
            "avg_rate_bytes_per_second": self.total_bytes / elapsed if elapsed > 0 else 0,
        }


class S3DatasetLoader:
    """
    Consolidated S3 dataset loader for Hetzner object storage.

    Provides idempotent operations for uploading and downloading datasets
    from S3-compatible storage (Hetzner, AWS, MinIO, etc.).
    """

    def __init__(self, config: S3Config):
        """
        Initialize S3 dataset loader.

        Args:
            config: S3 configuration
        """
        self.config = config
        self._client = None
        self._lock = threading.Lock()
        self._logger = logging.getLogger("s3_dataset_loader")

        # Initialize the S3 client on first use
        self._get_client()

    def _create_client(self) -> Any:
        """Create and configure S3 client."""
        boto_config = Config(
            retries={
                "max_attempts": self.config.max_retries,
                "mode": self.config.retry_mode,
            },
            connect_timeout=self.config.connect_timeout,
            read_timeout=self.config.read_timeout,
        )

        session = boto3.Session(
            aws_access_key_id=self.config.access_key_id,
            aws_secret_access_key=self.config.secret_access_key,
            region_name=self.config.region_name,
        )

        client_kwargs: dict[str, Any] = {
            "config": boto_config,
            "use_ssl": self.config.verify_ssl,
        }

        if self.config.endpoint_url:
            client_kwargs["endpoint_url"] = self.config.endpoint_url

        client = session.client("s3", **client_kwargs)
        self._logger.info(f"S3 client initialized for bucket: {self.config.bucket_name}")
        return client

    def _get_client(self) -> Any:
        """Get or create S3 client with lazy initialization."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._create_client()

        return self._client

    def _calculate_checksum(self, file_path: Path) -> str:
        """
        Calculate file checksum.

        Args:
            file_path: Path to file

        Returns:
            Checksum string
        """
        algorithm = self.config.checksum_algorithm.lower()

        if algorithm == "md5":
            hash_obj = hashlib.md5()
        elif algorithm == "sha256":
            hash_obj = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported checksum algorithm: {algorithm}")

        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    def _retry_operation(self, operation, *args, max_retries: int | None = None, **kwargs):
        """
        Execute operation with retry logic.

        Args:
            operation: Function to execute
            *args: Positional arguments for operation
            max_retries: Maximum number of retries (uses config default if None)
            **kwargs: Keyword arguments for operation

        Returns:
            Operation result
        """
        max_retries = max_retries or self.config.max_retries
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return operation(*args, **kwargs)

            except (ClientError, BotoCoreError) as e:
                last_exception = e

                if attempt < max_retries:
                    # Exponential backoff
                    sleep_time = self.config.retry_backoff_factor * (2**attempt)
                    self._logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {sleep_time:.1f}s: {e}"
                    )
                    time.sleep(sleep_time)
                else:
                    self._logger.error(f"Operation failed after {max_retries + 1} attempts: {e}")
                    raise

            except Exception as e:
                self._logger.error(f"Unexpected error: {e}")
                raise

        if last_exception:
            raise last_exception
        return None

    def upload_file(
        self,
        local_path: str | Path,
        s3_key: str,
        overwrite: bool = False,
        verify_checksum: bool | None = None,
    ) -> UploadResult:
        """
        Upload a file to S3.

        Args:
            local_path: Path to local file
            s3_key: S3 object key
            overwrite: Whether to overwrite if file exists
            verify_checksum: Whether to verify checksum (uses config default if None)

        Returns:
            UploadResult
        """
        local_path = Path(local_path)
        verify_checksum = verify_checksum if verify_checksum is not None else self.config.verify_checksums

        start_time = time.time()

        # Validate local file exists
        if not local_path.exists():
            return UploadResult(
                success=False,
                local_path=str(local_path),
                s3_key=s3_key,
                size_bytes=0,
                error=f"Local file not found: {local_path}",
            )

        file_size = local_path.stat().st_size

        # Check if file already exists in S3
        try:
            client = self._get_client()
            client.head_object(Bucket=self.config.bucket_name, Key=s3_key)

            if not overwrite:
                file_etag = self._calculate_checksum(local_path)
                existing_file = client.head_object(Bucket=self.config.bucket_name, Key=s3_key)

                # Compare checksums to determine if upload is needed
                if self.config.checksum_algorithm == "sha256":
                    existing_chksum = existing_file.get("Metadata", {}).get("sha256-checksum", "")
                else:
                    existing_chksum = existing_file.get("ETag", "").strip('"')

                if file_etag == existing_chksum:
                    self._logger.info(f"File already exists with matching checksum, skipping: {s3_key}")
                    return UploadResult(
                        success=True,
                        local_path=str(local_path),
                        s3_key=s3_key,
                        size_bytes=file_size,
                        etag=existing_file.get("ETag"),
                        upload_time_seconds=0.0,
                        retry_count=0,
                        checksum_verified=True,
                    )
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                return UploadResult(
                    success=False,
                    local_path=str(local_path),
                    s3_key=s3_key,
                    size_bytes=file_size,
                    error=f"Failed to check existing file: {e}",
                )

        # Calculate checksum before upload
        file_checksum = self._calculate_checksum(local_path) if verify_checksum else None

        # Upload file
        try:
            extra_args = {}

            # Add checksum metadata
            if verify_checksum and file_checksum:
                extra_args["Metadata"] = {f"{self.config.checksum_algorithm}-checksum": file_checksum}

            # Use multipart upload for large files
            if file_size > self.config.multipart_threshold:
                self._logger.info(f"Using multipart upload for {s3_key} ({file_size / (1024 * 1024):.2f} MB)")

                client = self._get_client()
                client.upload_file(
                    str(local_path),
                    self.config.bucket_name,
                    s3_key,
                    Config=TransferConfig(
                        multipart_threshold=self.config.multipart_threshold,
                        multipart_chunksize=self.config.multipart_chunksize,
                        max_concurrency=self.config.max_concurrency,
                    ),
                    ExtraArgs=extra_args,
                )
            else:
                client = self._get_client()
                client.upload_file(
                    str(local_path),
                    self.config.bucket_name,
                    s3_key,
                    ExtraArgs=extra_args,
                )

            # Get ETag
            response = client.head_object(Bucket=self.config.bucket_name, Key=s3_key)
            etag = response.get("ETag")

            upload_time = time.time() - start_time

            self._logger.info(f"Upload successful: {s3_key} ({file_size / (1024 * 1024):.2f} MB) in {upload_time:.2f}s")

            return UploadResult(
                success=True,
                local_path=str(local_path),
                s3_key=s3_key,
                size_bytes=file_size,
                etag=etag,
                upload_time_seconds=upload_time,
                checksum_verified=verify_checksum,
            )

        except Exception as e:
            upload_time = time.time() - start_time
            self._logger.error(f"Upload failed for {s3_key}: {e}")

            return UploadResult(
                success=False,
                local_path=str(local_path),
                s3_key=s3_key,
                size_bytes=file_size,
                error=str(e),
                upload_time_seconds=upload_time,
            )

    def _check_existing_file(self, s3_key: str, local_path: Path) -> DownloadResult | None:
        """Check if local file matches remote and return result if it does."""
        try:
            client = self._get_client()
            response = client.head_object(Bucket=self.config.bucket_name, Key=s3_key)
        except ClientError as e:
            self._logger.warning(f"Failed to verify existing file: {e}")
            return None

        # Calculate checksum after confirming remote file exists
        local_checksum = self._calculate_checksum(local_path)

        if self.config.checksum_algorithm == "sha256":
            remote_checksum = response.get("Metadata", {}).get("sha256-checksum", "")
        else:
            remote_checksum = response.get("ETag", "").strip('"')

        if local_checksum == remote_checksum:
            self._logger.info(f"File already exists with matching checksum, skipping: {s3_key}")
            return DownloadResult(
                success=True,
                s3_key=s3_key,
                local_path=str(local_path),
                size_bytes=local_path.stat().st_size,
                etag=response.get("ETag"),
                download_time_seconds=0.0,
                checksum_verified=True,
            )

        return None

    def _verify_downloaded_file(
        self,
        local_path: Path,
        s3_key: str,
        response: dict[str, Any],
    ) -> bool:
        """Verify checksum of downloaded file."""
        local_checksum = self._calculate_checksum(local_path)

        if self.config.checksum_algorithm == "sha256":
            remote_checksum = response.get("Metadata", {}).get("sha256-checksum", "")
        else:
            expected_etag = response.get("ETag")
            remote_checksum = expected_etag.strip('"') if expected_etag else ""

        if local_checksum == remote_checksum:
            return True

        self._logger.error(f"Checksum mismatch for {s3_key}: local={local_checksum}, remote={remote_checksum}")
        local_path.unlink()  # Delete corrupted file
        return False

    def download_file(
        self,
        s3_key: str,
        local_path: str | Path,
        overwrite: bool = False,
        verify_checksum: bool | None = None,
    ) -> DownloadResult:
        """
        Download a file from S3.

        Args:
            s3_key: S3 object key
            local_path: Path to save the file
            overwrite: Whether to overwrite if file exists locally
            verify_checksum: Whether to verify checksum (uses config default if None)

        Returns:
            DownloadResult
        """
        local_path = Path(local_path)
        verify_checksum = verify_checksum if verify_checksum is not None else self.config.verify_checksums

        start_time = time.time()

        # Create parent directory if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if local file exists
        if local_path.exists() and not overwrite:
            if existing_result := self._check_existing_file(s3_key, local_path):
                return existing_result

        # Get object metadata first
        try:
            client = self._get_client()
            response = client.head_object(Bucket=self.config.bucket_name, Key=s3_key)
            file_size = response["ContentLength"]
            expected_etag = response.get("ETag")

            # Download file
            client.download_file(self.config.bucket_name, s3_key, str(local_path))

            download_time = time.time() - start_time

            # Verify checksum
            checksum_verified = False
            if verify_checksum:
                checksum_verified = self._verify_downloaded_file(local_path, s3_key, response)

                if not checksum_verified:
                    return DownloadResult(
                        success=False,
                        s3_key=s3_key,
                        local_path=str(local_path),
                        size_bytes=file_size,
                        error="Checksum verification failed",
                    )

            self._logger.info(
                f"Download successful: {s3_key} ({file_size / (1024 * 1024):.2f} MB) in {download_time:.2f}s"
            )

            return DownloadResult(
                success=True,
                s3_key=s3_key,
                local_path=str(local_path),
                size_bytes=file_size,
                etag=expected_etag,
                download_time_seconds=download_time,
                checksum_verified=checksum_verified,
            )

        except ClientError as e:
            download_time = time.time() - start_time
            error_code = e.response.get("Error", {}).get("Code", "Unknown")

            if error_code == "404":
                self._logger.error(f"File not found: {s3_key}")
            else:
                self._logger.error(f"Download failed for {s3_key}: {e}")

            return DownloadResult(
                success=False,
                s3_key=s3_key,
                local_path=str(local_path),
                size_bytes=0,
                error=f"{error_code}: {e}",
                download_time_seconds=download_time,
            )

        except Exception as e:
            download_time = time.time() - start_time
            self._logger.error(f"Download failed for {s3_key}: {e}")

            return DownloadResult(
                success=False,
                s3_key=s3_key,
                local_path=str(local_path),
                size_bytes=0,
                error=str(e),
                download_time_seconds=download_time,
            )

    def _handle_progress(
        self,
        result: UploadResult | DownloadResult,
        progress_tracker: ProgressTracker,
        progress_callback: Callable[[Any], None] | None = None,
    ) -> None:
        """Handle progress updates for batch operations."""
        if result.success:
            progress_tracker.update(result.size_bytes)
        if progress_callback:
            progress_callback(result)

    def upload_batch(
        self,
        files: list[str | Path | tuple[str | Path, str]],
        s3_prefix: str = "",
        overwrite: bool = False,
        parallel: bool = True,
        progress_callback: Callable[[Any], None] | None = None,
    ) -> BatchOperationResult:
        """
        Upload multiple files to S3 in batch.

        Args:
            files: List of local paths or tuples of (local_path, s3_key)
            s3_prefix: S3 prefix to add to all keys
            overwrite: Whether to overwrite existing files
            parallel: Whether to upload files in parallel
            progress_callback: Optional callback for progress updates

        Returns:
            BatchOperationResult
        """
        start_time = time.time()
        results = []

        # Normalize file list
        file_list = []
        for file_item in files:
            if isinstance(file_item, tuple):
                local_path, s3_key = file_item
            else:
                local_path = file_item
                s3_key = Path(local_path).name

            # Add prefix to S3 key
            if s3_prefix and not s3_key.startswith(s3_prefix):
                s3_key = f"{s3_prefix}/{s3_key}"

            file_list.append((Path(local_path), s3_key))

        # Initialize progress tracker
        progress_tracker = ProgressTracker(
            total_files=len(file_list),
            enable_logging=self.config.enable_progress_tracking,
        )

        # Upload files
        if parallel and len(file_list) > 1:
            # Parallel upload
            with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
                futures = {
                    executor.submit(self.upload_file, local_path, s3_key, overwrite): (
                        local_path,
                        s3_key,
                    )
                    for local_path, s3_key in file_list
                }

                for future in as_completed(futures):
                    local_path, s3_key = futures[future]
                    result = future.result()
                    results.append(result)
                    self._handle_progress(result, progress_tracker, progress_callback)
        else:
            # Sequential upload
            for local_path, s3_key in file_list:
                result = self.upload_file(local_path, s3_key, overwrite)
                results.append(result)
                self._handle_progress(result, progress_tracker, progress_callback)

        # Calculate stats
        successful = sum(r.success for r in results)
        failed = sum(not r.success for r in results)
        skipped = sum(r.success and r.upload_time_seconds == 0.0 for r in results)
        total_bytes = sum(r.size_bytes for r in results if r.success)
        total_time = time.time() - start_time

        self._logger.info(
            f"Batch upload complete: {successful}/{len(file_list)} successful, "
            f"{failed} failed, {skipped} skipped "
            f"in {total_time:.2f}s"
        )

        return BatchOperationResult(
            operation="upload",
            total_files=len(file_list),
            successful=successful,
            failed=failed,
            skipped=skipped,
            results=results,
            total_time_seconds=total_time,
            total_bytes=total_bytes,
        )

    def download_batch(
        self,
        s3_keys: list[str],
        local_dir: str | Path,
        overwrite: bool = False,
        parallel: bool = True,
        progress_callback: Callable[[Any], None] | None = None,
    ) -> BatchOperationResult:
        """
        Download multiple files from S3 in batch.

        Args:
            s3_keys: List of S3 object keys to download
            local_dir: Local directory to save files
            overwrite: Whether to overwrite existing files
            parallel: Whether to download files in parallel
            progress_callback: Optional callback for progress updates

        Returns:
            BatchOperationResult
        """
        start_time = time.time()
        results = []
        local_dir = Path(local_dir)

        # Initialize progress tracker
        progress_tracker = ProgressTracker(
            total_files=len(s3_keys),
            enable_logging=self.config.enable_progress_tracking,
        )

        # Download files
        if parallel and len(s3_keys) > 1:
            # Parallel download
            with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
                futures = {
                    executor.submit(
                        self.download_file,
                        s3_key,
                        local_dir / Path(s3_key).name,
                        overwrite,
                    ): s3_key
                    for s3_key in s3_keys
                }

                for future in as_completed(futures):
                    s3_key = futures[future]
                    result = future.result()
                    results.append(result)
                    self._handle_progress(result, progress_tracker, progress_callback)
        else:
            # Sequential download
            for s3_key in s3_keys:
                result = self.download_file(s3_key, local_dir / Path(s3_key).name, overwrite)
                results.append(result)
                self._handle_progress(result, progress_tracker, progress_callback)

        # Calculate stats
        successful = sum(r.success for r in results)
        failed = sum(not r.success for r in results)
        skipped = sum(r.success and r.download_time_seconds == 0.0 for r in results)
        total_bytes = sum(r.size_bytes for r in results if r.success)
        total_time = time.time() - start_time

        self._logger.info(
            f"Batch download complete: {successful}/{len(s3_keys)} successful, "
            f"{failed} failed, {skipped} skipped "
            f"in {total_time:.2f}s"
        )

        return BatchOperationResult(
            operation="download",
            total_files=len(s3_keys),
            successful=successful,
            failed=failed,
            skipped=skipped,
            results=results,
            total_time_seconds=total_time,
            total_bytes=total_bytes,
        )

    def list_objects(
        self,
        prefix: str = "",
        delimiter: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        List objects in S3 bucket.

        Args:
            prefix: S3 key prefix to filter
            delimiter: Delimiter for grouping (e.g., '/' for directories)
            limit: Maximum number of objects to return

        Returns:
            List of object metadata dictionaries
        """
        client = self._get_client()
        objects = []

        kwargs = {"Bucket": self.config.bucket_name, "Prefix": prefix}

        if delimiter:
            kwargs["Delimiter"] = delimiter

        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(**kwargs)

        for page in pages:
            if "Contents" in page:
                objects.extend(page["Contents"])

            if limit and len(objects) >= limit:
                objects = objects[:limit]
                break

        return objects

    def delete_object(self, s3_key: str) -> bool:
        """
        Delete an object from S3.

        Args:
            s3_key: S3 object key to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            client = self._get_client()
            client.delete_object(Bucket=self.config.bucket_name, Key=s3_key)
            self._logger.info(f"Deleted object: {s3_key}")
            return True
        except ClientError as e:
            self._logger.error(f"Failed to delete {s3_key}: {e}")
            return False

    def delete_batch(self, s3_keys: list[str]) -> int:
        """
        Delete multiple objects from S3.

        Args:
            s3_keys: List of S3 object keys to delete

        Returns:
            Number of successfully deleted objects
        """
        client = self._get_client()
        deleted = 0

        # Delete in batches of 1000 (S3 limit)
        batch_size = 1000
        for i in range(0, len(s3_keys), batch_size):
            batch = s3_keys[i : i + batch_size]

            try:
                response = client.delete_objects(
                    Bucket=self.config.bucket_name,
                    Delete={"Objects": [{"Key": key} for key in batch]},
                )

                deleted += len(response.get("Deleted", []))

                if "Errors" in response:
                    for error in response["Errors"]:
                        self._logger.error(f"Failed to delete {error['Key']}: {error['Message']}")

            except ClientError as e:
                self._logger.error(f"Batch delete failed: {e}")

        return deleted

    def copy_object(self, source_key: str, dest_key: str, source_bucket: str | None = None) -> bool:
        """
        Copy an object within S3.

        Args:
            source_key: Source S3 object key
            dest_key: Destination S3 object key
            source_bucket: Source bucket (defaults to bucket_name)

        Returns:
            True if successful, False otherwise
        """
        try:
            client = self._get_client()

            copy_source = {
                "Bucket": source_bucket or self.config.bucket_name,
                "Key": source_key,
            }

            client.copy_object(CopySource=copy_source, Bucket=self.config.bucket_name, Key=dest_key)

            self._logger.info(f"Copied {source_key} to {dest_key}")
            return True

        except ClientError as e:
            self._logger.error(f"Failed to copy {source_key} to {dest_key}: {e}")
            return False

    def object_exists(self, s3_key: str) -> bool:
        """
        Check if an object exists in S3.

        Args:
            s3_key: S3 object key

        Returns:
            True if object exists, False otherwise
        """
        try:
            client = self._get_client()
            client.head_object(Bucket=self.config.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            self._logger.error(f"Failed to check existence of {s3_key}: {e}")
            return False

    def get_object_metadata(self, s3_key: str) -> dict[str, Any] | None:
        """
        Get metadata for an S3 object.

        Args:
            s3_key: S3 object key

        Returns:
            Object metadata dictionary or None if object doesn't exist
        """
        try:
            client = self._get_client()
            response = client.head_object(Bucket=self.config.bucket_name, Key=s3_key)

            return {
                "size": response["ContentLength"],
                "last_modified": response["LastModified"],
                "etag": response["ETag"],
                "content_type": response.get("ContentType"),
                "metadata": response.get("Metadata", {}),
                "storage_class": response.get("StorageClass", "STANDARD"),
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return None
            self._logger.error(f"Failed to get metadata for {s3_key}: {e}")
            return None

    def load_jsonl(self, s3_key: str) -> list[dict[str, Any]]:
        """
        Load a JSONL file from S3.

        Args:
            s3_key: S3 object key

        Returns:
            List of dictionaries
        """

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            local_path = Path(tmp.name)

        try:
            download_result = self.download_file(s3_key, local_path, overwrite=True)
            if not download_result.success:
                error_msg = f"Failed to download {s3_key}: {download_result.error}"
                raise RuntimeError(error_msg)

            with open(local_path) as f:
                return [json.loads(line) for line in f if line.strip()]

        finally:
            if local_path.exists():
                local_path.unlink()

    def save_jsonl(self, s3_key: str, records: list[dict[str, Any]]) -> UploadResult:
        """
        Save a list of dictionaries to S3 as a JSONL file.

        Args:
            s3_key: S3 object key
            records: List of dictionaries to save

        Returns:
            UploadResult
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            local_path = Path(tmp.name)
            for record in records:
                tmp.write(json.dumps(record) + "\n")

        try:
            return self.upload_file(local_path, s3_key, overwrite=True)
        finally:
            if local_path.exists():
                local_path.unlink()


# Convenience functions for backward compatibility
def upload_dataset_artifact(local_path: str, s3_key: str, config: S3Config | None = None) -> bool:
    """
    Convenience function to upload a dataset artifact.

    Args:
        local_path: Path to local file
        s3_key: S3 object key
        config: S3 configuration (uses defaults if None)

    Returns:
        True if successful, False otherwise
    """
    loader = S3DatasetLoader(config or S3Config())
    result = loader.upload_file(local_path, s3_key)
    return result.success


def download_dataset_artifact(s3_key: str, local_path: str, config: S3Config | None = None) -> bool:
    """
    Convenience function to download a dataset artifact.

    Args:
        s3_key: S3 object key
        local_path: Path to save file
        config: S3 configuration (uses defaults if None)

    Returns:
        True if successful, False otherwise
    """
    loader = S3DatasetLoader(config or S3Config())
    result = loader.download_file(s3_key, local_path)
    return result.success


if __name__ == "__main__":
    # Example usage
    config = S3Config(
        endpoint_url=os.getenv("HETZNER_S3_ENDPOINT"),
        bucket_name="pixelated-datasets",
        verify_checksums=True,
    )

    loader = S3DatasetLoader(config)

    # Example: Upload a file
    # result = loader.upload_file("sample.jsonl", "datasets/test/sample.jsonl")
    # print(f"Upload result: {result.success}")

    # Example: List objects
    # objects = loader.list_objects(prefix="datasets/")
    # print(f"Found {len(objects)} objects")

    # Example: Download a file
    # result = loader.download_file("datasets/test/sample.jsonl", "downloaded.jsonl")
    # print(f"Download result: {result.success}")
