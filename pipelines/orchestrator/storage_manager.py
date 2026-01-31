#!/usr/bin/env python3
"""
Storage Manager for Dataset Pipeline
Handles file uploads/downloads to configured storage backend (S3, GCS, or local)
"""

import hashlib
import json
from pathlib import Path
from typing import Optional, BinaryIO, Dict, Any
import boto3
from botocore.exceptions import ClientError

# Optional GCS imports
try:
    from google.cloud import storage as gcs_storage
    from google.cloud.exceptions import NotFound, GoogleCloudError
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

from .storage_config import StorageConfig, StorageBackend, get_storage_config


class StorageManager:
    """Manages file storage operations for dataset pipeline"""

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or get_storage_config()
        self._s3_client = None
        self._gcs_client = None
        self._gcs_bucket = None

        # Validate config
        is_valid, error = self.config.validate()
        if not is_valid:
            raise ValueError(f"Invalid storage configuration: {error}")

    def _get_s3_client(self):
        """Get or create S3 client"""
        if self._s3_client is None and self.config.backend == StorageBackend.S3:
            session_kwargs = {}
            if self.config.s3_access_key_id:
                session_kwargs['aws_access_key_id'] = self.config.s3_access_key_id
            if self.config.s3_secret_access_key:
                session_kwargs['aws_secret_access_key'] = self.config.s3_secret_access_key
            if self.config.s3_region:
                session_kwargs['region_name'] = self.config.s3_region

            session = boto3.Session(**session_kwargs)
            client_kwargs = {}
            if self.config.s3_endpoint_url:
                client_kwargs['endpoint_url'] = self.config.s3_endpoint_url
            if self.config.s3_region:
                client_kwargs['region_name'] = self.config.s3_region

            self._s3_client = session.client('s3', **client_kwargs)
        return self._s3_client

    def _get_gcs_client(self):
        """Get or create GCS client and bucket"""
        if not GCS_AVAILABLE:
            raise ImportError("google-cloud-storage is not installed. Install it with: pip install google-cloud-storage")

        if self._gcs_client is None and self.config.backend == StorageBackend.GCS:
            from google.oauth2 import service_account
            import os

            credentials = None
            if self.config.gcs_credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    self.config.gcs_credentials_path
                )
            elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                credentials = service_account.Credentials.from_service_account_file(
                    os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                )

            client_kwargs = {}
            if credentials:
                client_kwargs['credentials'] = credentials
            if self.config.gcs_project_id:
                client_kwargs['project'] = self.config.gcs_project_id

            self._gcs_client = gcs_storage.Client(**client_kwargs)
            self._gcs_bucket = self._gcs_client.bucket(self.config.gcs_bucket)

        return self._gcs_client, self._gcs_bucket

    def upload_file(self, local_path: Path, storage_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Upload a file to storage and return the storage URL"""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        if self.config.backend == StorageBackend.LOCAL:
            # Copy to local storage path
            storage_full_path = Path(storage_path)
            storage_full_path.parent.mkdir(parents=True, exist_ok=True)

            import shutil
            shutil.copy2(local_path, storage_full_path)

            # Store metadata if provided
            if metadata:
                metadata_path = storage_full_path.with_suffix(storage_full_path.suffix + '.metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

            return str(storage_full_path.absolute())

        elif self.config.backend == StorageBackend.S3:
            s3_client = self._get_s3_client()
            if not s3_client:
                raise RuntimeError("S3 client not initialized")

            extra_args = {}
            if metadata:
                # Convert metadata to S3 metadata (string values only)
                extra_args['Metadata'] = {k: str(v) for k, v in metadata.items()}

            try:
                s3_client.upload_file(
                    str(local_path),
                    self.config.s3_bucket,
                    storage_path,
                    ExtraArgs=extra_args
                )
                return f"s3://{self.config.s3_bucket}/{storage_path}"
            except ClientError as e:
                raise RuntimeError(f"Failed to upload to S3: {e}")

        elif self.config.backend == StorageBackend.GCS:
            gcs_client, gcs_bucket = self._get_gcs_client()
            if not gcs_bucket:
                raise RuntimeError("GCS bucket not initialized")

            blob = gcs_bucket.blob(storage_path)

            # Set metadata if provided
            if metadata:
                blob.metadata = {k: str(v) for k, v in metadata.items()}

            try:
                blob.upload_from_filename(str(local_path))
                return f"gs://{self.config.gcs_bucket}/{storage_path}"
            except GoogleCloudError as e:
                raise RuntimeError(f"Failed to upload to GCS: {e}")

        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

    def download_file(self, storage_path: str, local_path: Path) -> Path:
        """Download a file from storage to local path"""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.backend == StorageBackend.LOCAL:
            # Copy from local storage
            source_path = Path(storage_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Storage file not found: {storage_path}")

            import shutil
            shutil.copy2(source_path, local_path)
            return local_path

        elif self.config.backend == StorageBackend.S3:
            s3_client = self._get_s3_client()
            if not s3_client:
                raise RuntimeError("S3 client not initialized")

            try:
                s3_client.download_file(
                    self.config.s3_bucket,
                    storage_path,
                    str(local_path)
                )
                return local_path
            except ClientError as e:
                raise RuntimeError(f"Failed to download from S3: {e}")

        elif self.config.backend == StorageBackend.GCS:
            gcs_client, gcs_bucket = self._get_gcs_client()
            if not gcs_bucket:
                raise RuntimeError("GCS bucket not initialized")

            blob = gcs_bucket.blob(storage_path)
            try:
                blob.download_to_filename(str(local_path))
                return local_path
            except NotFound:
                raise FileNotFoundError(f"Storage file not found: {storage_path}")
            except GoogleCloudError as e:
                raise RuntimeError(f"Failed to download from GCS: {e}")

        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

    def file_exists(self, storage_path: str) -> bool:
        """Check if a file exists in storage"""
        if self.config.backend == StorageBackend.LOCAL:
            return Path(storage_path).exists()

        elif self.config.backend == StorageBackend.S3:
            s3_client = self._get_s3_client()
            if not s3_client:
                return False

            try:
                s3_client.head_object(Bucket=self.config.s3_bucket, Key=storage_path)
                return True
            except ClientError:
                return False

        elif self.config.backend == StorageBackend.GCS:
            if not GCS_AVAILABLE:
                return False
            gcs_client, gcs_bucket = self._get_gcs_client()
            if not gcs_bucket:
                return False

            blob = gcs_bucket.blob(storage_path)
            return blob.exists()

        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def upload_with_checksum(self, local_path: Path, storage_path: str,
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Upload file and return checksum and storage URL"""
        checksum = self.calculate_checksum(local_path)
        file_size = local_path.stat().st_size

        upload_metadata = metadata or {}
        upload_metadata['sha256'] = checksum
        upload_metadata['size_bytes'] = file_size

        storage_url = self.upload_file(local_path, storage_path, upload_metadata)

        return {
            'storage_url': storage_url,
            'storage_path': storage_path,
            'sha256': checksum,
            'size_bytes': file_size,
            'local_path': str(local_path)
        }

