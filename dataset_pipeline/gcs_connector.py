"""GCS connector for dataset pipeline ingestion.

Implements the IngestionConnector interface for Google Cloud Storage access,
with enhanced retry, rate limiting, and security features.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, List
from google.cloud import storage
from google.cloud.exceptions import NotFound, Forbidden, GoogleCloudError
from google.oauth2 import service_account
import tempfile
import requests

from .ingestion_interface import IngestionConnector, IngestRecord, IngestionError
from .ingest_utils import read_with_retry, RateLimiter
from .validation import validate_record
from .quarantine import get_quarantine_store


@dataclass
class GCSConfig:
    """Configuration for GCS connector."""
    bucket_name: str
    project_id: Optional[str] = None
    credentials_path: Optional[str] = None  # Path to service account key file
    credentials_info: Optional[dict] = None  # Credentials as dict
    prefix: Optional[str] = ''
    max_concurrent: int = 10  # Limit concurrent downloads
    retry_options: Optional[dict] = None
    rate_limit: Optional[dict] = None


class GCSConnector(IngestionConnector):
    """GCS connector implementing the IngestionConnector interface."""

    def __init__(
        self,
        config: GCSConfig,
        name: Optional[str] = None,
        allowed_domains: Optional[List[str]] = None
    ):
        super().__init__(name=name or f"gcs_{config.bucket_name}")
        self.config = config
        self.allowed_domains = allowed_domains or []
        
        # Set up credentials
        credentials = None
        credentials_path = config.credentials_path or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
        elif config.credentials_info:
            credentials = service_account.Credentials.from_service_account_info(config.credentials_info)
        
        # Initialize GCS client
        client_kwargs = {}
        if credentials:
            client_kwargs['credentials'] = credentials
        if config.project_id:
            client_kwargs['project'] = config.project_id
            
        self.gcs_client = storage.Client(**client_kwargs)
        self.bucket = self.gcs_client.bucket(config.bucket_name)
        
        # Set up rate limiting
        self.rate_limiter = None
        if config.rate_limit:
            self.rate_limiter = RateLimiter(
                capacity=config.rate_limit.get('capacity', 10),
                refill_rate=config.rate_limit.get('refill_rate', 1.0)
            )
        
        # Thread-safe semaphore for concurrent downloads
        self.download_semaphore = asyncio.Semaphore(config.max_concurrent)

    def connect(self) -> None:
        """Validate GCS connection and permissions."""
        try:
            # Test access to the bucket
            if not self.bucket.exists():
                raise IngestionError(f"Bucket {self.config.bucket_name} does not exist")
        except Forbidden:
            raise IngestionError(f"Access denied to bucket {self.config.bucket_name}")
        except GoogleCloudError as e:
            raise IngestionError(f"GCS connection error: {e}")
        except Exception as e:
            raise IngestionError(f"Unexpected error connecting to GCS: {e}")

    def _is_valid_gcs_blob_name(self, blob_name: str) -> bool:
        """Validate GCS blob name for security."""
        # Check for path traversal attempts
        if '..' in blob_name or blob_name.startswith('/') or blob_name.startswith('../'):
            return False
        # Disallow certain dangerous file extensions
        dangerous_extensions = ['.exe', '.bat', '.sh', '.cmd', '.ps1', '.jar']
        ext = blob_name.split('.')[-1].lower() if '.' in blob_name else ''
        if ext in dangerous_extensions:
            return False
        return True

    def _fetch_blob_metadata(self, blob_name: str) -> Dict[str, Any]:
        """Fetch blob metadata."""
        try:
            blob = self.bucket.blob(blob_name)
            if not blob.exists():
                raise IngestionError(f"Blob {blob_name} does not exist")
                
            # Load blob metadata
            blob.reload()
            return {
                'size': blob.size,
                'updated': blob.updated.isoformat() if blob.updated else None,
                'content_type': blob.content_type,
                'etag': blob.etag,
                'md5_hash': blob.md5_hash,
            }
        except Exception as e:
            raise IngestionError(f"Failed to fetch metadata for {blob_name}: {e}")

    def fetch(self) -> Iterable[IngestRecord]:
        """Fetch blobs from GCS bucket."""
        try:
            # List blobs in the specified prefix
            blobs = self.bucket.list_blobs(prefix=self.config.prefix)
            
            for blob in blobs:
                blob_name = blob.name
                
                # Validate blob name for security
                if not self._is_valid_gcs_blob_name(blob_name):
                    continue
                
                try:
                    # Honor rate limiting if configured
                    if self.rate_limiter:
                        acquired = self.rate_limiter.acquire(blocking=True, timeout=5)
                        if not acquired:
                            raise IngestionError(f"Rate limiter timeout reading {blob_name}")
                    
                    # Check if the blob exists before downloading
                    if not blob.exists():
                        continue
                    
                    # Download blob content
                    content = blob.download_as_bytes()
                    
                    # Deduplication check at ingestion stage
                    from .ingestion_deduplication import add_content_check_duplicate
                    if not add_content_check_duplicate(content):
                        # Content is a duplicate, skip
                        continue
                    
                    # Create IngestRecord
                    rec = IngestRecord(
                        id=f"gcs://{self.config.bucket_name}/{blob_name}",
                        payload=content,
                        metadata={
                            'source_type': 'gcs_object',
                            'bucket': self.config.bucket_name,
                            'blob_name': blob_name,
                            'size': len(content),
                            'content_type': blob.content_type or 'application/octet-stream',
                            'last_modified': str(blob.updated) if blob.updated else None,
                        }
                    )
                    
                    # Source-level validation
                    if not self.validate(rec):
                        store = get_quarantine_store()
                        errors = [f"Source validation failed for {self.name} connector: {blob_name}"]
                        store.quarantine_record(rec, errors)
                        continue
                    
                    # Schema validation
                    try:
                        validated = validate_record(rec)
                        yield validated
                    except Exception as ve:
                        # Quarantine validation failures
                        store = get_quarantine_store()
                        errors = [str(ve)]
                        store.quarantine_record(rec, errors)
                        continue

                except Exception as e:
                    # Log and continue processing other blobs
                    store = get_quarantine_store()
                    errors = [f"GCS fetch error: {e}"]
                    try:
                        store.quarantine_record(rec, errors)  # type: ignore[name-defined]
                    except Exception:
                        # If rec isn't defined or quarantine fails, ignore
                        pass
                    continue

        except Exception as e:
            raise IngestionError(f"Error listing blobs in bucket {self.config.bucket_name}: {e}")

    def close(self) -> None:
        """Close GCS client connections."""
        # GCS client doesn't have a specific close method, but we can clear references
        self.gcs_client = None
        self.bucket = None


# Register the GCS connector
from .ingestion_interface import register_connector

register_connector('gcs', GCSConnector)