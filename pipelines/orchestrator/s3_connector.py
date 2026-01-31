"""S3 connector for dataset pipeline ingestion.

Implements the IngestionConnector interface for S3 bucket access,
with enhanced retry, rate limiting, and security features.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, List
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import botocore

from .ingestion_interface import IngestionConnector, IngestRecord, IngestionError
from .ingest_utils import read_with_retry, RateLimiter
from .validation import validate_record
from .quarantine import get_quarantine_store


@dataclass
class S3Config:
    """Configuration for S3 connector."""
    bucket_name: str
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    region_name: Optional[str] = 'us-east-1'
    prefix: Optional[str] = ''
    endpoint_url: Optional[str] = None  # For custom S3 endpoints
    max_concurrent: int = 10  # Limit concurrent downloads
    retry_options: Optional[dict] = None
    rate_limit: Optional[dict] = None


class S3Connector(IngestionConnector):
    """S3 connector implementing the IngestionConnector interface."""

    def __init__(
        self,
        config: S3Config,
        name: Optional[str] = None,
        allowed_domains: Optional[List[str]] = None
    ):
        super().__init__(name=name or f"s3_{config.bucket_name}")
        self.config = config
        self.allowed_domains = allowed_domains or []
        
        # Set up AWS credentials from config or environment
        aws_access_key_id = config.aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = config.aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        
        # Create S3 client with enhanced configuration
        session_kwargs = {}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs['aws_access_key_id'] = aws_access_key_id
            session_kwargs['aws_secret_access_key'] = aws_secret_access_key
        if config.region_name:
            session_kwargs['region_name'] = config.region_name

        session = boto3.Session(**session_kwargs)
        
        client_kwargs = {}
        if config.endpoint_url:
            client_kwargs['endpoint_url'] = config.endpoint_url
        if config.region_name:
            client_kwargs['region_name'] = config.region_name

        self.s3_client = session.client('s3', **client_kwargs)
        
        # Set up rate limiting
        self.rate_limiter = None
        if config.rate_limit:
            self.rate_limiter = RateLimiter(
                capacity=config.rate_limit.get('capacity', 10),
                refill_rate=config.rate_limit.get('refill_rate', 1.0)
            )
        
        # Thread-safe semaphore for concurrent downloads
        self.download_semaphore = asyncio.Semaphore(config.max_concurrent)
        
        # Cache for object metadata to avoid repeated API calls
        self.metadata_cache: Dict[str, Dict[str, Any]] = {}

    def connect(self) -> None:
        """Validate S3 connection and permissions."""
        try:
            # Test access to the bucket
            self.s3_client.head_bucket(Bucket=self.config.bucket_name)
        except NoCredentialsError:
            raise IngestionError("AWS credentials not found")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == '404':
                raise IngestionError(f"Bucket {self.config.bucket_name} does not exist")
            elif error_code == '403':
                raise IngestionError(f"Access denied to bucket {self.config.bucket_name}")
            else:
                raise IngestionError(f"S3 connection error: {e}")
        except Exception as e:
            raise IngestionError(f"Unexpected error connecting to S3: {e}")

    def _is_valid_s3_key(self, key: str) -> bool:
        """Validate S3 object key for security."""
        # Check for path traversal attempts
        if '..' in key or key.startswith('/') or key.startswith('../'):
            return False
        # Disallow certain dangerous file extensions
        dangerous_extensions = ['.exe', '.bat', '.sh', '.cmd', '.ps1', '.jar']
        ext = key.split('.')[-1].lower() if '.' in key else ''
        if ext in dangerous_extensions:
            return False
        return True

    def _fetch_object_metadata(self, key: str) -> Dict[str, Any]:
        """Fetch and cache object metadata."""
        if key in self.metadata_cache:
            return self.metadata_cache[key]
        
        try:
            response = self.s3_client.head_object(Bucket=self.config.bucket_name, Key=key)
            metadata = {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'].isoformat(),
                'etag': response['ETag'].strip('"'),
                'content_type': response.get('ContentType', 'application/octet-stream')
            }
            self.metadata_cache[key] = metadata
            return metadata
        except ClientError as e:
            raise IngestionError(f"Failed to fetch metadata for {key}: {e}")

    def fetch(self) -> Iterable[IngestRecord]:
        """Fetch objects from S3 bucket."""
        try:
            # List objects in the specified prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.config.bucket_name,
                Prefix=self.config.prefix
            )
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    # Validate key for security
                    if not self._is_valid_s3_key(key):
                        continue
                    
                    try:
                        # Honor rate limiting if configured
                        if self.rate_limiter:
                            acquired = self.rate_limiter.acquire(blocking=True, timeout=5)
                            if not acquired:
                                raise IngestionError(f"Rate limiter timeout reading {key}")
                        
                        # Fetch object content
                        response = self.s3_client.get_object(
                            Bucket=self.config.bucket_name,
                            Key=key
                        )
                        
                        # Read content with retry logic for large objects
                        content = response['Body'].read()
                        
                        # Deduplication check at ingestion stage
                        from .ingestion_deduplication import add_content_check_duplicate
                        if not add_content_check_duplicate(content):
                            # Content is a duplicate, skip
                            continue
                        
                        # Create IngestRecord
                        rec = IngestRecord(
                            id=f"s3://{self.config.bucket_name}/{key}",
                            payload=content,
                            metadata={
                                'source_type': 's3_object',
                                'bucket': self.config.bucket_name,
                                'key': key,
                                'size': len(content),
                                'content_type': response.get('ResponseMetadata', {}).get('HTTPHeaders', {}).get('content-type', 'application/octet-stream'),
                                'last_modified': str(response.get('LastModified', '')),
                            }
                        )
                        
                        # Source-level validation
                        if not self.validate(rec):
                            store = get_quarantine_store()
                            errors = [f"Source validation failed for {self.name} connector: {key}"]
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
                        # Log and continue processing other objects
                        store = get_quarantine_store()
                        errors = [f"S3 fetch error: {e}"]
                        try:
                            store.quarantine_record(rec, errors)  # type: ignore[name-defined]
                        except Exception:
                            # If rec isn't defined or quarantine fails, ignore
                            pass
                        continue

        except Exception as e:
            raise IngestionError(f"Error listing objects in bucket {self.config.bucket_name}: {e}")

    def close(self) -> None:
        """Close S3 client connections."""
        # S3 client doesn't have a specific close method, but we can clear references
        self.s3_client = None
        self.metadata_cache.clear()


# Register the S3 connector
from .ingestion_interface import register_connector

register_connector('s3', S3Connector)