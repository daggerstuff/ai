#!/usr/bin/env python3
"""
S3 Dataset Loader

Loads datasets directly from S3 (OVH or AWS-compatible) object storage.
Supports JSONL and JSON formats with streaming for large datasets.
"""

import json
import os
import sys
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List
import logging
from io import BytesIO

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logging.warning("boto3 not available. Install with: uv pip install boto3")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3DatasetLoader:
    """Loads datasets from S3 object storage"""

    def __init__(self):
        """Initialize S3 client"""
        if not BOTO3_AVAILABLE:
            raise RuntimeError("boto3 not available. Install with: uv pip install boto3")

        # Initialize S3 client (OVH or AWS-compatible)
        endpoint_url = os.getenv("OVH_S3_ENDPOINT") or os.getenv("DATASET_S3_ENDPOINT_URL")
        access_key = (
            os.getenv("OVH_S3_ACCESS_KEY")
            or os.getenv("AWS_ACCESS_KEY_ID")
            or os.getenv("DATASET_S3_ACCESS_KEY_ID")
        )
        secret_key = (
            os.getenv("OVH_S3_SECRET_KEY")
            or os.getenv("AWS_SECRET_ACCESS_KEY")
            or os.getenv("DATASET_S3_SECRET_ACCESS_KEY")
        )
        region_name = (
            os.getenv("OVH_S3_REGION")
            or os.getenv("DATASET_S3_REGION")
            or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )

        session_kwargs: Dict[str, Any] = {}
        if access_key and secret_key:
            session_kwargs["aws_access_key_id"] = access_key
            session_kwargs["aws_secret_access_key"] = secret_key
        if region_name:
            session_kwargs["region_name"] = region_name
        if endpoint_url:
            session_kwargs["endpoint_url"] = endpoint_url

        self.s3_client = boto3.client("s3", **session_kwargs) if BOTO3_AVAILABLE else None

        if not self.s3_client:
            raise RuntimeError("Failed to initialize S3 client. Check credentials.")

    def parse_s3_path(self, s3_path: str) -> tuple[str, str]:
        """
        Parse S3 path into bucket and key

        Args:
            s3_path: S3 path in format 's3://bucket/key' or 'bucket/key'

        Returns:
            Tuple of (bucket, key)
        """
        if s3_path.startswith("s3://"):
            s3_path = s3_path[5:]

        parts = s3_path.split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            # Assume default bucket from env
            bucket = os.getenv("OVH_S3_BUCKET") or os.getenv("S3_BUCKET", "pixel-data")
            return bucket, s3_path

    def load_jsonl(self, s3_path: str, max_records: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        Load JSONL file from S3, streaming records

        Args:
            s3_path: S3 path to JSONL file
            max_records: Maximum number of records to load (None for all)

        Yields:
            Dictionary records from JSONL file
        """
        bucket, key = self.parse_s3_path(s3_path)

        logger.info(f"📥 Loading JSONL from s3://{bucket}/{key}")

        try:
            # Get object from S3
            response = self.s3_client.get_object(Bucket=bucket, Key=key)

            # Stream and parse JSONL
            record_count = 0
            buffer = ""

            for chunk in response['Body'].iter_chunks(chunk_size=8192):
                buffer += chunk.decode('utf-8')

                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()

                    if line:
                        try:
                            record = json.loads(line)
                            yield record
                            record_count += 1

                            if max_records and record_count >= max_records:
                                logger.info(f"  ✅ Loaded {record_count} records (limited to {max_records})")
                                return
                        except json.JSONDecodeError as e:
                            logger.warning(f"  ⚠️  Skipping invalid JSON line: {e}")

            # Process remaining buffer
            if buffer.strip():
                try:
                    record = json.loads(buffer.strip())
                    yield record
                    record_count += 1
                except json.JSONDecodeError:
                    pass

            logger.info(f"  ✅ Loaded {record_count} records")

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"File not found in S3: s3://{bucket}/{key}")
            else:
                raise RuntimeError(f"S3 error loading {s3_path}: {e}")

    def load_json(self, s3_path: str) -> Dict[str, Any]:
        """
        Load JSON file from S3

        Args:
            s3_path: S3 path to JSON file

        Returns:
            Parsed JSON data
        """
        bucket, key = self.parse_s3_path(s3_path)

        logger.info(f"📥 Loading JSON from s3://{bucket}/{key}")

        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            data = json.loads(content)
            logger.info(f"  ✅ Loaded JSON successfully")
            return data

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"File not found in S3: s3://{bucket}/{key}")
            else:
                raise RuntimeError(f"S3 error loading {s3_path}: {e}")

    def list_datasets(self, prefix: str = "datasets/", max_keys: int = 1000) -> List[str]:
        """
        List datasets in S3 bucket

        Args:
            prefix: S3 key prefix to search
            max_keys: Maximum number of keys to return

        Returns:
            List of S3 paths
        """
        bucket = os.getenv("OVH_S3_BUCKET") or os.getenv("S3_BUCKET", "pixel-data")

        logger.info(f"📋 Listing datasets in s3://{bucket}/{prefix}")

        try:
            paths = []
            paginator = self.s3_client.get_paginator('list_objects_v2')

            for page in paginator.paginate(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        paths.append(f"s3://{bucket}/{obj['Key']}")

            logger.info(f"  ✅ Found {len(paths)} datasets")
            return paths

        except ClientError as e:
            raise RuntimeError(f"S3 error listing datasets: {e}")

    def dataset_exists(self, s3_path: str) -> bool:
        """
        Check if dataset exists in S3

        Args:
            s3_path: S3 path to check

        Returns:
            True if exists, False otherwise
        """
        bucket, key = self.parse_s3_path(s3_path)

        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False


def load_dataset_from_s3(s3_path: str, format: str = "auto", max_records: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """
    Convenience function to load dataset from S3

    Args:
        s3_path: S3 path to dataset
        format: File format ('jsonl', 'json', or 'auto' to detect from extension)
        max_records: Maximum records to load (for JSONL only)

    Returns:
        Iterator of records (for JSONL) or dict (for JSON)
    """
    loader = S3DatasetLoader()

    if format == "auto":
        format = "jsonl" if s3_path.endswith(".jsonl") else "json"

    if format == "jsonl":
        return loader.load_jsonl(s3_path, max_records=max_records)
    elif format == "json":
        data = loader.load_json(s3_path)
        # If it's a list, yield items
        if isinstance(data, list):
            for item in data:
                yield item
        else:
            yield data
    else:
        raise ValueError(f"Unsupported format: {format}")


if __name__ == "__main__":
    # Test the loader
    import sys

    if len(sys.argv) < 2:
        print("Usage: python s3_dataset_loader.py <s3_path> [max_records]")
        print("Example: python s3_dataset_loader.py s3://pixel-data/datasets/huggingface/ShreyaR_DepressionDetection.jsonl 10")
        sys.exit(1)

    s3_path = sys.argv[1]
    max_records = int(sys.argv[2]) if len(sys.argv) > 2 else None

    try:
        loader = S3DatasetLoader()
        print(f"Loading from: {s3_path}")
        print("=" * 60)

        count = 0
        for record in loader.load_jsonl(s3_path, max_records=max_records):
            print(f"Record {count + 1}:")
            print(json.dumps(record, indent=2)[:200] + "...")
            print()
            count += 1

        print(f"✅ Successfully loaded {count} records")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

