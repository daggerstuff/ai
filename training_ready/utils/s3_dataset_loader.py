#!/usr/bin/env python3
"""
S3 Dataset Loader - Streaming JSON/JSONL loader for S3 training data
S3 is the training mecca - all training data should be loaded from S3
"""

import contextlib
import json
import logging
import os
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import boto3
    from botocore.exceptions import ClientError as _BotocoreClientError
except ImportError:
    # Keep runtime behavior (error on use) while making type checkers happy.
    boto3 = None  # type: ignore[assignment]
    _BotocoreClientError = None  # type: ignore[assignment]

if TYPE_CHECKING:
    # Minimal shape we rely on in this module.
    class ClientError(Exception):
        response: dict[str, Any]
else:
    ClientError = _BotocoreClientError if _BotocoreClientError is not None else Exception  # type: ignore[assignment]

BOTO3_AVAILABLE = boto3 is not None

# Load .env file if available
with contextlib.suppress(ImportError):
    from dotenv import load_dotenv

    # Try loading from ai/ directory first (where .env actually is), then project root
    # Module is at: ai/training_ready/utils/s3_dataset_loader.py
    # So parents[0] = ai/training_ready/utils/, parents[1] = ai/training_ready/,
    #    parents[2] = ai/, parents[3] = project root
    module_path = Path(__file__).resolve()
    env_paths = [
        module_path.parents[2] / ".env",  # ai/.env (actual location)
        module_path.parents[3] / ".env",  # project root/.env (fallback)
    ]
    for env_path in env_paths:
        if env_path.exists() and env_path.is_file():  # Check it's a file, not a pipe
            load_dotenv(env_path, override=False)  # Don't override existing env vars
            break

logger = logging.getLogger(__name__)


class S3DatasetLoader:
    """
    Load datasets from S3 with streaming support for large files.
    S3 is the canonical training data location.
    """

    def __init__(
        self,
        bucket: str = "pixelated-training-data",
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        region_name: str = "us-east-va",
    ):
        """
        Initialize S3 client for dataset loading.

        Args:
            bucket: S3 bucket name (default: pixelated-training-data)
            endpoint_url: S3 endpoint URL (default: OVH S3 endpoint)
            aws_access_key_id: AWS access key (from env if not provided)
            aws_secret_access_key: AWS secret key (from env if not provided)
            region_name: AWS region (default: us-east-va for OVH)
        """
        if boto3 is None:
            raise ImportError(
                "boto3 is required for S3 dataset loading. Install with: uv pip install boto3"
            )

        # Allow env to override only when using the default bucket argument
        if bucket == "pixelated-training-data":
            self.bucket = os.getenv("OVH_S3_BUCKET", bucket)
        else:
            self.bucket = bucket
        self.endpoint_url = endpoint_url or os.getenv(
            "OVH_S3_ENDPOINT", "https://s3.us-east-va.cloud.ovh.us"
        )

        # Get credentials from params or environment
        access_key = (
            aws_access_key_id
            or os.getenv("OVH_S3_ACCESS_KEY")
            or os.getenv("OVH_ACCESS_KEY")
            or os.getenv("AWS_ACCESS_KEY_ID")
        )
        secret_key = (
            aws_secret_access_key
            or os.getenv("OVH_S3_SECRET_KEY")
            or os.getenv("OVH_SECRET_KEY")
            or os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        if not access_key or not secret_key:
            raise ValueError(
                "S3 credentials not found. Set OVH_S3_ACCESS_KEY/OVH_S3_SECRET_KEY "
                "(or OVH_ACCESS_KEY/OVH_SECRET_KEY, "
                "or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)."
            )

        # Initialize S3 client (OVH S3 compatible)
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name or os.getenv("OVH_S3_REGION", "us-east-va"),
        )

        logger.info(f"S3DatasetLoader initialized for bucket: {bucket}")

    def _parse_s3_path(self, s3_path: str) -> tuple[str, str]:
        """
        Parse S3 path into bucket and key.

        Args:
            s3_path: S3 path (s3://bucket/key or just key)

        Returns:
            Tuple of (bucket, key)
        """
        s3_path = s3_path.removeprefix("s3://")

        if "/" in s3_path:
            parts = s3_path.split("/", 1)
            return parts[0], parts[1]

        # Just key, use default bucket
        return self.bucket, s3_path

    def object_exists(self, s3_path: str) -> bool:
        """Check if S3 object exists"""
        try:
            bucket, key = self._parse_s3_path(s3_path)
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def load_json(
        self,
        s3_path: str,
        cache_local: Path | None = None,
    ) -> dict[str, Any]:
        """
        Load JSON dataset from S3.

        Args:
            s3_path: S3 path (s3://bucket/key or just key)
            cache_local: Optional local cache path

        Returns:
            Parsed JSON data
        """
        bucket, key = self._parse_s3_path(s3_path)

        # Check local cache first
        if cache_local and cache_local.exists():
            logger.info(f"Loading from local cache: {cache_local}")
            with open(cache_local) as f:
                return json.load(f)

        # Load from S3
        logger.info(f"Loading from S3: s3://{bucket}/{key}")
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            data = json.loads(response["Body"].read())

            # Cache locally if requested
            if cache_local:
                cache_local.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_local, "w") as f:
                    json.dump(data, f)
                logger.info(f"Cached to: {cache_local}")

            return data
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"Dataset not found in S3: s3://{bucket}/{key}") from e
            raise

    def load_bytes(self, s3_path: str) -> bytes:
        """
        Load raw bytes from S3.

        Args:
            s3_path: S3 path (s3://bucket/key or just key)

        Returns:
            Raw bytes of the object body
        """
        bucket, key = self._parse_s3_path(s3_path)
        logger.info(f"Loading bytes from S3: s3://{bucket}/{key}")

        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"Dataset not found in S3: s3://{bucket}/{key}") from e
            raise

    def load_text(
        self,
        s3_path: str,
        *,
        encoding: str = "utf-8",
        errors: str = "replace",
    ) -> str:
        """
        Load a text object from S3.

        This is primarily for transcript corpora (e.g. .txt) that need to be
        converted into ChatML examples.
        """
        data = self.load_bytes(s3_path)
        return data.decode(encoding, errors=errors)

    def _parse_jsonl_line(self, line: bytes) -> dict[str, Any] | None:
        """
        Parse a single JSONL line with robust error handling.

        Args:
            line: Raw bytes of a JSONL line

        Returns:
            Parsed JSON object or None if parsing failed
        """
        try:
            return json.loads(line.decode("utf-8"))
        except UnicodeDecodeError:
            try:
                return json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSONL line: {e}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSONL line: {e}")
        return None

    def _stream_with_iter_lines(self, body) -> Iterator[dict[str, Any]]:
        """
        Stream JSONL using iter_lines() method.

        Args:
            body: S3 response body with iter_lines capability

        Yields:
            Parsed JSON objects
        """
        for raw_line in body.iter_lines():
            if not raw_line:
                continue
            parsed = self._parse_jsonl_line(raw_line)
            if parsed is not None:
                yield parsed

    def _stream_with_manual_buffering(self, body) -> Iterator[dict[str, Any]]:
        """
        Stream JSONL using manual buffering as fallback.

        Args:
            body: S3 response body

        Yields:
            Parsed JSON objects
        """
        buffer = BytesIO()
        for chunk in body.iter_chunks(chunk_size=8192):
            buffer.write(chunk)
            while True:
                buffer.seek(0)
                line = buffer.readline()
                if not line:
                    buffer = BytesIO()
                    break
                if not line.endswith(b"\n"):
                    # Keep incomplete tail in buffer
                    rest = buffer.read()
                    buffer = BytesIO(line + rest)
                    break

                parsed = self._parse_jsonl_line(line)
                if parsed is not None:
                    yield parsed

                rest = buffer.read()
                buffer = BytesIO(rest)

    def stream_jsonl(self, s3_path: str) -> Iterator[dict[str, Any]]:
        """
        Stream JSONL dataset from S3 (memory-efficient for large files).

        Args:
            s3_path: S3 path (s3://bucket/key or just key)

        Yields:
            Parsed JSON objects (one per line)
        """
        bucket, key = self._parse_s3_path(s3_path)

        logger.info(f"Streaming JSONL from S3: s3://{bucket}/{key}")
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            body = response["Body"]

            # Prefer iter_lines() which handles chunk boundaries robustly
            iter_lines = getattr(body, "iter_lines", None)
            if callable(iter_lines):
                yield from self._stream_with_iter_lines(body)
            else:
                # Fallback to manual buffering
                yield from self._stream_with_manual_buffering(body)

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"Dataset not found in S3: s3://{bucket}/{key}") from e
            raise

    def list_datasets(self, prefix: str = "gdrive/processed/") -> list[str]:
        """
        List available datasets in S3.

        Args:
            prefix: S3 prefix to search (default: gdrive/processed/)

        Returns:
            List of S3 paths
        """
        logger.info(f"Listing datasets with prefix: {prefix}")
        datasets: list[str] = []

        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)

            for page in pages:
                if "Contents" in page:
                    datasets.extend(
                        f"s3://{self.bucket}/{obj['Key']}"
                        for obj in page["Contents"]
                        if obj["Key"].endswith((".json", ".jsonl"))
                    )

        except ClientError as e:
            logger.error(f"Failed to list S3 objects: {e}")
            raise

        return datasets


def get_s3_dataset_path(
    dataset_name: str,
    category: str | None = None,
    bucket: str = "pixelated-training-data",
    prefer_processed: bool = True,
) -> str:
    """
    Get S3 path for dataset - S3 is canonical training data location.

    Args:
        dataset_name: Name of the dataset file
        category: Optional category (cot_reasoning, professional_therapeutic, etc.)
        bucket: S3 bucket name
        prefer_processed: Prefer processed/canonical structure over raw

    Returns:
        S3 path (s3://bucket/path)
    """
    loader = S3DatasetLoader(bucket=bucket)

    # Try canonical processed structure first
    if category and prefer_processed:
        path = f"s3://{bucket}/gdrive/processed/{category}/{dataset_name}"
        if loader.object_exists(path):
            return path

    # Fallback to raw structure
    if prefer_processed:
        path = f"s3://{bucket}/gdrive/raw/{dataset_name}"
        if loader.object_exists(path):
            return path

    # Fallback to acquired
    path = f"s3://{bucket}/acquired/{dataset_name}"
    if loader.object_exists(path):
        return path

    # If category provided, construct path even if doesn't exist yet
    if category:
        return f"s3://{bucket}/gdrive/processed/{category}/{dataset_name}"

    return f"s3://{bucket}/gdrive/raw/{dataset_name}"


def load_dataset_from_s3(
    dataset_name: str,
    category: str | None = None,
    cache_local: Path | None = None,
    bucket: str = "pixelated-training-data",
) -> dict[str, Any]:
    """
    Load dataset from S3 with automatic path resolution.

    Args:
        dataset_name: Name of the dataset file
        category: Optional category for canonical structure
        cache_local: Optional local cache path
        bucket: S3 bucket name

    Returns:
        Dataset data
    """
    loader = S3DatasetLoader(bucket=bucket)
    s3_path = get_s3_dataset_path(dataset_name, category, bucket)

    if dataset_name.endswith(".jsonl"):
        # For JSONL, convert to list
        return {"conversations": list(loader.stream_jsonl(s3_path))}
    return loader.load_json(s3_path, cache_local)
