#!/usr/bin/env python3
"""Helper for creating S3 clients with custom endpoint support (Hetzner Object Storage, MinIO, etc)."""

import os

import boto3
from botocore.config import Config


def get_s3_client():
    """
    Create an S3 client with support for custom endpoints.

    Supports:
    - AWS S3 (default)
    - Hetzner Object Storage (via AWS_S3_ENDPOINT or HETZNER_S3_ENDPOINT)
    - MinIO (via MINIO_ENDPOINT)

    Environment variables:
    - AWS_S3_ENDPOINT: Custom S3 endpoint URL
    - HETZNER_S3_ENDPOINT: Hetzner Object Storage endpoint
    - MINIO_ENDPOINT: MinIO endpoint
    - AWS_ACCESS_KEY_ID: Access key
    - AWS_SECRET_ACCESS_KEY: Secret key
    - AWS_REGION: Region name

    Returns:
        Configured boto3 S3 client
    """
    endpoint_url = (
        os.environ.get("HETZNER_S3_ENDPOINT")
        or os.environ.get("AWS_S3_ENDPOINT")
        or os.environ.get("MINIO_ENDPOINT")
        or "https://hel1.your-objectstorage.com"
    )

    region = os.environ.get("HETZNER_S3_REGION") or os.environ.get("AWS_REGION", "sfo3")

    access_key = (
        os.environ.get("HETZNER_S3_ACCESS_KEY")
        or os.environ.get("HETZNER_ACCESS_KEY")
        or os.environ.get("AWS_ACCESS_KEY_ID")
    )
    secret_key = (
        os.environ.get("HETZNER_S3_SECRET_KEY")
        or os.environ.get("HETZNER_SECRET_KEY")
        or os.environ.get("AWS_SECRET_ACCESS_KEY")
    )

    config = Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"})

    client_kwargs: dict[str, object] = {
        "region_name": region,
        "config": config,
    }

    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    if access_key:
        client_kwargs["aws_access_key_id"] = access_key
    if secret_key:
        client_kwargs["aws_secret_access_key"] = secret_key

    return boto3.client("s3", **client_kwargs)


def get_s3_bucket_name() -> str:
    """
    Get the S3 bucket name from environment or default.

    Checks:
    - HETZNER_S3_BUCKET
    - AWS_S3_BUCKET
    - Or defaults to 'pixel-data'

    Returns:
        Bucket name
    """
    return os.environ.get("HETZNER_S3_BUCKET") or os.environ.get("AWS_S3_BUCKET") or "pixel-data"
