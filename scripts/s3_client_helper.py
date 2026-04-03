#!/usr/bin/env python3
"""Helper for creating S3 clients with custom endpoint support (DigitalOcean Spaces, MinIO, etc)."""

import os
import boto3
from botocore.config import Config


def get_s3_client() -> boto3.client:
    """
    Create an S3 client with support for custom endpoints.

    Supports:
    - AWS S3 (default)
    - DigitalOcean Spaces (via AWS_S3_ENDPOINT or DO_S3_ENDPOINT)
    - MinIO (via MINIO_ENDPOINT)

    Environment variables:
    - AWS_S3_ENDPOINT: Custom S3 endpoint URL
    - DO_S3_ENDPOINT: DigitalOcean Spaces endpoint
    - MINIO_ENDPOINT: MinIO endpoint
    - AWS_ACCESS_KEY_ID: Access key
    - AWS_SECRET_ACCESS_KEY: Secret key
    - AWS_REGION: Region name

    Returns:
        Configured boto3 S3 client
    """
    endpoint_url = (
        os.environ.get("AWS_S3_ENDPOINT")
        or os.environ.get("DO_S3_ENDPOINT")
        or os.environ.get("MINIO_ENDPOINT")
        or "https://sfo3.digitaloceanspaces.com"
    )

    region = os.environ.get("AWS_REGION", "sfo3")

    config = Config(
        signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}
    )

    client_kwargs = {
        "region_name": region,
        "config": config,
    }

    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    return boto3.client("s3", **client_kwargs)


def get_s3_bucket_name() -> str:
    """
    Get the S3 bucket name from environment or default.

    Checks:
    - DO_S3_BUCKET (for DigitalOcean)
    - AWS_S3_BUCKET
    - Or defaults to 'pixel-data'

    Returns:
        Bucket name
    """
    return (
        os.environ.get("DO_S3_BUCKET")
        or os.environ.get("AWS_S3_BUCKET")
        or "pixel-data"
    )
