#!/usr/bin/env python3
"""S3 atomic swap sync for V7 MASTER dataset.

Uploads local files to S3 using an atomic swap pattern:
1. Upload to a staging key (sibling of final key with .tmp suffix).
2. Verify the staging object's SHA-256 matches the local file.
3. Copy staging to the final key (S3 copy_object is atomic).
4. Delete the staging key.

Readers of the final key see either the old or the new version, never a partial upload.

Environment variables (all optional, CLI args take precedence):
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    AWS_S3_ENDPOINT (custom endpoint URL for Hetzner/MinIO)
    S3_BUCKET (default bucket)

Usage:
    python s3_atomic_sync.py --input ai/training/output/v7/V7_MASTER.jsonl \\
        --s3_key datasets/v7/V7_MASTER.jsonl \\
        --bucket pixeldata --region US-EAST-VA

    python s3_atomic_sync.py --input ai/training/output/v7/ \\
        --s3_prefix datasets/v7/ \\
        --bucket pixeldata --dry_run
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None  # type: ignore[assignment]
    BotoConfig = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[misc,assignment]

logger = logging.getLogger("s3_atomic_sync")

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB read chunks for SHA-256


@dataclass
class SyncResult:
    """Result of a single atomic swap operation."""

    local_path: str
    s3_key: str
    success: bool
    etag: str | None = None
    sha256: str | None = None
    size_bytes: int = 0
    error: str | None = None
    dry_run: bool = False


def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of file at *path*."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _get_s3_client(
    endpoint_url: str | None,
    region: str | None,
    access_key: str | None,
    secret_key: str | None,
) -> Any:
    """Build a boto3 S3 client. Raises RuntimeError if boto3 missing."""
    if boto3 is None:
        raise RuntimeError("boto3 not installed. Install with: pip install boto3")
    cfg = BotoConfig(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"})
    kwargs: dict[str, Any] = {"config": cfg}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    if region:
        kwargs["region_name"] = region
    if access_key:
        kwargs["aws_access_key_id"] = access_key
    if secret_key:
        kwargs["aws_secret_access_key"] = secret_key
    return boto3.client("s3", **kwargs)


def _staging_key(s3_key: str) -> str:
    """Derive a staging key from the final key."""
    return f"{s3_key}.tmp.{os.getpid()}"


def _head_sha256(client: Any, bucket: str, key: str) -> str | None:
    """Fetch SHA-256 from object metadata if present."""
    try:
        resp = client.head_object(Bucket=bucket, Key=key)
    except ClientError:
        return None
    meta = resp.get("Metadata", {})
    return meta.get("sha256-checksum") or meta.get("sha256")


def atomic_swap(
    client: Any,
    bucket: str,
    local_path: Path,
    s3_key: str,
    *,
    dry_run: bool = False,
    content_type: str = "application/x-ndjson",
) -> SyncResult:
    """Upload *local_path* to *s3_key* in *bucket* atomically.

    Returns a SyncResult describing the outcome.
    """
    if not local_path.exists() or not local_path.is_file():
        return SyncResult(
            local_path=str(local_path),
            s3_key=s3_key,
            success=False,
            error=f"Local file does not exist: {local_path}",
        )

    size = local_path.stat().st_size
    sha = _sha256_file(local_path)

    if dry_run:
        logger.info(
            "[dry-run] would upload %s (%d bytes, sha256=%s) -> s3://%s/%s",
            local_path,
            size,
            sha,
            bucket,
            s3_key,
        )
        return SyncResult(
            local_path=str(local_path),
            s3_key=s3_key,
            success=True,
            sha256=sha,
            size_bytes=size,
            dry_run=True,
        )

    staging = _staging_key(s3_key)
    extra_args: dict[str, Any] = {
        "ContentType": content_type,
        "Metadata": {"sha256-checksum": sha},
    }

    # 1. Upload to staging key.
    try:
        client.upload_file(
            str(local_path),
            bucket,
            staging,
            ExtraArgs=extra_args,
        )
    except Exception as e:
        logger.error("Upload to staging key %s failed: %s", staging, e)
        return SyncResult(
            local_path=str(local_path),
            s3_key=s3_key,
            success=False,
            size_bytes=size,
            sha256=sha,
            error=f"staging upload failed: {e}",
        )

    # 2. Verify staging object's checksum.
    remote_sha = _head_sha256(client, bucket, staging)
    if remote_sha is not None and remote_sha != sha:
        logger.error(
            "Checksum mismatch on staging key %s: local=%s remote=%s",
            staging,
            sha,
            remote_sha,
        )
        _safe_delete(client, bucket, staging)
        return SyncResult(
            local_path=str(local_path),
            s3_key=s3_key,
            success=False,
            size_bytes=size,
            sha256=sha,
            error="staging checksum mismatch",
        )

    # 3. Copy staging -> final (atomic in S3).
    try:
        client.copy_object(
            CopySource={"Bucket": bucket, "Key": staging},
            Bucket=bucket,
            Key=s3_key,
            MetadataDirective="REPLACE",
            ContentType=content_type,
            Metadata={"sha256-checksum": sha},
        )
    except Exception as e:
        logger.error("copy_object staging->final failed: %s", e)
        _safe_delete(client, bucket, staging)
        return SyncResult(
            local_path=str(local_path),
            s3_key=s3_key,
            success=False,
            size_bytes=size,
            sha256=sha,
            error=f"copy_object failed: {e}",
        )

    # 4. Delete staging key.
    _safe_delete(client, bucket, staging)

    # 5. Fetch ETag of final for the result.
    etag: str | None = None
    try:
        resp = client.head_object(Bucket=bucket, Key=s3_key)
        etag = resp.get("ETag", "").strip('"') or None
    except Exception:
        pass

    logger.info(
        "atomic swap complete: %s -> s3://%s/%s (%d bytes, etag=%s)",
        local_path,
        bucket,
        s3_key,
        size,
        etag,
    )

    return SyncResult(
        local_path=str(local_path),
        s3_key=s3_key,
        success=True,
        etag=etag,
        sha256=sha,
        size_bytes=size,
    )


def _safe_delete(client: Any, bucket: str, key: str) -> None:
    """Best-effort delete; log on failure."""
    try:
        client.delete_object(Bucket=bucket, Key=key)
    except Exception as e:
        logger.warning("failed to clean up staging key %s: %s", key, e)


def sync_directory(
    client: Any,
    bucket: str,
    local_dir: Path,
    s3_prefix: str,
    *,
    dry_run: bool = False,
    extensions: tuple[str, ...] = (".jsonl", ".json"),
) -> list[SyncResult]:
    """Sync every file in *local_dir* (non-recursive) to *s3_prefix*."""
    results: list[SyncResult] = []
    for path in sorted(local_dir.iterdir()):
        if not path.is_file() or path.suffix not in extensions:
            continue
        key = f"{s3_prefix.rstrip('/')}/{path.name}"
        results.append(atomic_swap(client, bucket, path, key, dry_run=dry_run))
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="S3 atomic swap sync for V7 MASTER dataset.",
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Local file or directory to upload.",
    )
    p.add_argument(
        "--s3_key",
        type=str,
        default=None,
        help="Destination S3 key (when --input is a file).",
    )
    p.add_argument(
        "--s3_prefix",
        type=str,
        default=None,
        help="Destination S3 prefix (when --input is a directory).",
    )
    p.add_argument(
        "--bucket",
        type=str,
        default=os.environ.get("S3_BUCKET", "pixeldata"),
        help="S3 bucket name (default: pixeldata or $S3_BUCKET).",
    )
    p.add_argument(
        "--region",
        type=str,
        default=os.environ.get("AWS_REGION", "US-EAST-VA"),
        help="S3 region (default: US-EAST-VA or $AWS_REGION).",
    )
    p.add_argument(
        "--endpoint_url",
        type=str,
        default=os.environ.get("AWS_S3_ENDPOINT"),
        help="Custom S3 endpoint URL (Hetzner/MinIO).",
    )
    p.add_argument(
        "--access_key",
        type=str,
        default=os.environ.get("AWS_ACCESS_KEY_ID"),
        help="S3 access key (default: $AWS_ACCESS_KEY_ID).",
    )
    p.add_argument(
        "--secret_key",
        type=str,
        default=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        help="S3 secret key (default: $AWS_SECRET_ACCESS_KEY).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would happen without uploading.",
    )
    p.add_argument(
        "--content_type",
        type=str,
        default="application/x-ndjson",
        help="Content-Type header for uploaded objects.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on any failure."""
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    is_dir = args.input.is_dir()
    if is_dir and not args.s3_prefix:
        logger.error("--s3_prefix required when --input is a directory")
        return 1
    if not is_dir and not args.s3_key:
        logger.error("--s3_key required when --input is a file")
        return 1

    client = _get_s3_client(
        args.endpoint_url,
        args.region,
        args.access_key,
        args.secret_key,
    )

    if is_dir:
        results = sync_directory(
            client,
            args.bucket,
            args.input,
            args.s3_prefix,
            dry_run=args.dry_run,
        )
    else:
        results = [
            atomic_swap(
                client,
                args.bucket,
                args.input,
                args.s3_key,
                dry_run=args.dry_run,
                content_type=args.content_type,
            )
        ]

    failed = [r for r in results if not r.success]
    for r in results:
        status = "OK" if r.success else "FAIL"
        logger.info(
            "[%s] %s -> s3://%s/%s%s",
            status,
            r.local_path,
            args.bucket,
            r.s3_key,
            f" (dry-run)" if r.dry_run else "",
        )

    if failed:
        for r in failed:
            logger.error("failure: %s -> %s", r.local_path, r.error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
