#!/usr/bin/env python3
"""Download CPTSD source transcripts from HETZNER S3 using boto3.

Reads the source inventory and download catalog, then pulls each
file into local cptsd_sources/ subdirectories.

Usage:
    uv run ai/training/ready_packages/scripts/download_cptsd_sources.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
CATALOG_PATH = DATA_DIR / "cptsd_sources" / "download_catalog.json"
INVENTORY_PATH = DATA_DIR / "cptsd_source_inventory.json"
OUTPUT_DIR = DATA_DIR / "cptsd_sources"
REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_env() -> None:
    """Load HETZNER_S3_ vars from .env.staging if not already set."""
    if os.getenv("HETZNER_S3_ACCESS_KEY"):
        return
    for name in (".env.staging", ".env.local", ".env.production"):
        env_file = REPO_ROOT / name
        if env_file.exists():
            logger.info("Loading S3 credentials from %s", env_file)
            with open(env_file) as fh:
                for line in fh:
                    stripped = line.strip()
                    if (
                        stripped
                        and not stripped.startswith("#")
                        and "=" in stripped
                        and stripped.startswith("HETZNER_S3_")
                    ):
                        key, _, val = stripped.partition("=")
                        val = val.strip("'\"")
                        os.environ.setdefault(key, val)
            return


def _get_s3_client():
    """Build a boto3 S3 client for HETZNER."""
    try:
        import boto3
    except ImportError:
        logger.error("boto3 is required. Install: uv add boto3")
        sys.exit(1)

    endpoint = os.getenv("HETZNER_S3_ENDPOINT")
    access_key = os.getenv("HETZNER_S3_ACCESS_KEY")
    secret_key = os.getenv("HETZNER_S3_SECRET_KEY")
    region = os.getenv("HETZNER_S3_REGION", "hel1")

    if not all([endpoint, access_key, secret_key]):
        logger.error(
            "Missing S3 credentials. Set HETZNER_S3_ENDPOINT, "
            "HETZNER_S3_ACCESS_KEY, HETZNER_S3_SECRET_KEY"
        )
        sys.exit(1)

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )


def _author_dir(s3_key: str, source_name: str) -> str:
    """Determine the author directory name from the S3 key."""
    key_lower = s3_key.lower()
    if "heidi priebe" in key_lower:
        return "heidi_priebe"
    if "patrick teahan" in key_lower:
        return "patrick_teahan"
    if "tim fletcher" in key_lower:
        return "tim_fletcher"
    if "pixelated-v2" in key_lower:
        return "processed_transcripts"
    if "crappy childhood" in key_lower:
        return "crappy_childhood_fairy"
    return source_name


def _collect_s3_keys() -> list[dict]:
    """Collect all S3 keys to download from inventory."""
    files_to_download: list[dict] = []

    if INVENTORY_PATH.exists():
        with open(INVENTORY_PATH) as fh:
            inventory = json.load(fh)
        for source_name, source in inventory.get("sources", {}).items():
            if source_name == "existing_cptsd_dataset":
                continue
            for file_entry in source.get("files", []):
                files_to_download.append(
                    {
                        "key": file_entry["key"],
                        "source": source_name,
                    }
                )
    elif CATALOG_PATH.exists():
        with open(CATALOG_PATH) as fh:
            catalog = json.load(fh)
        for source_name, source in catalog.get("sources", {}).items():
            for file_entry in source.get("files", []):
                files_to_download.append(
                    {
                        "key": file_entry["s3_key"],
                        "source": source_name,
                    }
                )
    else:
        logger.error(
            "No inventory or catalog found at %s or %s",
            INVENTORY_PATH,
            CATALOG_PATH,
        )
        sys.exit(1)

    return files_to_download


def download_all() -> dict:
    """Download all CPTSD sources from S3."""
    _load_env()

    bucket = os.getenv("HETZNER_S3_BUCKET", "pixel-data")
    client = _get_s3_client()
    files_to_download = _collect_s3_keys()

    stats = {
        "total": len(files_to_download),
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
    }

    for entry in files_to_download:
        s3_key = entry["key"]
        source = entry["source"]
        filename = Path(s3_key).name
        author = _author_dir(s3_key, source)

        local_dir = OUTPUT_DIR / author
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / filename

        if local_path.exists() and local_path.stat().st_size > 0:
            logger.info("SKIP (exists): %s", filename[:60])
            stats["skipped"] += 1
            continue

        logger.info(
            "Downloading: %s -> %s/%s",
            s3_key[:70],
            author,
            filename[:40],
        )
        try:
            client.download_file(bucket, s3_key, str(local_path))
            size = local_path.stat().st_size
            logger.info(
                "  OK: %s (%.1f KB)",
                filename[:50],
                size / 1024,
            )
            stats["downloaded"] += 1
        except Exception as exc:
            logger.warning(
                "  FAIL: %s — %s",
                s3_key[:50],
                str(exc)[:80],
            )
            stats["failed"] += 1
            stats["errors"].append({"key": s3_key, "error": str(exc)})

    # Update catalog status
    catalog_out = OUTPUT_DIR / "download_results.json"
    with open(catalog_out, "w") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Results saved to %s", catalog_out)

    return stats


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    logger.info("CPTSD Source Downloader (boto3)")
    logger.info("Output: %s", OUTPUT_DIR)

    stats = download_all()

    logger.info(
        "Done: %d/%d downloaded, %d skipped, %d failed",
        stats["downloaded"],
        stats["total"],
        stats["skipped"],
        stats["failed"],
    )

    if stats["errors"]:
        logger.warning("Errors:")
        for err in stats["errors"][:5]:
            logger.warning(
                "  %s: %s",
                err["key"][:40],
                err["error"][:60],
            )

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
