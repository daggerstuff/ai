#!/usr/bin/env python3
"""
Generate S3 Manifest - Complete inventory of what's on S3
Shows sync status and generates manifest JSON
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    script_path = Path(__file__).resolve()
    # Go up 3 levels: scripts -> training_ready -> ai -> project_root
    return script_path.parents[3]


def ensure_project_root_on_path(project_root: Path) -> None:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size"""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def list_all_objects(loader: object, prefix: str = "") -> list[dict[str, Any]]:
    """List all objects in S3 with metadata"""
    objects: list[dict[str, Any]] = []

    try:
        s3_client = loader.s3_client  # type: ignore[attr-defined]
        bucket = loader.bucket  # type: ignore[attr-defined]

        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        for page in pages:
            objects.extend(
                [
                    {
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "size_formatted": format_size(obj["Size"]),
                        "last_modified": (
                            obj["LastModified"].isoformat() if "LastModified" in obj else None
                        ),
                        "etag": obj.get("ETag", "").strip('"'),
                    }
                    for obj in page.get("Contents", [])
                ]
            )
    except Exception:
        logger.exception("⚠️  Error listing %s", prefix)

    return objects


def categorize_objects(objects: list[dict[str, Any]]) -> dict[str, Any]:
    """Categorize objects by structure"""
    categories = {
        "gdrive": {
            "raw": [],
            "processed": defaultdict(list),
        },
        "acquired": [],
        "lightning": [],
        "voice": [],
        "pixel_voice": [],
        "metadata": {
            "consolidation": [],
        },
        "other": [],
    }

    for obj in objects:
        key = obj["key"]
        normalized_key = key.removeprefix("datasets/")
        normalized_key_lower = normalized_key.lower()

        if normalized_key.startswith("gdrive/raw/"):
            categories["gdrive"]["raw"].append(obj)
            continue

        if normalized_key.startswith("gdrive/processed/"):
            # gdrive/processed/<category>/<file> OR gdrive/processed/<file>
            parts = normalized_key.split("/")
            category = parts[2] if len(parts) >= 4 else "root"
            categories["gdrive"]["processed"][category].append(obj)
            continue

        if normalized_key.startswith("acquired/"):
            categories["acquired"].append(obj)
            continue

        if normalized_key.startswith("lightning/"):
            categories["lightning"].append(obj)
            continue

        if normalized_key.startswith("voice/"):
            categories["voice"].append(obj)
            continue

        if normalized_key.startswith("pixel_voice/"):
            categories["pixel_voice"].append(obj)
            continue

        if normalized_key.startswith("metadata/consolidation/"):
            categories["metadata"]["consolidation"].append(obj)
            continue

        if "raw" in normalized_key_lower or "gdrive" in normalized_key_lower:
            categories["gdrive"]["raw"].append(obj)
            continue

        categories["other"].append(obj)

    return categories


def build_category_section(objects: list[dict[str, Any]]) -> dict[str, Any]:
    size_bytes = sum(obj["size"] for obj in objects)
    return {
        "count": len(objects),
        "size_bytes": size_bytes,
        "size_formatted": format_size(size_bytes),
        "objects": objects,
    }


def generate_manifest(loader: object) -> dict[str, Any]:
    """Generate complete S3 manifest"""
    bucket = loader.bucket  # type: ignore[attr-defined]
    endpoint_url = loader.endpoint_url  # type: ignore[attr-defined]

    # List all objects
    logger.info("📊 Generating S3 Manifest...")
    logger.info("%s", "=" * 60)
    logger.info("1. Scanning S3 bucket...")
    all_objects = list_all_objects(loader)
    logger.info("   ✅ Found %s objects", len(all_objects))

    if not all_objects:
        logger.info("   ⚠️  No objects found in S3")
        logger.info("   💡 Sync may still be in progress")
        return {
            "bucket": bucket,
            "endpoint": endpoint_url,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "total_objects": 0,
            "total_size_bytes": 0,
            "total_size_formatted": format_size(0),
            "categories": {},
        }

    # Calculate totals
    total_size = sum(obj["size"] for obj in all_objects)

    # Categorize
    logger.info("2. Categorizing objects...")
    categories = categorize_objects(all_objects)

    # Convert defaultdict to regular dict for JSON serialization
    processed_categories = dict(categories["gdrive"]["processed"])

    manifest = {
        "bucket": bucket,
        "endpoint": endpoint_url,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "total_objects": len(all_objects),
        "total_size_bytes": total_size,
        "total_size_formatted": format_size(total_size),
        "categories": {
            "gdrive": {
                "raw": build_category_section(categories["gdrive"]["raw"]),
                "processed": {},
            },
            "acquired": build_category_section(categories["acquired"]),
            "lightning": build_category_section(categories["lightning"]),
            "voice": build_category_section(categories["voice"]),
            "pixel_voice": build_category_section(categories["pixel_voice"]),
            "metadata": {
                "consolidation": build_category_section(categories["metadata"]["consolidation"]),
            },
            "other": build_category_section(categories["other"]),
        },
    }

    # Add processed categories
    processed_section = manifest["categories"]["gdrive"]["processed"]
    for category, objects in processed_categories.items():
        processed_section[category] = build_category_section(objects)

    return manifest


def print_summary(manifest: dict[str, Any]) -> None:
    """Log a human-readable summary."""
    logger.info("%s", "=" * 60)
    logger.info("📋 S3 Manifest Summary")
    logger.info("%s", "=" * 60)

    logger.info("📦 Bucket: %s", manifest["bucket"])
    logger.info("🌐 Endpoint: %s", manifest["endpoint"])
    logger.info("📅 Generated: %s", manifest["generated_at"])

    logger.info("📊 Totals:")
    logger.info("   Objects: %s", f"{manifest['total_objects']:,}")
    logger.info("   Size: %s", manifest["total_size_formatted"])

    categories = manifest.get("categories", {})
    if not categories:
        return

    raw = categories.get("gdrive", {}).get("raw", {})
    logger.info("📁 Google Drive Raw (backup/staging):")
    logger.info("   Files: %s", f"{raw.get('count', 0):,}")
    logger.info("   Size: %s", raw.get("size_formatted", format_size(0)))
    raw_status = (
        "✅ Synced" if raw.get("count", 0) > 0 else "⏳ In Progress (per .notes/markdown/one.md)"
    )
    logger.info(
        "   Status: %s",
        raw_status,
    )

    processed = categories.get("gdrive", {}).get("processed", {})
    processed_total = sum(cat["count"] for cat in processed.values()) if processed else 0
    processed_size = sum(cat["size_bytes"] for cat in processed.values()) if processed else 0
    logger.info("📁 Google Drive Processed (canonical):")
    logger.info("   Files: %s", f"{processed_total:,}")
    logger.info("   Size: %s", format_size(processed_size))
    if processed:
        logger.info("   Categories:")
        for category, data in sorted(processed.items()):
            logger.info(
                "      - %s: %s files (%s)",
                category,
                data["count"],
                data["size_formatted"],
            )
    else:
        logger.info("   Status: ⏳ Not yet organized")

    metadata = categories.get("metadata", {})
    if metadata.get("consolidation", {}).get("count", 0) > 0:
        logger.info("📋 Consolidation Metadata:")
        logger.info("   Files: %s", f"{metadata['consolidation']['count']:,}")
        logger.info("   Size: %s", metadata["consolidation"]["size_formatted"])
        logger.info("   Location: s3://%s/datasets/metadata/consolidation/", manifest["bucket"])

    for name, key in [
        ("Acquired", "acquired"),
        ("Lightning", "lightning"),
        ("Voice", "voice"),
        ("Pixel Voice", "pixel_voice"),
    ]:
        data = categories.get(key, {})
        if data.get("count", 0) > 0:
            logger.info("📁 %s:", name)
            logger.info("   Files: %s", f"{data['count']:,}")
            logger.info("   Size: %s", data["size_formatted"])

    other = categories.get("other", {})
    if other.get("count", 0) > 0:
        logger.info("📁 Other:")
        logger.info("   Files: %s", f"{other['count']:,}")
        logger.info("   Size: %s", other["size_formatted"])


def save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def generate_and_save_manifest(*, project_root: Path, loader: object) -> Path:
    manifest = generate_manifest(loader)
    print_summary(manifest)

    manifest_path = project_root / "ai" / "training_ready" / "data" / "s3_manifest.json"
    save_manifest(manifest_path, manifest)

    logger.info("%s", "=" * 60)
    logger.info("✅ Manifest saved to: %s", manifest_path)
    logger.info("💡 Next steps:")
    logger.info("   1. Review manifest to see what's synced")
    logger.info("   2. Check sync progress in tmux session (per .notes/markdown/one.md)")
    logger.info("   3. Once raw sync completes, organize into processed/ structure")
    return manifest_path


def main() -> int:
    """Generate and save S3 manifest."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    project_root = get_project_root()
    ensure_project_root_on_path(project_root)

    if not (project_root / "ai").exists():
        logger.warning("⚠️  Warning: Could not find 'ai' directory at %s", project_root)
        logger.warning("   Script location: %s", Path(__file__).resolve())

    # Import after adding project root to sys.path
    from ai.training.ready_packages.utils.s3_dataset_loader import (  # noqa: PLC0415
        S3DatasetLoader,
    )

    logger.info("🔍 S3 Manifest Generator")
    logger.info("%s", "=" * 60)
    logger.info("📝 Note: This checks the bucket configured in S3DatasetLoader.")

    try:
        logger.info("1. Connecting to S3...")
        loader = S3DatasetLoader()
        logger.info("   ✅ Connected to %s", loader.bucket)
        generate_and_save_manifest(project_root=project_root, loader=loader)
        return 0
    except Exception:
        logger.exception("❌ Error generating S3 manifest")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
