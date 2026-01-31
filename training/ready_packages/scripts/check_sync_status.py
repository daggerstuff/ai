#!/usr/bin/env python3
"""
Check S3 Sync Status - Quick status of what's synced vs what's expected
Compares S3 contents with expected structure from documentation
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_project_root_on_path(project_root: Path) -> None:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def log_lines(*lines: str) -> None:
    for line in lines:
        logger.info("%s", line)


def list_datasets(loader: object, *, title: str, prefix: str) -> list[str]:
    logger.info("%s", title)
    result = loader.list_datasets(prefix=prefix)  # type: ignore[attr-defined]
    logger.info("   Found: %s datasets", len(result))
    return result


def log_dataset_list(*, title: str, datasets: list[str], limit: int = 5) -> None:
    logger.info("%s", title)
    for dataset in datasets[:limit]:
        logger.info("      - %s", dataset)
    if len(datasets) > limit:
        logger.info("      ... and %s more", len(datasets) - limit)


def run_sync_checks(loader: object) -> int:
    logger.info("")
    logger.info("🔍 Checking S3: %s", loader.bucket)  # type: ignore[attr-defined]
    logger.info("%s", "=" * 60)

    processed = list_datasets(
        loader,
        title="\n1. Checking processed tier (canonical)...",
        prefix="datasets/gdrive/processed/",
    ) or list_datasets(
        loader,
        title="   (fallback) Checking processed tier...",
        prefix="gdrive/processed/",
    )

    if processed:
        log_dataset_list(
            title="   ✅ Processed tier has data",
            datasets=processed,
        )
    else:
        logger.info("   ⚠️  No processed datasets found")

    raw = list_datasets(
        loader,
        title="\n2. Checking raw tier (backup/staging)...",
        prefix="datasets/gdrive/raw/",
    ) or list_datasets(
        loader,
        title="   (fallback) Checking raw tier...",
        prefix="gdrive/raw/",
    )

    if raw:
        log_dataset_list(
            title="   ✅ Raw tier has data (sync in progress)",
            datasets=raw,
        )
    else:
        logger.info("   ⏳ Raw tier empty (sync may not have started)")

    acquired = list_datasets(
        loader,
        title="\n3. Checking acquired datasets...",
        prefix="acquired/",
    )
    if acquired:
        log_dataset_list(
            title="   ✅ Acquired datasets",
            datasets=acquired,
            limit=50,
        )

    logger.info("")
    logger.info("%s", "=" * 60)
    logger.info("📋 Summary:")
    logger.info("   Processed (canonical): %s datasets", len(processed))
    logger.info("   Raw (backup): %s datasets", len(raw))
    logger.info("   Acquired: %s datasets", len(acquired))

    if raw:
        log_lines(
            "",
            "💡 Sync Status:",
            "   ⏳ Raw sync appears to be in progress",
            "   💡 Check tmux session for progress:",
            "      tmux attach",
            "      # Look for rclone process",
            "   💡 Check log file: upload_raw_final.log",
        )

    if not processed and raw:
        log_lines(
            "",
            "⚠️  Note:",
            "   Raw data exists but not yet organized into processed/ structure",
            "   This is expected - processed/ is the canonical organized structure",
        )

    return 0


def main() -> int:
    """Check sync status based on .notes/markdown/one.md."""
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

    logger.info("📊 S3 Sync Status Check")
    logger.info("%s", "=" * 60)

    log_lines(
        "📝 Expected Status (from .notes/markdown/one.md):",
        "   ✅ processed tier: DONE",
        "   ⏳ raw tier: IN PROGRESS",
        "   🔄 Running in tmux session:",
        "      rclone copy gdrive:datasets ovh:pixel-data/datasets/gdrive/raw",
        "   📋 Log: upload_raw_final.log",
        "",
        "⚠️  Note: Sync uses bucket 'pixel-data',",
        "   but loader checks 'pixel-data'",
        "   If no data found, check if buckets are different",
        "   or sync target needs update",
    )

    try:
        loader = S3DatasetLoader()
        return run_sync_checks(loader)
    except Exception as e:
        logger.error("❌ Error: %s", e)
        logger.error("%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
