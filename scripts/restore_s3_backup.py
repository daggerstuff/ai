#!/usr/bin/env python3
"""
Restore S3 backup from Google Drive to Hetzner Object Storage.

Syncs gdrive:backups/S3-Complete/* to BackupStorageS3:pixel-data/
"""

from datetime import datetime, timezone

import subprocess
import sys


def run_rclone(args: list[str], dry_run: bool = False) -> tuple[int, str, str]:
    """Run rclone command."""
    cmd = ["rclone"] + args
    if dry_run and "sync" in args[0]:
        cmd.insert(1, "--dry-run")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Restore S3 backup from Google Drive to DO Spaces"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without doing it",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check source and destination before syncing",
    )
    parser.add_argument("--log", type=str, default=None, help="Log file path")

    args = parser.parse_args()

    source = "gdrive:backups/S3-Complete"
    dest = "BackupStorageS3:pixel-data"

    print("=" * 80)
    print("S3 BACKUP RESTORE")
    print("=" * 80)
    print(f"Source: {source}")
    print(f"Destination: {dest}")
    print(f"Dry run: {args.dry_run}")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}Z")
    print()

    if args.check:
        print("### Checking source ###")
        rc, out, err = run_rclone(["lsf", source])
        if rc != 0:
            print(f"ERROR: Cannot access source: {err}")
            return 1
        print(f"Source directories:\n{out}")

        print("\n### Checking destination ###")
        rc, out, err = run_rclone(["lsf", dest])
        if rc != 0:
            print(f"WARNING: Destination may be empty or inaccessible: {err}")
        else:
            print(f"Destination contents:\n{out}")

        print("\n### Source size ###")
        rc, out, err = run_rclone(["size", source])
        print(out)
        return 0

    print("### Starting sync ###")
    print("This will sync all files from Google Drive backup to DO Spaces")
    print()

    # Use sync with --progress for visibility
    sync_args = [
        "sync",
        source,
        dest,
        "--progress",
        "--transfers",
        "4",
        "--checkers",
        "8",
        "--contimeout",
        "60s",
        "--timeout",
        "300s",
        "--retries",
        "3",
        "--low-level-retries",
        "10",
        "--stats",
        "30s",
        "--stats-one-line",
    ]

    if args.log:
        sync_args.extend(["--log-file", args.log])

    rc, out, err = run_rclone(sync_args, dry_run=args.dry_run)

    print()
    print("=" * 80)
    if rc == 0:
        print("✅ SYNC COMPLETE")
        if args.dry_run:
            print("(Dry run - no files were actually transferred)")
    else:
        print("❌ SYNC FAILED")
        print(f"Error: {err}")
    print("=" * 80)

    return rc


if __name__ == "__main__":
    sys.exit(main())
