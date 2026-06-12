#!/usr/bin/env python3
"""
Restore S3 backup from Google Drive to Hetzner Object Storage.

Syncs gdrive:backups/S3-Complete/* to BackupStorageS3:pixel-data/
"""

import argparse
import subprocess
import sys


def run_rclone(args: list[str], dry_run: bool = False) -> tuple[int, str, str]:
    """Run rclone command."""
    cmd = ["rclone", *args]
    if dry_run and "sync" in args[0]:
        cmd.insert(1, "--dry-run")

    result = subprocess.run(cmd, capture_output=True, text=True, shell=False, check=False)
    return result.returncode, result.stdout, result.stderr


def main():

    parser = argparse.ArgumentParser(description="Restore S3 backup from Google Drive to DO Spaces")
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

    if args.check:
        rc, _out, _err = run_rclone(["lsf", source])
        if rc != 0:
            return 1

        rc, _out, _err = run_rclone(["lsf", dest])
        if rc != 0:
            pass
        else:
            pass

        _rc, _out, _err = run_rclone(["size", source])
        return 0

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

    rc, _out, _err = run_rclone(sync_args, dry_run=args.dry_run)

    if rc == 0:
        if args.dry_run:
            pass
    else:
        pass

    return rc


if __name__ == "__main__":
    sys.exit(main())
