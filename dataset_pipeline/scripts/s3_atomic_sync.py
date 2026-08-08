#!/usr/bin/env python3
"""S3 atomic swap sync for V7 MASTER dataset.

Uploads a local output directory to S3 using a staging path, verifies
integrity, then atomically promotes staging to the final path.  This
ensures consumers never see a partial or corrupt dataset.

Workflow:
  1. Upload all files to ``<s3_prefix>/_staging/<timestamp>/``
  2. Verify: list remote files, compare count + size against local
  3. Promote: copy staging → final path (overwrite)
  4. Cleanup: purge staging, purge stale files in final not in local set
  5. Write sync manifest to ``<s3_prefix>/_sync_manifest.json``

Usage:
    python -m dataset_pipeline.scripts.s3_atomic_sync \
        --local_dir ai/data/prepared/v7_master \
        --s3_prefix ai/training_ready/v7_master \
        --remote HetznerS3:pixeldata
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("s3_atomic_sync")

STAGING_DIR = "_staging"
SYNC_MANIFEST = "_sync_manifest.json"
MIN_REMOTE_PARTS = 2


@dataclass
class SyncFile:
    """Metadata for a single file in the sync set."""

    name: str
    local_size: int
    remote_size: int | None = None
    verified: bool = False


@dataclass
class SyncReport:
    """Result of an atomic sync operation."""

    success: bool = False
    local_dir: str = ""
    s3_prefix: str = ""
    remote: str = ""
    staging_path: str = ""
    final_path: str = ""
    files_planned: list[SyncFile] = field(default_factory=list)
    files_uploaded: list[SyncFile] = field(default_factory=list)
    files_verified: list[SyncFile] = field(default_factory=list)
    files_promoted: int = 0
    stale_files_removed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        logger.error(message)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "local_dir": self.local_dir,
            "s3_prefix": self.s3_prefix,
            "remote": self.remote,
            "staging_path": self.staging_path,
            "final_path": self.final_path,
            "files_planned": [f.__dict__ for f in self.files_planned],
            "files_uploaded": [f.__dict__ for f in self.files_uploaded],
            "files_verified": [f.__dict__ for f in self.files_verified],
            "files_promoted": self.files_promoted,
            "stale_files_removed": self.stale_files_removed,
            "errors": self.errors,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class S3AtomicSync:
    """Atomic swap S3 sync using rclone.

    Parameters
    ----------
    remote : str
        rclone remote name (e.g. ``HetznerS3:pixeldata``).
    dry_run : bool
        If True, log actions without executing rclone commands.
    """

    def __init__(self, remote: str = "HetznerS3:pixeldata", dry_run: bool = False) -> None:
        self.remote = remote.rstrip("/")
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # rclone helpers
    # ------------------------------------------------------------------

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        """Execute an rclone command, returning the completed process."""
        logger.debug("rclone: %s", " ".join(cmd))
        if self.dry_run:
            logger.info("[dry-run] %s", " ".join(cmd))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logger.warning("rclone failed (%d): %s", result.returncode, result.stderr.strip())
        return result

    def _remote_path(self, *parts: str) -> str:
        """Build a full remote path from components."""
        return "/".join([self.remote, *parts])

    def _list_remote(self, prefix: str) -> dict[str, int]:
        """List files under a remote prefix, returning {name: size}."""
        cmd = ["rclone", "lsf", "--format", "p", "-s", self._remote_path(prefix)]
        result = self._run(cmd)
        files: dict[str, int] = {}
        for raw_line in result.stdout.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # lsf -s format: "size path" or "path size" depending on flags
            # Using --format "p" gives path, -s gives size as separate field
            # Actually rclone lsf -s gives: "size path"
            parts = line.split(None, 1)
            if len(parts) == MIN_REMOTE_PARTS:
                try:
                    size = int(parts[0])
                    files[parts[1].strip()] = size
                except ValueError:
                    files[parts[0].strip()] = -1
            else:
                files[line] = -1
        return files

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync(
        self,
        local_dir: str | Path,
        s3_prefix: str,
        purge_stale: bool = True,
    ) -> SyncReport:
        """Run the full atomic swap sync.

        Parameters
        ----------
        local_dir : str | Path
            Local directory containing the consolidated V7 output.
        s3_prefix : str
            Destination prefix within the S3 bucket (e.g. ``ai/training_ready/v7_master``).
        purge_stale : bool
            If True, remove files in the final path that are not in the local set.

        Returns
        -------
        SyncReport
            Detailed report of the sync operation.
        """
        report = SyncReport(
            local_dir=str(local_dir),
            s3_prefix=s3_prefix,
            remote=self.remote,
            started_at=datetime.now(UTC).isoformat(),
        )

        local_path = Path(local_dir)
        if not local_path.is_dir():
            report.add_error(f"Local directory does not exist: {local_dir}")
            report.completed_at = datetime.now(UTC).isoformat()
            return report

        # Collect local files
        local_files = sorted(f for f in local_path.iterdir() if f.is_file())
        if not local_files:
            report.add_error(f"No files found in local directory: {local_dir}")
            report.completed_at = datetime.now(UTC).isoformat()
            return report

        # Build staging path with timestamp
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        staging_prefix = f"{s3_prefix}/{STAGING_DIR}/{ts}"
        final_prefix = s3_prefix

        report.staging_path = staging_prefix
        report.final_path = final_prefix

        # Phase 1: Plan
        for f in local_files:
            report.files_planned.append(SyncFile(name=f.name, local_size=f.stat().st_size))

        logger.info(
            "Phase 1: %d files to sync (%s → %s/%s)",
            len(local_files), local_dir, self.remote, s3_prefix,
        )

        # Phase 2: Upload to staging
        uploaded = self._upload_to_staging(local_path, staging_prefix, report)
        if not uploaded:
            report.completed_at = datetime.now(UTC).isoformat()
            return report

        # Phase 3: Verify
        verified = self._verify_staging(staging_prefix, report)
        if not verified:
            logger.error("Verification failed, aborting sync")
            self._purge_staging(staging_prefix)
            report.completed_at = datetime.now(UTC).isoformat()
            return report

        # Phase 4: Promote staging → final
        promoted = self._promote(staging_prefix, final_prefix, report)
        if not promoted:
            self._purge_staging(staging_prefix)
            report.completed_at = datetime.now(UTC).isoformat()
            return report

        # Phase 5: Cleanup
        self._purge_staging(staging_prefix)
        if purge_stale:
            self._purge_stale_files(final_prefix, local_files, report)

        # Phase 6: Write sync manifest
        self._write_manifest(final_prefix, report)

        report.success = not report.errors
        report.completed_at = datetime.now(UTC).isoformat()
        logger.info(
            "Sync complete: %d files promoted to %s/%s",
            report.files_promoted, self.remote, final_prefix,
        )
        return report

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _upload_to_staging(
        self,
        local_path: Path,
        staging_prefix: str,
        report: SyncReport,
    ) -> bool:
        """Upload all local files to the staging path."""
        local_files = sorted(f for f in local_path.iterdir() if f.is_file())
        for f in local_files:
            remote_key = f"{staging_prefix}/{f.name}"
            cmd = ["rclone", "copyto", str(f), self._remote_path(remote_key)]
            result = self._run(cmd)
            if result.returncode != 0:
                report.add_error(f"Upload failed: {f.name} — {result.stderr.strip()}")
                return False
            report.files_uploaded.append(SyncFile(name=f.name, local_size=f.stat().st_size))
            logger.info("Uploaded %s → %s", f.name, remote_key)
        return len(report.files_uploaded) == len(local_files)

    def _verify_staging(self, staging_prefix: str, report: SyncReport) -> bool:
        """Verify all staged files exist with matching sizes."""
        remote_files = self._list_remote(staging_prefix)
        planned = {f.name: f for f in report.files_planned}

        if len(remote_files) != len(planned):
            report.add_error(
                f"File count mismatch: planned {len(planned)}, remote {len(remote_files)}",
            )
            return False

        for name, planned_file in planned.items():
            remote_size = remote_files.get(name)
            if remote_size is None:
                report.add_error(f"Missing on remote: {name}")
                return False
            if remote_size not in (planned_file.local_size, -1):
                report.add_error(
                    f"Size mismatch: {name} (local={planned_file.local_size}, remote={remote_size})",
                )
                return False
            verified = SyncFile(
                name=name,
                local_size=planned_file.local_size,
                remote_size=remote_size,
                verified=True,
            )
            report.files_verified.append(verified)

        logger.info("Verification passed: %d files", len(report.files_verified))
        return True

    def _promote(self, staging_prefix: str, final_prefix: str, report: SyncReport) -> bool:
        """Copy files from staging to final path."""
        for vf in report.files_verified:
            src = f"{staging_prefix}/{vf.name}"
            dst = f"{final_prefix}/{vf.name}"
            cmd = ["rclone", "copyto", self._remote_path(src), self._remote_path(dst)]
            result = self._run(cmd)
            if result.returncode != 0:
                report.add_error(f"Promote failed: {vf.name} — {result.stderr.strip()}")
                return False
            report.files_promoted += 1
            logger.info("Promoted %s → %s", src, dst)
        return report.files_promoted == len(report.files_verified)

    def _purge_staging(self, staging_prefix: str) -> None:
        """Delete the staging directory."""
        cmd = ["rclone", "purge", self._remote_path(staging_prefix)]
        result = self._run(cmd)
        if result.returncode != 0 and not self.dry_run:
            logger.warning("Failed to purge staging: %s", result.stderr.strip())
        else:
            logger.info("Purged staging: %s", staging_prefix)

    def _purge_stale_files(
        self,
        final_prefix: str,
        local_files: list[Path],
        report: SyncReport,
    ) -> None:
        """Remove files in the final path that are not in the local set."""
        local_names = {f.name for f in local_files}
        remote_files = self._list_remote(final_prefix)
        for remote_name in remote_files:
            if remote_name in local_names:
                continue
            if remote_name == SYNC_MANIFEST:
                continue
            key = f"{final_prefix}/{remote_name}"
            cmd = ["rclone", "deletefile", self._remote_path(key)]
            result = self._run(cmd)
            if result.returncode == 0:
                report.stale_files_removed.append(remote_name)
                logger.info("Removed stale file: %s", remote_name)
            else:
                logger.warning("Failed to remove stale file: %s", remote_name)

    def _write_manifest(self, final_prefix: str, report: SyncReport) -> None:
        """Write a sync manifest to the final path."""
        manifest = {
            "synced_at": datetime.now(UTC).isoformat(),
            "remote": self.remote,
            "prefix": final_prefix,
            "file_count": report.files_promoted,
            "files": [f.name for f in report.files_verified],
            "stale_removed": report.stale_files_removed,
            "success": report.success,
        }
        key = f"{final_prefix}/{SYNC_MANIFEST}"
        if self.dry_run:
            logger.info("[dry-run] Would write manifest to %s", key)
            return
        # Write manifest via rclone rcat
        cmd = ["rclone", "rcat", self._remote_path(key)]
        proc = subprocess.run(
            cmd,
            input=json.dumps(manifest, indent=2),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            logger.warning("Failed to write sync manifest: %s", proc.stderr.strip())
        else:
            logger.info("Wrote sync manifest to %s", key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="S3 atomic swap sync for V7 MASTER dataset")
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Local directory containing consolidated V7 output",
    )
    parser.add_argument(
        "--s3_prefix",
        required=True,
        help="Destination prefix within S3 bucket (e.g. ai/training_ready/v7_master)",
    )
    parser.add_argument(
        "--remote",
        default="HetznerS3:pixeldata",
        help="rclone remote name (default: HetznerS3:pixeldata)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Log actions without executing rclone commands",
    )
    parser.add_argument(
        "--no_purge_stale",
        action="store_true",
        help="Do not remove stale files in the final path",
    )
    parser.add_argument(
        "--report_path",
        default=None,
        help="Optional path to write the sync report JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    syncer = S3AtomicSync(remote=args.remote, dry_run=args.dry_run)
    report = syncer.sync(
        local_dir=args.local_dir,
        s3_prefix=args.s3_prefix,
        purge_stale=not args.no_purge_stale,
    )

    if args.report_path:
        Path(args.report_path).write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8",
        )

    if report.success:
        print(f"Sync complete: {report.files_promoted} files promoted")
    else:
        print(f"Sync failed: {len(report.errors)} errors")
        for err in report.errors:
            print(f"  - {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
