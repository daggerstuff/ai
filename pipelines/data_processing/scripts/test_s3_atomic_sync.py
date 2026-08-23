"""Tests for S3 atomic swap sync module."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure ai/ is on sys.path
_ai_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ai_root not in sys.path:
    sys.path.insert(0, _ai_root)

from pipelines.data_processing.scripts.s3_atomic_sync import (
    S3AtomicSync,
    SyncFile,
    SyncReport,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_report() -> SyncReport:
    return SyncReport()


def make_syncer(dry_run: bool = False) -> S3AtomicSync:
    return S3AtomicSync(remote="HetznerS3:pixeldata", dry_run=dry_run)


class FakeCompletedProcess:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def make_local_files(tmp_path: Path, names: list[str], content: str = "x" * 100) -> Path:
    """Create a local dir with files (100 bytes each), return the dir path."""
    d = tmp_path / "v7_master"
    d.mkdir(exist_ok=True)
    for name in names:
        f = d / name
        f.write_text(content, encoding="utf-8")
    return d


def mock_list_remote(files: dict[str, int]) -> str:
    """Build rclone lsf output string."""
    lines = []
    for name, size in files.items():
        lines.append(f"{size} {name}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SyncFile
# ---------------------------------------------------------------------------

class TestSyncFile:
    def test_defaults(self) -> None:
        f = SyncFile(name="shard_0000.jsonl", local_size=1024)
        assert f.name == "shard_0000.jsonl"
        assert f.local_size == 1024
        assert f.remote_size is None
        assert f.verified is False

    def test_with_remote(self) -> None:
        f = SyncFile(name="test.jsonl", local_size=100, remote_size=100, verified=True)
        assert f.remote_size == 100
        assert f.verified is True


# ---------------------------------------------------------------------------
# SyncReport
# ---------------------------------------------------------------------------

class TestSyncReport:
    def test_defaults(self) -> None:
        r = SyncReport()
        assert r.success is False
        assert r.files_planned == []
        assert r.errors == []
        assert r.files_promoted == 0

    def test_add_error(self) -> None:
        r = SyncReport()
        r.add_error("test error")
        assert len(r.errors) == 1
        assert r.errors[0] == "test error"

    def test_to_dict(self) -> None:
        r = SyncReport(
            success=True,
            local_dir="/tmp/v7",
            s3_prefix="ai/v7",
            remote="HetznerS3:pixeldata",
        )
        d = r.to_dict()
        assert d["success"] is True
        assert d["local_dir"] == "/tmp/v7"
        assert d["s3_prefix"] == "ai/v7"
        assert d["files_planned"] == []
        assert d["files_promoted"] == 0


# ---------------------------------------------------------------------------
# S3AtomicSync._run
# ---------------------------------------------------------------------------

class TestRun:
    def test_dry_run(self) -> None:
        syncer = make_syncer(dry_run=True)
        result = syncer._run(["rclone", "lsf", "test"])
        assert result.returncode == 0
        assert result.stdout == ""

    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_real_run_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = FakeCompletedProcess(0, "output", "")
        syncer = make_syncer()
        result = syncer._run(["rclone", "lsf", "test"])
        assert result.returncode == 0
        assert result.stdout == "output"

    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_real_run_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = FakeCompletedProcess(1, "", "error")
        syncer = make_syncer()
        result = syncer._run(["rclone", "lsf", "test"])
        assert result.returncode == 1


# ---------------------------------------------------------------------------
# S3AtomicSync._list_remote
# ---------------------------------------------------------------------------

class TestListRemote:
    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_normal(self, mock_run: MagicMock) -> None:
        mock_run.return_value = FakeCompletedProcess(
            0, "1024 shard_0000.jsonl\n2048 shard_0001.jsonl\n", "",
        )
        syncer = make_syncer()
        files = syncer._list_remote("ai/v7")
        assert files == {"shard_0000.jsonl": 1024, "shard_0001.jsonl": 2048}

    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_empty(self, mock_run: MagicMock) -> None:
        mock_run.return_value = FakeCompletedProcess(0, "", "")
        syncer = make_syncer()
        files = syncer._list_remote("ai/v7")
        assert files == {}


# ---------------------------------------------------------------------------
# S3AtomicSync._remote_path
# ---------------------------------------------------------------------------

class TestRemotePath:
    def test_basic(self) -> None:
        syncer = make_syncer()
        assert syncer._remote_path("a", "b") == "HetznerS3:pixeldata/a/b"

    def test_trailing_slash_remote(self) -> None:
        syncer = S3AtomicSync(remote="HetznerS3:pixeldata/")
        assert syncer._remote_path("a") == "HetznerS3:pixeldata/a"


# ---------------------------------------------------------------------------
# S3AtomicSync.sync — full flow
# ---------------------------------------------------------------------------

class TestSync:
    def test_missing_local_dir(self, tmp_path: Path) -> None:
        syncer = make_syncer()
        report = syncer.sync(str(tmp_path / "nonexistent"), "ai/v7")
        assert not report.success
        assert any("does not exist" in e for e in report.errors)

    def test_empty_local_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        syncer = make_syncer()
        report = syncer.sync(d, "ai/v7")
        assert not report.success
        assert any("No files found" in e for e in report.errors)

    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_upload_failure_aborts(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = FakeCompletedProcess(1, "", "upload error")
        d = make_local_files(tmp_path, ["shard_0000.jsonl"])
        syncer = make_syncer()
        report = syncer.sync(d, "ai/v7")
        assert not report.success
        assert any("Upload failed" in e for e in report.errors)

    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_verify_count_mismatch(self, mock_run: MagicMock, tmp_path: Path) -> None:
        # Upload succeeds, but listing shows fewer files
        call_count = [0]

        def side_effect(cmd, **_kwargs):
            call_count[0] += 1
            if cmd[0] == "rclone" and cmd[1] == "copyto" and call_count[0] <= 2:
                return FakeCompletedProcess(0, "", "")
            if cmd[0] == "rclone" and cmd[1] == "lsf":
                return FakeCompletedProcess(0, "100 shard_0000.jsonl\n", "")
            return FakeCompletedProcess(0, "", "")

        mock_run.side_effect = side_effect
        d = make_local_files(tmp_path, ["shard_0000.jsonl", "shard_0001.jsonl"])
        syncer = make_syncer()
        report = syncer.sync(d, "ai/v7")
        assert not report.success
        assert any("File count mismatch" in e for e in report.errors)

    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_verify_size_mismatch(self, mock_run: MagicMock, tmp_path: Path) -> None:
        call_count = [0]

        def side_effect(cmd, **_kwargs):
            call_count[0] += 1
            if cmd[0] == "rclone" and cmd[1] == "copyto" and call_count[0] <= 2:
                return FakeCompletedProcess(0, "", "")
            if cmd[0] == "rclone" and cmd[1] == "lsf":
                return FakeCompletedProcess(0, "1 shard_0000.jsonl\n1 shard_0001.jsonl\n", "")
            return FakeCompletedProcess(0, "", "")

        mock_run.side_effect = side_effect
        d = make_local_files(tmp_path, ["shard_0000.jsonl", "shard_0001.jsonl"])
        syncer = make_syncer()
        report = syncer.sync(d, "ai/v7")
        assert not report.success
        assert any("Size mismatch" in e for e in report.errors)

    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_full_success(self, mock_run: MagicMock, tmp_path: Path) -> None:
        call_count = [0]

        def side_effect(cmd, **_kwargs):
            call_count[0] += 1
            if cmd[0] == "rclone" and cmd[1] == "copyto":
                return FakeCompletedProcess(0, "", "")
            if cmd[0] == "rclone" and cmd[1] == "lsf":
                # First call: verify staging (2 files)
                # Second call: stale check (only manifest + 2 files)
                if call_count[0] <= 4:
                    return FakeCompletedProcess(
                        0,
                        "100 shard_0000.jsonl\n100 shard_0001.jsonl\n", "",
                    )
                return FakeCompletedProcess(
                    0,
                    "100 shard_0000.jsonl\n100 shard_0001.jsonl\n", "",
                )
            if cmd[0] == "rclone" and cmd[1] == "purge":
                return FakeCompletedProcess(0, "", "")
            return FakeCompletedProcess(0, "", "")

        mock_run.side_effect = side_effect
        d = make_local_files(tmp_path, ["shard_0000.jsonl", "shard_0001.jsonl"])
        syncer = make_syncer()
        report = syncer.sync(d, "ai/v7")
        assert report.success
        assert report.files_promoted == 2

    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.Popen")
    def test_full_success_with_manifest(self, mock_popen: MagicMock, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        call_count = [0]

        def side_effect(cmd, **_kwargs):
            call_count[0] += 1
            if cmd[0] == "rclone" and cmd[1] == "copyto":
                return FakeCompletedProcess(0, "", "")
            if cmd[0] == "rclone" and cmd[1] == "lsf":
                return FakeCompletedProcess(
                    0, "100 manifest.json\n100 shard_0000.jsonl\n", "",
                )
            if cmd[0] == "rclone" and cmd[1] == "purge":
                return FakeCompletedProcess(0, "", "")
            return FakeCompletedProcess(0, "", "")

        mock_run.side_effect = side_effect
        d = make_local_files(tmp_path, ["shard_0000.jsonl", "manifest.json"])
        syncer = make_syncer()
        report = syncer.sync(d, "ai/v7")
        assert report.success

    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_dry_run_success(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = FakeCompletedProcess(0, "", "")
        d = make_local_files(tmp_path, ["shard_0000.jsonl"])
        syncer = make_syncer(dry_run=True)
        report = syncer.sync(d, "ai/v7")
        # Dry run: _run returns empty stdout, so _list_remote returns {}
        # verify will fail because count mismatch (0 remote vs 1 local)
        # This is expected — dry run can't verify
        assert not report.success

    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_promote_failure_aborts(self, mock_run: MagicMock, tmp_path: Path) -> None:
        call_count = [0]

        def side_effect(cmd, **_kwargs):
            call_count[0] += 1
            if cmd[0] == "rclone" and cmd[1] == "copyto":
                # First 2 calls: upload to staging (success)
                # Later calls: promote (fail)
                if call_count[0] <= 2:
                    return FakeCompletedProcess(0, "", "")
                return FakeCompletedProcess(1, "", "promote error")
            if cmd[0] == "rclone" and cmd[1] == "lsf":
                return FakeCompletedProcess(
                    0, "100 shard_0000.jsonl\n100 shard_0001.jsonl\n", "",
                )
            if cmd[0] == "rclone" and cmd[1] == "purge":
                return FakeCompletedProcess(0, "", "")
            return FakeCompletedProcess(0, "", "")

        mock_run.side_effect = side_effect
        d = make_local_files(tmp_path, ["shard_0000.jsonl", "shard_0001.jsonl"])
        syncer = make_syncer()
        report = syncer.sync(d, "ai/v7")
        assert not report.success
        assert any("Promote failed" in e for e in report.errors)

    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_stale_files_removed(self, mock_run: MagicMock, tmp_path: Path) -> None:
        call_count = [0]

        def side_effect(cmd, **_kwargs):
            call_count[0] += 1
            if cmd[0] == "rclone" and cmd[1] == "copyto":
                return FakeCompletedProcess(0, "", "")
            if cmd[0] == "rclone" and cmd[1] == "lsf":
                # Verify: 1 file in staging
                if call_count[0] <= 2:
                    return FakeCompletedProcess(0, "100 shard_0000.jsonl\n", "")
                # Stale check: has old file + new file
                return FakeCompletedProcess(
                    0,
                    "100 shard_0000.jsonl\n100 old_shard.jsonl\n", "",
                )
            if cmd[0] == "rclone" and cmd[1] == "deletefile":
                return FakeCompletedProcess(0, "", "")
            if cmd[0] == "rclone" and cmd[1] == "purge":
                return FakeCompletedProcess(0, "", "")
            return FakeCompletedProcess(0, "", "")

        mock_run.side_effect = side_effect
        d = make_local_files(tmp_path, ["shard_0000.jsonl"])
        syncer = make_syncer()
        report = syncer.sync(d, "ai/v7")
        assert report.success
        assert "old_shard.jsonl" in report.stale_files_removed

    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_no_purge_stale(self, mock_run: MagicMock, tmp_path: Path) -> None:
        call_count = [0]

        def side_effect(cmd, **_kwargs):
            call_count[0] += 1
            if cmd[0] == "rclone" and cmd[1] == "copyto":
                return FakeCompletedProcess(0, "", "")
            if cmd[0] == "rclone" and cmd[1] == "lsf":
                return FakeCompletedProcess(0, "100 shard_0000.jsonl\n", "")
            if cmd[0] == "rclone" and cmd[1] == "purge":
                return FakeCompletedProcess(0, "", "")
            return FakeCompletedProcess(0, "", "")

        mock_run.side_effect = side_effect
        d = make_local_files(tmp_path, ["shard_0000.jsonl"])
        syncer = make_syncer()
        report = syncer.sync(d, "ai/v7", purge_stale=False)
        assert report.success
        assert report.stale_files_removed == []


# ---------------------------------------------------------------------------
# S3AtomicSync._upload_to_staging
# ---------------------------------------------------------------------------

class TestUploadToStaging:
    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_success(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = FakeCompletedProcess(0, "", "")
        d = make_local_files(tmp_path, ["a.jsonl", "b.jsonl"])
        syncer = make_syncer()
        report = make_report()
        report.files_planned = [
            SyncFile(name="a.jsonl", local_size=100),
            SyncFile(name="b.jsonl", local_size=200),
        ]
        result = syncer._upload_to_staging(d, "staging/prefix", report)
        assert result is True
        assert len(report.files_uploaded) == 2

    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_failure(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = FakeCompletedProcess(1, "", "error")
        d = make_local_files(tmp_path, ["a.jsonl"])
        syncer = make_syncer()
        report = make_report()
        result = syncer._upload_to_staging(d, "staging/prefix", report)
        assert result is False
        assert len(report.errors) > 0


# ---------------------------------------------------------------------------
# S3AtomicSync._verify_staging
# ---------------------------------------------------------------------------

class TestVerifyStaging:
    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = FakeCompletedProcess(0, "100 a.jsonl\n200 b.jsonl\n", "")
        syncer = make_syncer()
        report = make_report()
        report.files_planned = [
            SyncFile(name="a.jsonl", local_size=100),
            SyncFile(name="b.jsonl", local_size=200),
        ]
        result = syncer._verify_staging("staging/prefix", report)
        assert result is True
        assert len(report.files_verified) == 2

    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_missing_file(self, mock_run: MagicMock) -> None:
        mock_run.return_value = FakeCompletedProcess(0, "100 a.jsonl\n", "")
        syncer = make_syncer()
        report = make_report()
        report.files_planned = [
            SyncFile(name="a.jsonl", local_size=100),
            SyncFile(name="b.jsonl", local_size=200),
        ]
        result = syncer._verify_staging("staging/prefix", report)
        assert result is False

    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_size_mismatch(self, mock_run: MagicMock) -> None:
        mock_run.return_value = FakeCompletedProcess(0, "50 a.jsonl\n", "")
        syncer = make_syncer()
        report = make_report()
        report.files_planned = [
            SyncFile(name="a.jsonl", local_size=100),
        ]
        result = syncer._verify_staging("staging/prefix", report)
        assert result is False


# ---------------------------------------------------------------------------
# S3AtomicSync._promote
# ---------------------------------------------------------------------------

class TestPromote:
    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = FakeCompletedProcess(0, "", "")
        syncer = make_syncer()
        report = make_report()
        report.files_verified = [
            SyncFile(name="a.jsonl", local_size=100, remote_size=100, verified=True),
            SyncFile(name="b.jsonl", local_size=200, remote_size=200, verified=True),
        ]
        result = syncer._promote("staging", "final", report)
        assert result is True
        assert report.files_promoted == 2

    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = FakeCompletedProcess(1, "", "error")
        syncer = make_syncer()
        report = make_report()
        report.files_verified = [
            SyncFile(name="a.jsonl", local_size=100, verified=True),
        ]
        result = syncer._promote("staging", "final", report)
        assert result is False
        assert any("Promote failed" in e for e in report.errors)


# ---------------------------------------------------------------------------
# S3AtomicSync._purge_stale_files
# ---------------------------------------------------------------------------

class TestPurgeStale:
    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_removes_stale(self, mock_run: MagicMock) -> None:
        call_count = [0]

        def side_effect(cmd, **_kwargs):
            call_count[0] += 1
            if cmd[0] == "rclone" and cmd[1] == "lsf":
                return FakeCompletedProcess(
                    0, "100 new.jsonl\n100 old.jsonl\n", "",
                )
            if cmd[0] == "rclone" and cmd[1] == "deletefile":
                return FakeCompletedProcess(0, "", "")
            return FakeCompletedProcess(0, "", "")

        mock_run.side_effect = side_effect
        syncer = make_syncer()
        report = make_report()
        local_files = [Path("new.jsonl")]
        syncer._purge_stale_files("final", local_files, report)
        assert "old.jsonl" in report.stale_files_removed

    @patch("dataset_pipeline.scripts.s3_atomic_sync.subprocess.run")
    def test_keeps_manifest(self, mock_run: MagicMock) -> None:
        mock_run.return_value = FakeCompletedProcess(
            0, "100 new.jsonl\n100 _sync_manifest.json\n", "",
        )
        syncer = make_syncer()
        report = make_report()
        local_files = [Path("new.jsonl")]
        syncer._purge_stale_files("final", local_files, report)
        assert report.stale_files_removed == []


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------

class TestCli:
    def test_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc_info, patch("sys.argv", [
            "s3_atomic_sync",
            "--local_dir", str(tmp_path / "nope"),
            "--s3_prefix", "ai/v7",
        ]):
            main()
        assert exc_info.value.code == 1
