"""Tests for training.scripts.s3_atomic_sync.

Covers atomic_swap happy/failure paths, sync_directory, CLI validation.
S3 client is mocked — no real AWS calls.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from training.scripts.s3_atomic_sync import (
    CHUNK_SIZE,
    SyncResult,
    _sha256_file,
    _staging_key,
    atomic_swap,
    build_parser,
    main,
    sync_directory,
)

# --- constants --------------------------------------------------------------

ONE_RECORD = 1
ZERO_BYTES = 0
DUMMY_SHA = "a" * 64
SMALL_CONTENT = '{"messages": [{"role": "user", "content": "hi"}]}'

# --- helpers ----------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _mock_client() -> MagicMock:
    """S3 client mock that returns matching SHA-256 metadata from head_object."""
    c = MagicMock()
    # head_object returns the metadata stored by upload_file ExtraArgs
    store: dict[str, dict] = {}

    def _upload(local, bucket, key, ExtraArgs=None, **kw):
        meta = (ExtraArgs or {}).get("Metadata", {})
        store[key] = {"Metadata": meta}

    def _head(Bucket, Key):
        return store.get(Key, {"Metadata": {}})

    c.upload_file.side_effect = _upload
    c.head_object.side_effect = _head
    c._store = store
    return c


# --- SyncResult -------------------------------------------------------------


class TestSyncResult:
    def test_defaults(self):
        r = SyncResult(local_path="a", s3_key="b", success=True)
        assert r.etag is None
        assert r.sha256 is None
        assert r.size_bytes == ZERO_BYTES
        assert r.error is None
        assert r.dry_run is False

    def test_failure_fields(self):
        r = SyncResult(local_path="a", s3_key="b", success=False, error="boom")
        assert not r.success
        assert r.error == "boom"


# --- _sha256_file -----------------------------------------------------------


class TestSha256File:
    def test_empty(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_bytes(b"")
        assert _sha256_file(f) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_known(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_bytes(b"hello")
        assert _sha256_file(f) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_large_chunk(self, tmp_path):
        f = tmp_path / "big.bin"
        f.write_bytes(b"x" * (CHUNK_SIZE + 100))
        h = _sha256_file(f)
        assert len(h) == 64  # hex digest


# --- _staging_key ----------------------------------------------------------


class TestStagingKey:
    def test_format(self):
        k = _staging_key("datasets/v7/V7_MASTER.jsonl")
        assert k.startswith("datasets/v7/V7_MASTER.jsonl.tmp.")
        assert str(os.getpid()) in k

    def test_unique_per_pid(self):
        # same key, same pid should be deterministic within process
        assert _staging_key("a/b.json") == _staging_key("a/b.json")


# --- atomic_swap happy path -------------------------------------------------


class TestAtomicSwapHappy:
    def test_dry_run(self, tmp_path):
        f = tmp_path / "V7_MASTER.jsonl"
        _write_jsonl(f, [{"messages": [{"role": "user", "content": "hi"}]}])
        client = _mock_client()
        r = atomic_swap(client, "pixeldata", f, "ds/v7/V7_MASTER.jsonl", dry_run=True)
        assert r.success
        assert r.dry_run
        assert r.size_bytes > ZERO_BYTES
        assert len(r.sha256) == 64
        client.upload_file.assert_not_called()
        client.copy_object.assert_not_called()
        client.delete_object.assert_not_called()

    def test_real_upload(self, tmp_path):
        f = tmp_path / "V7_MASTER.jsonl"
        _write_jsonl(f, [{"messages": [{"role": "user", "content": "hi"}]}])
        client = _mock_client()
        r = atomic_swap(client, "pixeldata", f, "ds/v7/V7_MASTER.jsonl")
        assert r.success
        # upload to staging key
        assert client.upload_file.call_count == 1
        staging = client.upload_file.call_args[0][2]
        assert ".tmp." in staging
        # copy staging -> final
        assert client.copy_object.call_count == 1
        cp = client.copy_object.call_args.kwargs
        assert cp["Key"] == "ds/v7/V7_MASTER.jsonl"
        assert cp["CopySource"]["Key"] == staging
        # delete staging
        assert client.delete_object.call_count == 1
        assert client.delete_object.call_args.kwargs["Key"] == staging

    def test_metadata_carries_sha256(self, tmp_path):
        f = tmp_path / "V7_MASTER.jsonl"
        _write_jsonl(f, [{"messages": [{"role": "user", "content": "hi"}]}])
        client = _mock_client()
        r = atomic_swap(client, "pixeldata", f, "ds/v7/V7_MASTER.jsonl")
        assert r.success
        upload_extra = client.upload_file.call_args.kwargs["ExtraArgs"]
        assert "sha256-checksum" in upload_extra["Metadata"]
        copy_meta = client.copy_object.call_args.kwargs["Metadata"]
        assert "sha256-checksum" in copy_meta


# --- atomic_swap failure paths ---------------------------------------------


class TestAtomicSwapFailures:
    def test_missing_file(self, tmp_path):
        f = tmp_path / "nope.jsonl"
        client = _mock_client()
        r = atomic_swap(client, "pixeldata", f, "ds/v7/V7_MASTER.jsonl")
        assert not r.success
        assert "does not exist" in (r.error or "")
        client.upload_file.assert_not_called()

    def test_upload_failure(self, tmp_path):
        f = tmp_path / "V7_MASTER.jsonl"
        _write_jsonl(f, [{"messages": [{"role": "user", "content": "hi"}]}])
        client = _mock_client()
        client.upload_file.side_effect = RuntimeError("network down")
        r = atomic_swap(client, "pixeldata", f, "ds/v7/V7_MASTER.jsonl")
        assert not r.success
        assert "staging upload failed" in (r.error or "")
        # no cleanup since upload failed
        client.copy_object.assert_not_called()
        client.delete_object.assert_not_called()

    def test_checksum_mismatch(self, tmp_path):
        f = tmp_path / "V7_MASTER.jsonl"
        _write_jsonl(f, [{"messages": [{"role": "user", "content": "hi"}]}])
        client = _mock_client()
        # override head_object to return wrong sha
        client.head_object.side_effect = lambda Bucket, Key: {"Metadata": {"sha256-checksum": "wrong"}}
        r = atomic_swap(client, "pixeldata", f, "ds/v7/V7_MASTER.jsonl")
        assert not r.success
        assert "checksum mismatch" in (r.error or "")
        # staging cleanup attempted
        client.delete_object.assert_called_once()

    def test_copy_failure(self, tmp_path):
        f = tmp_path / "V7_MASTER.jsonl"
        _write_jsonl(f, [{"messages": [{"role": "user", "content": "hi"}]}])
        client = _mock_client()
        client.copy_object.side_effect = RuntimeError("permission denied")
        r = atomic_swap(client, "pixeldata", f, "ds/v7/V7_MASTER.jsonl")
        assert not r.success
        assert "copy_object failed" in (r.error or "")
        # staging cleanup attempted
        client.delete_object.assert_called_once()


# --- sync_directory ---------------------------------------------------------


class TestSyncDirectory:
    def test_walks_dir(self, tmp_path):
        d = tmp_path / "v7"
        d.mkdir()
        _write_jsonl(d / "V7_MASTER.jsonl", [{"messages": [{"role": "user", "content": "a"}]}])
        _write_jsonl(d / "dedup_report.json", {"schema": "v7"})
        (d / "README.txt").write_text("ignored")
        client = _mock_client()
        results = sync_directory(client, "pixeldata", d, "ds/v7/")
        assert len(results) == 2
        keys = sorted(r.s3_key for r in results)
        assert keys == ["ds/v7/V7_MASTER.jsonl", "ds/v7/dedup_report.json"]
        assert all(r.success for r in results)

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        client = _mock_client()
        results = sync_directory(client, "pixeldata", d, "ds/v7/")
        assert results == []

    def test_extension_filter(self, tmp_path):
        d = tmp_path / "v7"
        d.mkdir()
        (d / "V7_MASTER.jsonl").write_text("{}\n")
        (d / "notes.md").write_text("ignored")
        (d / "data.csv").write_text("ignored")
        client = _mock_client()
        results = sync_directory(client, "pixeldata", d, "ds/v7/")
        assert len(results) == 1
        assert results[0].s3_key.endswith("V7_MASTER.jsonl")


# --- CLI --------------------------------------------------------------------


class TestCLI:
    def test_build_parser_defaults(self, monkeypatch):
        monkeypatch.delenv("S3_BUCKET", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        p = build_parser()
        ns = p.parse_args(["--input", "x.jsonl", "--s3_key", "y"])
        assert ns.bucket == "pixeldata"
        assert ns.region == "US-EAST-VA"
        assert ns.dry_run is False

    def test_main_missing_s3_key_for_file(self, tmp_path):
        f = tmp_path / "x.jsonl"
        f.write_text("{}\n")
        rc = main(["--input", str(f)])
        assert rc == 1

    def test_main_missing_s3_prefix_for_dir(self, tmp_path):
        d = tmp_path / "v7"
        d.mkdir()
        rc = main(["--input", str(d)])
        assert rc == 1

    def test_main_dry_run_file(self, tmp_path):
        f = tmp_path / "V7_MASTER.jsonl"
        _write_jsonl(f, [{"messages": [{"role": "user", "content": "hi"}]}])
        rc = main(
            [
                "--input",
                str(f),
                "--s3_key",
                "ds/v7/V7_MASTER.jsonl",
                "--bucket",
                "pixeldata",
                "--dry_run",
            ]
        )
        assert rc == 0

    def test_main_dry_run_dir(self, tmp_path):
        d = tmp_path / "v7"
        d.mkdir()
        _write_jsonl(d / "V7_MASTER.jsonl", [{"messages": [{"role": "user", "content": "hi"}]}])
        _write_jsonl(d / "dedup_report.json", {"ok": True})
        rc = main(
            [
                "--input",
                str(d),
                "--s3_prefix",
                "ds/v7/",
                "--bucket",
                "pixeldata",
                "--dry_run",
            ]
        )
        assert rc == 0


# --- subprocess smoke -------------------------------------------------------


class TestSubprocess:
    def test_dry_run_exit_zero(self, tmp_path):
        f = tmp_path / "V7_MASTER.jsonl"
        _write_jsonl(f, [{"messages": [{"role": "user", "content": "hi"}]}])
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "training.scripts.s3_atomic_sync",
                "--input",
                str(f),
                "--s3_key",
                "ds/v7/V7_MASTER.jsonl",
                "--dry_run",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert r.returncode == 0

    def test_missing_file_exit_one(self):
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "training.scripts.s3_atomic_sync",
                "--input",
                "/nonexistent/file.jsonl",
                "--s3_key",
                "ds/v7/x.jsonl",
                "--dry_run",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert r.returncode == 1
