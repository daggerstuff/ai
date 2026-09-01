"""Tests for DVC versioned dataset loading (data_versioning.py)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from training.data_versioning import (
    _md5_file,
    _read_dvc_md5,
    list_available_versions,
    load_versioned,
)

# Expected number of records in the mock train split.
EXPECTED_TRAIN_RECORDS = 2


class TestMd5File:
    def test_known_content(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")
        # md5("hello world") = 5eb63bbbe01eeed093cb22bb8f5acdc3
        assert _md5_file(f) == "5eb63bbbe01eeed093cb22bb8f5acdc3"

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        # md5("") = d41d8cd98f00b204e9800998ecf8427e
        assert _md5_file(f) == "d41d8cd98f00b204e9800998ecf8427e"

    def test_large_file(self, tmp_path: Path) -> None:
        f = tmp_path / "large.txt"
        content = b"x" * 100_000
        f.write_bytes(content)
        expected = hashlib.md5(content).hexdigest()
        assert _md5_file(f) == expected


class TestReadDvcMd5:
    def test_valid_dvc_file(self, tmp_path: Path) -> None:
        dvc = tmp_path / "train.jsonl.dvc"
        dvc.write_text(json.dumps({"outs": [{"md5": "abc123", "path": "train.jsonl", "size": 100}]}))
        assert _read_dvc_md5(dvc) == "abc123"

    def test_missing_file(self, tmp_path: Path) -> None:
        assert _read_dvc_md5(tmp_path / "nonexistent.dvc") is None

    def test_no_outs(self, tmp_path: Path) -> None:
        dvc = tmp_path / "empty.dvc"
        dvc.write_text(json.dumps({"outs": []}))
        assert _read_dvc_md5(dvc) is None

    def test_invalid_json(self, tmp_path: Path) -> None:
        dvc = tmp_path / "bad.dvc"
        dvc.write_text("not json {{{")
        assert _read_dvc_md5(dvc) is None

    def test_missing_md5_key(self, tmp_path: Path) -> None:
        dvc = tmp_path / "no_md5.dvc"
        dvc.write_text(json.dumps({"outs": [{"path": "train.jsonl", "size": 100}]}))
        assert _read_dvc_md5(dvc) is None


class TestLoadVersioned:
    @pytest.fixture
    def mock_repo(self, tmp_path: Path) -> Path:
        """Create a mock repo structure with .dvc pointers and data files."""
        curated = tmp_path / "data" / "curated" / "sft_chatml"
        curated.mkdir(parents=True)
        # Create sample data files.
        train_data = [
            {"messages": [{"role": "user", "content": "hi"}]},
            {"messages": [{"role": "user", "content": "bye"}]},
        ]
        val_data = [{"messages": [{"role": "user", "content": "val"}]}]
        test_data = [{"messages": [{"role": "user", "content": "test"}]}]
        for split, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            data_path = curated / f"{split}.jsonl"
            with open(data_path, "w", encoding="utf-8") as f:
                for rec in data:
                    f.write(json.dumps(rec) + "\n")
            # Create .dvc pointer with matching md5.
            md5 = _md5_file(data_path)
            dvc_path = curated / f"{split}.jsonl.dvc"
            dvc_path.write_text(
                json.dumps(
                    {
                        "outs": [
                            {
                                "md5": md5,
                                "path": f"{split}.jsonl",
                                "size": data_path.stat().st_size,
                            }
                        ]
                    }
                )
            )
        return tmp_path

    def test_load_train(self, mocker: MockerFixture, mock_repo: Path) -> None:
        mocker.patch("subprocess.run")
        records = load_versioned("train", version="dataset-v1.0.0", repo_root=mock_repo)
        assert len(records) == EXPECTED_TRAIN_RECORDS
        assert records[0]["messages"][0]["content"] == "hi"
        assert records[1]["messages"][0]["content"] == "bye"

    def test_load_val(self, mocker: MockerFixture, mock_repo: Path) -> None:
        mocker.patch("subprocess.run")
        records = load_versioned("val", version="dataset-v1.0.0", repo_root=mock_repo)
        assert len(records) == 1
        assert records[0]["messages"][0]["content"] == "val"

    def test_load_test(self, mocker: MockerFixture, mock_repo: Path) -> None:
        mocker.patch("subprocess.run")
        records = load_versioned("test", version="dataset-v1.0.0", repo_root=mock_repo)
        assert len(records) == 1
        assert records[0]["messages"][0]["content"] == "test"

    def test_hash_verification_passes(self, mocker: MockerFixture, mock_repo: Path) -> None:
        """load_versioned should succeed when hashes match."""
        mocker.patch("subprocess.run")
        records = load_versioned("train", repo_root=mock_repo)
        assert len(records) == EXPECTED_TRAIN_RECORDS

    def test_hash_verification_fails(self, mocker: MockerFixture, mock_repo: Path) -> None:
        """load_versioned should raise RuntimeError on hash mismatch."""
        mocker.patch("subprocess.run")
        # Corrupt the .dvc pointer with wrong md5.
        dvc_path = mock_repo / "data" / "curated" / "sft_chatml" / "train.jsonl.dvc"
        meta = json.loads(dvc_path.read_text())
        meta["outs"][0]["md5"] = "deadbeef00000000000000000000000"
        dvc_path.write_text(json.dumps(meta))
        with pytest.raises(RuntimeError, match="Hash mismatch"):
            load_versioned("train", repo_root=mock_repo)

    def test_missing_data_file_after_pull(self, mocker: MockerFixture, mock_repo: Path) -> None:
        """load_versioned should raise FileNotFoundError if data is missing."""
        mocker.patch("subprocess.run")
        data_path = mock_repo / "data" / "curated" / "sft_chatml" / "train.jsonl"
        data_path.unlink()
        with pytest.raises(FileNotFoundError):
            load_versioned("train", repo_root=mock_repo)

    def test_git_checkout_called_with_correct_args(self, mocker: MockerFixture, mock_repo: Path) -> None:
        mock_run = mocker.patch("subprocess.run")
        load_versioned("val", version="dataset-v2.0.0", repo_root=mock_repo)
        calls = mock_run.call_args_list
        git_call = calls[0]
        assert "git" in git_call.args[0]
        assert "checkout" in git_call.args[0]
        assert "dataset-v2.0.0" in git_call.args[0]
        assert "data/curated/sft_chatml/val.jsonl.dvc" in git_call.args[0]

    def test_dvc_pull_called(self, mocker: MockerFixture, mock_repo: Path) -> None:
        mock_run = mocker.patch("subprocess.run")
        load_versioned("test", repo_root=mock_repo)
        calls = mock_run.call_args_list
        dvc_call = calls[1]
        assert "dvc" in dvc_call.args[0]
        assert "pull" in dvc_call.args[0]

    def test_empty_jsonl(self, mocker: MockerFixture, mock_repo: Path) -> None:
        """An empty JSONL file should return an empty list."""
        mocker.patch("subprocess.run")
        data_path = mock_repo / "data" / "curated" / "sft_chatml" / "train.jsonl"
        data_path.write_text("")
        # Fix the .dvc pointer to match.
        dvc_path = mock_repo / "data" / "curated" / "sft_chatml" / "train.jsonl.dvc"
        md5 = _md5_file(data_path)
        meta = json.loads(dvc_path.read_text())
        meta["outs"][0]["md5"] = md5
        dvc_path.write_text(json.dumps(meta))
        records = load_versioned("train", repo_root=mock_repo)
        assert records == []

    def test_subprocess_error_propagates(self, mocker: MockerFixture, mock_repo: Path) -> None:
        """If git checkout fails, the error should propagate."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.side_effect = subprocess.CalledProcessError(1, ["git", "checkout"])
        with pytest.raises(subprocess.CalledProcessError):
            load_versioned("train", repo_root=mock_repo)


class TestListAvailableVersions:
    def test_lists_dataset_tags(self, mocker: MockerFixture) -> None:
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = mock_run.return_value.__class__(
            stdout="dataset-v1.0.0\ndataset-v1.1.0\ndataset-v2.0.0\n"
        )
        versions = list_available_versions()
        assert versions == ["dataset-v1.0.0", "dataset-v1.1.0", "dataset-v2.0.0"]

    def test_no_tags(self, mocker: MockerFixture) -> None:
        mocker.patch("subprocess.run")
        versions = list_available_versions()
        assert versions == []

    def test_filters_only_dataset_prefix(self, mocker: MockerFixture) -> None:
        mocker.patch("subprocess.run")
        versions = list_available_versions()
        assert all(v.startswith("dataset-v") for v in versions)
