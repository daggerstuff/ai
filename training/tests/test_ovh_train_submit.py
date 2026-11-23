"""Tests for the OVH training submitter's job environment.

The trainer reads its release from the environment, so the launcher's env list is
the only thing tying a submission to a corpus. Two failures are worth pinning:
an override that never reaches the container (the run silently trains the wrong
recipe) and an empty value (ovhai rejects the whole job spec).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from training.launch import r2_sync
from training.ovh_v23_qwen3_8b import submit


def _env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    entries = submit.build_job_env(
        plan={"download": {}, "upload": {}}, run_name="run-name", flavor="l4-1-gpu", overrides=overrides
    )
    return dict(entry.split("=", 1) for entry in entries)


def test_the_env_carries_the_run_name_and_the_flavor() -> None:
    env = _env()

    assert env["RUN_NAME"] == "run-name"
    assert env["TRAIN_FLAVOR"] == "l4-1-gpu"
    assert env["ROOT"] == str(Path(tempfile.gettempdir()) / "work")


def test_the_env_round_trips_the_signed_map() -> None:
    plan = {"download": {"code/launch.sh": "https://example.invalid/get"}, "upload": {}}

    env = submit.build_job_env(plan=plan, run_name="run-name", flavor="l40s-1-gpu")

    assert r2_sync.decode_map(dict(entry.split("=", 1) for entry in env)["R2_MAP_B64"]) == plan


def test_overrides_reach_the_container() -> None:
    env = _env({"PER_DEVICE_BATCH": "1", "GRAD_ACCUM": "8"})

    assert env["PER_DEVICE_BATCH"] == "1"
    assert env["GRAD_ACCUM"] == "8"


def test_no_override_is_emitted_when_none_are_given() -> None:
    env = _env()

    assert "PER_DEVICE_BATCH" not in env
    assert "GRAD_ACCUM" not in env


def test_an_empty_override_is_dropped_because_ovhai_rejects_it() -> None:
    env = _env({"PER_DEVICE_BATCH": ""})

    assert "PER_DEVICE_BATCH" not in env


def _stage_bundle(tmp_path: Path) -> Path:
    stage_dir = tmp_path / "stage"
    (stage_dir / "code").mkdir(parents=True)
    (stage_dir / "data").mkdir()
    for relative in (
        "code/run_ovh.py",
        "code/train.py",
        "code/launch.sh",
        "data/tinker-sft-train.jsonl",
        "data/tinker-dataset.json",
    ):
        (stage_dir / relative).write_text("x")
    (stage_dir / "data" / "stale-export.json").write_text("leftover")
    return stage_dir


def _patch_r2(monkeypatch: Any, upload: Any) -> list[tuple[Path, dict[str, Any]]]:
    uploaded: list[tuple[Path, dict[str, Any]]] = []

    def fake_upload_dir(source: Path, **kwargs: Any) -> list[str]:
        uploaded.append((source, kwargs))
        return upload(kwargs)

    monkeypatch.setattr(r2_sync, "client", object)
    monkeypatch.setattr(r2_sync, "credentials", lambda: {r2_sync.DEFAULT_BUCKET_ENV: "training"})
    monkeypatch.setattr(r2_sync, "upload_dir", fake_upload_dir)
    monkeypatch.setattr(
        r2_sync, "presign", lambda keys, **_kwargs: {key: f"https://example.invalid/{key}" for key in keys}
    )
    return uploaded


def test_only_the_signed_bundle_is_uploaded_from_a_reused_stage(tmp_path: Path, monkeypatch: Any) -> None:
    """--no-restage can reuse a directory with leftovers; they must not reach R2."""
    stage_dir = _stage_bundle(tmp_path)
    uploaded = _patch_r2(
        monkeypatch,
        lambda kwargs: [f"{kwargs['prefix']}/{relative}" for relative in kwargs["include"]],
    )

    result = submit.submit(tag="v23-test", stage_dir=stage_dir, restage=False)

    assert uploaded[0][1]["include"] == submit.staged_files(stage_dir)
    assert "data/stale-export.json" not in uploaded[0][1]["include"]
    assert result["download_objects"] == len(submit.staged_files(stage_dir))


def test_an_upload_outside_the_signed_bundle_is_refused(tmp_path: Path, monkeypatch: Any) -> None:
    stage_dir = _stage_bundle(tmp_path)
    _patch_r2(
        monkeypatch,
        lambda kwargs: [
            *(f"{kwargs['prefix']}/{relative}" for relative in kwargs["include"]),
            f"{kwargs['prefix']}/data/stale-export.json",
        ],
    )

    with pytest.raises(RuntimeError, match="outside the signed bundle"):
        submit.submit(tag="v23-test", stage_dir=stage_dir, restage=False)


def test_parse_overrides_reads_key_value_pairs() -> None:
    assert submit.parse_overrides(["PER_DEVICE_BATCH=1", "GRAD_ACCUM=8"]) == {
        "PER_DEVICE_BATCH": "1",
        "GRAD_ACCUM": "8",
    }


def test_parse_overrides_keeps_equals_signs_in_the_value() -> None:
    assert submit.parse_overrides(["RUN_NAME=a=b"]) == {"RUN_NAME": "a=b"}


def test_parse_overrides_rejects_a_bare_key() -> None:
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        submit.parse_overrides(["PER_DEVICE_BATCH"])
