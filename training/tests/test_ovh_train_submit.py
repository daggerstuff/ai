"""Tests for the OVH training submitter's job environment.

The trainer reads its release from the environment, so the launcher's env list is
the only thing tying a submission to a corpus. Two failures are worth pinning:
an override that never reaches the container (the run silently trains the wrong
recipe) and an empty value (ovhai rejects the whole job spec).
"""

from __future__ import annotations

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
    assert env["ROOT"] == "/tmp/work"


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
