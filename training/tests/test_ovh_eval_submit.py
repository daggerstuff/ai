"""Tests for the OVH eval submitter that carries everything over presigned URLs.

The volume mount is gone, so the two things that can silently break a run are the
signed map (if it does not cover every result file, the run is lost at teardown)
and the in-container bootstrap (if it does not actually pull its inputs and hand
control to launch.sh, the job fails before it scores anything).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

from training.launch import r2_sync
from training.ovh_v22_eval import submit


def test_run_names_covers_every_adapter_and_split() -> None:
    names = submit.run_names()

    assert names == ["eval-bf16-dev", "eval-bf16-sealed", "eval-nf4-dev", "eval-nf4-sealed"]


def test_run_names_applies_the_output_prefix() -> None:
    names = submit.run_names(labels=("bf16",), splits=("dev",), out_prefix="retry-")

    assert names == ["retry-eval-bf16-dev"]


def test_result_keys_sign_every_file_of_every_run() -> None:
    names = submit.run_names()

    keys = submit.result_keys("v23", names=names)

    assert len(keys) == len(names) * len(submit.RESULT_FILES)
    assert keys["eval-bf16-dev/results.jsonl"] == "evals/v23/eval-bf16-dev/results.jsonl"


def test_job_env_round_trips_the_signed_map() -> None:
    plan = {"tag": "v23", "bucket": "training", "download": {"code/launch.sh": "https://example.test/a"}}

    env = dict(entry.split("=", 1) for entry in submit.build_job_env(tag="v23", plan=plan, out_prefix="", flavor="l4"))
    decoded = r2_sync.decode_map(env["R2_MAP_B64"])

    assert decoded == plan
    assert env["ROOT"] == "/tmp/work"
    assert env["EVAL_FLAVOR"] == "l4"


def test_command_mounts_no_volume_and_runs_the_bootstrap() -> None:
    argv = submit.build_command(image="img:1", name="job", flavor="l4-1-gpu", gpu=1, timeout="2h", env=["A=1", "B=2"])

    assert "--volume" not in argv
    assert [argv[index + 1] for index, part in enumerate(argv) if part == "--env"] == ["A=1", "B=2"]
    assert argv[-4:] == ["--", "bash", "-lc", 'python3 -c "$BOOTSTRAP"']


def test_bootstrap_fetches_inputs_and_hands_off_to_the_launcher(tmp_path: Path) -> None:
    source = tmp_path / "source" / "launch.sh"
    source.parent.mkdir(parents=True)
    launcher_exit = 7
    source.write_text(f'#!/bin/sh\necho "launcher ran"\nexit {launcher_exit}\n')

    plan = {"download": {"code/launch.sh": source.as_uri()}}
    root = tmp_path / "work"
    env = {
        **os.environ,
        "R2_MAP_B64": r2_sync.encode_map(plan),
        "ROOT": str(root),
    }

    completed = subprocess.run(
        [sys.executable, "-c", submit.BOOTSTRAP], env=env, capture_output=True, text=True, check=False
    )

    assert "fetched code/launch.sh" in completed.stdout
    assert completed.stdout.strip().endswith("launcher ran")
    # The bootstrap must pass the launcher's exit code through, not swallow it.
    assert completed.returncode == launcher_exit
    assert (root / "code" / "launch.sh").read_text() == source.read_text()


class FakeS3:
    """Signs URLs offline and reports the objects the test says are stored."""

    def __init__(self, keys: tuple[str, ...] = ()) -> None:
        self.keys = keys

    def generate_presigned_url(self, operation: str, Params: dict[str, str], ExpiresIn: int) -> str:
        return f"https://r2.test/{operation}/{Params['Key']}?expires={ExpiresIn}"

    def list_objects_v2(self, **request: str) -> dict[str, object]:
        prefix = request.get("Prefix", "")
        contents = [{"Key": key, "Size": 1} for key in self.keys if key.startswith(prefix)]
        return {"Contents": contents, "IsTruncated": False}


def test_submit_signs_every_input_and_every_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage_dir = tmp_path / "stage"
    (stage_dir / "code").mkdir(parents=True)
    (stage_dir / "code" / "launch.sh").write_text("#!/bin/sh\n")
    (stage_dir / "data").mkdir()
    (stage_dir / "data" / "scenario-crisis-dev-v1.jsonl").write_text("{}\n")

    monkeypatch.setattr(
        submit.r2_sync,
        "client",
        lambda **_: FakeS3(("inputs/v22/code/launch.sh", "inputs/v22/data/scenario-crisis-dev-v1.jsonl")),
    )
    monkeypatch.setattr(submit.r2_sync, "credentials", lambda **_: {"R2_BUCKET": "training"})
    monkeypatch.setattr(submit.r2_sync, "upload_dir", lambda *_, **__: ["code/launch.sh"])

    report = submit.submit(tag="v23", stage_dir=stage_dir, restage=False)

    env = dict(
        entry.split("=", 1)
        for entry in report["command"]
        if isinstance(entry, str) and "=" in entry and not entry.startswith("--")
    )
    plan = r2_sync.decode_map(env["R2_MAP_B64"])

    # One GET per staged input, one PUT per result file of every run.
    staged_files = [path for path in stage_dir.rglob("*") if path.is_file()]
    assert report["download_objects"] == len(staged_files)
    assert report["upload_objects"] == len(submit.run_names()) * len(submit.RESULT_FILES)
    assert report["executed"] is False
    assert set(plan["download"]) == {"code/launch.sh", "data/scenario-crisis-dev-v1.jsonl"}
    assert plan["download"]["code/launch.sh"].startswith("https://r2.test/get_object/inputs/v22/")
    assert plan["upload"]["eval-nf4-sealed/results.jsonl"].startswith("https://r2.test/put_object/evals/v23/")
    assert plan["bucket"] == "training"


def test_submit_refuses_to_launch_when_an_input_did_not_land(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A presigned GET signs a key whether or not the object exists.

    v28's eval job spent its whole life on a 404 because the upload was one file
    short and the signature did not care. The submitter now lists the prefix
    back and stops before the job is created.
    """
    stage_dir = tmp_path / "stage"
    (stage_dir / "code").mkdir(parents=True)
    (stage_dir / "code" / "launch.sh").write_text("#!/bin/sh\n")
    (stage_dir / "adapters" / "sft").mkdir(parents=True)
    (stage_dir / "adapters" / "sft" / "adapter_config.json").write_text("{}\n")

    monkeypatch.setattr(submit.r2_sync, "client", lambda **_: FakeS3(("inputs/v22/code/launch.sh",)))
    monkeypatch.setattr(submit.r2_sync, "credentials", lambda **_: {"R2_BUCKET": "training"})
    monkeypatch.setattr(submit.r2_sync, "upload_dir", lambda *_, **__: ["code/launch.sh"])

    with pytest.raises(SystemExit, match=r"adapters/sft/adapter_config\.json"):
        submit.submit(tag="v28", stage_dir=stage_dir, restage=False)


def _fake_repo(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """A miniature ai/ tree plus adapters, so stage() runs without 700 MB copies."""
    ai_root = tmp_path / "ai"
    for relative in submit.CODE_FILES:
        target = ai_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# {relative}\n")

    cases = tmp_path / "cases"
    cases.mkdir()
    for split in submit.DEFAULT_SPLITS:
        for suite in ("v1", "v2"):
            (cases / f"scenario-crisis-{split}-{suite}.jsonl").write_text("{}\n")

    adapters = {}
    for label in submit.DEFAULT_LABELS:
        adapters[label] = tmp_path / "adapters" / label
        adapters[label].mkdir(parents=True)
        (adapters[label] / "adapter_config.json").write_text("{}\n")
    return cases, adapters


def test_stage_lays_out_exactly_what_the_launcher_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cases, adapters = _fake_repo(tmp_path)
    monkeypatch.setattr(submit, "AI_ROOT", tmp_path / "ai")
    monkeypatch.setattr(submit, "CASES_DIR", cases)
    monkeypatch.setattr(submit, "DEFAULT_ADAPTERS", adapters)

    copied = submit.stage(tmp_path / "stage")

    staged = tmp_path / "stage"
    assert (staged / "code" / "launch.sh").is_file()
    assert (staged / "data" / "scenario-crisis-sealed-v2.jsonl").is_file()
    assert (staged / "adapters" / "nf4" / "adapter_config.json").is_file()
    assert len(copied) == len(submit.CODE_FILES) + 1 + len(submit.DEFAULT_SPLITS) + len(adapters)


def test_stage_refuses_to_run_without_an_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cases, _adapters = _fake_repo(tmp_path)
    monkeypatch.setattr(submit, "AI_ROOT", tmp_path / "ai")
    monkeypatch.setattr(submit, "CASES_DIR", cases)
    monkeypatch.setattr(submit, "DEFAULT_ADAPTERS", {"bf16": tmp_path / "missing"})

    with pytest.raises(SystemExit, match="adapter for bf16 is missing"):
        submit.stage(tmp_path / "stage")


def test_stage_copies_only_the_adapters_it_is_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cases, adapters = _fake_repo(tmp_path)
    monkeypatch.setattr(submit, "AI_ROOT", tmp_path / "ai")
    monkeypatch.setattr(submit, "CASES_DIR", cases)

    submit.stage(tmp_path / "stage", adapters={"v23": adapters["bf16"]})

    assert (tmp_path / "stage" / "adapters" / "v23" / "adapter_config.json").is_file()
    assert not (tmp_path / "stage" / "adapters" / "nf4").exists()


def _run_args(labels: str, adapters: list[str]) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=labels)
    parser.add_argument("--adapter", action="append", default=adapters)
    return parser, parser.parse_args([])


def test_selected_adapters_stages_only_the_labels_being_scored() -> None:
    parser, args = _run_args("nf4", [])

    selected = submit._selected_adapters(parser, args)

    assert selected == {"nf4": submit.DEFAULT_ADAPTERS["nf4"]}


def test_selected_adapters_takes_a_label_path_override(tmp_path: Path) -> None:
    adapter = tmp_path / "v23-adapter"
    parser, args = _run_args("v23", [f"v23={adapter}"])

    selected = submit._selected_adapters(parser, args)

    assert selected == {"v23": adapter}


def test_selected_adapters_rejects_a_label_with_no_adapter() -> None:
    parser, args = _run_args("v23", [])

    with pytest.raises(SystemExit):
        submit._selected_adapters(parser, args)


def test_job_env_carries_the_labels_the_launcher_will_score() -> None:
    plan = {"tag": "v23", "bucket": "training", "download": {}}

    env = dict(
        entry.split("=", 1)
        for entry in submit.build_job_env(
            tag="v23", plan=plan, out_prefix="", flavor="l4", labels=("v23",), run_label="v23"
        )
    )

    assert env["LABELS"] == "v23"
    assert env["RUN_LABEL"] == "v23"


def test_job_env_never_emits_an_empty_value() -> None:
    plan = {"tag": "v23", "bucket": "training", "download": {}}

    env = submit.build_job_env(tag="v23", plan=plan, out_prefix="", flavor="l4")

    assert all(entry.split("=", 1)[1] for entry in env)
    assert "OUT_PREFIX" not in dict(entry.split("=", 1) for entry in env)
