"""Tests for the release paired-eval config generator."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from training.launch import prepare_release_eval as prep

V15_URI = "wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-73504a99ed05:v1"
V16_URI = "wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-562d43347133:v1"
RELEASE = prep.Release("v16", V16_URI)
COMPARE = prep.Release(prep.DEFAULT_COMPARE_LABEL, prep.DEFAULT_COMPARE_URI)
VERSUS_BASE, VERSUS_COMPARE = prep.pairs(RELEASE, COMPARE)
CAPABILITY_CAP_USD = 0.2
CRISIS_CAP_USD = 0.5
CONTRACT_CAP_USD = 0.2
LITERAL_CAP_USD = 0.3


def arm_by_id(config: dict[str, object], arm_id: str) -> dict[str, str]:
    arms = config["arms"]
    assert isinstance(arms, list)
    return next(arm for arm in arms if arm["id"] == arm_id)


def make_cases_dir(root: Path) -> Path:
    cases_dir = root / "cases"
    cases_dir.mkdir()
    for _, case_file, _ in prep.SUITES:
        (cases_dir / case_file).write_text('{"id": "case-1"}\n', encoding="utf-8")
    return cases_dir


def test_pairs_name_the_release_and_its_predecessor() -> None:
    assert VERSUS_BASE == "v16-vs-base"
    assert VERSUS_COMPARE == "v16-vs-v15"


def test_vs_base_pairs_the_base_model_against_the_adapter() -> None:
    config = prep.build_config("crisis-dev", VERSUS_BASE, RELEASE, COMPARE, CRISIS_CAP_USD)

    assert arm_by_id(config, "base")["model"] == prep.BASE_MODEL
    assert arm_by_id(config, "sft")["model"] == V16_URI
    assert config["pricing"]["hard_cap_usd"] == CRISIS_CAP_USD


def test_vs_compare_puts_the_previous_adapter_in_the_base_slot() -> None:
    config = prep.build_config("literal-dev", VERSUS_COMPARE, RELEASE, COMPARE, LITERAL_CAP_USD)

    assert arm_by_id(config, "base")["model"] == COMPARE.uri
    assert arm_by_id(config, "sft")["model"] == V16_URI
    assert config["decoding"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_run_id_is_stable_per_artifact_and_changes_with_it() -> None:
    first = prep.build_config("crisis-dev", VERSUS_COMPARE, RELEASE, COMPARE, CRISIS_CAP_USD)["run_id"]
    repeat = prep.build_config("crisis-dev", VERSUS_COMPARE, RELEASE, COMPARE, CRISIS_CAP_USD)["run_id"]
    other_release = prep.Release(RELEASE.label, V16_URI.replace(":v1", ":v2"))
    other = prep.build_config("crisis-dev", VERSUS_COMPARE, other_release, COMPARE, CRISIS_CAP_USD)["run_id"]

    assert first == repeat
    assert first != other
    assert first.startswith(f"wandb-crisis-dev-{VERSUS_COMPARE}-")


def test_run_id_differs_between_comparison_pairs() -> None:
    versus_base = prep.build_config("crisis-dev", VERSUS_BASE, RELEASE, COMPARE, CRISIS_CAP_USD)["run_id"]
    versus_compare = prep.build_config("crisis-dev", VERSUS_COMPARE, RELEASE, COMPARE, CRISIS_CAP_USD)["run_id"]

    assert versus_base != versus_compare


def test_unknown_pair_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown comparison pair"):
        prep.build_config("crisis-dev", "v16-vs-v14", RELEASE, COMPARE, CRISIS_CAP_USD)


def test_write_configs_covers_every_suite_and_pair(tmp_path: Path) -> None:
    cases_dir = make_cases_dir(tmp_path)
    out_dir = tmp_path / "out"

    written = prep.write_configs(cases_dir, out_dir, RELEASE, COMPARE)

    assert len(written) == len(prep.SUITES) * len(prep.pairs(RELEASE, COMPARE))
    sealed = {label for label, _, _, is_sealed in written if is_sealed}
    assert sealed == {
        f"{suite}-{pair}"
        for suite, _, _ in prep.SUITES
        if suite.endswith("-sealed")
        for pair in prep.pairs(RELEASE, COMPARE)
    }
    config = json.loads((out_dir / f"wandb-crisis-dev-{VERSUS_BASE}.config.json").read_text(encoding="utf-8"))
    assert [arm["id"] for arm in config["arms"]] == ["base", "sft"]
    assert config["schema_version"] == 1


def test_write_configs_refuses_a_missing_case_file(tmp_path: Path) -> None:
    cases_dir = make_cases_dir(tmp_path)
    (cases_dir / "scenario-crisis-dev-v1.jsonl").unlink()

    with pytest.raises(FileNotFoundError, match=re.escape("scenario-crisis-dev-v1.jsonl")):
        prep.write_configs(cases_dir, tmp_path / "out", RELEASE, COMPARE)


def test_render_plan_marks_sealed_suites_and_uses_the_results_root(tmp_path: Path) -> None:
    cases_dir = make_cases_dir(tmp_path)
    written = prep.write_configs(cases_dir, tmp_path / "out", RELEASE, COMPARE)

    plan = prep.render_plan(RELEASE, written, tmp_path / "results")

    assert f"{RELEASE.label} artifact: {V16_URI}" in plan
    assert f"--results-dir {tmp_path / 'results' / f'crisis-dev-{VERSUS_BASE}'}" in plan
    assert f"--results-dir {tmp_path / 'results' / f'crisis-sealed-{VERSUS_BASE}'}" in plan
    assert f"--max-input-tokens {prep.MAX_INPUT_TOKENS} --max-tool-rounds {prep.MAX_TOOL_ROUNDS} --allow-sealed" in plan
    assert plan.count("--allow-sealed") == len([row for row in written if row[3]])


def test_render_plan_carries_the_preflight_flags_the_cap_depends_on(tmp_path: Path) -> None:
    cases_dir = make_cases_dir(tmp_path)
    written = prep.write_configs(cases_dir, tmp_path / "out", RELEASE, COMPARE)

    plan = prep.render_plan(RELEASE, written, tmp_path / "results")

    for _, _, config, _ in written:
        assert f"--config {config} --results-dir" in plan
    preflight = f"--max-input-tokens {prep.MAX_INPUT_TOKENS} --max-tool-rounds {prep.MAX_TOOL_ROUNDS}"
    assert plan.count(preflight) == len(written)


def test_estimate_matches_the_runner_formula() -> None:
    config = prep.build_config("capability-dev", VERSUS_BASE, RELEASE, COMPARE, CAPABILITY_CAP_USD)

    estimate = prep.estimate_max_cost_usd(config, 5)

    # 5 cases * 2 arms * 5 calls * (4096 * $0.25/M + 2048 * $1.25/M)
    assert estimate == pytest.approx(0.1792)


def test_write_configs_rejects_a_suite_that_outgrows_its_cap(tmp_path: Path) -> None:
    cases_dir = make_cases_dir(tmp_path)
    (cases_dir / "scenario-contract-dev-v1.jsonl").write_text('{"id": "case"}\n' * 6, encoding="utf-8")

    with pytest.raises(ValueError, match=f"contract-dev-{VERSUS_BASE} worst case"):
        prep.write_configs(cases_dir, tmp_path / "out", RELEASE, COMPARE)


class _FakeArtifact:
    def __init__(self, name: str, version: str, artifact_type: str = "lora") -> None:
        self.name = name
        self.version = version
        self.type = artifact_type


class _FakeRun:
    def __init__(self, state: str, artifacts: list[_FakeArtifact]) -> None:
        self.state = state
        self._artifacts = artifacts

    def logged_artifacts(self) -> list[_FakeArtifact]:
        return self._artifacts


def fake_wandb(monkeypatch: pytest.MonkeyPatch, run: _FakeRun) -> None:
    class _Api:
        def run(self, _ref: str) -> _FakeRun:
            return run

    monkeypatch.setattr(prep, "wandb", type("W", (), {"Api": _Api}))


def test_resolve_picks_the_newest_lora_and_keeps_one_version_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = "arc-sft-baseline-73504a99ed05"
    run = _FakeRun(
        "finished",
        [
            _FakeArtifact(f"{collection}:v0", "v0"),
            _FakeArtifact(f"{collection}:v1", "v1"),
            _FakeArtifact(f"{collection}-train-state:v0", "v0", artifact_type="train-state"),
        ],
    )
    fake_wandb(monkeypatch, run)

    assert prep.resolve_release_uri(collection) == f"wandb-artifact:///{prep.ENTITY}/{prep.PROJECT}/{collection}:v1"


def test_resolve_refuses_an_unfinished_run(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_wandb(monkeypatch, _FakeRun("running", []))

    with pytest.raises(RuntimeError, match="only final once it finishes"):
        prep.resolve_release_uri("arc-sft-baseline-73504a99ed05")


def test_resolve_refuses_a_run_with_no_lora_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_wandb(monkeypatch, _FakeRun("finished", [_FakeArtifact("state:v0", "v0", "train-state")]))

    with pytest.raises(RuntimeError, match="logged no lora artifact"):
        prep.resolve_release_uri("arc-sft-baseline-73504a99ed05")


def test_default_compare_is_the_previous_release() -> None:
    assert prep.DEFAULT_COMPARE_LABEL == "v15"
    assert prep.DEFAULT_COMPARE_URI == V15_URI
    assert RELEASE.label != prep.DEFAULT_COMPARE_LABEL


def test_main_refuses_to_compare_a_release_against_itself(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases_dir = make_cases_dir(tmp_path)
    argv = [
        "prepare_release_eval",
        "--release-label",
        "v16",
        "--release-uri",
        V16_URI,
        "--compare-label",
        "v16",
        "--cases-dir",
        str(cases_dir),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit):
        prep.main()

    assert "must differ from --release-label" in capsys.readouterr().err
