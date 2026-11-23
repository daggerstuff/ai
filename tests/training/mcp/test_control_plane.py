"""Unit tests for the training-infra MCP registry, plan binding, and gate.

These run without the service, the network, or real credentials: they exercise
the pure logic that keeps the control plane safe — recipe validation, the
plan/submit binding, the audit log, and the eval gate's fail-closed behaviour.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

AI_ROOT = Path(__file__).resolve().parents[3]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from training.mcp import (  # noqa: E402
    backends,
    config as mcp_config,
    db,
    foresight,
    plan,
    recipes,
    runner,
    server,
    sync,
)
from training.mcp.config import IMPORT_ACTOR, Settings  # noqa: E402
from training.mcp.server import count_toward_daily_cap  # noqa: E402

VALID_ENTRY = {
    "id": "eval-draft-v1",
    "kind": "eval_paired",
    "description": "Draft eval recipe.",
    "backend": "ovhai",
    "entrypoint": "python -m training.ovh_v22_eval.submit run --execute --tag <tag> --suite v2",
    "cost_class": "gpu-hour",
    "est_cost_usd": 2.0,
    "gpu": "l4-1-gpu",
    "timeout": "2h",
    "inputs": {"tag": "result tag", "suite": "v2"},
    "outputs": {"r2_prefix": "evals/<tag>/"},
    "invariants": {"r2_only": True},
}


# --- recipes.validate_recipe_entry (P3.3) ----------------------------------


def test_valid_draft_entry_has_no_errors():
    assert recipes.validate_recipe_entry(VALID_ENTRY) == []


def test_missing_required_fields_are_reported():
    errors = recipes.validate_recipe_entry({"id": "x", "kind": "eval_paired"})
    assert any("missing fields" in error and "backend" in error for error in errors)


def test_unknown_kind_backend_and_cost_class_are_reported():
    errors = recipes.validate_recipe_entry(
        {**VALID_ENTRY, "id": "y", "kind": "not_a_kind", "backend": "nope", "cost_class": "gpu-year"}
    )
    assert any("unknown kind" in e for e in errors)
    assert any("unknown backend" in e for e in errors)
    assert any("unknown cost_class" in e for e in errors)


def test_duplicate_id_is_reported_against_known_ids():
    errors = recipes.validate_recipe_entry({**VALID_ENTRY, "id": "eval-crisis-v2"}, known_ids={"eval-crisis-v2"})
    assert any("duplicate recipe id" in e for e in errors)


def test_override_needs_type_unless_locked():
    bad = recipes.validate_recipe_entry({**VALID_ENTRY, "id": "z", "overrides": {"epochs": {"min": 1}}})
    assert any("must declare a type unless locked" in e for e in bad)
    good = recipes.validate_recipe_entry(
        {**VALID_ENTRY, "id": "z", "overrides": {"epochs": {"type": "int", "min": 1}, "lr": {"locked": True}}}
    )
    assert good == []


def test_launching_recipe_without_execute_flag_is_rejected():
    """A launching recipe must carry --execute (P4.14).

    Both remote entrypoints default to a dry run, so a recipe missing the flag
    would submit, report success, and launch nothing.
    """
    for kind, entrypoint in (
        ("train_sft", "python -m training.launch.wandb_sft_v15_execute"),
        ("eval_paired", "python -m training.ovh_v22_eval.submit run --tag <tag>"),
    ):
        entry = {**VALID_ENTRY, "id": "x", "kind": kind, "entrypoint": entrypoint}
        errors = recipes.validate_recipe_entry(entry)
        assert any("--execute" in error for error in errors), (kind, errors)

    ok = {
        **VALID_ENTRY,
        "id": "x",
        "kind": "train_sft",
        "entrypoint": "python -m training.launch.wandb_sft_v15_execute --execute",
    }
    assert recipes.validate_recipe_entry(ok) == []

    local = {
        **VALID_ENTRY,
        "id": "x",
        "kind": "curate_dataset",
        "backend": "local",
        "entrypoint": "python -m training.build_agent_tool_sft_v30",
    }
    assert recipes.validate_recipe_entry(local) == []


def test_every_bundled_launching_recipe_has_execute():
    for recipe in recipes.load_recipes():
        if recipe.kind not in recipes.LAUNCH_KINDS:
            continue
        assert "--execute" in recipe.entrypoint, f"{recipe.id} cannot launch"


def test_multi_node_shapes_are_rejected():
    for patch in (
        {"nodes": 2},
        {"num_nodes": 4},
        {"world_size": 8},
        {"multi_node": True},
        {"gpu": "l4-4-gpu"},
        {"gpu": "multi-gpu"},
    ):
        errors = recipes.validate_recipe_entry({**VALID_ENTRY, "id": "mn", **patch})
        assert any("P3.4" in e for e in errors), patch


def test_single_gpu_flavour_still_passes():
    assert recipes.validate_recipe_entry({**VALID_ENTRY, "id": "sg", "gpu": "l4-1-gpu"}) == []
    assert recipes.validate_recipe_entry({**VALID_ENTRY, "id": "sg2", "gpu": "none"}) == []


EXPECTED_RECIPE_COUNT = 6


def test_bundled_registry_loads_and_has_no_multi_gpu():
    loaded = recipes.load_recipes()
    assert len(loaded) >= EXPECTED_RECIPE_COUNT
    assert all("multi" not in r.gpu.lower() for r in loaded)


def test_bad_registry_file_fails_fast(tmp_path):
    bad = tmp_path / "recipes.yaml"
    bad.write_text("recipes:\n  - id: broken\n    kind: train_sft\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        recipes.load_recipes(bad)


def test_duplicate_registry_id_fails_fast(tmp_path):
    bad = tmp_path / "recipes.yaml"
    # Both entries must otherwise be valid, or the loader fails on the first
    # one's unrelated problem before it ever reaches the duplicate check.
    bad.write_text(
        "recipes:\n"
        "  - id: same\n    kind: eval_paired\n    description: d\n    backend: local\n"
        "    entrypoint: python -m x --execute\n"
        "  - id: same\n    kind: eval_paired\n    description: d\n    backend: local\n"
        "    entrypoint: python -m x --execute\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate recipe id"):
        recipes.load_recipes(bad)


# --- db: registry, plan cache, audit (X.2) ---------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "jobs.db"


def test_plan_cache_roundtrip(db_path: Path):
    db.cache_plan(db_path, "ph1", "eval-crisis-v2", {"recipe_id": "eval-crisis-v2"})
    cached = db.get_cached_plan(db_path, "ph1")
    assert cached is not None
    assert cached["recipe_id"] == "eval-crisis-v2"
    assert cached["spec"]["recipe_id"] == "eval-crisis-v2"
    assert db.get_cached_plan(db_path, "missing") is None


def test_audit_log_records_hashes_and_filters(db_path: Path):
    db.write_audit(
        db_path,
        action="plan",
        actor="pipeline-agent",
        recipe_id="eval-crisis-v2",
        spec_hash="s1",
        plan_hash="p1",
        detail={"est_cost_usd": 2.0},
        remote_addr="127.0.0.1",
    )
    db.write_audit(db_path, action="submit", actor="pipeline-agent", job_id="job-1", plan_hash="p1")
    db.write_audit(db_path, action="submit.rejected", actor="pipeline-agent", detail={"reason": "no plan_hash"})

    plans = db.list_audit(db_path, action="plan")
    assert len(plans) == 1
    assert plans[0]["spec_hash"] == "s1"
    assert plans[0]["detail"] == {"est_cost_usd": 2.0}
    assert plans[0]["remote_addr"] == "127.0.0.1"

    rejects = db.list_audit(db_path, action="submit.rejected")
    assert len(rejects) == 1
    assert rejects[0]["detail"]["reason"] == "no plan_hash"

    by_job = db.list_audit(db_path, job_id="job-1")
    assert len(by_job) == 1
    assert by_job[0]["action"] == "submit"

    assert db.list_audit(db_path, limit=1)[0]["action"] == "submit.rejected"


def test_idempotency_lookup(db_path: Path):
    db.upsert_job(
        db_path,
        {
            "job_id": "job-abc",
            "kind": "eval_paired",
            "source": "ovhai",
            "name": "eval",
            "status": "queued",
            "idempotency_key": "idem-1",
        },
    )
    found = db.find_by_idempotency(db_path, "idem-1")
    assert found is not None
    assert found["job_id"] == "job-abc"
    assert db.find_by_idempotency(db_path, "idem-2") is None
    assert db.find_by_idempotency(db_path, "") is None


# --- plan: hashes bind spec + recipe ---------------------------------------


class _FakeSettings:
    def __init__(self, db_path: Path):
        self.db_path = db_path


def test_plan_job_is_deterministic_and_bound_to_recipe(db_path: Path):
    settings = _FakeSettings(db_path)
    first = plan.plan_job(
        settings,
        "eval-crisis-v2",
        {"tag": "v30-eval-v2"},
        None,
    )
    second = plan.plan_job(
        settings,
        "eval-crisis-v2",
        {"tag": "v30-eval-v2"},
        None,
    )
    assert first["plan_hash"] == second["plan_hash"]
    assert first["spec_hash"] == second["spec_hash"]
    assert first["command"].startswith("python -m training.ovh_v22_eval.submit")
    assert "v30-eval-v2" in first["command"]
    assert first["plan_hash"] != plan.plan_job(settings, "eval-crisis-v2", {"tag": "other-tag"}, None)["plan_hash"]


def test_plan_job_rejects_missing_inputs_and_unknown_recipe(db_path: Path):
    settings = _FakeSettings(db_path)
    with pytest.raises(plan.PlanError, match="missing required input"):
        plan.plan_job(settings, "eval-crisis-v2", {}, None)
    with pytest.raises(plan.PlanError, match="unknown recipe"):
        plan.plan_job(settings, "no-such-recipe", {}, None)
    # An input the recipe does not declare is rejected: ignoring it would let a
    # caller believe it changed the outcome (the wrong-suite bug class).
    with pytest.raises(plan.PlanError, match="unknown input 'suite'"):
        plan.plan_job(settings, "eval-crisis-v2", {"tag": "t", "suite": "v1"}, None)


def test_plan_job_rejects_locked_and_badly_typed_overrides(db_path: Path):
    settings = _FakeSettings(db_path)
    with pytest.raises(plan.PlanError, match="locked"):
        plan.plan_job(
            settings,
            "train-sft-latest",
            {"launch_config": "c"},
            {"learning_rate": 1e-4},
        )
    with pytest.raises(plan.PlanError, match="epochs"):
        plan.plan_job(
            settings,
            "train-sft-latest",
            {"launch_config": "c"},
            {"epochs": 99},
        )


def test_dry_run_override_adds_and_omits_a_flag(db_path: Path):
    """A boolean override substitutes as a CLI flag, and vanishes when false."""
    settings = _FakeSettings(db_path)
    off = plan.plan_job(settings, "curate-sft-v30", {"corpus": "/data/corpus"}, {})
    assert off["command"] == "python -m training.build_agent_tool_sft_v30"
    # `corpus` is recorded in the spec but the builder reads its own ROOT, so
    # the plan must say so rather than implying the path was passed through.
    assert any("input 'corpus' is recorded in the spec" in w for w in off["warnings"])
    on = plan.plan_job(settings, "curate-sft-v30", {"corpus": "/data/corpus"}, {"dry_run": True})
    assert on["command"] == "python -m training.build_agent_tool_sft_v30 --dry-run"
    explicit_off = plan.plan_job(
        settings,
        "build-release-v30",
        {"train_payload": "sha256:x", "eval_tag": "v30"},
        {"dry_run": False},
    )
    assert explicit_off["command"] == "python -m training.build_v30_release"


def test_unconsumed_recipe_inputs_are_warned_about(db_path: Path):
    """An input the command never receives must be visible, not silently dropped."""
    settings = _FakeSettings(db_path)
    planned = plan.plan_job(settings, "curate-sft-v30", {"corpus": "/data/corpus"}, {})
    assert any("input 'corpus' is recorded in the spec" in w for w in planned["warnings"])
    # Every recipe must report accurately: an input is either substituted into
    # the command or warned about. None may vanish silently.
    for recipe in recipes.load_recipes():
        inputs = dict.fromkeys(recipe.inputs, "x")
        overrides = {}
        for name, spec in recipe.overrides.items():
            if not spec.get("locked") and "default" in spec:
                overrides[name] = spec["default"]
        resolved = plan.plan_job(settings, recipe.id, inputs, overrides)
        for name in recipe.inputs:
            if f"<{name}>" in recipe.entrypoint:
                continue
            assert any(name in w and "recorded in the spec" in w for w in resolved["warnings"]), (
                recipe.id,
                name,
                resolved["warnings"],
            )


def test_unconsumed_recipe_overrides_are_warned_about(db_path: Path):
    """An override the command never receives must be visible, not silently dropped."""
    settings = _FakeSettings(db_path)
    planned = plan.plan_job(settings, "train-sft-latest", {"launch_config": "c"}, {"epochs": 5})
    assert planned["spec"]["overrides"] == {"epochs": 5}
    assert "<epochs>" not in planned["recipe"]["entrypoint"]
    assert any("override 'epochs' is recorded in the spec" in w for w in planned["warnings"]), planned["warnings"]


def test_dry_run_override_rejects_non_boolean(db_path: Path):
    settings = _FakeSettings(db_path)
    with pytest.raises(plan.PlanError, match="must be a boolean"):
        plan.plan_job(settings, "curate-sft-v30", {"corpus": "/c"}, {"dry_run": "yes"})


def test_local_recipes_support_dry_run_and_validate():
    """Every local recipe must be verifiable without writing artifacts."""
    local = [r for r in recipes.load_recipes() if r.backend == "local"]
    assert local, "expected at least one local recipe"
    for recipe in local:
        assert "dry_run" in recipe.overrides, f"{recipe.id} cannot be dry-run"
        assert recipe.overrides["dry_run"].get("type") == "bool"


# --- backends.evaluate_gate (P3.2) -----------------------------------------


def _summary(sft: float, base: float, run: str = "run-1") -> dict:
    return {
        "tag": "t",
        "runs": {run: {"arm_pass_rates": {"sft": {"rate": sft}, "base": {"rate": base}}}},
    }


def test_gate_passes_when_delta_and_rate_clear_thresholds():
    verdict = backends.evaluate_gate(_summary(0.8, 0.7), min_delta=0.05, min_pass_rate=0.75)
    assert verdict["passed"] is True
    assert verdict["runs"]["run-1"]["delta"] == pytest.approx(0.1)


def test_gate_fails_on_small_delta_and_low_rate():
    verdict = backends.evaluate_gate(_summary(0.6, 0.59), min_delta=0.05, min_pass_rate=0.75)
    assert verdict["passed"] is False
    reasons = " ".join(verdict["reasons"])
    assert "delta sft-base" in reasons
    assert "sft pass rate" in reasons


def test_gate_fails_closed_without_grades_or_runs():
    no_grades = {"tag": "t", "runs": {"r": {"arm_pass_rates": None}}}
    verdict = backends.evaluate_gate(no_grades)
    assert verdict["passed"] is False
    assert any("no graded results" in r for r in verdict["reasons"])

    empty = backends.evaluate_gate({"tag": "t", "runs": {}})
    assert empty["passed"] is False
    assert any("no eval runs" in r for r in empty["reasons"])

    errored = backends.evaluate_gate({"error": "r2_not_configured"})
    assert errored["passed"] is False
    assert any("r2_not_configured" in r for r in errored["reasons"])


def test_gate_requires_base_arm_when_delta_threshold_is_positive():
    summary = {"tag": "t", "runs": {"r": {"arm_pass_rates": {"sft": {"rate": 0.9}}}}}
    strict = backends.evaluate_gate(summary, min_delta=0.01)
    assert strict["passed"] is False
    assert any("base arm missing" in r for r in strict["reasons"])

    lenient = backends.evaluate_gate(summary, min_delta=0.0)
    assert lenient["passed"] is True


def test_gate_fails_when_any_run_fails():
    summary = _summary(0.9, 0.8, run="run-ok")
    summary["runs"]["run-bad"] = {"arm_pass_rates": {"sft": {"rate": 0.2}, "base": {"rate": 0.1}}}
    verdict = backends.evaluate_gate(summary, min_pass_rate=0.5)
    assert verdict["passed"] is False
    assert verdict["runs"]["run-ok"]["passed"] is True
    assert verdict["runs"]["run-bad"]["passed"] is False


def test_gate_thresholds_default_to_a_non_zero_operator_policy():
    """A caller that omits thresholds must not get a rubber stamp.

    With 0/0 defaults, any graded run passes -- including one where the
    fine-tune is no better than the base model -- so the promotion gate would
    be decorative for every caller that forgets the arguments.
    """
    s = Settings(host="127.0.0.1", port=8766, db_path=Path("/tmp/x.db"), env={})
    assert s.gate_min_delta > 0
    assert s.gate_min_pass_rate > 0
    want_delta, want_rate = 0.5, 0.8
    overridden = Settings(
        host="127.0.0.1",
        port=8766,
        db_path=Path("/tmp/x.db"),
        env={
            "TRAINING_MCP_GATE_MIN_DELTA": str(want_delta),
            "TRAINING_MCP_GATE_MIN_PASS_RATE": str(want_rate),
        },
    )
    assert overridden.gate_min_delta == want_delta
    assert overridden.gate_min_pass_rate == want_rate


def test_gate_unwraps_the_guard_envelope_from_the_server_tool():
    """``server.eval_gate`` passes ``_guard``'s envelope straight through.

    Reading ``runs`` only at the top level made every verdict report
    "no eval runs found under the tag", permanently blocking all promotions
    no matter what the real eval data said.
    """
    wrapped = {"ok": True, "result": _summary(0.9, 0.2)}
    verdict = backends.evaluate_gate(wrapped, min_delta=0.5, min_pass_rate=0.8)
    assert verdict["passed"] is True, verdict
    assert verdict["runs"]["run-1"]["sft_rate"] == pytest.approx(0.9)


def test_gate_reports_backend_errors_inside_a_guard_envelope():
    wrapped = {"ok": True, "result": {"error": "r2_not_configured"}}
    verdict = backends.evaluate_gate(wrapped)
    assert verdict["passed"] is False
    assert any("r2_not_configured" in r for r in verdict["reasons"])


# --- real R2 results.jsonl grading ------------------------------------------


def _result_row(arm: str, checks: list[bool | None]) -> str:
    metrics = [{"type": "response_omits", "passed": c} for c in checks if c is not None]
    return json.dumps({"arm_id": arm, "status": "success", "state_metrics": metrics, "response": "x"})


def test_results_parser_grades_from_state_metrics_not_a_top_level_field():
    """Real rows carry no top-level ``passed``; the grade lives in state_metrics."""
    body = "\n".join(
        [
            _result_row("base", [True, True, False, False]),
            _result_row("base", [False, False]),
            _result_row("sft", [True, True, True, True]),
            _result_row("sft", [True, True]),
        ]
    ).encode()
    parsed = backends._parse_results_body(body)
    rates = parsed["arm_pass_rates"]
    assert rates["base"] == {"passed": 0, "total": 2, "rate": 0.0}
    assert rates["sft"] == {"passed": 2, "total": 2, "rate": 1.0}
    assert parsed["delta_sft_minus_base"] == pytest.approx(1.0)


def test_results_parser_requires_every_check_to_pass():
    """Any-check-pass scores 1.0 for both arms on real data and gates nothing."""
    body = "\n".join([_result_row("base", [True, False]), _result_row("sft", [True, False])]).encode()
    parsed = backends._parse_results_body(body)
    assert parsed["arm_pass_rates"]["base"]["rate"] == 0.0
    assert parsed["arm_pass_rates"]["sft"]["rate"] == 0.0


def test_results_parser_honours_an_explicit_top_level_passed():
    row = json.dumps({"arm_id": "sft", "passed": True, "state_metrics": [{"passed": False}]})
    parsed = backends._parse_results_body(row.encode())
    assert parsed["arm_pass_rates"]["sft"]["rate"] == 1.0


def test_results_parser_counts_ungraded_rows_against_their_arm():
    """Ungraded rows count against their arm once any row carries a grade."""
    body = "\n".join(
        [
            json.dumps({"arm_id": "sft", "state_metrics": [], "response": "x"}),
            _result_row("sft", [True, True]),
        ]
    ).encode()
    parsed = backends._parse_results_body(body)
    assert parsed["arm_pass_rates"]["sft"] == {"passed": 1, "total": 2, "rate": 0.5}


def test_an_all_ungradeable_file_fails_closed():
    """Zero graded rows must not read as a valid zero-rate arm."""
    row = json.dumps({"arm_id": "sft", "state_metrics": [], "response": "x"})
    parsed = backends._parse_results_body(row.encode())
    assert parsed["arm_pass_rates"] is None
    assert "no graded rows" in parsed["note"]


def test_results_parser_counts_errored_rows_against_the_pass_rate():
    """10 passes out of 50 attempts must not read as a perfect rate."""
    body = "\n".join(
        [
            _result_row("sft", [True, True]),
            _result_row("sft", [True, True]),
            json.dumps({"arm_id": "sft", "status": "error", "response": None}),
            json.dumps({"arm_id": "sft", "status": "error", "response": None}),
        ]
    ).encode()
    parsed = backends._parse_results_body(body)
    assert parsed["arm_pass_rates"]["sft"] == {"passed": 2, "total": 4, "rate": 0.5}


def test_results_parser_reports_an_empty_results_file():
    parsed = backends._parse_results_body(b"")
    assert parsed["arm_pass_rates"] is None
    assert "no results rows" in parsed["note"]


# --- runner lifecycle (finished_at, cancel-vs-monitor, task4) ----------------


def _settings(db_path, env=None):
    return Settings(host="127.0.0.1", port=8766, db_path=db_path, env=env or {})


def test_terminal_rows_carry_finished_at(db_path):
    """A terminal transition must set finished_at; it was always null before."""
    db.upsert_job(
        db_path,
        {"job_id": "job-t1", "kind": "build_release", "source": "local", "name": "r", "status": "queued"},
    )
    assert db.get_job(db_path, "job-t1")["finished_at"] is None

    terminal = db.get_job(db_path, "job-t1")
    runner._record_terminal_outcome(_settings(db_path), "job-t1")
    db.upsert_job(
        db_path,
        {
            "job_id": "job-t1",
            "kind": terminal["kind"],
            "source": terminal["source"],
            "name": terminal["name"],
            "status": "succeeded",
            "stage": "succeeded",
            "error": None,
            "finished_at": db.now(),
        },
    )
    assert db.get_job(db_path, "job-t1")["finished_at"] is not None


def test_cancel_is_refused_for_terminal_jobs(db_path):
    """cancel() must not resurrect or re-terminate a finished job."""
    db.upsert_job(
        db_path,
        {"job_id": "job-done", "kind": "build_release", "source": "local", "status": "succeeded"},
    )
    assert runner.cancel(_settings(db_path), "job-done") is False
    assert db.get_job(db_path, "job-done")["status"] == "succeeded"


def test_cancel_marks_finished_at(db_path):
    db.upsert_job(
        db_path,
        {"job_id": "job-live", "kind": "build_release", "source": "local", "status": "running"},
    )
    assert runner.cancel(_settings(db_path), "job-live") is True
    job = db.get_job(db_path, "job-live")
    assert job["status"] == "cancelled"
    assert job["finished_at"] is not None


def test_monitor_preserves_operator_cancel(db_path, monkeypatch):
    """The monitor must not rewrite a cancel() as 'failed'."""
    settings = _settings(db_path)
    db.upsert_job(
        db_path,
        {"job_id": "job-c", "kind": "build_release", "source": "local", "status": "cancelled"},
    )

    class _Proc:
        returncode = -15

        def wait(self, *_args, **_kwargs):
            return self.returncode

    monkeypatch.setattr(runner, "_record_terminal_outcome", lambda *_a, **_k: None)
    runner._monitor(settings, "job-c", _Proc(), "/tmp/whatever.log")
    assert db.get_job(db_path, "job-c")["status"] == "cancelled"


# --- sync: lost-backend reconciliation (PIX-5012) --------------------------


def test_monitor_loses_a_cancel_that_lands_during_collecting(db_path, monkeypatch):
    """cancel() runs on the request thread and must win over the monitor.

    The monitor reads the row before its half-second collecting window, so a
    cancel that lands inside that window used to be overwritten with `failed`.
    """
    settings = _settings(db_path)
    db.upsert_job(
        db_path,
        {"job_id": "job-race", "kind": "build_release", "source": "local", "status": "running"},
    )
    recorded: list[str] = []
    monkeypatch.setattr(runner, "_record_terminal_outcome", lambda _s, job_id: recorded.append(job_id))

    class _Proc:
        returncode = -15

        def wait(self, *_args, **_kwargs):
            return self.returncode

        def terminate(self):
            return None

    def _cancel_during_collecting(_seconds):
        runner.cancel(settings, "job-race")

    monkeypatch.setattr(runner.time, "sleep", _cancel_during_collecting)
    try:
        runner._monitor(settings, "job-race", _Proc(), "/tmp/whatever.log")
    finally:
        runner._forget_cancelled("job-race")

    job = db.get_job(db_path, "job-race")
    assert job["status"] == "cancelled"
    assert job["error"] == "cancelled by operator"
    assert job["finished_at"] is not None
    assert recorded == ["job-race"]


def test_reconcile_marks_vanished_backend_job_failed(db_path):
    db.upsert_job(
        db_path,
        {
            "job_id": "ovh-gone",
            "kind": "eval_paired",
            "source": "ovhai",
            "status": "running",
            "backend_id": "uuid-gone",
        },
    )
    result = sync.reconcile_lost_backends(_settings(db_path), seen={"ovhai": set()})
    job = db.get_job(db_path, "ovh-gone")
    assert job["status"] == "failed"
    assert job["error"] == "backend_lost"
    assert job["finished_at"] is not None
    assert result["lost"] == 1


def test_reconcile_keeps_live_and_skips_unknown_sources(db_path):
    db.upsert_job(
        db_path,
        {"job_id": "ovh-live", "kind": "eval_paired", "source": "ovhai", "status": "running", "backend_id": "u1"},
    )
    db.upsert_job(
        db_path,
        {"job_id": "wandb-x", "kind": "train_sft", "source": "wandb", "status": "running", "backend_id": "r1"},
    )
    result = sync.reconcile_lost_backends(_settings(db_path), seen={"ovhai": {"u1"}})
    assert db.get_job(db_path, "ovh-live")["status"] == "running"
    # wandb was not queried, so it must not be called lost
    assert db.get_job(db_path, "wandb-x")["status"] == "running"
    assert result["lost"] == 0


def test_reconcile_ignores_terminal_rows(db_path):
    db.upsert_job(
        db_path,
        {"job_id": "ovh-done", "kind": "eval_paired", "source": "ovhai", "status": "succeeded", "backend_id": "u9"},
    )
    result = sync.reconcile_lost_backends(_settings(db_path), seen={"ovhai": set()})
    assert db.get_job(db_path, "ovh-done")["status"] == "succeeded"
    assert result["lost"] == 0


# --- sync: historical import (P0.10) ---------------------------------------


def test_import_manual_rows_validates_required_fields(db_path):
    result = sync.import_manual_rows(
        _settings(db_path),
        [
            {"job_id": "hist-1", "kind": "train_sft", "status": "succeeded", "name": "v29"},
            {"job_id": "hist-2", "kind": "train_sft"},
        ],
    )
    assert result["imported"] == 1
    assert result["rejected"] == [{"index": 1, "missing": ["status"]}]
    job = db.get_job(db_path, "hist-1")
    assert job["source"] == "manual"
    assert job["requested_by"] == "import"


def test_normalize_wandb_run_maps_states():
    assert backends.normalize_wandb_run({"id": "abc", "state": "finished"})["status"] == "succeeded"
    assert backends.normalize_wandb_run({"id": "abc", "state": "crashed"})["status"] == "failed"
    assert backends.normalize_wandb_run({"id": "abc", "state": "running"})["status"] == "running"
    row = backends.normalize_wandb_run({"id": "abc", "state": "finished", "name": "sft-v30"})
    assert row["job_id"] == "wandb-abc"
    assert row["source"] == "wandb"


# --- foresight payload (PIX-5000 provenance) --------------------------------


def test_foresight_payload_is_best_effort_on_bad_url(db_path):
    settings = _settings(db_path, {"FORESIGHT_URL": "http://127.0.0.1:1/mcp"})
    outcome = foresight.store_terminal_memory(settings, {"job_id": "j", "status": "succeeded"})
    assert outcome["stored"] is False
    assert "error" in outcome


def test_foresight_disabled_by_config(db_path):
    assert _settings(db_path, {"TRAINING_MCP_FORESIGHT": "0"}).foresight_enabled is False
    assert _settings(db_path).foresight_enabled is True


def test_daily_cap_counting_ignores_imported_rows():
    """A backfill sweep must not burn the day's launch budget.

    Imported rows carry the ``import`` actor; the daily cap must ignore them or
    one ``include_backends=True`` sweep blocks every real submit until midnight.
    """
    today = datetime.now(UTC).date().isoformat()
    yesterday = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()

    def row(actor, when=today):
        return {"requested_by": actor, "created_at": f"{when}T09:00:00+00:00"}

    imported = [row(IMPORT_ACTOR) for _ in range(30)]
    real = [row("pipeline-agent"), row("schedule:weekly-train")]
    anonymous = [row(None)]
    stale = [row("pipeline-agent", yesterday)]

    assert count_toward_daily_cap(imported) == 0
    assert count_toward_daily_cap(imported + anonymous) == 0
    assert count_toward_daily_cap(imported + stale) == 0
    assert count_toward_daily_cap(imported + real) == len(real)


def test_command_interpreter_is_rewritten_to_the_service_venv():
    """Recipes say ``python -m ...``; systemd's PATH has only ``python3``.

    Without the rewrite the spawn raises FileNotFoundError on a daemon thread.
    """
    rewritten = runner._resolve_interpreter("python -m training.build_v30_release --dry-run")
    assert rewritten.startswith(sys.executable)
    assert rewritten.endswith("-m training.build_v30_release --dry-run")
    assert runner._resolve_interpreter("python3 -m x") == f"{sys.executable} -m x"
    # Non-python entrypoints are untouched.
    assert runner._resolve_interpreter("ovhai job list") == "ovhai job list"


def test_spawn_failure_lands_in_a_terminal_state(db_path, monkeypatch):
    """A Popen OSError must fail the job, not strand it at `submitted`."""
    settings = _settings(db_path)
    db.upsert_job(
        db_path,
        {
            "job_id": "job-spawnfail",
            "kind": "build_release",
            "source": "local",
            "name": "spawnfail",
            "status": "queued",
            "requested_by": "test",
        },
    )
    monkeypatch.setattr(runner, "_record_terminal_outcome", lambda *_a, **_k: None)

    def boom(*_a, **_k):
        raise FileNotFoundError("no such file: python")

    monkeypatch.setattr(runner.subprocess, "Popen", boom)
    # Run the job body synchronously instead of on the daemon thread.
    row = db.get_job(db_path, "job-spawnfail")
    started = threading.Event()

    class Immediate:
        def __init__(self, target=None, **_kw):
            self._target = target

        def start(self):
            started.set()
            self._target()

    monkeypatch.setattr(runner.threading, "Thread", Immediate)
    runner.submit(settings, "job-spawnfail", row, "python -m training.build_v30_release", "local")
    assert started.is_set()
    final = db.get_job(db_path, "job-spawnfail")
    assert final["status"] == "failed"
    assert "spawn failed" in (final["error"] or "")
    assert final["finished_at"] is not None


# --- audit fixes (PIX-5000 deep audit) ---------------------------------------
#
# Each test below pins a defect found by running the control plane end to end
# rather than reading it: a guard that was advertised but absent, a value that
# silently pointed somewhere else, or a state machine that could be talked out
# of its terminal state.


#: ai/training/mcp/runner.py acquires the per-backend lock with this bound.
_SINGLE_FLIGHT_TIMEOUT = 300


class ImmediateThread:
    """Run a job body on the calling thread so assertions are deterministic."""

    def __init__(self, target=None, **_kw):
        self._target = target

    def start(self):
        self._target()


def test_boolean_override_flag_is_derived_from_its_name(tmp_path):
    """A bool override must not hardcode `--dry-run`.

    The old codec turned *every* truthy bool into the literal ``--dry-run``, so
    a recipe with two boolean overrides rendered the same flag twice and the
    other one silently vanished.
    """
    registry = tmp_path / "recipes.yaml"
    registry.write_text(
        """
recipes:
  - id: multi-flag-v1
    kind: curate_dataset
    description: Two boolean overrides.
    backend: local
    entrypoint: "python -m training.thing <verbose> <dry_run>"
    cost_class: cpu-minute
    est_cost_usd: 0
    inputs: {}
    overrides:
      verbose: {type: boolean, default: false}
      dry_run: {type: boolean, default: false}
    outputs: {}
"""
    )
    found = {r.id: r for r in recipes.load_recipes(registry)}
    recipe = found["multi-flag-v1"]
    command = plan._resolve_command(recipe, {}, {"verbose": True, "dry_run": True})
    assert command == "python -m training.thing --verbose --dry-run"
    # A falsy bool omits the flag entirely, leaving no stray placeholder.
    assert plan._resolve_command(recipe, {}, {"verbose": False, "dry_run": False}) == "python -m training.thing"
    # `flag:` in the override spec overrides the derived name.
    recipe.overrides["verbose"]["flag"] = "--loud"
    assert (
        plan._resolve_command(recipe, {}, {"verbose": True, "dry_run": True})
        == "python -m training.thing --loud --dry-run"
    )


def test_plan_cache_expires_and_prunes(db_path: Path):
    """A plan approved long ago must not stay submittable forever.

    The rejection message said "not found or expired" while nothing expired.
    """
    db.cache_plan(db_path, "hash-fresh", "eval-crisis-v2", {"inputs": {"tag": "t"}, "overrides": {}})
    assert db.get_cached_plan(db_path, "hash-fresh", max_age_seconds=3600) is not None
    # An unparseable timestamp fails closed (treated as ancient).
    assert db._age_seconds("not-a-date") == float("inf")
    assert db._age_seconds(None) == float("inf")
    assert db.prune_plans(db_path, max_age_seconds=3600) == 0
    # Age the row past its TTL: it disappears from the lookup and from the table.
    conn = db._conn(db_path)
    conn.execute(
        "UPDATE plan_cache SET created_at = ? WHERE plan_hash = ?", ("2020-01-01T00:00:00+00:00", "hash-fresh")
    )
    conn.commit()
    assert db.get_cached_plan(db_path, "hash-fresh", max_age_seconds=3600) is None
    assert db.prune_plans(db_path, max_age_seconds=3600) == 1
    assert db.get_cached_plan(db_path, "hash-fresh") is None


def test_cap_rows_is_unpaged_while_list_jobs_clamps(db_path: Path):
    """Budget checks must see every row, not one 500-row page."""
    page_size = 500
    row_count = page_size + 1
    rows_per_job_cost = 1.5
    for index in range(row_count):
        db.upsert_job(
            db_path,
            {
                "job_id": f"job-cap-{index:04d}",
                "kind": "eval_paired",
                "source": "local",
                "name": "cap",
                "status": "succeeded",
                "requested_by": "tester",
                "created_at": "2026-11-20T09:00:00+00:00",
                "cost_estimate": str(rows_per_job_cost),
            },
        )
    assert len(db.list_jobs(db_path, limit=page_size)) == page_size
    rows = db.cap_rows(db_path, since="2026-11-20")
    assert len(rows) == row_count
    assert db.cap_rows(db_path, since="2026-11-21") == []
    assert sum(float(r["cost_estimate"]) for r in rows) == row_count * rows_per_job_cost


def test_registry_files_are_owner_only(tmp_path):
    """The registry holds job specs and the audit trail; not world-readable."""
    path = tmp_path / "state" / "jobs.db"
    file_mode = 0o600
    dir_mode = 0o700
    db.upsert_job(
        path, {"job_id": "job-mode", "kind": "eval_paired", "source": "local", "name": "m", "status": "queued"}
    )
    assert (path.stat().st_mode & 0o777) == file_mode
    assert (path.parent.stat().st_mode & 0o777) == dir_mode
    for sidecar in (path.with_name("jobs.db-wal"), path.with_name("jobs.db-shm")):
        if sidecar.exists():
            assert (sidecar.stat().st_mode & 0o777) == file_mode


def test_schema_version_is_recorded(tmp_path):
    """`_migrate` must stamp PRAGMA user_version so operators can tell the schema."""
    path = tmp_path / "state" / "jobs.db"
    db.upsert_job(
        path, {"job_id": "job-ver", "kind": "eval_paired", "source": "local", "name": "v", "status": "queued"}
    )
    version = db._conn(path).execute("PRAGMA user_version").fetchone()[0]
    assert version == db.SCHEMA_VERSION
    # Re-running the migration is idempotent.
    db._migrate(db._conn(path))
    assert db._conn(path).execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION


def test_job_cwd_and_pythonpath_point_at_real_directories():
    """The spawn cwd/PYTHONPATH were off by one directory.

    `parents[2]` of the registry path lands on `ai/training`, not `ai`, so a
    recipe's cwd-relative path (the launch config) never resolved.
    """
    assert runner.JOB_CWD == mcp_config.WORKSPACE_ROOT
    assert runner.JOB_CWD.is_dir()
    env = runner._build_env(_settings(Path("/tmp") / "x" / "state" / "jobs.db"))
    assert str(mcp_config.AI_ROOT) in (env.get("PYTHONPATH") or "")


def test_build_env_does_not_leak_unlisted_secrets(monkeypatch):
    """Job subprocesses get only the allowlisted variables."""
    monkeypatch.setenv("TOTALLY_UNRELATED_SECRET", "leak-me")
    monkeypatch.setenv("PATH", os.environ.get("PATH", "/usr/bin"))
    env = runner._build_env(
        _settings(
            Path("/tmp") / "y" / "state" / "jobs.db",
            env={"WANDB_API_KEY": "allowed", "UNRELATED_DOTENV_SECRET": "leak-me-too"},
        )
    )
    assert "TOTALLY_UNRELATED_SECRET" not in env
    assert "UNRELATED_DOTENV_SECRET" not in env
    assert env["WANDB_API_KEY"] == "allowed"
    assert "PATH" in env


def test_cancel_before_spawn_prevents_the_run(db_path, monkeypatch):
    """Cancelling a queued job must stop it, not be overwritten by `succeeded`.

    The submit thread acquired the per-backend lock before writing `submitted`;
    cancel saw no live process, wrote `cancelled`, and the thread then spawned
    and the monitor overwrote it with `succeeded`.
    """
    settings = _settings(db_path)
    db.upsert_job(
        db_path,
        {"job_id": "job-cancelrace", "kind": "eval_paired", "source": "local", "name": "r", "status": "queued"},
    )
    row = db.get_job(db_path, "job-cancelrace")
    spawned: list[str] = []

    def fake_popen(*_a, **_k):  # pragma: no cover - must not be reached
        spawned.append("popen")
        raise AssertionError("a cancelled job must not spawn")

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner, "_record_terminal_outcome", lambda *_a, **_k: None)

    # Simulate cancel arriving while the thread waits for the backend lock.
    acquired = threading.Event()

    class SlowLock:
        def acquire(self, timeout=None):
            # The runner must bound its wait; an unbounded wait would strand
            # every queued job behind one hung backend.
            assert timeout == _SINGLE_FLIGHT_TIMEOUT
            acquired.set()
            return True

        def release(self):
            pass

    monkeypatch.setattr(runner, "_lock_for", lambda _backend: SlowLock())
    monkeypatch.setattr(runner.threading, "Thread", ImmediateThread)
    runner._mark_cancelled("job-cancelrace")
    runner.submit(settings, "job-cancelrace", row, "python -m training.build_v30_release", "local")
    assert acquired.is_set()
    assert spawned == []
    assert db.get_job(db_path, "job-cancelrace")["status"] == "queued"
    runner._forget_cancelled("job-cancelrace")


def test_single_flight_timeout_records_a_finished_row(db_path, monkeypatch):
    """The single-flight timeout path must complete the row, not leave it open."""
    settings = _settings(db_path)
    db.upsert_job(
        db_path,
        {"job_id": "job-sf", "kind": "eval_paired", "source": "local", "name": "r", "status": "queued"},
    )
    row = db.get_job(db_path, "job-sf")
    recorded: list[str] = []

    class BusyLock:
        def acquire(self, timeout=None):
            assert timeout == _SINGLE_FLIGHT_TIMEOUT
            return False

        def release(self):
            pass

    monkeypatch.setattr(runner, "_lock_for", lambda _backend: BusyLock())
    monkeypatch.setattr(runner, "_record_terminal_outcome", lambda _s, job_id: recorded.append(job_id))
    monkeypatch.setattr(runner.threading, "Thread", ImmediateThread)
    runner.submit(settings, "job-sf", row, "python -m training.build_v30_release", "local")
    final = db.get_job(db_path, "job-sf")
    assert final["status"] == "failed"
    assert "single-flight timeout" in (final["error"] or "")
    assert final["finished_at"] is not None
    assert recorded == ["job-sf"]


def test_train_recipe_and_docs_agree_with_the_real_entrypoints():
    """Every launching recipe must pass --execute; docs must point at real files."""
    for recipe in recipes.load_recipes():
        if recipe.kind in {"train_sft", "eval_paired"}:
            assert "--execute" in recipe.entrypoint, recipe.id
    here = Path(__file__).resolve().parents[3] / "training" / "mcp"
    for name in ("RECIPE_WORKFLOW.md", "MULTI_NODE_DECISION.md"):
        assert (here / name).is_file()


# --- submit_job guards (the defects an operator would actually hit) ---------


def _server_settings(tmp_path, **overrides):
    """A real Settings pointed at a temp registry, with no backends configured."""
    env = {
        "TRAINING_MCP_MAX_CONCURRENT_JOBS": str(overrides.get("max_concurrent_jobs", 1)),
        "TRAINING_MCP_DAILY_JOB_CAP": str(overrides.get("daily_job_cap", 5)),
        "TRAINING_MCP_MONTHLY_COST_CAP_USD": str(overrides.get("monthly_cost_cap_usd", 100.0)),
    }
    return Settings(host="127.0.0.1", port=8766, db_path=tmp_path / "jobs.db", env=env)


@pytest.fixture
def server_env(tmp_path, monkeypatch):
    """Wire the server module at a temp registry and stub the spawn."""
    cfg = _server_settings(tmp_path)
    monkeypatch.setattr(server, "settings", lambda: cfg)
    spawned: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        server.runner,
        "submit",
        lambda _s, job_id, _row, command, backend: spawned.append((job_id, command, backend)),
    )
    return server, cfg, spawned


def test_submit_rejects_a_spec_that_differs_from_the_approved_plan(server_env):
    """The plan hash must bind the *spec*, not just the recipe name.

    The old check compared only `recipe`; a caller could plan a harmless dry run,
    show it to an operator, and submit the same hash with `dry_run: false` so
    different work ran than the one that was approved.
    """
    server, _cfg, spawned = server_env
    planned = server.plan_job("curate-sft-v30", {"corpus": "x"}, {"dry_run": True}, "operator")
    assert planned["ok"] is True
    approved_hash = planned["plan_hash"]
    assert "--dry-run" in planned["command"]

    tampered = server.submit_job(
        plan_hash=approved_hash,
        recipe="curate-sft-v30",
        inputs={"corpus": "x"},
        overrides={"dry_run": False},
        requested_by="attacker",
    )
    assert tampered["ok"] is False
    assert "differ from the approved plan" in tampered["error"]
    assert "overrides" in tampered["error"]
    assert spawned == []
    rejections = db.list_audit(_cfg.db_path, action="submit.rejected")
    assert rejections
    assert rejections[0]["detail"]["reason"] == "submitted spec differs from the approved plan"
    assert rejections[0]["detail"]["fields"] == ["overrides"]


def test_submit_runs_the_approved_spec_even_when_the_caller_repeats_it(server_env):
    """A matching submit succeeds and launches the approved command."""
    server, _cfg, spawned = server_env
    planned = server.plan_job("curate-sft-v30", {"corpus": "x"}, {"dry_run": True}, "operator")
    result = server.submit_job(
        plan_hash=planned["plan_hash"],
        recipe="curate-sft-v30",
        inputs={"corpus": "x"},
        overrides={"dry_run": True},
        requested_by="operator",
    )
    assert result["ok"] is True
    assert len(spawned) == 1
    _job_id, command, backend = spawned[0]
    assert "--dry-run" in command
    assert backend == "local"


def test_submit_accepts_a_repeat_of_the_callers_own_overrides(server_env):
    """The cached spec carries recipe defaults, so the same submit must expand too.

    plan_job records `epochs` (a declared default) even when the caller passed no
    overrides; re-submitting that empty dict then compared {} against
    {"epochs": 3} and was rejected as a tampered spec.
    """
    server, _cfg, spawned = server_env
    planned = server.plan_job("train-sft-latest", {"launch_config": "c"}, {}, "operator")
    assert planned["spec"]["overrides"] == {"epochs": 3}
    result = server.submit_job(
        plan_hash=planned["plan_hash"],
        recipe="train-sft-latest",
        inputs={"launch_config": "c"},
        overrides={},
        requested_by="operator",
    )
    assert result["ok"] is True, result
    assert len(spawned) == 1


def test_submit_rejects_a_recipe_changed_after_approval(server_env, monkeypatch):
    """The recomputed plan must reproduce the approved hash, not merely parse."""
    server, cfg, spawned = server_env
    planned = server.plan_job("train-sft-latest", {"launch_config": "c"}, {}, "operator")
    assert planned["ok"] is True

    drifted = [
        replace(recipe, entrypoint=recipe.entrypoint.replace("wandb_sft_v15_execute", "wandb_sft_v16_execute"))
        if recipe.id == "train-sft-latest"
        else recipe
        for recipe in recipes.load_recipes()
    ]
    monkeypatch.setattr(recipes, "load_recipes", lambda: drifted)

    result = server.submit_job(plan_hash=planned["plan_hash"], requested_by="operator")
    assert result["ok"] is False
    assert "changed since this plan was approved" in result["error"]
    assert spawned == []
    rejections = db.list_audit(cfg.db_path, action="submit.rejected")
    assert rejections[0]["detail"]["reason"] == "plan changed since approval"


def test_submit_accepts_a_repeat_after_a_recipe_default_changed(server_env, monkeypatch):
    """A registry default must not turn a faithful repeat into a tampered spec."""
    server, _cfg, spawned = server_env
    planned = server.plan_job("train-sft-latest", {"launch_config": "c"}, {}, "operator")
    assert planned["spec"]["overrides"] == {"epochs": 3}

    drifted = [
        replace(
            recipe,
            overrides={**recipe.overrides, "epochs": {**recipe.overrides["epochs"], "default": 4}},
        )
        if recipe.id == "train-sft-latest"
        else recipe
        for recipe in recipes.load_recipes()
    ]
    monkeypatch.setattr(recipes, "load_recipes", lambda: drifted)

    result = server.submit_job(
        plan_hash=planned["plan_hash"],
        inputs={"launch_config": "c"},
        overrides={},
        requested_by="operator",
    )
    assert result["ok"] is True, result
    assert len(spawned) == 1


def test_submit_rejects_a_plan_older_than_its_ttl(server_env):
    """An approval must expire, or an old hash stays submittable forever."""
    server, cfg, spawned = server_env
    planned = server.plan_job("curate-sft-v30", {"corpus": "x"}, {"dry_run": True}, "operator")
    conn = db._conn(cfg.db_path)
    conn.execute("UPDATE plan_cache SET created_at = ?", ("2020-01-01T00:00:00+00:00",))
    conn.commit()
    result = server.submit_job(plan_hash=planned["plan_hash"], requested_by="operator")
    assert result["ok"] is False
    assert "not found or expired" in result["error"]
    assert spawned == []


def test_submit_enforces_the_concurrency_cap(server_env):
    """`max_concurrent_jobs` was audited nowhere and enforced nowhere."""
    server, cfg, spawned = server_env
    db.upsert_job(
        cfg.db_path,
        {"job_id": "job-busy", "kind": "eval_paired", "source": "local", "name": "b", "status": "running"},
    )
    planned = server.plan_job("curate-sft-v30", {"corpus": "x"}, {"dry_run": True}, "operator")
    result = server.submit_job(plan_hash=planned["plan_hash"], requested_by="operator")
    assert result["ok"] is False
    assert "max concurrent jobs reached" in result["error"]
    assert spawned == []
    rejection = db.list_audit(cfg.db_path, action="submit.rejected")[0]
    assert rejection["detail"] == {"reason": "max concurrent jobs reached", "cap": 1, "active_count": 1}


def test_submit_monthly_cap_counts_the_whole_month(server_env):
    """A month of sequential jobs must accumulate, not each pass individually."""
    server, cfg, spawned = server_env
    # One succeeded job earlier this month already spent $99 of the $100 cap.
    month_start = datetime.now(UTC).replace(day=1, hour=1, minute=0, second=0, microsecond=0)
    db.upsert_job(
        cfg.db_path,
        {
            "job_id": "job-spent",
            "kind": "eval_paired",
            "source": "local",
            "name": "s",
            "status": "succeeded",
            "requested_by": "operator",
            "created_at": month_start.isoformat(timespec="seconds"),
            "cost_estimate": "99.0",
        },
    )
    planned = server.plan_job("eval-crisis-v2", {"tag": "t"}, None, "operator")
    assert planned["est_cost_usd"] > 1.0
    result = server.submit_job(plan_hash=planned["plan_hash"], requested_by="operator")
    assert result["ok"] is False
    assert "monthly cost cap would be exceeded" in result["error"]
    assert spawned == []


def test_submit_daily_cap_ignores_imported_rows(server_env):
    """Imported history must not consume the day's launch budget."""
    server, cfg, _spawned = server_env
    for index in range(30):
        db.upsert_job(
            cfg.db_path,
            {
                "job_id": f"job-import-{index}",
                "kind": "eval_paired",
                "source": "wandb",
                "name": "i",
                "status": "succeeded",
                "requested_by": IMPORT_ACTOR,
                "created_at": f"{datetime.now(UTC).date().isoformat()}T01:00:00+00:00",
            },
        )
    planned = server.plan_job("curate-sft-v30", {"corpus": "x"}, {"dry_run": True}, "operator")
    result = server.submit_job(plan_hash=planned["plan_hash"], requested_by="operator")
    assert result["ok"] is True, result


# --- PR 619 threads: post-spawn cancel race, stale cancel read, orphans, tokens ---


def test_cancel_wins_the_row_while_the_process_spawns(db_path, monkeypatch):
    """The running write must lose to a cancel that lands during the spawn.

    The worker used to upsert `running` unconditionally after registering the
    process, so a cancel that had already won the row was overwritten, the
    process kept running, and the monitor's marker check returned without a
    terminal state — the job stayed `running` forever while holding a slot.
    """
    settings = _settings(db_path)
    db.upsert_job(
        db_path,
        {"job_id": "job-r2", "kind": "build_release", "source": "local", "name": "r", "status": "queued"},
    )
    row = db.get_job(db_path, "job-r2")
    recorded: list[str] = []
    monkeypatch.setattr(runner, "_record_terminal_outcome", lambda _s, job_id: recorded.append(job_id))

    terminated: list[str] = []
    popened: list[str] = []

    class _Proc:
        def terminate(self):
            terminated.append("terminate")

        def wait(self, *_a, **_k):
            return -15

        def kill(self):
            terminated.append("kill")

    def fake_popen(*_a, **_k):
        popened.append("spawned")
        return _Proc()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    real_transition = db.transition_job

    def racing_transition(path, job_id, **kwargs):
        if kwargs.get("status") == "running":
            # The operator's cancel lands after the process is registered but
            # before the worker writes `running`: it finds the process,
            # terminates it, and wins the row.
            assert runner.cancel(settings, job_id) is True
        return real_transition(path, job_id, **kwargs)

    monkeypatch.setattr(db, "transition_job", racing_transition)
    monkeypatch.setattr(runner.threading, "Thread", ImmediateThread)
    runner.submit(settings, "job-r2", row, "python -m training.build_v30_release", "local")
    job = db.get_job(db_path, "job-r2")
    assert popened == ["spawned"]
    assert terminated, "the cancelled job's process must be stopped"
    assert job["status"] == "cancelled"
    assert job["error"] == "cancelled by operator"
    assert job["finished_at"] is not None
    # One terminal outcome, one provenance record: cancel() recorded it.
    assert recorded == ["job-r2"]


def test_cancel_loses_to_a_terminal_result_committed_after_its_read(db_path, monkeypatch):
    """A stale read must not let cancel() rewrite a committed result.

    cancel() read `collecting` before the monitor committed `succeeded`; the
    unconditional upsert used to rewrite that terminal row and record its
    provenance a second time. The guarded update keeps the monitor's outcome.
    """
    settings = _settings(db_path)
    db.upsert_job(
        db_path,
        {"job_id": "job-stale", "kind": "build_release", "source": "local", "status": "running"},
    )
    recorded: list[str] = []
    monkeypatch.setattr(runner, "_record_terminal_outcome", lambda _s, job_id: recorded.append(job_id))
    # The monitor commits the terminal result after cancel() read the row.
    db.upsert_job(
        db_path,
        {
            "job_id": "job-stale",
            "kind": "build_release",
            "source": "local",
            "status": "succeeded",
            "finished_at": db.now(),
        },
    )
    stale = dict(db.get_job(db_path, "job-stale"), status="collecting")
    real_get = db.get_job
    monkeypatch.setattr(db, "get_job", lambda _path, _job_id: stale)
    assert runner.cancel(settings, "job-stale") is False
    monkeypatch.undo()
    job = real_get(db_path, "job-stale")
    assert job["status"] == "succeeded"
    assert recorded == []


def test_orphaned_process_rows_fail_at_startup(db_path):
    """A restart orphans every in-process job row; the startup sweep clears them."""
    db.upsert_job(db_path, {"job_id": "loc-1", "kind": "curate_dataset", "source": "local", "status": "running"})
    db.upsert_job(
        db_path,
        {"job_id": "ws-1", "kind": "train_sft", "source": "wandb-serverless", "status": "queued"},
    )
    db.upsert_job(
        db_path,
        {"job_id": "ovh-1", "kind": "eval_paired", "source": "ovhai", "status": "running", "backend_id": "u1"},
    )
    db.upsert_job(db_path, {"job_id": "ovh-nb", "kind": "eval_paired", "source": "ovhai", "status": "queued"})
    db.upsert_job(db_path, {"job_id": "loc-done", "kind": "curate_dataset", "source": "local", "status": "succeeded"})
    db.upsert_job(db_path, {"job_id": "loc-plan", "kind": "curate_dataset", "source": "local", "status": "planned"})
    swept = db.fail_orphaned_process_jobs(db_path)
    assert sorted(swept) == ["loc-1", "ovh-nb", "ws-1"]
    failed = db.get_job(db_path, "loc-1")
    assert failed["status"] == "failed"
    assert "orphaned" in (failed["error"] or "")
    assert failed["finished_at"] is not None
    assert db.get_job(db_path, "ws-1")["status"] == "failed"
    assert db.get_job(db_path, "ovh-nb")["status"] == "failed"
    # Rows that name remote work keep their backend_id and are the
    # reconciler's business, not the sweep's; rows that never launched or
    # already finished are left alone.
    assert db.get_job(db_path, "ovh-1")["status"] == "running"
    assert db.get_job(db_path, "loc-done")["status"] == "succeeded"
    assert db.get_job(db_path, "loc-plan")["status"] == "planned"


def test_recipe_inputs_must_be_single_tokens(db_path: Path):
    """Inputs substitute into reviewed CLI commands: anything that is not one
    safe token is rejected before the entrypoint is ever resolved."""
    settings = _FakeSettings(db_path)
    for bad in ["two tokens", "x; rm -rf /", "$(boom)", "a'b", "--execute", "tag\ntwo", "tag\n"]:
        with pytest.raises(plan.PlanError, match="single token"):
            plan.plan_job(settings, "eval-crisis-v2", {"tag": bad}, {})
    # Non-string values are rejected too, not silently str()ed.
    with pytest.raises(plan.PlanError, match="single token"):
        plan.plan_job(settings, "eval-crisis-v2", {"tag": 30}, {})
    good = plan.plan_job(settings, "eval-crisis-v2", {"tag": "v30-eval-v2"}, {})
    assert good["command"].endswith("--tag v30-eval-v2 --suite v2")


def test_terminal_rows_survive_a_backend_import_refresh(db_path):
    """Thread on db.py: a backend import must not resurrect a terminal row.

    ``sync`` mirrors what OVH/W&B list back into the registry through
    ``upsert_job``; the refresh side of the upsert is terminal-guarded, so a
    late refresh cannot rewrite an outcome that was already written.
    """
    db.upsert_job(
        db_path,
        {"job_id": "ovh-x", "kind": "eval_paired", "source": "ovhai", "status": "cancelled", "backend_id": "u2"},
    )
    db.upsert_job(
        db_path,
        {"job_id": "ovh-x", "kind": "eval_paired", "source": "ovhai", "status": "running", "backend_id": "u2"},
    )
    assert db.get_job(db_path, "ovh-x")["status"] == "cancelled"
    # The guard is on the refresh side only: a terminal row imports fine the
    # first time.
    db.upsert_job(db_path, {"job_id": "ovh-import", "kind": "eval_paired", "source": "ovhai", "status": "succeeded"})
    assert db.get_job(db_path, "ovh-import")["status"] == "succeeded"


def test_resolve_command_preserves_whitespace_inside_a_quoted_value():
    """Thread on plan.py: normalization must be shell-aware after quoting.

    A plain whitespace split collapses the repeated space inside a quoted
    value (`a  b` became `a b`), silently changing what the recipe runs. The
    resolved command is parsed and re-serialized instead, exactly the way the
    runner builds the argv.
    """
    recipe = recipes.Recipe(
        id="resolve-probe",
        kind="other",
        description="probe",
        backend="local",
        entrypoint="python -m training.probe --note <note>",
        overrides={"note": {"type": "str"}},
    )
    assert plan._resolve_command(recipe, {}, {"note": "a  b"}) == "python -m training.probe --note 'a  b'"
    assert plan._resolve_command(recipe, {}, {"note": "b"}) == "python -m training.probe --note b"


def test_log_dir_failure_lands_in_a_terminal_state(db_path, monkeypatch):
    """A preparation failure must fail the job, not strand it at `submitted`.

    mkdir ran before the guarded spawn block: a permission error or a full disk
    killed the daemon thread with the row still `submitted`, holding a
    concurrency slot forever with no process and no provenance.
    """
    settings = _settings(db_path)
    db.upsert_job(
        db_path, {"job_id": "job-mkdir", "kind": "build_release", "source": "local", "name": "r", "status": "queued"}
    )
    row = db.get_job(db_path, "job-mkdir")
    recorded: list[str] = []
    monkeypatch.setattr(runner, "_record_terminal_outcome", lambda _s, job_id: recorded.append(job_id))

    def boom(_self, *_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(runner.Path, "mkdir", boom)
    monkeypatch.setattr(runner.threading, "Thread", ImmediateThread)
    runner.submit(settings, "job-mkdir", row, "python -m training.build_v30_release", "local")
    final = db.get_job(db_path, "job-mkdir")
    assert final["status"] == "failed"
    assert "spawn failed" in (final["error"] or "")
    assert final["finished_at"] is not None
    assert recorded == ["job-mkdir"]


@pytest.mark.parametrize("status", ["queued", "staging", "submitted", "running", "collecting"])
def test_submit_counts_every_in_flight_phase_against_the_cap(server_env, status):
    """A job holds its slot from `queued` through `collecting`, so every one of
    those phases counts against `max_concurrent_jobs`."""
    server, cfg, spawned = server_env
    db.upsert_job(
        cfg.db_path,
        {"job_id": "job-phase", "kind": "eval_paired", "source": "local", "name": "b", "status": status},
    )
    planned = server.plan_job("curate-sft-v30", {"corpus": "x"}, {"dry_run": True}, "operator")
    result = server.submit_job(plan_hash=planned["plan_hash"], requested_by="operator")
    assert result["ok"] is False
    assert "max concurrent jobs reached" in result["error"]
    assert spawned == []


def test_a_full_page_listing_does_not_mark_jobs_lost(db_path, monkeypatch):
    """A truncated backend listing is not proof that a job vanished."""
    settings = _settings(db_path)
    db.upsert_job(
        db_path,
        {"job_id": "ovh-busy", "kind": "eval_paired", "source": "ovhai", "status": "running", "backend_id": "old-job"},
    )
    db.upsert_job(
        db_path,
        {"job_id": "wand-gone", "kind": "train_sft", "source": "wandb", "status": "running", "backend_id": "r-old"},
    )
    monkeypatch.setattr(
        sync,
        "sync_ovhai",
        lambda _s, **_k: {
            "source": "ovhai",
            "configured": True,
            "imported": 0,
            "total_seen": 50,
            "limit": 50,
            "backend_ids": [f"job-{i}" for i in range(50)],
        },
    )
    monkeypatch.setattr(
        sync,
        "sync_wandb",
        lambda _s, **_k: {
            "source": "wandb",
            "configured": True,
            "imported": 0,
            "total_seen": 1,
            "limit": 20,
            "backend_ids": ["r-new"],
        },
    )
    monkeypatch.setattr(
        sync,
        "sync_r2_evals",
        lambda _s: {"source": "r2", "configured": False, "imported": 0, "total_seen": 0, "backend_ids": []},
    )

    result = sync.sync_all(settings)

    # The full-page listing is not trusted: the job just past the window stays.
    assert db.get_job(db_path, "ovh-busy")["status"] == "running"
    assert result["ovhai"]["listing_may_be_truncated"] is True
    # The provably complete listing still reconciles its vanished job.
    assert db.get_job(db_path, "wand-gone")["status"] == "failed"
    assert "ovhai" not in result["reconcile"]["checked_sources"]
    assert result["reconcile"]["checked_sources"] == ["wandb"]


def test_a_killed_process_is_reaped():
    """The terminate-kill path returns without _monitor, so it must wait."""
    calls: list[str] = []

    class _Proc:
        waited = False

        def terminate(self):
            calls.append("terminate")

        def wait(self, *_args, **_kwargs):
            calls.append("wait")
            if not self.waited:
                self.waited = True
                raise subprocess.TimeoutExpired(cmd="job", timeout=10)
            return 0

        def kill(self):
            calls.append("kill")

    runner._terminate_proc(_Proc())
    assert calls == ["terminate", "wait", "kill", "wait"]


def test_a_locked_default_survives_the_plan_roundtrip(db_path, monkeypatch):
    """The cached spec carries locked defaults; submit re-plans with them."""
    recipe = recipes.Recipe(
        id="locked-probe",
        kind="other",
        description="probe",
        backend="local",
        entrypoint="python -m training.probe --lr <learning_rate>",
        overrides={"learning_rate": {"type": "float", "locked": True, "default": 0.0001}},
    )
    monkeypatch.setattr(recipes, "load_recipes", lambda: [recipe])
    settings = _FakeSettings(db_path)
    planned = plan.plan_job(settings, "locked-probe", {}, {})
    assert planned["spec"]["overrides"] == {"learning_rate": 0.0001}
    again = plan.plan_job(settings, "locked-probe", {}, planned["spec"]["overrides"])
    assert again["plan_hash"] == planned["plan_hash"]


def test_draft_validation_reports_unhashable_fields_as_errors():
    """A draft with a list or dict where an enum belongs is an error, not a crash."""
    errors = recipes.validate_recipe_entry(
        {
            **VALID_ENTRY,
            "kind": ["train_sft"],
            "backend": {"name": "ovhai"},
            "cost_class": ["cpu-minute"],
        }
    )
    assert any("kind" in e for e in errors)
    assert any("backend" in e for e in errors)
    assert any("cost_class" in e for e in errors)


def test_the_marker_is_forgotten_when_the_slot_wait_times_out(db_path, monkeypatch):
    """The single-flight timeout path returns before the try/finally that
    clears the marker, so it must clear it itself."""
    settings = _settings(db_path)
    db.upsert_job(
        db_path, {"job_id": "job-sf2", "kind": "eval_paired", "source": "local", "name": "r", "status": "queued"}
    )
    row = db.get_job(db_path, "job-sf2")
    monkeypatch.setattr(runner, "_record_terminal_outcome", lambda *_a, **_k: None)

    class BusyLock:
        def acquire(self, timeout=None):
            assert timeout == _SINGLE_FLIGHT_TIMEOUT
            return False

        def release(self):
            pass

    monkeypatch.setattr(runner, "_lock_for", lambda _backend: BusyLock())
    monkeypatch.setattr(runner.threading, "Thread", ImmediateThread)
    runner._mark_cancelled("job-sf2")
    runner.submit(settings, "job-sf2", row, "python -m training.build_v30_release", "local")
    assert db.get_job(db_path, "job-sf2")["status"] == "failed"
    assert runner._is_cancelled("job-sf2") is False


def test_the_marker_is_forgotten_when_an_external_row_is_cancelled(db_path, monkeypatch):
    """An imported row has no runner thread to clear its marker in `finally`."""
    settings = _settings(db_path)
    db.upsert_job(
        db_path, {"job_id": "job-ext", "kind": "train_sft", "source": "wandb", "name": "r", "status": "running"}
    )
    monkeypatch.setattr(runner, "_record_terminal_outcome", lambda *_a, **_k: None)
    assert runner.cancel(settings, "job-ext") is True
    assert db.get_job(db_path, "job-ext")["status"] == "cancelled"
    assert runner._is_cancelled("job-ext") is False


def test_concurrent_admits_cannot_both_pass_the_cap(db_path):
    """The count and the insert share one IMMEDIATE transaction."""
    results: list[bool] = []
    barrier = threading.Barrier(2)
    in_flight = ("queued", "staging", "submitted", "running", "collecting")

    def try_admit(index: int) -> None:
        barrier.wait()
        results.append(
            db.admit_job_under_cap(
                db_path,
                {
                    "job_id": f"job-race-{index}",
                    "kind": "eval_paired",
                    "source": "local",
                    "name": "r",
                    "status": "queued",
                },
                in_flight=in_flight,
                cap=1,
            )
        )

    threads = [threading.Thread(target=try_admit, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [False, True]
    assert db.count_jobs(db_path, in_flight=("queued",)) == 1


def test_the_cap_counts_rows_past_one_page(db_path):
    """A paged count reads 500 and admits; the real count reads 501 and refuses."""
    for i in range(501):
        db.upsert_job(db_path, {"job_id": f"job-pg-{i}", "kind": "eval_paired", "source": "local", "status": "running"})
    assert db.count_jobs(db_path, in_flight=("running",)) == 501
    assert not db.admit_job_under_cap(
        db_path,
        {"job_id": "job-pg-new", "kind": "eval_paired", "source": "local", "status": "queued"},
        in_flight=("running",),
        cap=501,
    )


def test_a_locked_default_rejects_equal_values_of_the_wrong_type(db_path, monkeypatch):
    recipe = recipes.Recipe(
        id="locked-probe",
        kind="other",
        description="probe",
        backend="local",
        entrypoint="python -m training.probe --flag <flag>",
        overrides={"flag": {"type": "bool", "locked": True, "default": False}},
    )
    monkeypatch.setattr(recipes, "load_recipes", lambda: [recipe])
    with pytest.raises(plan.PlanError, match="is locked"):
        plan.plan_job(_FakeSettings(db_path), "locked-probe", {}, {"flag": 0})


def test_a_retried_case_counts_once_last_record_wins():
    """A resumed run's duplicate record is the retry, not a second failure."""
    body = "\n".join(
        [
            json.dumps({"case_id": "c1", "arm_id": "sft", "status": "error", "response": None}),
            json.dumps(
                {
                    "case_id": "c1",
                    "arm_id": "sft",
                    "status": "success",
                    "response": "ok",
                    "state_metrics": [{"passed": True}],
                }
            ),
            json.dumps(
                {
                    "case_id": "c2",
                    "arm_id": "sft",
                    "status": "success",
                    "response": "ok",
                    "state_metrics": [{"passed": True}],
                }
            ),
        ]
    ).encode()
    parsed = backends._parse_results_body(body)
    assert parsed["arm_pass_rates"]["sft"] == {"passed": 2, "total": 2, "rate": 1.0}


def test_draft_validation_rejects_non_mapping_fields_and_non_finite_costs():
    errors = recipes.validate_recipe_entry(
        {
            **VALID_ENTRY,
            "inputs": ["tag"],
            "outputs": "outputs/",
            "invariants": [True],
            "est_cost_usd": "nan",
        }
    )
    assert any("inputs must be a mapping" in e for e in errors)
    assert any("outputs must be a mapping" in e for e in errors)
    assert any("invariants must be a mapping" in e for e in errors)
    assert any("est_cost_usd" in e and "finite" in e for e in errors)


def test_a_value_error_during_spawn_lands_in_a_terminal_state(db_path, monkeypatch):
    """Popen raises ValueError for an embedded NUL; that must not strand the row."""
    settings = _settings(db_path)
    db.upsert_job(
        db_path,
        {"job_id": "job-nul", "kind": "build_release", "source": "local", "name": "r", "status": "queued"},
    )
    row = db.get_job(db_path, "job-nul")
    recorded: list[str] = []
    monkeypatch.setattr(runner, "_record_terminal_outcome", lambda _s, job_id: recorded.append(job_id))

    def boom(*_a, **_k):
        raise ValueError("embedded null byte")

    monkeypatch.setattr(runner.subprocess, "Popen", boom)
    monkeypatch.setattr(runner.threading, "Thread", ImmediateThread)
    runner.submit(settings, "job-nul", row, "python -m training.build_v30_release", "local")
    final = db.get_job(db_path, "job-nul")
    assert final["status"] == "failed"
    assert "spawn failed" in (final["error"] or "")
    assert recorded == ["job-nul"]


def test_admission_initializes_the_registry_like_every_other_call(db_path):
    assert db.admit_job_under_cap(
        db_path,
        {"job_id": "job-first", "kind": "eval_paired", "source": "local", "status": "queued"},
        in_flight=("queued",),
        cap=0,
    )
    assert (os.stat(db_path).st_mode & 0o777) == 0o600
    journal = sqlite3.connect(str(db_path)).execute("PRAGMA journal_mode").fetchone()[0]
    assert journal == "wal"


def test_an_unhashable_case_id_is_counted_as_is():
    body = json.dumps({"case_id": ["x"], "arm_id": "sft", "state_metrics": [{"passed": True}]}).encode()
    parsed = backends._parse_results_body(body)
    assert parsed["arm_pass_rates"]["sft"] == {"passed": 1, "total": 1, "rate": 1.0}


def test_draft_validation_rejects_boolean_and_overflowing_costs():
    errors = recipes.validate_recipe_entry({**VALID_ENTRY, "est_cost_usd": False})
    assert any("est_cost_usd" in e for e in errors)
    errors = recipes.validate_recipe_entry({**VALID_ENTRY, "est_cost_usd": 10**400})
    assert any("est_cost_usd" in e for e in errors)


def test_draft_validation_rejects_a_malformed_timeout():
    errors = recipes.validate_recipe_entry({**VALID_ENTRY, "timeout": "soon"})
    assert any("timeout" in e for e in errors)


def test_draft_validation_rejects_bad_override_defaults():
    errors = recipes.validate_recipe_entry(
        {
            **VALID_ENTRY,
            "overrides": {
                "epochs": {"type": "int", "min": 1, "default": 0},
                "label": {"type": "str", "default": 5},
                "flag": {"type": "bool", "default": "yes"},
                "rate": {"type": "float", "default": True},
            },
        }
    )
    assert any("epochs" in e and "default" in e for e in errors)
    assert any("label" in e and "default" in e for e in errors)
    assert any("flag" in e and "default" in e for e in errors)
    assert any("rate" in e and "default" in e for e in errors)


def test_a_recipe_timeout_kills_a_hung_job(db_path, monkeypatch):
    db.upsert_job(db_path, {"job_id": "job-hang", "kind": "build_release", "source": "local", "status": "running"})
    killed: list[str] = []

    class _Proc:
        returncode = None

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd="job", timeout=timeout)
            return None

        def kill(self):
            self.returncode = -9
            killed.append("kill")

    monkeypatch.setattr(runner.time, "sleep", lambda _s: None)
    monkeypatch.setattr(runner, "_record_terminal_outcome", lambda *_a, **_k: None)
    runner._monitor(_settings(db_path), "job-hang", _Proc(), "/tmp/x.log", timeout_seconds=3600.0)
    job = db.get_job(db_path, "job-hang")
    assert job["status"] == "failed"
    assert job["error"] == "exceeded its recipe timeout"
    assert killed == ["kill"]


def test_timeout_strings_parse_to_seconds():
    assert runner._timeout_seconds({"timeout": "2h"}) == 7200.0
    assert runner._timeout_seconds({"timeout": "30m"}) == 1800.0
    assert runner._timeout_seconds({"timeout": "2d"}) == 172800.0
    assert runner._timeout_seconds({"timeout": ""}) is None
    assert runner._timeout_seconds({}) is None
    assert runner._timeout_seconds({"timeout": "soon"}) is None
