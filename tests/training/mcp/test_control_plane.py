"""Unit tests for the training-infra MCP registry, plan binding, and gate.

These run without the service, the network, or real credentials: they exercise
the pure logic that keeps the control plane safe — recipe validation, the
plan/submit binding, the audit log, and the eval gate's fail-closed behaviour.
"""

from __future__ import annotations

import json
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

AI_ROOT = Path(__file__).resolve().parents[3]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from training.mcp import backends, db, foresight, plan, recipes, runner, sync  # noqa: E402
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


def test_results_parser_reports_no_grades_when_metrics_are_absent():
    row = json.dumps({"arm_id": "sft", "state_metrics": [], "response": "x"})
    parsed = backends._parse_results_body(row.encode())
    assert parsed["arm_pass_rates"] is None
    assert "raw outputs only" in parsed["note"]


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
