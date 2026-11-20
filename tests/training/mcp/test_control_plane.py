"""Unit tests for the training-infra MCP registry, plan binding, and gate.

These run without the service, the network, or real credentials: they exercise
the pure logic that keeps the control plane safe — recipe validation, the
plan/submit binding, the audit log, and the eval gate's fail-closed behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

AI_ROOT = Path(__file__).resolve().parents[3]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from training.mcp import backends, db, plan, recipes  # noqa: E402

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
    good = VALID_ENTRY
    bad = tmp_path / "recipes.yaml"
    bad.write_text(
        "recipes:\n"
        "  - id: same\n    kind: eval_paired\n    description: d\n    backend: local\n    entrypoint: python -m x\n"
        "  - id: same\n    kind: eval_paired\n    description: d\n    backend: local\n    entrypoint: python -m x\n",
        encoding="utf-8",
    )
    del good
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
        {"tag": "v30-eval-v2", "suite": "v2"},
        None,
    )
    second = plan.plan_job(
        settings,
        "eval-crisis-v2",
        {"tag": "v30-eval-v2", "suite": "v2"},
        None,
    )
    assert first["plan_hash"] == second["plan_hash"]
    assert first["spec_hash"] == second["spec_hash"]
    assert first["command"].startswith("python -m training.ovh_v22_eval.submit")
    assert "v30-eval-v2" in first["command"]
    assert (
        first["plan_hash"]
        != plan.plan_job(settings, "eval-crisis-v2", {"tag": "other-tag", "suite": "v2"}, None)["plan_hash"]
    )


def test_plan_job_rejects_missing_inputs_and_unknown_recipe(db_path: Path):
    settings = _FakeSettings(db_path)
    with pytest.raises(plan.PlanError, match="missing required input"):
        plan.plan_job(settings, "eval-crisis-v2", {"tag": "t"}, None)
    with pytest.raises(plan.PlanError, match="unknown recipe"):
        plan.plan_job(settings, "no-such-recipe", {}, None)


def test_plan_job_rejects_locked_and_badly_typed_overrides(db_path: Path):
    settings = _FakeSettings(db_path)
    with pytest.raises(plan.PlanError, match="locked"):
        plan.plan_job(
            settings,
            "train-sft-latest",
            {"manifest": "m", "launch_config": "c"},
            {"learning_rate": 1e-4},
        )
    with pytest.raises(plan.PlanError, match="epochs"):
        plan.plan_job(
            settings,
            "train-sft-latest",
            {"manifest": "m", "launch_config": "c"},
            {"epochs": 99},
        )


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
