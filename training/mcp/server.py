"""FastMCP server for the training-infra control plane (PIX-5000, Phase 0).

Read-only observability: health, recipe catalog, job registry queries,
eval results, and W&B metrics. Submit/plan tools arrive in Phase 1+.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

from training.mcp import backends, db, plan, recipes, runner, sync
from training.mcp.config import Settings, load_settings
from training.mcp.plan import PlanError

_STATE: dict[str, Any] = {}
_HEALTH_TTL_S = 60.0
_RECONCILE_INTERVAL_S = 60.0


def settings() -> Settings:
    cached = _STATE.get("settings")
    if cached is None:
        cached = load_settings()
        _STATE["settings"] = cached
    return cached


def _build_auth():
    token = settings().mcp_token
    if not token:
        return None
    return StaticTokenVerifier(
        tokens={token: {"client_id": "training-infra-client", "scopes": [], "sub": "agent"}},
    )


@asynccontextmanager
async def lifespan(_app):
    s = settings()
    # Reconcile on startup.
    with contextlib.suppress(Exception):
        sync.sync_all(s)
    # Periodic reconciler.
    async def _loop():
        while True:
            await asyncio.sleep(_RECONCILE_INTERVAL_S)
            with contextlib.suppress(Exception):
                sync.sync_all(settings())
    task = asyncio.create_task(_loop())
    try:
        yield
    finally:
        task.cancel()


mcp = FastMCP("pixelated-training-infra", auth=_build_auth(), lifespan=lifespan)
_STATE.setdefault("settings", load_settings())


def _guard(fn, *args, **kwargs) -> dict[str, Any]:
    """Run a backend call and normalize failures into tool-friendly errors."""
    try:
        return {"ok": True, "result": fn(*args, **kwargs)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


@mcp.tool
def health() -> dict[str, Any]:
    """Probe every backend the training pipeline depends on.

    Result is cached for 60s — backend probes (esp. W&B) are slow and
    agents poll this before pipeline runs.
    """
    now = time.monotonic()
    cached = _STATE.get("health")
    if cached is not None and now - cached[0] < _HEALTH_TTL_S:
        report = dict(cached[1])
        report["cached"] = True
        return report
    report = _health_probe()
    _STATE["health"] = (now, report)
    return report


def _health_probe() -> dict[str, Any]:
    s = settings()
    started = time.monotonic()

    ovhai_ok = backends.ovhai_configured(s)
    ovhai_detail: dict[str, Any] = {"configured": ovhai_ok}
    if ovhai_ok:
        probe = _guard(backends.ovhai_list_jobs, s, limit=1)
        ovhai_detail["reachable"] = probe["ok"]
        if not probe["ok"]:
            ovhai_detail["error"] = probe["error"]

    wandb_ok = backends.wandb_configured(s)
    wandb_detail: dict[str, Any] = {"configured": wandb_ok}
    if wandb_ok:
        probe = _guard(backends.wandb_runs, s, limit=1)
        wandb_detail["reachable"] = probe["ok"]
        if not probe["ok"]:
            wandb_detail["error"] = probe["error"]

    r2_ok = backends.r2_configured(s)
    r2_detail: dict[str, Any] = {"configured": r2_ok}
    if r2_ok:
        probe = _guard(backends.r2_list_eval_tags, s)
        r2_detail["reachable"] = probe["ok"]
        if probe["ok"]:
            r2_detail["eval_tags"] = len(probe["result"])
        else:
            r2_detail["error"] = probe["error"]

    try:
        db.list_jobs(s.db_path, limit=1)
        db_ok = True
    except Exception:
        db_ok = False

    return {
        "service": "pixelated-training-infra",
        "phase": 0,
        "read_only": True,
        "latency_ms": round((time.monotonic() - started) * 1000, 1),
        "checks": {
            "ovhai": ovhai_detail,
            "wandb": wandb_detail,
            "r2": r2_detail,
            "registry_db": {"ok": db_ok, "path": str(s.db_path)},
        },
        "healthy": all(
            d.get("reachable", d.get("ok", d.get("configured", False))) for d in (ovhai_detail, wandb_detail, r2_detail)
        )
        and db_ok,
    }


@mcp.tool
def list_recipes(kind: str | None = None) -> dict[str, Any]:
    """List registered training recipes. Agents select; they never define."""
    try:
        all_recipes = recipes.load_recipes()
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if kind:
        all_recipes = [r for r in all_recipes if r.kind == kind]
    return {
        "ok": True,
        "count": len(all_recipes),
        "recipes": [r.to_dict() for r in all_recipes],
    }


@mcp.tool
def get_recipe(recipe_id: str) -> dict[str, Any]:
    """Fetch one recipe by id."""
    try:
        for recipe in recipes.load_recipes():
            if recipe.id == recipe_id:
                return {"ok": True, "recipe": recipe.to_dict()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": False, "error": f"unknown recipe: {recipe_id}"}


@mcp.tool
def sync_jobs() -> dict[str, Any]:
    """Reconcile live backends (ovhai job list) into the job registry."""
    return {"ok": True, "result": sync.sync_all(settings())}


@mcp.tool
def list_jobs(
    status: str | None = None,
    kind: str | None = None,
    source: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Query the job registry. Run sync_jobs first for fresh backend state."""
    s = settings()
    jobs = db.list_jobs(s.db_path, status=status, kind=kind, source=source, limit=limit)
    return {"ok": True, "count": len(jobs), "jobs": jobs}


@mcp.tool
def get_job(job_id: str) -> dict[str, Any]:
    """Fetch one registry job by id (e.g. 'ovh-cca2876a-2a78')."""
    job = db.get_job(settings().db_path, job_id)
    if job is None:
        return {"ok": False, "error": f"unknown job: {job_id}"}
    return {"ok": True, "job": job}


@mcp.tool
def plan_job(
    recipe: str,
    inputs: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a recipe into an exact, costed, runnable spec (no side effects)."""
    s = settings()
    try:
        result = plan.plan_job(s, recipe, inputs, overrides)
    except PlanError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, **result}


@mcp.tool
def submit_job(
    plan_hash: str | None = None,
    recipe: str | None = None,
    inputs: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
    requested_by: str = "agent",
    force: bool = False,
) -> dict[str, Any]:
    """Submit a planned job. Binds to a recent plan_job via plan_hash unless force=true."""
    s = settings()
    # Idempotency.
    if idempotency_key:
        existing = db.find_by_idempotency(s.db_path, idempotency_key)
        if existing is not None:
            return {"ok": True, "job_id": existing["job_id"], "status": existing["status"], "idempotent": True}
    # Plan binding.
    cached = None
    if plan_hash:
        cached = db.get_cached_plan(s.db_path, plan_hash)
        if cached is None and not force:
            return {"ok": False, "error": "plan_hash not found or expired; re-run plan_job"}
    if cached is not None:
        if recipe is None:
            recipe = cached["recipe_id"]
        if inputs is None:
            inputs = cached["spec"].get("inputs", {})
        if overrides is None:
            overrides = cached["spec"].get("overrides", {})
        if not force and recipe != cached["recipe_id"]:
            return {"ok": False, "error": "plan_hash does not match recipe"}
    if recipe is None:
        return {"ok": False, "error": "recipe required (or a valid plan_hash)"}
    if not force and not plan_hash:
        return {"ok": False, "error": "plan_hash required (run plan_job first) unless force=true"}
    # Resolve plan (recompute spec) for cost/command.
    try:
        planned = plan.plan_job(s, recipe, inputs, overrides)
    except PlanError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    spec = planned["spec"]
    cost = float(planned["est_cost_usd"] or 0.0)
    # Budget caps.
    running = db.list_jobs(s.db_path, status="running")
    queued = db.list_jobs(s.db_path, status="queued")
    active_count = len(running) + len(queued)
    if active_count >= s.max_concurrent_jobs and s.max_concurrent_jobs > 0:
        # Allow queueing but refuse if already at cap for queued? We still queue; but cap daily.
        pass
    # daily job cap
    today = datetime.now(UTC).date().isoformat()
    all_jobs = db.list_jobs(s.db_path, limit=500)
    today_count = sum(1 for j in all_jobs if (j.get("created_at") or "").startswith(today) and j.get("requested_by"))
    if today_count >= s.daily_job_cap:
        return {"ok": False, "error": f"daily job cap reached ({s.daily_job_cap})"}
    # monthly cost cap
    active_cost = sum(float(j.get("cost_estimate") or 0) for j in running + queued)
    if active_cost + cost > s.monthly_cost_cap_usd:
        return {"ok": False, "error": (
            f"monthly cost cap would be exceeded ({s.monthly_cost_cap_usd})"
        )}
    # Single-flight handled in runner; create job row.
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    job_row = {
        "job_id": job_id,
        "kind": spec["kind"],
        "source": spec["backend"],
        "name": spec["recipe_id"],
        "status": "queued",
        "stage": "queued",
        "idempotency_key": idempotency_key,
        "spec": spec,
        "spec_hash": planned["spec_hash"],
        "plan_hash": planned["plan_hash"],
        "cost_class": spec["cost_class"],
        "cost_estimate": cost,
        "requested_by": requested_by,
    }
    db.upsert_job(s.db_path, job_row)
    # Spawn.
    runner.submit(s, job_id, job_row, spec["command"], spec["backend"])
    return {"ok": True, "job_id": job_id, "status": "queued", "plan_hash": planned["plan_hash"]}


@mcp.tool
def cancel_job(job_id: str) -> dict[str, Any]:
    """Best-effort cancel: terminate subprocess, mark cancelled, attempt backend cancel."""
    job = db.get_job(settings().db_path, job_id)
    if job is None:
        return {"ok": False, "error": f"unknown job: {job_id}"}
    ok = runner.cancel(settings(), job_id)
    return {"ok": ok, "job_id": job_id}

@mcp.tool
def get_job_logs(job_id: str, tail: int = 200) -> dict[str, Any]:
    """Fetch a tail of backend logs for a job (ovhai jobs only in Phase 0)."""
    job = db.get_job(settings().db_path, job_id)
    if job is None:
        return {"ok": False, "error": f"unknown job: {job_id}"}
    if job.get("source") != "ovhai" or not job.get("backend_id"):
        return {"ok": False, "error": "logs only available for ovhai jobs in Phase 0"}
    return {"ok": True, "logs": backends.ovhai_job_logs(job["backend_id"], tail=tail)}


@mcp.tool
def job_metrics(run_id: str) -> dict[str, Any]:
    """W&B metric summary for a run id (from wandb_runs or a train job)."""
    return _guard(backends.wandb_run_metrics, settings(), run_id)


@mcp.tool
def list_wandb_runs(limit: int = 20) -> dict[str, Any]:
    """Recent W&B runs in the configured project."""
    return _guard(backends.wandb_runs, settings(), limit=limit)


@mcp.tool
def eval_results(tag: str) -> dict[str, Any]:
    """Eval outputs for an R2 tag like 'v30-eval-v2': manifests + result previews + deltas."""
    return _guard(backends.r2_eval_summary, settings(), tag)


@mcp.tool
def list_eval_tags() -> dict[str, Any]:
    """All eval result tags present under evals/ in the training bucket."""
    return _guard(backends.r2_list_eval_tags, settings())


def main() -> None:
    s = settings()
    mcp.run(transport="streamable-http", host=s.host, port=s.port, path="/mcp")


if __name__ == "__main__":
    main()
