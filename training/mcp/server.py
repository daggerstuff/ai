"""FastMCP server for the training-infra control plane (PIX-5000, Phase 0).

Read-only observability: health, recipe catalog, job registry queries,
eval results, and W&B metrics. Submit/plan tools arrive in Phase 1+.
"""

from __future__ import annotations

import time
from typing import Any

from fastmcp import FastMCP

from training.mcp import backends, db, recipes, sync
from training.mcp.config import Settings, load_settings

mcp = FastMCP("pixelated-training-infra")
_STATE: dict[str, Any] = {}
_HEALTH_TTL_S = 60.0


def settings() -> Settings:
    cached = _STATE.get("settings")
    if cached is None:
        cached = load_settings()
        _STATE["settings"] = cached
    return cached


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
def job_metrics(run_id: str) -> dict[str, Any]:
    """W&B metric summary for a run id (from wandb_runs or a train job)."""
    return _guard(backends.wandb_run_metrics, settings(), run_id)


@mcp.tool
def list_wandb_runs(limit: int = 20) -> dict[str, Any]:
    """Recent W&B runs in the configured project."""
    return _guard(backends.wandb_runs, settings(), limit=limit)


@mcp.tool
def eval_results(tag: str) -> dict[str, Any]:
    """Eval outputs for an R2 tag like 'v30-eval-v2': manifests + result previews."""
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
