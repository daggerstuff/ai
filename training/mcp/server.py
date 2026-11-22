"""FastMCP server for the training-infra control plane (PIX-5000).

Read-only observability (health, recipe catalog, job registry, eval results,
W&B metrics) plus the control plane: ``plan_job`` resolves a reviewed recipe
into an exact costed spec, ``submit_job`` binds to that plan and launches it
under budget caps, and ``cancel_job`` stops it. ``validate_recipe`` checks a
draft recipe before a human lands it by PR, and ``eval_gate`` decides whether
eval results clear the promotion gate. Every mutating action is recorded in the
registry's ``audit_log`` table.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from fastmcp.server.dependencies import get_http_request

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


def _remote_addr() -> str | None:
    """Best-effort caller address for the audit trail (HTTP transport only)."""
    try:
        request = get_http_request()
        client = getattr(request, "client", None)
        return getattr(client, "host", None)
    except Exception:
        return None


def audit(
    action: str,
    *,
    actor: str | None = None,
    job_id: str | None = None,
    recipe_id: str | None = None,
    spec_hash: str | None = None,
    plan_hash: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """Append an audit row, never letting an audit failure break the action."""
    with contextlib.suppress(Exception):
        db.write_audit(
            settings().db_path,
            action=action,
            actor=actor,
            job_id=job_id,
            recipe_id=recipe_id,
            spec_hash=spec_hash,
            plan_hash=plan_hash,
            detail=detail,
            remote_addr=_remote_addr(),
        )


def _build_auth() -> StaticTokenVerifier | None:
    token = settings().mcp_token
    if not token:
        return None
    return StaticTokenVerifier(
        tokens={token: {"client_id": "training-infra-client", "scopes": [], "sub": "agent"}},
    )


@asynccontextmanager
async def lifespan(_app: Any) -> AsyncIterator[None]:
    s = settings()
    # Reconcile on startup.
    with contextlib.suppress(Exception):
        sync.sync_all(s)
    # Expire plans left over from previous runs.
    with contextlib.suppress(Exception):
        db.prune_plans(s.db_path, max_age_seconds=s.plan_ttl_seconds)

    # Periodic reconciler.
    async def _loop() -> None:
        while True:
            await asyncio.sleep(_RECONCILE_INTERVAL_S)
            with contextlib.suppress(Exception):
                sync.sync_all(settings())
            # Drop plan_cache rows past their TTL so the table does not grow
            # without bound and an expired approval cannot be resubmitted.
            with contextlib.suppress(Exception):
                current = settings()
                db.prune_plans(current.db_path, max_age_seconds=current.plan_ttl_seconds)

    task = asyncio.create_task(_loop())
    try:
        yield
    finally:
        task.cancel()


mcp = FastMCP("pixelated-training-infra", auth=_build_auth(), lifespan=lifespan)
_STATE.setdefault("settings", load_settings())


def _guard(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
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
        "phase": 4,
        "read_only": False,
        "audit": True,
        "eval_gate": True,
        "recipe_validation": True,
        "historical_import": True,
        "reconciler": {"active": True, "lost_backend": True},
        "foresight": {"enabled": s.foresight_enabled, "url": s.foresight_url},
        "budget_caps": {
            "max_concurrent_jobs": s.max_concurrent_jobs,
            "daily_job_cap": s.daily_job_cap,
            "monthly_cost_cap_usd": s.monthly_cost_cap_usd,
        },
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
def validate_recipe(entry: dict[str, Any]) -> dict[str, Any]:
    """Validate a *draft* recipe entry against the registry rules (P3.3).

    This is the proposal half of the recipe-addition workflow: an advisor can
    draft an entry and have it checked before a human opens the PR that lands
    it in ``recipes.yaml``. The MCP never writes the registry itself.
    """
    try:
        known = {r.id for r in recipes.load_recipes()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    errors = recipes.validate_recipe_entry(entry, known_ids=known)
    if errors:
        return {
            "ok": True,
            "valid": False,
            "errors": errors,
            "note": "Fix the errors, then land the entry via a reviewed PR to training/mcp/recipes.yaml.",
        }
    return {
        "ok": True,
        "valid": True,
        "errors": [],
        "note": (
            "Entry is valid; land it via a reviewed PR to training/mcp/recipes.yaml. "
            "See training/mcp/RECIPE_WORKFLOW.md."
        ),
    }


@mcp.tool
def sync_jobs() -> dict[str, Any]:
    """Reconcile every backend into the registry, then mark lost jobs.

    Imports recent ovhai jobs, W&B runs, and R2 eval tags; a registry row still
    ``running`` whose backend job is no longer listed becomes ``failed`` with
    ``error: backend_lost`` (PIX-5012). A backend that is unconfigured or whose
    listing failed is skipped, so an unreachable backend never looks like a
    vanished job.
    """
    return {"ok": True, "result": sync.sync_all(settings())}


@mcp.tool
def import_history(
    rows: list[dict[str, Any]] | None = None,
    include_backends: bool = False,
) -> dict[str, Any]:
    """Seed the registry from historical records (P0.10).

    Two sources:
    * ``rows`` — operator-supplied manual log entries. Each needs at minimum
      ``job_id``, ``kind``, and ``status``; the rest of the registry fields are
      accepted as-is (``spec``, ``artifact_refs``, ``cost_estimate``, …).
    * ``include_backends`` — additionally import what W&B/R2 can enumerate.
      R2 contributes one terminal ``succeeded`` row per ``evals/<tag>/`` tag,
      which is what makes pre-control-plane evals visible and lets
      ``eval_results`` line up with a registry row.

    ``ovhai`` is always imported by ``sync_jobs``; this tool exists for the
    history those listings cannot reconstruct. Idempotent: rows are upserted.
    """
    s = settings()
    results: dict[str, Any] = {}
    if rows:
        results["manual"] = sync.import_manual_rows(s, rows)
    if include_backends:
        for name, fn in (("wandb", sync.sync_wandb), ("r2", sync.sync_r2_evals)):
            try:
                results[name] = fn(s)
            except Exception as exc:
                results[name] = {"error": f"{type(exc).__name__}: {exc}"}
    if not rows and not include_backends:
        return {"ok": False, "error": "nothing to import: pass rows and/or include_backends=true"}
    return {"ok": True, "result": results}


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
    requested_by: str = "agent",
) -> dict[str, Any]:
    """Resolve a recipe into an exact, costed, runnable spec (no side effects)."""
    s = settings()
    try:
        result = plan.plan_job(s, recipe, inputs, overrides)
    except PlanError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    audit(
        "plan",
        actor=requested_by,
        recipe_id=recipe,
        spec_hash=str(result.get("spec_hash") or ""),
        plan_hash=str(result.get("plan_hash") or ""),
        detail={
            "est_cost_usd": result.get("est_cost_usd"),
            "cost_class": result.get("cost_class"),
        },
    )
    return {"ok": True, **result}


def count_toward_daily_cap(jobs: list[dict[str, Any]]) -> int:
    """Count jobs that consume the daily launch cap.

    Rows created before this date or by an actor that did not request a launch
    through the control plane (``import``) must not count: a single
    ``import_history(include_backends=True)`` sweep can mirror dozens of
    historical runs, and counting them would block every real submit until
    midnight.
    """
    today = datetime.now(UTC).date().isoformat()
    return sum(
        1
        for j in jobs
        if (j.get("created_at") or "").startswith(today)
        and j.get("requested_by")
        and j.get("requested_by") not in sync.NON_REQUESTING_ACTORS
    )


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
            audit(
                "submit.idempotent",
                actor=requested_by,
                job_id=existing["job_id"],
                recipe_id=existing.get("name"),
                plan_hash=existing.get("plan_hash"),
                detail={"idempotency_key": idempotency_key},
            )
            return {"ok": True, "job_id": existing["job_id"], "status": existing["status"], "idempotent": True}
    # Plan binding. When a plan_hash is supplied (and not force), the *approved*
    # plan is the authority: the caller's recipe/inputs/overrides must either be
    # omitted or repeat exactly what was approved, and the recomputed plan below
    # must still hash to the same value.
    #
    # Without this check a caller could plan a harmless dry run, show that to
    # the operator, and then submit the same plan_hash with a different override
    # — so the reviewed artifact never described the work that actually ran.
    cached = None
    if plan_hash:
        cached = db.get_cached_plan(s.db_path, plan_hash, max_age_seconds=s.plan_ttl_seconds)
        if cached is None and not force:
            audit(
                "submit.rejected",
                actor=requested_by,
                recipe_id=recipe,
                plan_hash=plan_hash,
                detail={"reason": "plan_hash not found or expired"},
            )
            return {"ok": False, "error": "plan_hash not found or expired; re-run plan_job"}
    if cached is not None and not force:
        approved = cached["spec"]
        # The cached spec records overrides with the recipe's defaults filled
        # in. Compare only the keys the caller supplied against the approved
        # values: a faithful repeat that omits defaults must pass, while any
        # supplied key whose value differs is a tampered spec. Expanding the
        # caller's dict from the live registry would judge that repeat against a
        # recipe that may have changed since approval (and could raise outside
        # the error boundary below).
        approved_overrides = approved.get("overrides") or {}
        mismatched = [
            name
            for name, supplied, approved_value in (
                ("recipe", recipe, cached["recipe_id"]),
                ("inputs", inputs, approved.get("inputs", {})),
            )
            if supplied is not None and supplied != approved_value
        ]
        changed_overrides: list[str] = []
        if overrides is not None:
            changed_overrides = sorted(
                name
                for name, value in overrides.items()
                if name not in approved_overrides or value != approved_overrides[name]
            )
            if changed_overrides:
                mismatched.append("overrides")
        if mismatched:
            audit(
                "submit.rejected",
                actor=requested_by,
                recipe_id=recipe or cached["recipe_id"],
                plan_hash=plan_hash,
                detail={
                    "reason": "submitted spec differs from the approved plan",
                    "fields": mismatched,
                    "overrides": changed_overrides,
                },
            )
            return {
                "ok": False,
                "error": f"submitted {', '.join(mismatched)} differ from the approved plan; re-run plan_job",
            }
    if cached is not None:
        if force:
            # force=true is the explicit escape hatch: the caller's spec wins,
            # defaulting to the cached plan only where they said nothing.
            if recipe is None:
                recipe = cached["recipe_id"]
            if inputs is None:
                inputs = cached["spec"].get("inputs", {})
            if overrides is None:
                overrides = cached["spec"].get("overrides", {})
        else:
            # Bound submit: run exactly the approved spec, never the caller's.
            recipe = cached["recipe_id"]
            inputs = cached["spec"].get("inputs", {})
            overrides = cached["spec"].get("overrides", {})
    if recipe is None:
        audit("submit.rejected", actor=requested_by, detail={"reason": "recipe required"})
        return {"ok": False, "error": "recipe required (or a valid plan_hash)"}
    if not force and not plan_hash:
        audit("submit.rejected", actor=requested_by, recipe_id=recipe, detail={"reason": "plan_hash required"})
        return {"ok": False, "error": "plan_hash required (run plan_job first) unless force=true"}
    # Resolve plan (recompute spec) for cost/command.
    try:
        planned = plan.plan_job(s, recipe, inputs, overrides)
    except PlanError as exc:
        audit("submit.rejected", actor=requested_by, recipe_id=recipe, detail={"reason": str(exc)})
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        audit("submit.rejected", actor=requested_by, recipe_id=recipe, detail={"reason": str(exc)})
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    spec = planned["spec"]
    if cached is not None and not force and planned["plan_hash"] != cached["plan_hash"]:
        # The approved plan is the authority: recomputing it from the live
        # registry must reproduce the same hash. A recipe edited inside the TTL
        # would otherwise launch commands the operator never reviewed.
        audit(
            "submit.rejected",
            actor=requested_by,
            recipe_id=spec["recipe_id"],
            plan_hash=plan_hash,
            detail={
                "reason": "plan changed since approval",
                "approved_plan_hash": cached["plan_hash"],
                "recomputed_plan_hash": planned["plan_hash"],
            },
        )
        return {
            "ok": False,
            "error": "the recipe changed since this plan was approved; re-run plan_job for a fresh approval",
        }
    cost = float(planned["est_cost_usd"] or 0.0)
    # Budget caps.
    running = db.list_jobs(s.db_path, status="running")
    queued = db.list_jobs(s.db_path, status="queued")
    active_count = len(running) + len(queued)
    if s.max_concurrent_jobs > 0 and active_count >= s.max_concurrent_jobs:
        # RUNBOOK documents this cap as "reject a submit that would exceed it",
        # and it is the only guard on how many jobs an agent can have in flight
        # at once. It used to be a bare `pass`, so the cap was advertised in
        # `health.budget_caps` but never enforced.
        audit(
            "submit.rejected",
            actor=requested_by,
            recipe_id=recipe,
            plan_hash=planned["plan_hash"],
            detail={
                "reason": "max concurrent jobs reached",
                "cap": s.max_concurrent_jobs,
                "active_count": active_count,
            },
        )
        return {
            "ok": False,
            "error": (
                f"max concurrent jobs reached ({s.max_concurrent_jobs}); wait for a running job to finish or cancel one"
            ),
        }
    # daily job cap — counts jobs a human or agent actually asked the control
    # plane to launch. Rows discovered by `import_history`/`sync_jobs` carry the
    # `import` actor and are excluded: a single backfill sweep would otherwise
    # consume the whole cap and block every real submit until midnight.
    #
    # Read through `db.cap_rows` rather than `db.list_jobs`, which pages at 500
    # rows: a bounded scan would silently under-count once the registry grows
    # past one page and let the cap be exceeded.
    now_utc = datetime.now(UTC)
    today_count = count_toward_daily_cap(db.cap_rows(s.db_path, since=now_utc.date().isoformat()))
    if today_count >= s.daily_job_cap:
        audit(
            "submit.rejected",
            actor=requested_by,
            recipe_id=recipe,
            plan_hash=planned["plan_hash"],
            detail={"reason": "daily job cap reached", "cap": s.daily_job_cap, "today_count": today_count},
        )
        return {"ok": False, "error": f"daily job cap reached ({s.daily_job_cap})"}
    # monthly cost cap — the month's real spend, not just what is active right
    # now. Summing only running+queued rows let a month of sequential jobs each
    # pass the check while the accumulated cost sailed past the cap. Imported
    # history is excluded for the same reason as the daily cap: it was not
    # launched (or paid for) by this control plane.
    month_start = now_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat(timespec="seconds")
    month_spend = sum(
        float(j.get("cost_estimate") or 0)
        for j in db.cap_rows(s.db_path, since=month_start)
        if j.get("requested_by") not in sync.NON_REQUESTING_ACTORS
    )
    if month_spend + cost > s.monthly_cost_cap_usd:
        audit(
            "submit.rejected",
            actor=requested_by,
            recipe_id=recipe,
            plan_hash=planned["plan_hash"],
            detail={
                "reason": "monthly cost cap would be exceeded",
                "cap": s.monthly_cost_cap_usd,
                "month_spend": month_spend,
                "new_cost": cost,
            },
        )
        return {"ok": False, "error": (f"monthly cost cap would be exceeded ({s.monthly_cost_cap_usd})")}
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
    audit(
        "submit",
        actor=requested_by,
        job_id=job_id,
        recipe_id=spec["recipe_id"],
        spec_hash=planned["spec_hash"],
        plan_hash=planned["plan_hash"],
        detail={
            "kind": spec["kind"],
            "backend": spec["backend"],
            "cost_class": spec["cost_class"],
            "est_cost_usd": cost,
            "gpu": spec.get("gpu"),
            "idempotency_key": idempotency_key,
        },
    )
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
    audit(
        "cancel",
        job_id=job_id,
        recipe_id=job.get("name"),
        spec_hash=job.get("spec_hash"),
        plan_hash=job.get("plan_hash"),
        detail={"ok": ok, "previous_status": job.get("status")},
    )
    return {"ok": ok, "job_id": job_id}


@mcp.tool
def list_audit(job_id: str | None = None, action: str | None = None, limit: int = 100) -> dict[str, Any]:
    """Query the append-only audit log (actor, action, recipe, hashes, ts).

    Filter by ``job_id`` and/or ``action``. Newest rows first.
    """
    rows = db.list_audit(settings().db_path, job_id=job_id, action=action, limit=limit)
    return {"ok": True, "count": len(rows), "entries": rows}


# --- Security posture (PIX-5037) -------------------------------------------


def security_report() -> dict[str, Any]:
    """Assess credential isolation, bind scope, and token enforcement."""
    s = settings()
    job_env = runner._build_env(s)
    unexpected = sorted(set(job_env) - runner.JOB_ENV_ALLOWLIST)
    issues: list[str] = []
    if not s.binding_is_loopback and not s.allow_remote:
        issues.append("service bound to non-loopback without TRAINING_MCP_ALLOW_REMOTE=1")
    if not s.mcp_token:
        issues.append("no bearer token configured; unauthenticated clients can submit jobs")
    if unexpected:
        issues.append(f"job subprocess env carries non-allowlisted keys: {', '.join(unexpected)}")
    return {
        "ok": True,
        "bind_host": s.host,
        "loopback_only": s.binding_is_loopback,
        "allow_remote": s.allow_remote,
        "auth_enabled": bool(s.mcp_token),
        "credential_isolation": {
            "job_env_source": "allowlisted process env + pipeline dotenv secrets (runner._build_env)",
            "allowlisted_keys": sorted(runner.JOB_ENV_ALLOWLIST),
            "non_allowlisted_keys_in_job_env": unexpected,
            "isolated": not unexpected,
        },
        "issues": issues,
        "secure": not issues,
    }


@mcp.tool
def security_status() -> dict[str, Any]:
    """Report the security posture: credential isolation, bind scope, token auth.

    Read-only. ``secure`` is true when the service is loopback-bound (or
    explicitly opted into remote), a bearer token is enforced, and the
    subprocess environment exposes no unrelated secrets.
    """
    return security_report()


@mcp.tool
def get_job_logs(job_id: str, tail: int = 200) -> dict[str, Any]:
    """Fetch a tail of logs for a job.

    Local/subprocess jobs read the captured ``state/logs/<job_id>.log`` that
    the runner wrote; backend jobs (ovhai) are queried live. The MCP-owned
    capture is preferred when present because it is the same stream the
    subprocess actually produced.
    """
    job = db.get_job(settings().db_path, job_id)
    if job is None:
        return {"ok": False, "error": f"unknown job: {job_id}"}
    captured = _read_captured_log(job_id, tail=tail)
    if captured is not None:
        return {"ok": True, "job_id": job_id, "source": "captured", "logs": captured}
    if job.get("source") == "ovhai" and job.get("backend_id"):
        return {
            "ok": True,
            "job_id": job_id,
            "source": "ovhai",
            "logs": backends.ovhai_job_logs(job["backend_id"], tail=tail),
        }
    return {
        "ok": False,
        "error": "no captured log and no backend log source for this job "
        "(job has not started, or its backend is not log-addressable)",
    }


def _read_captured_log(job_id: str, *, tail: int) -> str | None:
    """Return the last ``tail`` lines of a captured job log, if any."""
    if "/" in job_id or "\\" in job_id or job_id.startswith("."):
        return None
    path = settings().db_path.parent / "logs" / f"{job_id}.log"
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    limit = max(1, min(tail, 5000))
    return "\n".join(lines[-limit:])


@mcp.tool
def job_metrics(job_id: str) -> dict[str, Any]:
    """W&B metric summary for a registry job (PIX-5010).

    Accepts a registry ``job_id`` and resolves the W&B run id from the row's
    ``backend_id``/``backend_ids`` — falling back to treating the argument as a
    bare W&B run id for hand-launched runs. Unknown jobs and unconfigured W&B
    return a real error rather than an ``ok: true`` wrapper hiding one.
    """
    s = settings()
    job = db.get_job(s.db_path, job_id)
    run_id = job_id
    if job is not None:
        candidates = [job.get("backend_id"), *(job.get("backend_ids") or [])]
        run_id = next((str(c) for c in candidates if c), job_id)
        if job.get("status") in {"planned", "queued", "staging", "submitted"}:
            return {
                "ok": True,
                "job_id": job_id,
                "run_id": run_id or None,
                "status": job["status"],
                "metrics": None,
                "note": "job has not started; W&B metrics are unavailable while queued",
            }
    metrics = backends.wandb_run_metrics(s, run_id)
    if isinstance(metrics, dict) and metrics.get("error"):
        return {"ok": False, "job_id": job_id, "run_id": run_id, "error": metrics["error"]}
    return {"ok": True, "job_id": job_id, "run_id": run_id, "metrics": metrics}


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


@mcp.tool
def eval_gate(tag: str, min_delta: float | None = None, min_pass_rate: float | None = None) -> dict[str, Any]:
    """Decide whether the eval results under ``tag`` clear the promotion gate.

    Fails closed: missing grades, missing arms, or a backend error all block
    promotion. Promotion tools must consult this before moving a release.

    ``min_delta`` / ``min_pass_rate`` default to the operator policy in
    ``TRAINING_MCP_GATE_MIN_DELTA`` / ``TRAINING_MCP_GATE_MIN_PASS_RATE``
    (0.05 / 0.5). They are not zero, so a caller that omits them cannot turn the
    gate into a rubber stamp.
    """
    s = settings()
    resolved_delta = s.gate_min_delta if min_delta is None else min_delta
    resolved_rate = s.gate_min_pass_rate if min_pass_rate is None else min_pass_rate
    summary = _guard(backends.r2_eval_summary, s, tag)
    verdict = backends.evaluate_gate(summary, min_delta=resolved_delta, min_pass_rate=resolved_rate)
    audit(
        "eval.gate",
        actor="gate",
        recipe_id=None,
        detail={
            "tag": tag,
            "passed": verdict["passed"],
            "min_delta": resolved_delta,
            "min_pass_rate": resolved_rate,
            "reasons": verdict["reasons"][:5],
        },
    )
    return {"ok": True, "tag": tag, **verdict}


def main() -> None:
    s = settings()
    if not s.binding_is_loopback and not s.allow_remote:
        raise SystemExit(f"refusing to bind to non-loopback host {s.host!r} without TRAINING_MCP_ALLOW_REMOTE=1")
    mcp.run(transport="streamable-http", host=s.host, port=s.port, path="/mcp")


if __name__ == "__main__":
    main()
