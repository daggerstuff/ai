"""Backend clients: OVH AI Jobs, W&B, and R2 result storage.

All functions are read-only in Phase 0. They return plain dicts so the MCP
tools stay thin and testable without live credentials.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
from typing import Any

import wandb
from training.launch import r2_sync
from training.mcp.config import Settings


def _run(cmd: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


# ---------------------------------------------------------------- OVH / ovhai


def ovhai_configured(settings: Settings) -> bool:
    return settings.ovhai_available


def ovhai_list_jobs(settings: Settings, *, limit: int = 50) -> list[dict[str, Any]]:
    """Recent ovhai jobs, newest first. Returns [] when the CLI is unusable."""
    if not ovhai_configured(settings):
        return []
    proc = _run(["ovhai", "job", "list", "--all", "-o", "json"], timeout=90)
    if proc.returncode != 0:
        raise RuntimeError(f"ovhai job list failed: {proc.stderr.strip()[:300]}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ovhai job list returned non-JSON output") from exc
    items = payload if isinstance(payload, list) else payload.get("jobs", [])
    return items[:limit]


def ovhai_job_detail(job_id: str) -> dict[str, Any] | None:
    proc = _run(["ovhai", "job", "get", job_id, "-o", "json"], timeout=60)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def ovhai_job_logs(job_id: str, *, tail: int = 200) -> str:
    proc = _run(["ovhai", "job", "logs", job_id], timeout=60)
    if proc.returncode != 0:
        return f"<log fetch failed: {proc.stderr.strip()[:200]}>"
    lines = proc.stdout.splitlines()
    return "\n".join(lines[-tail:])


def _ovhai_status(entry: dict[str, Any]) -> tuple[str, str]:
    """Map an ovhai job list entry to (registry status, backend state)."""
    status = entry.get("status") or {}
    state = str(status.get("state") or entry.get("state") or "UNKNOWN").upper()
    backend_state = state
    mapping = {
        "DONE": "succeeded",
        "FAILED": "failed",
        "ERROR": "failed",
        "TIMEOUT": "failed",
        "INTERRUPTED": "cancelled",
        "INTERRUPTING": "cancelled",
        "SYNC_FAILED": "failed",
        "CANCELLED": "cancelled",
        "QUEUED": "queued",
        "PENDING": "queued",
        "RUNNING": "running",
        "INITIALIZING": "running",
        "FINALIZING": "running",
    }
    return mapping.get(state, "unknown"), backend_state


def normalize_ovhai_job(entry: dict[str, Any]) -> dict[str, Any]:
    """Convert one `ovhai job list -o json` entry into a registry job dict."""
    spec = entry.get("spec") or {}
    status = entry.get("status") or {}
    job_uuid = str(entry.get("id") or "")
    registry_status, backend_state = _ovhai_status(entry)
    name = spec.get("name") or job_uuid
    kind = "eval_paired" if "eval" in name.lower() else "train_sft" if "train" in name.lower() else "other"
    duration = status.get("duration")
    return {
        "job_id": f"ovh-{job_uuid[:12]}",
        "kind": kind,
        "source": "ovhai",
        "name": name,
        "status": registry_status,
        "stage": registry_status,
        "backend_id": job_uuid,
        "backend_ids": [job_uuid],
        "backend_state": backend_state,
        "spec": {
            "image": spec.get("image"),
            "flavor": (spec.get("resources") or {}).get("flavor"),
            "gpu": (spec.get("resources") or {}).get("gpu"),
            "timeout": spec.get("timeout"),
            "command": spec.get("command"),
            "url": status.get("url"),
        },
        "artifacts": {"ovhai_url": status.get("url"), "duration": duration},
        "cost_class": "gpu-hour" if kind != "other" else None,
        "error": (status.get("info") or {}).get("message") if registry_status == "failed" else None,
    }


# ------------------------------------------------------------------- W&B


def wandb_configured(settings: Settings) -> bool:
    return bool(settings.wandb_api_key)


_STATE: dict[str, Any] = {}


def _wandb_api(settings: Settings):
    """Lazy singleton — wandb.Api() init is slow enough to amortize."""
    api = _STATE.get("wandb_api")
    if api is None:
        os.environ.setdefault("WANDB_API_KEY", settings.wandb_api_key)
        api = wandb.Api()
        _STATE["wandb_api"] = api
    return api


def wandb_runs(settings: Settings, *, limit: int = 20) -> list[dict[str, Any]]:
    """Recent W&B runs in the configured project. Requires wandb package."""
    if not wandb_configured(settings):
        return []
    api = _wandb_api(settings)
    path = f"{settings.wandb_entity}/{settings.wandb_project}"
    runs = api.runs(path, per_page=min(max(limit, 1), 50))
    out = []
    for run in itertools.islice(runs, limit):
        out.append(
            {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": str(run.created_at),
                "url": run.url,
                "tags": list(run.tags or []),
                "summary": {k: v for k, v in (run.summary or {}).items() if isinstance(v, (int, float, str))},
            }
        )
    return out


def wandb_run_metrics(settings: Settings, run_id: str, *, max_points: int = 50) -> dict[str, Any]:
    """Metric summary for one W&B run: last values per key + sample history."""
    if not wandb_configured(settings):
        return {"error": "wandb_not_configured"}
    api = _wandb_api(settings)
    path = f"{settings.wandb_entity}/{settings.wandb_project}/{run_id}"
    try:
        run = api.run(path)
    except Exception as exc:  # wandb raises CommError/KeyError variants
        return {"error": f"run_not_found: {type(exc).__name__}"}
    summary = {k: v for k, v in (run.summary or {}).items() if isinstance(v, (int, float, str))}
    history = run.history(samples=max_points, pandas=False)
    keys = [k for k in summary if isinstance(summary[k], (int, float))][:20]
    series = {k: [row.get(k) for row in history[-max_points:]] for k in keys}
    return {
        "run_id": run_id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "summary": summary,
        "history": series,
    }


# --------------------------------------------------------------------- R2


def r2_configured(settings: Settings) -> bool:
    return settings.r2_configured


def _r2_env_file(settings: Settings) -> None:
    """Point r2_sync at the merged env by materializing the needed keys."""
    for name in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET"):
        if settings.env.get(name):
            os.environ.setdefault(name, settings.env[name])


MIN_TAG_PARTS = 3
RESULT_PREVIEW_ROWS = 5


def r2_list_eval_tags(settings: Settings) -> list[str]:
    """Eval result tags present under ``evals/`` in the training bucket."""
    if not r2_configured(settings):
        return []
    _r2_env_file(settings)
    s3 = r2_sync.client()
    bucket = r2_sync.credentials()[r2_sync.DEFAULT_BUCKET_ENV]
    tags: set[str] = set()
    for entry in r2_sync.list_prefix(bucket=bucket, prefix="evals/", s3_client=s3):
        parts = entry["key"].split("/")
        if len(parts) >= MIN_TAG_PARTS:
            tags.add(parts[1])
    return sorted(tags)


def _parse_results_body(body: bytes) -> dict[str, Any]:
    """Parse a results.jsonl body into preview + per-arm pass rates."""
    lines = [ln for ln in body.decode("utf-8", "replace").splitlines() if ln.strip()]
    parsed: list[dict[str, Any]] = []
    for line in lines:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    out: dict[str, Any] = {
        "results_preview": parsed[:RESULT_PREVIEW_ROWS],
        "results_rows": len(parsed),
    }
    by_arm: dict[str, dict[str, int]] = {}
    has_passed = False
    for row in parsed:
        if "passed" not in row:
            continue
        has_passed = True
        arm = str(row.get("arm_id", "unknown"))
        slot = by_arm.setdefault(arm, {"total": 0, "passed": 0})
        arm_slot = slot
        arm_slot["total"] += 1
        if row.get("passed") is True:
            arm_slot["passed"] += 1
    if has_passed and by_arm:
        rates = {arm: (s["passed"] / s["total"] if s["total"] else 0.0) for arm, s in by_arm.items()}
        out["arm_pass_rates"] = {
            arm: {"passed": by_arm[arm]["passed"], "total": by_arm[arm]["total"], "rate": rates[arm]}
            for arm in by_arm
        }
        if "base" in rates and "sft" in rates:
            out["delta_sft_minus_base"] = rates["sft"] - rates["base"]
    else:
        out["arm_pass_rates"] = None
        out["note"] = "no pass/fail grades in results.jsonl (raw outputs only)"
    return out


def r2_eval_summary(settings: Settings, tag: str) -> dict[str, Any]:
    """Fetch manifest.json + first results.jsonl rows for each run under a tag."""
    if not r2_configured(settings):
        return {"error": "r2_not_configured"}
    _r2_env_file(settings)
    s3 = r2_sync.client()
    bucket = r2_sync.credentials()[r2_sync.DEFAULT_BUCKET_ENV]
    prefix = f"evals/{tag}/"
    entries = list(r2_sync.list_prefix(bucket=bucket, prefix=prefix, s3_client=s3))
    runs: dict[str, dict[str, Any]] = {}
    for entry in entries:
        rel = entry["key"][len(prefix) :]
        run_name, _, filename = rel.partition("/")
        if not run_name or not filename:
            continue
        slot = runs.setdefault(run_name, {"files": {}})
        slot["files"][filename] = entry["key"]
    result: dict[str, Any] = {"tag": tag, "bucket": bucket, "runs": {}}
    for run_name, slot in runs.items():
        run_out: dict[str, Any] = {"files": list(slot["files"])}
        for wanted in ("manifest.json", "results.jsonl"):
            key = slot["files"].get(wanted)
            if not key:
                continue
            obj = s3.get_object(Bucket=bucket, Key=key)
            body = obj["Body"].read()
            if wanted == "manifest.json":
                try:
                    run_out["manifest"] = json.loads(body)
                except json.JSONDecodeError:
                    run_out["manifest_raw"] = body[:2000].decode("utf-8", "replace")
            else:
                run_out.update(_parse_results_body(body))
        result["runs"][run_name] = run_out
    return result
