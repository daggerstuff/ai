"""Reconcile live backends into the job registry.

The registry is the source of truth, but it is fed from two directions:

* **Import** — mirror what a backend already knows (``ovhai job list``, recent
  W&B runs, R2 eval tags, operator-supplied manual rows) into the registry so
  ``list_jobs`` answers truthfully about work that was launched by hand.
* **Reconcile** — a row that is still active in the registry but whose backend
  job is no longer visible is *lost*, not eternally ``running``. It is marked
  ``failed`` with ``error: backend_lost`` (PIX-5012).

Lost-detection is deliberately conservative: it only examines sources whose
listing actually succeeded. A source that is unconfigured or whose listing
raised is skipped entirely, because "the backend is unreachable" must never be
mistaken for "the job vanished".
"""

from __future__ import annotations

import math
from typing import Any

from training.mcp import backends, db
from training.mcp.config import IMPORT_ACTOR, NON_REQUESTING_ACTORS, Settings

#: Statuses a registry row can still leave. Anything else is terminal and is
#: never rewritten by the reconciler.
ACTIVE_STATUSES = frozenset({"planned", "queued", "staging", "submitted", "running", "collecting"})

#: Terminal statuses, mirrored from runner.TERMINAL_STATUSES without importing
#: runner (which would create a module cycle risk for a constant).
_BACKEND_LOST = "backend_lost"

__all__ = [
    "ACTIVE_STATUSES",
    "IMPORT_ACTOR",
    "NON_REQUESTING_ACTORS",
    "import_manual_rows",
    "reconcile_lost_backends",
    "sync_all",
    "sync_ovhai",
    "sync_r2_evals",
    "sync_wandb",
]


def _import_rows(settings: Settings, rows: list[dict[str, Any]]) -> int:
    imported = 0
    for row in rows:
        db.upsert_job(settings.db_path, row)
        imported += 1
    return imported


def sync_ovhai(settings: Settings, *, limit: int = 50) -> dict[str, Any]:
    """Mirror ``ovhai job list`` into the registry."""
    entries = backends.ovhai_list_jobs(settings, limit=limit)
    rows = [backends.normalize_ovhai_job(entry) for entry in entries]
    return {
        "source": "ovhai",
        "configured": backends.ovhai_configured(settings),
        "imported": _import_rows(settings, rows),
        "total_seen": len(entries),
        "limit": limit,
        "backend_ids": [row["backend_id"] for row in rows if row.get("backend_id")],
    }


def sync_wandb(settings: Settings, *, limit: int = 20) -> dict[str, Any]:
    """Mirror recent W&B runs into the registry (hand-launched train jobs)."""
    if not backends.wandb_configured(settings):
        return {"source": "wandb", "configured": False, "imported": 0, "total_seen": 0, "backend_ids": []}
    runs = backends.wandb_runs(settings, limit=limit)
    rows = [backends.normalize_wandb_run(run) for run in runs]
    return {
        "source": "wandb",
        "configured": True,
        "imported": _import_rows(settings, rows),
        "total_seen": len(runs),
        "limit": limit,
        "backend_ids": [row["backend_id"] for row in rows if row.get("backend_id")],
    }


def sync_r2_evals(settings: Settings) -> dict[str, Any]:
    """Import R2 eval tags as historical ``eval_paired`` rows.

    R2 holds results, not jobs, so these rows are recorded as terminal
    ``succeeded`` artifacts with no backend id. They exist so the registry can
    answer "what evals have we ever run" for tags produced before the control
    plane existed — and so ``eval_gate``/``eval_results`` tags line up with a
    registry row.
    """
    if not backends.r2_configured(settings):
        return {"source": "r2", "configured": False, "imported": 0, "total_seen": 0, "backend_ids": []}
    tags = backends.r2_list_eval_tags(settings)
    rows = [
        {
            "job_id": f"r2-{tag}"[:64],
            "kind": "eval_paired",
            "source": "r2",
            "name": tag,
            "status": "succeeded",
            "stage": "succeeded",
            "requested_by": IMPORT_ACTOR,
            "spec": {"tag": tag, "imported_from": "r2:evals/"},
            "artifact_refs": {"r2_prefix": f"evals/{tag}/"},
            "cost_class": "gpu-hour",
        }
        for tag in tags
    ]
    return {
        "source": "r2",
        "configured": True,
        "imported": _import_rows(settings, rows),
        "total_seen": len(tags),
        "backend_ids": [],
    }


MANUAL_REQUIRED_FIELDS = ("job_id", "kind", "status")


def import_manual_rows(settings: Settings, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Import operator-supplied historical rows from a manual log.

    Used to seed runs that no backend can enumerate (a hand-run script whose
    only record is a log file). Each row must at minimum carry ``job_id``,
    ``kind``, and ``status``; anything else is passed through to the registry.
    Validation failures are reported per row rather than aborting the batch.
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        normalized = dict(row)
        missing = [field for field in MANUAL_REQUIRED_FIELDS if not normalized.get(field)]
        if missing:
            rejected.append({"index": index, "missing": missing})
            continue
        cost = normalized.get("cost_estimate")
        if cost is not None:
            if isinstance(cost, bool):
                rejected.append({"index": index, "invalid": "cost_estimate"})
                continue
            try:
                parsed = float(cost)
            except (TypeError, ValueError, OverflowError):
                rejected.append({"index": index, "invalid": "cost_estimate"})
                continue
            if not math.isfinite(parsed) or parsed < 0:
                rejected.append({"index": index, "invalid": "cost_estimate"})
                continue
            normalized["cost_estimate"] = parsed
        accepted.append(
            {
                **normalized,
                "source": normalized.get("source") or "manual",
                "stage": normalized.get("stage") or normalized["status"],
                "requested_by": normalized.get("requested_by") or IMPORT_ACTOR,
            }
        )
    return {
        "source": "manual",
        "imported": _import_rows(settings, accepted),
        "total_seen": len(rows),
        "rejected": rejected,
        "backend_ids": [],
    }


def reconcile_lost_backends(
    settings: Settings,
    *,
    seen: dict[str, set[str]],
) -> dict[str, Any]:
    """Mark active rows whose backend job is no longer listed as ``backend_lost``.

    ``seen`` maps a job ``source`` to the backend ids the listing returned. A
    source absent from ``seen`` is never reconciled: we cannot distinguish
    "vanished" from "not queried".
    """
    lost: list[dict[str, Any]] = []
    for source, backend_ids in seen.items():
        for job in db.list_jobs(settings.db_path, source=source, limit=500):
            if job.get("status") not in ACTIVE_STATUSES:
                continue
            backend_id = job.get("backend_id")
            if not backend_id or backend_id in backend_ids:
                continue
            db.upsert_job(
                settings.db_path,
                {
                    "job_id": job["job_id"],
                    "kind": job["kind"],
                    "source": job["source"],
                    "name": job.get("name"),
                    "status": "failed",
                    "stage": "failed",
                    "backend_id": backend_id,
                    "error": _BACKEND_LOST,
                    "finished_at": db.now(),
                },
            )
            lost.append({"job_id": job["job_id"], "source": source, "backend_id": backend_id})
    return {"checked_sources": sorted(seen), "lost": len(lost), "jobs": lost}


def sync_all(settings: Settings) -> dict[str, Any]:
    """Import + reconcile every configured backend. Per-source failures are isolated."""
    results: dict[str, Any] = {}
    seen: dict[str, set[str]] = {}
    sources = (
        ("ovhai", sync_ovhai),
        ("wandb", sync_wandb),
        ("r2", sync_r2_evals),
    )
    for name, fn in sources:
        try:
            result = fn(settings)
        except Exception as exc:
            results[name] = {"error": f"{type(exc).__name__}: {exc}", "configured": True}
            continue
        results[name] = result
        if result.get("configured") and isinstance(result.get("backend_ids"), list):
            listing_limit = result.get("limit")
            if listing_limit is None or result.get("total_seen", 0) < listing_limit:
                seen[name] = set(result["backend_ids"])
            else:
                # A full page is not proof of absence: an active job just past
                # the window would be failed as lost on a listing that never
                # saw it, so only a provably complete page may mark ids lost.
                results[name]["listing_may_be_truncated"] = True
    try:
        results["reconcile"] = reconcile_lost_backends(settings, seen=seen)
    except Exception as exc:
        results["reconcile"] = {"error": f"{type(exc).__name__}: {exc}"}
    return results
