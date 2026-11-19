"""Reconcile live backends into the job registry.

Phase 0 is read-only against backends: ``sync_ovhai`` mirrors ``ovhai job
list`` into the registry so ``list_jobs`` answers truthfully about work
that was launched by hand. Future submit paths write rows directly.
"""

from __future__ import annotations

from typing import Any

from training.mcp import backends, db
from training.mcp.config import Settings


def sync_ovhai(settings: Settings, *, limit: int = 50) -> dict[str, Any]:
    entries = backends.ovhai_list_jobs(settings, limit=limit)
    imported = 0
    for entry in entries:
        job = backends.normalize_ovhai_job(entry)
        db.upsert_job(settings.db_path, job)
        imported += 1
    return {"source": "ovhai", "imported": imported, "total_seen": len(entries)}


def sync_all(settings: Settings) -> dict[str, Any]:
    """Reconcile every configured backend. Per-source failures are isolated."""
    results: dict[str, Any] = {}
    try:
        results["ovhai"] = sync_ovhai(settings)
    except Exception as exc:
        results["ovhai"] = {"error": f"{type(exc).__name__}: {exc}"}
    return results
