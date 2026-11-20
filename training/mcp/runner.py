"""Subprocess runner + single-flight for training jobs (Phase 1).

The MCP never runs training in-process. ``submit_job`` resolves a plan,
writes a job row, and spawns the recipe's CLI entrypoint as a managed
subprocess with a wrapped env that injects credentials server-side. This
module owns the lifecycle: queue, single-flight, status reconciliation.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import subprocess
import threading
import time
from typing import Any

from training.mcp import db
from training.mcp.config import Settings

# Single-flight locks per backend kind.
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_PROCESSES: dict[str, subprocess.Popen[bytes]] = {}
_PROCESSES_GUARD = threading.Lock()


def _lock_for(backend: str) -> threading.Lock:
    with _LOCKS_GUARD:
        if backend not in _LOCKS:
            _LOCKS[backend] = threading.Lock()
        return _LOCKS[backend]


def _build_env(settings: Settings) -> dict[str, str]:
    env = dict(os.environ)
    env.update(settings.env)
    # Ensure python can import training module from ai root.
    ai_root = str(settings.db_path.parents[2])  # ai/training/mcp/state -> ai
    env.setdefault("PYTHONPATH", ai_root)
    if ai_root not in env.get("PYTHONPATH", ""):
        env["PYTHONPATH"] = ai_root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _monitor(settings: Settings, job_id: str, proc: subprocess.Popen[bytes], _log_path: str) -> None:
    with contextlib.suppress(Exception):
        proc.wait()
    # Collect tail of stdout/stderr into log file (already redirected).
    rc = proc.returncode
    job = db.get_job(settings.db_path, job_id)
    if job is None:
        return
    # Determine terminal status.
    if rc == 0:
        new_status = "succeeded"
        error = None
    elif rc is None:
        new_status = "cancelled"
        error = "cancelled by operator"
    else:
        new_status = "failed"
        error = f"process exited with code {rc}"
    # Collecting phase: briefly mark collecting then terminal.
    db.upsert_job(settings.db_path, {
        "job_id": job_id,
        "kind": job["kind"],
        "source": job["source"],
        "name": job.get("name"),
        "status": "collecting",
        "stage": "collecting",
        "backend_id": job.get("backend_id"),
    })
    time.sleep(0.5)
    db.upsert_job(settings.db_path, {
        "job_id": job_id,
        "kind": job["kind"],
        "source": job["source"],
        "name": job.get("name"),
        "status": new_status,
        "stage": new_status,
        "backend_id": job.get("backend_id"),
        "error": error,
        "finished_at": None,
    })
    with _PROCESSES_GUARD:
        _PROCESSES.pop(job_id, None)


def submit(settings: Settings, job_id: str, job_row: dict[str, Any], command: str, backend: str) -> None:
    """Spawn the subprocess for a queued job, respecting single-flight."""
    lock = _lock_for(backend)

    def _run() -> None:
        # Wait for single-flight slot.
        acquired = lock.acquire(timeout=300)
        if not acquired:
            db.upsert_job(settings.db_path, {
                "job_id": job_id, "kind": job_row["kind"], "source": job_row["source"],
                "name": job_row.get("name"), "status": "failed", "stage": "failed",
                "backend_id": job_row.get("backend_id"), "error": "single-flight timeout",
            })
            return
        try:
            db.upsert_job(settings.db_path, {
                "job_id": job_id, "kind": job_row["kind"], "source": job_row["source"],
                "name": job_row.get("name"), "status": "submitted", "stage": "submitted",
                "backend_id": job_row.get("backend_id"),
            })
            env = _build_env(settings)
            log_dir = settings.db_path.parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = str(log_dir / f"{job_id}.log")
            try:
                cmd_list = shlex.split(command)
            except ValueError:
                cmd_list = command.split()
            with open(log_path, "w") as lf:
                proc = subprocess.Popen(
                    cmd_list,
                    cwd=str(settings.db_path.parents[2]),  # ai root
                    env=env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )
            with _PROCESSES_GUARD:
                _PROCESSES[job_id] = proc
            db.upsert_job(settings.db_path, {
                "job_id": job_id, "kind": job_row["kind"], "source": job_row["source"],
                "name": job_row.get("name"), "status": "running", "stage": "running",
                "backend_id": job_row.get("backend_id"),
                "artifacts": {"log_path": log_path},
            })
            _monitor(settings, job_id, proc, log_path)
        finally:
            lock.release()

    threading.Thread(target=_run, daemon=True).start()


def cancel(settings: Settings, job_id: str) -> bool:
    with _PROCESSES_GUARD:
        proc = _PROCESSES.get(job_id)
    if proc is not None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass
    job = db.get_job(settings.db_path, job_id)
    if job is None:
        return False
    db.upsert_job(settings.db_path, {
        "job_id": job_id, "kind": job["kind"], "source": job["source"],
        "name": job.get("name"), "status": "cancelled", "stage": "cancelled",
        "backend_id": job.get("backend_id"), "error": "cancelled by operator",
    })
    # Best-effort backend cancel for ovhai.
    if job.get("source") == "ovhai" and job.get("backend_id"):
        with contextlib.suppress(Exception):
            subprocess.run(["ovhai", "job", "stop", job["backend_id"]], capture_output=True, timeout=30, check=False)
    return True
