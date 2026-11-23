"""Subprocess runner + single-flight for training jobs.

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
import sys
import threading
import time
from pathlib import Path
from typing import Any

from training.mcp import db, foresight
from training.mcp.config import AI_ROOT, WORKSPACE_ROOT, Settings

# Single-flight locks per backend kind.
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_PROCESSES: dict[str, subprocess.Popen[bytes]] = {}
_PROCESSES_GUARD = threading.Lock()
#: Jobs an operator cancelled while they were still queued (waiting for the
#: single-flight slot). cancel() records the id here *before* touching the
#: registry so the waiting thread cannot resurrect the job by spawning it.
_CANCELLED: set[str] = set()

#: Statuses a job can never leave. The reconciler and cancel() both use this
#: to avoid resurrecting or re-terminating a job that already finished.
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


def _lock_for(backend: str) -> threading.Lock:
    with _LOCKS_GUARD:
        if backend not in _LOCKS:
            _LOCKS[backend] = threading.Lock()
        return _LOCKS[backend]


# Environment variables a launched job legitimately needs. Anything not
# listed here is NOT inherited from the service process — this is the
# credential-isolation boundary: the MCP holds fleet secrets (Linear, Slack,
# GitHub, Cloudflare) that a training subprocess has no business seeing.
_ENV_ALLOWLIST = (
    "HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "PYTHONHASHSEED",
    "TRAINING_MCP_ENV_FILE",
)

# Dotenv keys the training pipeline genuinely needs (W&B + R2 + OVH).
_JOB_SECRET_ALLOWLIST = (
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "WANDB_BASE_URL",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_ENDPOINT",
    "R2_BUCKET",
    "OVHAI_AI_APP_TOKEN",
    "HF_TOKEN",
)


def _first_existing(*candidates: Path) -> Path:
    """Return the first candidate that exists, else the first candidate.

    Recipe commands are written to be run from the repository root the
    training scripts themselves assume (``AI_ROOT.parent`` — the workspace,
    which owns ``docs/training/execution-*/``). The interpreter is resolved to
    the service venv, so no candidate needs to be on ``sys.path``; the value
    only has to exist on disk and be stable.
    """
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


#: Working directory for spawned jobs: the workspace root, not the `ai`
#: submodule. Training recipes reference ``docs/training/execution-*/`` and
#: those manifests live in the workspace, so a job started from ``ai/`` cannot
#: resolve the relative ``launch_config`` the agent tools send.
JOB_CWD = _first_existing(WORKSPACE_ROOT, AI_ROOT)


def _build_env(settings: Settings) -> dict[str, str]:
    """Build a minimal, credential-scoped env for a job subprocess.

    Only the allow-listed process vars and the pipeline's own dotenv secrets
    are passed through; unrelated fleet secrets never reach the child.
    """
    env: dict[str, str] = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    for name in _JOB_SECRET_ALLOWLIST:
        value = settings.env.get(name)
        if value:
            env[name] = value
    # Ensure python can import ``training`` even without the editable install.
    # The scripts live under the `ai` submodule, so that is the path that
    # matters — not the cwd.
    paths = [str(AI_ROOT)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


#: Every key a job subprocess may legitimately carry. Used by the security
#: review to prove no unrelated secret leaked into the child environment.
JOB_ENV_ALLOWLIST = frozenset(_ENV_ALLOWLIST) | frozenset(_JOB_SECRET_ALLOWLIST)


def _monitor(settings: Settings, job_id: str, proc: subprocess.Popen[bytes], _log_path: str) -> None:
    with contextlib.suppress(Exception):
        proc.wait()
    # Collect tail of stdout/stderr into log file (already redirected).
    rc = proc.returncode
    job = db.get_job(settings.db_path, job_id)
    if job is None:
        _drop_process(job_id)
        return
    # A cancel() already wrote the terminal row; don't overwrite it with a
    # process-exit status (terminate() gives a non-zero rc). The marker covers
    # the window between cancel() registering the id and writing its row.
    if _is_cancelled(job_id) or job.get("status") == "cancelled":
        _drop_process(job_id)
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
    # Collecting phase: briefly mark collecting then terminal. Both writes are
    # conditional on the row not being cancelled yet, because cancel() runs on
    # the request thread and can land inside this window: the monitor read the
    # row before it, so an unconditional write here resurrected the operator's
    # cancellation as a failure.
    if not db.transition_job(settings.db_path, job_id, status="collecting", stage="collecting"):
        _drop_process(job_id)
        return
    time.sleep(0.5)
    if not db.transition_job(
        settings.db_path,
        job_id,
        status=new_status,
        stage=new_status,
        error=error,
        # Terminal rows must carry finished_at so provenance and duration
        # queries work; it also gates the "is this job over" checks.
        finished_at=db.now(),
    ):
        _drop_process(job_id)
        return
    _record_terminal_outcome(settings, job_id)
    _drop_process(job_id)


def _record_terminal_outcome(settings: Settings, job_id: str) -> None:
    """Write the terminal job outcome to Foresight (best-effort, epic criterion).

    Doing it here rather than agent-side means a job nobody polls still leaves
    a provenance record for ``promote_to_*`` to find. Foresight being down must
    never change the job's result, so failures are swallowed.
    """
    if not settings.foresight_enabled:
        return
    with contextlib.suppress(Exception):
        job = db.get_job(settings.db_path, job_id)
        if job is None or job.get("status") not in TERMINAL_STATUSES:
            return
        outcome = foresight.store_terminal_memory(settings, job)
        db.write_audit(
            settings.db_path,
            action="foresight.terminal",
            actor="mcp",
            job_id=job_id,
            recipe_id=job.get("name"),
            plan_hash=job.get("plan_hash"),
            detail={"stored": outcome.get("stored"), "error": outcome.get("error")},
        )


def _drop_process(job_id: str) -> None:
    with _PROCESSES_GUARD:
        _PROCESSES.pop(job_id, None)


def _terminate_proc(proc: subprocess.Popen[bytes]) -> None:
    """Stop a job process: terminate, wait, then kill. Never raises."""
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


def _is_cancelled(job_id: str) -> bool:
    with _PROCESSES_GUARD:
        return job_id in _CANCELLED


def _mark_cancelled(job_id: str) -> None:
    with _PROCESSES_GUARD:
        _CANCELLED.add(job_id)


def _forget_cancelled(job_id: str) -> None:
    with _PROCESSES_GUARD:
        _CANCELLED.discard(job_id)


def _resolve_interpreter(command: str) -> str:
    """Run recipe commands in the MCP's own interpreter.

    Recipes spell entrypoints as ``python -m training...``, but the systemd unit
    runs with a PATH that has only ``python3`` (no ``python``), so a bare
    ``python`` raised FileNotFoundError. Rather than depend on the caller's
    PATH, rewrite a leading ``python``/``python3`` token to ``sys.executable`` —
    the venv the service itself runs in, which is also where the training
    package and its dependencies live.
    """
    try:
        parts = shlex.split(command)
    except ValueError:
        return command
    if parts and os.path.basename(parts[0]) in {"python", "python3", "python3.14"}:
        parts[0] = sys.executable
    return shlex.join(parts)


def submit(settings: Settings, job_id: str, job_row: dict[str, Any], command: str, backend: str) -> None:
    """Spawn the subprocess for a queued job, respecting single-flight."""
    lock = _lock_for(backend)
    resolved_command = _resolve_interpreter(command)

    def _run() -> None:
        # Wait for single-flight slot.
        acquired = lock.acquire(timeout=300)
        if not acquired:
            # Guarded like every other terminal write: a cancel that landed
            # while this job waited for the slot owns the row and the record.
            if db.mark_job_terminal(settings.db_path, job_id, status="failed", error="single-flight timeout"):
                _record_terminal_outcome(settings, job_id)
            return
        try:
            # A queued job can be cancelled while it waits for the slot.
            # cancel() registers the id in _CANCELLED before it writes the
            # terminal row, so a cancel that lands during the wait must win:
            # spawning here would overwrite the operator's cancellation with
            # `submitted`/`running` and eventually report `succeeded` for work
            # that was explicitly stopped.
            if _is_cancelled(job_id):
                # cancel() owns the row and the terminal record from here: it
                # wins (or already won) the guarded write and records the
                # outcome itself. Recording here too duplicated provenance.
                return
            # Conditional on the row still being active: a cancellation that
            # landed while this job waited for the slot must not be overwritten
            # by `submitted`, or the operator's stop turns into a `succeeded`.
            if not db.transition_job(settings.db_path, job_id, status="submitted", stage="submitted"):
                _forget_cancelled(job_id)
                return
            env = _build_env(settings)
            log_dir = settings.db_path.parent / "logs"
            log_path = str(log_dir / f"{job_id}.log")
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                cmd_list = shlex.split(resolved_command)
            except ValueError:
                cmd_list = resolved_command.split()
            except OSError as exc:
                # Preparation failed (permissions, ENOSPC, ...): land the job
                # in a terminal state instead of stranding the row at
                # `submitted` with no process and no provenance.
                if db.mark_job_terminal(
                    settings.db_path,
                    job_id,
                    status="failed",
                    error=f"spawn failed: {type(exc).__name__}: {exc}",
                ):
                    _record_terminal_outcome(settings, job_id)
                return
            try:
                with open(log_path, "w") as lf:
                    proc = subprocess.Popen(
                        cmd_list,
                        # The workspace root, not `ai/`: every recipe entrypoint
                        # resolves docs/training/... relative to AI_ROOT.parent.
                        cwd=str(JOB_CWD),
                        env=env,
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                    )
            except OSError as exc:
                # A spawn failure must land the job in a terminal state: this
                # runs on a daemon thread, so an escaping exception would leave
                # the row stuck at `submitted` forever with no operator signal.
                # The write is guarded, so a cancel that won the row meanwhile
                # keeps its own outcome and its single provenance record.
                if db.mark_job_terminal(
                    settings.db_path,
                    job_id,
                    status="failed",
                    error=f"spawn failed: {type(exc).__name__}: {exc}",
                ):
                    with open(log_path, "a") as lf:
                        lf.write(f"[runner] spawn failed: {type(exc).__name__}: {exc}\n")
                    _record_terminal_outcome(settings, job_id)
                return
            # Coordinate the marker, the process registration, and the
            # running write: a cancel() that started while the process spawned
            # either finds the process here and terminates it, or is already
            # in _CANCELLED by the time the registration completes, so one of
            # the two is guaranteed to stop the child. The running write itself
            # is guarded so a cancel that has already won the row cannot be
            # overwritten and the job left stuck `running`.
            with _PROCESSES_GUARD:
                _PROCESSES[job_id] = proc
                lost = job_id in _CANCELLED
            if lost or not db.transition_job(
                settings.db_path,
                job_id,
                status="running",
                stage="running",
                artifacts={"log_path": log_path},
            ):
                # The cancellation owns the terminal record: it won (or is
                # about to win) the guarded row write and records the outcome.
                _terminate_proc(proc)
                _drop_process(job_id)
                return
            _monitor(settings, job_id, proc, log_path)
        finally:
            _forget_cancelled(job_id)
            lock.release()

    threading.Thread(target=_run, daemon=True).start()


def cancel(settings: Settings, job_id: str) -> bool:
    # Register the cancellation BEFORE inspecting anything: a job still waiting
    # for the single-flight slot has no entry in _PROCESSES, so without this
    # the waiting thread would spawn it moments later and overwrite the
    # `cancelled` row with `succeeded`.
    _mark_cancelled(job_id)
    with _PROCESSES_GUARD:
        proc = _PROCESSES.get(job_id)
    if proc is not None:
        _terminate_proc(proc)
    job = db.get_job(settings.db_path, job_id)
    if job is None:
        _forget_cancelled(job_id)
        return False
    if job.get("status") in TERMINAL_STATUSES:
        # Already over — do not leave a stale marker that would block a
        # re-submitted job that happened to reuse the id (ids are unique, but
        # the marker set is process-global and long-lived).
        _forget_cancelled(job_id)
        return False
    # The UPDATE carries the terminal guard, so a monitor that committed
    # `succeeded`/`failed` after the read above wins instead of being rewritten
    # here: one terminal outcome per job.
    if not db.mark_job_terminal(settings.db_path, job_id, status="cancelled", error="cancelled by operator"):
        _forget_cancelled(job_id)
        return False
    # Best-effort backend cancel for ovhai.
    if job.get("source") == "ovhai" and job.get("backend_id"):
        with contextlib.suppress(Exception):
            subprocess.run(["ovhai", "job", "stop", job["backend_id"]], capture_output=True, timeout=30, check=False)
    _record_terminal_outcome(settings, job_id)
    return True
