"""Run every paired eval for a release once its SFT job finishes.

Each release needs the same three readings -- the adapter against the base
model and against the two adapters before it -- across eight suites, and the
adapter URI only exists after the training job logs its lora artifact. Doing
that by hand means babysitting a two-hour job and then transcribing 24 commands
from the prep plan, which is how the v16 batch ended up with one run that never
finished and a stale scoreboard.

This script owns the whole sequence: poll the job, resolve the artifact from the
run that produced it, write the configs, run them in parallel, render the
scoreboard, and publish the results root to W&B. A run that dies on a provider
5xx is retried once from a clean results directory -- the harness records its
own flag set in the run manifest, so resuming a partially written directory is
rejected as an identity change.

Usage:

    python -m training.launch.wandb_release_evals \
        --release-label v17 \
        --task-id b6886c9f8dde3c9bcd9b9cd4fd5e81d0637ecd783cb3d1521fa7f1bc7c94bd68 \
        --job-id 10019e23-9f44-46f2-afbc-b65eb3dc0f1b \
        --exec-dir docs/training/execution-2026-11-09 \
        --compare v15=wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-73504a99ed05:v1

The default comparison is whatever ``prepare_release_eval`` points at, so the
common case is just ``--release-label``/``--task-id``/``--job-id``/``--exec-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from training.launch import prepare_release_eval as prep

REPO = Path(__file__).resolve().parents[3]
AI = REPO / "ai"
TRAINING_API = "https://api.training.wandb.ai/v1/preview/training-jobs"
PYTHON = sys.executable
RUN_PREFIX = "arc-sft-baseline-"
RUN_HASH_LENGTH = 12
JOB_ACTIVE_STATES = {"RUNNING", "PENDING", "UNKNOWN", "QUEUED", "CREATING"}
JOB_SUCCESS_STATES = {"FINISHED", "SUCCEEDED", "COMPLETED"}
# A provider hiccup is worth one clean retry; anything else is a real failure.
TRANSIENT_MARKERS = ("http_502", "http_503", "http_504", "http_500", "ProviderHTTPError", "ReadTimeout")
# Transport failures are not model behaviour, so they are re-asked instead of
# being scored. The eval subprocess records them in results.jsonl, which is the
# only reliable signal: the manifest it prints carries counts, not error types.
TRANSIENT_ERROR_TYPES = frozenset({"TimeoutError", "URLError", "RemoteDisconnected", "ConnectionResetError"})
TRANSIENT_HTTP_STATUSES = frozenset({500, 502, 503, 504})
MAX_TRANSIENT_RETRIES = 2

# Labels flow into child-process argv (results dirs, config names) and W&B run
# names. Restrict them to a safe token alphabet so a crafted value cannot smuggle
# flags or path traversal into the eval subprocesses.
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _validate_token(value: str, flag: str) -> str:
    if not _SAFE_TOKEN.fullmatch(value):
        raise argparse.ArgumentTypeError(f"{flag} must match {_SAFE_TOKEN.pattern}, got {value!r}")
    return value


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S", time.gmtime())
    sys.stdout.write(f"{stamp} {message}\n")
    sys.stdout.flush()


def api_key() -> str:
    for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("WANDB_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("WANDB_API_KEY missing from .env")


def job_status(key: str, job_id: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{TRAINING_API}/{job_id}",
        headers={"Authorization": f"Bearer {key}", "User-Agent": "curl/8.5.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload: dict[str, Any] = json.load(response)
    return payload


def wait_for_job(key: str, job_id: str, poll_seconds: int, max_polls: int) -> str:
    for poll in range(max_polls):
        try:
            status = str(job_status(key, job_id).get("status", "unknown")).upper()
        except Exception as error:  # transient network blips must not end the poll loop
            log(f"poll {poll} failed: {error}")
            status = "UNKNOWN"
        if poll % 5 == 0:
            log(f"poll {poll}: {status}")
        if status not in JOB_ACTIVE_STATES:
            log(f"job reached terminal status: {status}")
            return status
        time.sleep(poll_seconds)
    raise SystemExit("job never left the active states")


def parse_compare(value: str) -> tuple[str, str]:
    label, _, uri = value.partition("=")
    if not label or not uri.startswith("wandb-artifact:///"):
        raise argparse.ArgumentTypeError(f"expected label=wandb-artifact:///... , got {value!r}")
    return _validate_token(label, "--compare label"), uri


def resolve_compares(
    requested: list[tuple[str, str]],
    default: tuple[str, str],
    release_label: str,
    include_default: bool = True,
) -> list[tuple[str, str]]:
    """Return the comparisons to run, keeping the default predecessor in the set.

    ``--compare`` used to replace the default outright, so a release launched
    with an explicit predecessor silently lost its reading against the adapter
    the prep defaults point at -- the one the last scoreboard called the best.
    The requested comparisons come first so their runs are scheduled ahead of
    the default, and a label is never compared against itself.
    """
    compares = [(label, uri) for label, uri in requested if label != release_label]
    labels = {label for label, _ in compares}
    if include_default and default[0] != release_label and default[0] not in labels:
        compares.append(default)
    if not compares:
        raise SystemExit("no comparisons to run: every label named matches the release itself")
    return compares


def build_commands(
    release_label: str,
    release_uri: str,
    compares: list[tuple[str, str]],
    exec_dir: Path,
) -> list[tuple[str, list[str]]]:
    cases_dir = prep.default_cases_dir()
    release = prep.Release(release_label, release_uri)
    log(f"release artifact: {release.uri}")

    commands: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for label, uri in compares:
        written = prep.write_configs(cases_dir, exec_dir / "evals", release, prep.Release(label, uri))
        for run_label, cases, config, sealed in written:
            if run_label in seen:
                # the vs-base reading is identical for every compare; keep one.
                continue
            if not _SAFE_TOKEN.fullmatch(run_label):
                raise ValueError(f"unsafe run_label from prepare_release_eval: {run_label!r}")
            seen.add(run_label)
            argv = [
                PYTHON,
                "training/run_wandb_paired_eval.py",
                "--cases",
                str(cases),
                "--config",
                str(config),
                "--results-dir",
                str(exec_dir / "evals" / "results" / run_label),
                "--max-input-tokens",
                str(prep.MAX_INPUT_TOKENS),
                "--max-tool-rounds",
                str(prep.MAX_TOOL_ROUNDS),
            ]
            if sealed:
                argv.append("--allow-sealed")
            commands.append((run_label, argv))
    log(f"prepared {len(commands)} eval runs")
    return commands


def transient_failures(results_dir: Path) -> list[str]:
    """Failed records caused by the transport rather than by the model.

    A resumed run appends its retry to the same file, so the last record for a
    case is the one that counts; an earlier timeout that was later answered
    must not trigger another pass.
    """
    path = results_dir / "results.jsonl"
    if not path.is_file():
        return []
    latest: dict[tuple[Any, Any], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            latest[(record.get("arm_id"), record.get("case_id"))] = record
    names: list[str] = []
    for record in latest.values():
        if record.get("status") == "success":
            continue
        if (
            record.get("error_type") in TRANSIENT_ERROR_TYPES
            or record.get("provider_http_status") in TRANSIENT_HTTP_STATUSES
        ):
            names.append(f"{record.get('arm_id')}/{record.get('case_id')}={record.get('error_type')}")
    return names


def run_one(label: str, argv: list[str], key: str) -> tuple[str, int]:
    env = dict(os.environ, PYTHONPATH=str(AI), WANDB_API_KEY=key)
    results_dir = Path(argv[argv.index("--results-dir") + 1])
    log(f"start {label}")
    completed = subprocess.run(
        argv,
        cwd=AI,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    tail = (completed.stdout or "").strip().splitlines()[-3:]
    log(f"done {label} rc={completed.returncode} {' | '.join(tail)}")
    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0 and stderr:
        log(f"stderr {label}: {stderr[-800:]}")
    for attempt in range(1, MAX_TRANSIENT_RETRIES + 1):
        if completed.returncode == 0:
            return label, completed.returncode
        hiccups = transient_failures(results_dir)
        if hiccups:
            # Non-fatal: the harness resumes, so only the failed cases are re-asked.
            log(f"retry {label} {attempt} for {len(hiccups)} transport failure(s): {'; '.join(hiccups[:6])}")
            completed = subprocess.run(
                argv,
                cwd=AI,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            log(f"retry {label} {attempt} rc={completed.returncode}")
            continue
        if any(marker in stderr or marker in (completed.stdout or "") for marker in TRANSIENT_MARKERS):
            # Fatal provider metadata failure blocks resume, so start clean.
            log(f"retrying {label} from a clean results directory")
            shutil.rmtree(results_dir, ignore_errors=True)
            completed = subprocess.run(
                [*argv, "--continue-on-transient-http-errors"],
                cwd=AI,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            log(f"retry {label} rc={completed.returncode}")
            continue
        return label, completed.returncode
    return label, completed.returncode


def render_scoreboard(exec_dir: Path, release_label: str) -> int:
    completed = subprocess.run(
        [PYTHON, "training/launch/summarize_eval_run.py", "--results-root", str(exec_dir / "evals" / "results")],
        cwd=AI,
        env=dict(os.environ, PYTHONPATH=str(AI)),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        log(f"scoreboard failed rc={completed.returncode}: {completed.stderr.strip()[-800:]}")
        return completed.returncode
    target = exec_dir / f"{release_label}-scoreboard.md"
    target.write_text(completed.stdout, encoding="utf-8")
    log(f"scoreboard written to {target} ({len(completed.stdout.splitlines())} lines)")
    return 0


def track_eval_run(exec_dir: Path, release_label: str) -> int:
    """Publish the finished results root to W&B (metrics + artifact)."""
    completed = subprocess.run(
        [
            PYTHON,
            "training/launch/log_eval_tracking.py",
            "--results-root",
            str(exec_dir / "evals" / "results"),
            "--release-label",
            release_label,
            "--exec-dir",
            str(exec_dir),
        ],
        cwd=AI,
        env=dict(os.environ, PYTHONPATH=str(AI)),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        log(f"eval tracking failed rc={completed.returncode}: {completed.stderr.strip()[-800:]}")
        return completed.returncode
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        log(f"eval tracking: {completed.stdout.strip()[-400:]}")
        return 0
    log(
        f"eval tracking: run {report.get('run_id')} "
        f"({report.get('scalar_count')} scalars, {report.get('case_rows')} case rows, "
        f"artifact {report.get('artifact')})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-label", required=True, help="e.g. v17")
    parser.add_argument("--task-id", required=True, help="g04 task id the launch printed")
    parser.add_argument("--job-id", required=True, help="W&B training job id the launch printed")
    parser.add_argument("--exec-dir", type=Path, required=True, help="docs/training/execution-YYYY-MM-DD")
    parser.add_argument(
        "--compare",
        type=parse_compare,
        action="append",
        default=[],
        metavar="LABEL=URI",
        help="repeatable; the default predecessor from prepare_release_eval is always added",
    )
    parser.add_argument(
        "--no-default-compare",
        action="store_true",
        help="drop the default predecessor comparison; only the --compare set runs",
    )
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-polls", type=int, default=240)
    parser.add_argument("--skip-wait", action="store_true", help="assume the job already finished")
    parser.add_argument(
        "--skip-track",
        action="store_true",
        help="do not publish the results root to W&B (offline runs)",
    )
    args = parser.parse_args(argv)
    try:
        release_label = _validate_token(args.release_label, "--release-label")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    exec_dir = args.exec_dir if args.exec_dir.is_absolute() else REPO / args.exec_dir
    if not exec_dir.is_dir():
        parser.error(f"--exec-dir does not exist: {exec_dir}")
    compares = resolve_compares(
        args.compare,
        (prep.DEFAULT_COMPARE_LABEL, prep.DEFAULT_COMPARE_URI),
        release_label,
        include_default=not args.no_default_compare,
    )
    run_name = f"{RUN_PREFIX}{args.task_id[:RUN_HASH_LENGTH]}"

    key = api_key()
    if not args.skip_wait:
        status = wait_for_job(key, args.job_id, args.poll_seconds, args.max_polls)
        if status not in JOB_SUCCESS_STATES:
            log(f"job did not finish successfully ({status}); stopping")
            return 1

    release_uri = prep.resolve_release_uri(run_name)
    commands = build_commands(release_label, release_uri, compares, exec_dir)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run_one, label, argv, key) for label, argv in commands]
        for future in futures:
            _, rc = future.result()
            if rc != 0:
                failures += 1

    failures += 1 if render_scoreboard(exec_dir, release_label) != 0 else 0
    if not args.skip_track:
        failures += 1 if track_eval_run(exec_dir, release_label) != 0 else 0
    log(f"PIPELINE DONE failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
