"""Poll a W&B full-corpus job and run its bounded paired eval at step 1."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.launch.wandb_sft_g04 import TrainingAPI

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = ROOT / (
    "docs/training/execution-2026-10-18/wandb-full-corpus-v2-state/"
    "wandb-sft-556695bfd5c4dcd55a12d698290840432a5e0c193d5a3180aef274e2ed8e1790.json"
)
DEFAULT_CONFIG = ROOT / (
    "docs/training/execution-2026-10-18/inference-configs/wandb-full-corpus-v2-vs-base-2048-512.json"
)
DEFAULT_RESULTS = ROOT / "docs/training/execution-2026-10-18/results/wandb-full-corpus-v2-step1-paired"
POLL_INTERVAL_SECONDS = 60
MAX_POLLS = 1_440
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELED", "CANCELLED", "ERROR"}
STEP_ONE = 1
HTTP_SUCCESS_STATUS = 0


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def load_state(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("release_id") not in {
        "wandb-full-corpus-20261018-v1",
        "wandb-full-corpus-20261018-v2",
    }:
        raise ValueError("state file is not for a versioned full-corpus experiment")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL_SECONDS)
    parser.add_argument("--max-polls", type=int, default=MAX_POLLS)
    args = parser.parse_args()
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise ValueError("WANDB_API_KEY must be provided through the environment")
    state = load_state(args.state)
    api = TrainingAPI(api_key)
    for poll_index in range(args.max_polls):
        job = api.request("GET", f"/preview/training-jobs/{state['training_job_id']}")
        checkpoint_payload = api.request("GET", f"/preview/models/{state['model_id']}/checkpoints")
        checkpoints = checkpoint_payload.get("data", []) if isinstance(checkpoint_payload, dict) else checkpoint_payload
        state["last_status"] = job.get("status", job.get("state", "unknown"))
        state["last_status_response"] = {
            key: job[key]
            for key in ("id", "status", "state", "created_at", "updated_at", "completed_at", "step", "total_steps")
            if key in job
        }
        state["checkpoints"] = checkpoints
        state["last_polled_at_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        atomic_write(args.state, state)
        steps = sorted(checkpoint.get("step", -1) for checkpoint in checkpoints)
        sys.stdout.write(
            json.dumps({"status": state["last_status"], "checkpoint_steps": steps, "poll": poll_index + 1}) + "\n"
        )
        sys.stdout.flush()
        trained_checkpoint = next(
            (checkpoint for checkpoint in checkpoints if checkpoint.get("step", -1) >= STEP_ONE), None
        )
        if trained_checkpoint is not None:
            checkpoint_model = (
                "wandb-artifact:///wutang/pixelated-empathy-kan28/"
                f"{trained_checkpoint['artifact_name']}:step{trained_checkpoint['step']}"
            )
            config = json.loads(args.config.read_text(encoding="utf-8"))
            sft_arm = next(arm for arm in config["arms"] if arm["id"] == "sft")
            sft_arm["model"] = checkpoint_model
            config["run_id"] = (
                f"{state['release_id']}-step{trained_checkpoint['step']}-vs-base-2048-512"
            )
            args.results_dir.mkdir(parents=True, exist_ok=True)
            run_config = args.results_dir / "run-config.json"
            run_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT / "ai") + os.pathsep + environment.get("PYTHONPATH", "")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "ai/training/run_wandb_paired_eval.py"),
                    "--cases",
                    str(ROOT / "docs/training/execution-2026-10-17/scenario-dev-v1.jsonl"),
                    "--config",
                    str(run_config),
                    "--results-dir",
                    str(args.results_dir),
                    "--max-input-tokens",
                    "2048",
                    "--max-tool-rounds",
                    "2",
                ],
                check=False,
                env=environment,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            state["paired_eval"] = {
                "checkpoint_step": trained_checkpoint["step"],
                "checkpoint_artifact_name": trained_checkpoint["artifact_name"],
                "results_dir": str(args.results_dir.relative_to(ROOT)),
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "completed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }
            atomic_write(args.state, state)
            sys.stdout.write(result.stdout)
            sys.stdout.flush()
            return result.returncode
        if state["last_status"] in TERMINAL_STATES:
            return HTTP_SUCCESS_STATUS if state["last_status"] == "COMPLETED" else 1
        if poll_index + 1 < args.max_polls:
            time.sleep(args.poll_interval)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
