"""Track full-corpus W&B jobs and evaluate their final retained resume artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import wandb
from training.launch.wandb_sft_g04 import TrainingAPI

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)
ENTITY = "wutang"
PROJECT = "pixelated-empathy-kan28"
POLL_SECONDS = 300
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELED", "CANCELLED", "ERROR"}
CASES = ROOT / "docs/training/execution-2026-10-17/scenario-dev-v1.jsonl"
RUNNER = ROOT / "ai/training/run_wandb_paired_eval.py"
JOBS: dict[str, dict[str, Any]] = {
    "v1": {
        "job_id": "2087f223-a48f-45f7-9999-daece5743bba",
        "state": ROOT / "docs/training/execution-2026-10-18/wandb-full-corpus-state/"
        "wandb-sft-8ce8354c67ef10915f78ebd385950aaed8fcac505578645dba907d330befbe91.json",
        "config": ROOT / "docs/training/execution-2026-10-18/inference-configs/"
        "wandb-full-corpus-v1-vs-base-2048-512.json",
        "resume_artifact": "arc-sft-baseline-8ce8354c67ef-sft-resume",
        "results": ROOT / "docs/training/execution-2026-10-18/results/"
        "wandb-full-corpus-v1-final-paired",
    },
    "v2": {
        "job_id": "f914f0fb-f892-4e1a-8291-8cea3757274e",
        "state": ROOT / "docs/training/execution-2026-10-18/wandb-full-corpus-v2-state/"
        "wandb-sft-556695bfd5c4dcd55a12d698290840432a5e0c193d5a3180aef274e2ed8e1790.json",
        "config": ROOT / "docs/training/execution-2026-10-18/inference-configs/"
        "wandb-full-corpus-v2-vs-base-2048-512.json",
        "resume_artifact": "arc-sft-baseline-556695bfd5c4-sft-resume",
        "results": ROOT / "docs/training/execution-2026-10-18/results/"
        "wandb-full-corpus-v2-final-paired",
    },
}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def latest_resume_artifact(api: wandb.Api, name: str) -> dict[str, Any] | None:
    collection = next(
        (
            item
            for item in api.artifact_collections(f"{ENTITY}/{PROJECT}", "lora", per_page=100)
            if item.name == name
        ),
        None,
    )
    if collection is None:
        return None
    artifacts = list(collection.artifacts(per_page=100))
    if not artifacts:
        return None
    artifact = max(artifacts, key=lambda item: int(item.version.removeprefix("v")))
    return {
        "name": name,
        "version": artifact.version,
        "uri": f"wandb-artifact:///{ENTITY}/{PROJECT}/{name}:{artifact.version}",
        "created_at": artifact.created_at,
        "size_bytes": artifact.size,
        "completed_batches": artifact.metadata.get("sft_completed_batches"),
        "internal_checkpoint": artifact.metadata.get("sft_internal_checkpoint"),
    }


def run_final_eval(label: str, config_path: Path, result_dir: Path, model_uri: str) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    next(arm for arm in config["arms"] if arm["id"] == "sft")["model"] = model_uri
    config["run_id"] = f"wandb-full-corpus-{label}-final-vs-base-2048-512"
    result_dir.mkdir(parents=True, exist_ok=True)
    run_config = result_dir / "run-config.json"
    run_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "ai") + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--cases",
            str(CASES),
            "--config",
            str(run_config),
            "--results-dir",
            str(result_dir),
            "--arm",
            "both",
            "--max-cases",
            "5",
            "--max-input-tokens",
            "2048",
            "--max-tool-rounds",
            "2",
        ],
        check=False,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(result.stdout)
    sys.stdout.flush()
    if result.stderr:
        logger.warning("paired eval stderr: %s", result.stderr.rstrip())
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-interval", type=int, default=POLL_SECONDS)
    parser.add_argument("--max-polls", type=int, default=432)
    args = parser.parse_args()
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise ValueError("WANDB_API_KEY must be provided through the environment")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    training_api = TrainingAPI(api_key)
    public_api = wandb.Api(timeout=30)
    log_path = ROOT / "docs/training/execution-2026-10-18/live-monitor/full-corpus-jobs.jsonl"

    for poll_index in range(args.max_polls):
        all_terminal = True
        terminal_without_artifact = False
        for label, task in JOBS.items():
            job_payload = training_api.request("GET", f"/preview/training-jobs/{task['job_id']}")
            job = job_payload if isinstance(job_payload, dict) else {}
            status = str(job.get("status", job.get("state", "unknown"))).upper()
            artifact = latest_resume_artifact(public_api, task["resume_artifact"])
            snapshot = {
                "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "release": label,
                "job_id": task["job_id"],
                "status": status,
                "job_updated_at": job.get("updated_at"),
                "latest_resume_artifact": artifact,
            }
            task["results"].mkdir(parents=True, exist_ok=True)
            atomic_json(task["results"] / "live-status.json", snapshot)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
            logger.info(
                f"{label} status={status} "
                f"artifact={(artifact or {}).get('version', 'none')} "
                f"batches={(artifact or {}).get('completed_batches', 'unknown')}"
            )
            if status not in TERMINAL_STATES:
                all_terminal = False
                continue
            if not artifact:
                logger.error("%s reached terminal state without a resume artifact", label)
                terminal_without_artifact = True
                continue
            manifest_path = task["results"] / "manifest.json"
            if manifest_path.exists():
                prior = json.loads(manifest_path.read_text(encoding="utf-8"))
                sft_arm: dict[str, Any] = next((arm for arm in prior.get("arms", []) if arm.get("id") == "sft"), {})
                if prior.get("status") == "complete" and sft_arm.get("model") == artifact["uri"]:
                    logger.info("%s final artifact already evaluated", label)
                    continue
            if run_final_eval(label, task["config"], task["results"], artifact["uri"]):
                return 1
        if all_terminal:
            return 1 if terminal_without_artifact else 0
        if poll_index + 1 < args.max_polls:
            time.sleep(args.poll_interval)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
