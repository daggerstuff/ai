#!/usr/bin/env python3
"""Preserve checkpoints from the bounded Gemma/Nemotron W&B SFT comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import wandb
from ai.training.launch.wandb_sft_g04 import TrainingAPI

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / "ai/training/output/dual_model_sft_20261020"
DEFAULT_BACKUP_ROOT = STATE_DIR / "checkpoints"
ENTITY = "wutang"
PROJECT = "pixelated-empathy-kan28"
TERMINAL_STATES = {"COMPLETED", "SUCCEEDED", "FAILED", "CANCELED", "CANCELLED", "ERROR"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def backup_checkpoint(
    api: wandb.Api,
    backup_root: Path,
    label: str,
    checkpoint: dict[str, Any],
    campaign_id: str = "dual-model-agent-tool-sft-20261020-v1",
) -> dict[str, Any]:
    step = checkpoint.get("step")
    version = checkpoint.get("version")
    artifact_name = checkpoint.get("artifact_name")
    if not isinstance(step, int) or not isinstance(version, int) or not isinstance(artifact_name, str):
        raise ValueError("checkpoint response lacks a valid step, version, or artifact name")
    destination = backup_root / label / f"step-{step:08d}-v{version}"
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["local_files"]:
            path = destination / entry["name"]
            if not path.is_file() or path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
                raise RuntimeError("previously mirrored checkpoint failed local integrity verification")
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination.parent, 0o700)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise RuntimeError("unfinished checkpoint backup exists; refusing to overwrite it")
    temporary.mkdir(mode=0o700)
    artifact_uri = f"{ENTITY}/{PROJECT}/{artifact_name}:step{step}"
    try:
        artifact = api.artifact(artifact_uri)
        artifact.download(root=str(temporary))
        files = []
        for remote_file in artifact.files():
            path = temporary / remote_file.name
            if not path.is_file() or path.stat().st_size != remote_file.size:
                raise RuntimeError("downloaded checkpoint file size does not match W&B metadata")
            os.chmod(path, 0o600)
            files.append({"name": remote_file.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
        manifest = {
            "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "campaign_id": campaign_id,
            "label": label,
            "artifact_uri": f"wandb-artifact:///{artifact_uri}",
            "artifact_version": artifact.version,
            "checkpoint_step": step,
            "checkpoint_version": version,
            "artifact_size_bytes": artifact.size,
            "local_files": files,
        }
        atomic_json(temporary / "manifest.json", manifest)
        temporary.rename(destination)
        return manifest
    except Exception:
        # Preserve failed downloads for inspection; never overwrite a partial tree.
        raise


def poll_model(
    training_api: TrainingAPI,
    public_api: wandb.Api,
    backup_root: Path,
    label: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    job = training_api.request("GET", f"/preview/training-jobs/{state['training_job_id']}")
    route = f"/preview/models/{state['model_id']}/checkpoints"
    checkpoints = []
    cursors = set()
    path = route
    while True:
        response = training_api.request("GET", path)
        checkpoints.extend(response.get("data", []) if isinstance(response, dict) else response)
        if not isinstance(response, dict) or not response.get("has_more"):
            break
        cursor = response.get("last_id")
        if not isinstance(cursor, str) or not cursor or cursor in cursors:
            raise RuntimeError("checkpoint pagination has no advancing cursor")
        cursors.add(cursor)
        path = f"{route}?{urlencode({'after': cursor})}"
    backed_up = []
    for checkpoint in sorted(checkpoints, key=lambda item: (item.get("step", -1), item.get("version", -1))):
        manifest = backup_checkpoint(
            public_api,
            backup_root,
            label,
            checkpoint,
            state.get("release_id") or "dual-model-agent-tool-sft-20261020-v1",
        )
        backed_up.append(
            {
                "step": manifest["checkpoint_step"],
                "version": manifest["checkpoint_version"],
                "bytes": manifest["artifact_size_bytes"],
            }
        )
    return {
        "job_id": state["training_job_id"],
        "model_id": state["model_id"],
        "status": str(job.get("status", job.get("state", "unknown"))).upper(),
        "updated_at": job.get("updated_at"),
        "checkpoints": backed_up,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument(
        "--state-file",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help="watch an explicit campaign state file (repeat for each model); defaults to the Oct 20 pair",
    )
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--max-polls", type=int, default=1440)
    args = parser.parse_args()
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise ValueError("WANDB_API_KEY must be provided through the environment")
    training_api = TrainingAPI(api_key)
    public_api = wandb.Api(api_key=api_key)
    jobs = {}
    state_paths = {label: STATE_DIR / f"{label}.json" for label in ("gemma4", "nemotron35")}
    if args.state_file:
        state_paths = {}
        for value in args.state_file:
            label, separator, path_value = value.partition("=")
            if not separator or label not in {"gemma4", "nemotron35"} or not path_value:
                raise ValueError("--state-file must use gemma4=PATH or nemotron35=PATH")
            if label in state_paths:
                raise ValueError(f"duplicate checkpoint state label: {label}")
            state_paths[label] = Path(path_value)
    for label, state_path in state_paths.items():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not state.get("training_job_id") or not state.get("model_id"):
            raise ValueError(f"{label} job state has no model or training-job UUID")
        jobs[label] = state
    campaign_ids = {state.get("release_id") for state in jobs.values() if state.get("release_id")}
    campaign_id = next(iter(campaign_ids)) if len(campaign_ids) == 1 else "dual-model-agent-tool-sft-20261020-v1"

    args.backup_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.backup_root, 0o700)
    status_path = args.backup_root / "watch-status.json"
    for poll in range(args.max_polls):
        all_terminal = True
        status = {
            "campaign_id": campaign_id,
            "polled_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "models": {},
        }
        for label, state in jobs.items():
            try:
                observation = poll_model(training_api, public_api, args.backup_root, label, state)
            except Exception as exc:
                # Keep polling both jobs after transport/export failures. Never
                # turn an observation failure into a terminal training status.
                observation = {
                    "job_id": state["training_job_id"],
                    "model_id": state["model_id"],
                    "status": "UNKNOWN",
                    "observation_error": type(exc).__name__,
                    "checkpoints": [],
                }
            terminal = observation["status"] in TERMINAL_STATES
            if observation["status"] in {"COMPLETED", "SUCCEEDED"}:
                # A completed job may publish its trained adapter after the
                # terminal status. Keep polling until an export is mirrored.
                terminal = any(checkpoint["step"] > 0 for checkpoint in observation["checkpoints"])
            all_terminal = all_terminal and terminal
            status["models"][label] = observation
        atomic_json(status_path, status)
        sys.stdout.write(
            json.dumps(
                {
                    "poll": poll + 1,
                    "models": {
                        k: {"status": v["status"], "checkpoint_steps": [c["step"] for c in v["checkpoints"]]}
                        for k, v in status["models"].items()
                    },
                }
            )
            + "\n"
        )
        sys.stdout.flush()
        if all_terminal:
            return 0
        time.sleep(args.poll_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
