"""Keep private local rolling copies of active W&B full-corpus resume state."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wandb.sdk.artifacts.artifact_file_cache import get_artifact_file_cache

import wandb
from training.backup_wandb_full_corpus_lora import (
    DEFAULT_BACKUP_ROOT,
    backup_artifact,
    prune_local_backups,
    training_job_status,
)
from training.launch.wandb_sft_g04 import TrainingAPI

ROOT = Path(__file__).resolve().parents[2]
ENTITY = "wutang"
PROJECT = "pixelated-empathy-kan28"
JOBS = {
    "v1": {
        "job_id": "2087f223-a48f-45f7-9999-daece5743bba",
        "artifact": "arc-sft-baseline-8ce8354c67ef-train-state-sft-resume",
    },
    "v2": {
        "job_id": "f914f0fb-f892-4e1a-8291-8cea3757274e",
        "artifact": "arc-sft-baseline-556695bfd5c4-train-state-sft-resume",
    },
    "v4": {
        "job_id": "24cc2d06-3ec3-4356-ad3a-179694dd86a7",
        "artifact": "arc-sft-baseline-30e158011725-train-state-sft-resume",
    },
    "v5-gemma4": {
        "job_id": "f41de815-2789-48d8-914d-23c3b9e256b3",
        "artifact": "arc-sft-baseline-5c243e8ea483",
    },
    "v5-nemotron": {
        "job_id": "b46708dd-9726-44f1-a699-62641ecc49e1",
        "artifact": "arc-sft-baseline-08ea5de19e58",
    },
}
KEEP_VERSIONS = 2


def latest_train_state_artifact(api: wandb.Api, name: str) -> Any | None:
    collection = next(
        (
            item
            for item in api.artifact_collections(f"{ENTITY}/{PROJECT}", "train-state", per_page=100)
            if item.name == name or item.name.startswith(name + "-")
        ),
        None,
    )
    if collection is None:
        return None
    artifacts = list(collection.artifacts(per_page=200))
    if not artifacts:
        return None
    return max(artifacts, key=lambda item: int(item.version.removeprefix("v")))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--poll-interval", type=int, default=300)
    parser.add_argument("--max-polls", type=int, default=432)
    parser.add_argument("--keep-versions", type=int, default=KEEP_VERSIONS)
    parser.add_argument("--only", nargs="+", choices=sorted(JOBS), default=None)
    args = parser.parse_args()
    selected_jobs = {label: JOBS[label] for label in args.only} if args.only else JOBS
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise ValueError("WANDB_API_KEY must be provided through the environment")
    args.backup_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.backup_root, 0o700)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    training_api = TrainingAPI(api_key)
    artifact_api = wandb.Api(timeout=30)

    for poll_index in range(args.max_polls):
        all_terminal = True
        manifest_path = args.backup_root / "train-state-manifest.json"
        if manifest_path.exists():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoints: dict[str, Any] = dict(existing_manifest.get("checkpoints", {}))
        else:
            checkpoints = {}
        for label, task in selected_jobs.items():
            status = training_job_status(training_api, task["job_id"])
            try:
                artifact = latest_train_state_artifact(artifact_api, task["artifact"])
                artifact_lookup_error = None
            except Exception as exc:
                logging.warning("train-state artifact lookup unavailable for %s (%s)", label, type(exc).__name__)
                artifact = None
                artifact_lookup_error = type(exc).__name__
            backup_label = f"{label}-train-state"
            checkpoint = None
            if artifact is not None:
                checkpoint = backup_artifact(artifact, args.backup_root, backup_label, artifact.collection.name)
                prune_local_backups(args.backup_root, backup_label, keep_versions=args.keep_versions)
                with suppress(Exception):
                    get_artifact_file_cache().cleanup(target_size=2 * 1024 * 1024 * 1024, remove_temp=True)
                checkpoints[label] = {
                    "artifact_uri": checkpoint["artifact_uri"],
                    "version": checkpoint["version"],
                    "completed_batches": checkpoint["completed_batches"],
                    "size_bytes": checkpoint["artifact_size_bytes"],
                    "local_path": str(args.backup_root / backup_label / checkpoint["version"]),
                }
                atomic_json(
                    args.backup_root / f"{backup_label}-latest.json",
                    {
                        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                        "job_id": task["job_id"],
                        "status": status,
                        **checkpoints[label],
                    },
                )
            atomic_json(
                args.backup_root / f"{backup_label}-observation.json",
                {
                    "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "job_id": task["job_id"],
                    "status": status,
                    "checkpoint_version": (checkpoint or {}).get("version"),
                    "completed_batches": (checkpoint or {}).get("completed_batches"),
                    "artifact_lookup_error": artifact_lookup_error,
                },
            )
            logging.info(
                "%s status=%s train_state=%s batches=%s",
                label,
                status,
                (checkpoint or {}).get("version"),
                (checkpoint or {}).get("completed_batches"),
            )
            if status not in {"COMPLETED", "FAILED", "CANCELED", "CANCELLED", "ERROR"}:
                all_terminal = False
        atomic_json(
            manifest_path,
            {
                "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "purpose": "private rolling backups of W&B full-corpus training resume state",
                "retention": f"Keep the {args.keep_versions} newest versions per job; no remote artifacts are deleted.",
                "checkpoints": checkpoints,
            },
        )
        if all_terminal:
            return 0
        if poll_index + 1 < args.max_polls:
            time.sleep(args.poll_interval)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
