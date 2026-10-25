"""Keep a small private local copy of the newest W&B full-corpus LoRA checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import wandb
from training.launch.wandb_sft_g04 import TrainingAPI

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BACKUP_ROOT = ROOT / "ai/training/output/wandb_local_checkpoints_20261019"
ENTITY = "wutang"
PROJECT = "pixelated-empathy-kan28"
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELED", "CANCELLED", "ERROR"}
JOBS = {
    "v1": {
        "job_id": "2087f223-a48f-45f7-9999-daece5743bba",
        "artifact": "arc-sft-baseline-8ce8354c67ef-sft-resume",
    },
    "v2": {
        "job_id": "f914f0fb-f892-4e1a-8291-8cea3757274e",
        "artifact": "arc-sft-baseline-556695bfd5c4-sft-resume",
    },
    "v4": {
        "job_id": "24cc2d06-3ec3-4356-ad3a-179694dd86a7",
        "artifact": "arc-sft-baseline-30e158011725-sft-resume",
    },
    "v5-gemma4": {
        "job_id": "f41de815-2789-48d8-914d-23c3b9e256b3",
        "artifact": "arc-sft-baseline-5c243e8ea483",
    },
    "v5-nemotron": {
        "job_id": "b46708dd-9726-44f1-a699-62641ecc49e1",
        "artifact": "arc-sft-baseline-08ea5de19e58",
    },
    "v5-gemma4-trained": {
        "job_id": "f41de815-2789-48d8-914d-23c3b9e256b3",
        "artifact": "arc-sft-baseline-5c243e8ea483-sft-resume",
    },
    "v5-nemotron-trained": {
        "job_id": "b46708dd-9726-44f1-a699-62641ecc49e1",
        "artifact": "arc-sft-baseline-08ea5de19e58-sft-resume",
    },
}
KEEP_VERSIONS = 20


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def latest_artifact(api: wandb.Api, name: str) -> Any:
    collection = next(
        (item for item in api.artifact_collections(f"{ENTITY}/{PROJECT}", "lora", per_page=100) if item.name == name),
        None,
    )
    if collection is None:
        return None
    artifacts = list(collection.artifacts(per_page=200))
    if not artifacts:
        return None
    return max(artifacts, key=lambda item: int(item.version.removeprefix("v")))


def training_job_status(training_api: TrainingAPI, job_id: str) -> str:
    """Return UNKNOWN when the training-status service is unavailable."""
    try:
        job = training_api.request("GET", f"/preview/training-jobs/{job_id}")
    except Exception as exc:
        logging.warning("job status unavailable for %s (%s)", job_id, type(exc).__name__)
        return "UNKNOWN"
    if not isinstance(job, dict):
        return "UNKNOWN"
    return str(job.get("status", job.get("state", "unknown"))).upper()


def backup_artifact(artifact: Any, backup_root: Path, label: str, collection_name: str) -> dict[str, Any]:
    version = artifact.version
    job_root = backup_root / label
    job_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(job_root, 0o700)
    final = job_root / version
    if final.exists():
        manifest_path = final / "manifest.json"
        if manifest_path.exists():
            cached_manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
            return cached_manifest
        raise RuntimeError(f"existing local backup lacks a manifest: {label}/{version}")

    temporary = job_root / f".{version}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(mode=0o700)
    try:
        artifact.download(root=str(temporary))
        entries = []
        for remote_file in artifact.files():
            path = temporary / remote_file.name
            if not path.is_file() or path.stat().st_size != remote_file.size:
                raise RuntimeError(f"download verification failed for {label}/{version}/{remote_file.name}")
            os.chmod(path, 0o600)
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            entries.append({"name": remote_file.name, "size_bytes": path.stat().st_size, "sha256": digest})
        manifest = {
            "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "label": label,
            "artifact_uri": f"wandb-artifact:///{ENTITY}/{PROJECT}/{collection_name}:{version}",
            "version": version,
            "completed_batches": artifact.metadata.get("sft_completed_batches"),
            "artifact_size_bytes": artifact.size,
            "local_files": entries,
        }
        atomic_json(temporary / "manifest.json", manifest)
        temporary.rename(final)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prune_local_backups(backup_root: Path, label: str, keep_versions: int = KEEP_VERSIONS) -> None:
    job_root = backup_root / label
    versions = sorted(
        (path for path in job_root.iterdir() if path.is_dir() and path.name.startswith("v")),
        key=lambda path: int(path.name.removeprefix("v")),
        reverse=True,
    )
    for old_path in versions[keep_versions:]:
        shutil.rmtree(old_path)


def update_index(backup_root: Path) -> None:
    checkpoints = {}
    for label in JOBS:
        latest_path = backup_root / f"{label}-latest.json"
        if not latest_path.exists():
            continue
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        version = latest.get("checkpoint_version")
        if not version:
            continue
        version_manifest_path = backup_root / label / version / "manifest.json"
        version_manifest = json.loads(version_manifest_path.read_text(encoding="utf-8"))
        checkpoints[label] = {
            "artifact_uri": version_manifest["artifact_uri"],
            "version": version,
            "completed_batches": version_manifest["completed_batches"],
            "local_path": str((backup_root / label / version).resolve()),
            "files": version_manifest["local_files"],
        }
    atomic_json(
        backup_root / "manifest.json",
        {
            "schema_version": 1,
            "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "purpose": "private local preservation of latest active W&B full-corpus LoRA checkpoints",
            "security": "mode 0700 directory, mode 0600 files; do not commit or publish",
            "retention": (
                f"Keep the {KEEP_VERSIONS} newest local versioned adapters per run; no W&B artifacts are deleted."
            ),
            "checkpoints": checkpoints,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--poll-interval", type=int, default=300)
    parser.add_argument("--max-polls", type=int, default=432)
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
        for label, task in selected_jobs.items():
            status = training_job_status(training_api, task["job_id"])
            try:
                artifact = latest_artifact(artifact_api, task["artifact"])
                artifact_lookup_error = None
            except Exception as exc:
                logging.warning("artifact lookup unavailable for %s (%s)", label, type(exc).__name__)
                artifact = None
                artifact_lookup_error = type(exc).__name__
            checkpoint = None
            if artifact is not None:
                checkpoint = backup_artifact(artifact, args.backup_root, label, task["artifact"])
                prune_local_backups(args.backup_root, label)
            snapshot = {
                "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "job_id": task["job_id"],
                "status": status,
                "checkpoint_version": (checkpoint or {}).get("version"),
                "completed_batches": (checkpoint or {}).get("completed_batches"),
                "backup_path": str(args.backup_root / label / checkpoint["version"]) if checkpoint else None,
                "artifact_lookup_error": artifact_lookup_error,
            }
            observation_path = args.backup_root / f"{label}-observation.json"
            atomic_json(observation_path, snapshot)
            if checkpoint is not None:
                atomic_json(args.backup_root / f"{label}-latest.json", snapshot)
            logging.info(
                "%s status=%s checkpoint=%s batches=%s",
                label,
                status,
                snapshot["checkpoint_version"],
                snapshot["completed_batches"],
            )
            if status not in TERMINAL_STATES:
                all_terminal = False
        update_index(args.backup_root)
        if all_terminal:
            return 0
        if poll_index + 1 < args.max_polls:
            time.sleep(args.poll_interval)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
