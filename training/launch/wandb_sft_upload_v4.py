"""Upload the v4 full-corpus train payload to W&B as a dataset artifact.

Replicates the v2 upload method (training.launch.wandb_sft_g04) but stops
after _upload_artifact: no model or training job is created here — that is
the separate launch step. Idempotent via the launcher state file.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "ai"))

import training.launch.wandb_sft_g04 as g04

MANIFEST = ROOT / "docs/training/execution-2026-10-19/wandb-full-corpus-current-master-20261019-v4.json"
STATE_DIR = ROOT / "docs/training/execution-2026-10-19/wandb-sft-state"
WORK_DIR = ROOT / "docs/training/execution-2026-10-19/wandb-sft-artifacts"
ARM = "baseline"

manifest, payload = g04.resolve_train_payload(MANIFEST, ARM)
config = g04.BASELINE_CONFIG
task_id = g04.canonical_hash(
    {"release": manifest["release_id"], "payload": payload.sha256, "arm": ARM, "config": config}
)
state_path = STATE_DIR / f"wandb-sft-{task_id}.json"
existing = g04._load_state(state_path, task_id, payload.sha256)
if existing.get("training_data_url"):
    sys.stdout.write(
        json.dumps(
            {"already_uploaded": True, "task_id": task_id, "training_data_url": existing["training_data_url"]},
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(0)

api_key = os.environ.get("WANDB_API_KEY")
if not api_key:
    raise SystemExit("WANDB_API_KEY unset")
api = g04.TrainingAPI(api_key)


def upload(directory: Path, artifact_name: str) -> str:
    import wandb

    run = None
    try:
        run = wandb.init(entity=g04.ENTITY, project=g04.PROJECT, job_type="sft-data-upload", mode="online")
        artifact = wandb.Artifact(
            artifact_name,
            type="dataset",
            metadata={
                "release_id": manifest["release_id"],
                "payload_sha256": payload.sha256,
                "total_len": payload.rows,
            },
        )
        artifact.add_file(str(directory / "trajectories.jsonl"), name="trajectories.jsonl")
        artifact.add_file(str(directory / "metadata.json"), name="metadata.json")
        logged = run.log_artifact(artifact)
        logged.wait()
        version = getattr(logged, "version", "v0")
        return f"wandb-artifact:///{g04.ENTITY}/{g04.PROJECT}/{artifact_name}:{version}"
    except Exception as exc:
        raise g04.QualificationError(
            f"W&B artifact upload failed ({type(exc).__name__}); reconcile artifact before resuming"
        ) from None
    finally:
        if run is not None:
            run.finish()


artifact = g04.make_training_artifact(
    payload, manifest, config, WORK_DIR / manifest["release_id"] / ARM
)
plan = g04.TrainingPlan(manifest, payload, config, artifact, task_id)
request = g04.ExecutionRequest(
    manifest_path=MANIFEST,
    arm=ARM,
    state_dir=STATE_DIR,
    work_dir=WORK_DIR,
    api=api,
    api_key=api_key,
    artifact_upload=upload,
    max_polls=1,
    poll_interval=30,
)
g04._verify_remote_access(request)
state = existing if existing else g04._initial_state(request, plan)
g04.atomic_json(state_path, state)
artifact_uri = g04._upload_artifact(request, state, plan)
sys.stdout.write(
    json.dumps(
        {
            "task_id": task_id,
            "phase": state["phase"],
            "training_data_url": artifact_uri,
            "payload_sha256": payload.sha256,
            "rows": payload.rows,
            "state_path": str(state_path),
        },
        indent=2,
    )
    + "\n"
)
