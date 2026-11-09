#!/usr/bin/env python3
"""Run a verified full-corpus release through W&B Serverless SFT.

One launcher for every release. The only per-release input is the launch config
produced by ``make_launch_config.py``: it carries the manifest path, the train
payload identity, and the training config, so nothing about a release needs to
be restated in code. Earlier releases had a launcher apiece that differed only
in three constants, which meant a release could silently train a payload the
config no longer described.

The release is located by its short tag (``--release v16``) anywhere under
``docs/training/execution-*``, or given explicitly with ``--config``. Run
state, artifacts, and the job's work directory live beside that config.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wandb_sft_g04 as g04

_MODEL = g04.MODEL_ID
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXECUTION_GLOB = "docs/training/execution-*"


@dataclass(frozen=True)
class Campaign:
    config_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    payload: g04.Payload
    config: dict[str, Any]
    task_id: str


def find_config(release: str, root: Path = _REPO_ROOT) -> Path:
    """Return the single launch config for ``release`` under the docs submodule."""
    matches = sorted(root.glob(f"{_EXECUTION_GLOB}/{release}-launch-config.json"))
    if not matches:
        raise g04.QualificationError(f"no launch config found for release {release!r} under {_EXECUTION_GLOB}")
    if len(matches) > 1:
        found = ", ".join(str(path.relative_to(root)) for path in matches)
        raise g04.QualificationError(f"release {release!r} has more than one launch config: {found}")
    return matches[0]


def _expected_steps(train_rows: int, batch_size: int) -> int:
    """Return the optimiser step count a launch config must schedule."""
    if train_rows <= 0 or batch_size <= 0:
        raise g04.QualificationError("launch config needs a positive row count and batch size")
    return math.ceil(train_rows / batch_size)


def _ensure_api_key() -> str:
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    env_file = _REPO_ROOT / ".env"
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("WANDB_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise g04.QualificationError("WANDB_API_KEY not found in environment or .env")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--release", help="short release tag, e.g. v16")
    target.add_argument("--config", type=Path, help="explicit launch config path")
    parser.add_argument("--arm", choices=("baseline", "data-ablation"), default="baseline")
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--output", type=Path, help="artifact work directory (default: beside the config)")
    parser.add_argument("--state-dir", type=Path, help="run state directory (default: beside the config)")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--polls", type=int, default=1)
    parser.add_argument("--poll-interval", type=float, default=30)
    return parser.parse_args(argv)


def _load_campaign(args: argparse.Namespace) -> Campaign:
    config_path = args.config or find_config(args.release)
    if args.model not in g04.ALLOWED_MODEL_IDS:
        raise g04.QualificationError(f"model {args.model!r} is not on the launcher allowlist")
    with config_path.open() as stream:
        launch = json.load(stream)
    config: dict[str, Any] = launch["config"]
    manifest_path = Path(launch["manifest"])
    manifest, payload = g04.resolve_train_payload(manifest_path, args.arm)
    if payload.sha256 != launch["train_sha256"] or payload.rows != launch["train_rows"]:
        raise g04.QualificationError(f"{manifest['release_id']} train payload identity disagrees with launch config")
    expected_steps = _expected_steps(payload.rows, config["batch_size"])
    if len(config["learning_rate"]) != expected_steps:
        raise g04.QualificationError(
            f"{manifest['release_id']} learning-rate schedule has {len(config['learning_rate'])} steps, "
            f"payload needs {expected_steps}; regenerate the launch config"
        )
    task_id = g04.canonical_hash(
        {
            "release": manifest["release_id"],
            "payload": payload.sha256,
            "arm": args.arm,
            "model": args.model,
            "config": config,
        }
    )
    return Campaign(config_path, manifest_path, manifest, payload, config, task_id)


def _artifact_uploader(manifest: dict[str, Any], payload: g04.Payload, api_key: str) -> Callable[[Path, str], str]:
    wandb = importlib.import_module("wandb")
    os.environ["WANDB_API_KEY"] = api_key

    def upload(directory: Path, artifact_name: str) -> str:
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

    return upload


def _dry_run(args: argparse.Namespace, campaign: Campaign) -> int:
    work_path = _work_dir(args, campaign) / args.arm
    artifact = g04.make_training_artifact(campaign.payload, campaign.manifest, campaign.config, work_path)
    lr = campaign.config["learning_rate"]
    report = {
        "mode": "dry-run",
        "task_id": campaign.task_id,
        "release_id": campaign.manifest["release_id"],
        "config": str(campaign.config_path),
        "payload_sha256": campaign.payload.sha256,
        "rows": campaign.payload.rows,
        "model": args.model,
        "batch_size": campaign.config["batch_size"],
        "learning_rate_schedule": {
            "length": len(lr),
            "first": lr[0],
            "peak": max(lr),
            "last": lr[-1],
        },
        "artifact_bytes": artifact["artifact_bytes"],
        "no_upload_or_job_created": True,
    }
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0


def _work_dir(args: argparse.Namespace, campaign: Campaign) -> Path:
    return args.output or campaign.config_path.parent / "wandb-sft-artifacts"


def _state_dir(args: argparse.Namespace, campaign: Campaign) -> Path:
    return args.state_dir or campaign.config_path.parent / "wandb-sft-state"


def _execute(args: argparse.Namespace, campaign: Campaign) -> int:
    api_key = _ensure_api_key()
    state_dir = _state_dir(args, campaign)
    request = g04.ExecutionRequest(
        manifest_path=campaign.manifest_path,
        arm=args.arm,
        state_dir=state_dir,
        work_dir=_work_dir(args, campaign),
        api=g04.TrainingAPI(api_key),
        api_key=api_key,
        artifact_upload=_artifact_uploader(campaign.manifest, campaign.payload, api_key),
        max_polls=args.polls,
        poll_interval=args.poll_interval,
    )

    work_path = _work_dir(args, campaign) / campaign.manifest["release_id"] / args.arm
    artifact = g04.make_training_artifact(campaign.payload, campaign.manifest, campaign.config, work_path)
    plan = g04.TrainingPlan(campaign.manifest, campaign.payload, campaign.config, artifact, campaign.task_id)

    state_path = state_dir / f"wandb-sft-{campaign.task_id}.json"
    state = g04._load_state(state_path, campaign.task_id, campaign.payload.sha256)
    g04._verify_remote_access(request)
    if not state:
        state = g04._initial_state(request, plan)
        g04.atomic_json(state_path, state)
    artifact_uri = g04._upload_artifact(request, state, plan)
    model_id = g04._create_model(request, state, campaign.task_id)
    job_id = g04._create_job(request, state, plan, model_id, artifact_uri)
    g04._poll_job(request, state, job_id, campaign.task_id)
    sys.stdout.write(
        json.dumps(
            {
                "task_id": campaign.task_id,
                "phase": state["phase"],
                "training_job_id": job_id,
                "model_id": model_id,
                "artifact_uri": artifact_uri,
                "state_path": str(state_path),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    campaign = _load_campaign(args)
    if not args.execute:
        return _dry_run(args, campaign)
    return _execute(args, campaign)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except g04.QualificationError as exc:
        sys.stderr.write(f"qualification failed: {exc}\n")
        raise SystemExit(2) from None
