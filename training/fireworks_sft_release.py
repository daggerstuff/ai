#!/usr/bin/env python3
"""Train the exported release selection on the Fireworks Training API.

Fireworks turned self-serve dataset uploads off in October 2026 while it
investigated a security incident, which retires both ``firectl sftj create``
and the Batch-Inference dataset path. The Training API is what is left: it
reads datums out of this process and streams batches at a trainer, so no
dataset file is ever stored on Fireworks.

The trainer is reachable two ways, and only one of them is free. The managed
path (``managed_config``) provisions a dedicated ``rlorTrainerJobs`` trainer and
lets the backend pick the accelerator — it chose ``NVIDIA_B200_180GB``, which
bills against the account. The serverless gateway at
``/training/v1/serverless`` runs the same trainer pooled and is free through
2026-11-30, so every run goes through it and ``assert_serverless`` refuses
anything else.

The trainer itself is algorithm-agnostic, so the release recipe replays
unchanged — ``training.tinker_sft_release.train`` owns the loop, the LR trapezoid
and the checkpoint saves. This module swaps the transport for
``FiretitanServiceClient``, defaults to the Muse-Glimmer base, and adds the
Fireworks reporting and promotion steps.

The base model is not on the Hub, so the export has to be made with
``--base-model accounts/fireworks/models/muse-glimmer-30b
--tokenizer-model meta-models/Muse-Glimmer-30B --end-token '<|eot|>'``;
``--dry-run`` refuses an export tokenized for anything else.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from training.export_tinker_release import DEFAULT_OUT_DIR
from training.launch.make_launch_config import BATCH_SIZE, PEAK_LR
from training.tinker_sft_release import (
    DEFAULT_RANK,
    DEFAULT_SEED,
    Plan,
    http_get,
    plan_run,
    train,
)

ROOT = Path(__file__).resolve().parents[2]

FIREWORKS_BASE_MODEL = "accounts/fireworks/models/muse-glimmer-30b"
TOKENIZER_MODEL = "meta-models/Muse-Glimmer-30B"
BASE_URL = "https://api.fireworks.ai"

# Serverless training is the free route (through 2026-11-30). The managed path
# builds a dedicated rlorTrainerJobs trainer, and with no pinned shape the
# backend picks the accelerator itself -- it chose NVIDIA_B200_180GB, which
# bills against the account. Every run goes through the gateway instead.
SERVERLESS_ROUTE = "/training/v1/serverless"
SERVERLESS_BASE_URL = f"{BASE_URL}{SERVERLESS_ROUTE}"

RUN_PREFIX = "fireworks"
RUN_SCHEMA = "fireworks-sft-run-v1"
RUN_NAME = "fireworks-run.json"

# A sampler checkpoint name has to be a DNS label: 63 characters max. The SDK
# appends its own "-<8 hex>" suffix, so this budget covers what we control.
CHECKPOINT_PREFIX = "fw"
CHECKPOINT_ROOM = 54

HTTP_OK = 200

# The flags that decide whether this base can take a supervised LoRA run. They
# are top-level model fields; only the architecture lives in baseModelDetails.
MODEL_FLAGS = (
    "supervisedLoraTunable",
    "supportsServerlessTraining",
    "supportsLora",
    "trainingContextLength",
    "contextLength",
)


def load_api_key() -> str | None:
    """Return the Fireworks key from the environment, else from the repo .env."""
    key = os.environ.get("FIREWORKS_API_KEY")
    if key:
        return key
    env = ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("FIREWORKS_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def checkpoint_name(plan: Plan) -> str:
    """Return a DNS-label-safe sampler checkpoint name for this plan.

    The release id alone overflows the 63-character label limit once the prefix
    and the SDK's suffix are counted, so the id is truncated from the front: the
    release version at the end is what distinguishes one run from another.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", (plan.release_id or plan.dataset_dir.name).lower())
    room = CHECKPOINT_ROOM - len(CHECKPOINT_PREFIX) - 1
    if len(slug) > room:
        slug = slug[-room:].lstrip("-")
    return f"{CHECKPOINT_PREFIX}-{slug.strip('-')}"


def describe_model(api_key: str | None, *, base_url: str = BASE_URL) -> dict[str, Any]:
    """Report the base model's training flags without creating a trainer."""
    status, body = http_get(f"{base_url}/v1/{FIREWORKS_BASE_MODEL}", api_key)
    report: dict[str, Any] = {
        "base_model": FIREWORKS_BASE_MODEL,
        "tokenizer_model": TOKENIZER_MODEL,
        "status": status,
    }
    if status != HTTP_OK:
        report["reason"] = body[:400]
        return report

    payload = json.loads(body)
    details = payload.get("baseModelDetails") or payload.get("base_model_details") or {}
    report["state"] = payload.get("state")
    report["parameter_count"] = details.get("parameterCount")
    report["moe"] = details.get("moe")
    for flag in MODEL_FLAGS:
        report[flag] = payload.get(flag)
    return report


def assert_serverless(service: Any) -> None:
    """Fail closed if this service would provision a dedicated, billed trainer.

    A managed config is what routes ``create_training_client`` through
    ``rlorTrainerJobs``, where the backend auto-selects an accelerator.
    """
    if getattr(service, "_managed_config", None) is not None:
        raise RuntimeError(
            "refusing to train on the managed path: it provisions a dedicated "
            "trainer (NVIDIA_B200_180GB) and bills the account. Use the "
            f"serverless gateway at {SERVERLESS_BASE_URL}"
        )
    base_url = str(getattr(service, "_managed_base_url", "") or "").rstrip("/")
    if not base_url.endswith(SERVERLESS_ROUTE):
        raise RuntimeError(
            f"refusing to train against {base_url or '<no base url>'}: only the "
            f"serverless gateway at {SERVERLESS_BASE_URL} is free to run"
        )


def build_service(*, api_key: str) -> Any:
    """Return a serverless Firetitan service.

    With no managed config there is no dedicated trainer job: the gateway mints
    a training session (``ts-...``) and the run bills as serverless training.
    """
    from fireworks.training.sdk import FiretitanServiceClient

    service = FiretitanServiceClient(api_key=api_key, base_url=SERVERLESS_BASE_URL)
    assert_serverless(service)
    # Supervised runs on a shared trainer can lose their capacity to another
    # tenant mid-run; this is the seam the SDK's own SFT examples call so those
    # 409s are retried instead of killing the run.
    service._enable_serverless_supervised_409_retry()
    return service


def check_promote_target(model_id: str) -> list[str]:
    """Reject a bad promotion id before a run spends GPU time on it.

    The SDK wants the bare model name (``pixelated-glimmer-v22``), not a
    resource path; passing ``accounts/<acct>/models/<name>`` fails validation
    only after training has already finished.
    """
    from fireworks.training.sdk.fireworks_client import validate_output_model_id

    return validate_output_model_id(model_id)


def promote(
    service: Any,
    model_id: str,
    *,
    api_key: str,
    base_model: str = FIREWORKS_BASE_MODEL,
) -> dict[str, Any]:
    """Promote this session's newest promotable checkpoint to a model.

    A serverless run has no trainer job: its checkpoints live under
    ``accounts/<acct>/trainingSessions/<ts>/checkpoints`` and promote through the
    session-scoped endpoint. The session is gone once the service closes.
    """
    from fireworks.training.sdk.fireworks_client import FireworksClient

    session = service.training_session_name
    if not session:
        raise RuntimeError("no serverless training session is bound to this service")
    client = FireworksClient(api_key=api_key, base_url=BASE_URL)
    rows = [row for row in client.list_training_session_checkpoints(session) if row.get("promotable")]
    if not rows:
        raise RuntimeError(f"training session {session} has no promotable checkpoints")
    # Both list endpoints return oldest first, so the newest is the max.
    newest = max(rows, key=lambda row: row.get("createTime") or "")
    return client.promote_session_checkpoint(
        name=newest["name"],
        output_model_id=model_id,
        base_model=base_model,
    )


def _wandb_run(plan: Plan, name: str) -> Any:
    """Open a W&B run for this training job, or return None when unavailable."""
    if not os.environ.get("WANDB_API_KEY"):
        return None
    import wandb
    from training.launch.log_eval_tracking import ENTITY, PROJECT

    return wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type="fireworks-sft",
        name=name,
        mode="online",
        config=plan.to_dict(),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--base-model",
        default=FIREWORKS_BASE_MODEL,
        help="assert the export was tokenized for this Fireworks model",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=None, help="train on the first N datums only")
    parser.add_argument("--save-every", type=int, default=0, help="checkpoint every N steps")
    parser.add_argument(
        "--promote",
        metavar="MODEL_ID",
        default=None,
        help="promote the final checkpoint to this bare model name (no account prefix)",
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--check", action="store_true", help="report the base model's flags and exit")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--execute", action="store_true", help="actually train")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the trainer from the command line."""
    args = _parse_args(argv)
    api_key = load_api_key()

    if args.check:
        report = describe_model(api_key)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("status") == HTTP_OK else 1

    plan = plan_run(
        args.dataset_dir,
        base_model=args.base_model,
        batch_size=args.batch_size,
        rank=args.rank,
        seed=args.seed,
        limit=args.limit,
    )
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    if plan.recipe_degenerate:
        print(
            f"warning: {plan.steps} step(s) cannot carry the release schedule; holding {PEAK_LR:g} instead",
            file=sys.stderr,
        )
    if args.dry_run or not args.execute:
        if not args.dry_run:
            print("refusing to train without --execute", file=sys.stderr)
        return 0
    if not api_key:
        print("FIREWORKS_API_KEY is not set", file=sys.stderr)
        return 1
    if args.promote:
        problems = check_promote_target(args.promote)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 1

    name = f"{RUN_PREFIX}-{plan.release_id or plan.dataset_dir.name}"
    run = None if args.no_wandb else _wandb_run(plan, name)
    service = build_service(api_key=api_key)
    try:
        record = train(
            plan,
            service=service,
            run=run,
            save_every=args.save_every,
            run_prefix=RUN_PREFIX,
            schema=RUN_SCHEMA,
            run_name=checkpoint_name(plan),
        )
        if args.promote:
            record["promoted"] = promote(
                service,
                args.promote,
                api_key=api_key,
                base_model=plan.base_model,
            )
    finally:
        service.close()
        if run is not None:
            run.finish()

    out = plan.dataset_dir / RUN_NAME
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: record.get(k) for k in ("run_name", "sampling_path", "state_path")}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
