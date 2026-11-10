#!/usr/bin/env python3
"""Train the exported release selection on Tinker with the release LR recipe.

The Fireworks/W&B runs hand the provider a row list and let it pick the recipe.
Tinker does not: it exposes ``forward_backward`` and ``optim_step``, so the
recipe has to be replayed here. That is deliberate — ``learning_rate_schedule``
from ``make_launch_config`` is the same trapezoid every release run has used
(warmup 5% 1e-5 -> 5e-5, hold to 60%, cosine to 5e-6), so a Tinker adapter is
comparable to the W&B adapters instead of being a different experiment.

The data comes from ``export_tinker_release.py``: token ids plus a per-token
loss weight, so the model is only scored on assistant turns. ``plan_run``
re-verifies that export (sha256, datum count, trained-token total) before a
single API call, because a silent mismatch there would train on the wrong
tokens and only show up in the eval scoreboard.

Tinker bills per trained token and the org has to have credit on file; a
blocked org answers every call with HTTP 402. ``--check`` probes that and the
server capabilities without training anything.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.export_tinker_release import DEFAULT_BASE_MODEL, DEFAULT_OUT_DIR
from training.launch.make_launch_config import (
    BATCH_SIZE,
    PEAK_LR,
    RECIPE_SOURCE,
    learning_rate_schedule,
    total_batch,
)

ROOT = Path(__file__).resolve().parents[2]
DATUM_NAME = "tinker-sft-train.jsonl"
RECORD_NAME = "tinker-dataset.json"
RUN_NAME = "tinker-run.json"
RUN_SCHEMA = "tinker-sft-run-v1"

# The rank the account-proven Fireworks recipe used (lora_r 32 / alpha 32).
DEFAULT_RANK = 32
DEFAULT_SEED = 20261110

# The SDK defaults to the same host; it is named here so ``--check`` can probe
# it without constructing a client.
DEFAULT_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod"

# ``learning_rate_schedule`` refuses a degenerate schedule, which is the right
# call for a release run and the wrong one for a two-step smoke test.
LOSS_KEYS = ("loss:mean", "loss", "cross_entropy:mean")

# Tinker answers every call with this when the organisation has no billing set up.
HTTP_PAYMENT_REQUIRED = 402

# Cloudflare sits in front of the API and rejects urllib's default user agent.
USER_AGENT = "pixelated-tinker-sft/1.0"


class PlanError(RuntimeError):
    """The export on disk does not match its own record."""


@dataclass(frozen=True)
class Plan:
    """What a Tinker run of this export would do, computed before any API call."""

    dataset_dir: Path
    base_model: str
    datums: int
    trained_tokens: int
    batch_size: int
    steps: int
    schedule: list[float]
    rank: int
    seed: int
    release_id: str | None = None
    recipe_degenerate: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as it is written into the run record."""
        return {
            "dataset_dir": str(self.dataset_dir),
            "base_model": self.base_model,
            "datums": self.datums,
            "trained_tokens": self.trained_tokens,
            "batch_size": self.batch_size,
            "steps": self.steps,
            "peak_lr": max(self.schedule),
            "first_lr": self.schedule[0],
            "last_lr": self.schedule[-1],
            "rank": self.rank,
            "seed": self.seed,
            "release_id": self.release_id,
            "recipe_source": RECIPE_SOURCE,
            "recipe_degenerate": self.recipe_degenerate,
        }


def _sha256(path: Path) -> str:
    """Return the SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_datums(dataset_dir: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Return the exported datums, optionally truncated to ``limit``."""
    path = dataset_dir / DATUM_NAME
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def plan_run(
    dataset_dir: Path = DEFAULT_OUT_DIR,
    *,
    base_model: str | None = None,
    batch_size: int = BATCH_SIZE,
    rank: int = DEFAULT_RANK,
    seed: int = DEFAULT_SEED,
    limit: int | None = None,
) -> Plan:
    """Return the plan for this export, verifying the export against its record.

    ``base_model=None`` trusts the model named in the export record; passing a
    model explicitly only asserts it matches.
    """
    record_path = dataset_dir / RECORD_NAME
    if not record_path.exists():
        raise PlanError(f"{record_path} is missing; run export_tinker_release.py first")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    datum_path = dataset_dir / DATUM_NAME
    if not datum_path.exists():
        raise PlanError(f"{datum_path} is missing; run export_tinker_release.py first")

    actual_sha = _sha256(datum_path)
    if actual_sha != record.get("sha256"):
        raise PlanError(f"{datum_path} sha256 is {actual_sha}, the record says {record.get('sha256')}")

    rows = load_datums(dataset_dir, limit)
    if not rows:
        raise PlanError(f"{datum_path} has no datums")
    if limit is None and len(rows) != record.get("datums"):
        raise PlanError(f"{datum_path} has {len(rows)} datums, the record says {record.get('datums')}")

    trained = sum(sum(row["weights"]) for row in rows)
    if limit is None and trained != record.get("counts", {}).get("trained_tokens"):
        raise PlanError(
            f"{datum_path} trains {trained} tokens, the record says {record.get('counts', {}).get('trained_tokens')}"
        )

    steps = total_batch(len(rows), batch_size)
    try:
        schedule = learning_rate_schedule(steps)
        degenerate = False
    except ValueError:
        schedule = [PEAK_LR] * steps
        degenerate = True
    record_model = record.get("base_model")
    if base_model is not None and record_model and record_model != base_model:
        # The datum token ids only mean anything to the tokenizer that produced
        # them, so training them on a different base silently destroys the run.
        raise PlanError(f"{datum_path} was tokenized for {record_model}, not {base_model}")
    return Plan(
        dataset_dir=dataset_dir,
        base_model=record_model or base_model or DEFAULT_BASE_MODEL,
        datums=len(rows),
        trained_tokens=trained,
        batch_size=batch_size,
        steps=steps,
        schedule=schedule,
        rank=rank,
        seed=seed,
        release_id=record.get("source_release_id"),
        recipe_degenerate=degenerate,
    )


def datum_from_row(tinker: Any, row: dict[str, Any]) -> Any:
    """Return one Tinker ``Datum`` for an exported row."""
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(row["input_ids"]),
        loss_fn_inputs={
            "target_tokens": row["target_ids"],
            "weights": row["weights"],
        },
    )


def _loss(fb_result: Any) -> float | None:
    """Return the batch loss from a forward/backward result, if it reported one."""
    metrics = getattr(fb_result, "metrics", None) or {}
    for key in LOSS_KEYS:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _get(url: str, api_key: str | None, timeout: float = 30.0) -> tuple[int, str]:
    """Return ``(status, body)`` for a GET, keeping HTTP errors as data."""
    if urllib.parse.urlparse(url).scheme not in {"http", "https"}:
        raise ValueError(f"refusing to open a non-HTTP URL: {url}")
    request = urllib.request.Request(url, method="GET")
    # Cloudflare fronts the API and rejects urllib's default user agent.
    request.add_header("User-Agent", USER_AGENT)
    request.add_header("Accept", "application/json")
    if api_key:
        request.add_header("X-Api-Key", api_key)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


def check_api(
    base_url: str | None = None,
    api_key: str | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
) -> dict[str, Any]:
    """Probe health, capabilities and billing without training anything."""
    base_url = (base_url or os.environ.get("TINKER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    api_key = api_key or os.environ.get("TINKER_API_KEY")
    report: dict[str, Any] = {
        "base_url": base_url,
        "api_key_present": bool(api_key),
        "base_model": base_model,
    }

    status, _ = _get(f"{base_url}/api/v1/healthz", api_key)
    report["health_status"] = status

    status, body = _get(f"{base_url}/api/v1/get_server_capabilities", api_key)
    report["capabilities_status"] = status
    if status == HTTP_PAYMENT_REQUIRED:
        report["blocked"] = True
        report["reason"] = body[:400]
        return report
    report["blocked"] = False
    try:
        models = json.loads(body).get("supported_models", [])
    except json.JSONDecodeError:
        report["reason"] = body[:400]
        return report
    names = [entry.get("model_name") for entry in models if isinstance(entry, dict)]
    report["model_supported"] = base_model in names
    report["model_count"] = len(names)
    return report


def train(
    plan: Plan,
    *,
    service: Any | None = None,
    tinker: Any | None = None,
    run: Any | None = None,
    log: Any = print,
    save_every: int = 0,
) -> dict[str, Any]:
    """Run the plan against Tinker and return the run record."""
    tinker = tinker or importlib.import_module("tinker")
    if service is None:
        service = tinker.ServiceClient(
            base_url=os.environ.get("TINKER_BASE_URL") or DEFAULT_BASE_URL,
            api_key=os.environ.get("TINKER_API_KEY"),
        )

    rows = load_datums(plan.dataset_dir, plan.datums)
    run_name = f"tinker-{plan.release_id or plan.dataset_dir.name}"
    client = service.create_lora_training_client(
        base_model=plan.base_model,
        rank=plan.rank,
        seed=plan.seed,
    )

    losses: list[float] = []
    started = time.time()
    for step in range(plan.steps):
        start = step * plan.batch_size
        batch = rows[start : start + plan.batch_size]
        data = [datum_from_row(tinker, row) for row in batch]
        learning_rate = plan.schedule[step]
        result = client.forward_backward(data, loss_fn="cross_entropy").result()
        client.optim_step(tinker.AdamParams(learning_rate=learning_rate)).result()
        loss = _loss(result)
        if loss is not None:
            losses.append(loss)
        log(
            f"step {step + 1}/{plan.steps} rows {len(batch)} "
            f"lr {learning_rate:.3g} loss {loss if loss is not None else 'n/a'}"
        )
        if run is not None:
            scalars: dict[str, float] = {"train/step": float(step + 1), "train/lr": learning_rate}
            if loss is not None:
                scalars["train/loss"] = loss
            run.log(scalars)
        if save_every and (step + 1) % save_every == 0 and step + 1 < plan.steps:
            client.save_state(name=f"{run_name}-step{step + 1}").result()

    sampling = client.save_weights_for_sampler(name=run_name).result()
    state = client.save_state(name=f"{run_name}-final").result()
    record = {
        "schema": RUN_SCHEMA,
        "run_name": run_name,
        "plan": plan.to_dict(),
        "losses": losses,
        "elapsed_seconds": round(time.time() - started, 1),
        "sampling_path": getattr(sampling, "path", None),
        "state_path": getattr(state, "path", None),
    }
    if run is not None:
        run.config.update({"sampling_path": record["sampling_path"]})
        run.finish()
    return record


def _wandb_run(plan: Plan, name: str) -> Any:
    """Open a W&B run for this training job, or return None when unavailable."""
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        return None
    import wandb
    from training.launch.log_eval_tracking import ENTITY, PROJECT

    return wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type="tinker-sft",
        name=name,
        mode="online",
        config=plan.to_dict(),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--base-model",
        default=None,
        help=f"assert the export was tokenized for this model (default: the export record, {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=None, help="train on the first N datums only")
    parser.add_argument("--save-every", type=int, default=0, help="checkpoint every N steps")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--check", action="store_true", help="probe the Tinker API and exit")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--execute", action="store_true", help="actually train")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the trainer from the command line."""
    args = _parse_args(argv)

    if args.check:
        report = check_api(base_model=args.base_model or DEFAULT_BASE_MODEL)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1 if report.get("blocked") or not report.get("model_supported", False) else 0

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

    name = f"tinker-{plan.release_id or plan.dataset_dir.name}"
    run = None if args.no_wandb else _wandb_run(plan, name)
    try:
        record = train(plan, run=run, save_every=args.save_every)
    finally:
        if run is not None:
            run.finish()

    out = plan.dataset_dir / RUN_NAME
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("run_name", "sampling_path", "state_path")}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
