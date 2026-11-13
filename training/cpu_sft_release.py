"""LoRA SFT on a CPU-only host, consuming the same tokenized datums as the hosted launchers.

Every hosted trainer we can reach is unavailable: the W&B serverless control plane answers
502 on both routes, Tinker's org is billing-blocked, the Fireworks accounts are suspended, and
the free tiers we hold keys for (Kaggle, Lightning, Novita, Nscale) reject their tokens or have
no balance. What is left is the box itself, so this module trains the exported corpus in fp32 on
CPU and writes the same run record the other launchers do.

The datum contract is shared: ``{"input_ids", "target_ids", "weights"}`` from
``training/export_tinker_release.py``, with ``target_ids`` already shifted by one and ``weights``
aligned to them. Nothing here re-tokenizes, so the mask you inspect in the export is the mask
that gets trained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_DIR = ROOT / "ai/training/output/cpu-v22-qwen06"
DATASET_FILE = "tinker-sft-train.jsonl"
RECORD_FILE = "tinker-dataset.json"
RUN_NAME = "cpu-run.json"
RUN_SCHEMA = "cpu-sft-run-v1"

DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 32
DEFAULT_LR = 1e-4
DEFAULT_MIN_LR = 1e-6
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 1
DEFAULT_SEED = 20261110
WARMUP_FRACTION = 0.05
HOLD_FRACTION = 0.6
IGNORE_INDEX = -100
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class Recipe:
    """The training knobs, grouped so a run is described by one object."""

    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    rank: int = DEFAULT_LORA_R
    alpha: int = DEFAULT_LORA_ALPHA
    lr: float = DEFAULT_LR
    min_lr: float = DEFAULT_MIN_LR
    seed: int = DEFAULT_SEED
    limit: int | None = None


@dataclass(frozen=True)
class Plan:
    """What a run will do, decided before a single gradient is computed."""

    dataset_dir: Path
    base_model: str
    datums: int
    trained_tokens: int
    batch_size: int
    steps: int
    epochs: int
    rank: int
    alpha: int
    lr: float
    min_lr: float
    seed: int
    release_id: str | None


def load_datums(dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the export and refuse it if the file no longer matches its recorded digest."""
    record = json.loads((dataset_dir / RECORD_FILE).read_text())
    payload = (dataset_dir / DATASET_FILE).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != record["sha256"]:
        raise ValueError(f"{DATASET_FILE} sha256 {digest} does not match the recorded {record['sha256']}")
    rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    if len(rows) != record["datums"]:
        raise ValueError(f"{DATASET_FILE} holds {len(rows)} datums, the record claims {record['datums']}")
    return rows, record


def trained_tokens(rows: list[dict[str, Any]]) -> int:
    return sum(sum(row["weights"]) for row in rows)


def make_batches(rows: list[dict[str, Any]], batch_size: int, rng: random.Random) -> list[list[dict[str, Any]]]:
    """Shuffle, then sort within a window so a batch pads to its own longest datum."""
    order = list(rows)
    rng.shuffle(order)
    window = batch_size * 32
    batches: list[list[dict[str, Any]]] = []
    for start in range(0, len(order), window):
        part = sorted(order[start : start + window], key=lambda row: len(row["input_ids"]))
        batches.extend(part[at : at + batch_size] for at in range(0, len(part), batch_size))
    return batches


def lr_at(progress: float, *, peak: float, floor: float, start: float) -> float:
    """The release trapezoid: linear warmup, hold, cosine decay."""
    if progress < WARMUP_FRACTION:
        return start + (peak - start) * (progress / WARMUP_FRACTION)
    if progress < HOLD_FRACTION:
        return peak
    span = 1.0 - HOLD_FRACTION
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * (progress - HOLD_FRACTION) / span))


def plan_run(
    dataset_dir: Path,
    *,
    base_model: str | None = None,
    recipe: Recipe | None = None,
) -> tuple[Plan, list[dict[str, Any]], dict[str, Any]]:
    recipe = recipe or Recipe()
    rows, record = load_datums(dataset_dir)
    if recipe.limit is not None:
        rows = rows[: recipe.limit]
    if not rows:
        raise ValueError(f"{dataset_dir} selected no datums")
    model = base_model or record["base_model"]
    if model != record["base_model"]:
        raise ValueError(f"{model} was not the tokenizer behind this export ({record['base_model']})")
    steps = math.ceil(len(rows) / recipe.batch_size) * recipe.epochs
    plan = Plan(
        dataset_dir=dataset_dir,
        base_model=model,
        datums=len(rows),
        trained_tokens=trained_tokens(rows),
        batch_size=recipe.batch_size,
        steps=steps,
        epochs=recipe.epochs,
        rank=recipe.rank,
        alpha=recipe.alpha,
        lr=recipe.lr,
        min_lr=recipe.min_lr,
        seed=recipe.seed,
        release_id=record.get("source_release_id"),
    )
    return plan, rows, record


def _collate(batch: list[dict[str, Any]], pad_id: int, torch: Any) -> tuple[Any, Any, Any]:
    width = max(len(row["input_ids"]) for row in batch)
    inputs, targets, weights = [], [], []
    for row in batch:
        pad = width - len(row["input_ids"])
        inputs.append(row["input_ids"] + [pad_id] * pad)
        targets.append(row["target_ids"] + [IGNORE_INDEX] * pad)
        weights.append([float(value) for value in row["weights"]] + [0.0] * pad)
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(weights, dtype=torch.float32),
    )


def masked_cross_entropy(logits: Any, targets: Any, weights: Any, torch: Any) -> Any:
    """Per-token mean over the trained span only, matching the tinker/fireworks loss."""
    losses = torch.nn.functional.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        targets.view(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view(targets.shape)
    return (losses * weights).sum() / weights.sum().clamp(min=1.0)


def build_model(base_model: str, *, rank: int, alpha: float, torch: Any) -> Any:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    model.config.use_cache = False
    return get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=int(alpha),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(TARGET_MODULES),
        ),
    )


def _wandb_run(plan: Plan, name: str) -> Any:
    import wandb
    from training.launch.log_eval_tracking import ENTITY, PROJECT

    return wandb.init(
        project=PROJECT,
        entity=ENTITY,
        name=name,
        job_type="cpu-sft",
        config={
            "base_model": plan.base_model,
            "datums": plan.datums,
            "trained_tokens": plan.trained_tokens,
            "batch_size": plan.batch_size,
            "steps": plan.steps,
            "epochs": plan.epochs,
            "lora_r": plan.rank,
            "lora_alpha": plan.alpha,
            "lr": plan.lr,
            "min_lr": plan.min_lr,
            "release_id": plan.release_id,
        },
    )


def train(
    plan: Plan,
    rows: list[dict[str, Any]],
    *,
    save_every: int = 250,
    run: Any = None,
    log: Any = print,
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    torch.manual_seed(plan.seed)
    torch.set_num_threads(os.cpu_count() or 1)
    tokenizer = AutoTokenizer.from_pretrained(plan.base_model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    model = build_model(plan.base_model, rank=plan.rank, alpha=plan.alpha, torch=torch)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=plan.lr
    )

    rng = random.Random(plan.seed)
    start_lr = plan.lr / 5.0
    started = time.time()
    step = 0
    losses: list[float] = []
    out_dir = plan.dataset_dir

    for epoch in range(plan.epochs):
        for batch in make_batches(rows, plan.batch_size, rng):
            progress = step / max(plan.steps - 1, 1)
            lr = lr_at(progress, peak=plan.lr, floor=plan.min_lr, start=start_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            inputs, targets, weights = _collate(batch, pad_id, torch)
            logits = model(input_ids=inputs).logits
            loss = masked_cross_entropy(logits, targets, weights, torch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            value = float(loss.detach())
            losses.append(value)
            if run is not None:
                run.log({"train/loss": value, "train/lr": lr, "train/epoch": epoch, "train/step": step})
            if step == 1 or step % 10 == 0 or step == plan.steps:
                elapsed = time.time() - started
                rate = elapsed / step
                log(
                    f"step {step}/{plan.steps} rows {len(batch)} lr {lr:.2e} loss {value:.2f} "
                    f"elapsed {elapsed / 60:.1f}m eta {rate * (plan.steps - step) / 60:.1f}m"
                )
            if save_every and step % save_every == 0 and step != plan.steps:
                checkpoint = out_dir / f"adapter-step-{step}"
                model.save_pretrained(checkpoint)
                log(f"saved {checkpoint.name}")

    adapter_dir = out_dir / "adapter-final"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    record = {
        "schema": RUN_SCHEMA,
        "run_name": f"cpu-{plan.release_id or plan.dataset_dir.name}",
        "base_model": plan.base_model,
        "release_id": plan.release_id,
        "datums": plan.datums,
        "trained_tokens": plan.trained_tokens,
        "batch_size": plan.batch_size,
        "steps": plan.steps,
        "epochs": plan.epochs,
        "lora_r": plan.rank,
        "lora_alpha": plan.alpha,
        "lr": plan.lr,
        "min_lr": plan.min_lr,
        "seed": plan.seed,
        "adapter": str(adapter_dir),
        "elapsed_seconds": round(time.time() - started, 1),
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "mean_loss_last_10": sum(losses[-10:]) / len(losses[-10:]) if losses else None,
    }
    (plan.dataset_dir / RUN_NAME).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--base-model", default=None, help="must match the tokenizer behind the export")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--rank", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--min-lr", type=float, default=DEFAULT_MIN_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=None, help="train on the first N datums only")
    parser.add_argument("--save-every", type=int, default=250, help="checkpoint every N steps (0 disables)")
    parser.add_argument("--no-wandb", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    mode.add_argument("--execute", action="store_true", help="train")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan, rows, _record = plan_run(
        args.dataset_dir,
        base_model=args.base_model,
        recipe=Recipe(
            batch_size=args.batch_size,
            epochs=args.epochs,
            rank=args.rank,
            alpha=args.alpha,
            lr=args.lr,
            min_lr=args.min_lr,
            seed=args.seed,
            limit=args.limit,
        ),
    )
    print(
        f"base_model {plan.base_model}\n"
        f"datums {plan.datums} trained_tokens {plan.trained_tokens}\n"
        f"steps {plan.steps} batch_size {plan.batch_size} epochs {plan.epochs}\n"
        f"lora r {plan.rank} alpha {plan.alpha} lr {plan.lr:g} -> {plan.min_lr:g}\n"
        f"release_id {plan.release_id} seed {plan.seed}"
    )
    if args.dry_run:
        return 0

    run = None if args.no_wandb else _wandb_run(plan, f"cpu-{plan.release_id or plan.dataset_dir.name}")
    try:
        record = train(plan, rows, save_every=args.save_every, run=run)
    finally:
        if run is not None:
            run.finish()
    summary = {key: record[key] for key in ("run_name", "steps", "elapsed_seconds", "final_loss", "adapter")}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
