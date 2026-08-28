#!/usr/bin/env python3
"""ORPO (Odds Ratio Preference Optimization) trainer for therapeutic AI.

ORPO combines SFT and preference alignment in a single pass — no reference
model is needed, reducing memory by ~25% vs DPO.  The dataset format is
identical to DPO (``prompt`` / ``chosen`` / ``rejected``), so the proven
``load_preference_dataset`` loader from ``dpo_trainer`` is reused.

Blueprint reference: §2 (ORPO preferred over DPO for new pipelines) and
Appendix C (``use_orpo: true``, ``beta: 0.1``).

Enhanced beyond the initial implementation:
  - DeepSpeed ZeRO-3 support (``--deepspeed`` flag, blueprint Appendix D)
  - WandB experiment tracking (``--wandb_project``, ``--wandb_run_name``)
  - Warmup ratio + cosine schedule (Appendix C SFT hyperparams)
  - Gradient checkpointing toggle (memory vs compute trade-off)
  - Early stopping patience (Appendix E Step 3: "early stop if val perplexity
    increases for 2 consecutive epochs")
  - Save total limit (checkpoint disk management)
  - Evaluation strategy + eval steps (Appendix C: val_set_size 0.05)
  - Flash attention toggle (Appendix C: flash_attention: true)
  - DoRA / VeRA adapter variant support (Appendix C: LoRA Variant Configs)

Usage::

    python -m ai.training.orpo_trainer \
        --data_path data/preference_pairs.jsonl \
        --base_model_checkpoint Qwen/Qwen2.5-32B \
        --output_dir models/orpo-out \
        --beta 0.1 \
        --deepspeed ai/training/configs/ds_config_zero3.json \
        --wandb_project pixelated-empathy-orpo

Usage (Axolotl config equivalent)::

    axolotl train ai/training/configs/orpo_axolotl.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from .shared_config import (
        add_lora_args,
        build_lora_config,
        log_token_length_distribution,
        shared_qlora_config,
    )
except ModuleNotFoundError:
    try:
        from ai.training.shared_config import (
            add_lora_args,
            build_lora_config,
            log_token_length_distribution,
            shared_qlora_config,
        )
    except ModuleNotFoundError:
        from shared_config import (
            add_lora_args,
            build_lora_config,
            log_token_length_distribution,
            shared_qlora_config,
        )

try:
    from .dpo_trainer import (
        CheckpointVerificationCallback,
        load_preference_dataset,
    )
except ModuleNotFoundError:
    try:
        from ai.training.dpo_trainer import (
            CheckpointVerificationCallback,
            load_preference_dataset,
        )
    except ModuleNotFoundError:
        from dpo_trainer import (
            CheckpointVerificationCallback,
            load_preference_dataset,
        )

logger = logging.getLogger("orpo_trainer")

DEFAULT_BETA = 0.1
DEFAULT_LR = 5e-6
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_LOGGING_STEPS = 10
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_SAVE_TOTAL_LIMIT = 3
DEFAULT_EVAL_STEPS = 50


def save_metrics(
    output_dir: Path,
    metrics: dict[str, Any],
    beta: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save training metrics JSON to output directory.

    Args:
        output_dir: Where to write ``orpo_metrics.json``.
        metrics: Core training metrics (train_loss, train_runtime, etc.).
        beta: ORPO odds-ratio penalty weight.
        extra: Additional top-level fields merged into the report (e.g.
            ``deepspeed_config``, ``wandb_run_id``, ``adapter_variant``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "method": "orpo",
        "beta": beta,
        "metrics": metrics,
    }
    if extra:
        report.update(extra)
    metrics_path = output_dir / "orpo_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    logger.info("Metrics saved to %s", metrics_path)


def _build_training_args(args: argparse.Namespace):
    """Build ``ORPOConfig`` (trl >= 0.14) from parsed CLI args.

    Handles version differences between trl releases gracefully:
    - ``ORPOConfig`` available (trl >= 0.14): use it.
    - Fallback: construct via ``TrainingArguments`` with ORPO-specific kwargs.
    """
    try:
        from trl import ORPOConfig
    except ImportError:
        try:
            from trl.experimental.orpo import ORPOConfig
        except ImportError:
            from transformers import TrainingArguments as ORPOConfig

    kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "max_length": args.max_seq_length,
        "beta": args.beta,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "remove_unused_columns": False,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": "wandb" if args.wandb_project else "none",
    }

    if args.eval_strategy != "no":
        kwargs["eval_strategy"] = args.eval_strategy
        kwargs["eval_steps"] = args.eval_steps

    if args.early_stopping_patience > 0:
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = "eval_loss"
        kwargs["greater_is_better"] = False

    if args.deepspeed:
        kwargs["deepspeed"] = str(args.deepspeed)

    if args.flash_attention:
        kwargs["flash_attention"] = True

    try:
        return ORPOConfig(**kwargs)
    except TypeError:
        import inspect

        sig = inspect.signature(ORPOConfig.__init__)
        supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return ORPOConfig(**supported)


def _setup_wandb(args: argparse.Namespace) -> dict[str, Any] | None:
    """Initialize WandB if configured, return metadata for metrics report."""
    if not args.wandb_project:
        return None

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — skipping experiment tracking")
        return None

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or f"orpo-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
        config={
            "method": "orpo",
            "beta": args.beta,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "deepspeed": str(args.deepspeed) if args.deepspeed else None,
            "gradient_checkpointing": args.gradient_checkpointing,
            "flash_attention": args.flash_attention,
        },
    )

    return {
        "wandb_run_id": run.id,
        "wandb_project": args.wandb_project,
    }


def run_orpo(args: argparse.Namespace) -> None:
    """Run ORPO training.

    ORPO merges the SFT loss and the odds-ratio preference loss into a single
    objective, so no separate SFT pass is required.  The ``beta`` parameter
    controls the strength of the preference penalty (default 0.1 per
    Appendix C).
    """
    from datasets import Dataset
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from trl import ORPOTrainer
    except ImportError:
        try:
            from trl.experimental.orpo import ORPOTrainer
        except ImportError:
            logger.error("ORPOTrainer requires trl >= 0.14. Install with: pip install trl>=0.14")
            return

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    max_seq_length = args.max_seq_length

    pairs = load_preference_dataset(data_path, max_seq_length, logger)

    if pairs:
        chosen_lengths = [len(p["chosen"].split()) for p in pairs]
        rejected_lengths = [len(p["rejected"].split()) for p in pairs]
        log_token_length_distribution(
            chosen_lengths,
            max_seq_length,
            logger,
            "orpo_chosen",
        )
        log_token_length_distribution(
            rejected_lengths,
            max_seq_length,
            logger,
            "orpo_rejected",
        )

    logger.info("Loading model from %s", args.base_model_checkpoint)
    bnb_config = shared_qlora_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_checkpoint,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = build_lora_config(args)
    logger.info(
        "LoRA config: r=%d, alpha=%d, targets=%s",
        lora_config.r,
        lora_config.lora_alpha,
        lora_config.target_modules,
    )

    adapter_info = {"adapter_variant": "lora"}

    if getattr(args, "use_dora", False):
        try:
            lora_config.use_dora = True
            adapter_info["adapter_variant"] = "dora"
            logger.info("Using DoRA (Weight-Decomposed LoRA)")
        except Exception:
            logger.warning("use_dora requested but PEFT version does not support it")

    if getattr(args, "use_vera", False):
        try:
            lora_config.use_vera = True
            adapter_info["adapter_variant"] = "vera"
            logger.info("Using VeRA (Vector-based Random Adaptation)")
        except Exception:
            logger.warning("use_vera requested but PEFT version does not support it")

    dataset = Dataset.from_list(pairs)

    callback = CheckpointVerificationCallback()

    wandb_meta = _setup_wandb(args)

    training_args = _build_training_args(args)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "processing_class": tokenizer,
        "peft_config": lora_config,
    }

    callbacks = []
    if args.early_stopping_patience > 0:
        try:
            from transformers import EarlyStoppingCallback

            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
            )
            logger.info("Early stopping enabled (patience=%d)", args.early_stopping_patience)
        except ImportError:
            logger.warning("EarlyStoppingCallback not available — skipping")

    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    trainer = ORPOTrainer(**trainer_kwargs)

    logger.info(
        "ORPO config: beta=%.4f, lr=%.2e, epochs=%d, batch=%d, seq_len=%d, deepspeed=%s, grad_ckpt=%s, flash_attn=%s",
        args.beta,
        args.learning_rate,
        args.epochs,
        args.batch_size,
        max_seq_length,
        str(args.deepspeed) if args.deepspeed else "none",
        args.gradient_checkpointing,
        args.flash_attention,
    )

    train_result = trainer.train()

    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    verification = callback.verify(final_dir)
    logger.info("Checkpoint verification: %s", verification)

    metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        "beta": args.beta,
        "checkpoint_verification": verification,
        "num_train_samples": len(pairs),
    }

    extra = {
        "adapter_variant": adapter_info["adapter_variant"],
        "deepspeed_config": str(args.deepspeed) if args.deepspeed else None,
        "gradient_checkpointing": args.gradient_checkpointing,
        "flash_attention": args.flash_attention,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
    }

    if wandb_meta:
        extra.update(wandb_meta)

    save_metrics(output_dir, metrics, args.beta, extra=extra)

    if wandb_meta:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass

    logger.info("ORPO training complete. Final model at %s", final_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ORPO trainer — single-pass SFT + preference alignment.",
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--base_model_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help=f"Odds-ratio penalty weight (Appendix C default: {DEFAULT_BETA}).",
    )
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--logging_steps", type=int, default=DEFAULT_LOGGING_STEPS)
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
        help="Fraction of steps for linear warmup (Appendix C: 0.1).",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "constant", "constant_with_warmup"],
        help="LR scheduler type (Appendix C: cosine).",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing (~30% compute for memory).",
    )
    parser.add_argument(
        "--flash_attention",
        action="store_true",
        default=False,
        help="Enable Flash Attention 2 (Appendix C: flash_attention: true).",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed ZeRO-3 config JSON (Appendix D).",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=DEFAULT_SAVE_TOTAL_LIMIT,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="no",
        choices=["no", "steps", "epoch"],
        help="Evaluation strategy (Appendix C: eval_steps 50).",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=DEFAULT_EVAL_STEPS,
        help="Evaluation steps when eval_strategy='steps'.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Stop if eval_loss doesn't improve for N evaluations (Appendix E: 2).",
    )
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    add_lora_args(parser)
    parser.add_argument(
        "--use_dora",
        action="store_true",
        default=False,
        help="Use DoRA (Weight-Decomposed LoRA) — better perf for same rank.",
    )
    parser.add_argument(
        "--use_vera",
        action="store_true",
        default=False,
        help="Use VeRA (Vector-based Random Adaptation) — fewer params.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_orpo(args)


if __name__ == "__main__":
    main()