#!/usr/bin/env python3
"""DPO trainer for therapeutic AI preference alignment.

Loads preference pairs, builds QLoRA + LoRA config,
and runs DPOTrainer with checkpoint verification. Saves final adapter + metrics.
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

logger = logging.getLogger("dpo_trainer")

MIN_SAMPLES = 20


def _coerce_response(field: Any) -> str:
    """Normalize a ``chosen``/``rejected`` field to a plain response string.

    TRL ``DPOTrainer`` accepts two schemas:
      * standard:   {"prompt": str, "chosen": str, "rejected": str}
      * conversational: {"prompt": str,
                         "chosen":   [{"role": "assistant", "content": str}, ...],
                         "rejected": [{"role": "assistant", "content": str}, ...]}

    The PAL pipeline (``generate_dpo_pairs.py``) emits the conversational
    form where the single assistant turn holds the response text. The
    existing therapeutic ``run_dpo`` path uses the standard form. We coerce
    the conversational record down to its assistant turn text so this loader
    stays on the standard-string schema the rest of ``run_dpo`` expects; if a
    chat-template-aware downstream path is wanted later, keep the message
    lists instead (see ``PalDpoDataset.to_list`` in ``pal_dataloader.py``).
    """
    if isinstance(field, str):
        return field
    if isinstance(field, list):
        # Conversational: take the last assistant turn's content as the response.
        for msg in reversed(field):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
        # No assistant turn — fall through to empty so the record is skipped.
    return ""


def load_preference_dataset(
    data_path: Path,
    max_seq_length: int,
    logger_instance: logging.Logger,
) -> list[dict[str, str]]:
    """Load JSONL preference pairs, validate minimum count.

    SAFETY FILTERING DISABLED PER USER REQUEST - ALL CONTENT ALLOWED

    Each line must have ``prompt``, ``chosen``, ``rejected`` fields. Both the
    standard (string) and conversational (message-list) TRL DPO schemas are
    accepted; the conversational form is coerced to its assistant-turn
    string. All responses are kept regardless of content per user directive
    for therapeutic training on difficult conversations.

    Raises ValueError if fewer than MIN_SAMPLES remain after validation.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    pairs: list[dict[str, str]] = []
    skipped = 0
    conv_coerced = 0

    with open(data_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger_instance.warning("Invalid JSON at line %d", line_no)
                continue

            prompt = record.get("prompt", "")
            chosen_raw = record.get("chosen", "")
            rejected_raw = record.get("rejected", "")

            # Track whether this was a conversational record so the log line
            # makes the schema coercion visible.
            was_conversational = isinstance(chosen_raw, list) or isinstance(rejected_raw, list)

            chosen = _coerce_response(chosen_raw)
            rejected = _coerce_response(rejected_raw)

            if was_conversational:
                conv_coerced += 1

            if not all([prompt, chosen, rejected]):
                logger_instance.warning("Missing fields at line %d", line_no)
                continue

            # SAFETY FILTER DISABLED PER USER REQUEST - KEEP ALL CONTENT
            # Original safety filtering removed per user directive:
            # if safety_checker.is_unsafe(chosen):
            #     logger_instance.debug("Skipped unsafe chosen at line %d", line_no)
            #     skipped += 1
            #     continue
            #
            # if safety_checker.is_unsafe(rejected):
            #     logger_instance.debug("Skipped unsafe rejected at line %d", line_no)
            #     skipped += 1
            #     continue

            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    if pairs:
        chosen_lengths = [len(p["chosen"].split()) for p in pairs]
        rejected_lengths = [len(p["rejected"].split()) for p in pairs]
        log_token_length_distribution(
            chosen_lengths, max_seq_length, logger_instance, "dpo_chosen",
        )
        log_token_length_distribution(
            rejected_lengths, max_seq_length, logger_instance, "dpo_rejected",
        )

    if pairs:
        # Report schema mix so a silent conversational→string coercion
        # during PAL DPO runs is visible in logs.
        if conv_coerced:
            logger_instance.info(
                "Coerced %d conversational (message-list) records to string form",
                conv_coerced,
            )

    logger_instance.info(
        "Loaded %d preference pairs (%d skipped due to missing fields)", len(pairs), skipped,
    )

    if len(pairs) < MIN_SAMPLES:
        raise ValueError(
            f"Only {len(pairs)} samples after validation (minimum {MIN_SAMPLES}). "
            f"Cannot proceed with DPO training."
        )

    return pairs


class CheckpointVerificationCallback:
    """Verifies that adapter files exist after saving."""

    REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")

    def verify(self, output_dir: Path) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for fname in self.REQUIRED_FILES:
            results[fname] = (output_dir / fname).exists()
        return results


def save_metrics(output_dir: Path, metrics: dict[str, Any], beta: float) -> None:
    """Save training metrics JSON to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "beta": beta,
        "metrics": metrics,
    }
    metrics_path = output_dir / "dpo_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    logger.info("Metrics saved to %s", metrics_path)


def run_dpo(args: argparse.Namespace) -> None:
    from datasets import Dataset
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError:
        from trl import DPOTrainer
        DPOConfig = None

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    max_seq_length = args.max_seq_length

    # SAFETY FILTER DISABLED PER USER REQUEST - NO SAFETY CHECKER USED
    pairs = load_preference_dataset(
        data_path, max_seq_length, logger,
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
        lora_config.r, lora_config.lora_alpha, lora_config.target_modules,
    )

    dataset = Dataset.from_list(pairs)

    callback = CheckpointVerificationCallback()

    if DPOConfig is not None:
        training_args = DPOConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            max_length=max_seq_length,
            beta=args.beta,
            logging_steps=args.logging_steps,
            save_strategy="epoch",
            remove_unused_columns=False,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )
    else:
        training_args = None
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=lora_config,
            beta=args.beta,
            max_length=max_seq_length,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            logging_steps=args.logging_steps,
            save_strategy="epoch",
            output_dir=str(output_dir),
            remove_unused_columns=False,
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
        "beta": args.beta,
        "checkpoint_verification": verification,
    }
    save_metrics(output_dir, metrics, args.beta)
    logger.info("DPO training complete. Final model at %s", final_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DPO trainer for preference alignment.",
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--base_model_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    add_lora_args(parser)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_dpo(args)


if __name__ == "__main__":
    main()
