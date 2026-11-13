"""Train Qwen3-8B on the v22 export in full precision on an OVH L40S.

The Kaggle kernel is stuck at 4-bit NF4 because a T4 has 16 GB, and the loss
plateaued at 3.3 while the 0.6B reached 2.25. This run removes the only
variable that cannot be changed on a T4: the quantized base. Everything else
(export, mask, rank, schedule, seed, effective batch, LR) matches the Kaggle v6
run, so the two curves differ by precision alone.

The data and the shared helpers ride along in the object storage container that
this job mounts at /workspace.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import train as shared
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from train import TrapezoidTrainer, build_collator, verify_export
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

DATA_DIR = Path("/workspace/data")
OUT_DIR = Path("/workspace/out")
SCHEMA = "ovh-sft-run-v1"
RUN_NAME = "ovh-wandb-full-corpus-current-master-20261110-v22-fp16"

EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUM = 2
LR = 1e-4
MIN_LR = 1e-6
RANK = 32
ALPHA = 32
SEED = 20261110
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_model() -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        shared.BASE_MODEL_ID,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=RANK,
            lora_alpha=ALPHA,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ),
    )
    model.print_trainable_parameters()
    return model


def main() -> None:
    started = time.time()
    print(
        f"torch {torch.__version__}, cuda {torch.cuda.is_available()}, "
        f"devices {torch.cuda.device_count()}, "
        f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
        flush=True,
    )
    if LR != shared.LR or MIN_LR != shared.MIN_LR:
        raise ValueError("the shared trapezoid schedule carries its own LR, so they must agree")

    verified = verify_export(DATA_DIR)
    record = verified["record"]
    rows = verified["rows"]
    trained = sum(sum(row["weights"]) for row in rows)
    print(
        f"export verified: {len(rows)} datums, {trained} trained tokens, "
        f"sha256 {verified['sha256'][:16]}, release {record.get('source_release_id')}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(shared.BASE_MODEL_ID)
    dataset = Dataset.from_list(rows)
    model = build_model()

    steps_per_epoch = math.ceil(len(rows) / (BATCH_SIZE * GRAD_ACCUM))
    print(
        f"{steps_per_epoch * EPOCHS} optimisation steps "
        f"({steps_per_epoch}/epoch, {EPOCHS} epochs) at effective batch {BATCH_SIZE * GRAD_ACCUM}",
        flush=True,
    )

    arguments = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        run_name=RUN_NAME,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="constant",
        logging_steps=10,
        save_strategy="steps",
        save_steps=max(steps_per_epoch // 4, 1),
        save_total_limit=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        seed=SEED,
        data_seed=SEED,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = TrapezoidTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=build_collator(tokenizer.pad_token_id or tokenizer.eos_token_id),
    )
    if arguments.train_batch_size != trainer.per_step_batch:
        print(
            f"pinned per-step batch {arguments.train_batch_size} -> {trainer.per_step_batch} (n_gpu={arguments.n_gpu})",
            flush=True,
        )
    result = trainer.train()

    adapter_dir = OUT_DIR / "adapter-final"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    log_history = trainer.state.log_history
    losses = [entry["loss"] for entry in log_history if "loss" in entry]
    tail = losses[-10:]
    run = {
        "schema": SCHEMA,
        "run_name": RUN_NAME,
        "release_id": record.get("source_release_id"),
        "base_model": shared.BASE_MODEL_ID,
        "precision": "bf16-full",
        "datums": len(rows),
        "trained_tokens": trained,
        "export_sha256": verified["sha256"],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "per_step_batch": trainer.per_step_batch,
        "lora_r": RANK,
        "lora_alpha": ALPHA,
        "lr": LR,
        "min_lr": MIN_LR,
        "seed": SEED,
        "steps": int(result.global_step),
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "mean_loss_last_10": sum(tail) / len(tail) if tail else None,
        "elapsed_seconds": round(time.time() - started, 1),
        "adapter": str(adapter_dir),
        "log_history": log_history[-40:],
    }
    (OUT_DIR / "ovh-run.json").write_text(json.dumps(run, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in run.items() if k != "log_history"}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
