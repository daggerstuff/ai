"""Train Qwen3-8B on the v22 crisis export with QLoRA on Kaggle's free T4 pair.

The export is the same tinker-tokenized JSONL the CPU launcher reads, so the ids
transfer unchanged: Qwen3-8B shares the Qwen3 tokenizer vocabulary. Weights are
already masked to the assistant span, which becomes -100 in the labels here.

Every free GPU route is exhausted, so this kernel is the only path to a model
bigger than the 0.6B that fits on the local box. Kaggle caps a run at 12h and
the weekly GPU quota at 30h, so the recipe stays conservative: 4-bit NF4, LoRA
r=32, a small per-device batch with heavy accumulation, and a batch size that
halves itself rather than dying if the activations do not fit.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

INPUT_ROOT = Path("/kaggle/input")
OUTPUT_DIR = Path("/kaggle/working")
DATASET_FILE = "tinker-sft-train.jsonl"
RECORD_FILE = "tinker-dataset.json"

BASE_MODEL = "Qwen3-8B"
BASE_MODEL_ID = "Qwen/Qwen3-8B"
EXPORT_TOKENIZERS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B")
SCHEMA = "kaggle-sft-run-v1"
RUN_NAME = "kaggle-wandb-full-corpus-current-master-20261110-v22"

# None trains the whole export. Set to a small number to prove the plumbing.
LIMIT: int | None = None

EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 1e-4
MIN_LR = 1e-6
WARMUP_FRACTION = 0.05
HOLD_FRACTION = 0.6
RANK = 32
ALPHA = 32
SEED = 20261110
IGNORE_INDEX = -100


def install_dependencies() -> None:
    """Kaggle's image predates Qwen3 support, so top the stack up before importing."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            "transformers>=4.51",
            "peft>=0.14",
            "bitsandbytes>=0.45",
            "accelerate>=1.0",
            "datasets>=3.0",
        ],
        check=True,
    )


if Path("/kaggle/input").exists():
    install_dependencies()

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

if TYPE_CHECKING:
    # transformers resolves to Any under the strict ratchet's --no-site-packages
    # run, and strict mode refuses to subclass Any; model the surface used here.
    class _TrainerBase:
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None
        optimizer: torch.optim.Optimizer
        state: Any

        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

        def train(self) -> Any: ...

else:
    from transformers import Trainer as _TrainerBase


def locate_export(root: Path) -> Path:
    """Kaggle's mount layout shifts between image generations, so find the files."""
    for candidate in sorted(root.rglob(RECORD_FILE)):
        if (candidate.parent / DATASET_FILE).exists():
            return candidate.parent
    listing = "\n".join(str(path) for path in sorted(root.rglob("*"))[:40])
    raise FileNotFoundError(f"no export under {root}; found:\n{listing}")


def verify_export(input_dir: Path) -> dict[str, Any]:
    """Refuse to train on a file that is not the one the record describes."""
    record = json.loads((input_dir / RECORD_FILE).read_text())
    payload = (input_dir / DATASET_FILE).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != record["sha256"]:
        raise ValueError(f"{DATASET_FILE} does not match the recorded sha256")
    rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    if len(rows) != record["datums"]:
        raise ValueError(f"{DATASET_FILE} holds {len(rows)} datums, record says {record['datums']}")
    if record.get("tokenizer") not in EXPORT_TOKENIZERS:
        raise ValueError(f"{record.get('tokenizer')} was not the tokenizer behind this export")
    return {"record": record, "rows": rows, "sha256": digest}


def lr_lambda(steps: int) -> Callable[[float], float]:
    """Trapezoid: linear warmup, hold, then cosine down to the floor."""
    floor = MIN_LR / LR

    def schedule(step: float) -> float:
        progress = step / max(steps, 1)
        if progress < WARMUP_FRACTION:
            start = 0.2
            return start + (1.0 - start) * progress / WARMUP_FRACTION
        if progress < HOLD_FRACTION:
            return 1.0
        tail = (progress - HOLD_FRACTION) / (1.0 - HOLD_FRACTION)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(tail, 1.0)))

    return schedule


def build_collator(pad_id: int) -> Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]:
    """The export is already shifted for a raw cross-entropy, but Trainer shifts again.

    ``target_ids[i]`` is ``input_ids[i + 1]``, so the label at position ``i`` is
    ``input_ids[i]`` masked by ``weights[i - 1]``. Passing ``target_ids`` straight
    through would make the model predict one token too far ahead.
    """

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in batch)
        input_ids, labels, attention = [], [], []
        for row in batch:
            ids = list(row["input_ids"])
            weights = [0, *list(row["weights"])[:-1]]
            pad = width - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append([t if w else IGNORE_INDEX for t, w in zip(ids, weights, strict=True)] + [IGNORE_INDEX] * pad)
            attention.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    return collate


class TrapezoidTrainer(_TrainerBase):
    """transformers 5 dropped the "lambda" scheduler name, so own the schedule."""

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None
    ) -> torch.optim.lr_scheduler.LRScheduler:
        if self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer or self.optimizer,
                lr_lambda(num_training_steps),
            )
        return self.lr_scheduler


def build_model() -> Any:
    quantisation = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantisation,
        device_map={"": 0},
        dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=RANK,
            lora_alpha=ALPHA,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()
    return model


def main() -> None:
    started = time.time()
    print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}, gpus {torch.cuda.device_count()}", flush=True)

    verified = verify_export(locate_export(INPUT_ROOT))
    record = verified["record"]
    rows = verified["rows"]
    if LIMIT is not None:
        rows = rows[:LIMIT]
    trained = sum(sum(row["weights"]) for row in rows)
    print(
        f"export verified: {len(rows)} datums, {trained} trained tokens, "
        f"sha256 {verified['sha256'][:16]}, release {record.get('source_release_id')}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    dataset = Dataset.from_list(rows)
    model = build_model()

    steps_per_epoch = math.ceil(len(rows) / (BATCH_SIZE * GRAD_ACCUM))
    total_steps = steps_per_epoch * EPOCHS
    print(f"{total_steps} optimisation steps ({steps_per_epoch}/epoch, {EPOCHS} epochs)", flush=True)

    arguments = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        run_name=RUN_NAME,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        auto_find_batch_size=True,
        learning_rate=LR,
        lr_scheduler_type="constant",
        logging_steps=10,
        save_strategy="steps",
        save_steps=max(steps_per_epoch // 4, 1),
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        seed=SEED,
        data_seed=SEED,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = TrapezoidTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=build_collator(tokenizer.pad_token_id or tokenizer.eos_token_id),
    )
    result = trainer.train()

    adapter_dir = OUTPUT_DIR / "adapter-final"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    log_history = trainer.state.log_history
    losses = [entry["loss"] for entry in log_history if "loss" in entry]
    tail = losses[-10:]
    run = {
        "schema": SCHEMA,
        "run_name": RUN_NAME,
        "release_id": record.get("source_release_id"),
        "base_model": BASE_MODEL_ID,
        "datums": len(rows),
        "trained_tokens": trained,
        "export_sha256": verified["sha256"],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "lora_r": RANK,
        "lora_alpha": ALPHA,
        "lr": LR,
        "min_lr": MIN_LR,
        "seed": SEED,
        "steps": int(result.global_step),
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "mean_loss_last_10": sum(tail) / len(tail) if tail else None,
        "elapsed_seconds": round(time.time() - started, 1),
        "adapter": str(adapter_dir),
        "log_history": log_history[-40:],
    }
    (OUTPUT_DIR / "kaggle-run.json").write_text(json.dumps(run, indent=2, sort_keys=True))
    print(json.dumps(run, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
