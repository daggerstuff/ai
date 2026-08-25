"""Training-parameter optimizer: pick a batch profile and estimate runtime.

Selects a (batch_size, gradient_accumulation_steps, max_length) profile that
fits the H100 12-hour training window, estimates total hours, and returns a
ready-to-use ``transformers.TrainingArguments``.

The throughput constants below are conservative QLoRA-on-H100 heuristics
(tokens/sec per effective batch). They are intentionally coarse: a real run
should be calibrated once per base model, but the structure (profile ->
estimate -> TrainingArguments) is stable and unit-testable without a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from transformers import TrainingArguments

# tokens/sec delivered by the trainer at the profile's effective batch size.
# Larger effective batches amortize kernel launch overhead and lift throughput.
_TOKENS_PER_SEC: dict[str, float] = {
    "fast": 6000.0,
    "balanced": 3000.0,
    "quality": 1500.0,
}

_PROFILES: dict[str, dict[str, int]] = {
    "fast": {"batch_size": 8, "gradient_accumulation_steps": 2, "max_length": 1024},
    "balanced": {"batch_size": 4, "gradient_accumulation_steps": 8, "max_length": 2048},
    "quality": {"batch_size": 2, "gradient_accumulation_steps": 16, "max_length": 4096},
}

_SECONDS_PER_HOUR = 3600.0


@dataclass
class TrainingProfile:
    """Selected batch/context profile for a training run."""

    batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    num_epochs: int = 3
    priority: str = "balanced"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class TrainingEstimate:
    """Runtime estimate and any recommended parameter adjustments."""

    estimated_hours: float
    fits_in_window: bool
    recommended_adjustments: dict[str, float] = field(default_factory=dict)


def _estimate_hours(
    num_samples: int,
    avg_tokens_per_sample: int,
    num_epochs: int,
    tokens_per_sec: float,
) -> float:
    total_tokens = num_samples * avg_tokens_per_sample * num_epochs
    return total_tokens / tokens_per_sec / _SECONDS_PER_HOUR


def optimize_for_dataset(
    num_samples: int,
    avg_tokens_per_sample: int,
    num_epochs: int,
    priority: str = "balanced",
    max_hours: float = 12.0,
    deepspeed: str | None = None,
) -> tuple[TrainingProfile, TrainingEstimate, TrainingArguments]:
    """Pick a training profile that fits ``max_hours`` and build training args.

    Args:
        num_samples: Number of training samples.
        avg_tokens_per_sample: Mean token count per sample.
        num_epochs: Desired epochs (may be reduced by the estimator).
        priority: One of ``"fast"``, ``"balanced"``, ``"quality"``.
        max_hours: Hard wall-clock budget for the run.
        deepspeed: Optional path to a DeepSpeed config (e.g. ``ds_config_zero3.json``);
            passed through to ``TrainingArguments`` to enable ZeRO-3 multi-GPU training.

    Returns:
        Tuple of (profile, estimate, training_args).
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if avg_tokens_per_sample <= 0:
        raise ValueError("avg_tokens_per_sample must be positive")
    if priority not in _PROFILES:
        raise ValueError(f"priority must be one of {sorted(_PROFILES)}, got {priority!r}")
    if max_hours <= 0:
        raise ValueError("max_hours must be positive")

    spec = _PROFILES[priority]
    tokens_per_sec = _TOKENS_PER_SEC[priority]

    profile = TrainingProfile(
        batch_size=spec["batch_size"],
        gradient_accumulation_steps=spec["gradient_accumulation_steps"],
        max_length=spec["max_length"],
        num_epochs=num_epochs,
        priority=priority,
    )

    estimated_hours = _estimate_hours(
        num_samples, avg_tokens_per_sample, num_epochs, tokens_per_sec
    )

    recommended_adjustments: dict[str, float] = {}
    fits_in_window = estimated_hours <= max_hours
    if not fits_in_window:
        hours_per_epoch = estimated_hours / num_epochs
        max_epochs = int(max_hours / hours_per_epoch)
        if max_epochs < 1:
            max_epochs = 1
        if max_epochs < num_epochs:
            recommended_adjustments["new_num_epochs"] = float(max_epochs)
            estimated_hours = hours_per_epoch * max_epochs
            fits_in_window = estimated_hours <= max_hours

    estimate = TrainingEstimate(
        estimated_hours=round(estimated_hours, 2),
        fits_in_window=fits_in_window,
        recommended_adjustments=recommended_adjustments,
    )

    training_args = TrainingArguments(
        output_dir="./therapeutic_moe_model",
        num_train_epochs=profile.num_epochs,
        per_device_train_batch_size=profile.batch_size,
        per_device_eval_batch_size=profile.batch_size,
        gradient_accumulation_steps=profile.gradient_accumulation_steps,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        remove_unused_columns=True,
        deepspeed=deepspeed,
    )

    return profile, estimate, training_args