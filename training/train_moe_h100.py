#!/usr/bin/env python3
"""
Therapeutic AI Training with MoE Architecture on H100
Optimized for 12-hour training window with LoRA fine-tuning
"""

import contextlib
import json
import os
import signal
import time
from datetime import UTC, datetime
from typing import Any

import torch
import wandb
from datasets import Dataset
from tqdm import tqdm

try:
    from ai.models.moe_architecture import MoEConfig, create_therapeutic_moe_model
except ImportError:  # pragma: no cover - run as a plain script
    from models.moe_architecture import MoEConfig, create_therapeutic_moe_model
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Global shutdown flag
shutdown_requested = False
training_start_time = None
MAX_TRAINING_HOURS = 12


def should_apply_safety_filter(sample: dict, safety_config: dict) -> bool:
    """
    Determine if safety filtering should be applied to a sample.

    Edge cases (crisis/trauma content marked for training) bypass safety filtering
    to ensure the model learns to handle these critical scenarios appropriately.

    Args:
        sample: Training sample with optional is_training_edge_case flag
        safety_config: Safety configuration dict

    Returns:
        True if safety filtering should be applied, False if bypassed
    """
    # Check if edge case bypass is enabled
    if safety_config.get("edge_case_bypass", {}).get("enabled", True):
        # Bypass safety filtering for training edge cases
        if sample.get("is_training_edge_case", False):
            return False

    # Default: apply safety filtering
    return True


def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


class TimeConstraintCallback(TrainerCallback):
    """Callback to enforce 12-hour training window"""

    def __init__(self, max_hours: int = 12):
        self.max_hours = max_hours
        self.start_time = None
        self.last_checkpoint_time = None
        self.checkpoint_interval_minutes = 30

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time

    def on_step_end(self, args, state, control, **kwargs):
        global shutdown_requested

        if shutdown_requested:
            control.should_training_stop = True
            control.should_save = True
            return control

        current_time = time.time()
        elapsed_hours = (current_time - self.start_time) / 3600

        # Check if we're approaching time limit
        if elapsed_hours >= self.max_hours - 0.5:  # Stop 30 min before limit
            control.should_training_stop = True
            control.should_save = True

            wandb.log(
                {
                    "training/stopped_reason": "time_limit",
                    "training/elapsed_hours": elapsed_hours,
                }
            )

        # Periodic checkpointing (every 30 minutes)
        elapsed_since_checkpoint = (current_time - self.last_checkpoint_time) / 60
        if elapsed_since_checkpoint >= self.checkpoint_interval_minutes:
            control.should_save = True
            self.last_checkpoint_time = current_time

        # Log time progress
        if state.global_step % 100 == 0:
            remaining_hours = self.max_hours - elapsed_hours
            wandb.log(
                {
                    "training/elapsed_hours": elapsed_hours,
                    "training/remaining_hours": remaining_hours,
                    "training/progress_percent": (elapsed_hours / self.max_hours) * 100,
                }
            )

        return control


class MoETrainingCallback(TrainerCallback):
    """Callback for MoE-specific monitoring"""

    def __init__(self, safety_config: dict[str, Any]):
        self.safety_config = safety_config
        self.step_count = 0
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.patience = 3
        self.edge_case_count = 0
        self.edge_case_bypass_count = 0

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        global shutdown_requested

        if shutdown_requested:
            control.should_training_stop = True
            return control

        if logs:
            self.step_count += 1
            logs.get("loss", logs.get("train_loss", 0))

            # Early stopping based on validation loss
            if "eval_loss" in logs:
                eval_loss = logs["eval_loss"]
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    control.should_training_stop = True
                    control.should_save = True

                    wandb.log(
                        {
                            "training/stopped_reason": "early_stopping",
                            "training/best_eval_loss": self.best_loss,
                        }
                    )

            # Log progress
            if "epoch" in logs:
                (logs["epoch"] / args.num_train_epochs) * 100

            # Enhanced logging
            enhanced_logs = logs.copy()
            enhanced_logs.update(
                {
                    "training/steps_completed": self.step_count,
                    "training/best_loss": self.best_loss,
                    "training/patience_counter": self.patience_counter,
                    "system/shutdown_requested": shutdown_requested,
                "safety/edge_case_count": self.edge_case_count,
                "safety/edge_case_bypass_count": self.edge_case_bypass_count,
                }
            )

            wandb.log(enhanced_logs, step=state.global_step)

        return control


def setup_wandb(config_path: str = "wandb_config.json"):
    """Setup Weights & Biases logging"""
    with open(config_path) as f:
        config = json.load(f)

    if not torch.cuda.is_available():
        os.environ["WANDB_MODE"] = "offline"

    return wandb.init(
        project=config["project"],
        entity=config.get("entity"),
        name=f"{config['name']}_moe_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        tags=config["tags"] + ["moe", "h100", "lora"],
        notes=f"MoE training with LoRA on H100 - {config['notes']}",
        config=config["config"],
    )


def _nemotron_record_to_text(record: dict[str, Any]) -> str | None:
    """
    Convert a Nemotron3 evaluation record into a plain-text training example.

    Expected record schema (from nemotron3_evaluate.py):
      {
        "input": { "messages" | "conversations" | "prompt"/"input": ... },
        "nemotron3_response": { "role": "assistant", "content": "..." },
        ...
      }
    """
    inp = record.get("input") or {}
    messages = inp.get("messages") or inp.get("conversations")

    lines = []
    if messages and isinstance(messages, list):
        for msg in messages:
            role = msg.get("role", "user")
            if content := msg.get("content", ""):
                lines.append(f"{role}: {content}")
    elif prompt := inp.get("prompt") or inp.get("input"):
        # Fallback to single-turn prompt
        lines.append(f"user: {prompt}")

    answer = record.get("nemotron3_response") or {}
    if answer_content := answer.get("content", ""):
        lines.append(f"{answer.get('role', 'assistant')}: {answer_content}")

    text = "\n".join(lines).strip()
    return text or None


def load_training_data(
    dataset_path: str | None = None,
    s3_path: str | None = None,
    nemotron_teacher_s3_path: str | None = None,
    nemotron_teacher_max_samples: int = 0,
) -> tuple[Dataset, list]:
    """
    Load and prepare training dataset from S3 (canonical) or local (fallback).

    Args:
        dataset_path: Local file path (for backward compatibility)
        s3_path: S3 path (s3://bucket/key) - preferred, S3 is canonical
    """
    # Prefer S3 if provided
    if s3_path:
        from ai.training.utils.s3_dataset_loader import S3DatasetLoader

        loader = S3DatasetLoader()
        data = loader.load_json(s3_path)
    elif dataset_path:
        # Fallback to local file
        with open(dataset_path) as f:
            data = json.load(f)
    else:
        # Try to find dataset in S3
        from ai.training.utils.s3_dataset_loader import (
            S3DatasetLoader,
            get_s3_dataset_path,
            load_dataset_from_s3,
        )

        try:
            s3_path = get_s3_dataset_path("training_dataset.json")
            data = load_dataset_from_s3("training_dataset.json")
        except Exception as e:
            raise FileNotFoundError(
                f"Dataset not found. Provide dataset_path or s3_path, "
                f"or ensure training_dataset.json exists in S3. Error: {e}"
            ) from e

    texts = [conv["text"] for conv in tqdm(data["conversations"], desc="Loading")]

    # Optionally append Nemotron teacher SFT examples (generic reasoning warmup)
    if nemotron_teacher_s3_path and nemotron_teacher_max_samples > 0:
        loader = S3DatasetLoader()

        teacher_count = 0
        for record in loader.stream_jsonl(nemotron_teacher_s3_path):
            if not isinstance(record, dict):
                continue
            if text := _nemotron_record_to_text(record):
                texts.append(text)
                teacher_count += 1
            if teacher_count >= nemotron_teacher_max_samples:
                break


    dataset = Dataset.from_dict({"text": texts})


    return dataset, texts


def create_h100_training_args(
    output_dir: str = "./therapeutic_moe_model",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 3e-4,
    warmup_steps: int = 1000,
    max_steps: int = -1,
    deepspeed: str | None = None,
) -> TrainingArguments:
    """
    Create H100-optimized training arguments

    Optimized for:
    - 12-hour training window
    - H100 GPU memory (80GB)
    - LoRA fine-tuning efficiency
    """

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        # Batch size optimization for H100
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # Learning rate with warmup
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        # Regularization
        weight_decay=0.01,
        max_grad_norm=1.0,
        # H100 optimizations
        bf16=True,  # BFloat16 for H100
        bf16_full_eval=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        # Logging and checkpointing
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,
        # Evaluation
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # WandB reporting
        report_to="wandb",
        # Performance
        optim="adamw_torch_fused",  # Fused optimizer for H100
        # Disable unnecessary features
        push_to_hub=False,
        remove_unused_columns=True,
        # Distributed: DeepSpeed ZeRO-3 (optional). Path to ds_config_zero3.json.
        deepspeed=deepspeed,
    )


def main():
    global shutdown_requested, training_start_time


    training_start_time = datetime.now(UTC)
    wandb_run = None

    try:
        # Setup WandB
        wandb_run = setup_wandb()

        # Load configurations
        with open("training_config.json") as f:
            training_config = json.load(f)

        with open("safety_config.json") as f:
            safety_config = json.load(f)

        # Optional Nemotron teacher config (for generic reasoning warmup/distillation)
        nemotron_cfg = training_config.get("nemotron_teacher", {})
        nemotron_teacher_s3_path = None
        nemotron_teacher_max_samples = 0

        if isinstance(nemotron_cfg, dict) and nemotron_cfg.get("enabled"):
            nemotron_teacher_s3_path = nemotron_cfg.get("s3_path")
            nemotron_teacher_max_samples = int(nemotron_cfg.get("max_samples", 0))
            if not nemotron_teacher_s3_path:
                nemotron_teacher_s3_path = None
                nemotron_teacher_max_samples = 0

        # Load dataset (optionally augmented with Nemotron teacher SFT examples)
        dataset, texts = load_training_data(
            nemotron_teacher_s3_path=nemotron_teacher_s3_path,
            nemotron_teacher_max_samples=nemotron_teacher_max_samples,
        )

        wandb.log(
            {
                "dataset/total_conversations": len(texts),
                "dataset/avg_length": sum(len(text.split()) for text in texts)
                / len(texts),
            }
        )

        # Setup model
        BASE_MODEL_NAME = training_config.get(
            "base_model", "deepseek-ai/DeepSeek-V4-Pro"
        )
        device_available = torch.cuda.is_available()

        if not device_available:
            return


        # Create MoE configuration
        moe_config = MoEConfig(
            num_experts=4,
            expert_domains=[
                "psychology",
                "mental_health",
                "bias_detection",
                "general_therapeutic",
            ],
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            max_position_embeddings=8192,
            expert_capacity=2,
            load_balancing_weight=0.01,
        )

        # Create therapeutic MoE model
        model = create_therapeutic_moe_model(
            BASE_MODEL_NAME, moe_config=moe_config, device="auto"
        )


        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        wandb.log(
            {
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/trainable_percent": (trainable_params / total_params) * 100,
                "model/num_experts": moe_config.num_experts,
                "model/lora_rank": moe_config.lora_r,
                "model/context_length": moe_config.max_position_embeddings,
            }
        )

        (trainable_params / total_params) * 100

        # Setup tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize dataset
        def tokenize_function(examples):
            result = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=2048,  # Use 2048 for training, model supports 8192
            )
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"], desc="Tokenizing"
        )


        if len(tokenized_dataset) == 0:
            raise ValueError("Empty dataset after tokenization!")

        # Split into train/eval
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]


        # Create H100-optimized training arguments
        training_args = create_h100_training_args(
            output_dir="./therapeutic_moe_model",
            num_train_epochs=training_config.get("num_train_epochs", 3),
            per_device_train_batch_size=training_config.get(
                "per_device_train_batch_size", 4
            ),
            gradient_accumulation_steps=training_config.get(
                "gradient_accumulation_steps", 8
            ),
            learning_rate=training_config.get("learning_rate", 3e-4),
            warmup_steps=training_config.get("warmup_steps", 1000),
        )

        # Create trainer with callbacks
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=[
                TimeConstraintCallback(max_hours=MAX_TRAINING_HOURS),
                MoETrainingCallback(safety_config),
            ],
        )

        # Train
        if not shutdown_requested:
            wandb.log({"training/status": "started"})

            (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
            )

            trainer.train()

            trainer.save_model()
            tokenizer.save_pretrained(training_args.output_dir)

            # Save MoE-specific components
            model.save_pretrained(training_args.output_dir)

            wandb.log({"training/status": "completed"})

            (
                datetime.now(UTC) - training_start_time
            ).total_seconds() / 3600

    except KeyboardInterrupt:
        if wandb_run:
            wandb.log({"training/status": "interrupted"})

    except Exception as e:
        if wandb_run:
            with contextlib.suppress(Exception):
                wandb.log({"training/status": "failed", "training/error": str(e)})
        raise

    finally:
        if wandb_run:
            with contextlib.suppress(Exception):
                wandb.finish()
        if training_start_time:
            (
                datetime.now(UTC) - training_start_time
            ).total_seconds() / 3600



if __name__ == "__main__":
    main()
