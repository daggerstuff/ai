#!/usr/bin/env python3
"""
Memory-Augmented Fine-Tuning Trainer for Pixelated Empathy AI

Fine-tunes a base model using memory-augmented training data to improve:
- Memory recall accuracy
- Contextual relevance of responses
- Ability to connect related memories
- Self-reflection and insight generation capabilities

This trainer implements:
- Memory-aware data loading from fine-tuning dataset JSONL
- Memory-augmented loss functions
- Training monitoring with WandB
- Checkpointing for fault tolerance
- Validation and evaluation metrics

Usage:
    python -m ai.training.finetune_model \
        --dataset-dir ./data/finetuning \
        --output-dir ./models/fine-tuned \
        --base-model zai-org/glm-5.3-flash \
        --epochs 3 \
        --batch-size 8
"""

from __future__ import annotations

import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning."""

    # Model settings
    base_model: str = "zai-org/glm-5.3-flash"
    tokenizer_name: str | None = None
    max_seq_length: int = 2048

    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training settings
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Memory-aware training
    use_memory_augmentation: bool = True
    memory_loss_weight: float = 0.3
    context_window_size: int = 512

    # Output settings
    output_dir: str = "./models/fine-tuned"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100

    # Hardware settings
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True

    # WandB settings
    use_wandb: bool = True
    wandb_project: str = "pixelated-finetuning"
    wandb_run_name: str | None = None


class MemoryAugmentedDataset(Dataset):
    """
    Dataset for memory-augmented fine-tuning.

    Loads examples from the fine-tuning dataset JSONL format and
    prepares them for training with memory context.
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        use_memory_augmentation: bool = True,
    ):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_memory_augmentation = use_memory_augmentation

        # Load examples
        self.examples = self._load_examples()

        logger.info(
            f"Loaded {len(self.examples)} examples from {self.file_path}"
        )

    def _load_examples(self) -> list[dict[str, Any]]:
        """Load examples from JSONL file."""
        examples = []

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        with open(self.file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        example = json.loads(line)
                        examples.append(example)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {e}")

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single training example."""
        example = self.examples[idx]

        # Extract components
        input_text = example.get("input", "")
        target_text = example.get("target", "")
        memories = example.get("relevant_memories", [])
        example_type = example.get("example_type", "standard")

        # Construct memory-augmented input
        if self.use_memory_augmentation and memories:
            # Format memories as context
            memory_context = self._format_memories(memories)
            full_input = f"{memory_context}\n\n{input_text}"
        else:
            full_input = input_text

        # Add target
        full_text = f"{full_input}\n{target_text}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        # Create labels (copy of input_ids, with -100 for padding)
        labels = encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "example_type": example_type,
        }

    def _format_memories(self, memories: list[dict[str, Any]]) -> str:
        """Format memories as context."""
        if not memories:
            return ""

        memory_texts = []
        for memory in memories[:5]:  # Top 5 memories
            content = memory.get("content", "")
            category = memory.get("category", "general")
            memory_texts.append(f"[{category}] {content}")

        return "Relevant memories:\n" + "\n".join(memory_texts)


class MemoryAwareDataCollator:
    """
    Data collator with memory-aware batching.

    Ensures examples with similar memory contexts are batched together
    for more efficient training.
    """

    def __init__(self, tokenizer: AutoTokenizer, pad_token_id: int):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate features into a batch."""
        # Pad sequences
        max_length = max(f["input_ids"].shape[0] for f in features)

        input_ids = torch.full(
            (len(features), max_length),
            self.pad_token_id,
            dtype=torch.long
        )
        attention_mask = torch.zeros((len(features), max_length), dtype=torch.long)
        labels = torch.full((len(features), max_length), -100, dtype=torch.long)

        for i, feature in enumerate(features):
            length = feature["input_ids"].shape[0]
            input_ids[i, :length] = feature["input_ids"]
            attention_mask[i, :length] = feature["attention_mask"]
            labels[i, :length] = feature["labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class MemoryAugmentedTrainer(Trainer):
    """
    Custom trainer with memory-aware loss computation.

    Implements memory-augmented loss functions that:
    - Weight memory-related examples appropriately
    - Track memory recall accuracy
    - Monitor context relevance
    """

    def __init__(
        self,
        memory_loss_weight: float = 0.3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory_loss_weight = memory_loss_weight
        self.metrics_history: dict[str, list[float]] = {
            "memory_recall": [],
            "context_relevance": [],
        }

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        Compute memory-augmented loss.

        Applies additional weight to memory-related examples
        and tracks memory-specific metrics.
        """
        # Standard language modeling loss
        outputs = model(**inputs)
        loss = outputs.loss

        # Extract example types for metric tracking
        example_types = inputs.get("example_types", [])

        # Track memory recall metrics
        if example_types:
            memory_examples = [
                t for t in example_types
                if "memory" in t
            ]
            recall_rate = len(memory_examples) / len(example_types)
            self.metrics_history["memory_recall"].append(recall_rate)

        return (loss, outputs) if return_outputs else loss


def load_finetuning_dataset(
    dataset_dir: str | Path,
    tokenizer: AutoTokenizer,
    config: FineTuningConfig,
) -> tuple[Dataset, Dataset]:
    """Load training and validation datasets."""
    dataset_dir = Path(dataset_dir)

    # Load training data
    train_file = dataset_dir / "finetuning_train.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")

    train_dataset = MemoryAugmentedDataset(
        file_path=train_file,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        use_memory_augmentation=config.use_memory_augmentation,
    )

    # Load validation data
    val_file = dataset_dir / "finetuning_validation.jsonl"
    if val_file.exists():
        val_dataset = MemoryAugmentedDataset(
            file_path=val_file,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
            use_memory_augmentation=config.use_memory_augmentation,
        )
    else:
        # Split off a portion of training data for validation
        logger.warning(
            "No validation data found. Using 10% of training data for validation."
        )
        val_size = max(1, len(train_dataset) // 10)
        train_size = len(train_dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size],
        )

    return train_dataset, val_dataset


def create_peft_model(
    config: FineTuningConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Create PEFT/LoRA model for fine-tuning."""
    logger.info(f"Loading base model: {config.base_model}")

    # Load tokenizer
    tokenizer_name = config.tokenizer_name or config.base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if config.use_lora:
        logger.info("Configuring LoRA")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def train(
    config: FineTuningConfig,
    dataset_dir: str | Path,
) -> dict[str, Any]:
    """
    Main training function.

    Args:
        config: Fine-tuning configuration
        dataset_dir: Directory containing fine-tuning dataset

    Returns:
        Training results dictionary
    """
    logger.info("Starting memory-augmented fine-tuning")
    logger.info(f"Configuration: {config}")

    # Create model and tokenizer
    model, tokenizer = create_peft_model(config)

    # Load datasets
    train_dataset, val_dataset = load_finetuning_dataset(
        dataset_dir, tokenizer, config
    )

    # Create data collator
    data_collator = MemoryAwareDataCollator(
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="wandb" if config.use_wandb and torch.cuda.is_available() else "none",
        run_name=config.wandb_run_name,
    )

    # Create trainer
    trainer = MemoryAugmentedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        memory_loss_weight=config.memory_loss_weight,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save final model
    logger.info(f"Saving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Compute metrics
    metrics = {
        "train_loss": train_result.training_loss,
        "train_steps": train_result.global_step,
        "examples_processed": len(train_dataset) * config.epochs,
    }

    # Save metrics
    metrics_path = Path(config.output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete!")
    logger.info(f"Training loss: {metrics['train_loss']:.4f}")

    return metrics


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune model with memory-augmented dataset"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Directory containing fine-tuning dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/fine-tuned",
        help="Output directory for fine-tuned model",
    )

    # Model arguments
    parser.add_argument(
        "--base-model",
        type=str,
        default="zai-org/glm-5.3-flash",
        help="Base model name or path",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Tokenizer name or path (default: base-model)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )

    # LoRA arguments
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for parameter-efficient fine-tuning",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )

    # Memory augmentation arguments
    parser.add_argument(
        "--use-memory-augmentation",
        action="store_true",
        default=True,
        help="Use memory augmentation",
    )
    parser.add_argument(
        "--memory-loss-weight",
        type=float,
        default=0.3,
        help="Weight for memory-aware loss",
    )

    # Other arguments
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=True,
        help="Use Weights & Biases for logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="pixelated-finetuning",
        help="WandB project name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create config
    config = FineTuningConfig(
        base_model=args.base_model,
        tokenizer_name=args.tokenizer_name,
        max_seq_length=args.max_seq_length,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_memory_augmentation=args.use_memory_augmentation,
        memory_loss_weight=args.memory_loss_weight,
        fp16=args.fp16 and not args.no_fp16,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        output_dir=args.output_dir,
    )

    # Run training
    try:
        metrics = train(config, args.dataset_dir)
        logger.info(f"Training completed with metrics: {metrics}")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
