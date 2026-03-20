#!/usr/bin/env python3
"""
Lightning.ai H100 Therapeutic AI Training Script
4-Expert MoE LoRA training for therapeutic conversation AI
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict

import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import lightning as L

# Suppress standard PEFT warning regarding modules in eval mode
warnings.filterwarnings("ignore", ".*Found \d+ module\(s\) in eval mode.*")

# Add repo root to path to import S3DatasetLoader
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from ai.utils.s3_dataset_loader import S3DatasetLoader
except ImportError:
    S3DatasetLoader = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TherapeuticConversationDataset(torch.utils.data.IterableDataset):
    """Iterable Dataset for therapeutic conversation training.

    Streams directly from S3 JSONL files.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        is_val: bool = False,
        val_split: float = 0.05,
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_val = is_val
        self.val_split = val_split

        self.files = []
        if self.data_path.startswith("s3://"):
            if S3DatasetLoader is None:
                raise ImportError("S3DatasetLoader missing")
            self.loader = S3DatasetLoader()
            if any(self.data_path.endswith(ext) for ext in [".json", ".jsonl"]):
                self.files = [self.data_path]
            else:
                prefix = self.data_path.replace("s3://", "").split("/", 1)
                prefix_path = prefix[1] if len(prefix) > 1 else ""
                all_files = self.loader.list_datasets(prefix=prefix_path)
                shard_prefix = "val_" if self.is_val else "train_"
                self.files = [f for f in all_files if shard_prefix in f.split("/")[-1]]

                # Sort them so they are deterministic across workers
                self.files.sort()
        else:
            path = Path(self.data_path)
            if path.is_file():
                self.files = [str(path)]
            else:
                shard_prefix = "val_" if self.is_val else "train_"
                self.files = [str(f) for f in path.glob(f"*{shard_prefix}*.jsonl")] + [
                    str(f) for f in path.glob(f"*{shard_prefix}*.json")
                ]
                self.files.sort()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # Get rank info if in DDP
        rank = 0
        world_size = 1
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        # First, split files across DDP ranks
        files_for_rank = [
            self.files[i] for i in range(len(self.files)) if i % world_size == rank
        ]

        if not worker_info:
            # Single-process data loading, yield all files for this rank
            active_files = files_for_rank
        else:
            # Multi-process data loading, split files_for_rank across workers
            active_files = [
                files_for_rank[i]
                for i in range(len(files_for_rank))
                if i % worker_info.num_workers == worker_info.id
            ]

        for file_path in active_files:
            # S3 streams can break mid-transfer (IncompleteRead, connection
            # resets). Retry with backoff; if all attempts fail, skip the
            # shard and continue training. Losing a few records from one
            # shard is far less damaging than crashing the entire job.
            #
            # botocore exceptions also cannot survive PyTorch DataLoader
            # cross-process serialization, so we convert them to
            # RuntimeError if they do bubble up.
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    iterator = []
                    if file_path.startswith("s3://"):
                        if file_path.endswith(".jsonl"):
                            iterator = self.loader.stream_jsonl(file_path)
                        elif file_path.endswith(".json"):
                            logger.warning(f"Streaming JSON loads to mem: {file_path}")
                            data = self.loader.load_json(file_path)
                            if isinstance(data, list):
                                conversations = data
                            else:
                                conversations = data.get("conversations", [])
                            conversations.reverse()

                            def popping_iterator(convs):
                                while convs:
                                    yield convs.pop()

                            iterator = popping_iterator(conversations)
                    else:
                        if file_path.endswith(".jsonl"):
                            iterator = (
                                json.loads(line)
                                for line in open(file_path, "r", encoding="utf-8")
                                if line.strip()
                            )
                        else:
                            with open(file_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    conversations = data
                                else:
                                    conversations = data.get("conversations", [])
                                conversations.reverse()

                                def popping_iterator(convs):
                                    while convs:
                                        yield convs.pop()

                                iterator = popping_iterator(conversations)

                    for conversation in iterator:
                        if not conversation:
                            continue
                        yield self._process_conversation(conversation)

                    # Success — break retry loop
                    break

                except RuntimeError:
                    raise
                except Exception as exc:
                    if attempt < max_retries:
                        import time

                        wait = 2**attempt
                        logger.warning(
                            f"S3 stream error on {file_path} "
                            f"(attempt {attempt}/{max_retries}): "
                            f"{type(exc).__name__}: {exc}. "
                            f"Retrying in {wait}s..."
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            f"S3 stream failed after {max_retries} attempts "
                            f"for {file_path}: {type(exc).__name__}: {exc}. "
                            f"Skipping shard."
                        )
                        break

    def _process_conversation(self, conversation):
        conv_data = conversation.get("messages", conversation.get("conversation", []))
        text_parts = []

        for turn in conv_data:
            role = turn.get("role", "")
            role_str = "Human" if role in ("user", "client", "human") else "Assistant"
            text_parts.append(f"{role_str}: {turn.get('content', '')}")

        full_text = "\n".join(text_parts)
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class TherapeuticTrainer(L.LightningModule):
    """Lightning trainer for therapeutic AI with MoE LoRA"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # Initialize model and tokenizer
        model_name = config.get("base_model", "meta-llama/Llama-3.2-3B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization if requested
        quant_config = None
        if config.get("quantization") == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=(
                    torch.bfloat16
                    if config.get("precision") == "bf16"
                    else torch.float16
                ),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=(
                torch.bfloat16 if config.get("precision") == "bf16" else torch.float16
            ),
            quantization_config=quant_config,
            device_map=(
                {"": int(os.environ.get("LOCAL_RANK", 0))} if quant_config else None
            ),
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        if config.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
            logger.info("🚀 Gradient checkpointing enabled")

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)

        logger.info(f"✅ Model initialized: {model_name} with LoRA")
        logger.info(f"   Trainable parameters: {self.model.num_parameters()}")

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train/perplexity",
            torch.exp(loss),
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        # Explicitly log validation loss on every step to see progress in WandB
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "val/perplexity",
            torch.exp(loss),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.get("learning_rate", 2e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.get("epochs", 3)
        )

        return [optimizer], [scheduler]


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Therapeutic AI Training")
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Training stage (1=foundation, 2=reasoning, 3=voice)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a quick verification pass without full training",
    )
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="Max steps (used for dry runs)"
    )

    args = parser.parse_args()

    config_map = {
        1: "stage1_foundation.json",
        2: "stage2_reasoning.json",
        3: "stage3_voice.json",
    }

    config_file = config_map[args.stage]
    config_path = Path(f"ai/lightning/production/stage_configs/{config_file}")

    logger.info(
        f"🚀 Starting Lightning.ai H100 Therapeutic AI Training - Stage {args.stage}"
    )
    logger.info(f"Loading config from {config_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Dataset path
    data_path = config["train_data_path"]

    # Determine base model id
    model_name = config.get("base_model", "meta-llama/Llama-3.2-3B-Instruct")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prevent IndexErrors by capping to the model's absolute maximum length.
    # Usually 1024 for DialoGPT.
    model_max_length = getattr(tokenizer, "model_max_length", 1024)
    # Some tokenizers incorrectly report huge numbers like 100000000000000
    if model_max_length > 100000:
        model_max_length = 1024

    actual_max_length = min(config.get("context_length", 1024), model_max_length)

    # Create datasets as IterableDatasets for memory safety
    train_dataset = TherapeuticConversationDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=actual_max_length,
        is_val=False,
        val_split=0.05,
    )
    val_dataset = TherapeuticConversationDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=actual_max_length,
        is_val=True,
        val_split=0.05,
    )

    logger.info(f"Initialized IterableDatasets streaming from {data_path}")

    # Create data loaders
    # Optimize num_workers, pin_memory, and persistent_workers for GPU performance
    num_workers = config.get("num_workers", 4)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 8),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # Initialize model
    model = TherapeuticTrainer(config)

    # Setup WandB logger
    wandb_logger = WandbLogger(
        project=config.get("project_name", "pixelated-empathy-training"),
        name=config.get("run_name", f"stage{args.stage}_training"),
        log_model="all",
    )

    precision_mapping = {"bf16": "bf16-mixed", "fp16": "16-mixed", "32": "32-true"}

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=f"./lightning_logs/stage{args.stage}/checkpoints",
            filename="wayfarer-{epoch:02d}-{val/loss:.2f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            every_n_train_steps=None if args.dry_run else config.get("save_steps", 500),
        ),
    ]

    # Configure trainer
    trainer_kwargs = dict(
        max_epochs=config.get("epochs", 3),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_false"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        precision=precision_mapping.get(config.get("precision", "fp16"), "16-mixed"),
        gradient_clip_val=1.0,
        accumulate_grad_batches=config.get("gradient_accumulation_steps", 4),
        val_check_interval=(2 if args.dry_run else config.get("eval_steps", 100))
        * config.get("gradient_accumulation_steps", 4),
        limit_val_batches=2 if args.dry_run else 50,  # Prevent massive S3 val hangs
        enable_checkpointing=True,
        default_root_dir=f"./lightning_logs/stage{args.stage}",
        logger=wandb_logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )

    if args.dry_run:
        trainer_kwargs["max_steps"] = args.max_steps if args.max_steps > 0 else 1
        trainer_kwargs["limit_train_batches"] = 2
        trainer_kwargs["limit_val_batches"] = 2
        logger.info("🧪 Running in DRY RUN mode")

    trainer = L.Trainer(**trainer_kwargs)

    # Start training
    logger.info(f"🔥 Starting training (Stage {args.stage})...")

    ckpt_path = config.get("resume_from_checkpoint")
    if ckpt_path and Path(ckpt_path).exists() and not args.dry_run:
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, train_loader, val_loader)

    # Save final model
    output_dir = f"./therapeutic_ai_final_stage{args.stage}"
    model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"🎉 Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
