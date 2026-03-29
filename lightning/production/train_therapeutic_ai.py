#!/usr/bin/env python3
"""
Lightning.ai H100 Therapeutic AI Training Script
LoRA training for therapeutic conversation AI
"""

import argparse
import json
import logging
import os
import random
import sys
import warnings
from io import TextIOWrapper
from pathlib import Path
from typing import Dict

import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import lightning as L

# Suppress standard PEFT warning regarding modules in eval mode
warnings.filterwarnings("ignore", r".*Found \d+ module\(s\) in eval mode.*")

# Add repo root to path to import S3DatasetLoader
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from ai.utils.s3_dataset_loader import S3DatasetLoader
except ImportError:
    try:
        from ai.infrastructure.s3.s3_dataset_loader import S3DatasetLoader
    except ImportError:
        S3DatasetLoader = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override the configured DataLoader worker count.",
    )
    return parser.parse_args()


def load_stage_config(config_file: str) -> dict:
    config_path = Path(f"ai/lightning/production/stage_configs/{config_file}")
    logger.info(f"Loading config from {config_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as handle:
        config = json.load(handle)

    train_data_path_override = os.getenv("TRAIN_DATA_PATH")
    if train_data_path_override:
        config["train_data_path"] = train_data_path_override
        logger.info(f"Overriding train_data_path from env: {train_data_path_override}")

    return config


def path_exists(path: str) -> bool:
    if not path:
        return False
    if path.startswith("s3://"):
        if S3DatasetLoader is None:
            logger.warning("S3DatasetLoader unavailable; cannot verify S3 path existence")
            return False
        try:
            return S3DatasetLoader().object_exists(path)
        except Exception as exc:
            logger.warning(f"Unable to verify S3 path {path}: {type(exc).__name__}: {exc}")
            return False
    return Path(path).exists()


def build_dataloaders(
    *,
    config: dict,
    data_path: str,
    tokenizer,
    max_length: int,
    num_workers_override: int | None,
):
    train_dataset = TherapeuticConversationDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        is_val=False,
        val_split=0.05,
    )
    val_dataset = TherapeuticConversationDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        is_val=True,
        val_split=0.05,
    )

    logger.info(f"Initialized IterableDatasets streaming from {data_path}")

    num_workers = num_workers_override
    if num_workers is None:
        num_workers = config.get("dataloader_num_workers", config.get("num_workers", 4))

    dataloader_kwargs = {
        "batch_size": config.get("batch_size", 8),
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    return (
        torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs),
        torch.utils.data.DataLoader(val_dataset, **dataloader_kwargs),
    )


def save_training_artifacts(model, tokenizer, output_dir: str, base_model: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifact_manifest = {
        "base_model": base_model,
        "tokenizer_dir": ".",
    }
    peft_model = model.model
    if isinstance(peft_model, PeftModel):
        adapter_dir = output_path / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(adapter_dir)
        artifact_manifest["artifact_type"] = "peft_adapter"
        artifact_manifest["adapter_dir"] = adapter_dir.name
    else:
        peft_model.save_pretrained(output_path)
        artifact_manifest["artifact_type"] = "full_model"

    tokenizer.save_pretrained(output_path)
    with open(output_path / "artifact_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(artifact_manifest, handle, indent=2)


def build_tokenizer_and_context(config: dict):
    model_name = config.get("base_model", "meta-llama/Llama-3.2-3B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_max_length = getattr(tokenizer, "model_max_length", 1024)
    if model_max_length > 100000:
        model_max_length = 1024

    actual_max_length = min(config.get("context_length", 1024), model_max_length)
    return model_name, tokenizer, actual_max_length


def build_trainer(args, config: dict, wandb_logger):
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

    precision_mapping = {"bf16": "bf16-mixed", "fp16": "16-mixed", "32": "32-true"}
    trainer_kwargs = dict(
        max_epochs=config.get("epochs", 3),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="ddp_find_unused_parameters_false"
        if torch.cuda.device_count() > 1
        else "auto",
        precision=precision_mapping.get(config.get("precision", "fp16"), "16-mixed"),
        gradient_clip_val=1.0,
        accumulate_grad_batches=config.get("gradient_accumulation_steps", 4),
        val_check_interval=(2 if args.dry_run else config.get("eval_steps", 100))
        * config.get("gradient_accumulation_steps", 4),
        limit_val_batches=2 if args.dry_run else 50,
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

    return L.Trainer(**trainer_kwargs)


def run_training(trainer, model, train_loader, val_loader, config: dict, *, dry_run: bool):
    logger.info("🔥 Starting training...")
    ckpt_path = config.get("resume_from_checkpoint")
    if ckpt_path and path_exists(ckpt_path) and not dry_run:
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
        return
    trainer.fit(model, train_loader, val_loader)


class ConversationShardStreamer:
    """Low-level shard streaming for local and S3-backed conversation files."""

    def __init__(self, loader=None):
        self.loader = loader

    @staticmethod
    def iter_conversations_from_sequence(conversations):
        for conversation in conversations:
            if conversation:
                yield conversation

    def iter_json_array_stream(self, text_stream, *, initial_buffer: str = ""):
        decoder = json.JSONDecoder()
        buffer = initial_buffer
        started = False
        ended = False

        while True:
            chunk = ""
            if not ended:
                chunk = text_stream.read(65536)
                if chunk:
                    buffer += chunk

            index = 0
            while True:
                while index < len(buffer) and buffer[index].isspace():
                    index += 1

                if not started:
                    if index >= len(buffer):
                        break
                    if buffer[index] != "[":
                        raise ValueError("Expected top-level JSON array shard")
                    started = True
                    index += 1
                    continue

                while index < len(buffer) and buffer[index].isspace():
                    index += 1

                if index < len(buffer) and buffer[index] == "]":
                    ended = True
                    index += 1
                    break

                try:
                    conversation, index = decoder.raw_decode(buffer, index)
                except json.JSONDecodeError:
                    break

                if conversation:
                    yield conversation

                while index < len(buffer) and buffer[index].isspace():
                    index += 1
                if index < len(buffer) and buffer[index] == ",":
                    index += 1

            buffer = buffer[index:]
            if ended:
                return
            if not chunk:
                break

        if buffer.strip():
            raise ValueError("Incomplete JSON array shard")

    def iter_json_object_conversations(self, data):
        conversations = data if isinstance(data, list) else data.get("conversations", [])
        yield from self.iter_conversations_from_sequence(conversations)

    def iter_file_conversations(self, file_path: str):
        if file_path.startswith("s3://"):
            if self.loader is None:
                raise ImportError("S3DatasetLoader missing")
            if file_path.endswith(".jsonl"):
                yield from self.loader.stream_jsonl(file_path)
                return

            bucket, key = self.loader._parse_s3_path(file_path)
            response = self.loader.s3_client.get_object(Bucket=bucket, Key=key)
            with TextIOWrapper(response["Body"], encoding="utf-8", errors="replace") as handle:
                prefix = handle.read(4096)
                first_non_ws = next((char for char in prefix if not char.isspace()), "")
                if first_non_ws == "[":
                    yield from self.iter_json_array_stream(handle, initial_buffer=prefix)
                    return

            logger.warning(f"JSON object shard loaded to mem: {file_path}")
            data = self.loader.load_json(file_path)
        else:
            if file_path.endswith(".jsonl"):
                with open(file_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            yield json.loads(line)
                return

            with open(file_path, "r", encoding="utf-8") as handle:
                prefix = handle.read(4096)
                first_non_ws = next((char for char in prefix if not char.isspace()), "")
                if first_non_ws == "[":
                    yield from self.iter_json_array_stream(handle, initial_buffer=prefix)
                    return
                handle.seek(0)
                data = json.load(handle)

        yield from self.iter_json_object_conversations(data)


class TherapeuticConversationDataset(torch.utils.data.IterableDataset):
    """Iterable Dataset for therapeutic conversation training."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        is_val: bool = False,
        val_split: float = 0.05,
        shuffle_buffer_size: int = 128,
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_val = is_val
        self.val_split = val_split
        self.shuffle_buffer_size = max(1, shuffle_buffer_size)

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
        self.streamer = ConversationShardStreamer(getattr(self, "loader", None))

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

        shuffle_buffer = []

        def yield_buffered(conversation):
            shuffle_buffer.append(conversation)
            if len(shuffle_buffer) < self.shuffle_buffer_size:
                return
            index = random.randrange(len(shuffle_buffer))
            shuffle_buffer[index], shuffle_buffer[-1] = (
                shuffle_buffer[-1],
                shuffle_buffer[index],
            )
            yield self._process_conversation(shuffle_buffer.pop())

        for file_path in active_files:
            # S3 streams can break mid-transfer (IncompleteRead, connection
            # resets). Retry with backoff and convert terminal failures to
            # RuntimeError so DataLoader worker errors stay debuggable.
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    for conversation in self.streamer.iter_file_conversations(file_path):
                        yield from yield_buffered(conversation)

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
                            f"Escalating as RuntimeError."
                        )
                        raise RuntimeError(
                            f"Failed to stream shard {file_path}: {type(exc).__name__}: {exc}"
                        ) from exc

        while shuffle_buffer:
            index = random.randrange(len(shuffle_buffer))
            shuffle_buffer[index], shuffle_buffer[-1] = (
                shuffle_buffer[-1],
                shuffle_buffer[index],
            )
            yield self._process_conversation(shuffle_buffer.pop())

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
    """Lightning trainer for therapeutic AI with standard LoRA adapters."""

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
                bnb_4bit_compute_dtype=torch.bfloat16
                if config.get("precision") == "bf16"
                else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
            if config.get("precision") == "bf16"
            else torch.float16,
            quantization_config=quant_config,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

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

        if config.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
            logger.info("🚀 Gradient checkpointing enabled")

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
    args = parse_args()
    config_map = {
        1: "stage1_foundation.json",
        2: "stage2_reasoning.json",
        3: "stage3_voice.json",
    }

    config_file = config_map[args.stage]
    config = load_stage_config(config_file)

    logger.info(
        f"🚀 Starting Lightning.ai H100 Therapeutic AI Training - Stage {args.stage}"
    )

    data_path = config["train_data_path"]
    model_name, tokenizer, actual_max_length = build_tokenizer_and_context(config)
    train_loader, val_loader = build_dataloaders(
        config=config,
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=actual_max_length,
        num_workers_override=args.num_workers,
    )

    model = TherapeuticTrainer(config)
    wandb_logger = WandbLogger(
        project=config.get("project_name", "pixelated-empathy-training"),
        name=config.get("run_name", f"stage{args.stage}_training"),
        log_model="all",
    )
    trainer = build_trainer(args, config, wandb_logger)
    run_training(
        trainer,
        model,
        train_loader,
        val_loader,
        config,
        dry_run=args.dry_run,
    )

    output_dir = f"./therapeutic_ai_final_stage{args.stage}"
    save_training_artifacts(model, tokenizer, output_dir, model_name)

    logger.info(f"🎉 Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
