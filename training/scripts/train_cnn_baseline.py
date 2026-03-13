#!/usr/bin/env python3
"""
Training script for CNN Feature Extraction Baseline (PIX-002).

This script trains the lightweight CNN model for emotional feature extraction
from text. The model outputs VAD (Valence-Arousal-Dominance) scores and
emotion classification logits.

Usage:
    uv run python ai/training/scripts/train_cnn_baseline.py \
        --config configs/cnn_baseline.yaml

The training uses:
    - Mixed precision (FP16/BF16) for faster training
    - Gradient checkpointing for memory efficiency
    - Cosine annealing learning rate schedule
    - Early stopping based on validation loss
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, random_split

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from training.models.base.cnn_feature_extractor import (
    CNNFeatureConfig,
    CNNFeatureExtractor,
    create_cnn_baseline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for CNN baseline training."""

    output_dir: str = "checkpoints/cnn_baseline"

    num_epochs: int = 10
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    eval_steps: int = 100
    save_steps: int = 500
    early_stopping_patience: int = 3

    use_amp: bool = True
    amp_dtype: str = "bf16"
    seed: int = 42
    dataloader_num_workers: int = 4

    model_config: Dict[str, Any] = field(default_factory=dict)

    log_wandb: bool = False
    wandb_project: str = "pixelated-empathy"
    wandb_run_name: str = ""

    def __post_init__(self):
        if self.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
            logger.warning("BF16 not supported, falling back to FP16")
            self.amp_dtype = "fp16"


class EmotionDataset(Dataset):
    """
    Dataset for emotion classification training.

    Expects data in format:
        {
            "text": "The input text",
            "input_ids": [token ids],
            "vad": [valence, arousal, dominance],
            "emotion": 3  # emotion class index
        }
    """

    EMOTION_LABELS = [
        "neutral",
        "happiness",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "disgust",
        "calm",
    ]

    def __init__(
        self,
        data_path: Optional[str] = None,
        max_length: int = 128,
        tokenizer: Any = None,
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.data: List[Dict[str, Any]] = []

        if data_path and os.path.exists(data_path):
            self._load_data(data_path)
        else:
            logger.info("No data path provided, generating synthetic data for testing")
            self.data = self._generate_synthetic_data(1000)

    def _load_data(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.data.append(item)
        logger.info(f"Loaded {len(self.data)} samples from {path}")

    def _generate_synthetic_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic data for testing the training pipeline."""
        import random

        data = []
        for i in range(num_samples):
            seq_len = random.randint(10, 64)
            input_ids = [random.randint(1, 30000) for _ in range(seq_len)]

            vad = [
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 1),
            ]

            emotion = random.randint(0, 7)

            data.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * seq_len,
                    "vad": vad,
                    "emotion": emotion,
                }
            )
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        input_ids = item["input_ids"]
        attention_mask = item.get("attention_mask", [1] * len(input_ids))

        if len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids = input_ids + [0] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        else:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "vad": torch.tensor(item["vad"], dtype=torch.float),
            "emotion": torch.tensor(item["emotion"], dtype=torch.long),
        }


class VADLoss(nn.Module):
    """
    Combined loss for VAD regression and emotion classification.

    Uses:
        - MSE loss for VAD values
        - Cross-entropy loss for emotion classification
        - Optional consistency loss between VAD and emotion predictions
    """

    EMOTION_TO_VAD = {
        0: [0.0, 0.0, 0.0],
        1: [0.8, 0.6, 0.5],
        2: [-0.7, -0.4, -0.3],
        3: [-0.6, 0.8, 0.7],
        4: [-0.7, 0.7, -0.5],
        5: [0.5, 0.8, 0.3],
        6: [-0.6, 0.4, 0.4],
        7: [0.0, -0.6, 0.0],
    }

    def __init__(
        self,
        vad_weight: float = 1.0,
        emotion_weight: float = 1.0,
        consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.vad_weight = vad_weight
        self.emotion_weight = emotion_weight
        self.consistency_weight = consistency_weight

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        vad_means = torch.tensor(list(self.EMOTION_TO_VAD.values()))
        self.register_buffer("_emotion_vad_mapping", vad_means)

    def forward(
        self,
        vad_pred: torch.Tensor,
        emotion_logits: torch.Tensor,
        vad_target: torch.Tensor,
        emotion_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        vad_loss = self.mse_loss(vad_pred, vad_target)

        emotion_loss = self.ce_loss(emotion_logits, emotion_target)

        consistency_loss = torch.tensor(0.0, device=vad_pred.device)
        if self.consistency_weight > 0:
            mapping = self.get_buffer("_emotion_vad_mapping")
            assert isinstance(mapping, torch.Tensor)
            emotion_vad_target = mapping[emotion_target]
            consistency_loss = self.mse_loss(vad_pred, emotion_vad_target)

        total_loss = (
            self.vad_weight * vad_loss
            + self.emotion_weight * emotion_loss
            + self.consistency_weight * consistency_loss
        )

        return total_loss, {
            "vad_loss": vad_loss.item(),
            "emotion_loss": emotion_loss.item(),
            "consistency_loss": consistency_loss.item(),
        }


class Trainer:
    """Training orchestrator for CNN baseline model."""

    def __init__(
        self,
        model: CNNFeatureExtractor,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.warmup_steps,
            T_mult=2,
        )

        self.loss_fn = VADLoss()

        if config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            self.amp_dtype = (
                torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
            )
        else:
            self.scaler = None
            self.amp_dtype = torch.float32

        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.patience_counter = 0

        os.makedirs(config.output_dir, exist_ok=True)

        if config.log_wandb:
            try:
                import wandb

                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name
                    or f"cnn_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=vars(config),
                )
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, disabling logging")
                config.log_wandb = False
                self.wandb = None
        else:
            self.wandb = None

    def train(self) -> Dict[str, float]:
        """Run the full training loop."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True,
        )

        if self.eval_dataset:
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=True,
            )
        else:
            eval_loader = None

        logger.info(f"Starting training on {self.device}")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            logger.info(f"Evaluation samples: {len(self.eval_dataset)}")

        for epoch in range(self.config.num_epochs):
            train_metrics = self._train_epoch(train_loader, epoch)

            if eval_loader and (epoch + 1) % 1 == 0:
                eval_metrics = self._evaluate(eval_loader, epoch)

                if eval_metrics["loss"] < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics["loss"]
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        self._save_checkpoint(self.config.num_epochs - 1, is_best=False, is_final=True)

        return {"best_eval_loss": self.best_eval_loss}

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {
            "vad_loss": 0.0,
            "emotion_loss": 0.0,
            "consistency_loss": 0.0,
        }

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            vad_target = batch["vad"].to(self.device)
            emotion_target = batch["emotion"].to(self.device)

            with torch.cuda.amp.autocast(
                enabled=self.config.use_amp, dtype=self.amp_dtype
            ):
                outputs = self.model(input_ids, attention_mask)
                vad_pred = outputs["vad"]
                emotion_logits = outputs["emotion_logits"]

                loss, components = self.loss_fn(
                    vad_pred, emotion_logits, vad_target, emotion_target
                )

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                loss.backward()
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

            total_loss += loss.item()
            for k, v in components.items():
                loss_components[k] += v

            self.global_step += 1

            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint(epoch, step=step)

        num_steps = len(dataloader)
        metrics = {
            "loss": total_loss / num_steps,
            **{k: v / num_steps for k, v in loss_components.items()},
        }

        logger.info(
            f"Epoch {epoch + 1}/{self.config.num_epochs} - "
            f"Loss: {metrics['loss']:.4f} - "
            f"VAD: {metrics['vad_loss']:.4f} - "
            f"Emotion: {metrics['emotion_loss']:.4f}"
        )

        if self.wandb:
            self.wandb.log({f"train/{k}": v for k, v in metrics.items()})

        return metrics

    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        loss_components = {
            "vad_loss": 0.0,
            "emotion_loss": 0.0,
            "consistency_loss": 0.0,
        }
        correct_emotions = 0
        total_samples = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            vad_target = batch["vad"].to(self.device)
            emotion_target = batch["emotion"].to(self.device)

            with torch.cuda.amp.autocast(
                enabled=self.config.use_amp, dtype=self.amp_dtype
            ):
                outputs = self.model(input_ids, attention_mask)
                vad_pred = outputs["vad"]
                emotion_logits = outputs["emotion_logits"]

                loss, components = self.loss_fn(
                    vad_pred, emotion_logits, vad_target, emotion_target
                )

            total_loss += loss.item()
            for k, v in components.items():
                loss_components[k] += v

            emotion_pred = emotion_logits.argmax(dim=-1)
            correct_emotions += (emotion_pred == emotion_target).sum().item()
            total_samples += emotion_target.size(0)

        num_steps = len(dataloader)
        metrics = {
            "loss": total_loss / num_steps,
            **{k: v / num_steps for k, v in loss_components.items()},
            "emotion_accuracy": correct_emotions / total_samples,
        }

        logger.info(
            f"Eval - Loss: {metrics['loss']:.4f} - "
            f"VAD: {metrics['vad_loss']:.4f} - "
            f"Emotion Acc: {metrics['emotion_accuracy']:.2%}"
        )

        if self.wandb:
            self.wandb.log({f"eval/{k}": v for k, v in metrics.items()})

        return metrics

    def _save_checkpoint(
        self,
        epoch: int,
        step: Optional[int] = None,
        is_best: bool = False,
        is_final: bool = False,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": vars(self.config),
            "best_eval_loss": self.best_eval_loss,
        }

        if is_final:
            path = os.path.join(self.config.output_dir, "final_model.pt")
        elif is_best:
            path = os.path.join(self.config.output_dir, "best_model.pt")
        else:
            step_str = f"_step{step}" if step else ""
            path = os.path.join(
                self.config.output_dir, f"checkpoint_epoch{epoch}{step_str}.pt"
            )

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")


def load_config(config_path: str) -> TrainingConfig:
    """Load training configuration from YAML or JSON file."""
    import yaml

    with open(config_path, "r") as f:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            config_dict = yaml.safe_load(f)
        else:
            config_dict = json.load(f)

    return TrainingConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN Feature Extraction Baseline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to training data (JSONL format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/cnn_baseline",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.config:
        config = load_config(args.config)
    else:
        config = TrainingConfig(
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )

    logger.info("Creating model...")
    model_config = config.model_config if config.model_config else {}
    model = create_cnn_baseline(**model_config)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    logger.info("Loading datasets...")
    full_dataset = EmotionDataset(data_path=args.data_path)

    train_size = int(0.9 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(
        full_dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    trainer = Trainer(model, config, train_dataset, eval_dataset)

    logger.info("Starting training...")
    results = trainer.train()

    logger.info(f"Training complete. Best eval loss: {results['best_eval_loss']:.4f}")


if __name__ == "__main__":
    main()
