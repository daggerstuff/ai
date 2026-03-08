"""
Defense Mechanism Training Orchestrator (DEPRECATED)

NOTE: This script is for training local DeBERTa-v3 models.
The project has transitioned to using NVIDIA NIM (Remote Inference)
as implemented in NIMEmbeddingClassifier.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ai.training.defense_mechanisms import DEFENSE_LABELS
from ai.training.defense_mechanisms.dataset import (
    compute_class_weights,
    create_fold_datasets,
    load_psydefconv,
)
from ai.training.defense_mechanisms.model import DefenseClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class DefenseTrainer:
    """
    Training orchestrator for defense mechanism classification.

    Supports cross-validated training with per-fold checkpointing,
    focal loss, R-Drop regularization, and mixed precision.
    """

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.device = self._select_device()

        self.model_name = self.config.get("base_model", "microsoft/deberta-v3-base")
        self.num_labels = self.config.get("num_labels", 9)
        self.max_length = self.config.get("max_length", 512)
        self.max_turns = self.config.get("max_turns", 40)

        train_params = self.config.get("training_parameters", {})
        self.num_epochs = train_params.get("num_train_epochs", 6)
        self.batch_size = train_params.get("per_device_train_batch_size", 8)
        self.eval_batch_size = train_params.get("per_device_eval_batch_size", 16)
        self.grad_accum = train_params.get("gradient_accumulation_steps", 1)
        self.lr = train_params.get("learning_rate", 2e-5)
        self.warmup_ratio = train_params.get("warmup_ratio", 0.1)
        self.weight_decay = train_params.get("weight_decay", 0.01)
        self.use_fp16 = train_params.get("fp16", False) and torch.cuda.is_available()
        self.output_dir = Path(
            train_params.get("output_dir", "./outputs/defense_deberta")
        )
        self.max_steps = train_params.get("max_steps", -1)

        loss_config = self.config.get("loss", {})
        self.focal_gamma = loss_config.get("gamma", 2.0)
        self.label_smoothing = loss_config.get("label_smoothing", 0.05)

        r_drop_config = self.config.get("r_drop", {})
        self.r_drop_enabled = r_drop_config.get("enabled", True)
        self.r_drop_lambda = r_drop_config.get("lambda", 0.5)

        cv_config = self.config.get("cross_validation", {})
        self.num_folds = cv_config.get("num_folds", 5)
        self.group_field = cv_config.get("group_field", "dialogue_id")

        logger.info(
            "DefenseTrainer initialized: model=%s, device=%s",
            self.model_name,
            self.device,
        )

    def _load_config(self, path: str) -> dict:
        """Load and validate training config."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        logger.info("Loaded config from %s", config_path)
        return config

    def _select_device(self) -> torch.device:
        """Select best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple MPS")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device

    def train_fold(
        self,
        fold_index: int,
        samples: list,
        tokenizer: AutoTokenizer,
    ) -> dict:
        """
        Train a single cross-validation fold.

        Args:
            fold_index: Which fold to use as validation
            samples: All labeled samples
            tokenizer: Tokenizer instance

        Returns:
            Dict with fold metrics (macro_f1, weighted_f1, per_class)
        """
        logger.info("=" * 60)
        logger.info("Starting fold %d/%d", fold_index + 1, self.num_folds)
        logger.info("=" * 60)

        train_ds, val_ds = create_fold_datasets(
            samples=samples,
            tokenizer=tokenizer,
            num_folds=self.num_folds,
            fold_index=fold_index,
            max_length=self.max_length,
            max_turns=self.max_turns,
        )

        class_weights = compute_class_weights(train_ds.get_labels())
        logger.info("Class weights: %s", class_weights.tolist())

        model = DefenseClassifier(
            model_name=self.model_name,
            num_labels=self.num_labels,
            class_weights=class_weights,
            focal_gamma=self.focal_gamma,
            label_smoothing=self.label_smoothing,
            r_drop_lambda=self.r_drop_lambda,
            r_drop_enabled=self.r_drop_enabled,
        ).to(self.device)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Optimizer with weight decay
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        total_steps = len(train_loader) // self.grad_accum * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            eps=1e-8,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        scaler = torch.amp.GradScaler("cuda") if self.use_fp16 else None

        best_macro_f1 = 0.0
        fold_dir = self.output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        global_step = 0

        for epoch in range(self.num_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_steps = 0

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                if self.use_fp16:
                    with torch.amp.autocast("cuda"):
                        outputs = model(input_ids, attention_mask, labels)
                        loss = outputs["loss"] / self.grad_accum
                    scaler.scale(loss).backward()
                else:
                    outputs = model(input_ids, attention_mask, labels)
                    loss = outputs["loss"] / self.grad_accum
                    loss.backward()

                epoch_loss += loss.item() * self.grad_accum

                if (step + 1) % self.grad_accum == 0:
                    if self.use_fp16:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % 50 == 0:
                        avg_loss = epoch_loss / (epoch_steps + 1)
                        lr_current = scheduler.get_last_lr()[0]
                        logger.info(
                            "Fold %d | Epoch %d | Step %d | Loss: %.4f | LR: %.2e",
                            fold_index + 1,
                            epoch + 1,
                            global_step,
                            avg_loss,
                            lr_current,
                        )

                epoch_steps += 1

                if 0 < self.max_steps <= global_step:
                    logger.info(
                        "Reached max_steps=%d, stopping training",
                        self.max_steps,
                    )
                    break

            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            logger.info(
                "Fold %d | Epoch %d complete | Avg loss: %.4f",
                fold_index + 1,
                epoch + 1,
                avg_epoch_loss,
            )

            # Evaluate
            metrics = self._evaluate(model, val_loader)
            logger.info(
                "Fold %d | Epoch %d | Val macro-F1: %.4f | Val weighted-F1: %.4f",
                fold_index + 1,
                epoch + 1,
                metrics["macro_f1"],
                metrics["weighted_f1"],
            )

            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                checkpoint_path = fold_dir / "best_model.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": self.config,
                        "fold_index": fold_index,
                        "epoch": epoch + 1,
                        "macro_f1": best_macro_f1,
                        "metrics": metrics,
                    },
                    checkpoint_path,
                )
                logger.info(
                    "New best model saved: macro-F1=%.4f → %s",
                    best_macro_f1,
                    checkpoint_path,
                )

            if 0 < self.max_steps <= global_step:
                break

        logger.info(
            "Fold %d complete | Best macro-F1: %.4f",
            fold_index + 1,
            best_macro_f1,
        )

        return {"best_macro_f1": best_macro_f1, **metrics}

    @torch.no_grad()
    def _evaluate(
        self,
        model: DefenseClassifier,
        val_loader: DataLoader,
    ) -> dict:
        """Evaluate model on validation set."""
        model.eval()
        all_preds = []
        all_labels = []

        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask)
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        label_names = [DEFENSE_LABELS.get(i, str(i)) for i in range(self.num_labels)]
        report = classification_report(
            all_labels,
            all_preds,
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        )

        return {
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "classification_report": report,
        }

    def train(
        self,
        data_path: str = "training/defense_mechanisms/data/train.json",
        fold_indices: list[int] | None = None,
    ):
        """
        Run the full training pipeline.

        Args:
            data_path: Path to PSYDEFCONV train.json
            fold_indices: Which folds to train. None = all folds
        """
        start_time = time.time()

        samples = load_psydefconv(data_path, has_labels=True)
        if not samples:
            raise ValueError(f"No samples loaded from {data_path}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if fold_indices is None:
            fold_indices = list(range(self.num_folds))

        all_fold_metrics = []
        for fold_idx in fold_indices:
            fold_metrics = self.train_fold(fold_idx, samples, tokenizer)
            all_fold_metrics.append(fold_metrics)

        # Summary
        avg_macro_f1 = np.mean([m["best_macro_f1"] for m in all_fold_metrics])
        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info("Folds trained: %s", fold_indices)
        logger.info(
            "Average best macro-F1: %.4f (±%.4f)",
            avg_macro_f1,
            np.std([m["best_macro_f1"] for m in all_fold_metrics]),
        )
        logger.info("Total time: %.1f seconds", elapsed)
        logger.info("Checkpoints saved to: %s", self.output_dir)

        # Save summary
        summary_path = self.output_dir / "training_summary.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": self.model_name,
                    "num_folds": self.num_folds,
                    "folds_trained": fold_indices,
                    "avg_macro_f1": float(avg_macro_f1),
                    "fold_metrics": [
                        {
                            "fold": fold_indices[i],
                            "best_macro_f1": float(m["best_macro_f1"]),
                        }
                        for i, m in enumerate(all_fold_metrics)
                    ],
                    "elapsed_seconds": elapsed,
                    "config": self.config,
                },
                f,
                indent=2,
            )
        logger.info("Training summary saved to %s", summary_path)


def main():
    parser = argparse.ArgumentParser(description="Train defense mechanism classifier")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON training config",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="training/defense_mechanisms/data/train.json",
        help="Path to PSYDEFCONV train.json",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="Train a single fold (0-indexed). Default: all folds",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum training steps (for dry-run). -1 = no limit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Override max_steps to 5 for quick validation",
    )
    args = parser.parse_args()

    trainer = DefenseTrainer(args.config)

    if args.dry_run:
        trainer.max_steps = 5
        logger.info("DRY RUN: max_steps overridden to 5")
    elif args.max_steps > 0:
        trainer.max_steps = args.max_steps

    fold_indices = None
    if args.fold_index is not None:
        fold_indices = [args.fold_index]

    trainer.train(data_path=args.data, fold_indices=fold_indices)


if __name__ == "__main__":
    main()
