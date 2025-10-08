from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ai.pixel.models.pixel_base_model import PixelBaseModel
from ai.pixel.training.multi_objective_loss import MultiObjectiveLoss


@dataclass
class TrainerConfig:
    val_loader: DataLoader | None = None
    device: str | None = None
    scheduler: Any | None = None
    max_epochs: int = 1
    on_epoch_end: Callable | None = None

    def get_device(self):
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")


class MultiObjectiveTrainer:
    """
    Trainer for Pixel model with multi-objective loss.
    Supports extensible hooks for dynamic loss scheduling, monitoring, and validation.
    """

    def __init__(
        self,
        model: PixelBaseModel,
        loss_fn: MultiObjectiveLoss,
        optimizer: Optimizer,
        train_loader: DataLoader,
        config: TrainerConfig | None = None,
    ):
        config = config or TrainerConfig()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = config.val_loader
        self.device = config.get_device()
        self.scheduler = config.scheduler
        self.max_epochs = config.max_epochs
        self.on_epoch_end = config.on_epoch_end

        self.model.to(self.device)

    def train(self):
        for epoch in range(self.max_epochs):
            self.model.train()
            total_loss = 0.0
            for batch in self.train_loader:
                outputs, targets = batch["outputs"], batch["targets"]
                outputs = {k: v.to(self.device) for k, v in outputs.items()}
                targets = {k: v.to(self.device) for k, v in targets.items()}

                self.optimizer.zero_grad()
                preds = self.model(**outputs)
                loss = self.loss_fn(preds, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            if self.scheduler and hasattr(self.scheduler, "step"):
                self.scheduler.step()
            if self.on_epoch_end:
                self.on_epoch_end(epoch, avg_loss)

            if self.val_loader:
                self.validate()

    def validate(self):
        if self.val_loader is None:
            return None
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                outputs, targets = batch["outputs"], batch["targets"]
                outputs = {k: v.to(self.device) for k, v in outputs.items()}
                targets = {k: v.to(self.device) for k, v in targets.items()}
                preds = self.model(**outputs)
                loss = self.loss_fn(preds, targets)
                total_loss += loss.item()
                count += 1
        if count > 0:
            return total_loss / count
        return None
