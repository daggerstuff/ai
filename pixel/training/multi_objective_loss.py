"""
MultiObjectiveLoss: Combines language modeling, EQ, persona, clinical accuracy, and empathy losses for Pixel training.
"""

from typing import Any

import torch
from torch import nn


class MultiObjectiveLoss(nn.Module):
    """
    Combines multiple loss components for Pixel model training.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        super().__init__()
        # Default weights for each objective
        self.weights = weights or {
            "language": 1.0,
            "eq": 1.0,
            "persona": 1.0,
            "clinical": 1.0,
            "empathy": 1.0,
        }
        # Placeholders for individual loss functions
        self.language_loss_fn = nn.CrossEntropyLoss()
        self.eq_loss_fn = nn.MSELoss()
        self.persona_loss_fn = nn.CrossEntropyLoss()
        self.clinical_loss_fn = nn.BCEWithLogitsLoss()
        self.empathy_loss_fn = nn.MSELoss()

    def forward(self, outputs: dict[str, Any], targets: dict[str, Any]) -> torch.Tensor:
        """
        Compute the weighted sum of all objective losses.
        Args:
            outputs: Model outputs (dict)
            targets: Target values (dict)
        Returns:
            Weighted total loss (scalar tensor)
        """
        # Compute individual losses (placeholders)
        language_loss = self.language_loss_fn(outputs["language"], targets["language"])
        eq_loss = self.eq_loss_fn(outputs["eq"], targets["eq"])
        persona_loss = self.persona_loss_fn(outputs["persona"], targets["persona"])
        clinical_loss = self.clinical_loss_fn(outputs["clinical"], targets["clinical"])
        empathy_loss = self.empathy_loss_fn(outputs["empathy"], targets["empathy"])

        losses = [
            self.weights["language"] * language_loss,
            self.weights["eq"] * eq_loss,
            self.weights["persona"] * persona_loss,
            self.weights["clinical"] * clinical_loss,
            self.weights["empathy"] * empathy_loss,
        ]
        return sum(losses, start=torch.tensor(0.0, device=language_loss.device))
