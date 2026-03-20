import math
from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

from training.defense_mechanisms.constants import (
    DEFENSE_LABELS,
    DEFENSE_MATURITY,
    NUM_LABELS,
)


@dataclass
class DefensePrediction:
    """Structured output for a defense mechanism prediction."""

    label: int
    label_name: str
    confidence: float
    probabilities: list[float]
    maturity_score: float | None
    raw_logits: list[float] = field(repr=False)


class FocalLoss(nn.Module):
    """
    Focal Loss with Label Smoothing for imbalanced classification.
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)

        if self.label_smoothing > 0:
            smooth = self.label_smoothing / num_classes
            one_hot = torch.zeros_like(log_probs).scatter(1, targets.unsqueeze(1), 1.0)
            one_hot = one_hot * (1.0 - self.label_smoothing) + smooth

            loss = -(one_hot * log_probs).sum(dim=-1)
            pt = (one_hot * probs).sum(dim=-1)
        else:
            loss = F.nll_loss(log_probs, targets, reduction="none")
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * loss

        if self.alpha is not None:
            at = self.alpha.to(inputs.device).gather(0, targets)
            loss = at * loss

        return loss.mean()


def compute_r_drop_loss(logits_1, logits_2, reduction="batchmean"):
    """
    Computes bidirectional KL divergence for R-Drop regularization.
    """
    p = F.log_softmax(logits_1, dim=-1)
    q = F.log_softmax(logits_2, dim=-1)
    kl_pq = F.kl_div(p, q.exp(), reduction=reduction)
    kl_qp = F.kl_div(q, p.exp(), reduction=reduction)
    return (kl_pq + kl_qp) / 2.0


class DefenseClassifier(nn.Module):
    """
    DeBERTa-based classifier for defense mechanism detection.
    Integrates Focal Loss and R-Drop regularization for robustness against dataset imbalance.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = NUM_LABELS,
        class_weights: torch.Tensor | None = None,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        r_drop_lambda: float = 0.5,
        r_drop_enabled: bool = True,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.r_drop_enabled = r_drop_enabled
        self.r_drop_lambda = r_drop_lambda

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,
        )

        self.criterion = FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass. If training + labels provided + r_drop_enabled, performs two
        forward passes to compute R-Drop loss in addition to focal classification loss.
        """
        outputs_1 = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits_1 = outputs_1.logits

        result = {"logits": logits_1}

        if labels is not None:
            cls_loss = self.criterion(logits_1, labels)

            if self.training and self.r_drop_enabled:
                outputs_2 = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits_2 = outputs_2.logits

                cls_loss_2 = self.criterion(logits_2, labels)
                cls_loss = (cls_loss + cls_loss_2) / 2.0
                r_drop_loss = compute_r_drop_loss(logits_1, logits_2)

                total_loss = cls_loss + (self.r_drop_lambda * r_drop_loss)
                result["loss"] = total_loss
                result["cls_loss"] = cls_loss
                result["kl_loss"] = r_drop_loss
            else:
                result["loss"] = cls_loss

        return result

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[DefensePrediction]:
        """
        Inference method returning structured prediction objects.
        """
        self.eval()
        outputs = self(input_ids, attention_mask)
        logits = outputs["logits"]
        probs = F.softmax(logits, dim=-1)

        batch_size = logits.size(0)
        predictions = []

        for i in range(batch_size):
            p = probs[i]
            l = logits[i]

            pred_label = int(torch.argmax(p).item())
            confidence = float(p[pred_label].item())

            predictions.append(
                DefensePrediction(
                    label=pred_label,
                    label_name=DEFENSE_LABELS.get(pred_label, "Unknown"),
                    confidence=confidence,
                    probabilities=p.tolist(),
                    maturity_score=DEFENSE_MATURITY.get(pred_label),
                    raw_logits=l.tolist(),
                )
            )

        return predictions
