"""
Defense Mechanism Classifier Model

DeBERTa-based sequence classifier with Focal Loss and R-Drop
regularization for 9-class defense mechanism detection.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.defense_mechanisms.constants import DEFENSE_LABELS, DEFENSE_MATURITY

logger = logging.getLogger(__name__)

NUM_LABELS = 9


@dataclass
class DefensePrediction:
    """Prediction output for a single utterance."""

    label: int
    label_name: str
    confidence: float
    maturity_score: Optional[float]
    probabilities: list[float]


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification with class imbalance.

    Focal loss reduces the relative loss for well-classified examples,
    focusing training on hard, misclassified examples. This is critical
    for the PSYDEFCONV dataset where class 7 (High-Adaptive) represents
    52% of samples.

    Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Per-class weights tensor of shape (num_classes,)
        gamma: Focusing parameter. Default 2.0
        label_smoothing: Label smoothing factor. Default 0.0
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Model logits of shape (batch_size, num_classes)
            targets: Integer labels of shape (batch_size,)

        Returns:
            Scalar loss value
        """
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)

        # Apply label smoothing
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / num_classes
            one_hot = torch.zeros_like(log_probs).scatter(1, targets.unsqueeze(1), 1.0)
            one_hot = one_hot * (1.0 - self.label_smoothing) + smooth
            loss = -(one_hot * log_probs).sum(dim=-1)
            pt = (one_hot * probs).sum(dim=-1)
        else:
            loss = F.nll_loss(log_probs, targets, reduction="none")
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * loss

        # Class weight
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)
            at = alpha_t.gather(0, targets)
            loss = at * loss

        return loss.mean()


def compute_r_drop_loss(
    logits_1: torch.Tensor,
    logits_2: torch.Tensor,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """
    Compute R-Drop KL divergence regularization loss.

    R-Drop regularizes the model by minimizing the bidirectional
    KL divergence between outputs from two forward passes with
    different dropout masks.

    Args:
        logits_1: Logits from first forward pass
        logits_2: Logits from second forward pass
        reduction: Reduction method for KL divergence

    Returns:
        Mean bidirectional KL divergence
    """
    p = F.log_softmax(logits_1, dim=-1)
    q = F.log_softmax(logits_2, dim=-1)

    kl_pq = F.kl_div(p, q.exp(), reduction=reduction)
    kl_qp = F.kl_div(q, p.exp(), reduction=reduction)

    return (kl_pq + kl_qp) / 2.0


class DefenseClassifier(nn.Module):
    """
    DeBERTa-based defense mechanism classifier.

    Wraps a HuggingFace sequence classification model with
    Focal Loss and R-Drop support.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = NUM_LABELS,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        r_drop_lambda: float = 0.5,
        r_drop_enabled: bool = True,
    ):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_labels = num_labels
        self.r_drop_lambda = r_drop_lambda
        self.r_drop_enabled = r_drop_enabled

        self.criterion = FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )

        logger.info(
            "Initialized DefenseClassifier: model=%s, labels=%d, "
            "focal_gamma=%.1f, r_drop=%s (lambda=%.2f)",
            model_name,
            num_labels,
            focal_gamma,
            r_drop_enabled,
            r_drop_lambda,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass with optional R-Drop regularization.

        When labels are provided AND R-Drop is enabled, performs two
        forward passes with different dropout masks and adds the
        KL divergence penalty to the classification loss.

        Returns:
            Dict with 'loss' (scalar), 'logits' (batch_size, num_labels)
        """
        outputs_1 = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs_1.logits

        result = {"logits": logits}

        if labels is not None:
            cls_loss = self.criterion(logits, labels)

            if self.r_drop_enabled and self.training:
                outputs_2 = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits_2 = outputs_2.logits

                cls_loss_2 = self.criterion(logits_2, labels)
                cls_loss = (cls_loss + cls_loss_2) / 2.0

                r_drop_loss = compute_r_drop_loss(logits, logits_2)
                total_loss = cls_loss + self.r_drop_lambda * r_drop_loss
            else:
                total_loss = cls_loss

            result["loss"] = total_loss

        return result

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[DefensePrediction]:
        """
        Predict defense mechanisms for a batch of inputs.

        Returns a list of DefensePrediction objects with label,
        confidence, maturity score, and full probability distribution.
        """
        self.eval()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        probs = F.softmax(outputs.logits, dim=-1)

        predictions = []
        for i in range(probs.size(0)):
            prob_vec = probs[i].cpu().tolist()
            label = int(probs[i].argmax().item())
            confidence = float(prob_vec[label])

            maturity = DEFENSE_MATURITY.get(label)
            maturity_score: Optional[float] = None
            if maturity is not None:
                maturity_score = float(maturity)

            predictions.append(
                DefensePrediction(
                    label=label,
                    label_name=DEFENSE_LABELS.get(label, f"Unknown ({label})"),
                    confidence=confidence,
                    maturity_score=maturity_score,
                    probabilities=prob_vec,
                )
            )

        return predictions

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
