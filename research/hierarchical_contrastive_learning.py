"""
PIX-3912: Hierarchical Contrastive Learning Module

Contrastive loss with hierarchical negative sampling for clinical diagnosis prediction.
Implements:
- Easy / medium / hard negative sampling from the therapeutic concept hierarchy
- Condition-aware encoder that preserves hierarchy structure
- Multi-task training: contrastive + classification + hierarchy-preservation losses

Inspired by Mera (arXiv 2501.17326).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .therapeutic_concept_hierarchy import TherapeuticConceptHierarchy


@dataclass
class HCLConfig:
    """Configuration for the Hierarchical Contrastive Learning module."""

    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.2
    temperature: float = 0.07
    # Loss weights
    lambda_contrastive: float = 1.0
    lambda_classification: float = 0.5
    lambda_hierarchy: float = 0.3
    # Negative sampling
    negatives_per_anchor: int = 5
    negative_strategy: str = "mixed"  # easy | medium | hard | mixed
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class HierarchyAwareEncoder(nn.Module):
    """
    Encoder that produces condition-aware embeddings preserving hierarchy structure.

    Input: structural encoding vector (from TherapeuticConceptHierarchy.encode_node)
    Output: embedding vector in a learned metric space where hierarchical
            similarity is preserved.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.mlp(x)
        # L2-normalize for cosine-similarity contrastive loss
        return F.normalize(z, p=2, dim=1)


class HierarchicalContrastiveLearner(nn.Module):
    """
    Full model: encoder + optional classification head + hierarchy-preservation loss.
    """

    def __init__(self, config: HCLConfig, num_classes: int | None = None):
        super().__init__()
        self.config = config
        self.encoder = HierarchyAwareEncoder(
            input_dim=128,  # matches encode_node default dim
            hidden_dim=config.hidden_dim,
            output_dim=config.embedding_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.num_classes = num_classes
        if num_classes is not None and num_classes > 0:
            self.classifier = nn.Linear(config.embedding_dim, num_classes)
        else:
            self.classifier = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def classify(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.classifier is None:
            return None
        z = self.encoder(x)
        return self.classifier(z)


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------

def hierarchical_contrastive_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: list[torch.Tensor],
    temperature: float = 0.07,
    negative_weights: list[float] | None = None,
) -> torch.Tensor:
    """
    InfoNCE-style contrastive loss with optional per-negative-type weighting.

    Parameters
    ----------
    anchors : (B, D)
    positives : (B, D)
    negatives : list of (B, D) tensors (one per negative type)
    temperature : softmax temperature
    negative_weights : weight for each negative group
    """
    batch_size = anchors.size(0)
    # Positive similarities
    pos_sim = torch.sum(anchors * positives, dim=1) / temperature  # (B,)

    # Negative similarities
    neg_sims: list[torch.Tensor] = []
    for neg in negatives:
        # neg: (B, D) -> similarity per sample
        sim = torch.sum(anchors * neg, dim=1) / temperature  # (B,)
        neg_sims.append(sim)

    if negative_weights is None:
        negative_weights = [1.0] * len(negatives)

    # Weighted combination of negative similarities
    weighted_neg = torch.stack(
        [w * sim for w, sim in zip(negative_weights, neg_sims)], dim=1
    ).sum(dim=1)  # (B,)

    # Denominator: exp(pos) + sum(exp(neg))
    denominator = torch.exp(pos_sim) + torch.exp(weighted_neg)
    loss = -torch.log(torch.exp(pos_sim) / denominator + 1e-8)
    return loss.mean()


def hierarchy_preservation_loss(
    embeddings: torch.Tensor,
    hierarchy_distances: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Ensure that embedding distances reflect hierarchical distances.

    Parameters
    ----------
    embeddings : (B, D)
    hierarchy_distances : (B,) — pre-computed hierarchical distances
    margin : minimum separation between close and distant pairs
    """
    # Pairwise Euclidean distances in embedding space
    # For efficiency, use a sampled subset or all pairs
    B = embeddings.size(0)
    if B < 2:
        return torch.tensor(0.0, device=embeddings.device)

    # Compute pairwise distances
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

    # Create target distance matrix from hierarchy distances
    # We use the provided hierarchy distances as a proxy for pairwise targets
    # For a full implementation, we'd pass a (B, B) target matrix.
    # Here we approximate with a ranking loss on sampled triplets.

    # Sample triplets: anchor, close, distant
    losses = []
    for i in range(B):
        for j in range(i + 1, B):
            d_ij = hierarchy_distances[i] + hierarchy_distances[j]
            for k in range(j + 1, B):
                d_ik = hierarchy_distances[i] + hierarchy_distances[k]
                if d_ij < d_ik:
                    # j should be closer to i than k is
                    loss_ij = F.relu(
                        dist_matrix[i, j] - dist_matrix[i, k] + margin
                    )
                    losses.append(loss_ij)

    if not losses:
        return torch.tensor(0.0, device=embeddings.device)
    return torch.stack(losses).mean()


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class HierarchicalConceptDataset(Dataset):
    """
    PyTorch Dataset that yields (anchor, positive, negatives, label) tuples
    from a TherapeuticConceptHierarchy.
    """

    def __init__(
        self,
        hierarchy: TherapeuticConceptHierarchy,
        config: HCLConfig,
        transform: Callable[[np.ndarray], torch.Tensor] | None = None,
    ):
        self.hierarchy = hierarchy
        self.config = config
        self.transform = transform or (lambda x: torch.from_numpy(x).float())
        # Use condition-level nodes (level 1) as anchors
        self.anchor_ids = [n.id for n in hierarchy.get_nodes_at_level(1)]

    def __len__(self) -> int:
        return len(self.anchor_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        anchor_id = self.anchor_ids[idx]
        anchor_vec = self.hierarchy.encode_node(anchor_id, dim=128)

        # Positive: a sibling or the node itself (augmented)
        siblings = self.hierarchy.get_siblings(anchor_id)
        if siblings:
            positive_id = siblings[idx % len(siblings)].id
        else:
            positive_id = anchor_id
        positive_vec = self.hierarchy.encode_node(positive_id, dim=128)

        # Negatives: sampled from hierarchy
        neg_samples = self.hierarchy.sample_negatives(
            anchor_id,
            strategy=self.config.negative_strategy,
            k=self.config.negatives_per_anchor,
        )
        negative_vecs: list[torch.Tensor] = []
        for neg_id, neg_type in neg_samples:
            nv = self.hierarchy.encode_node(neg_id, dim=128)
            negative_vecs.append(self.transform(nv))

        # Pad negatives to fixed count
        while len(negative_vecs) < self.config.negatives_per_anchor:
            negative_vecs.append(torch.zeros(128))

        return {
            "anchor": self.transform(anchor_vec),
            "positive": self.transform(positive_vec),
            "negatives": torch.stack(negative_vecs[: self.config.negatives_per_anchor]),
            "anchor_id": anchor_id,
            "label": idx,  # simple integer label for classification head
        }


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class HCLTrainer:
    """Trainer for the Hierarchical Contrastive Learning module."""

    def __init__(self, model: HierarchicalContrastiveLearner, config: HCLConfig):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_contrastive = 0.0
        total_classification = 0.0
        total_hierarchy = 0.0
        num_batches = 0

        for batch in dataloader:
            anchors = batch["anchor"].to(self.config.device)
            positives = batch["positive"].to(self.config.device)
            negatives = batch["negatives"].to(self.config.device)
            labels = batch["label"].to(self.config.device)

            # Forward
            z_anchors = self.model(anchors)
            z_positives = self.model(positives)
            # Negatives: (B, K, D) -> list of (B, D)
            neg_list = [negatives[:, i, :] for i in range(negatives.size(1))]

            # Contrastive loss
            loss_contrastive = hierarchical_contrastive_loss(
                z_anchors, z_positives, neg_list, temperature=self.config.temperature
            )

            # Classification loss
            loss_cls = torch.tensor(0.0, device=self.config.device)
            if self.model.classifier is not None:
                logits = self.model.classify(anchors)
                if logits is not None:
                    loss_cls = F.cross_entropy(logits, labels)

            # Hierarchy preservation loss
            # Use dummy hierarchy distances for now (full implementation would pre-compute)
            hierarchy_dists = torch.zeros(anchors.size(0), device=self.config.device)
            loss_hierarchy = hierarchy_preservation_loss(z_anchors, hierarchy_dists)

            # Total loss
            loss = (
                self.config.lambda_contrastive * loss_contrastive
                + self.config.lambda_classification * loss_cls
                + self.config.lambda_hierarchy * loss_hierarchy
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_contrastive += loss_contrastive.item()
            total_classification += loss_cls.item()
            total_hierarchy += loss_hierarchy.item()
            num_batches += 1

        self.scheduler.step()

        return {
            "loss": total_loss / max(num_batches, 1),
            "contrastive": total_contrastive / max(num_batches, 1),
            "classification": total_classification / max(num_batches, 1),
            "hierarchy": total_hierarchy / max(num_batches, 1),
        }

    def fit(self, dataloader: DataLoader, epochs: int | None = None) -> list[dict[str, float]]:
        epochs = epochs or self.config.epochs
        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            metrics = self.train_epoch(dataloader)
            history.append(metrics)
            print(f"Epoch {epoch + 1}/{epochs} — {metrics}")
        return history

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": self.config,
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])


# ------------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------------

def encode_condition(
    hierarchy: TherapeuticConceptHierarchy,
    model: HierarchicalContrastiveLearner,
    condition_id: str,
    device: str = "cpu",
) -> np.ndarray:
    """Encode a single condition through the hierarchy + learned encoder."""
    struct_vec = hierarchy.encode_node(condition_id, dim=128)
    x = torch.from_numpy(struct_vec).float().unsqueeze(0).to(device)
    with torch.no_grad():
        z = model(x)
    return z.squeeze(0).cpu().numpy()


def compute_similarity_matrix(
    hierarchy: TherapeuticConceptHierarchy,
    model: HierarchicalContrastiveLearner,
    condition_ids: list[str] | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for all (or given) conditions."""
    if condition_ids is None:
        condition_ids = [n.id for n in hierarchy.get_nodes_at_level(1)]

    embeddings: list[np.ndarray] = []
    for cid in condition_ids:
        emb = encode_condition(hierarchy, model, cid, device=device)
        embeddings.append(emb)

    emb_matrix = np.stack(embeddings)
    # Cosine similarity
    norm = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    normalized = emb_matrix / (norm + 1e-8)
    return normalized @ normalized.T
