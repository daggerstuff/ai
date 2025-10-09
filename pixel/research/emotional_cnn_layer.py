"""
emotional_cnn_layer.py

Production-grade CNN Emotional Pattern Detection for Advanced Emotional Intelligence (7.1)
----------------------------------------------------------------------------------------
Implements a convolutional neural network for detecting emotional patterns in textual data.
Supports multi-scale feature extraction and integration with Pixel's emotion processing pipeline.

Author: Roo (AI)
Date: 2025-07-17
"""

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as f


@dataclass
class EmotionalCNNConfig:
    embedding_dim: int = 128
    num_filters: int = 64
    kernel_sizes: list[int] | None = field(default_factory=lambda: [2, 3, 4, 5])
    output_dim: int = 32
    hidden_dim: int = 32  # Only used for classifier
    dropout: float = 0.2
    padding_idx: int = 0


class EmotionalCNNTextEncoder(nn.Module):
    """
    CNN-based encoder for extracting emotional features from text sequences.
    Supports multi-scale convolutional filters for nuanced emotional cue detection.
    """

    def __init__(
        self,
        vocab_size: int,
        config: EmotionalCNNConfig,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, config.embedding_dim, padding_idx=config.padding_idx
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=config.embedding_dim,
                    out_channels=config.num_filters,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in (config.kernel_sizes or [2, 3, 4, 5])
            ]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(
            config.num_filters * len(config.kernel_sizes or [2, 3, 4, 5]), config.output_dim
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with token indices.
        Returns:
            Tensor of shape (batch_size, output_dim) with extracted emotional features.
        """
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        conv_outputs = [
            f.relu(conv(embedded)) for conv in self.convs
        ]  # List of (batch_size, num_filters, seq_len)
        pooled = [
            f.max_pool1d(out, kernel_size=out.shape[2]).squeeze(2) for out in conv_outputs
        ]  # List of (batch_size, num_filters)
        cat = torch.cat(pooled, dim=1)  # (batch_size, num_filters * num_kernels)
        features = self.dropout(cat)
        return self.fc(features)  # (batch_size, output_dim)


class EmotionalCNNClassifier(nn.Module):
    """
    CNN-based classifier for emotion detection in text.
    Outputs emotion logits for each input sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        num_emotions: int,
        config: EmotionalCNNConfig,
    ):
        super().__init__()
        self.encoder = EmotionalCNNTextEncoder(
            vocab_size=vocab_size,
            config=config,
        )
        self.classifier = nn.Linear(config.hidden_dim, num_emotions)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with token indices.
        Returns:
            Logits of shape (batch_size, num_emotions).
        """
        features = self.encoder(input_ids)
        return self.classifier(features)


# Example integration (for documentation/testing, not for production import)
if __name__ == "__main__":
    vocab_size = 1000
    num_emotions = 8
    batch_size = 2
    seq_len = 20
    config = EmotionalCNNConfig()
    # Set output_dim and hidden_dim to the same value for encoder/classifier compatibility
    config.output_dim = config.hidden_dim
    model = EmotionalCNNClassifier(vocab_size, num_emotions, config)
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(dummy_input)
