#!/usr/bin/env python3
"""
CNN Feature Extraction Baseline for Emotional Text Analysis.

Implements lightweight convolutional neural network layers for multi-scale
emotional feature extraction from text tokens. Designed as a faster alternative
to transformer-based approaches for real-time inference (<10ms target).

Key Features:
  - Multi-scale convolutional kernels (1-gram to 5-gram patterns)
  - Micro-rhythm detection for emotional cadence analysis
  - Valence-Arousal-Dominance (VAD) output projection
  - Hierarchical feature aggregation

Example:
    >>> config = CNNFeatureConfig()
    >>> model = CNNFeatureExtractor(config)
    >>> input_ids = torch.randint(0, 30522, (2, 128))  # Batch of 2, seq len 128
    >>> features = model(input_ids)
    >>> print(features['vad'].shape)  # [2, 3] - valence, arousal, dominance
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CNNFeatureConfig:
    """Configuration for CNN feature extraction model."""

    vocab_size: int = 30522
    hidden_size: int = 256
    embedding_dim: int = 128
    num_filters: int = 64
    kernel_sizes: Tuple[int, ...] = (1, 2, 3, 4, 5)
    num_rhythm_layers: int = 3
    dropout: float = 0.1
    output_dim: int = 3
    max_position_embeddings: int = 512

    enable_micro_rhythm: bool = True
    rhythm_hidden_dim: int = 32
    rhythm_num_layers: int = 2

    num_emotion_classes: int = 8

    pad_token_id: int = 0

    def __post_init__(self):
        if len(self.kernel_sizes) == 0:
            raise ValueError("kernel_sizes must not be empty")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence order awareness."""

    def __init__(self, embedding_dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe: torch.Tensor = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("_pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe_buffer = self.get_buffer("_pe")
        assert isinstance(pe_buffer, torch.Tensor)
        x = x + pe_buffer[:, : x.size(1), :]
        return self.dropout(x)


class MultiScaleConvLayer(nn.Module):
    """
    Parallel convolutions at multiple kernel sizes for n-gram feature extraction.

    Each kernel size captures different granularities:
      - kernel_size=1: Unigram patterns (individual word sentiment)
      - kernel_size=2: Bigram patterns (word-pair relationships)
      - kernel_size=3: Trigram patterns (short phrases)
      - kernel_size=4-5: Longer phrase patterns
    """

    def __init__(
        self,
        embedding_dim: int,
        num_filters: int,
        kernel_sizes: Tuple[int, ...],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList(
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=(k - 1) // 2,
            )
            for k in kernel_sizes
        )

        self.batch_norms = nn.ModuleList(
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        )

        self.dropout = nn.Dropout(dropout)

        self.output_dim = num_filters * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)

        conv_outputs = []
        target_len = x.size(2)

        for conv, bn in zip(self.convs, self.batch_norms):
            conv_out = conv(x)
            conv_out = bn(conv_out)
            conv_out = F.gelu(conv_out)

            if conv_out.size(2) != target_len:
                diff = target_len - conv_out.size(2)
                conv_out = F.pad(conv_out, (0, diff))

            conv_outputs.append(conv_out)

        concatenated = torch.cat(conv_outputs, dim=1)
        concatenated = self.dropout(concatenated)

        return concatenated.transpose(1, 2)


class MicroRhythmLayer(nn.Module):
    """
    Detects textual micro-rhythms - patterns in emotional intensity and cadence.

    Analyzes the "rhythm" of emotional expression through:
      - Temporal convolution for local pattern detection
      - Dilated convolution for longer-range dependencies
      - Gating mechanism for dynamic feature selection

    This captures phenomena like:
      - Emotional escalation patterns
      - Rhetorical repetition
      - Punctuation and pause rhythms
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.temporal_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
        )

        self.dilated_convs = nn.ModuleList(
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=2**i,
                dilation=2**i,
            )
            for i in range(num_layers)
        )

        self.gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.output_projection = nn.Linear(input_dim + hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_conv = x.transpose(1, 2)

        x_conv = F.gelu(self.temporal_conv(x_conv))

        for dilated_conv in self.dilated_convs:
            x_conv = x_conv + F.gelu(dilated_conv(x_conv))

        x_conv = x_conv.transpose(1, 2)

        gate_input = torch.cat([residual, x_conv], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))

        x_conv = gate_values * x_conv
        x_conv = self.layer_norm(x_conv)
        x_conv = self.dropout(x_conv)

        combined = torch.cat([residual, x_conv], dim=-1)
        output = self.output_projection(combined)

        return output + residual


class HierarchicalAggregator(nn.Module):
    """
    Aggregates features across hierarchy levels for comprehensive representation.

    Combines:
      - Token-level features (fine-grained emotion signals)
      - Phrase-level features (n-gram patterns)
      - Sequence-level features (overall emotional trajectory)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        self.token_attention = nn.Linear(input_dim, 1)

        self.phrase_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        self.sequence_pool = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        self.output_layer = nn.Linear(input_dim * 3, input_dim)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        token_weights = F.softmax(self.token_attention(x), dim=1)
        token_features = (x * token_weights).sum(dim=1)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        phrase_features, _ = self.phrase_attention(
            x, x, x, key_padding_mask=key_padding_mask
        )
        phrase_features = phrase_features.mean(dim=1)

        sequence_features = self.sequence_pool(x.mean(dim=1))

        combined = torch.cat(
            [token_features, phrase_features, sequence_features], dim=-1
        )
        return self.output_layer(combined)


class CNNFeatureExtractor(nn.Module):
    """
    Main CNN-based feature extraction model for emotional text analysis.

    Architecture:
      1. Token embedding with positional encoding
      2. Multi-scale convolutional layers (n-gram features)
      3. Micro-rhythm detection (optional)
      4. Hierarchical feature aggregation
      5. Output heads for VAD and emotion classification

    Designed for sub-10ms inference on CPU, sub-1ms on GPU.
    """

    def __init__(self, config: CNNFeatureConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_token_id,
        )

        self.positional_encoding = PositionalEncoding(
            embedding_dim=config.embedding_dim,
            max_len=config.max_position_embeddings,
            dropout=config.dropout,
        )

        self.multi_scale_conv = MultiScaleConvLayer(
            embedding_dim=config.embedding_dim,
            num_filters=config.num_filters,
            kernel_sizes=config.kernel_sizes,
            dropout=config.dropout,
        )

        self.input_projection = nn.Linear(
            self.multi_scale_conv.output_dim, config.hidden_size
        )

        if config.enable_micro_rhythm:
            self.micro_rhythm = MicroRhythmLayer(
                input_dim=config.hidden_size,
                hidden_dim=config.rhythm_hidden_dim,
                num_layers=config.rhythm_num_layers,
                dropout=config.dropout,
            )
        else:
            self.micro_rhythm = None

        self.hierarchical_agg = HierarchicalAggregator(
            input_dim=config.hidden_size,
            hidden_dim=config.hidden_size // 2,
            dropout=config.dropout,
        )

        self.vad_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 3),
            nn.Tanh(),
        )

        self.emotion_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_emotion_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if self.config.pad_token_id is not None:
                    nn.init.zeros_(module.weight[self.config.pad_token_id])
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the CNN feature extractor.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Optional mask for padding [batch_size, seq_len]
            return_dict: Whether to return dict or tuple

        Returns:
            Dictionary containing:
                - vad: Valence-Arousal-Dominance scores [batch_size, 3]
                - emotion_logits: Emotion classification logits
                  [batch_size, num_classes]
                - hidden_states: Aggregated features [batch_size, hidden_size]
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).float()

        x = self.embedding(input_ids)
        x = self.positional_encoding(x)

        x = self.multi_scale_conv(x)

        x = self.input_projection(x)
        x = F.gelu(x)

        if self.micro_rhythm is not None:
            x = self.micro_rhythm(x)

        hidden_states = self.hierarchical_agg(x, attention_mask)

        vad = self.vad_head(hidden_states)
        emotion_logits = self.emotion_head(hidden_states)

        if return_dict:
            return {
                "vad": vad,
                "emotion_logits": emotion_logits,
                "hidden_states": hidden_states,
            }
        return vad, emotion_logits, hidden_states

    def get_feature_size(self) -> int:
        """Return the output feature dimension."""
        return self.config.hidden_size


def create_cnn_baseline(
    vocab_size: int = 30522,
    hidden_size: int = 256,
    num_filters: int = 64,
    enable_micro_rhythm: bool = True,
) -> CNNFeatureExtractor:
    """
    Factory function to create a CNN feature extractor with sensible defaults.

    Args:
        vocab_size: Vocabulary size for embedding layer
        hidden_size: Hidden dimension for feature representations
        num_filters: Number of filters per kernel size
        enable_micro_rhythm: Whether to include micro-rhythm detection

    Returns:
        Configured CNNFeatureExtractor instance
    """
    config = CNNFeatureConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_filters=num_filters,
        enable_micro_rhythm=enable_micro_rhythm,
    )
    return CNNFeatureExtractor(config)
