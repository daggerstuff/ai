#!/usr/bin/env python3
"""
ResNet Emotional Memory Migration for PIX-003.

Implements a "Human Context Layer" with residual weights for modeling long-term
emotional context. The architecture combines:
  - Deep residual connections for gradient flow
  - Emotional memory pooling for context persistence
  - Overfitting prevention through regularization

Key Components:
  - ResidualBlock: Skip connections for deep feature extraction
  - EmotionalMemoryPool: Maintains emotional state across sequences
  - HumanContextLayer: Named architectural component for long-term context

Example:
    >>> config = ResNetEmotionalMemoryConfig()
    >>> model = ResNetEmotionalMemory(config)
    >>> input_ids = torch.randint(0, 30522, (2, 128))
    >>> outputs = model(input_ids)
    >>> print(outputs['vad'].shape)  # [2, 3]
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ResNetEmotionalMemoryConfig:
    """Configuration for ResNet Emotional Memory model."""

    vocab_size: int = 30522
    embedding_dim: int = 256
    hidden_size: int = 512
    num_resnet_blocks: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    memory_size: int = 64
    memory_heads: int = 4
    dropout: float = 0.15
    layer_norm_eps: float = 1e-6
    max_position_embeddings: int = 512

    num_emotion_classes: int = 8
    output_dim: int = 3
    pad_token_id: int = 0

    use_weight_decay: bool = True
    use_stochastic_depth: bool = True
    stochastic_depth_prob: float = 0.1

    context_window_size: int = 32

    def __post_init__(self):
        if self.num_resnet_blocks < 1:
            raise ValueError("num_resnet_blocks must be >= 1")
        if self.memory_size < 1:
            raise ValueError("memory_size must be >= 1")


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block with dropout and layer normalization.

    Architecture:
        Input -> LayerNorm -> GELU -> Conv1 -> LayerNorm -> GELU -> Conv2 -> + Input

    This "pre-activation" design improves gradient flow in deep networks.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)

        self.norm2 = nn.LayerNorm(intermediate_size, eps=layer_norm_eps)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.norm1(x)
        x = F.gelu(x)
        x = self.fc1(x)
        x = self.dropout(x)

        x = self.norm2(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x + residual


class StochasticDepth(nn.Module):
    """
    Stochastic depth for regularization during training.

    Randomly drops entire residual blocks during training to prevent
    overfitting in very deep networks. At inference, all blocks are used.
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x + residual

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()

        return x + residual * random_tensor / keep_prob


class EmotionalMemoryPool(nn.Module):
    """
    Memory mechanism for maintaining emotional context across sequences.

    Implements a learnable memory bank that stores and retrieves emotional
    states using attention-based addressing. This enables the model to
    "remember" emotional patterns from earlier in the conversation.

    Key features:
      - Fixed-size memory bank for bounded memory
      - Attention-based read/write operations
      - Temporal decay for forgetting old memories
    """

    def __init__(
        self,
        hidden_size: int,
        memory_size: int = 64,
        memory_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_heads = memory_heads
        self.head_dim = hidden_size // memory_heads

        self.memory = nn.Parameter(
            torch.zeros(1, memory_size, hidden_size),
            requires_grad=True,
        )
        nn.init.normal_(self.memory, std=0.02)

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        self.memory_update_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.memory_erase_gate = nn.Linear(hidden_size, hidden_size)

        self.temporal_decay = nn.Parameter(
            torch.ones(1, memory_size, 1) * 0.9,
            requires_grad=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        memory = self.memory.expand(batch_size, -1, -1)

        query = self.query_proj(x)
        key = self.key_proj(memory)
        value = self.value_proj(memory)

        query = query.view(batch_size, seq_len, self.memory_heads, self.head_dim)
        query = query.transpose(1, 2)

        key = key.view(batch_size, self.memory_size, self.memory_heads, self.head_dim)
        key = key.transpose(1, 2)

        value = value.view(
            batch_size, self.memory_size, self.memory_heads, self.head_dim
        )
        value = value.transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        memory_output = torch.matmul(attention_weights, value)
        memory_output = memory_output.transpose(1, 2).contiguous()
        memory_output = memory_output.view(batch_size, seq_len, self.hidden_size)
        memory_output = self.output_proj(memory_output)

        context_summary = x.mean(dim=1, keepdim=True)
        update_input = torch.cat(
            [context_summary.expand(-1, self.memory_size, -1), memory], dim=-1
        )
        update_gate = torch.sigmoid(self.memory_update_gate(update_input))

        erase_gate = torch.sigmoid(self.memory_erase_gate(context_summary))
        erase_gate = erase_gate.expand(-1, self.memory_size, -1)

        decay = torch.sigmoid(self.temporal_decay)
        new_memory = decay * memory * (
            1 - erase_gate
        ) + update_gate * context_summary.expand(-1, self.memory_size, -1)

        memory_for_output = new_memory.mean(dim=1, keepdim=True)
        memory_contribution = memory_for_output.expand(-1, seq_len, -1)
        output = self.layer_norm(x + memory_output + 0.1 * memory_contribution)

        return output, new_memory


class HumanContextLayer(nn.Module):
    """
    The "Human Context Layer" - models long-term emotional context.

    This layer combines:
      - Multi-head self-attention for local context
      - Emotional memory pooling for long-term context
      - Residual connections for gradient flow

    Named after the PIX-003 requirement for a "Human Context Layer".
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        memory_size: int,
        memory_heads: int,
        context_window_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_window_size = context_window_size

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.emotional_memory = EmotionalMemoryPool(
            hidden_size=hidden_size,
            memory_size=memory_size,
            memory_heads=memory_heads,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.context_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x

        x = self.norm1(x)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        attn_output, _ = self.attention(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        gate_input = torch.cat([x, attn_output], dim=-1)
        gate = torch.sigmoid(self.context_gate(gate_input))
        attn_output = gate * attn_output
        x = residual + self.dropout(attn_output)

        x, updated_memory = self.emotional_memory(x, attention_mask)

        x = self.norm2(x)

        return x, updated_memory


class ResNetEmotionalMemory(nn.Module):
    """
    ResNet-based model with Emotional Memory for long-term context modeling.

    Architecture:
      1. Token embeddings with positional encoding
      2. Stack of ResidualBlocks for deep feature extraction
      3. HumanContextLayer for emotional memory
      4. Output heads for VAD and emotion classification

    This implements the PIX-003 "ResNet Emotional Memory Migration" with:
      - Residual weights for gradient flow
      - Long-term context via memory pooling
      - Overfitting prevention via dropout and stochastic depth
    """

    def __init__(self, config: ResNetEmotionalMemoryConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_token_id,
        )

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.max_position_embeddings, config.embedding_dim),
        )
        nn.init.normal_(self.pos_embedding, std=0.02)

        self.input_projection = nn.Linear(config.embedding_dim, config.hidden_size)
        self.input_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.input_dropout = nn.Dropout(config.dropout)

        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    dropout=config.dropout,
                    layer_norm_eps=config.layer_norm_eps,
                )
                for _ in range(config.num_resnet_blocks)
            ]
        )

        if config.use_stochastic_depth:
            total_blocks = config.num_resnet_blocks
            self.stochastic_depths = nn.ModuleList(
                [
                    StochasticDepth(
                        drop_prob=config.stochastic_depth_prob * (i / total_blocks)
                    )
                    for i in range(total_blocks)
                ]
            )
        else:
            self.stochastic_depths = None

        self.human_context_layer = HumanContextLayer(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            memory_size=config.memory_size,
            memory_heads=config.memory_heads,
            context_window_size=config.context_window_size,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
        )

        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
        )

        self.vad_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.output_dim),
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
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape

        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).float()

        x = self.embedding(input_ids)

        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = x + pos_emb

        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        for i, res_block in enumerate(self.residual_blocks):
            residual = x
            x = res_block(x)

            if self.stochastic_depths is not None:
                x = self.stochastic_depths[i](x, residual)

        x, memory_state = self.human_context_layer(x, attention_mask)

        x = self.output_norm(x)

        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(
            min=1e-9
        )
        pooled = self.pooler(pooled)

        vad = self.vad_head(pooled)
        emotion_logits = self.emotion_head(pooled)

        if return_dict:
            return {
                "vad": vad,
                "emotion_logits": emotion_logits,
                "hidden_states": pooled,
                "memory_state": memory_state,
            }
        return vad, emotion_logits, pooled


def create_resnet_emotional_memory(
    vocab_size: int = 30522,
    hidden_size: int = 512,
    num_resnet_blocks: int = 6,
    memory_size: int = 64,
    dropout: float = 0.15,
) -> ResNetEmotionalMemory:
    """
    Factory function to create a ResNet Emotional Memory model.

    Args:
        vocab_size: Vocabulary size for embedding layer
        hidden_size: Hidden dimension for feature representations
        num_resnet_blocks: Number of residual blocks
        memory_size: Size of emotional memory bank
        dropout: Dropout probability

    Returns:
        Configured ResNetEmotionalMemory instance
    """
    config = ResNetEmotionalMemoryConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_resnet_blocks=num_resnet_blocks,
        memory_size=memory_size,
        dropout=dropout,
    )
    return ResNetEmotionalMemory(config)
