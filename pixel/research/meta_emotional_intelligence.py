"""
meta_emotional_intelligence.py

Production-grade Meta-Emotional Intelligence and Self-Awareness (7.7)
---------------------------------------------------------------------
Implements systems for emotional self-monitoring, reflection, and adaptive learning.
Supports meta-cognition, self-regulation, and emotional state reporting for advanced AI agents.

Author: Roo (AI)
Date: 2025-07-17
"""

from typing import Any

import torch
from torch import nn


class MetaEmotionalIntelligence(nn.Module):
    """
    Provides meta-cognitive capabilities for emotional self-awareness and adaptation.
    Tracks internal emotional state, monitors changes, and enables self-regulation.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_tracker = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.self_reflection = nn.Linear(hidden_dim, 1)
        self.regulation_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, emotion_seq: torch.Tensor) -> dict[str, Any]:
        """
        Args:
            emotion_seq: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Dictionary with self-awareness metrics and regulation output.
        """
        rnn_out, _ = self.state_tracker(emotion_seq)
        # Self-awareness: how much the current state deviates from the mean
        mean_state = rnn_out.mean(dim=1, keepdim=True)
        deviation = (rnn_out - mean_state).abs().mean(dim=(1, 2))
        reflection_score = torch.sigmoid(self.self_reflection(rnn_out)).mean(dim=1)
        regulation = self.regulation_head(rnn_out)
        return {
            "deviation": deviation,
            "reflection_score": reflection_score,
            "regulation": regulation,
        }

    @staticmethod
    def report_self_awareness(metrics: dict[str, Any]) -> str:
        """
        Generates a human-readable report of the model's self-awareness metrics.
        """
        deviation = metrics.get("deviation")
        reflection = metrics.get("reflection_score")
        return (
            f"Meta-EI Report:\n"
            f"  Deviation from mean state: {deviation}\n"
            f"  Reflection score: {reflection}\n"
        )


# Example integration (for documentation/testing, not for production import)
if __name__ == "__main__":
    batch_size = 2
    seq_len = 8
    input_dim = 10
    model = MetaEmotionalIntelligence(input_dim)
    emotions = torch.randn(batch_size, seq_len, input_dim)
    metrics = model(emotions)
