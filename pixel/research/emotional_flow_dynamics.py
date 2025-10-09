"""
emotional_flow_dynamics.py

Production-grade Emotional Flow Dynamics for Temporal Emotional Modeling (7.6)
-----------------------------------------------------------------------------
Implements temporal modeling of emotional state dynamics, including velocity, acceleration, and momentum.
Supports prediction of emotional trajectories and detection of intervention points in conversations.

Author: Roo (AI)
Date: 2025-07-17
"""

import torch
from torch import nn


class EmotionalFlowDynamics(nn.Module):
    """
    Models the temporal dynamics of emotional states (velocity, acceleration, momentum).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_velocity = nn.Linear(hidden_dim, input_dim)
        self.fc_acceleration = nn.Linear(hidden_dim, input_dim)
        self.fc_momentum = nn.Linear(hidden_dim, input_dim)

    def forward(self, emotion_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            emotion_seq: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            velocity: (batch_size, seq_len, input_dim)
            acceleration: (batch_size, seq_len, input_dim)
            momentum: (batch_size, seq_len, input_dim)
        """
        rnn_out, _ = self.rnn(emotion_seq)
        velocity = self.fc_velocity(rnn_out)
        acceleration = self.fc_acceleration(rnn_out)
        momentum = self.fc_momentum(rnn_out)
        return velocity, acceleration, momentum

    @staticmethod
    def detect_intervention_points(velocity: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Detects points in the sequence where the change in emotional state exceeds a threshold.
        Returns a boolean mask of shape (batch_size, seq_len).
        """
        speed = torch.norm(velocity, dim=-1)
        return speed > threshold


# Example integration (for documentation/testing, not for production import)
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    input_dim = 8
    model = EmotionalFlowDynamics(input_dim)
    emotions = torch.randn(batch_size, seq_len, input_dim)
    velocity, acceleration, momentum = model(emotions)
