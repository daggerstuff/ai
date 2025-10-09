"""
refactored_neuroplasticity_layer.py

Production-grade Neuroplasticity-Inspired Dynamic Architecture Adaptation (7.4) - Refactored
-------------------------------------------------------------------------------------------
This version includes a vectorized implementation for efficient batch processing.

Author: Roo (AI)
Date: 2025-07-17
"""

import torch
from torch import nn
from torch.nn import functional


class RefactoredNeuroplasticityLayer(nn.Module):
    """
    Adaptive layer that dynamically adjusts connection strengths based on emotional learning signals.
    This version uses vectorized operations for efficient batch processing.
    """

    def __init__(self, input_dim: int, output_dim: int, plasticity_rate: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.plasticity_rate = plasticity_rate
        self.register_buffer("eligibility_trace", torch.zeros(output_dim, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)
        if isinstance(self.eligibility_trace, torch.Tensor):
            self.eligibility_trace.zero_()

    def forward(self, x: torch.Tensor, reward_signal: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass with optional reward signal for plasticity update.
        If reward_signal is provided, update eligibility trace and adjust weights.
        """
        out = functional.linear(x, self.weight, self.bias)
        if reward_signal is not None:
            # Vectorized Hebbian-like update
            # x is (batch, input_dim), reward_signal is (batch, output_dim)
            # We want to compute the sum of outer products: sum(r_i.T @ x_i)
            # This is equivalent to reward_signal.T @ x
            hebbian_update = torch.matmul(reward_signal.T, x)
            self.eligibility_trace.add_(hebbian_update)

            # Apply plasticity
            self.weight.data.add_(self.plasticity_rate * self.eligibility_trace)
            # Decay eligibility trace
            self.eligibility_trace.mul_(0.9)
        return out
