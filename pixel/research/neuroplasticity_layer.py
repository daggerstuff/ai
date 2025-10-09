"""
neuroplasticity_layer.py

Production-grade Neuroplasticity-Inspired Dynamic Architecture Adaptation (7.4)
------------------------------------------------------------------------------
Implements adaptive neural pathways and dynamic weight adjustment mechanisms inspired by neuroplasticity,
enabling the model to strengthen or weaken emotional pathways based on learning signals and success metrics.

Author: Roo (AI)
Date: 2025-07-17
"""

import torch
from torch import nn
from torch.nn import functional


class NeuroplasticityLayer(nn.Module):
    """
    Adaptive layer that dynamically adjusts connection strengths based on emotional learning signals.
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
        eligibility_trace = self.__dict__["eligibility_trace"]
        if reward_signal is not None and isinstance(eligibility_trace, torch.Tensor):
            # Hebbian-like update: eligibility_trace += outer(reward_signal, x)
            if reward_signal.dim() == 1:
                reward_signal = reward_signal.unsqueeze(0)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            batch_size = x.size(0)
            for b in range(batch_size):
                x_b = x[b].view(1, -1)  # (1, input_dim)
                r_b = reward_signal[b].view(-1, 1)  # (output_dim, 1)
                hebbian_update = r_b @ x_b  # (output_dim, input_dim)
                eligibility_trace.add_(hebbian_update)
            # Apply plasticity
            self.weight.data.add_(self.plasticity_rate * eligibility_trace)
            # Decay eligibility trace
            eligibility_trace.mul_(0.9)
        return out


# Example integration (for documentation/testing, not for production import)
if __name__ == "__main__":
    input_dim = 16
    output_dim = 8
    layer = NeuroplasticityLayer(input_dim, output_dim, plasticity_rate=0.05)
    x = torch.randn(4, input_dim)
    reward = torch.randn(4, output_dim)
    out = layer(x, reward_signal=reward)
