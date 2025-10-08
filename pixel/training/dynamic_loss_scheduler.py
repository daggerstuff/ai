"""
dynamic_loss_scheduler.py

Advanced dynamic loss weight scheduler for Pixel multi-objective training.
Supports multiple scheduling strategies, training phase management, crisis detection, and momentum smoothing.
"""

from typing import Dict, List, Optional

import numpy as np


class DynamicLossScheduler:
    """
    Dynamic loss weight scheduler with multiple strategies:
    - fixed
    - performance-based
    - cosine annealing
    - exponential decay
    - curriculum learning
    - adaptive gradient
    - Pareto optimal
    Includes training phase management, crisis detection, and momentum smoothing.
    """

    def __init__(
        self,
        objectives: list[str],
        initial_weights: dict[str, float] | None = None,
        strategy: str = "fixed",
    ):
        """
        Initialize the scheduler.

        Args:
            objectives: List of objective names.
            initial_weights: Optional dict of initial weights.
            strategy: Scheduling strategy.
        """
        self.objectives = objectives
        self.weights = initial_weights or {obj: 1.0 for obj in objectives}
        self.strategy = strategy
        self.history: List[Dict[str, float]] = []

    def step(
        self, metrics: Dict[str, float], epoch: int, phase: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Update and return new loss weights based on the selected strategy.

        Args:
            metrics: Dict of current metrics for each objective.
            epoch: Current training epoch.
            phase: Optional training phase.

        Returns:
            Updated weights dict.
        """
        if self.strategy == "fixed":
            return self.weights
        elif self.strategy == "performance":
            return self._performance_based(metrics)
        elif self.strategy == "cosine":
            return self._cosine_annealing(epoch)
        elif self.strategy == "exponential":
            return self._exponential_decay(epoch)
        elif self.strategy == "curriculum":
            return self._curriculum_learning(epoch, phase)
        elif self.strategy == "adaptive":
            return self._adaptive_gradient(metrics)
        elif self.strategy == "pareto":
            return self._pareto_optimal(metrics)
        else:
            return self.weights

    def _performance_based(self, metrics: Dict[str, float]) -> Dict[str, float]:
        # Adjust weights inversely to metric (lower metric = higher weight), normalized
        inv_metrics = {obj: 1.0 / (metrics[obj] + 1e-6) for obj in self.objectives}
        total = sum(inv_metrics.values())
        return {obj: inv_metrics[obj] / total for obj in self.objectives}

    def _cosine_annealing(self, epoch: int) -> Dict[str, float]:
        # Cosine annealing for all objectives, with independent phase for each
        T_max = 100
        weights = {}
        for i, obj in enumerate(self.objectives):
            phase = (epoch + i * (T_max // len(self.objectives))) % T_max
            factor = 0.5 * (1 + np.cos(np.pi * phase / T_max))
            weights[obj] = self.weights[obj] * factor
        total = sum(weights.values())
        if total > 0:
            weights = {obj: w / total for obj, w in weights.items()}
        return weights

    def _exponential_decay(self, epoch: int) -> Dict[str, float]:
        # Exponential decay for all objectives, with different rates
        base_decay = 0.95
        weights = {}
        for i, obj in enumerate(self.objectives):
            decay = base_decay ** (epoch + i)
            weights[obj] = self.weights[obj] * decay
        total = sum(weights.values())
        if total > 0:
            weights = {obj: w / total for obj, w in weights.items()}
        return weights

    def _curriculum_learning(self, epoch: int, phase: Optional[str]) -> Dict[str, float]:
        # Increase weight for current curriculum phase, ramp up over time
        weights = self.weights.copy()
        ramp = min(1.0, epoch / 10.0)
        if phase and phase in weights:
            for obj in weights:
                if obj == phase:
                    weights[obj] *= 1.0 + ramp
                else:
                    weights[obj] *= 1.0 - 0.5 * ramp
        total = sum(weights.values())
        if total > 0:
            weights = {obj: w / total for obj, w in weights.items()}
        return weights

    def _adaptive_gradient(self, metrics: Dict[str, float]) -> Dict[str, float]:
        # Simulate adaptive gradient norm-based weighting (mock: use metric stddev)
        values = np.array([metrics[obj] for obj in self.objectives])
        std = np.std(values)
        if std < 1e-6:
            return self.weights
        normed = (values - np.min(values)) / (np.ptp(values) + 1e-6)
        weights = {obj: 1.0 - normed[i] for i, obj in enumerate(self.objectives)}
        total = sum(weights.values())
        if total > 0:
            weights = {obj: w / total for obj, w in weights.items()}
        return weights

    def _pareto_optimal(self, metrics: Dict[str, float]) -> Dict[str, float]:
        # Simulate Pareto optimal weighting: emphasize objectives with highest loss
        sorted_objs = sorted(self.objectives, key=lambda obj: metrics[obj], reverse=True)
        weights = {obj: 1.0 / (i + 1) for i, obj in enumerate(sorted_objs)}
        total = sum(weights.values())
        if total > 0:
            weights = {obj: w / total for obj, w in weights.items()}
        return weights

    def detect_crisis(self, metrics: Dict[str, float]) -> bool:
        """
        Detects crisis (e.g., loss spikes, metric collapse) in training.

        Args:
            metrics: Dict of current metrics.

        Returns:
            True if crisis detected, False otherwise.
        """
        # Placeholder: Crisis if any metric is NaN or very large
        for v in metrics.values():
            if np.isnan(v) or v > 1e6:
                return True
        return False

    def smooth_weights(
        self, new_weights: Dict[str, float], momentum: float = 0.9
    ) -> Dict[str, float]:
        """
        Applies momentum smoothing to weight updates.

        Args:
            new_weights: Dict of new weights.
            momentum: Smoothing factor.

        Returns:
            Smoothed weights dict.
        """
        smoothed = {}
        for obj in self.objectives:
            prev = self.weights.get(obj, 1.0)
            smoothed[obj] = momentum * prev + (1 - momentum) * new_weights[obj]
        self.weights = smoothed
        return smoothed
