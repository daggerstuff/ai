"""
gradient_conflict_resolver.py

Advanced gradient conflict detection and resolution system for Pixel multi-objective training.
Supports multiple strategies: surgery, normalization, dynamic weighting, Pareto optimization.
"""

from typing import Dict

import numpy as np


class GradientConflictResolver:
    """
    Detects and resolves gradient conflicts in multi-objective optimization.
    """

    def __init__(self, strategy: str = "surgery"):
        """
        Initialize the resolver.

        Args:
            strategy: Conflict resolution strategy ("surgery", "normalization", "dynamic", "pareto").
        """
        self.strategy = strategy

    def detect_conflicts(self, gradients: dict[str, np.ndarray]) -> list[str]:
        """
        Detects conflicting gradients between objectives.

        Args:
            gradients: Dict of objective name to gradient vector.

        Returns:
            List of objective pairs with detected conflicts.
        """
        conflicts = []
        keys = list(gradients.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                g1, g2 = gradients[keys[i]], gradients[keys[j]]
                if np.dot(g1.flatten(), g2.flatten()) < 0:
                    conflicts.append(f"{keys[i]} <-> {keys[j]}")
        return conflicts

    def resolve(self, gradients: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Resolves gradient conflicts using the selected strategy.

        Args:
            gradients: Dict of objective name to gradient vector.

        Returns:
            Dict of objective name to resolved gradient vector.
        """
        if self.strategy == "surgery":
            return self._surgery(gradients)
        if self.strategy == "normalization":
            return self._normalize(gradients)
        if self.strategy == "dynamic":
            return self._dynamic_weighting(gradients)
        if self.strategy == "pareto":
            return self._pareto_optimization(gradients)
        return gradients

    def _surgery(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # PCGrad-style projection: for each pair, project conflicting gradients
        keys = list(gradients.keys())
        new_grads = {k: gradients[k].copy() for k in keys}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                g1, g2 = new_grads[keys[i]], new_grads[keys[j]]
                dot = np.dot(g1.flatten(), g2.flatten())
                if dot < 0:
                    proj = dot / (np.linalg.norm(g2.flatten()) ** 2 + 1e-8)
                    new_grads[keys[i]] = g1 - proj * g2
                    new_grads[keys[j]] = g2 - proj * g1
        return new_grads

    def _normalize(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Normalize all gradients to unit norm
        return {k: v / (np.linalg.norm(v) + 1e-8) for k, v in gradients.items()}

    def _dynamic_weighting(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Weight gradients by their norm (higher norm = lower weight)
        norms = {k: np.linalg.norm(v) for k, v in gradients.items()}
        total = sum(1.0 / (n + 1e-8) for n in norms.values())
        weights = {k: (1.0 / (norms[k] + 1e-8)) / total for k in gradients}
        return {k: gradients[k] * weights[k] for k in gradients}

    def _pareto_optimization(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Simple convex combination for Pareto optimality (equal weights)
        n = len(gradients)
        combined = sum(gradients[k] for k in gradients) / n
        # Ensure the returned dict values are all np.ndarray (broadcast shape)
        import numpy as np

        arr = np.array(combined, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return {k: arr.copy() for k in gradients}
