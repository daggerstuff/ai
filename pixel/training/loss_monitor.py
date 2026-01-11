"""
loss_monitor.py

Comprehensive loss component monitoring and visualization system for Pixel training.
Provides real-time tracking, logging, and summary statistics for all loss components.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt


class LossMonitor:
    """
    Real-time loss component monitoring and visualization.
    Tracks loss values for each objective, logs history, and provides summary statistics.
    """

    def __init__(self, objectives: List[str]):
        """
        Initialize the loss monitor.

        Args:
            objectives: List of objective names to track.
        """
        self.objectives = objectives
        self.history: Dict[str, List[float]] = {obj: [] for obj in objectives}

    def log(self, losses: Dict[str, float]):
        """
        Log current loss values for each objective.

        Args:
            losses: Dict of objective name to loss value.
        """
        for obj in self.objectives:
            self.history[obj].append(losses.get(obj, 0.0))

    def get_latest(self) -> Dict[str, float]:
        """
        Get the most recent loss value for each objective.

        Returns:
            Dict of objective name to latest loss value.
        """
        return {obj: self.history[obj][-1] if self.history[obj] else 0.0 for obj in self.objectives}

    def get_history(self, objective: str) -> List[float]:
        """
        Get the full loss history for a specific objective.

        Args:
            objective: Objective name.

        Returns:
            List of loss values.
        """
        return self.history.get(objective, [])

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics (mean, min, max) for each objective.

        Returns:
            Dict of objective name to summary stats.
        """
        stats = {}
        for obj, values in self.history.items():
            if values:
                stats[obj] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
            else:
                stats[obj] = {"mean": 0.0, "min": 0.0, "max": 0.0}
        return stats

    def plot(self, save_path: Optional[str] = None):
        """
        Plot loss curves for all objectives.

        Args:
            save_path: Optional path to save the plot image.
        """
        for obj in self.objectives:
            plt.plot(self.history[obj], label=obj)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Loss Component Monitoring")
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
