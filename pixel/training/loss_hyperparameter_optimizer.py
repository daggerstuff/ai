"""
loss_hyperparameter_optimizer.py

Comprehensive hyperparameter optimization system for Pixel multi-objective loss function.
Uses Optuna for Bayesian optimization of loss weights, learning rates, and training strategies.
"""

from typing import Dict, List, Callable, Any, Optional

try:
    import optuna
except ImportError:
    optuna = None


class LossHyperparameterOptimizer:
    """
    Hyperparameter optimizer for multi-objective loss function using Optuna.
    """

    def __init__(
        self,
        objectives: List[str],
        loss_fn: Callable[[Dict[str, float], Any], float],
        train_fn: Callable[[Dict[str, float]], float],
        n_trials: int = 50,
        study_name: str = "pixel_loss_optimization",
    ):
        """
        Initialize the optimizer.

        Args:
            objectives: List of objective names.
            loss_fn: Function to compute loss given weights and data.
            train_fn: Function to run training and return validation loss.
            n_trials: Number of optimization trials.
            study_name: Name for the Optuna study.
        """
        self.objectives = objectives
        self.loss_fn = loss_fn
        self.train_fn = train_fn
        self.n_trials = n_trials
        self.study_name = study_name
        self.study = None

    def objective(self, trial):
        """
        Optuna objective function for loss weight optimization.

        Args:
            trial: Optuna trial object.

        Returns:
            Validation loss (float).
        """
        weights = {obj: trial.suggest_float(f"weight_{obj}", 0.1, 2.0) for obj in self.objectives}
        # Optionally optimize learning rate, etc.
        # lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        val_loss = self.train_fn(weights)
        return val_loss

    def run(self):
        """
        Run the hyperparameter optimization.

        Returns:
            Best weights dict.
        """
        if optuna is None:
            raise ImportError("Optuna is not installed.")
        self.study = optuna.create_study(direction="minimize", study_name=self.study_name)
        self.study.optimize(self.objective, n_trials=self.n_trials)
        best_params = self.study.best_params
        best_weights = {obj: best_params[f"weight_{obj}"] for obj in self.objectives}
        return best_weights

    def report(self) -> Optional[Dict[str, Any]]:
        """
        Generate a report of the optimization results.

        Returns:
            Dict with best trial, all trials, and summary statistics.
        """
        if self.study is None:
            return None
        return {
            "best_trial": self.study.best_trial,
            "trials": [t for t in self.study.trials],
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
        }
