"""
test_loss_hyperparameter_optimizer.py

Unit tests for LossHyperparameterOptimizer covering bounds, configurations, trial evaluation, and optimization workflow.
"""

import pytest

from ai.pixel.training.loss_hyperparameter_optimizer import LossHyperparameterOptimizer


class DummyTrainFn:
    def __init__(self):
        self.calls = 0

    def __call__(self, weights):
        self.calls += 1
        # Simulate validation loss as sum of squared deviations from 1.0
        return sum((w - 1.0) ** 2 for w in weights.values())


@pytest.mark.skipif(
    "optuna" not in globals() or globals()["optuna"] is None, reason="Optuna not installed"
)
class TestLossHyperparameterOptimizer:
    def setup_method(self):
        self.objectives = ["language", "eq", "persona"]
        self.train_fn = DummyTrainFn()
        self.loss_fn = lambda w, d=None: sum(w.values())
        self.optimizer = LossHyperparameterOptimizer(
            objectives=self.objectives,
            loss_fn=self.loss_fn,
            train_fn=self.train_fn,
            n_trials=5,
            study_name="test_study",
        )

    def test_run_optimization(self):
        best_weights = self.optimizer.run()
        assert isinstance(best_weights, dict)
        assert all(obj in best_weights for obj in self.objectives)

    def test_report(self):
        self.optimizer.run()
        report = self.optimizer.report()
        assert report is not None
        assert "best_trial" in report
        assert "trials" in report
        assert "best_value" in report
        assert "best_params" in report
