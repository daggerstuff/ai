"""
test_dynamic_loss_scheduler.py

Unit tests for DynamicLossScheduler covering scheduling strategies, phase transitions, and crisis handling.
"""

import numpy as np

from ai.pixel.training.dynamic_loss_scheduler import DynamicLossScheduler


class TestDynamicLossScheduler:
    def setup_method(self):
        self.objectives = ["language", "eq", "persona", "clinical", "empathy"]
        self.initial_weights = {obj: 1.0 for obj in self.objectives}

    def test_fixed_strategy(self):
        scheduler = DynamicLossScheduler(self.objectives, self.initial_weights, strategy="fixed")
        weights = scheduler.step({}, 0)
        assert weights == self.initial_weights

    def test_performance_strategy(self):
        scheduler = DynamicLossScheduler(
            self.objectives, self.initial_weights, strategy="performance"
        )
        metrics = {obj: np.random.rand() + 0.1 for obj in self.objectives}
        weights = scheduler.step(metrics, 1)
        assert all(isinstance(w, float) for w in weights.values())

    def test_cosine_annealing(self):
        scheduler = DynamicLossScheduler(self.objectives, self.initial_weights, strategy="cosine")
        weights = scheduler.step({}, 10)
        assert all(isinstance(w, float) for w in weights.values())

    def test_exponential_decay(self):
        scheduler = DynamicLossScheduler(
            self.objectives, self.initial_weights, strategy="exponential"
        )
        weights = scheduler.step({}, 5)
        assert all(isinstance(w, float) for w in weights.values())

    def test_curriculum_learning(self):
        scheduler = DynamicLossScheduler(
            self.objectives, self.initial_weights, strategy="curriculum"
        )
        weights = scheduler.step({}, 3, phase="eq")
        assert "eq" in weights

    def test_adaptive_gradient(self):
        scheduler = DynamicLossScheduler(self.objectives, self.initial_weights, strategy="adaptive")
        metrics = {obj: np.random.rand() for obj in self.objectives}
        weights = scheduler.step(metrics, 2)
        assert all(isinstance(w, float) for w in weights.values())

    def test_pareto_optimal(self):
        scheduler = DynamicLossScheduler(self.objectives, self.initial_weights, strategy="pareto")
        metrics = {obj: np.random.rand() for obj in self.objectives}
        weights = scheduler.step(metrics, 2)
        assert all(isinstance(w, float) for w in weights.values())

    def test_crisis_detection(self):
        scheduler = DynamicLossScheduler(self.objectives, self.initial_weights)
        metrics = {obj: 1e7 for obj in self.objectives}
        assert scheduler.detect_crisis(metrics) is True
        metrics = {obj: 0.5 for obj in self.objectives}
        assert scheduler.detect_crisis(metrics) is False

    def test_momentum_smoothing(self):
        scheduler = DynamicLossScheduler(self.objectives, self.initial_weights)
        new_weights = {obj: 0.5 for obj in self.objectives}
        smoothed = scheduler.smooth_weights(new_weights, momentum=0.8)
        assert all(isinstance(w, float) for w in smoothed.values())
