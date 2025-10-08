"""
test_gradient_conflict_resolver.py

Unit tests for GradientConflictResolver covering conflict detection and resolution strategies.
"""

import numpy as np

from ai.pixel.training.gradient_conflict_resolver import GradientConflictResolver


class TestGradientConflictResolver:
    def setup_method(self):
        self.gradients = {
            "obj1": np.array([1.0, 2.0, 3.0]),
            "obj2": np.array([-1.0, -2.0, -3.0]),
            "obj3": np.array([0.5, 0.5, 0.5]),
        }

    def test_detect_conflicts(self):
        resolver = GradientConflictResolver()
        conflicts = resolver.detect_conflicts(self.gradients)
        assert any("obj1 <-> obj2" in c or "obj2 <-> obj1" in c for c in conflicts)

    def test_surgery_strategy(self):
        resolver = GradientConflictResolver(strategy="surgery")
        resolved = resolver.resolve(self.gradients)
        assert isinstance(resolved, dict)
        assert all(isinstance(v, np.ndarray) for v in resolved.values())

    def test_normalization_strategy(self):
        resolver = GradientConflictResolver(strategy="normalization")
        resolved = resolver.resolve(self.gradients)
        for v in resolved.values():
            norm = np.linalg.norm(v)
            assert np.isclose(norm, 1.0)

    def test_dynamic_weighting_strategy(self):
        resolver = GradientConflictResolver(strategy="dynamic")
        resolved = resolver.resolve(self.gradients)
        assert isinstance(resolved, dict)

    def test_pareto_optimization_strategy(self):
        resolver = GradientConflictResolver(strategy="pareto")
        resolved = resolver.resolve(self.gradients)
        assert isinstance(resolved, dict)
