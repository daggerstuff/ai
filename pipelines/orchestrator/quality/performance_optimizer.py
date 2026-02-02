"""Performance optimization utilities."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""

    average_operation_time: float = 0.0
    total_operations: int = 0
    success_count: int = 0
    error_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class PerformanceOptimizer:
    """Performance optimization for dataset operations."""

    def __init__(self):
        self._metrics = PerformanceMetrics()
        self._cache: dict[str, Any] = {}

    def clear_cache(self) -> None:
        """Clear the performance cache."""
        self._cache.clear()

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self._metrics

    def record_operation(self, duration: float, success: bool = True) -> None:
        """Record an operation for metrics tracking."""
        self._metrics.total_operations += 1
        if success:
            self._metrics.success_count += 1
        else:
            self._metrics.error_count += 1

        # Update running average
        if self._metrics.total_operations > 0:
            self._metrics.average_operation_time = (
                (
                    self._metrics.average_operation_time
                    * (self._metrics.total_operations - 1)
                )
                + duration
            ) / self._metrics.total_operations

    def shutdown(self) -> None:
        """Shutdown the performance optimizer."""
        self._cache.clear()
        self._metrics = PerformanceMetrics()
