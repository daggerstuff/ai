"""Backwards-compatible import path for the observability singleton.

The implementation lives in ``ai.monitoring.observability``; this module
keeps the older ``ai.qa.reports.observability`` import working.
"""

from ai.monitoring.observability import ObservabilityManager, observability

__all__ = [
    "ObservabilityManager",
    "observability",
]
