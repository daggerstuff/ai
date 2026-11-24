"""Compatibility alias for the live implementation.

PIX-511: Sprint 2 - Memory Ingestion Gating System.

The 5-gate pipeline (PII redaction, crisis detection, trauma-trigger filtering,
consent-gated retrieval, human review queue) is defined once, in
``ai.research.gates``. This module re-exports its public names rather than
restating the classes, so ``ai.memory.gates.GateResult`` and
``ai.research.gates.GateResult`` are the same class object — the duplicate
definition here is what made ``isinstance`` fail across the two import paths
once the gate submodules became aliases. The submodules under this package are
``sys.modules`` aliases onto their research counterparts.
"""

from ai.research.gates import GateDecision, GateResult, GatingReport

__all__ = ["GateDecision", "GateResult", "GatingReport"]
