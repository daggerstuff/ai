"""Tests for ai/research/reprioritization_engine.py — lightweight evidence-based ordering."""

from __future__ import annotations

from ai.research.reprioritization_engine import EvidenceItem, ReprioritizationEngine


def test_reprioritization_engine_basic():
    engine = ReprioritizationEngine(base_priority=["task_a", "task_b", "task_c"])
    evidence = [
        EvidenceItem(source_id="eval1", evidence_type="gap", score=2.5, details={"task_id": "task_c"}),
        EvidenceItem(source_id="eval2", evidence_type="issue", score=1.0, details={"task_id": "task_a"}),
    ]
    new_order = engine.compute_new_order(evidence)
    assert new_order == ["task_c", "task_a", "task_b"]


def test_reprioritization_engine_no_evidence():
    engine = ReprioritizationEngine(base_priority=["t1", "t2"])
    new_order = engine.compute_new_order([])
    assert new_order == ["t1", "t2"]


def test_evidence_item_has_timestamp():
    item = EvidenceItem(source_id="s1", evidence_type="gap", score=1.0, details={})
    assert item.timestamp is not None
    from datetime import datetime

    datetime.fromisoformat(item.timestamp)


def test_reprioritization_preserves_order_for_equal_scores():
    engine = ReprioritizationEngine(base_priority=["a", "b", "c"])
    evidence = [
        EvidenceItem(source_id="e1", evidence_type="x", score=0.5, details={"task_id": "a"}),
        EvidenceItem(source_id="e2", evidence_type="y", score=0.5, details={"task_id": "b"}),
    ]
    new_order = engine.compute_new_order(evidence)
    assert new_order.index("a") < new_order.index("b") < new_order.index("c")
