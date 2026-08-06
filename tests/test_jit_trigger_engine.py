"""Unit tests for JITTriggerEngine: per-clinician rolling windows + persistence."""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import UTC, datetime, timedelta

# Ensure ai/ is on sys.path for `triggers.*` imports
_ai_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ai_root not in sys.path:
    sys.path.insert(0, _ai_root)

from triggers.case_flag import CaseFlag, FlagType
from triggers.flag_store import FlagStore
from triggers.jit_trigger_engine import JITTriggerEngine


def _make_flag(
    clinician_id: str = "clin-1",
    flag_type: FlagType = FlagType.BIAS,
    minutes_ago: float = 0,
    bias_score: float = 0.5,
    session_id: str = "sess-1",
) -> CaseFlag:
    ts = datetime.now(UTC) - timedelta(minutes=minutes_ago)
    return CaseFlag(
        clinician_id=clinician_id,
        flag_type=flag_type,
        timestamp=ts,
        session_id=session_id,
        bias_score=bias_score,
        detected_biases=["authority_bias"],
    )


def test_per_clinician_grouping_no_cross_leakage():
    engine = JITTriggerEngine(threshold=3)
    engine.add_flag(_make_flag("clin-A", minutes_ago=10))
    engine.add_flag(_make_flag("clin-A", minutes_ago=5))
    result_a = engine.add_flag(_make_flag("clin-A", minutes_ago=1))
    assert result_a is not None
    assert result_a.should_trigger
    assert result_a.clinician_id == "clin-A"

    # clin-B has only 1 flag — no trigger
    result_b = engine.add_flag(_make_flag("clin-B", minutes_ago=1))
    assert result_b is None


def test_7_day_window_excludes_old_flags():
    engine = JITTriggerEngine(window_days=7, threshold=3)
    # Old flag (8 days ago) should not count
    engine.add_flag(_make_flag("clin-1", minutes_ago=8 * 24 * 60))
    # Two recent flags — not enough
    engine.add_flag(_make_flag("clin-1", minutes_ago=60))
    result = engine.add_flag(_make_flag("clin-1", minutes_ago=30))
    # Only 2 recent flags, not 3
    assert result is None


def test_threshold_exact_3_triggers():
    engine = JITTriggerEngine(threshold=3)
    engine.add_flag(_make_flag(minutes_ago=120))
    engine.add_flag(_make_flag(minutes_ago=60))
    result = engine.add_flag(_make_flag(minutes_ago=30))
    assert result is not None
    assert result.matching_flags == 3


def test_crisis_flag_type_also_triggers():
    engine = JITTriggerEngine(threshold=1)
    flag = _make_flag(flag_type=FlagType.CRISIS, bias_score=0.9)
    result = engine.add_flag(flag)
    assert result is not None
    assert result.should_trigger


def test_rolling_window_method_returns_only_recent():
    engine = JITTriggerEngine(window_days=7)
    engine.add_flag(_make_flag("clin-1", minutes_ago=10 * 24 * 60))  # 10 days ago
    engine.add_flag(_make_flag("clin-1", minutes_ago=60))  # 1 hour ago
    recent = engine.rolling_window("clin-1")
    assert len(recent) == 1
    assert recent[0].session_id == "sess-1"


def test_different_clinicians_isolated():
    engine = JITTriggerEngine(threshold=2)
    engine.add_flag(_make_flag("clin-A", minutes_ago=60))
    engine.add_flag(_make_flag("clin-A", minutes_ago=30))
    # One flag for clin-B — should not trigger even though clin-A already triggered
    result_b = engine.add_flag(_make_flag("clin-B", minutes_ago=10))
    assert result_b is None
    # clin-B needs one more
    result_b2 = engine.add_flag(_make_flag("clin-B", minutes_ago=5))
    assert result_b2 is not None
    assert result_b2.clinician_id == "clin-B"


def test_flags_persist_across_restart():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_flags.db")

        # First engine: add flags and persist
        store1 = FlagStore(database_url="", sqlite_path=db_path)
        engine1 = JITTriggerEngine(store=store1, threshold=3)
        engine1.add_flag(_make_flag("clin-1", minutes_ago=120))
        engine1.add_flag(_make_flag("clin-1", minutes_ago=60))
        store1.close()

        # Second engine: should backfill from store
        store2 = FlagStore(database_url="", sqlite_path=db_path)
        engine2 = JITTriggerEngine(store=store2, threshold=3)
        # Only 1 more flag needed (2 backfilled)
        result = engine2.add_flag(_make_flag("clin-1", minutes_ago=30))
        assert result is not None
        assert result.should_trigger
        store2.close()


def test_backfill_safe_multiple_clinicians():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_flags2.db")
        store = FlagStore(database_url="", sqlite_path=db_path)
        engine = JITTriggerEngine(store=store, threshold=2)
        engine.add_flag(_make_flag("clin-A", minutes_ago=60))
        engine.add_flag(_make_flag("clin-A", minutes_ago=30))
        engine.add_flag(_make_flag("clin-B", minutes_ago=60))
        store.close()

        # New engine backfills
        store2 = FlagStore(database_url="", sqlite_path=db_path)
        engine2 = JITTriggerEngine(store=store2, threshold=2)
        ids = sorted(engine2.get_clinician_ids())
        assert ids == ["clin-A", "clin-B"]
        # clin-A already has 2 flags — evaluate should trigger
        decision = engine2.evaluate_clinician("clin-A")
        assert decision.should_trigger
        # clin-B has 1 flag — no trigger
        decision_b = engine2.evaluate_clinician("clin-B")
        assert not decision_b.should_trigger
        store2.close()


def test_purge_old_flags():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_flags3.db")
        store = FlagStore(database_url="", sqlite_path=db_path)
        engine = JITTriggerEngine(store=store, window_days=7, threshold=10)
        engine.add_flag(_make_flag("clin-1", minutes_ago=10 * 24 * 60))  # old
        engine.add_flag(_make_flag("clin-1", minutes_ago=60))  # recent
        purged = engine.purge_old_flags()
        assert purged == 1
        remaining = engine.rolling_window("clin-1")
        assert len(remaining) == 1
        store.close()


def test_evaluate_below_threshold():
    engine = JITTriggerEngine(threshold=5)
    engine.add_flag(_make_flag("clin-1", minutes_ago=60))
    decision = engine.evaluate_clinician("clin-1")
    assert not decision.should_trigger
    assert decision.matching_flags == 1


def test_evaluate_at_threshold():
    engine = JITTriggerEngine(threshold=2)
    engine.add_flag(_make_flag("clin-1", minutes_ago=60))
    engine.add_flag(_make_flag("clin-1", minutes_ago=30))
    decision = engine.evaluate_clinician("clin-1")
    assert decision.should_trigger
    assert decision.matching_flags == 2
