"""Tests for the long-context timeline drift probe.

Two things can silently invalidate the probe: picking prefixes that are not
actually the deepest in their context band, and scoring a broken generation as
if it were a drift observation. These tests pin both, plus the rollup maths.
"""

from __future__ import annotations

import random

from training.probe_timeline_drift import (
    BUCKETS,
    BucketSummary,
    ProbeSettings,
    TimelinePrefix,
    _cap,
    _session_prefixes,
    bucket_for,
    build_messages,
    enumerate_prefixes,
    score_sample,
)


def _client(text: str = "I am tired.") -> dict[str, object]:
    return {"role": "client", "content": text}


def _therapist(tl: str | None, spoken: str = "Reply.") -> dict[str, object]:
    turn: dict[str, object] = {"role": "therapist", "content": spoken}
    if tl is not None:
        turn["ledger"] = {"dx": "x", "tl": tl}
    return turn


def _session(tls: list[str | None]) -> list[dict[str, object]]:
    turns: list[dict[str, object]] = []
    for tl in tls:
        turns.append(_client())
        turns.append(_therapist(tl))
    return turns


def _prefix(
    bucket: str = "short",
    inherited: tuple[str, ...] = ("now",),
    arc_id: str = "arc_1",
    turn_ordinal: int = 2,
) -> TimelinePrefix:
    return TimelinePrefix(
        arc_id=arc_id,
        session_n=1,
        turn_ordinal=turn_ordinal,
        bucket=bucket,
        messages=[{"role": "user", "content": "hello"}],
        inherited=frozenset(inherited),
        reference=frozenset(inherited),
    )


class TestBucketFor:
    def test_edges_follow_the_declared_strata(self) -> None:
        assert bucket_for(8) == "short"
        assert bucket_for(9) == "short"
        assert bucket_for(10) == "medium"
        assert bucket_for(19) == "medium"
        assert bucket_for(20) == "long"
        assert bucket_for(31) == "long"
        assert bucket_for(32) == "very_long"
        assert bucket_for(74) == "very_long"


class TestBuildMessages:
    def test_serialises_therapists_as_ledger_plus_spoken(self) -> None:
        messages = build_messages(_session(["now: x"])[:2])
        assert messages is not None
        assert messages[0] == {"role": "user", "content": "I am tired."}
        assert messages[1]["content"].startswith('{"dx": "x", "tl": "now: x"}')
        assert messages[1]["content"].endswith("Reply.")

    def test_rejects_a_therapist_turn_without_a_ledger(self) -> None:
        assert build_messages(_session([None])) is None

    def test_rejects_a_prefix_that_does_not_open_with_the_client(self) -> None:
        assert build_messages([_therapist("now: x")]) is None


class TestSessionPrefixes:
    def test_keeps_the_deepest_prefix_in_each_band(self) -> None:
        turns = _session([f"now: turn {i}" for i in range(12)])
        prefixes = _session_prefixes("arc_1", 1, turns)
        assert [prefix.bucket for prefix in prefixes] == ["short", "medium", "long"]
        assert len(prefixes[0].messages) == 8
        assert len(prefixes[1].messages) == 18
        assert len(prefixes[2].messages) == 22

    def test_inherited_anchors_come_from_the_prefix_ledgers(self) -> None:
        turns = _session(["now: a", "now: a; -1w: b", "now: a; -1w: b; -2w: c", "now: a; -1w: b; -2w: c; +1y: d"])
        prefix = _session_prefixes("arc_1", 1, turns)[0]
        assert prefix.inherited == frozenset({"now", "-1w", "-2w"})
        assert prefix.reference == frozenset({"now", "-1w", "-2w", "+1y"})
        assert prefix.reference_holds

    def test_reference_records_drift_in_the_accepted_turn(self) -> None:
        turns = _session(["now: a", "now: a; -1w: b"])
        turns[3]["ledger"] = {"tl": "-1w: b"}  # accepted turn dropped "now"
        prefix = _session_prefixes("arc_1", 1, turns)[0]
        assert not prefix.reference_holds

    def test_skips_a_prefix_with_no_following_therapist_turn(self) -> None:
        turns = _session(["now: a", "-1w: b"])[:-1]  # ends on a client turn
        assert _session_prefixes("arc_1", 1, turns) == []


class TestEnumeratePrefixes:
    def test_groups_by_bucket_and_drops_anchored_less_prefixes(self) -> None:
        long_session = _session([f"now: turn {i}" for i in range(16)])
        sessions = {
            ("arc_1", 1): long_session,
            ("arc_2", 1): _session([""] * 16),
        }
        grouped = enumerate_prefixes(sessions, excluded=set())
        assert set(grouped) == set(BUCKETS)
        assert all(prefix.arc_id == "arc_1" for bucket in grouped.values() for prefix in bucket)
        assert sum(len(rows) for rows in grouped.values()) == 3

    def test_excludes_arcs_from_heldout_splits(self) -> None:
        sessions = {("arc_1", 1): _session([f"now: turn {i}" for i in range(16)])}
        grouped = enumerate_prefixes(sessions, excluded={"arc_1"})
        assert all(not rows for rows in grouped.values())


class TestCap:
    def test_prefers_distinct_arcs_then_fills_the_remainder(self) -> None:
        grouped: dict[str, list[TimelinePrefix]] = {bucket: [] for bucket in BUCKETS}
        grouped["short"] = [
            _prefix(arc_id="arc_1", turn_ordinal=2),
            _prefix(arc_id="arc_1", turn_ordinal=6),
            _prefix(arc_id="arc_2", turn_ordinal=2),
        ]
        chosen = _cap(grouped, 2, random.Random(0))
        assert {prefix.arc_id for prefix in chosen} == {"arc_1", "arc_2"}
        chosen = _cap(grouped, 3, random.Random(0))
        assert len(chosen) == 3


class TestScoreSample:
    def test_unparseable_reply_is_a_parse_failure_not_a_drift(self) -> None:
        result = score_sample("base", _prefix(inherited=("now", "-1w")), "I hear you.")
        assert not result.parsed
        assert result.recall == 0.0
        assert result.missing == ["-1w", "now"]
        assert not result.collapsed

    def test_missing_reply_counts_as_a_parse_failure(self) -> None:
        assert not score_sample("base", _prefix(), None).parsed

    def test_retained_anchors_pass(self) -> None:
        reply = '{"tl": "now: still here; -1w: unchanged"}\nThat sounds hard.'
        result = score_sample("sft", _prefix(inherited=("now", "-1w")), reply)
        assert result.parsed
        assert result.missing == []
        assert result.recall == 1.0
        assert not result.collapsed

    def test_dropped_anchor_is_reported(self) -> None:
        reply = '{"tl": "now: still here"}\nThat sounds hard.'
        result = score_sample("sft", _prefix(inherited=("now", "-1w")), reply)
        assert result.parsed
        assert result.missing == ["-1w"]
        assert result.recall == 0.5

    def test_full_anchor_loss_collapses(self) -> None:
        reply = '{"tl": "+2y: a new thing"}\nThat sounds hard.'
        result = score_sample("sft", _prefix(inherited=("now", "-1w")), reply)
        assert result.collapsed
        assert result.invented == ["+2y"]


class TestBucketSummary:
    def test_drift_rate_ignores_unparsed_samples(self) -> None:
        summary = BucketSummary(arm="sft", bucket="long")
        summary.add(score_sample("sft", _prefix(inherited=("now", "-1w")), "not a ledger"))
        summary.add(score_sample("sft", _prefix(inherited=("now", "-1w")), '{"tl": "now: x; -1w: y"}'))
        summary.add(score_sample("sft", _prefix(inherited=("now", "-1w")), '{"tl": "now: x"}'))
        assert summary.samples == 3
        assert summary.parsed == 2
        assert summary.drift == 1
        assert summary.drift_rate == 0.5
        assert summary.missing_total == 1
        assert summary.mean_recall == 0.75

    def test_empty_summary_reports_zero(self) -> None:
        summary = BucketSummary(arm="base", bucket="short")
        assert summary.drift_rate == 0.0
        assert summary.mean_recall == 0.0


class TestProbeSettings:
    def test_carries_the_sampling_contract(self) -> None:
        settings = ProbeSettings(samples=2, temperature=0.7, concurrency=6, max_tokens=1400)
        assert settings.samples == 2
        assert settings.max_tokens == 1400
