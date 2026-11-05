"""Tests for the timeline anchor accounting.

The probe's numbers are only meaningful if its anchor extraction matches the
generator's gate, and if a drifted ledger is reported as drifted rather than as
a parse failure. These tests pin both.
"""

from __future__ import annotations

import re
from pathlib import Path

from training.timeline_gate import (
    TL_ANCHOR_RE,
    accumulate,
    check_drift,
    ledger_anchors,
    parse_ledger,
    tl_anchor_keys,
)

GENERATOR = Path(__file__).resolve().parents[1] / "generate_arc_corpus.py"


class TestAnchorParity:
    def test_pattern_matches_the_generator_gate(self) -> None:
        source = GENERATOR.read_text(encoding="utf-8")
        match = re.search(r'_TL_ANCHOR_RE = re\.compile\(r"([^"]*)"', source)
        assert match, "generator no longer defines _TL_ANCHOR_RE as a raw literal"
        assert match.group(1) == TL_ANCHOR_RE.pattern


class TestTlAnchorKeys:
    def test_reads_anchor_before_its_colon(self) -> None:
        assert tl_anchor_keys("now: questions the work") == {"now"}

    def test_reads_every_delimited_entry(self) -> None:
        tl = "now: presenting; -6w: left the job, -2y: divorce"
        assert tl_anchor_keys(tl) == {"now", "-6w", "-2y"}

    def test_ignores_colons_inside_descriptions(self) -> None:
        tl = "-3w: said 'it started: slowly' then stopped"
        assert tl_anchor_keys(tl) == {"-3w"}

    def test_accepts_plus_and_single_letter_units(self) -> None:
        assert tl_anchor_keys("+1y: retirement; -4d: argument") == {"+1y", "-4d"}

    def test_empty_input_has_no_anchors(self) -> None:
        assert tl_anchor_keys("") == set()


class TestLedgerAnchors:
    def test_reads_tl_field(self) -> None:
        assert ledger_anchors({"tl": "now: x; -1m: y"}) == {"now", "-1m"}

    def test_missing_or_non_string_tl_is_empty(self) -> None:
        assert ledger_anchors({}) == set()
        assert ledger_anchors({"tl": None}) == set()


class TestParseLedger:
    def test_parses_ledger_before_the_spoken_reply(self) -> None:
        ledger = parse_ledger('{"tl": "now: x"}\nThat sounds exhausting.')
        assert ledger == {"tl": "now: x"}

    def test_tolerates_leading_whitespace(self) -> None:
        assert parse_ledger('  \n{"tl": "now: x"}') == {"tl": "now: x"}

    def test_rejects_turn_without_leading_object(self) -> None:
        assert parse_ledger("That sounds exhausting.") is None
        assert parse_ledger(None) is None
        assert parse_ledger("") is None

    def test_rejects_truncated_json(self) -> None:
        assert parse_ledger('{"tl": "now: x"') is None

    def test_rejects_empty_object(self) -> None:
        assert parse_ledger("{}\nsome reply") is None


class TestDriftCheck:
    def test_retaining_every_anchor_holds(self) -> None:
        check = check_drift({"now", "-6w"}, {"now", "-6w", "+1y"})
        assert check.holds
        assert check.missing == frozenset()
        assert check.invented == frozenset({"+1y"})
        assert check.recall == 1.0

    def test_dropping_one_anchor_is_drift(self) -> None:
        check = check_drift({"now", "-6w"}, {"now"})
        assert not check.holds
        assert check.missing == frozenset({"-6w"})
        assert check.recall == 0.5
        assert not check.collapsed

    def test_dropping_everything_collapses(self) -> None:
        check = check_drift({"now", "-6w"}, {"+2y"})
        assert not check.holds
        assert check.collapsed
        assert check.recall == 0.0

    def test_nothing_inherited_always_holds(self) -> None:
        check = check_drift(set(), {"now"})
        assert check.holds
        assert check.recall == 1.0


class TestAccumulate:
    def test_unions_anchors_across_ledgers(self) -> None:
        ledgers = [{"tl": "now: x"}, {"tl": "now: x; -6w: y"}, {"tl": "-6w: y; now: x; +1y: z"}]
        assert accumulate(ledgers) == {"now", "-6w", "+1y"}

    def test_skips_ledgers_without_tl(self) -> None:
        assert accumulate([{"tl": "now: x"}, {"dx": "no timeline"}]) == {"now"}
