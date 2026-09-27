"""Ledger traceability gate tests (ai/training/ledger_gate.py).

Covers the number/anchor/quote checks on ledger fields, the spoken
specificity patterns on therapist lines, carry-forward evidence seeding
(prior client lines, plan gap, prior ledger state and tl anchors), and the
wiring into ``run_mechanical_gates``.
"""

from __future__ import annotations

import json
from typing import Any

from training.generate_arc_corpus import parse_transcript, run_mechanical_gates
from training.ledger_gate import ledger_gate_failures, number_matches

_BASE_LEDGER = {
    "dx": "generalized anxiety",
    "def": "worry she cannot switch off",
    "soma": "none",
    "risk": "none",
    "hx": "no history items",
    "onset": "not stated",
    "track": "steady",
    "tx": "none yet",
    "tl": "now: session opened",
}

TWO_SEPARATORS = 2  # 3 capped hit labels -> 2 "; " joins


def _ledger(**overrides: str) -> dict[str, Any]:
    return {**_BASE_LEDGER, **overrides}


def _transcript(lines: list[tuple[str, object]]) -> dict[str, Any]:
    """lines: (mark, content) where mark is C, T, or T|THINK."""
    text = "\n".join(f"[{mark}] {json.dumps(content) if mark == 'T|THINK' else content}" for mark, content in lines)
    return parse_transcript(text)


def _session_plan(n: int, turns: int, gap_before: str = "") -> dict[str, Any]:
    session: dict[str, Any] = {"n": n, "turns": turns}
    if gap_before:
        session["gap_before"] = gap_before
    return {"sessions": [session]}


# --- number matching -------------------------------------------------------


def test_digit_matches_digit_evidence() -> None:
    assert number_matches("3", {"3"})


def test_digit_matches_word_evidence() -> None:
    assert number_matches("3", {"three"})


def test_word_matches_digit_evidence() -> None:
    assert number_matches("three", {"3"})


def test_hyphenated_compound_matches_component_words() -> None:
    assert number_matches("twenty-one", {"twenty", "one"})


def test_digit_compound_matches_component_words() -> None:
    assert number_matches("21", {"twenty", "one"})


def test_zero_is_uncheckable() -> None:
    assert number_matches("0", set())
    assert number_matches("00", set())


def test_large_numbers_are_uncheckable() -> None:
    assert number_matches("1200", set())
    assert number_matches("2026", set())


def test_absent_number_fails() -> None:
    assert not number_matches("7", {"3"})


def test_one_stays_locked() -> None:
    # "one"/"zero" are excluded from the vocabulary on purpose: "the one
    # thing" is not a quantity, and "1" must not be satisfied by it.
    assert not number_matches("1", {"one"})
    assert not number_matches("one", {"1"})


def test_plural_evidence_satisfies_singular() -> None:
    # plural client speech ("tens") grounds the singular ledger form ("10 mg").
    assert number_matches("10", {"tens"})
    assert number_matches("three", {"threes"})


# --- ledger field anchors ----------------------------------------------------


def _one_session(parsed: dict[str, Any]) -> list[str]:
    return ledger_gate_failures(_session_plan(1, 2), 1, parsed, [])


def test_past_anchor_traceable() -> None:
    parsed = _transcript(
        [
            ("C", "It has been three months since the move."),
            ("T|THINK", _ledger(tl="-3m: last move (told)")),
            ("T", "How did the neighborhood change for you?"),
        ]
    )
    assert _one_session(parsed) == []


def test_past_anchor_untraceable() -> None:
    parsed = _transcript(
        [
            ("C", "The move was hard."),
            ("T|THINK", _ledger(tl="-3m: last move (told)")),
            ("T", "What did the move cost you?"),
        ]
    )
    failures = _one_session(parsed)
    assert len(failures) == 1
    assert failures[0].startswith("ledger_traceability: 1 untraceable ledger items")
    assert "anchor '-3m'" in failures[0]


def test_future_anchor_needs_unit_only() -> None:
    parsed = _transcript(
        [
            ("C", "I want a year of stability before deciding anything."),
            ("T|THINK", _ledger(tl="+1y: stability window (claim)")),
            ("T", "What would stability look like?"),
        ]
    )
    assert _one_session(parsed) == []


def test_now_anchor_is_free() -> None:
    parsed = _transcript(
        [
            ("C", "It has been rough."),
            ("T|THINK", _ledger(tl="now: session opened")),
            ("T", "Where does it sit heaviest?"),
        ]
    )
    assert _one_session(parsed) == []


def test_anchor_digit_not_double_reported() -> None:
    parsed = _transcript(
        [
            ("C", "The move was hard."),
            ("T|THINK", _ledger(tl="-3m: last move (told)")),
            ("T", "What did the move cost you?"),
        ]
    )
    hits = _one_session(parsed)[0].split(": ", 1)[1]
    assert hits.count("number") == 0


def test_session_number_bookkeeping_is_exempt() -> None:
    # "session 1 begins" is writer bookkeeping, not a client fact.
    parsed = _transcript(
        [
            ("C", "It has been a hard stretch."),
            ("T|THINK", _ledger(tl="now: session 1 begins (told)")),
            ("T", "Where shall we start?"),
        ]
    )
    assert _one_session(parsed) == []


# --- quotes ------------------------------------------------------------------


def test_fabricated_contraction_quote_flagged() -> None:
    parsed = _transcript(
        [
            ("C", "I never said that."),
            ("T|THINK", _ledger(tl="now: client quit claim 'I'll stop everything' (told)")),
            ("T", "Did you say that?"),
        ]
    )
    failures = _one_session(parsed)
    assert len(failures) == 1
    assert "quote" in failures[0]


def test_verbatim_quote_passes() -> None:
    parsed = _transcript(
        [
            ("C", "He said I'll stop everything for him, and maybe I will."),
            ("T|THINK", _ledger(tl="now: client quote 'I'll stop everything' (told)")),
            ("T", "What stops you?"),
        ]
    )
    assert _one_session(parsed) == []


def test_short_quotes_are_exempt() -> None:
    parsed = _transcript(
        [
            ("C", "It was a fight."),
            ("T|THINK", _ledger(tl="now: client said 'go away' (told)")),
            ("T", "What happened next?"),
        ]
    )
    assert _one_session(parsed) == []


def test_in_word_apostrophe_is_not_a_quote() -> None:
    parsed = _transcript(
        [
            ("C", "It is my mother's house."),
            ("T|THINK", _ledger(dx="client's mother is the stressor")),
            ("T", "What does the house hold for you?"),
        ]
    )
    assert _one_session(parsed) == []


def test_quote_overlap_threshold() -> None:
    # grounding is >= half the quote's tokens in evidence, not verbatim
    # containment — rhetorical paraphrase stays the auditor's domain.
    client = "The harbor seals gather every winter morning near the pier."
    parsed_pass = _transcript(
        [
            ("C", client),
            ("T|THINK", _ledger(tl="now: client said 'seals gather storm fright' (told)")),
            ("T", "What do the seals mean to you?"),
        ]
    )
    assert _one_session(parsed_pass) == []
    parsed_fail = _transcript(
        [
            ("C", client),
            ("T|THINK", _ledger(tl="now: client said 'seals storm fright panic' (told)")),
            ("T", "What do the seals mean to you?"),
        ]
    )
    failures = _one_session(parsed_fail)
    assert len(failures) == 1
    assert "quote" in failures[0]


def test_quote_only_told_claim_entries() -> None:
    # quotes on (untold) entries mirror plan background, not client speech.
    parsed = _transcript(
        [
            ("C", "It has been a long winter."),
            ("T|THINK", _ledger(tl="now: moved out 'not a person who does anything' (untold)")),
            ("T", "What did the move change?"),
        ]
    )
    assert _one_session(parsed) == []


def test_non_tl_quote_not_checked() -> None:
    parsed = _transcript(
        [
            ("C", "Things feel flat."),
            ("T|THINK", _ledger(dx="client presents 'not a person who does anything'")),
            ("T", "What does flat feel like?"),
        ]
    )
    assert _one_session(parsed) == []


def test_tl_entry_splitting() -> None:
    parsed = _transcript(
        [
            ("C", "It has been two months since I moved out."),
            (
                "T|THINK",
                _ledger(tl="now: opened 'nothing ever changes' (told), -2m: moved out 'nothing ever changes' (untold)"),
            ),
            ("T", "What changed with the move?"),
        ]
    )
    failures = _one_session(parsed)
    # only the (told) entry's quote is checked; the (untold) one is skipped
    assert len(failures) == 1
    assert failures[0].count("quote") == 1


# --- plan evidence seeding ----------------------------------------------------


def test_tl_mirrors_plan_timeline() -> None:
    plan = _session_plan(1, 2)
    plan["timeline"] = [{"anchor": "-4y", "event": "moved into Devon's apartment", "provenance": "untold"}]
    parsed = _transcript(
        [
            ("C", "I have been circling the apartment in my head."),
            ("T|THINK", _ledger(tl="-4y: moved into the apartment (untold)")),
            ("T", "What does the apartment hold for you?"),
        ]
    )
    # the ledger mirrors the plan's timeline without the client naming it
    assert ledger_gate_failures(plan, 1, parsed, []) == []
    # strip the timeline and the anchor becomes invention
    failures = ledger_gate_failures(_session_plan(1, 2), 1, parsed, [])
    assert len(failures) == 1
    assert "anchor '-4y'" in failures[0]


def test_hx_numbers_from_plan_notes() -> None:
    plan = _session_plan(1, 2)
    plan["client"] = {"name": "Mara", "notes": "son Diego, 12, lives at home"}
    parsed = _transcript(
        [
            ("C", "It has been a hard stretch."),
            ("T|THINK", _ledger(hx="son Diego is 12 (told)")),
            ("T", "How is Diego doing?"),
        ]
    )
    assert ledger_gate_failures(plan, 1, parsed, []) == []
    failures = ledger_gate_failures(_session_plan(1, 2), 1, parsed, [])
    assert len(failures) == 1
    assert "number '12'" in failures[0]


# --- spoken specificity ------------------------------------------------------


def _client_then_therapist(client: str, therapist: str, ledger: dict[str, Any] | None = None) -> list[str]:
    lines: list[tuple[str, object]] = [("C", client)]
    if ledger is not None:
        lines.append(("T|THINK", ledger))
    lines.append(("T", therapist))
    return _one_session(_transcript(lines))


def test_invented_duration_flagged() -> None:
    failures = _client_then_therapist("Work is bad.", "That was three weeks ago.")
    assert len(failures) == 1
    assert failures[0].startswith("spoken_specificity: 1 untraceable speech items")
    assert "three weeks ago" in failures[0]


def test_client_backed_duration_passes() -> None:
    failures = _client_then_therapist("It started three weeks ago.", "Three weeks ago, before the job change.")
    assert failures == []


def test_frequency_flagged() -> None:
    failures = _client_then_therapist("I barely sleep.", "You mean four nights a week?")
    assert len(failures) == 1
    assert "four nights a week" in failures[0]


def test_future_span_is_exempt() -> None:
    failures = _client_then_therapist("I am scared.", "In three weeks we can review this.")
    assert failures == []


def test_clock_time_flagged() -> None:
    failures = _client_then_therapist("We used to talk more.", "Back then you called at 3:00.")
    assert len(failures) == 1


def test_oclock_backed_by_client() -> None:
    failures = _client_then_therapist("He called at ten.", "At ten o'clock, right before bed.")
    assert failures == []


def test_in_session_backref_is_exempt() -> None:
    # elapsed time inside the session itself is not client-history fabrication
    failures = _client_then_therapist("Work is bad.", "Twenty minutes ago you told me the job was the whole story.")
    assert failures == []


def test_minute_and_hour_durations_are_in_session_scale() -> None:
    # Minutes/hours date the session's own hour, never client history —
    # exempt even without a back-reference verb (arc_0033 corpus pattern).
    failures = _client_then_therapist("Work is bad.", "Ten minutes ago it was clean the whole time.")
    assert failures == []
    failures = _client_then_therapist("Work is bad.", "Six hours ago you arrived.")
    assert failures == []


# --- carry-forward evidence ---------------------------------------------------


def test_prior_client_lines_are_evidence() -> None:
    # "seven" appears nowhere in the plan JSON; only prior client speech
    # grounds the session-2 ledger's reuse of it.
    prior = [{"turns": [{"role": "client", "content": "I have seven jobs."}]}]
    parsed = _transcript(
        [
            ("C", "Both are wearing me down."),
            ("T|THINK", _ledger(tl="now: seven jobs confirmed (told)")),
            ("T", "Which one is heavier?"),
        ]
    )
    assert ledger_gate_failures(_session_plan(2, 2, "two days later"), 2, parsed, prior) == []


def test_prior_ledger_state_seeds_evidence() -> None:
    # The carry-forward rule repeats prior ledger entries verbatim, so the
    # prior final ledger's numbers ground the next session's ledger.
    prior = [
        {
            "turns": [
                {"role": "client", "content": "I need help."},
                {
                    "role": "therapist",
                    "content": "Let us start there.",
                    "ledger": _ledger(tl="now: Daniel is 28 (told)", hx="three workups at the clinic (told)"),
                },
            ]
        }
    ]
    parsed = _transcript(
        [
            ("C", "It has kept going."),
            ("T|THINK", _ledger(tl="now: Daniel still 28 (told), three workups repeated (told)")),
            ("T", "How has that tracked?"),
        ]
    )
    assert ledger_gate_failures(_session_plan(2, 2), 2, parsed, prior) == []
    # a number absent from prior ledger state AND the plan is still invented.
    parsed_bad = _transcript(
        [
            ("C", "It has kept going."),
            ("T|THINK", _ledger(tl="now: nine workups repeated (told)")),
            ("T", "How has that tracked?"),
        ]
    )
    failures = ledger_gate_failures(_session_plan(2, 2), 2, parsed_bad, prior)
    assert len(failures) == 1
    assert "number 'nine'" in failures[0]


def test_plan_gap_seeds_evidence() -> None:
    prior = [{"turns": [{"role": "client", "content": "I have two jobs."}]}]
    parsed = _transcript(
        [
            ("C", "Both are wearing me down."),
            ("T|THINK", _ledger(tl="-3w: jobs doubled up (told)")),
            ("T", "What stacked up?"),
        ]
    )
    # gap stated in the plan gives the writer the "three weeks" anchor.
    assert ledger_gate_failures(_session_plan(2, 2, "three weeks later"), 2, parsed, prior) == []
    # with no stated gap the anchor is invented.
    failures = ledger_gate_failures(_session_plan(2, 2), 2, parsed, prior)
    assert len(failures) == 1
    assert "anchor '-3w'" in failures[0]


def test_prior_tl_anchors_seed_evidence() -> None:
    prior = [
        {
            "turns": [
                {"role": "client", "content": "I need help."},
                {
                    "role": "therapist",
                    "content": "Let us start there.",
                    "ledger": _ledger(tl="-2y: drinking resumed (told)"),
                },
            ]
        }
    ]
    parsed = _transcript(
        [
            ("C", "It has kept going."),
            ("T|THINK", _ledger(tl="-2y: drinking continued (told)")),
            ("T", "How has the drinking tracked?"),
        ]
    )
    assert ledger_gate_failures(_session_plan(2, 2), 2, parsed, prior) == []
    parsed_bad = _transcript(
        [
            ("C", "It has kept going."),
            ("T|THINK", _ledger(tl="-3w: drinking continued (told)")),
            ("T", "How has the drinking tracked?"),
        ]
    )
    failures = ledger_gate_failures(_session_plan(2, 2), 2, parsed_bad, prior)
    assert len(failures) == 1
    assert "anchor '-3w'" in failures[0]


# --- failure shaping ----------------------------------------------------------


def test_failed_parse_blocks_are_skipped() -> None:
    parsed = parse_transcript("[C] It has been rough.\n[T|THINK] not json {\n[T] Tell me more.")
    assert _one_session(parsed) == []


def test_non_dict_ledger_is_skipped() -> None:
    # A quoted string parses as valid JSON (ok=True) but the "ledger" is not
    # a mapping; the gate must not crash on it (ledger_fields flags it).
    parsed = _transcript(
        [
            ("C", "It has been rough."),
            ("T|THINK", "not json {"),
            ("T", "Tell me more."),
        ]
    )
    assert _one_session(parsed) == []


def test_hit_cap_and_multiple_heads() -> None:
    parsed = _transcript(
        [
            ("C", "It has been rough."),
            ("T|THINK", _ledger(tl="-3m: work stress (told)", hx="two hospital stays (told)")),
            ("T", "Four hospital stays? That was two weeks back."),
            ("T|THINK", _ledger(tl="-2m: hospital visit (told)")),
            ("T", "Tell me about the stays."),
            ("C", "There was one, once."),
            ("T|THINK", _ledger(tl="-1m: second visit (told)")),
            ("T", "What happened there?"),
        ]
    )
    failures = _one_session(parsed)
    heads = [f.split(":", 1)[0] for f in failures]
    assert heads == ["ledger_traceability", "spoken_specificity"]
    detail = failures[0].split(": ", 1)[1]
    assert detail.startswith("4 untraceable ledger items: ")
    # cap: exactly 3 hit labels after the count
    assert detail.count("; ") == TWO_SEPARATORS


# --- run_mechanical_gates wiring ----------------------------------------------


def test_generator_gate_wiring_flags_fabrication() -> None:
    plan = _session_plan(1, 1)
    parsed = _transcript(
        [
            ("C", "It has been hard lately."),
            (
                "T|THINK",
                _ledger(tl="-3m: workload change (told)"),
            ),
            ("T", "Since when has the workload felt heaviest?"),
        ]
    )
    failures = run_mechanical_gates(plan, 1, parsed, [])
    traceability = [f for f in failures if f.startswith("ledger_traceability")]
    assert len(traceability) == 1


def test_generator_gate_wiring_clean_session() -> None:
    plan = _session_plan(1, 1)
    parsed = _transcript(
        [
            ("C", "It has been hard lately."),
            ("T|THINK", _ledger(tl="now: workload stress (told)")),
            ("T", "Where does it press hardest?"),
        ]
    )
    assert run_mechanical_gates(plan, 1, parsed, []) == []


# --- beat quote containment normalization -------------------------------------


def _beats_plan(n: int, revision: str) -> dict[str, Any]:
    plan = _session_plan(n, 2)
    plan["beats"] = [
        {
            "type": "misstatement",
            "revise_session": n,
            "revise_turn": 2,
            "revision": f"session-{n} revision: '{revision}'",
        }
    ]
    return plan


def test_beat_quote_em_dash_variant_passes() -> None:
    # arc_0024 shape: plan quotes the client with a spaced hyphen, the writer
    # renders the same words with an em-dash — typography, not paraphrase.
    plan = _beats_plan(2, "okay - most nights after the shift. Sometimes before I clock in.")
    parsed = _transcript(
        [
            ("C", "Okay\u2014most nights after the shift. Sometimes before I clock in."),
            ("T|THINK", _ledger()),
            ("T", "That is a different picture than an hour ago."),
        ]
    )
    assert run_mechanical_gates(plan, 2, parsed, []) == []


def test_beat_quote_curly_apostrophe_and_en_dash_pass() -> None:
    plan = _beats_plan(2, "I drive - ten minutes - and then I wait.")
    parsed = _transcript(
        [
            ("C", "I drive\u2013ten minutes\u2013and then I wait."),
            ("T|THINK", _ledger()),
            ("T", "And the waiting is the hard part?"),
        ]
    )
    assert run_mechanical_gates(plan, 2, parsed, []) == []


def test_beat_quote_paraphrase_still_fails() -> None:
    plan = _beats_plan(2, "okay - most nights after the shift. Sometimes before I clock in.")
    parsed = _transcript(
        [
            ("C", "I work most nights, honestly. Even before I clock in."),
            ("T|THINK", _ledger()),
            ("T", "How long has that been true?"),
        ]
    )
    failures = run_mechanical_gates(plan, 2, parsed, [])
    assert any(f.startswith("beat_content") for f in failures)
