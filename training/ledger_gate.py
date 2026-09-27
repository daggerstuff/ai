#!/usr/bin/env python3
"""Deterministic ledger traceability gates (ARC_CORPUS_SPEC §4.2 + §7.4).

Mechanical companions to the K3 auditor pass (§7). They catch the fabrication
classes decidable by text containment — invented numbers, durations, clock
times, and wholly ungrounded client quotations — while leaving paraphrase-
level fabrications (invented nouns, somatic cues, status facts, rhetorical
quotes) to the auditor.

Evidence model (what the writer legitimately draws on):

  speech — every client line up to and including the one preceding the turn
    under check, carry-forward client lines from prior sessions, the plan's
    stated inter-session gap (session 2+), and durations established by prior
    sessions' final ``tl`` anchors. Therapist spoken lines must trace to this
    alone: plan text is not something the therapist may say to the client.

  plan evidence — the plan's fact-schedule text (client profile, timeline,
    surface/real subject, beats, ending, per-session focus and gap text, with
    timeline anchors expanded to number + unit words), plus everything the
    carry-forward rule lets the writer reuse in session 2+: prior sessions'
    client lines and their final ledger state (§5.1 — the next ledger repeats
    and extends prior entries verbatim). The ledger mirrors this schedule, so
    ledger numbers, anchors, and grounded quotations trace to it.

A number is traceable when the evidence contains it after normalization
(digit and word forms are interchangeable, hyphenated compounds split, plural
evidence satisfies the singular). A quotation of 3+ tokens inside a tl entry
tagged (told)/(claim) is grounded when at least half of its tokens appear in
the evidence — full paraphrase-level rhetoric is the auditor's domain.
Checks are token-class only — common nouns are deliberately ungated (K3's
job) so the false-positive rate on accepted-arc replays stays near zero.
"""

from __future__ import annotations

import json
import re
from itertools import pairwise
from typing import Any

# Number vocabulary. "one"/"zero" are excluded from target scans: "the one
# thing" is not a quantity, and skipping them costs almost nothing (K3 still
# reads every turn).
_ONES: dict[int, str] = {
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}
_WORD_TO_NUM: dict[str, int] = {w: n for n, w in _ONES.items()}
# Ones-digit word forms for compound components only ("twenty-one"). "one" is
# excluded from the standalone scan vocabulary but is unambiguous inside a
# compound, so it lives here instead of _ONES.
_UNIT_MAX_DIGIT = 9
_UNIT_WORD_FOR: dict[int, str] = {1: "one", **{n: w for n, w in _ONES.items() if n <= _UNIT_MAX_DIGIT}}
_UNIT_NUM_FOR: dict[str, int] = {w: n for n, w in _UNIT_WORD_FOR.items()}
# Longest first so the alternation prefers the longest word at a position.
_WORD_ALT = "|".join(sorted(_WORD_TO_NUM, key=len, reverse=True))

_DIGIT_RE = re.compile(r"\b\d+\b")
_WORD_NUM_RE = re.compile(rf"\b({_WORD_ALT})\b", re.IGNORECASE)
_ANCHOR_RE = re.compile(r"(?<![\w.])([-+]?\d+)\s*([ymwd])\b", re.IGNORECASE)
_PROVENANCE_RE = re.compile(r"\((?:told|untold|claim|revision)\b[^)]*\)", re.IGNORECASE)
# Quote spans: single-quoted (must start after space/start so in-word
# apostrophes like client's never open a span) with word-internal
# contractions allowed inside, or any double-quoted span.
_QUOTE_RE = re.compile(
    r"(?:(?<=^)|(?<=\s))"
    r"'((?:[^']|'(?=[a-z]))*?)'(?![a-z])"
    r"|\"([^\"]+)\""
)
_TOKEN_RE = re.compile(r"[a-z0-9']+")

_UNIT_FOR_ANCHOR = {"w": "week", "d": "day", "m": "month", "y": "year"}

_NUM_ALT = r"\d+|" + _WORD_ALT
_SESSION_NUM_RE = re.compile(r"\bsessions?\s+\d+\b", re.IGNORECASE)
# A spoken duration can legally reference the session's own elapsed time
# ("Twenty minutes ago you told me...") — exempt when the sentence carries an
# in-session back-reference verb.
_IN_SESSION_REF_RE = re.compile(
    r"\b(?:you|we)\s+(?:told|said|say|asked|ask|wrote|mentioned|talked|agreed|noticed|brought)\b",
    re.IGNORECASE,
)
_SENT_SPLIT_RE = re.compile(r"[.!?]")
# Minute/hour durations are in-session scale ("ten minutes ago" = earlier in
# this hour, corpus-wide) and unrepresentable as client history — exempt.
# Day/week/month/year/night units date client history, where invented
# specificity is fabrication (K3-flagged class).
_SPOKEN_DUR_RE = re.compile(
    rf"\b({_NUM_ALT})\s+"
    r"(days?|weeks?|months?|years?|nights?|mornings?|evenings?)"
    r"\s+(ago|back|earlier)\b",
    re.IGNORECASE,
)
_SPOKEN_FREQ_RE = re.compile(
    rf"\b({_NUM_ALT})\s+\S+\s+(?:a|per)\s+(?:day|week|month|year)s?\b",
    re.IGNORECASE,
)
_SPOKEN_CLOCK_RE = re.compile(
    rf"\b({_NUM_ALT})\s+o'?clock\b|\b(\d{{1,2}}):(\d{{2}})\b",
    re.IGNORECASE,
)

LEDGER_NUMBER_FIELDS = ("tl", "hx", "onset", "soma")

# tl entries are "<anchor>: <event> (<provenance>)" segments; entries split at
# anchor-colon boundaries (mirrors the generator's own tl parsing).
_TL_ENTRY_RE = re.compile(r"(?:^|(?<=[;,]))\s*(now|[-+]?\d+\s*[ymwd])\s*:", re.IGNORECASE)
_TL_TOLD_CLAIM_RE = re.compile(r"\((?:told|claim)\b", re.IGNORECASE)

# Quotation grounding: >= half the quote's tokens must exist in the evidence.
# Corpus calibration (60-arc replay): accepted told/claim quotes are 99.2%
# >= 0.75 overlap; HR quotes never fall below 0.5. Below 0.5 a quote is
# invention, not paraphrase.
_MIN_QUOTE_OVERLAP = 0.5
_MIN_QUOTE_TOKENS = 3  # shorter quotes are too false-positive-prone
_PAIR_LEN = 2  # "twenty one" -> two word parts
_MIN_COMPOUND_NUMBER = 21  # tens-compounds: "twenty-one"
_MAX_COMPOUND_NUMBER = 99

# Plan fields whose text the writer is legitimately given (the fact schedule
# the arc is built from). Metadata (ids, titles, era_jitter seeds, turn
# counts) is excluded on purpose.
_PLAN_FIELDS = ("client", "timeline", "surface_subject", "real_subject", "beats", "ending")


def _num_to_words(n: int) -> list[str]:
    """Word components that may stand for ``n`` in client text.

    ``21`` -> ``["twenty", "one"]`` (evidence "twenty-one" or "twenty one"
    tokenizes to those two words). ``21`` is also satisfied by the literal
    digit "21" — callers check that separately.
    """
    if n in _ONES:
        return [_ONES[n]]
    if _MIN_COMPOUND_NUMBER <= n <= _MAX_COMPOUND_NUMBER and n % 10 and (n // 10) * 10 in _ONES:
        return [_ONES[(n // 10) * 10], _UNIT_WORD_FOR[n % 10]]
    return []


def _word_to_num(parts: list[str]) -> int | None:
    if len(parts) == 1:
        return _WORD_TO_NUM.get(parts[0])
    if len(parts) == _PAIR_LEN:
        tens = _WORD_TO_NUM.get(parts[0])
        ones = _WORD_TO_NUM.get(parts[1])
        if ones is None:
            ones = _UNIT_NUM_FOR.get(parts[1])
        if tens is not None and ones is not None and tens % 10 == 0:
            return tens + ones
    return None


def number_matches(token: str, evidence: set[str]) -> bool:
    """True when ``token`` (digit or number word) is traceable to ``evidence``.

    Digit and word forms are interchangeable, and plural evidence ("tens",
    "weeks") satisfies the singular ("ten")."""

    def _has(word: str) -> bool:
        return word in evidence or word + "s" in evidence

    parts = [p for p in re.split(r"[-\s]+", token.strip().lower()) if p]
    if not parts:
        return True
    if len(parts) == 1 and parts[0].isdigit():
        n = int(parts[0])
        # Zero and anything past the tens-compounds are uncheckable: 0 has no
        # word form ("3:00" minutes), and large numbers tokenize differently
        # than their client-text spellings ("1,200" vs "twelve hundred").
        if n == 0 or n > _MAX_COMPOUND_NUMBER:
            return True
        if parts[0] in evidence:
            return True
        words = _num_to_words(n)
        return bool(words) and all(_has(w) for w in words)
    word_n = _word_to_num(parts)
    if word_n is not None and str(word_n) in evidence:
        return True
    return all(_has(p) for p in parts)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _unit_in_evidence(unit: str, evidence: set[str]) -> bool:
    stem = _UNIT_FOR_ANCHOR[unit.lower()]
    return any(tok.startswith(stem) for tok in evidence)


def _anchor_ok(anchor_num: str, unit: str, evidence: set[str]) -> bool:
    """Anchors are relative-time shorthand: "-3w" needs the number AND the
    unit family in the evidence; a future anchor ("+1y") only needs the unit
    ("a year from now" states no digit)."""
    if anchor_num.startswith("+"):
        return _unit_in_evidence(unit, evidence)
    return number_matches(anchor_num, evidence) and _unit_in_evidence(unit, evidence)


def _quotes_in(text: str) -> list[str]:
    return [a if a is not None else b for a, b in _QUOTE_RE.findall(text)]


def _quote_grounded(quote: str, evidence_set: set[str]) -> bool:
    span = _tokenize(quote)
    if len(span) < _MIN_QUOTE_TOKENS:
        return True
    shared = sum(1 for tok in span if tok in evidence_set)
    return shared / len(span) >= _MIN_QUOTE_OVERLAP


class _Evidence:
    """Membership set of tokenized evidence text."""

    __slots__ = ("set",)

    def __init__(self) -> None:
        self.set: set[str] = set()

    def add(self, text: str) -> None:
        self.set.update(_tokenize(text))


def _plan_tokens(plan: dict[str, Any]) -> str:
    """The plan's fact-schedule text, which the ledger may mirror verbatim."""
    parts = [json.dumps(plan.get(f) or "", default=str) for f in _PLAN_FIELDS]
    for s in plan.get("sessions") or []:
        parts.append(str(s.get("focus") or ""))
        parts.append(str(s.get("gap_before") or ""))
    return " ".join(parts)


def _expand_anchor_tokens(text: str) -> list[str]:
    """Relative-time anchors ("-18y") flattened to checkable words."""
    out: list[str] = []
    for m in _ANCHOR_RE.finditer(text):
        num, unit = m.group(1), m.group(2)
        stem = _UNIT_FOR_ANCHOR[unit.lower()]
        if num.startswith("+"):
            out.extend((stem, stem + "s"))
            continue
        n = abs(int(num))
        out.append(str(n))
        out.extend(_num_to_words(n))
        out.extend((stem, stem + "s"))
    return out


def _check_field_numbers(field: str, text: str, evidence: _Evidence, hits: list[str], label: str) -> None:
    cleaned = _PROVENANCE_RE.sub(" ", text)
    if field == "tl":
        # "session 1 begins" is writer bookkeeping, not a client fact.
        cleaned = _SESSION_NUM_RE.sub(" ", cleaned)
    for m in _ANCHOR_RE.finditer(cleaned):
        if not _anchor_ok(m.group(1), m.group(2), evidence.set):
            hits.append(f"{label} {field} anchor '{m.group(0).strip()}'")
    # Anchors already got their dedicated check; strip them so the digit scan
    # doesn't double-report the same fabrication.
    cleaned = _ANCHOR_RE.sub(" ", cleaned)
    for m in _DIGIT_RE.finditer(cleaned):
        if not number_matches(m.group(0), evidence.set):
            hits.append(f"{label} {field} number '{m.group(0)}'")
    for m in _WORD_NUM_RE.finditer(cleaned):
        if not number_matches(m.group(1), evidence.set):
            hits.append(f"{label} {field} number '{m.group(1)}'")


def _check_field_quotes(field: str, text: str, evidence: _Evidence, hits: list[str], label: str) -> None:
    for quote in _quotes_in(text):
        if not _quote_grounded(quote, evidence.set):
            hits.append(f"{label} {field} quote '{quote[:60]}'")


def _sentence_has_in_session_ref(content: str, start: int, end: int) -> bool:
    """True when the sentence containing ``[start, end)`` back-references the
    ongoing exchange ("you told me", "we agreed") — in-session elapsed time."""
    s = 0
    for chunk in _SENT_SPLIT_RE.split(content):
        e = s + len(chunk)
        if start < e and end > s:
            return bool(_IN_SESSION_REF_RE.search(chunk))
        s = e + 1
    return False


def _check_therapist_line(content: str, evidence: _Evidence, hits: list[str], label: str) -> None:
    for m in _SPOKEN_DUR_RE.finditer(content):
        if _sentence_has_in_session_ref(content, m.start(), m.end()):
            continue
        if not number_matches(m.group(1), evidence.set):
            hits.append(f"{label} spoken '{m.group(0)[:40]}'")
    for m in _SPOKEN_FREQ_RE.finditer(content):
        if _sentence_has_in_session_ref(content, m.start(), m.end()):
            continue
        if not number_matches(m.group(1), evidence.set):
            hits.append(f"{label} spoken '{m.group(0)[:40]}'")
    for m in _SPOKEN_CLOCK_RE.finditer(content):
        nums = [m.group(1)] if m.group(1) is not None else [m.group(2), m.group(3)]
        if _sentence_has_in_session_ref(content, m.start(), m.end()):
            continue
        if any(not number_matches(num, evidence.set) for num in nums):
            hits.append(f"{label} spoken '{m.group(0)[:40]}'")


def _prior_ledger_text(prior_sessions: list[dict[str, Any]]) -> str:
    """Text of the most recent prior session's final ledger state fields.

    Carry-over is append-mostly (§4.1): the next session's ledger repeats and
    extends prior entries verbatim, so prior state text grounds later checks.
    """
    for ps in reversed(prior_sessions):
        last = next(
            (t["ledger"] for t in reversed(ps.get("turns") or []) if t.get("ledger")),
            None,
        )
        if last:
            return " ".join(str(last.get(f) or "") for f in LEDGER_NUMBER_FIELDS)
    return ""


def _prior_anchor_tokens(prior_sessions: list[dict[str, Any]]) -> str:
    """Durations established by the most recent prior ledger's tl anchors,
    flattened to client-usable words ("two", "years")."""
    out: list[str] = []
    for ps in reversed(prior_sessions):
        last = next(
            (t["ledger"] for t in reversed(ps.get("turns") or []) if t.get("ledger")),
            None,
        )
        if not last:
            continue
        for m in _ANCHOR_RE.finditer(str(last.get("tl") or "")):
            num, unit = m.group(1), m.group(2)
            if num.startswith("+"):
                out.append(_UNIT_FOR_ANCHOR[unit.lower()])
                out.append(_UNIT_FOR_ANCHOR[unit.lower()] + "s")
                continue
            n = abs(int(num))
            out.append(str(n))
            out.extend(_num_to_words(n))
            out.append(_UNIT_FOR_ANCHOR[unit.lower()])
            out.append(_UNIT_FOR_ANCHOR[unit.lower()] + "s")
        if out:
            break
    return " ".join(out)


def _tl_entries(tl_text: str) -> list[str]:
    """Entries of a tl string, split at anchor-colon boundaries."""
    matches = list(_TL_ENTRY_RE.finditer(tl_text))
    if not matches:
        return [tl_text] if tl_text.strip() else []
    bounds = [m.start(1) for m in matches] + [len(tl_text)]
    return [tl_text[a:b] for a, b in pairwise(bounds)]


def _seed_evidence(
    plan: dict[str, Any], session: dict[str, Any] | None, prior_sessions: list[dict[str, Any]]
) -> tuple[_Evidence, _Evidence]:
    """(speech, plan_ev): the two evidence sets.

    speech: client lines so far, the plan's inter-session gap, and durations
    from prior final tl anchors. Therapist speech must trace to this alone.

    plan_ev: speech plus the plan's fact-schedule text — the ledger mirrors
    that schedule verbatim (§4), so its numbers, anchors, and grounded
    quotes trace to it.
    """
    speech = _Evidence()
    if prior_sessions:
        for ps in prior_sessions:
            for t in ps.get("turns") or []:
                if t.get("role") == "client":
                    speech.add(str(t.get("content") or ""))
        speech.add((session or {}).get("gap_before") or "")
        speech.add(_prior_anchor_tokens(prior_sessions))
    plan_ev = _Evidence()
    plan_text = _plan_tokens(plan)
    plan_ev.add(plan_text)
    plan_ev.add(" ".join(_expand_anchor_tokens(plan_text)))
    if prior_sessions:
        # Carry-forward: session 2+'s ledger repeats the prior session's
        # client lines and final ledger state, so both ground later checks.
        # (Anchors get their word-form expansion too — "-28d" never appears
        # verbatim in prior client speech.)
        for ps in prior_sessions:
            for t in ps.get("turns") or []:
                if t.get("role") == "client":
                    plan_ev.add(str(t.get("content") or ""))
        plan_ev.add(_prior_ledger_text(prior_sessions))
        plan_ev.add(_prior_anchor_tokens(prior_sessions))
    return speech, plan_ev


def _check_ledger_block(ledger: dict[str, Any], label: str, evidence: _Evidence, ledger_hits: list[str]) -> None:
    """Numbers/anchors in the state fields and told/claim quotes on tl
    entries — all against plan-plus-speech evidence."""
    for field in LEDGER_NUMBER_FIELDS:
        text = str(ledger.get(field) or "")
        if field == "soma" and text.strip().lower() == "none":
            continue
        _check_field_numbers(field, text, evidence, ledger_hits, label)
    for entry in _tl_entries(str(ledger.get("tl") or "")):
        if not _TL_TOLD_CLAIM_RE.search(entry):
            continue
        _check_field_quotes("tl", entry, evidence, ledger_hits, label)


def ledger_gate_failures(
    plan: dict[str, Any], n: int, parsed: dict[str, Any], prior_sessions: list[dict[str, Any]]
) -> list[str]:
    """Traceability failures for one generated session.

    Walks the parsed transcript in order, growing both evidence sets with
    each client line. Ledger fields (tl/hx/onset/soma numbers + anchors,
    told/claim quotes on tl entries) trace to the plan's fact schedule plus
    client speech; therapist lines (invented past durations, frequencies,
    clock times) trace to client speech alone. Returns failure strings whose
    heads feed ``corrective_note_for``: ``ledger_traceability``,
    ``spoken_specificity``.
    """
    session = next((s for s in plan["sessions"] if s["n"] == n), None)
    speech, plan_ev = _seed_evidence(plan, session, prior_sessions)

    ledger_hits: list[str] = []
    spoken_hits: list[str] = []
    client_turn = 0
    for kind, val in parsed["entries"]:
        if kind == "client":
            client_turn += 1
            speech.add(val)
            plan_ev.add(val)
        elif kind == "think":
            ledger = val["ledger"]
            if val["ok"] and isinstance(ledger, dict):
                _check_ledger_block(ledger, f"t{client_turn}", plan_ev, ledger_hits)
        elif kind == "therapist":
            _check_therapist_line(val, speech, spoken_hits, f"t{client_turn}")

    failures: list[str] = []
    if ledger_hits:
        failures.append(
            f"ledger_traceability: {len(ledger_hits)} untraceable ledger items: " + "; ".join(ledger_hits[:3])
        )
    if spoken_hits:
        failures.append(
            f"spoken_specificity: {len(spoken_hits)} untraceable speech items: " + "; ".join(spoken_hits[:3])
        )
    return failures
