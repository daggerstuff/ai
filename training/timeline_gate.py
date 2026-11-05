"""Timeline (``tl``) anchor accounting for the arc ledger.

The arc generator enforces an append-only invariant on the ledger's ``tl``
field: every time anchor (``now``, ``-6w``, ``+1y`` ...) that appears in one
ledger must still appear in the next one, within a session and across the
sessions of an arc. ``generate_arc_corpus.run_mechanical_gates`` implements it
by replacing its accumulated anchor set with each ledger's anchors and failing
the session when any of them vanish.

That gate is the only structural check the corpus has on long-range temporal
coherence, and it is the most common failure after the sycophancy gate
(200 of 1,195 recorded generation failures). This module exposes the same
accounting so a probe can measure drift on sampled continuations instead of
only catching it after a full session regeneration.

``TL_ANCHOR_RE`` is a copy of ``_TL_ANCHOR_RE`` in ``generate_arc_corpus.py``;
that module loads ``.env`` with ``override=True`` at import time, so it cannot
be imported here. ``test_timeline_gate.py`` pins the two patterns together.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

TL_ANCHOR_RE = re.compile(r"(?:^|[;,])\s*(now|[-+]?\d+[ymwd])(?=\s*:)", re.M)


def tl_anchor_keys(tl: str) -> set[str]:
    """Extract time anchors from a ``tl`` ledger string.

    Ledger entries are ``anchor: description`` separated by ``;`` or ``,``.
    Descriptions may themselves contain commas, so splitting on a delimiter and
    taking the text before the first colon yields phantom keys; match the short
    anchor token that sits at an entry boundary immediately before its colon.
    """
    return set(TL_ANCHOR_RE.findall(tl or ""))


def ledger_anchors(ledger: Mapping[str, Any]) -> set[str]:
    """Anchors carried by a parsed ledger, empty when ``tl`` is absent."""
    tl = ledger.get("tl")
    return tl_anchor_keys(tl) if isinstance(tl, str) else set()


def parse_ledger(text: str | None) -> dict[str, Any] | None:
    """Parse the ledger object that leads a therapist turn.

    Training rows carry ``<one-line ledger JSON>\\n<spoken reply>``, so the
    ledger is whatever ``raw_decode`` finds at the start. A turn that does not
    open with an object, or whose object is empty, is a generation breakdown
    rather than a drift observation and returns ``None``.
    """
    if not text:
        return None
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return None
    try:
        ledger, _ = json.JSONDecoder().raw_decode(stripped)
    except ValueError:
        return None
    if not isinstance(ledger, dict) or not ledger:
        return None
    return ledger


@dataclass(frozen=True, slots=True)
class DriftCheck:
    """One ledger measured against the anchors it inherited."""

    accumulated: frozenset[str]
    emitted: frozenset[str]

    @property
    def missing(self) -> frozenset[str]:
        """Inherited anchors the ledger dropped."""
        return self.accumulated - self.emitted

    @property
    def invented(self) -> frozenset[str]:
        """Anchors the ledger introduced that were never inherited."""
        return self.emitted - self.accumulated

    @property
    def holds(self) -> bool:
        """True when the append-only invariant survived this turn."""
        return not self.missing

    @property
    def recall(self) -> float:
        """Share of inherited anchors retained; 1.0 when there was nothing to keep."""
        if not self.accumulated:
            return 1.0
        return len(self.accumulated & self.emitted) / len(self.accumulated)

    @property
    def collapsed(self) -> bool:
        """True when every inherited anchor was dropped, not just some."""
        return bool(self.accumulated) and not (self.accumulated & self.emitted)


def check_drift(accumulated: Iterable[str], emitted: Iterable[str]) -> DriftCheck:
    """Compare a ledger's anchors against the anchors it inherited."""
    return DriftCheck(frozenset(accumulated), frozenset(emitted))


def accumulate(ledgers: Iterable[Mapping[str, Any]]) -> set[str]:
    """Union of anchors across ledgers.

    Equivalent to the last ledger's anchors for a corpus that satisfies the
    invariant (append-only makes the chain monotone), and the stricter reading
    for anything that has already drifted.
    """
    anchors: set[str] = set()
    for ledger in ledgers:
        anchors |= ledger_anchors(ledger)
    return anchors
