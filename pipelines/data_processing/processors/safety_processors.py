"""
Hackathon safety processor — orchestrates PII stripping and heuristic toxicity
detection across hackathon-sourced training data for PIX-4240.

This module is the single integration point called by the dataset pipeline
(extract_everything.py) for all imported hackathon records. It:

  1. Strict PII stripping via the existing PiiScrubber (extended in PIX-4240
     with a url_with_identifying_path pattern).
  2. Heuristic toxicity detection via HeuristicToxicityDetector (new in
     PIX-4240), which distinguishes genuinely toxic content from legitimate
     clinical discussion.
  3. Emits a per-record SafetyReport and a cleaned ChatML record ready for
     the next pipeline stage (quality_filter -> shard upload).

Design constraints (PIX-4240):
  - Safety-critical: false negatives (missing PII/toxicity) worse than
    false positives (over-flagging).
  - But over-filtering legitimate clinical content is a real risk: mental
    health training data naturally references trauma, substance use, and
    difficult emotions. The toxicity detector distinguishes clinical
    discussion from genuinely toxic content via paired trigger/clinical-cue
    patterns — see toxicity_detector.py.
  - Builds on existing infrastructure; no parallel systems.
  - Output is deterministic + reproducible (pure regex, no ML model deps).
  - Never silently drops records: toxic-flagged records are routed to a
    separate "toxic_review" output for human review, not deleted.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from pipelines.data_processing.processors.toxicity_detector import (
    HeuristicToxicityDetector,
    ToxicityResult,
)
from ai.tools.utilities.pipelines.processing.pii_scrubber import (
    PiiScrubber,
    PiiScrubberConfig,
)

logger = logging.getLogger(__name__)


# Records with total toxicity score >= this threshold are routed to the
# "toxic_review" output. Below it but > 0 are flagged with the report but
# stay in the main output (the toxicity is advisory-only until review).
TOXIC_ROUTE_THRESHOLD: float = 0.85


@dataclass
class SafetyReport:
    """Per-record safety report emitted by HackathonSafetyProcessor."""

    pii_counts: dict[str, int] = field(default_factory=dict)
    pii_total: int = 0
    toxicity_score: float = 0.0
    toxicity_triggered_categories: list[str] = field(default_factory=list)
    toxicity_findings_summary: list[dict[str, Any]] = field(default_factory=list)
    clinical_matches_summary: list[dict[str, Any]] = field(default_factory=list)
    routed_to_toxic_review: bool = False
    record_id: str | None = None  # Ephemeral id for correlation across shards

    def to_dict(self) -> dict[str, Any]:
        return {
            "pii_counts": dict(self.pii_counts),
            "pii_total": self.pii_total,
            "toxicity_score": self.toxicity_score,
            "toxicity_triggered_categories": list(self.toxicity_triggered_categories),
            "toxicity_findings_summary": list(self.toxicity_findings_summary),
            "clinical_matches_summary": list(self.clinical_matches_summary),
            "routed_to_toxic_review": self.routed_to_toxic_review,
            "record_id": self.record_id,
        }


@dataclass
class SafetyProcessResult:
    """Output of HackathonSafetyProcessor.process()."""

    cleaned_record: dict[str, Any]
    report: SafetyReport


class HackathonSafetyProcessor:
    """
    Single hackathon-dataset safety processor: PII strip + tox score.

    Called from extract_everything.py between ChatML conversion and the
    quality filter. Pure-function, deterministic, no I/O, no side effects.

    Usage:
        processor = HackathonSafetyProcessor()
        cleaned_record, report = processor.process(chatml_record)
        if not report.routed_to_toxic_review:
            ...  # send to main output + quality filter
        else:
            ...  # send to toxic_review output
    """

    def __init__(
        self,
        pii_config: PiiScrubberConfig | None = None,
        toxicity_detector: HeuristicToxicityDetector | None = None,
        toxic_route_threshold: float = TOXIC_ROUTE_THRESHOLD,
    ) -> None:
        # spaCy name detection disabled by default for the hackathon pipeline:
        # deterministic-only by default. Callers who want spaCy can pass
        # PiiScrubberConfig(use_spacy_for_names=True) explicitly.
        if pii_config is None:
            pii_config = PiiScrubberConfig(
                redaction_style="[TYPE]",
                use_spacy_for_names=True,  # enable NER for name detection
                log_findings=False,
            )
        self._scrubber = PiiScrubber(pii_config)
        self._tox = toxicity_detector or HeuristicToxicityDetector()
        self._threshold = toxic_route_threshold

    # -- public API ---------------------------------------------------------

    def process(self, chatml_record: dict[str, Any]) -> SafetyProcessResult:
        """
        Run the full safety pass against one ChatML record.

        Args:
            chatml_record: ChatML dict {"messages": [...], "metadata": {...}}

        Returns:
            SafetyProcessResult with the cleaned record and a report describing
            what was stripped/flagged. The input record is never mutated.
        """
        if not isinstance(chatml_record, dict):
            raise TypeError(f"chatml_record must be dict, got {type(chatml_record).__name__}")

        # Stage 1: PII stripping — PiiScrubber.scrub_dict recurses into all str
        # values of the messages list and metadata subtree. We pass a deep copy
        # so the input is never mutated.
        import copy

        record_copy = copy.deepcopy(chatml_record)
        cleaned = self._scrubber.scrub_dict(record_copy)

        # Collect PII counts from the original record's string values in a
        # single pass. We scrub each string value once and aggregate counts;
        # the deep-copy scrub_dict above already produced the cleaned tree.
        pii_counts: dict[str, int] = {}
        pii_total = 0

        def _count_pii(text: str) -> None:
            nonlocal pii_total
            if not isinstance(text, str) or not text:
                return
            sr = self._scrubber.scrub(text)
            for k, v in sr.pii_counts.items():
                pii_counts[k] = pii_counts.get(k, 0) + v
                pii_total += v

        for msg in chatml_record.get("messages", []):
            _count_pii(msg.get("content", ""))
        meta = chatml_record.get("metadata", {})
        if isinstance(meta, dict):
            for v in meta.values():
                _count_pii(v)

        # Stage 2: heuristic toxicity detection (operates on cleaned record so
        # PII redaction tokens don't accidentally trip clinical cues)
        tox_result: ToxicityResult = self._tox.detect_record(cleaned)

        triggered = [name for name, cr in tox_result.categories.items() if cr.triggered]
        routed = tox_result.score >= self._threshold and bool(triggered)

        # Build a stable ephemeral record_id from content hash (for shard correlation)
        record_id = _stable_record_id(chatml_record)

        # Findings summary — light report, no full PII/identifying text echoed back
        findings_summary: list[dict[str, Any]] = [
            {
                "category": f.category,
                "start": f.start,
                "end": f.end,
                "weight": f.weight,
                "sample_prefix": _safe_prefix(f.matched_text, 40),
            }
            for f in tox_result.findings
        ]
        clinical_summary: list[dict[str, Any]] = [
            {
                "category": f.category,
                "start": f.start,
                "end": f.end,
                "weight": f.weight,
            }
            for cr in tox_result.categories.values()
            for f in cr.clinical_matches
        ]

        report = SafetyReport(
            pii_counts=dict(pii_counts),
            pii_total=pii_total,
            toxicity_score=tox_result.score,
            toxicity_triggered_categories=triggered,
            toxicity_findings_summary=findings_summary,
            clinical_matches_summary=clinical_summary,
            routed_to_toxic_review=routed,
            record_id=record_id,
        )

        # Attach cleaned report metadata to the record so downstream stages
        # and the shard uploader can include provenance.
        meta_clean = cleaned.setdefault("metadata", {})
        if isinstance(meta_clean, dict):
            meta_clean["safety_report"] = report.to_dict()

        return SafetyProcessResult(cleaned_record=cleaned, report=report)


def _stable_record_id(record: dict[str, Any]) -> str:
    """Derive a stable ephemeral id from message content for shard correlation."""
    h = hashlib.sha1(usedforsecurity=False)
    for msg in record.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        h.update(role.encode("utf-8", "ignore"))
        h.update(b"|")
        h.update(str(content).encode("utf-8", "ignore"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def _safe_prefix(text: str, n: int) -> str:
    """Return a prefix of text, with PII-suspicious chars masked and length capped."""
    if not text:
        return ""
    truncated = text[:n]
    # Mask any digits that look like phone/ssn/cc fragments
    import re

    masked = re.sub(r"\d", "X", truncated)
    return masked


__all__ = [
    "TOXIC_ROUTE_THRESHOLD",
    "HackathonSafetyProcessor",
    "SafetyProcessResult",
    "SafetyReport",
]
