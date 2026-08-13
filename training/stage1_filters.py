"""Stage 1 QA filters — deterministic, no LLM calls. PIX-4342.

Implements the four Stage 1 filters from blueprint Appendix B.2:

* B.2.1 Language detection — fasttext-langdetect + non-English script ratio.
* B.2.2 PII removal — regex Layer 1 + Presidio Layer 2 (lazy-loaded).
* B.2.3 Toxicity — Detoxify local (lazy) OR Perspective API; gate severe+threat.
* B.2.4 Dedup — exact (SHA-256) + near (Jaccard <50K / MinHash+LSH 50K+).

Heavy deps (fasttext, presidio, detoxify) lazy-import at call sites with
graceful regex-only fallback so the module imports cleanly without them.

Wired into ``curate_pipeline.py`` via ``run_stage1_filters`` + per-record
``Stage1Result`` verdict.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# dedup + non-english constants reuse existing modules
from training.dedup_normalize import (
    _MinHashIndex,
    _content_hash,
    _extract_text,
    _minhash_signature,
)
from training.clinical_validity_judge import _NON_ENGLISH_RATIO, _NON_ENGLISH_RE

logger = logging.getLogger("stage1_filters")

NEAR_DUP_JACCARD_THRESHOLD = 0.85
MINHASH_NUM_PERM = 128
MINHASH_BANDS = 16
MINHASH_ROWS = 8
SMALL_CORPUS_THRESHOLD = 50_000

# Toxicity gates per blueprint B.2.3
TOXICITY_SEVERE_MAX = 0.30
TOXICITY_THREAT_MAX = 0.15

# PII regex Layer 1 — fast strip of obvious high-sensitivity entities.
_PII_REGEXES: dict[str, re.Pattern[str]] = {
    "EMAIL_ADDRESS": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE_NUMBER": re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}
_PII_REDACTION = "[REDACTED]"

PRESIDIO_ENTITIES: tuple[str, ...] = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "MEDICAL_LICENSE",
    "IP_ADDRESS",
    "PERSON",
    "LOCATION",
    "DATE_TIME",
)


class FilterVerdict:
    PASS = "pass"
    DROP = "drop"
    TRANSFORMED = "transformed"


@dataclass
class Stage1Result:
    """Outcome of running a record through the four Stage 1 filters."""

    verdict: str
    reasons: list[str] = field(default_factory=list)
    transformed_text: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    def add_reason(self, reason: str) -> None:
        self.reasons.append(reason)

    @property
    def passed(self) -> bool:
        return self.verdict in {FilterVerdict.PASS, FilterVerdict.TRANSFORMED}


# ---------------------------------------------------------------------------
# B.2.1 Language detection
# ---------------------------------------------------------------------------


def detect_language(text: str) -> tuple[str, float]:
    """Return ``(language_code, confidence)``.

    Uses ``fasttext-langdetect`` when available (lid.176.bin); falls back to
    the ``_NON_ENGLISH_RE`` script-ratio gate from ``clinical_validity_judge``.
    Fallback never claims a specific language — it only flags non-English.
    """

    if not text or not text.strip():
        return ("und", 0.0)
    try:
        import fasttext_langdetect  # lazy — optional dep
    except ImportError:
        fasttext_langdetect = None  # type: ignore[assignment]
    if fasttext_langdetect is not None:
        try:
            detector = fasttext_langdetect.FastTextLanguageDetector()
            lang, confidence = detector.detect(text)
            return (lang, float(confidence))
        except Exception as exc:  # pragma: no cover — fasttext runtime failures
            logger.debug("fasttext detect failed: %s", exc)
    non_en_ratio = len(_NON_ENGLISH_RE.findall(text)) / max(1, len(text.strip()))
    if non_en_ratio > _NON_ENGLISH_RATIO:
        return ("non_en_script", non_en_ratio)
    return ("en", 1.0 - non_en_ratio)


def language_filter(text: str, *, target_lang: str = "en") -> tuple[bool, str, float]:
    """Return ``(keep, language_code, confidence)``."""

    lang, conf = detect_language(text)
    keep = lang == target_lang or lang == "und"
    return (keep, lang, conf)


# ---------------------------------------------------------------------------
# B.2.2 PII removal (Layer 1 regex + Layer 2 Presidio)
# ---------------------------------------------------------------------------


def strip_pii_regex(text: str) -> tuple[str, int]:
    """Layer 1 fast regex strip.  Returns ``(redacted_text, hit_count)``."""

    if not text:
        return (text, 0)
    redacted = text
    hits = 0
    for pattern in _PII_REGEXES.values():
        redacted, count = pattern.subn(_PII_REDACTION, redacted)
        hits += count
    return (redacted, hits)


def strip_pii_presidio(text: str) -> tuple[str, int]:
    """Layer 2 Presidio anonymization.  Returns ``(redacted_text, hit_count)``.

    Falls back to Layer 1 regex when Presidio is unavailable.
    """

    if not text:
        return (text, 0)
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
    except ImportError:
        return strip_pii_regex(text)
    try:
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        results = analyzer.analyze(
            text=text,
            entities=list(PRESIDIO_ENTITIES),
            language="en",
        )
        if not results:
            return (text, 0)
        out = anonymizer.anonymize(text=text, analyzer_results=results)
        return (out.text, len(results))
    except Exception as exc:  # pragma: no cover — presidio runtime failures
        logger.warning("Presidio failed, falling back to regex: %s", exc)
        return strip_pii_regex(text)


def pii_filter(text: str, *, use_presidio: bool = True) -> tuple[str, int]:
    """Two-layer PII strip.  Returns ``(redacted_text, total_hits)``."""

    if not text:
        return (text, 0)
    if use_presidio:
        redacted, hits = strip_pii_presidio(text)
        # Presidio may miss regex-obvious patterns; run Layer 1 as a complement.
        if hits == 0:
            redacted, hits = strip_pii_regex(text)
        return (redacted, hits)
    return strip_pii_regex(text)


# ---------------------------------------------------------------------------
# B.2.3 Toxicity — Detoxify (local) gated on severe_toxicity + threat
# ---------------------------------------------------------------------------


def score_toxicity(text: str) -> dict[str, float]:
    """Return toxicity scores.  Keys: ``severe_toxicity``, ``threat`` (0.0-1.0).

    Uses Detoxify when installed; returns zeros (pass) when unavailable so
    the pipeline still runs without the dep.  Cloud Perspective API path is
    a TODO tracked separately (integration-test time).
    """

    if not text or not text.strip():
        return {"severe_toxicity": 0.0, "threat": 0.0}
    try:
        from detoxify import Detoxify
    except ImportError:
        return {"severe_toxicity": 0.0, "threat": 0.0}
    try:
        model = Detoxify("unbiased")
        scores = model.predict(text)
        return {
            "severe_toxicity": float(scores.get("severe_toxicity", 0.0)),
            "threat": float(scores.get("threat", 0.0)),
        }
    except Exception as exc:  # pragma: no cover — detoxify runtime failures
        logger.warning("Detoxify failed: %s", exc)
        return {"severe_toxicity": 0.0, "threat": 0.0}


def toxicity_filter(
    text: str,
    *,
    severe_max: float = TOXICITY_SEVERE_MAX,
    threat_max: float = TOXICITY_THREAT_MAX,
) -> tuple[bool, dict[str, float]]:
    """Return ``(keep, scores)``."""

    scores = score_toxicity(text)
    keep = scores["severe_toxicity"] < severe_max and scores["threat"] < threat_max
    return (keep, scores)


# ---------------------------------------------------------------------------
# B.2.4 Dedup — exact SHA-256 + near Jaccard / MinHash+LSH
# ---------------------------------------------------------------------------


class NearDuplicateIndex:
    """Adapter selecting small-corpus Jaccard vs large-corpus MinHash+LSH."""

    def __init__(
        self,
        *,
        corpus_size: int = SMALL_CORPUS_THRESHOLD,
        jaccard_threshold: float = NEAR_DUP_JACCARD_THRESHOLD,
    ) -> None:
        self.corpus_size = corpus_size
        self.threshold = jaccard_threshold
        self._minhash_index: _MinHashIndex | None = None
        self._seen_hashes: set[str] = set()
        self._seen_texts: list[tuple[str, frozenset[str]]] = []
        if corpus_size >= SMALL_CORPUS_THRESHOLD:
            self._minhash_index = _MinHashIndex(
                num_perm=MINHASH_NUM_PERM,
                bands=MINHASH_BANDS,
                rows=MINHASH_ROWS,
                jaccard_threshold=jaccard_threshold,
            )

    def is_duplicate(self, text: str) -> bool:
        text_hash = _content_hash(text)
        if text_hash in self._seen_hashes:
            return True
        if self._minhash_index is not None:
            sig = _minhash_signature(text, num_perm=MINHASH_NUM_PERM)
            if self._minhash_index.is_near_duplicate(sig, text_hash):
                return True
            self._minhash_index.add(text_hash, sig)
        else:
            tokens = frozenset(text.lower().split())
            for entry in self._seen_texts[-2000:]:
                if _jaccard_local(tokens, entry[1]) > self.threshold:
                    return True
            self._seen_texts.append((text_hash, tokens))
        self._seen_hashes.add(text_hash)
        return False


def _jaccard_local(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# Orchestrator — run all four filters on one record
# ---------------------------------------------------------------------------


def run_stage1_on_record(
    record: Mapping[str, Any],
    *,
    dedup_index: NearDuplicateIndex,
    target_lang: str = "en",
    use_presidio: bool = True,
) -> Stage1Result:
    """Run the four Stage 1 filters against one record.

    Order: dedup (cheapest, by content hash) → language → PII → toxicity.
    Short-circuits on the first DROP so we never waste a Presidio/Detoxify
    call on a record we already rejected.
    """

    text = _extract_text(dict(record)) or ""
    result = Stage1Result(verdict=FilterVerdict.PASS, stats={"text_len": len(text)})

    # B.2.4 Dedup
    if dedup_index.is_duplicate(text):
        result.verdict = FilterVerdict.DROP
        result.add_reason("near_duplicate")
        return result

    # B.2.1 Language
    keep, lang, conf = language_filter(text, target_lang=target_lang)
    result.stats["language"] = lang
    result.stats["language_confidence"] = round(conf, 4)
    if not keep:
        result.verdict = FilterVerdict.DROP
        result.add_reason(f"language_{lang}")
        return result

    # B.2.2 PII
    redacted, pii_hits = pii_filter(text, use_presidio=use_presidio)
    result.stats["pii_hits"] = pii_hits
    if pii_hits > 0:
        result.transformed_text = redacted
        result.verdict = FilterVerdict.TRANSFORMED

    # B.2.3 Toxicity
    keep_tox, tox_scores = toxicity_filter(text)
    result.stats.update(tox_scores)
    if not keep_tox:
        result.verdict = FilterVerdict.DROP
        result.add_reason(
            f"toxicity_severe_{tox_scores['severe_toxicity']:.2f}_threat_{tox_scores['threat']:.2f}"
        )
        return result

    return result


def run_stage1_filters(
    records: list[Mapping[str, Any]],
    *,
    corpus_size: int | None = None,
    target_lang: str = "en",
    use_presidio: bool = True,
) -> tuple[list[dict[str, Any]], list[Stage1Result]]:
    """Run all four Stage 1 filters across a corpus.

    Returns ``(kept_records, results)``.  ``kept_records`` carry the
    transformed text (PII-redacted) when applicable and an added
    ``stage1`` metadata block.
    """

    size = corpus_size if corpus_size is not None else len(records)
    dedup_index = NearDuplicateIndex(corpus_size=size)
    kept: list[dict[str, Any]] = []
    all_results: list[Stage1Result] = []
    for record in records:
        res = run_stage1_on_record(
            record,
            dedup_index=dedup_index,
            target_lang=target_lang,
            use_presidio=use_presidio,
        )
        all_results.append(res)
        if not res.passed:
            continue
        enriched = dict(record)
        if res.transformed_text is not None:
            enriched["text"] = res.transformed_text
            if "messages" in enriched:
                for msg in enriched["messages"]:
                    if isinstance(msg, dict) and msg.get("content"):
                        msg["content"] = res.transformed_text
        enriched["stage1"] = {
            "verdict": res.verdict,
            "reasons": res.reasons,
            "stats": res.stats,
        }
        kept.append(enriched)
    return (kept, all_results)
