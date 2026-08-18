"""Stage 1 QA filter chain for the AI training curation pipeline (PIX-4342).

Streaming filter chain that takes an iterable of records and yields only
records that pass all filters. Filters run in order: language → PII →
toxicity → dedup (cheapest first).

Each filter is a callable ``(record) -> FilterResult`` where ``FilterResult``
is a ``(passed, reason, metadata)`` named tuple.

Heavy ML models (fasttext, presidio, detoxify) are loaded lazily and cached
so the chain can be constructed without the models present — tests inject
mocks via constructor parameters.

This module unifies two predecessor implementations:

* ``Stage1FilterChain`` + per-filter classes (PIX-4342, Steiner) — the
  canonical streaming API used by ``curate_pipeline.py``. Persistent SQLite
  dedup store, optional SimHash for 10M+ scale, Protocol-based filters,
  chain-level statistics.
* Functional helpers (PIX-4345, SDG pipeline) — ``detect_language``,
  ``language_filter``, ``strip_pii_regex``/``strip_pii_presidio``,
  ``pii_filter``. These add capabilities the streaming classes do not:
  the ``fasttext-langdetect`` package path, confidence-returning language
  detection, and **transforming** PII redaction (vs. the chain's drop-only
  PII filter). They are kept as module-level primitives.

Usage::

    chain = Stage1FilterChain()
    for record in chain.filter(records):
        process(record)
    print(chain.summary())
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Protocol

logger = logging.getLogger("stage1_filters")

# ---------------------------------------------------------------------------
# Shared constants — reuse _NON_ENGLISH_RE from clinical_validity_judge.py
# ---------------------------------------------------------------------------

# Non-English script detection — matches the pattern in clinical_validity_judge.py
_NON_ENGLISH_RE = re.compile(
    "["
    "一-鿿"  # CJK Unified Ideographs
    "぀-ゟ"  # Hiragana
    "゠-ヿ"  # Katakana
    "가-힯"  # Hangul Syllables
    "Ѐ-ӿ"  # Cyrillic
    "؀-ۿ"  # Arabic
    "฀-๿"  # Thai
    "ༀ-࿿"  # Tibetan
    "]",
)
_NON_ENGLISH_RATIO = 0.30

# PII entities reused from pii_scrubber.py
PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "CRYPTO",
    "IP_ADDRESS",
    "MEDICAL_LICENSE",
    "US_SSN",
    "US_PASSPORT",
]

# Presidio entity set for the transforming redaction path (PIX-4345).
# Broader than ``PII_ENTITIES`` — includes PERSON/LOCATION/DATE_TIME so the
# redactor can scrub contextual identifiers the drop-only filter ignores.
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

# Toxicity thresholds
SEVERE_TOXICITY_THRESHOLD = 0.30
THREAT_THRESHOLD = 0.15

# Dedup thresholds
JACCARD_THRESHOLD = 0.85
SIMHASH_SCALE_THRESHOLD = 10_000_000  # 10M records → switch to SimHash

# PII regex Layer 1 — fast strip of obvious high-sensitivity entities.
_PII_REGEXES: dict[str, re.Pattern[str]] = {
    "EMAIL_ADDRESS": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE_NUMBER": re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}
_PII_REDACTION = "[REDACTED]"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class FilterResult(NamedTuple):
    """Result of a single filter check."""

    passed: bool
    reason: str
    metadata: dict[str, Any]


class RecordFilter(Protocol):
    """Protocol for filter callables."""

    def __call__(self, record: dict[str, Any]) -> FilterResult: ...



# ---------------------------------------------------------------------------
# Helpers — reused by both the streaming chain and the functional primitives.
# ---------------------------------------------------------------------------


def _extract_text(record: dict[str, Any]) -> str:
    """Extract all text content from a record for analysis."""
    messages = record.get("messages", [])
    if messages:
        return " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
    prompt = record.get("prompt", "")
    chosen = record.get("chosen", "")
    rejected = record.get("rejected", "")

    def _text_part(part: Any) -> str:
        if isinstance(part, list):
            return " ".join(str(m.get("content", "")) for m in part if isinstance(m, dict))
        return str(part)

    preference_parts = [_text_part(p) for p in (prompt, chosen, rejected) if p]
    if preference_parts:
        return " ".join(preference_parts)
    return record.get("text", "") or f"{record.get('instruction', '')} {record.get('output', '')}"


def _content_hash(text: str) -> str:
    """SHA-256 content hash for exact dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_set(text: str) -> frozenset[str]:
    """Token set for Jaccard / MinHash."""
    return frozenset(text.lower().split())


def _jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


# ---------------------------------------------------------------------------
# SimHash implementation for 10M+ scale near-duplicate detection.
# ---------------------------------------------------------------------------


def _simhash(text: str, hash_bits: int = 64) -> int:
    """Compute SimHash fingerprint for near-duplicate detection at scale."""
    tokens = text.lower().split()
    if not tokens:
        return 0
    v = [0] * hash_bits
    for token in tokens:
        h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        for i in range(hash_bits):
            bit = (h >> i) & 1
            v[i] += 1 if bit else -1
    fingerprint = 0
    for i in range(hash_bits):
        if v[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def _hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# Functional language primitives (PIX-4345).
#
# Distinct from ``LanguageFilter``: these use the ``fasttext-langdetect``
# package, return ``(language, confidence)`` tuples, and never mutate state.
# Used by the SDG pipeline for per-record inspection where streaming isn't
# needed.
# ---------------------------------------------------------------------------


def detect_language(text: str) -> tuple[str, float]:
    """Return ``(language_code, confidence)``.

    Uses ``fasttext-langdetect`` when available (lid.176.bin); falls back to
    the ``_NON_ENGLISH_RE`` script-ratio gate. The fallback never claims a
    specific language — it only flags non-English.
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
# Functional PII primitives (PIX-4345).
#
# Distinct from ``PIIFilter``: these **redact** PII in-place and return the
# scrubbed text + a hit count, rather than dropping the record. Used by SDG
# when synthetic text should be cleaned, not discarded.
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
# Language Filter (streaming chain element, PIX-4342)
# ---------------------------------------------------------------------------


class LanguageFilter:
    """Language detection filter using fasttext-langdetect.

    Uses ``fasttext-langdetect`` with the ``lid.176.bin`` model (downloaded
    and cached locally if missing). Falls back to a regex-based non-English
    script check when fasttext is unavailable.

    Reuses ``_NON_ENGLISH_RE`` from ``clinical_validity_judge.py`` as a
    secondary check: if fasttext says English but the non-English regex
    matches >30% of tokens, the record is dropped.
    """

    _MODEL_CACHE: dict[str, Any] = {}
    _MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    _MODEL_DIR = os.path.join(tempfile.gettempdir(), "fasttext_models")
    _MODEL_PATH = os.path.join(_MODEL_DIR, "lid.176.bin")

    def __init__(
        self, model: Any | None = None, accepted_langs: frozenset[str] | None = None
    ) -> None:
        self._model = model
        self._accepted_langs = accepted_langs or frozenset({"en"})
        self._loaded = model is not None

    def _load_model(self) -> Any:
        """Lazy-load the fasttext model, downloading if necessary."""
        if self._model is not None:
            return self._model

        cache_key = "fasttext_lid176"
        if cache_key in self._MODEL_CACHE:
            return self._MODEL_CACHE[cache_key]

        try:
            import fasttext
        except ImportError:
            logger.warning("fasttext not installed — language filter will use regex-only mode")
            return None

        if not os.path.exists(self._MODEL_PATH):
            import stat

            os.makedirs(self._MODEL_DIR, exist_ok=True)
            os.chmod(self._MODEL_DIR, stat.S_IRWXU)
            logger.info("Downloading fasttext lid.176.bin model...")
            try:
                import requests

                with requests.get(self._MODEL_URL, timeout=60, stream=True) as resp:
                    resp.raise_for_status()
                    with open(self._MODEL_PATH, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                os.chmod(self._MODEL_PATH, stat.S_IRUSR | stat.S_IWUSR)
            except Exception as e:
                logger.warning("Failed to download fasttext model: %s — using regex-only mode", e)
                return None

        model = fasttext.load_model(self._MODEL_PATH)
        self._MODEL_CACHE[cache_key] = model
        self._model = model
        self._loaded = True
        return model

    def _detect_language(self, text: str) -> str:
        """Detect language code. Returns 'en' if fasttext unavailable or text empty."""
        if not text.strip():
            return "en"
        model = self._load_model()
        if model is None:
            # Fallback: regex-only — check if non-English scripts dominate
            non_english_chars = len(_NON_ENGLISH_RE.findall(text))
            total_chars = max(1, len(text.strip()))
            if (non_english_chars / total_chars) > _NON_ENGLISH_RATIO:
                return "und"  # undetermined non-English
            return "en"
        # fasttext predict returns (labels, probs)
        labels, probs = model.predict(text.replace("\n", " "), k=1)
        lang = labels[0].replace("__label__", "")
        return lang

    def _check_non_english_ratio(self, text: str) -> bool:
        """Secondary check: if >30% of chars are non-English script, flag as non-English."""
        if not text or not text.strip():
            return False
        non_english_chars = len(_NON_ENGLISH_RE.findall(text))
        total_chars = max(1, len(text.strip()))
        return (non_english_chars / total_chars) > _NON_ENGLISH_RATIO

    def __call__(self, record: dict[str, Any]) -> FilterResult:
        text = _extract_text(record)
        if not text.strip():
            return FilterResult(passed=False, reason="empty_text", metadata={"lang": "und"})

        lang = self._detect_language(text)

        # Primary: fasttext language check
        if lang not in self._accepted_langs:
            return FilterResult(passed=False, reason=f"non_english:{lang}", metadata={"lang": lang})

        # Secondary: regex override — fasttext says English but non-English scripts dominate
        if lang == "en" and self._check_non_english_ratio(text):
            return FilterResult(
                passed=False,
                reason="non_english_regex_override",
                metadata={"lang": lang, "non_english_ratio": True},
            )

        return FilterResult(passed=True, reason="", metadata={"lang": lang})


# ---------------------------------------------------------------------------
# PII Filter (streaming chain element, PIX-4342)
# ---------------------------------------------------------------------------


class PIIFilter:
    """PII detection filter using presidio-analyzer + presidio-anonymizer.

    Reuses entity patterns from ``pii_scrubber.py``. For borderline cases
    (Presidio confidence < 0.8), routes to an LLM second-opinion via
    ``utils/common/llm_client.py`` when ``llm_borderline=True``.

    Drops records with confirmed PII. For borderline-LLM-reviewed records,
    drops only if the LLM confirms PII.
    """

    PII_CONFIDENCE_THRESHOLD = 0.8

    def __init__(
        self,
        analyzer: Any | None = None,
        llm_client: Any | None = None,
        llm_borderline: bool = False,
        entities: list[str] | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._llm_client = llm_client
        self._llm_borderline = llm_borderline
        self._entities = entities or PII_ENTITIES

    def _load_analyzer(self) -> Any:
        if self._analyzer is not None:
            return self._analyzer
        try:
            from presidio_analyzer import AnalyzerEngine

            self._analyzer = AnalyzerEngine()
            return self._analyzer
        except (ImportError, OSError) as e:
            logger.warning(
                "presidio_analyzer / spaCy model unavailable — PII filter will pass all records: %s",
                e,
            )
            return None

    def _llm_confirm_pii(self, text: str) -> bool:
        """Ask LLM to confirm whether borderline text contains PII."""
        if self._llm_client is None:
            return False  # No LLM available → don't drop (conservative: keep the record)
        prompt = (
            "Does the following text contain personally identifiable information "
            "(emails, phone numbers, SSNs, credit cards, passport numbers, medical license numbers)? "
            "Respond with ONLY 'YES' or 'NO'.\n\n"
            f"Text: {text[:2000]}"
        )
        try:
            response = self._llm_client.generate(prompt)
            return "YES" in response.upper()
        except Exception as e:
            logger.warning("LLM PII confirmation failed: %s", e)
            return False

    def __call__(self, record: dict[str, Any]) -> FilterResult:
        text = _extract_text(record)
        if not text.strip():
            return FilterResult(passed=False, reason="empty_text", metadata={"pii_found": False})

        analyzer = self._load_analyzer()
        if analyzer is None:
            return FilterResult(
                passed=True, reason="", metadata={"pii_found": False, "analyzer_unavailable": True}
            )

        try:
            results = analyzer.analyze(text=text, entities=self._entities, language="en")
        except Exception as e:
            logger.warning("Presidio analysis failed: %s", e)
            return FilterResult(
                passed=True, reason="", metadata={"pii_found": False, "analyzer_error": str(e)}
            )

        if not results:
            return FilterResult(passed=True, reason="", metadata={"pii_found": False})

        # Check confidence levels
        high_confidence = [
            r for r in results if getattr(r, "score", 0) >= self.PII_CONFIDENCE_THRESHOLD
        ]
        borderline = [r for r in results if getattr(r, "score", 0) < self.PII_CONFIDENCE_THRESHOLD]

        # High-confidence PII → drop
        if high_confidence:
            entity_types = [getattr(r, "entity_type", "unknown") for r in high_confidence]
            return FilterResult(
                passed=False,
                reason=f"pii_confirmed:{','.join(entity_types)}",
                metadata={"pii_found": True, "entity_types": entity_types, "confidence": "high"},
            )

        # Borderline PII → LLM second-opinion if enabled, otherwise pass
        if borderline and self._llm_borderline:
            llm_confirms = self._llm_confirm_pii(text)
            if llm_confirms:
                entity_types = [getattr(r, "entity_type", "unknown") for r in borderline]
                return FilterResult(
                    passed=False,
                    reason=f"pii_llm_confirmed:{','.join(entity_types)}",
                    metadata={"pii_found": True, "entity_types": entity_types, "confidence": "llm"},
                )
            return FilterResult(
                passed=True,
                reason="pii_borderline_llm_rejected",
                metadata={"pii_found": False, "borderline_entities": len(borderline)},
            )

        # Borderline without LLM → pass (conservative)
        return FilterResult(
            passed=True,
            reason="pii_borderline_no_llm",
            metadata={"pii_found": False, "borderline_entities": len(borderline)},
        )


# ---------------------------------------------------------------------------
# Toxicity Filter (detoxify 4-model ensemble, streaming chain element)
# ---------------------------------------------------------------------------


class ToxicityFilter:
    """Toxicity filter using detoxify 4-model ensemble.

    Loads four detoxify models (original, unbiased, multilingual, original-small)
    and combines their predictions. Caches model loads so they load once and
    reuse across records.

    Gates: ``severe_toxicity < 0.30`` AND ``threat < 0.15``.
    Drops if either threshold is exceeded.
    """

    MODEL_NAMES = ("original", "unbiased", "multilingual", "original-small")

    _MODEL_CACHE: dict[str, Any] = {}

    def __init__(self, models: dict[str, Any] | None = None) -> None:
        self._models = models
        self._loaded = models is not None

    def _load_models(self) -> dict[str, Any]:
        if self._models is not None:
            return self._models

        models: dict[str, Any] = {}
        for name in self.MODEL_NAMES:
            if name in self._MODEL_CACHE:
                models[name] = self._MODEL_CACHE[name]
                continue
            try:
                from detoxify import Detoxify

                model = Detoxify(name)
                self._MODEL_CACHE[name] = model
                models[name] = model
            except (ImportError, Exception) as e:
                logger.warning("Failed to load detoxify model '%s': %s", name, e)
                models[name] = None
        self._models = models
        self._loaded = True
        return models

    def _predict(self, text: str) -> dict[str, float]:
        """Run all 4 models and return max scores across the ensemble."""
        models = self._load_models()
        all_scores: dict[str, list[float]] = {}

        for name, model in models.items():
            if model is None:
                continue
            try:
                if name == "unbiased":
                    # unbiased model has different output keys
                    results = model.predict(text)
                    # Map: toxicity, severe_toxicity, obscene, identity_attack, insult, threat
                    for key, val in results.items():
                        score = (
                            float(val)
                            if isinstance(val, (int, float))
                            else float(val[0])
                            if val
                            else 0.0
                        )
                        all_scores.setdefault(key, []).append(score)
                else:
                    results = model.predict(text)
                    for key, val in results.items():
                        score = (
                            float(val)
                            if isinstance(val, (int, float))
                            else float(val[0])
                            if val
                            else 0.0
                        )
                        all_scores.setdefault(key, []).append(score)
            except Exception as e:
                logger.warning("Detoxify model '%s' prediction failed: %s", name, e)

        # Take max across ensemble for each category
        return {key: max(vals) for key, vals in all_scores.items() if vals}

    def __call__(self, record: dict[str, Any]) -> FilterResult:
        text = _extract_text(record)
        if not text.strip():
            return FilterResult(passed=False, reason="empty_text", metadata={"toxicity_scores": {}})

        scores = self._predict(text)

        if not scores:
            # All models unavailable → pass (can't filter)
            return FilterResult(
                passed=True,
                reason="toxicity_models_unavailable",
                metadata={"toxicity_scores": {}, "models_available": False},
            )

        severe_toxicity = scores.get("severe_toxicity", 0.0)
        threat = scores.get("threat", 0.0)

        metadata = {"toxicity_scores": scores, "models_available": True}

        if severe_toxicity >= SEVERE_TOXICITY_THRESHOLD:
            return FilterResult(
                passed=False,
                reason=f"severe_toxicity:{severe_toxicity:.3f}",
                metadata=metadata,
            )

        if threat >= THREAT_THRESHOLD:
            return FilterResult(
                passed=False,
                reason=f"threat:{threat:.3f}",
                metadata=metadata,
            )

        return FilterResult(passed=True, reason="", metadata=metadata)


# ---------------------------------------------------------------------------
# Dedup Filter
# ---------------------------------------------------------------------------


class DedupFilter:
    """Deduplication filter with exact + near-duplicate detection.

    - SHA-256 exact dedup (drop exact content-hash duplicates).
    - MinHash + LSH via datasketch for near-duplicate detection (Jaccard threshold 0.85).
    - SimHash for 10M+ scale near-duplicate detection (optional, default to MinHash/LSH).
    - Persistent dedup store (SQLite) so dedup state survives across runs.

    Reuses normalization from ``dedup_normalize.py``.
    """

    def __init__(
        self,
        jaccard_threshold: float = JACCARD_THRESHOLD,
        dedup_store_path: str | None = None,
        use_simhash: bool = False,
        lsh_index: Any | None = None,
    ) -> None:
        self._jaccard_threshold = jaccard_threshold
        self._use_simhash = use_simhash
        self._seen_hashes: set[str] = set()
        self._seen_sighashes: set[int] = set()
        self._lsh_index = lsh_index
        self._lsh_initialized = False
        if self._lsh_index is not None:
            self._lsh = self._lsh_index
            self._minhash_set: dict[str, Any] = {}
            self._lsh_initialized = True

        # Persistent dedup store — default to a private user cache directory
        if dedup_store_path:
            self._db_path = dedup_store_path
        else:
            db_dir = os.path.join(Path.home(), ".cache", "pixelated")
            try:
                os.makedirs(db_dir, exist_ok=True)
                os.chmod(db_dir, 0o700)
                self._db_path = os.path.join(db_dir, "stage1_dedup.db")
            except OSError:
                self._db_path = os.path.join(
                    tempfile.gettempdir(), f"stage1_dedup_{os.getpid()}.db"
                )
        self._db: sqlite3.Connection | None = None
        self._init_db()

        # Load existing hashes from store
        self._load_existing_hashes()

        # Initialize LSH index if not using SimHash
        if not self._use_simhash and self._lsh_index is None:
            self._init_lsh_index()

    def _init_db(self) -> None:
        """Initialize SQLite dedup store."""
        try:
            self._db = sqlite3.connect(self._db_path)
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS dedup_hashes (
                    content_hash TEXT PRIMARY KEY,
                    simhash INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._db.commit()
        except Exception as e:
            logger.warning("Failed to initialize dedup SQLite store: %s", e)
            self._db = None

    def _load_existing_hashes(self) -> None:
        """Load existing hashes from the persistent store."""
        if self._db is None:
            return
        try:
            cursor = self._db.execute("SELECT content_hash, simhash FROM dedup_hashes")
            for chash, simhash in cursor:
                self._seen_hashes.add(chash)
                if simhash is not None:
                    self._seen_sighashes.add(simhash)
        except Exception as e:
            logger.warning("Failed to load existing dedup hashes: %s", e)

    def _store_hash(self, content_hash: str, simhash: int | None) -> None:
        """Persist a new hash to the dedup store."""
        if self._db is None:
            return
        try:
            self._db.execute(
                "INSERT OR IGNORE INTO dedup_hashes (content_hash, simhash) VALUES (?, ?)",
                (content_hash, simhash),
            )
            self._db.commit()
        except Exception as e:
            logger.warning("Failed to store dedup hash: %s", e)

    def _init_lsh_index(self) -> None:
        """Initialize MinHash LSH index."""
        try:
            from datasketch import MinHash, MinHashLSH

            self._lsh = MinHashLSH(threshold=self._jaccard_threshold, num_perm=128)
            self._minhash_set: dict[str, MinHash] = {}
            self._lsh_initialized = True
        except ImportError:
            logger.warning(
                "datasketch not installed — near-dedup will use token-set Jaccard fallback"
            )
            self._lsh = None
            self._token_sets: list[tuple[frozenset[str], str]] = []
            self._lsh_initialized = False

    def _compute_minhash(self, text: str, num_perm: int = 128) -> Any:
        """Compute MinHash signature using datasketch."""
        from datasketch import MinHash

        mh = MinHash(num_perm=num_perm)
        for token in text.lower().split():
            mh.update(token.encode("utf-8"))
        return mh

    def _check_near_duplicate_lsh(self, text: str, content_hash: str) -> bool:
        """Check near-duplicate using MinHash LSH (datasketch)."""
        if self._lsh is not None:
            mh = self._compute_minhash(text)
            candidates = self._lsh.query(mh)
            for cand_hash in candidates:
                cand_mh = self._minhash_set.get(cand_hash)
                if cand_mh is not None and mh.jaccard(cand_mh) > self._jaccard_threshold:
                    return True
            self._lsh.insert(content_hash, mh)
            self._minhash_set[content_hash] = mh
            return False
        # Fallback: token-set Jaccard
        tokens = _token_set(text)
        for existing_tokens, existing_hash in self._token_sets:
            if existing_hash == content_hash:
                continue
            if _jaccard_similarity(tokens, existing_tokens) > self._jaccard_threshold:
                return True
        self._token_sets.append((tokens, content_hash))
        return False

    def _check_near_duplicate_simhash(self, text: str) -> tuple[bool, int]:
        """Check near-duplicate using SimHash (for 10M+ scale)."""
        fingerprint = _simhash(text)
        # Check against all seen SimHashes with Hamming distance < 4
        for existing_fp in self._seen_sighashes:
            if _hamming_distance(fingerprint, existing_fp) < 4:
                return True, fingerprint
        self._seen_sighashes.add(fingerprint)
        return False, fingerprint

    def __call__(self, record: dict[str, Any]) -> FilterResult:
        text = _extract_text(record)
        if not text.strip():
            return FilterResult(passed=False, reason="empty_text", metadata={"dedup": "empty"})

        chash = _content_hash(text)

        # Exact dedup check
        if chash in self._seen_hashes:
            return FilterResult(
                passed=False,
                reason="exact_duplicate",
                metadata={"dedup": "exact", "content_hash": chash[:16]},
            )

        # Near-duplicate check
        simhash_val: int | None = None
        if self._use_simhash:
            is_near_dup, simhash_val = self._check_near_duplicate_simhash(text)
            if is_near_dup:
                return FilterResult(
                    passed=False,
                    reason="near_duplicate_simhash",
                    metadata={"dedup": "near_simhash", "content_hash": chash[:16]},
                )
        elif self._check_near_duplicate_lsh(text, chash):
            return FilterResult(
                passed=False,
                reason="near_duplicate_minhash",
                metadata={"dedup": "near_minhash", "content_hash": chash[:16]},
            )

        # Record the hash
        self._seen_hashes.add(chash)
        self._store_hash(chash, simhash_val)

        return FilterResult(
            passed=True, reason="", metadata={"dedup": "unique", "content_hash": chash[:16]}
        )

    def close(self) -> None:
        """Close the dedup store connection."""
        if self._db is not None:
            self._db.close()
            self._db = None


# ---------------------------------------------------------------------------
# Stage1FilterChain
# ---------------------------------------------------------------------------


@dataclass
class FilterStats:
    """Statistics for a single filter."""

    name: str
    total: int = 0
    passed: int = 0
    dropped: int = 0
    drop_reasons: Counter = field(default_factory=Counter)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0.0


@dataclass
class ChainStats:
    """Aggregate statistics for the entire filter chain."""

    total_input: int = 0
    total_output: int = 0
    filter_stats: dict[str, FilterStats] = field(default_factory=dict)

    def get_or_create(self, name: str) -> FilterStats:
        if name not in self.filter_stats:
            self.filter_stats[name] = FilterStats(name=name)
        return self.filter_stats[name]

    @property
    def pass_through_rate(self) -> float:
        return (self.total_output / self.total_input * 100) if self.total_input > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input": self.total_input,
            "total_output": self.total_output,
            "pass_through_rate": round(self.pass_through_rate, 2),
            "filters": {
                name: {
                    "total": fs.total,
                    "passed": fs.passed,
                    "dropped": fs.dropped,
                    "pass_rate": round(fs.pass_rate, 2),
                    "drop_reasons": dict(fs.drop_reasons.most_common()),
                }
                for name, fs in self.filter_stats.items()
            },
        }

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            "Stage1 Filter Chain Summary",
            f"  Input:  {self.total_input:,}",
            f"  Output: {self.total_output:,}",
            f"  Pass-through: {self.pass_through_rate:.1f}%",
            "",
        ]
        for name, fs in self.filter_stats.items():
            lines.append(f"  {name}: {fs.passed:,}/{fs.total:,} passed ({fs.pass_rate:.1f}%)")
            for reason, count in fs.drop_reasons.most_common(5):
                lines.append(f"    dropped {count:,}: {reason}")
        return "\n".join(lines)


class Stage1FilterChain:
    """Streaming filter chain for Stage 1 QA.

    Takes an iterable of records, yields only records that pass all filters.
    Filters run in order: language → PII → toxicity → dedup (cheapest first).

    Each filter is a callable ``(record) -> FilterResult``.

    Args:
        language_filter: Custom language filter (or None to create default).
        pii_filter: Custom PII filter (or None to create default).
        toxicity_filter: Custom toxicity filter (or None to create default).
        dedup_filter: Custom dedup filter (or None to create default).
        llm_borderline: Enable LLM second-opinion for PII borderline cases.
        dedup_store_path: Path for the persistent dedup SQLite store.
        use_simhash: Use SimHash instead of MinHash/LSH for near-dedup.

    Example::

        chain = Stage1FilterChain()
        for record in chain.filter(records):
            process(record)
        print(chain.summary())
    """

    FILTER_ORDER = ("language", "pii", "toxicity", "dedup")

    def __init__(
        self,
        language_filter: RecordFilter | None = None,
        pii_filter: RecordFilter | None = None,
        toxicity_filter: RecordFilter | None = None,
        dedup_filter: RecordFilter | None = None,
        llm_borderline: bool = False,
        dedup_store_path: str | None = None,
        use_simhash: bool = False,
    ) -> None:
        self.language_filter = language_filter or LanguageFilter()
        self.pii_filter = pii_filter or PIIFilter(llm_borderline=llm_borderline)
        self.toxicity_filter = toxicity_filter or ToxicityFilter()
        self.dedup_filter = dedup_filter or DedupFilter(
            dedup_store_path=dedup_store_path,
            use_simhash=use_simhash,
        )
        self.stats = ChainStats()

        # Map filter names to their callables (in order)
        self._filters: list[tuple[str, RecordFilter]] = [
            ("language", self.language_filter),
            ("pii", self.pii_filter),
            ("toxicity", self.toxicity_filter),
            ("dedup", self.dedup_filter),
        ]

    def filter(self, records: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        """Filter records through the chain, yielding only passing records.

        Logs drop reasons + counts per filter. Updates ``self.stats``
        with running statistics. Emits a summary at end of run (via logger).

        Args:
            records: Iterable of record dicts.

        Yields:
            Records that pass all filters.
        """
        for record in records:
            self.stats.total_input += 1
            passed_all = True

            for filter_name, filter_fn in self._filters:
                fs = self.stats.get_or_create(filter_name)
                fs.total += 1

                result = filter_fn(record)

                if not result.passed:
                    fs.dropped += 1
                    fs.drop_reasons[result.reason] += 1
                    logger.debug(
                        "Stage1 %s dropped record: %s (metadata: %s)",
                        filter_name,
                        result.reason,
                        result.metadata,
                    )
                    passed_all = False
                    break  # Stop at first filter that drops
                else:
                    fs.passed += 1
                    # Attach filter metadata to the record
                    record.setdefault("_stage1_metadata", {})[filter_name] = result.metadata

            if passed_all:
                self.stats.total_output += 1
                yield record

        # Summary is available via summary() / stats_dict() for the caller to log.

    def summary(self) -> str:
        """Return human-readable summary of the filter run."""
        return self.stats.summary()

    def stats_dict(self) -> dict[str, Any]:
        """Return stats as a dict (for JSON serialization)."""
        return self.stats.to_dict()

    def close(self) -> None:
        """Close any resources held by filters (e.g., dedup store)."""
        close_fn = getattr(self.dedup_filter, "close", None)
        if callable(close_fn):
            close_fn()


__all__ = [
    "PII_ENTITIES",
    "PRESIDIO_ENTITIES",
    "SEVERE_TOXICITY_THRESHOLD",
    "SIMHASH_SCALE_THRESHOLD",
    "THREAT_THRESHOLD",
    "ChainStats",
    "DedupFilter",
    "FilterResult",
    "FilterStats",
    "LanguageFilter",
    "PIIFilter",
    "RecordFilter",
    "Stage1FilterChain",
    "ToxicityFilter",
    "detect_language",
    "language_filter",
    "pii_filter",
    "strip_pii_presidio",
    "strip_pii_regex",
]
