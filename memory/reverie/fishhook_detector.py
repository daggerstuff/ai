"""
Fishhook Detector

Detects "fishhooks" — subtle cues in the current context that resonate with
latent memories. Inspired by Westworld's reveries: tiny gestures that pull
from the deep sea of consciousness.

Four detection modalities:
    1. Lexical — TF-IDF cosine similarity (subtle word overlap, not exact match)
    2. Emotional — VAD resonance (valence/arousal alignment)
    3. Pattern — recurring emotion categories (thematic echoes)
    4. Surprise — Bayesian surprise (deviation from expected pattern)

Python/TypeScript parity: src/lib/memory/reverie/fishhook_detector.ts mirrors this file.
"""

from __future__ import annotations

import math
import re
import time

from ai.memory.reverie_types import (
    DEFAULT_REVERIE_CONFIG,
    FishhookMatch,
    FishhookMatchType,
    ReverieConfig,
)
from ai.memory.schema import MemoryBlock

# ─── TF-IDF Helpers ──────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text to lowercase word tokens."""
    return _TOKEN_RE.findall(text.lower())


def build_idf(documents: list[str]) -> dict[str, float]:
    """Build inverse document frequency map from a corpus."""
    n = len(documents)
    if n == 0:
        return {}

    df: dict[str, int] = {}
    for doc in documents:
        tokens = set(tokenize(doc))
        for t in tokens:
            df[t] = df.get(t, 0) + 1

    idf: dict[str, float] = {}
    for term, freq in df.items():
        idf[term] = math.log((n + 1) / (freq + 1)) + 1
    return idf


def tfidf_vector(text: str, idf: dict[str, float]) -> dict[str, float]:
    """Compute TF-IDF vector for a single document using maxTF normalization."""
    tokens = tokenize(text)
    if not tokens:
        return {}

    tf: dict[str, int] = {}
    max_tf = 0
    for t in tokens:
        c = tf.get(t, 0) + 1
        tf[t] = c
        if c > max_tf:
            max_tf = c

    vec: dict[str, float] = {}
    for term, freq in tf.items():
        idf_val = idf.get(term)
        if idf_val is None:
            continue
        vec[term] = (freq / max_tf) * idf_val
    return vec


def cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    if not a or not b:
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for k, v in a.items():
        norm_a += v * v
        bv = b.get(k)
        if bv is not None:
            dot += v * bv

    for v in b.values():
        norm_b += v * v

    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0:
        return 0.0
    return dot / denom


# ─── Emotional Resonance ──────────────────────────────────────────────────


def emotional_resonance(
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    """
    Computes emotional resonance between two VAD profiles.
    Returns [0,1] where 1 = identical emotional tone.
    """
    # Valence: same sign = high resonance; opposite sign = low
    valence_sim = 1.0 - abs(a[0] - b[0]) / 2.0  # [-1..1] → [0..1]
    # Arousal: closer = higher resonance
    arousal_sim = 1.0 - abs(a[1] - b[1])
    return 0.5 * valence_sim + 0.5 * arousal_sim


# ─── Pattern Detection ────────────────────────────────────────────────────


def category_overlap(a: list[str], b: list[str]) -> float:
    """
    Computes category overlap between two sets of emotion categories.
    Returns [0,1] Jaccard similarity.
    """
    if not a or not b:
        return 0.0
    set_a = set(c.lower() for c in a)
    set_b = set(c.lower() for c in b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ─── Bayesian Surprise ────────────────────────────────────────────────────


def bayesian_surprise(
    current: tuple[float, float],
    expected: tuple[float, float],
    expected_variance: float = 0.5,
) -> float:
    """
    Computes Bayesian surprise: how much the current emotional state
    deviates from what would be expected given the latent memory's pattern.

    High surprise = the current context is emotionally unexpected relative
    to the latent memory, which can trigger a reverie (the "fishhook" pulls
    harder when reality doesn't match the stored pattern).

    Returns [0,1] where 1 = maximally surprising.
    """
    valence_diff = current[0] - expected[0]
    arousal_diff = current[1] - expected[1]
    dist_sq = valence_diff * valence_diff + arousal_diff * arousal_diff
    variance = max(expected_variance, 0.01)

    # Surprise = 1 - exp(-distSq / (2 * variance))
    return 1.0 - math.exp(-dist_sq / (2.0 * variance))


# ─── FishhookDetector ─────────────────────────────────────────────────────


class FishhookDetector:
    """Detects fishhooks between current context and latent memory pool."""

    def __init__(self, config: ReverieConfig = DEFAULT_REVERIE_CONFIG) -> None:
        self.config = config
        self._idf: dict[str, float] | None = None
        self._idf_corpus: list[str] = []

    def build_index(self, memories: list[MemoryBlock]) -> None:
        """Build or rebuild the IDF index from a corpus of memory contents."""
        self._idf_corpus = [m.content for m in memories]
        self._idf = build_idf(self._idf_corpus)

    def detect(
        self,
        current_message: str,
        current_emotions: tuple[float, float, list[str]],
        latent_pool: list[MemoryBlock],
        expected_variance: float = 0.5,
    ) -> list[FishhookMatch]:
        """
        Detect fishhooks between a current message and the latent memory pool.

        Args:
            current_message: The incoming message content
            current_emotions: (valence, arousal, categories) of current context
            latent_pool: Memories in the 'latent' consolidation phase
            expected_variance: Variance for Bayesian surprise (default 0.5)

        Returns:
            List of fishhook matches, sorted by resonance (descending)
        """
        if not latent_pool:
            return []

        if self._idf is None:
            self.build_index(latent_pool)
        assert self._idf is not None  # type narrowing after build_index

        now = int(time.time() * 1000)
        matches: list[FishhookMatch] = []

        current_valence, current_arousal, current_categories = current_emotions
        current_vec = tfidf_vector(current_message, self._idf)

        for latent in latent_pool:
            # Only check memories that are reverie-eligible
            if not latent.consolidation.reverie_eligible:
                continue

            features: list[str] = []
            scores: list[tuple[FishhookMatchType, float]] = []

            # 1. Lexical resonance (TF-IDF cosine)
            latent_vec = tfidf_vector(latent.content, self._idf)
            lexical_score = cosine_sim(current_vec, latent_vec)
            if lexical_score >= self.config.fishhook_threshold:
                scores.append(("lexical", lexical_score))
                features.append(f"lexical:{lexical_score:.3f}")

            # 2. Emotional resonance (VAD alignment)
            emo_score = emotional_resonance(
                (current_valence, current_arousal),
                (latent.emotions.valence, latent.emotions.arousal),
            )
            if emo_score >= self.config.fishhook_threshold:
                scores.append(("emotional", emo_score))
                features.append(f"emotional:{emo_score:.3f}")

            # 3. Pattern resonance (emotion category overlap)
            pattern_score = category_overlap(
                current_categories,
                latent.emotions.categories,
            )
            if pattern_score >= self.config.fishhook_threshold:
                scores.append(("pattern", pattern_score))
                features.append(f"pattern:{pattern_score:.3f}")

            # 4. Surprise resonance (Bayesian deviation)
            surprise_score = bayesian_surprise(
                (current_valence, current_arousal),
                (latent.emotions.valence, latent.emotions.arousal),
                expected_variance,
            )
            if surprise_score >= self.config.fishhook_threshold:
                scores.append(("surprise", surprise_score))
                features.append(f"surprise:{surprise_score:.3f}")

            # Need at least one modality above threshold to form a fishhook
            if not scores:
                continue

            composite_score = self._compute_composite_resonance(scores)

            matches.append(
                FishhookMatch(
                    latent_memory_id=latent.id,
                    trigger_memory_id=f"current_{now}",
                    match_type=scores[0][0],
                    resonance_score=composite_score,
                    matched_features=features,
                    timestamp=now,
                )
            )

        matches.sort(key=lambda m: m.resonance_score, reverse=True)
        return matches

    def _compute_composite_resonance(
        self,
        scores: list[tuple[FishhookMatchType, float]],
    ) -> float:
        """Compute weighted composite resonance from individual modality scores."""
        weighted = 0.0
        total_weight = 0.0

        weight_map = {
            "lexical": self.config.lexical_resonance_weight,
            "emotional": self.config.emotional_resonance_weight,
            "pattern": self.config.pattern_resonance_weight,
            "surprise": self.config.surprise_resonance_weight,
        }

        for match_type, score in scores:
            weight = weight_map[match_type]
            weighted += score * weight
            total_weight += weight

        return min(weighted / total_weight, 1.0) if total_weight > 0 else 0.0

    def should_run(self, message_count: int) -> bool:
        """Quick check: should fishhook detection run this message?"""
        return message_count > 0 and message_count % self.config.trigger_interval == 0
