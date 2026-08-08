#!/usr/bin/env python3
"""MinHash + LSH for scalable semantic near-duplicate detection.

Replaces O(n*W) windowed Jaccard with O(n) LSH-banded MinHash, enabling
near-duplicate detection across 10k+ records without pairwise comparison.

Architecture:
  1. MinHashSignature — compact `num_perms`-length signature approximating
     Jaccard similarity between token sets.
  2. LSHIndex — band-based locality-sensitive hashing that buckets
     signatures into candidate groups for exact comparison.
  3. SemanticDeduplicator — orchestrates MinHash + LSH to identify
     near-duplicate clusters and decide which records to keep.

Usage:
    dedup = SemanticDeduplicator(jaccard_threshold=0.85)
    dedup.add("rec1", tokens_a)
    dedup.add("rec2", tokens_b)
    clusters = dedup.get_clusters()  # list of sets of record IDs
    is_dup, dup_of = dedup.check("rec3", tokens_c)
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# MinHash signature
# ---------------------------------------------------------------------------
class MinHashSignature:
    """MinHash signature for a token set.

    Uses `num_perms` independent hash functions to produce a fixed-length
    signature where `EstimateJaccard(sig_a, sig_b) ≈ Jaccard(A, B)`.
    """

    def __init__(self, tokens: frozenset[str], num_perms: int = 128) -> None:
        self.num_perms = num_perms
        self.signature = self._compute(tokens)

    def _compute(self, tokens: frozenset[str]) -> list[int]:
        if not tokens:
            return [2**64 - 1] * self.num_perms
        sig: list[int] = [2**64 - 1] * self.num_perms
        for token in tokens:
            for i in range(self.num_perms):
                h = self._hash(token, i)
                sig[i] = min(sig[i], h)
        return sig

    @staticmethod
    def _hash(token: str, perm: int) -> int:
        """Produce a 64-bit hash for a token under permutation `perm`."""
        payload = struct.pack(">Q", perm) + token.encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return struct.unpack(">Q", digest)[0]

    def estimate_jaccard(self, other: MinHashSignature) -> float:
        """Estimate Jaccard similarity from two MinHash signatures."""
        if self.num_perms != other.num_perms:
            raise ValueError("Signature length mismatch")
        matches = sum(1 for a, b in zip(self.signature, other.signature, strict=True) if a == b)
        return matches / self.num_perms


# ---------------------------------------------------------------------------
# LSH index
# ---------------------------------------------------------------------------
@dataclass
class LSHConfig:
    """Configuration for LSH banding.

    The probability that a pair with Jaccard similarity `s` becomes a
    candidate is: 1 - (1 - s^rows_per_band)^num_bands

    With `num_perms=128`, `num_bands=32`, `rows_per_band=4`:
      - s=0.85 → P(candidate) ≈ 0.94
      - s=0.50 → P(candidate) ≈ 0.03
    """

    num_perms: int = 128
    num_bands: int = 32
    rows_per_band: int = 4

    def __post_init__(self) -> None:
        if self.num_bands * self.rows_per_band != self.num_perms:
            raise ValueError(
                f"num_bands * rows_per_band ({self.num_bands * self.rows_per_band}) "
                f"must equal num_perms ({self.num_perms})"
            )


class LSHIndex:
    """Locality-sensitive hash index for MinHash signatures.

    Splits each signature into `num_bands` bands of `rows_per_band` rows.
    Records sharing a band hash become candidates for exact comparison.
    """

    def __init__(self, config: LSHConfig | None = None) -> None:
        self.config = config or LSHConfig()
        # band_index -> band_hash -> set of record IDs
        self._bands: list[dict[int, set[str]]] = [
            {} for _ in range(self.config.num_bands)
        ]
        # record_id -> signature (for verification)
        self._signatures: dict[str, MinHashSignature] = {}

    def add(self, record_id: str, signature: MinHashSignature) -> list[str]:
        """Add a record and return candidate IDs (records sharing a band)."""
        candidates: set[str] = set()
        for band_idx in range(self.config.num_bands):
            start = band_idx * self.config.rows_per_band
            end = start + self.config.rows_per_band
            band_hash = self._band_hash(signature, start, end)
            bucket = self._bands[band_idx].setdefault(band_hash, set())
            candidates.update(bucket)
            bucket.add(record_id)
        self._signatures[record_id] = signature
        return [cid for cid in candidates if cid != record_id]

    def query(self, signature: MinHashSignature) -> list[str]:
        """Return candidate IDs for a signature without adding it."""
        candidates: set[str] = set()
        for band_idx in range(self.config.num_bands):
            start = band_idx * self.config.rows_per_band
            end = start + self.config.rows_per_band
            band_hash = self._band_hash(signature, start, end)
            bucket = self._bands[band_idx].get(band_hash)
            if bucket:
                candidates.update(bucket)
        return list(candidates)

    @staticmethod
    def _band_hash(signature: MinHashSignature, start: int, end: int) -> int:
        """Hash a contiguous slice of the signature into a single band key."""
        slice_bytes = b"".join(
            struct.pack(">Q", v) for v in signature.signature[start:end]
        )
        return struct.unpack(">Q", hashlib.blake2b(slice_bytes, digest_size=8).digest())[0]


# ---------------------------------------------------------------------------
# Semantic deduplicator
# ---------------------------------------------------------------------------
@dataclass
class DedupResult:
    """Result of checking a record against the index."""

    is_duplicate: bool
    duplicate_of: str | None = None
    similarity: float = 0.0
    candidates_checked: int = 0


@dataclass
class DuplicationStats:
    total_records: int = 0
    total_duplicates: int = 0
    total_unique: int = 0
    total_candidates: int = 0
    total_exact_comparisons: int = 0
    clusters: list[set[str]] = field(default_factory=list)


class SemanticDeduplicator:
    """Scalable near-duplicate detection using MinHash + LSH.

    Flow:
      1. Tokenize record → token set
      2. Generate MinHash signature (compact, fixed-length)
      3. Query LSH index for candidates (band collision)
      4. Verify each candidate with estimated Jaccard
      5. If above threshold → mark as duplicate
      6. Add signature to LSH index for future queries
    """

    def __init__(
        self,
        jaccard_threshold: float = 0.85,
        num_perms: int = 128,
        num_bands: int = 32,
        rows_per_band: int = 4,
    ) -> None:
        self.jaccard_threshold = jaccard_threshold
        self.config = LSHConfig(
            num_perms=num_perms,
            num_bands=num_bands,
            rows_per_band=rows_per_band,
        )
        self.lsh = LSHIndex(self.config)
        self.stats = DuplicationStats()
        # record_id -> set of duplicate IDs that map to it
        self._clusters: dict[str, set[str]] = {}

    def add(self, record_id: str, tokens: frozenset[str]) -> DedupResult:
        """Add a record and check if it's a near-duplicate of existing records.

        Returns DedupResult indicating whether this record is a duplicate
        and which existing record it duplicates.
        """
        self.stats.total_records += 1
        signature = MinHashSignature(tokens, self.config.num_perms)

        candidates = self.lsh.query(signature)
        self.stats.total_candidates += len(candidates)

        best_match: str | None = None
        best_similarity = 0.0

        for candidate_id in candidates:
            self.stats.total_exact_comparisons += 1
            candidate_sig = self.lsh._signatures[candidate_id]
            sim = signature.estimate_jaccard(candidate_sig)

            if sim >= self.jaccard_threshold and sim > best_similarity:
                best_match = candidate_id
                best_similarity = sim

        if best_match is not None:
            self.stats.total_duplicates += 1
            self.lsh.add(record_id, signature)
            # Merge into existing cluster or create new one.
            # If best_match is already a duplicate of another record,
            # add to that record's cluster instead.
            canonical = self._find_canonical(best_match)
            self._clusters.setdefault(canonical, set()).add(record_id)
            if canonical != best_match:
                self._clusters[canonical].add(best_match)
            return DedupResult(
                is_duplicate=True,
                duplicate_of=best_match,
                similarity=best_similarity,
                candidates_checked=len(candidates),
            )

        self.stats.total_unique += 1
        self.lsh.add(record_id, signature)
        return DedupResult(
            is_duplicate=False,
            duplicate_of=None,
            similarity=0.0,
            candidates_checked=len(candidates),
        )

    def check(self, tokens: frozenset[str]) -> DedupResult:
        """Check if a record would be a duplicate without adding it."""
        signature = MinHashSignature(tokens, self.config.num_perms)
        candidates = self.lsh.query(signature)
        self.stats.total_candidates += len(candidates)

        best_match: str | None = None
        best_similarity = 0.0

        for candidate_id in candidates:
            self.stats.total_exact_comparisons += 1
            candidate_sig = self.lsh._signatures[candidate_id]
            sim = signature.estimate_jaccard(candidate_sig)

            if sim >= self.jaccard_threshold and sim > best_similarity:
                best_match = candidate_id
                best_similarity = sim

        return DedupResult(
            is_duplicate=best_match is not None,
            duplicate_of=best_match,
            similarity=best_similarity,
            candidates_checked=len(candidates),
        )

    def get_clusters(self) -> list[set[str]]:
        """Return clusters of near-duplicate record IDs.

        Each cluster contains the canonical record ID (first seen) and
        all records that were detected as its duplicates.
        """
        clusters = []
        for canonical, dups in self._clusters.items():
            cluster = {canonical} | dups
            clusters.append(cluster)
        self.stats.clusters = clusters
        return clusters

    def _find_canonical(self, record_id: str) -> str:
        """Find the canonical (first-seen) record for a given record ID.

        If `record_id` is a duplicate of another record, walk the chain
        back to the original canonical record.
        """
        for canonical, dups in self._clusters.items():
            if record_id in dups:
                return canonical
        return record_id
