#!/usr/bin/env python3
"""MinHash + LSH for scalable semantic near-duplicate detection.

Uses ``datasketch`` (numpy-backed) for MinHash signatures and LSH indexing,
providing near-linear performance on 500k+ records.

Architecture:
  1. MinHash signature — compact `num_perms`-length signature approximating
     Jaccard similarity between token sets.
  2. LSH index — band-based locality-sensitive hashing that buckets
     signatures into candidate groups for exact comparison.
  3. SemanticDeduplicator — orchestrates MinHash + LSH to identify
     near-duplicate clusters and decide which records to keep.

Usage:
    dedup = SemanticDeduplicator(jaccard_threshold=0.85)
    dedup.add("rec1", tokens_a)
    dedup.add("rec2", tokens_b)
    clusters = dedup.get_clusters()  # list of sets of record IDs
    result = dedup.check(tokens_c)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from datasketch import MinHash, MinHashLSH


# ---------------------------------------------------------------------------
# Result / stats dataclasses
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


# ---------------------------------------------------------------------------
# Semantic deduplicator (datasketch-backed)
# ---------------------------------------------------------------------------
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
    ) -> None:
        self.jaccard_threshold = jaccard_threshold
        self.num_perms = num_perms
        self.lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perms)
        self.stats = DuplicationStats()
        # record_id -> MinHash signature (for cluster lookups)
        self._signatures: dict[str, MinHash] = {}
        # canonical_id -> set of duplicate IDs
        self._clusters: dict[str, set[str]] = {}

    def _make_minhash(self, tokens: frozenset[str]) -> MinHash:
        """Build a datasketch MinHash from a token set."""
        mh = MinHash(num_perm=self.num_perms)
        for token in tokens:
            mh.update(token.encode("utf-8"))
        return mh

    def add(self, record_id: str, tokens: frozenset[str]) -> DedupResult:
        """Add a record and check if it's a near-duplicate of existing records.

        Returns DedupResult indicating whether this record is a duplicate
        and which existing record it duplicates.
        """
        self.stats.total_records += 1
        mh = self._make_minhash(tokens)

        candidates = self.lsh.query(mh)
        self.stats.total_candidates += len(candidates)

        for candidate_id in candidates:
            self.stats.total_exact_comparisons += 1
            candidate_mh = self._signatures.get(candidate_id)
            if candidate_mh is None:
                continue
            sim = mh.jaccard(candidate_mh)

            if sim >= self.jaccard_threshold:
                # Early-exit: any match is enough for dedup
                self.stats.total_duplicates += 1
                # Track cluster but don't bloat LSH index with duplicates
                self._clusters.setdefault(candidate_id, set()).add(record_id)
                return DedupResult(
                    is_duplicate=True,
                    duplicate_of=candidate_id,
                    similarity=sim,
                    candidates_checked=len(candidates),
                )

        self.stats.total_unique += 1
        self.lsh.insert(record_id, mh)
        self._signatures[record_id] = mh
        return DedupResult(
            is_duplicate=False,
            duplicate_of=None,
            similarity=0.0,
            candidates_checked=len(candidates),
        )

    def check(self, tokens: frozenset[str]) -> DedupResult:
        """Check if a record would be a duplicate without adding it."""
        mh = self._make_minhash(tokens)
        candidates = self.lsh.query(mh)
        self.stats.total_candidates += len(candidates)

        best_match: str | None = None
        best_similarity = 0.0

        for candidate_id in candidates:
            self.stats.total_exact_comparisons += 1
            candidate_mh = self._signatures.get(candidate_id)
            if candidate_mh is None:
                continue
            sim = mh.jaccard(candidate_mh)

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
        """Return clusters of near-duplicate record IDs."""
        clusters = []
        for canonical, dups in self._clusters.items():
            cluster = {canonical} | dups
            clusters.append(cluster)
        self.stats.clusters = clusters
        return clusters

    def _find_canonical(self, record_id: str) -> str:
        """Find the canonical (first-seen) record for a given record ID."""
        for canonical, dups in self._clusters.items():
            if record_id in dups:
                return canonical
        return record_id
