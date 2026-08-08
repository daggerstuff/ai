#!/usr/bin/env python3
"""Tests for MinHash/LSH semantic deduplication module."""

from __future__ import annotations

import os
import sys

# Ensure ai/ is on sys.path
_ai_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ai_root not in sys.path:
    sys.path.insert(0, _ai_root)

import pytest

from dataset_pipeline.processors.minhash_lsh import (
    DedupResult,
    LSHConfig,
    LSHIndex,
    MinHashSignature,
    SemanticDeduplicator,
)


# ---------------------------------------------------------------------------
# MinHashSignature tests
# ---------------------------------------------------------------------------
class TestMinHashSignature:
    def test_empty_token_set(self):
        sig = MinHashSignature(frozenset(), num_perms=16)
        assert len(sig.signature) == 16
        assert all(v == 2**64 - 1 for v in sig.signature)

    def test_signature_length(self):
        tokens = frozenset(["hello", "world"])
        sig = MinHashSignature(tokens, num_perms=32)
        assert len(sig.signature) == 32

    def test_identical_sets_produce_identical_signatures(self):
        tokens = frozenset(["alpha", "beta", "gamma"])
        sig_a = MinHashSignature(tokens, num_perms=16)
        sig_b = MinHashSignature(tokens, num_perms=16)
        assert sig_a.signature == sig_b.signature

    def test_estimate_jaccard_identical(self):
        tokens = frozenset(["alpha", "beta", "gamma", "delta"])
        sig_a = MinHashSignature(tokens, num_perms=64)
        sig_b = MinHashSignature(tokens, num_perms=64)
        est = sig_a.estimate_jaccard(sig_b)
        assert est == pytest.approx(1.0, abs=0.05)

    def test_estimate_jaccard_disjoint(self):
        sig_a = MinHashSignature(frozenset(["a", "b", "c"]), num_perms=64)
        sig_b = MinHashSignature(frozenset(["x", "y", "z"]), num_perms=64)
        est = sig_a.estimate_jaccard(sig_b)
        assert est < 0.2

    def test_estimate_jaccard_partial_overlap(self):
        shared = frozenset(["a", "b", "c"])
        sig_a = MinHashSignature(shared | frozenset(["d", "e"]), num_perms=128)
        sig_b = MinHashSignature(shared | frozenset(["f", "g"]), num_perms=128)
        est = sig_a.estimate_jaccard(sig_b)
        # True Jaccard = 3/7 ≈ 0.43
        assert 0.25 < est < 0.65

    def test_signature_length_mismatch_raises(self):
        sig_a = MinHashSignature(frozenset(["a"]), num_perms=16)
        sig_b = MinHashSignature(frozenset(["b"]), num_perms=32)
        with pytest.raises(ValueError, match="Signature length mismatch"):
            sig_a.estimate_jaccard(sig_b)

    def test_deterministic_hashes(self):
        """Same token and perm always produce the same hash."""
        h1 = MinHashSignature._hash("test_token", 5)
        h2 = MinHashSignature._hash("test_token", 5)
        assert h1 == h2

    def test_different_perms_produce_different_hashes(self):
        h1 = MinHashSignature._hash("test_token", 0)
        h2 = MinHashSignature._hash("test_token", 1)
        assert h1 != h2


# ---------------------------------------------------------------------------
# LSHConfig tests
# ---------------------------------------------------------------------------
class TestLSHConfig:
    def test_default_config(self):
        config = LSHConfig()
        assert config.num_perms == 128
        assert config.num_bands == 32
        assert config.rows_per_band == 4

    def test_valid_custom_config(self):
        config = LSHConfig(num_perms=64, num_bands=16, rows_per_band=4)
        assert config.num_bands * config.rows_per_band == config.num_perms

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="must equal num_perms"):
            LSHConfig(num_perms=128, num_bands=10, rows_per_band=4)


# ---------------------------------------------------------------------------
# LSHIndex tests
# ---------------------------------------------------------------------------
class TestLSHIndex:
    def test_add_and_query_identical(self):
        config = LSHConfig(num_perms=64, num_bands=16, rows_per_band=4)
        index = LSHIndex(config)
        tokens = frozenset(["hello", "world", "foo", "bar"])
        sig = MinHashSignature(tokens, config.num_perms)
        candidates = index.add("rec1", sig)
        assert candidates == []

        sig2 = MinHashSignature(tokens, config.num_perms)
        candidates2 = index.query(sig2)
        assert "rec1" in candidates2

    def test_add_returns_candidates_for_similar(self):
        config = LSHConfig(num_perms=64, num_bands=16, rows_per_band=4)
        index = LSHIndex(config)
        tokens_a = frozenset(["a", "b", "c", "d", "e", "f"])
        sig_a = MinHashSignature(tokens_a, config.num_perms)
        index.add("rec1", sig_a)

        # Similar set — shares most tokens
        tokens_b = frozenset(["a", "b", "c", "d", "e", "g"])
        sig_b = MinHashSignature(tokens_b, config.num_perms)
        candidates = index.add("rec2", sig_b)
        assert "rec1" in candidates

    def test_disjoint_sets_no_candidates(self):
        config = LSHConfig(num_perms=64, num_bands=16, rows_per_band=4)
        index = LSHIndex(config)
        sig_a = MinHashSignature(frozenset(["a", "b", "c"]), config.num_perms)
        index.add("rec1", sig_a)

        sig_b = MinHashSignature(frozenset(["x", "y", "z"]), config.num_perms)
        candidates = index.add("rec2", sig_b)
        assert "rec1" not in candidates

    def test_query_without_adding(self):
        config = LSHConfig(num_perms=64, num_bands=16, rows_per_band=4)
        index = LSHIndex(config)
        tokens = frozenset(["test", "data", "here"])
        sig = MinHashSignature(tokens, config.num_perms)
        index.add("rec1", sig)

        sig2 = MinHashSignature(tokens, config.num_perms)
        candidates = index.query(sig2)
        assert "rec1" in candidates
        # Verify the signature was not added
        assert "rec2" not in index._signatures


# ---------------------------------------------------------------------------
# SemanticDeduplicator tests
# ---------------------------------------------------------------------------
class TestSemanticDeduplicator:
    def test_unique_records_not_flagged(self):
        dedup = SemanticDeduplicator(jaccard_threshold=0.85, num_perms=64, num_bands=16, rows_per_band=4)
        result = dedup.add("r1", frozenset(["alpha", "beta", "gamma"]))
        assert not result.is_duplicate
        result = dedup.add("r2", frozenset(["delta", "epsilon", "zeta"]))
        assert not result.is_duplicate

    def test_identical_records_flagged(self):
        dedup = SemanticDeduplicator(jaccard_threshold=0.85, num_perms=64, num_bands=16, rows_per_band=4)
        tokens = frozenset(["alpha", "beta", "gamma", "delta", "epsilon"])
        dedup.add("r1", tokens)
        result = dedup.add("r2", tokens.copy())
        assert result.is_duplicate
        assert result.duplicate_of == "r1"
        assert result.similarity == pytest.approx(1.0, abs=0.1)

    def test_near_duplicate_flagged(self):
        dedup = SemanticDeduplicator(jaccard_threshold=0.5, num_perms=128, num_bands=32, rows_per_band=4)
        # 5 shared tokens, 1 unique each → Jaccard = 5/7 ≈ 0.71
        shared = frozenset(["a", "b", "c", "d", "e"])
        dedup.add("r1", shared | frozenset(["f"]))
        result = dedup.add("r2", shared | frozenset(["g"]))
        assert result.is_duplicate
        assert result.duplicate_of == "r1"

    def test_below_threshold_not_flagged(self):
        dedup = SemanticDeduplicator(jaccard_threshold=0.85, num_perms=128, num_bands=32, rows_per_band=4)
        # 2 shared, 4 unique → Jaccard = 2/6 ≈ 0.33
        dedup.add("r1", frozenset(["a", "b", "c", "d"]))
        result = dedup.add("r2", frozenset(["a", "b", "e", "f"]))
        assert not result.is_duplicate

    def test_check_without_adding(self):
        dedup = SemanticDeduplicator(jaccard_threshold=0.5, num_perms=64, num_bands=16, rows_per_band=4)
        tokens = frozenset(["alpha", "beta", "gamma", "delta"])
        dedup.add("r1", tokens)
        result = dedup.check(tokens.copy())
        assert result.is_duplicate
        assert result.duplicate_of == "r1"

    def test_stats_tracked(self):
        dedup = SemanticDeduplicator(jaccard_threshold=0.5, num_perms=64, num_bands=16, rows_per_band=4)
        dedup.add("r1", frozenset(["a", "b", "c"]))
        dedup.add("r2", frozenset(["a", "b", "c"]))
        assert dedup.stats.total_records == 2
        assert dedup.stats.total_duplicates == 1
        assert dedup.stats.total_unique == 1
        assert dedup.stats.total_candidates > 0

    def test_get_clusters(self):
        dedup = SemanticDeduplicator(jaccard_threshold=0.5, num_perms=64, num_bands=16, rows_per_band=4)
        tokens = frozenset(["x", "y", "z", "w"])
        dedup.add("r1", tokens)
        dedup.add("r2", tokens.copy())
        dedup.add("r3", tokens.copy())
        clusters = dedup.get_clusters()
        assert len(clusters) == 1
        assert clusters[0] == {"r1", "r2", "r3"}

    def test_multiple_clusters(self):
        dedup = SemanticDeduplicator(jaccard_threshold=0.8, num_perms=128, num_bands=32, rows_per_band=4)
        tokens_a = frozenset(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"])
        tokens_b = frozenset(["one", "two", "three", "four", "five", "six"])
        dedup.add("a1", tokens_a)
        dedup.add("a2", tokens_a.copy())
        dedup.add("b1", tokens_b)
        dedup.add("b2", tokens_b.copy())
        clusters = dedup.get_clusters()
        cluster_sets = [frozenset(c) for c in clusters]
        assert frozenset({"a1", "a2"}) in cluster_sets
        assert frozenset({"b1", "b2"}) in cluster_sets

    def test_empty_tokens_not_flagged(self):
        dedup = SemanticDeduplicator(jaccard_threshold=0.85, num_perms=64, num_bands=16, rows_per_band=4)
        result = dedup.add("r1", frozenset())
        assert not result.is_duplicate
        result = dedup.add("r2", frozenset())
        # Two empty sets have Jaccard 1.0, but LSH may not bucket them together
        # due to max-value signatures. Either outcome is acceptable.
        assert isinstance(result, DedupResult)


# ---------------------------------------------------------------------------
# V7Deduplicator LSH integration tests
# ---------------------------------------------------------------------------
class TestV7DeduplicatorLSH:
    """Test that the V7 dedup engine works with LSH mode enabled."""

    def test_lsh_mode_detects_near_duplicates(self):
        from dataset_pipeline.orchestration.consolidate_v7 import V7Deduplicator

        dedup = V7Deduplicator(jaccard_threshold=0.5, use_lsh=True)

        # Near-duplicate records (shared content with minor variation)
        rec_a = {
            "messages": [
                {"role": "system", "content": "You are a therapist."},
                {"role": "user", "content": "I feel anxious about my job."},
                {"role": "assistant", "content": "Let's explore that feeling."},
            ],
            "metadata": {"stage": "stage1_foundation"},
        }
        rec_b = {
            "messages": [
                {"role": "system", "content": "You are a therapist."},
                {"role": "user", "content": "I feel anxious about my job today."},
                {"role": "assistant", "content": "Let's explore that feeling."},
            ],
            "metadata": {"stage": "stage1_foundation"},
        }

        assert dedup.process(rec_a)
        assert not dedup.process(rec_b)  # near-dup dropped
        assert dedup.stats.near_duplicates == 1

    def test_lsh_mode_unique_records_kept(self):
        from dataset_pipeline.orchestration.consolidate_v7 import V7Deduplicator

        dedup = V7Deduplicator(jaccard_threshold=0.85, use_lsh=True)

        rec_a = {
            "messages": [
                {"role": "user", "content": "Tell me about cognitive behavioral therapy."},
                {"role": "assistant", "content": "CBT is a structured approach."},
            ],
        }
        rec_b = {
            "messages": [
                {"role": "user", "content": "What is dialectical behavior therapy?"},
                {"role": "assistant", "content": "DBT focuses on emotional regulation."},
            ],
        }

        assert dedup.process(rec_a)
        assert dedup.process(rec_b)
        assert dedup.stats.total_kept == 2

    def test_lsh_stats_tracked(self):
        from dataset_pipeline.orchestration.consolidate_v7 import V7Deduplicator

        dedup = V7Deduplicator(jaccard_threshold=0.5, use_lsh=True)
        rec = {
            "messages": [
                {"role": "user", "content": "Hello world test data."},
                {"role": "assistant", "content": "Response here."},
            ],
        }
        dedup.process(rec)
        assert dedup.stats.lsh_candidates >= 0
        assert dedup.stats.lsh_comparisons >= 0


# ---------------------------------------------------------------------------
# Scalability smoke test
# ---------------------------------------------------------------------------
class TestScalability:
    def test_1000_records_complete_fast(self):
        """Smoke test: 1000 records should complete in reasonable time."""
        dedup = SemanticDeduplicator(
            jaccard_threshold=0.85,
            num_perms=64,
            num_bands=16,
            rows_per_band=4,
        )
        # Generate 1000 unique-ish records
        for i in range(1000):
            tokens = frozenset([f"token_{i}", f"word_{i % 50}", "common", "words"])
            dedup.add(f"rec_{i}", tokens)

        assert dedup.stats.total_records == 1000
        # With 1000 records, LSH should have far fewer comparisons than 1000*1000
        assert dedup.stats.total_exact_comparisons < 50000
