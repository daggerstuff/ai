#!/usr/bin/env python3
"""Tests for MinHash/LSH semantic deduplication module (datasketch-backed)."""

from __future__ import annotations

import os
import sys

# Ensure ai/ is on sys.path
_ai_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ai_root not in sys.path:
    sys.path.insert(0, _ai_root)

import pytest

from pipelines.data_processing.processors.minhash_lsh import (
    DedupResult,
    SemanticDeduplicator,
)


# ---------------------------------------------------------------------------
# SemanticDeduplicator tests
# ---------------------------------------------------------------------------
class TestSemanticDeduplicator:
    def test_unique_records_not_flagged(self):
        dedup = SemanticDeduplicator(jaccard_threshold=0.85, num_perms=64)
        result = dedup.add("r1", frozenset(["alpha", "beta", "gamma"]))
        assert not result.is_duplicate
        result = dedup.add("r2", frozenset(["delta", "epsilon", "zeta"]))
        assert not result.is_duplicate

    def test_identical_records_flagged(self):
        dedup = SemanticDeduplicator(jaccard_threshold=0.85, num_perms=64)
        tokens = frozenset(["alpha", "beta", "gamma", "delta", "epsilon"])
        dedup.add("r1", tokens)
        result = dedup.add("r2", tokens.copy())
        assert result.is_duplicate
        assert result.duplicate_of == "r1"
        assert result.similarity == pytest.approx(1.0, abs=0.1)

    def test_near_duplicate_flagged(self):
        dedup = SemanticDeduplicator(jaccard_threshold=0.5, num_perms=128)
        # 5 shared tokens, 1 unique each → Jaccard = 5/7 ≈ 0.71
        shared = frozenset(["a", "b", "c", "d", "e"])
        dedup.add("r1", shared | frozenset(["f"]))
        result = dedup.add("r2", shared | frozenset(["g"]))
        assert result.is_duplicate
        assert result.duplicate_of == "r1"

    def test_below_threshold_not_flagged(self):
        dedup = SemanticDeduplicator(jaccard_threshold=0.85, num_perms=128)
        # 2 shared, 4 unique → Jaccard = 2/6 ≈ 0.33
        dedup.add("r1", frozenset(["a", "b", "c", "d"]))
        result = dedup.add("r2", frozenset(["a", "b", "e", "f"]))
        assert not result.is_duplicate

    def test_check_without_adding(self):
        dedup = SemanticDeduplicator(jaccard_threshold=0.5, num_perms=64)
        tokens = frozenset(["alpha", "beta", "gamma", "delta"])
        dedup.add("r1", tokens)
        result = dedup.check(tokens.copy())
        assert result.is_duplicate
        assert result.duplicate_of == "r1"

    def test_stats_tracked(self):
        dedup = SemanticDeduplicator(jaccard_threshold=0.5, num_perms=64)
        dedup.add("r1", frozenset(["a", "b", "c"]))
        dedup.add("r2", frozenset(["a", "b", "c"]))
        assert dedup.stats.total_records == 2
        assert dedup.stats.total_duplicates == 1
        assert dedup.stats.total_unique == 1
        assert dedup.stats.total_candidates > 0

    def test_get_clusters(self):
        dedup = SemanticDeduplicator(jaccard_threshold=0.5, num_perms=64)
        tokens = frozenset(["x", "y", "z", "w"])
        dedup.add("r1", tokens)
        dedup.add("r2", tokens.copy())
        dedup.add("r3", tokens.copy())
        clusters = dedup.get_clusters()
        assert len(clusters) == 1
        assert clusters[0] == {"r1", "r2", "r3"}

    def test_multiple_clusters(self):
        dedup = SemanticDeduplicator(jaccard_threshold=0.8, num_perms=128)
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
        dedup = SemanticDeduplicator(jaccard_threshold=0.85, num_perms=64)
        result = dedup.add("r1", frozenset())
        assert not result.is_duplicate
        result = dedup.add("r2", frozenset())
        assert isinstance(result, DedupResult)


# ---------------------------------------------------------------------------
# V7Deduplicator LSH integration tests
# ---------------------------------------------------------------------------
class TestV7DeduplicatorLSH:
    """Test that the V7 dedup engine works with LSH mode enabled."""

    def test_lsh_mode_detects_near_duplicates(self):
        from pipelines.data_processing.orchestration.consolidate_v7 import V7Deduplicator

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
        assert not dedup.process(rec_b)  # near-duplicate dropped

        stats = dedup.stats
        assert stats.total_read == 2
        assert stats.total_kept == 1
        assert stats.near_duplicates == 1
        assert stats.lsh_candidates > 0

    def test_lsh_mode_keeps_disjoint_records(self):
        from pipelines.data_processing.orchestration.consolidate_v7 import V7Deduplicator

        dedup = V7Deduplicator(jaccard_threshold=0.85, use_lsh=True)

        rec_a = {
            "messages": [
                {"role": "user", "content": "I love hiking in the mountains."},
                {"role": "assistant", "content": "That sounds wonderful!"},
            ],
            "metadata": {"stage": "stage1_foundation"},
        }
        rec_b = {
            "messages": [
                {"role": "user", "content": "My favorite programming language is Rust."},
                {"role": "assistant", "content": "Rust has great safety features."},
            ],
            "metadata": {"stage": "stage1_foundation"},
        }

        assert dedup.process(rec_a)
        assert dedup.process(rec_b)  # different content — both kept
        assert dedup.stats.total_kept == 2
        assert dedup.stats.near_duplicates == 0

    def test_lsh_mode_edge_cases_bypass_near_dedup(self):
        from pipelines.data_processing.orchestration.consolidate_v7 import V7Deduplicator

        dedup = V7Deduplicator(jaccard_threshold=0.5, use_lsh=True)

        rec_a = {
            "messages": [
                {"role": "user", "content": "I feel sad today and need help."},
                {"role": "assistant", "content": "I understand. Tell me more."},
            ],
            "is_training_edge_case": True,
            "metadata": {"stage": "stage3_edge_stress_test"},
        }
        rec_b = {
            "messages": [
                {"role": "user", "content": "I feel sad today and need support."},
                {"role": "assistant", "content": "I understand. Tell me more."},
            ],
            "is_training_edge_case": True,
            "metadata": {"stage": "stage3_edge_stress_test"},
        }

        # Near-duplicate edge cases should both be kept (bypass near-dedup)
        assert dedup.process(rec_a)
        assert dedup.process(rec_b)
        assert dedup.stats.edge_cases_preserved == 2
