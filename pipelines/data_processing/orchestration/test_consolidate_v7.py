"""Unit tests for V7 dataset consolidation: hashing, dedup, normalization, I/O."""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure ai/ is on sys.path for `dataset_pipeline.*` imports
_ai_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ai_root not in sys.path:
    sys.path.insert(0, _ai_root)

from pipelines.data_processing.orchestration.consolidate_v7 import (
    DEFAULT_SYSTEM_PROMPT,
    V7Deduplicator,
    _extract_text,
    _get_source,
    _get_stage,
    _get_stage_priority,
    _get_task_type,
    _is_edge_case,
    _verify_chatml,
    _write_jsonl,
    _write_shards,
    compute_primary_hash,
    compute_token_set,
    jaccard_similarity,
    normalize_to_v7,
    run_consolidation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _chatml(
    user: str = "Hello",
    assistant: str = "Hi there",
    source: str = "test",
    task_type: str = "therapy_response_generation",
    stage: str = "stage1_foundation",
    is_edge: bool = False,
) -> dict:
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": source,
        "task_type": task_type,
        "metadata": {"stage": stage},
        "is_training_edge_case": is_edge,
    }


def _raw_msg(user: str, assistant: str) -> dict:
    """Build a minimal record without system prompt for token-set tests."""
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def _normalize(rec: dict, src: str = "test.jsonl") -> tuple[dict, bool]:
    """Normalize and assert non-None result, returning typed dict."""
    normalized, reformatted = normalize_to_v7(rec, src)
    assert normalized is not None
    return normalized, reformatted


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
class TestComputePrimaryHash:
    def test_same_content_same_hash(self):
        rec = _chatml("Hello", "World")
        assert compute_primary_hash(rec) == compute_primary_hash(rec)

    def test_different_content_different_hash(self):
        a = _chatml("Hello", "World")
        b = _chatml("Hello", "Different")
        assert compute_primary_hash(a) != compute_primary_hash(b)

    def test_case_insensitive(self):
        a = _chatml("Hello", "World")
        b = _chatml("hello", "world")
        assert compute_primary_hash(a) == compute_primary_hash(b)

    def test_empty_messages_returns_empty_hash(self):
        h = compute_primary_hash({})
        assert len(h) == 64  # SHA-256 hex digest

    def test_order_matters(self):
        a = {"messages": [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]}
        b = {"messages": [{"role": "assistant", "content": "B"}, {"role": "user", "content": "A"}]}
        assert compute_primary_hash(a) != compute_primary_hash(b)


class TestComputeTokenSet:
    def test_returns_frozenset(self):
        ts = compute_token_set("hello world hello")
        assert isinstance(ts, frozenset)

    def test_deduplicates_tokens(self):
        ts = compute_token_set("hello world hello")
        assert ts == frozenset({"hello", "world"})

    def test_case_insensitive(self):
        assert compute_token_set("Hello WORLD") == compute_token_set("hello world")


class TestJaccardSimilarity:
    def test_identical_sets(self):
        a = frozenset(["hello", "world"])
        assert jaccard_similarity(a, a) == 1.0

    def test_disjoint_sets(self):
        a = frozenset(["hello"])
        b = frozenset(["world"])
        assert jaccard_similarity(a, b) == 0.0

    def test_both_empty(self):
        assert jaccard_similarity(frozenset(), frozenset()) == 1.0

    def test_one_empty(self):
        a = frozenset(["hello"])
        assert jaccard_similarity(a, frozenset()) == 0.0

    def test_partial_overlap(self):
        a = frozenset(["a", "b", "c"])
        b = frozenset(["b", "c", "d"])
        assert jaccard_similarity(a, b) == 0.5


# ---------------------------------------------------------------------------
# Record inspection helpers
# ---------------------------------------------------------------------------
class TestRecordHelpers:
    def test_is_edge_case_true(self):
        assert _is_edge_case({"is_training_edge_case": True})

    def test_is_edge_case_false(self):
        assert not _is_edge_case({"is_training_edge_case": False})
        assert not _is_edge_case({})

    def test_get_stage_from_metadata(self):
        assert _get_stage({"metadata": {"stage": "stage2_therapeutic_expertise"}}) == "stage2_therapeutic_expertise"

    def test_get_stage_default(self):
        assert _get_stage({}) == "supplementary"
        assert _get_stage({"metadata": "not-a-dict"}) == "supplementary"

    def test_stage_priority_mapping(self):
        assert _get_stage_priority({"metadata": {"stage": "stage4_voice_persona"}}) == 5
        assert _get_stage_priority({"metadata": {"stage": "supplementary"}}) == 1
        assert _get_stage_priority({}) == 1

    def test_get_source_from_top_level(self):
        assert _get_source({"source": "github"}) == "github"

    def test_get_source_from_metadata(self):
        assert _get_source({"metadata": {"source": "hf"}}) == "hf"
        assert _get_source({"metadata": {"source_channel": "s3"}}) == "s3"

    def test_get_source_unknown(self):
        assert _get_source({}) == "unknown"

    def test_get_task_type_from_top_level(self):
        assert _get_task_type({"task_type": "risk_assessment"}) == "risk_assessment"

    def test_get_task_type_from_metadata(self):
        assert _get_task_type({"metadata": {"task_type": "empathy_scoring"}}) == "empathy_scoring"

    def test_get_task_type_default(self):
        assert _get_task_type({}) == "therapy_response_generation"

    def test_extract_text_from_messages(self):
        rec = {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
        assert _extract_text(rec) == "Hi Hello"

    def test_extract_text_from_dpo(self):
        rec = {"prompt": "Q", "chosen": "A", "rejected": "B"}
        assert _extract_text(rec) == "Q A B"

    def test_extract_text_from_instruction(self):
        rec = {"instruction": "What?", "output": "Answer"}
        assert _extract_text(rec) == "What? Answer"


# ---------------------------------------------------------------------------
# ChatML verification
# ---------------------------------------------------------------------------
class TestVerifyChatML:
    def test_valid_chatml(self):
        assert _verify_chatml(_chatml())

    def test_empty_messages(self):
        assert not _verify_chatml({})

    def test_invalid_role(self):
        rec = {"messages": [{"role": "narrator", "content": "x"}]}
        assert not _verify_chatml(rec)

    def test_empty_content(self):
        rec = {"messages": [{"role": "user", "content": ""}]}
        assert not _verify_chatml(rec)

    def test_inst_boundary_valid(self):
        rec = {"messages": [
            {"role": "user", "content": "[INST] Hello [/INST]"},
            {"role": "assistant", "content": "Hi"},
        ]}
        assert _verify_chatml(rec)

    def test_inst_without_close_tag_still_valid(self):
        # The boundary check only rejects [/INST] that doesn't match the regex;
        # [INST] without [/INST] is not rejected (no [/INST] to validate).
        rec = {"messages": [
            {"role": "user", "content": "[INST] Hello without closing"},
            {"role": "assistant", "content": "Hi"},
        ]}
        assert _verify_chatml(rec)


# ---------------------------------------------------------------------------
# V7 normalization
# ---------------------------------------------------------------------------
class TestNormalizeToV7:
    def test_valid_chatml_passes_through(self):
        rec = _chatml()
        normalized, reformatted = _normalize(rec)
        assert not reformatted
        assert "provenance" in normalized
        assert "v7_normalize" in normalized["provenance"]["transformations"]

    def test_instruction_output_reformatted(self):
        rec = {"instruction": "What is CBT?", "output": "Cognitive Behavioral Therapy."}
        normalized, reformatted = _normalize(rec)
        assert reformatted
        assert len(normalized["messages"]) == 3  # system + user + assistant
        assert normalized["messages"][1]["role"] == "user"
        assert normalized["messages"][1]["content"] == "What is CBT?"

    def test_dpo_reformatted(self):
        rec = {"prompt": "Pick one", "chosen": "A", "rejected": "B"}
        normalized, reformatted = _normalize(rec)
        assert reformatted
        assert normalized["messages"][1]["content"] == "Pick one"

    def test_garbage_returns_none(self):
        normalized, reformatted = normalize_to_v7({"random": "data"}, "test.jsonl")
        assert normalized is None
        assert not reformatted

    def test_enrich_adds_system_prompt(self):
        rec = {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
        normalized, _ = _normalize(rec)
        assert normalized["messages"][0]["role"] == "system"
        assert normalized["messages"][0]["content"] == DEFAULT_SYSTEM_PROMPT

    def test_enrich_defaults(self):
        rec = {"messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]}
        normalized, _ = _normalize(rec)
        assert normalized["diagnostic_tag"] is None
        assert normalized["demographic_tags"] == []
        assert normalized["linguistic_style"] == "mixed"
        assert normalized["clinical_reviewed"] is False

    def test_enrich_fixes_invalid_linguistic_style(self):
        rec = {"messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ], "linguistic_style": "invalid"}
        normalized, _ = _normalize(rec)
        assert normalized["linguistic_style"] == "mixed"

    def test_enrich_fixes_invalid_task_type(self):
        rec = {"messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ], "task_type": "nonexistent"}
        normalized, _ = _normalize(rec)
        assert normalized["task_type"] == "therapy_response_generation"

    def test_provenance_source_url(self):
        rec = _chatml()
        normalized, _ = _normalize(rec, "my_source.jsonl")
        assert normalized["provenance"]["source_url"] == "my_source.jsonl"


# ---------------------------------------------------------------------------
# V7Deduplicator
# ---------------------------------------------------------------------------
class TestV7Deduplicator:
    def test_unique_records_all_kept(self):
        dedup = V7Deduplicator(jaccard_threshold=0.99)
        for i in range(5):
            dedup.process(_chatml(user=f"msg-{i}", assistant=f"resp-{i}"))
        assert dedup.stats.total_kept == 5
        assert dedup.stats.exact_duplicates == 0
        assert dedup.stats.near_duplicates == 0

    def test_exact_duplicate_dropped(self):
        dedup = V7Deduplicator(jaccard_threshold=0.99)
        rec = _chatml()
        dedup.process(rec)
        dedup.process(rec.copy())
        assert dedup.stats.total_kept == 1
        assert dedup.stats.exact_duplicates == 1

    def test_stage_conflict_higher_wins(self):
        dedup = V7Deduplicator(jaccard_threshold=0.99)
        low = _chatml(user="same", assistant="same", stage="stage1_foundation")
        high = _chatml(user="same", assistant="same", stage="stage4_voice_persona")
        dedup.process(low)
        dedup.process(high)
        assert dedup.stats.total_kept == 1  # only 1 unique record
        assert dedup.stats.stage_conflicts_resolved == 1
        kept = dedup.kept_records[0]
        assert kept["metadata"]["stage"] == "stage4_voice_persona"

    def test_stage_conflict_lower_does_not_replace(self):
        dedup = V7Deduplicator(jaccard_threshold=0.99)
        high = _chatml(user="same", assistant="same", stage="stage4_voice_persona")
        low = _chatml(user="same", assistant="same", stage="stage1_foundation")
        dedup.process(high)
        dedup.process(low)
        assert dedup.stats.total_kept == 1
        assert dedup.stats.exact_duplicates == 1
        kept = dedup.kept_records[0]
        assert kept["metadata"]["stage"] == "stage4_voice_persona"

    def test_near_duplicate_dropped(self):
        dedup = V7Deduplicator(jaccard_threshold=0.5)
        a = _chatml(user="the quick brown fox jumps", assistant="over the lazy dog")
        b = _chatml(user="the quick brown fox jumps", assistant="over the lazy dog today")
        dedup.process(a)
        dedup.process(b)
        assert dedup.stats.total_kept == 1
        assert dedup.stats.near_duplicates == 1

    def test_near_duplicate_below_threshold_kept(self):
        dedup = V7Deduplicator(jaccard_threshold=0.99)
        a = _chatml(user="the quick brown fox", assistant="over the lazy dog")
        b = _chatml(user="completely different words", assistant="entirely unique text")
        dedup.process(a)
        dedup.process(b)
        assert dedup.stats.total_kept == 2

    def test_edge_case_bypasses_near_dedup(self):
        dedup = V7Deduplicator(jaccard_threshold=0.3)
        a = _chatml(user="the quick brown fox jumps", assistant="over the lazy dog", is_edge=True)
        b = _chatml(user="the quick brown fox jumps", assistant="over the lazy dog today")
        dedup.process(a)
        dedup.process(b)
        assert dedup.stats.edge_cases_preserved == 1
        assert dedup.stats.total_kept == 2

    def test_edge_case_exact_dup_dropped(self):
        dedup = V7Deduplicator(jaccard_threshold=0.99)
        rec = _chatml(is_edge=True)
        dedup.process(rec)
        dedup.process(rec.copy())
        assert dedup.stats.total_kept == 1
        assert dedup.stats.exact_duplicates == 1
        assert dedup.stats.edge_cases_preserved == 1

    def test_edge_case_then_non_edge_same_hash_dropped(self):
        dedup = V7Deduplicator(jaccard_threshold=0.99)
        edge = _chatml(user="same", assistant="same", is_edge=True)
        normal = _chatml(user="same", assistant="same", is_edge=False)
        dedup.process(edge)
        dedup.process(normal)
        assert dedup.stats.total_kept == 1
        assert dedup.stats.exact_duplicates == 1

    def test_window_limits_comparisons(self):
        dedup = V7Deduplicator(jaccard_threshold=0.5, near_dedup_window=1)
        a = _raw_msg("alpha beta gamma delta", "epsilon")
        b = _raw_msg("completely different content here", "totally unique")
        c = _raw_msg("alpha beta gamma delta zeta", "epsilon")
        dedup.process(a)
        dedup.process(b)
        dedup.process(c)
        assert dedup.stats.near_duplicates == 0
        assert dedup.stats.total_kept == 3

    def test_window_zero_means_unlimited(self):
        dedup = V7Deduplicator(jaccard_threshold=0.5, near_dedup_window=0)
        a = _raw_msg("alpha beta gamma delta", "epsilon")
        b = _raw_msg("completely different content here", "totally unique")
        c = _raw_msg("alpha beta gamma delta zeta", "epsilon")
        dedup.process(a)
        dedup.process(b)
        dedup.process(c)
        assert dedup.stats.near_duplicates == 1

    def test_stats_tracked(self):
        dedup = V7Deduplicator()
        dedup.process(_chatml(source="src1", task_type="risk_assessment"))
        dedup.process(_chatml(source="src2", task_type="empathy_scoring"))
        assert "src1" in dedup.stats.records_by_source
        assert "src2" in dedup.stats.records_by_source
        assert "risk_assessment" in dedup.stats.records_by_task_type
        assert "empathy_scoring" in dedup.stats.records_by_task_type
        assert dedup.stats.total_read == 2

    def test_kept_records_property(self):
        dedup = V7Deduplicator(jaccard_threshold=0.99)
        rec = _chatml()
        dedup.process(rec)
        kept = dedup.kept_records
        assert len(kept) == 1
        assert kept[0] is rec


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
class TestFileIO:
    def test_write_and_read_jsonl(self, tmp_path):
        records = [_chatml(user=f"u{i}", assistant=f"a{i}") for i in range(3)]
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, records)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3
        parsed = json.loads(lines[0])
        assert "messages" in parsed

    def test_write_shards(self, tmp_path):
        records = [_chatml(user=f"u{i}", assistant=f"a{i}") for i in range(10)]
        shard_count = _write_shards(records, tmp_path, shard_size=4)
        assert shard_count == 3  # 4 + 4 + 2
        for i in range(3):
            shard_path = tmp_path / f"shard_{i:04d}.jsonl"
            assert shard_path.exists()
        lines = (tmp_path / "shard_0002.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

    def test_write_shards_single_shard(self, tmp_path):
        records = [_chatml()]
        shard_count = _write_shards(records, tmp_path, shard_size=100)
        assert shard_count == 1
        assert (tmp_path / "shard_0000.jsonl").exists()


# ---------------------------------------------------------------------------
# Integration: full pipeline via run_consolidation
# ---------------------------------------------------------------------------
class TestRunConsolidation:
    def test_full_pipeline(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        records = [
            _chatml(user="unique one", assistant="response one"),
            _chatml(user="unique two", assistant="response two"),
            _chatml(user="unique one", assistant="response one"),  # exact dup
        ]
        input_file = input_dir / "data.jsonl"
        with input_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        args = argparse.Namespace(
            input_dirs=[str(input_dir)],
            output_dir=str(output_dir),
            jaccard_threshold=0.99,  # high threshold to avoid system prompt interference
            near_dedup_window=5000,
            shard_size=0,
            use_lsh=False,
            num_perms=64,
        )
        run_consolidation(args)

        master_path = output_dir / "MASTER_V7.jsonl"
        assert master_path.exists()
        kept_lines = master_path.read_text().strip().split("\n")
        assert len(kept_lines) == 2  # 2 unique

        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema_version"] == "v7"
        assert manifest["total_records"] == 2

        stats_path = output_dir / "stats.json"
        assert stats_path.exists()
        stats = json.loads(stats_path.read_text())
        assert stats["total_read"] == 3
        assert stats["total_kept"] == 2
        assert stats["exact_duplicates"] == 1

    def test_pipeline_with_shards(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        records = [_chatml(user=f"u{i}", assistant=f"a{i}") for i in range(5)]
        input_file = input_dir / "data.jsonl"
        with input_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        args = argparse.Namespace(
            input_dirs=[str(input_dir)],
            output_dir=str(output_dir),
            jaccard_threshold=0.99,
            near_dedup_window=5000,
            shard_size=2,
            use_lsh=False,
            num_perms=64,
        )
        run_consolidation(args)

        for i in range(3):
            assert (output_dir / f"shard_{i:04d}.jsonl").exists()
        manifest = json.loads((output_dir / "manifest.json").read_text())
        assert manifest["shard_count"] == 3

    def test_pipeline_excludes_report_files(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Valid data file
        good = input_dir / "data.jsonl"
        with good.open("w") as f:
            f.write(json.dumps(_chatml()) + "\n")

        # Report files that should be excluded
        for name in ("report.jsonl", "rejection_log.jsonl", "stats.json"):
            rp = input_dir / name
            with rp.open("w") as f:
                f.write(json.dumps(_chatml()) + "\n")

        args = argparse.Namespace(
            input_dirs=[str(input_dir)],
            output_dir=str(output_dir),
            jaccard_threshold=0.99,
            near_dedup_window=5000,
            shard_size=0,
            use_lsh=False,
            num_perms=64,
        )
        run_consolidation(args)

        stats = json.loads((output_dir / "stats.json").read_text())
        assert stats["total_read"] == 1  # Only the good file
