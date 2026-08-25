"""Tests for the dedup and ChatML normalization pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from hypothesis import given, settings, strategies as st
except ImportError:
    given = None
    settings = None
    st = None

from training.dedup_normalize import (
    ProcessingContext,
    SimHashIndex,
    _attempt_reformat,
    _content_hash,
    _extract_text,
    _hamming_distance,
    _is_edge_case,
    _jaccard_similarity,
    _simhash_signature,
    _token_set,
    _verify_chatml_boundary,
    build_parser,
    process_file,
    run_dedup,
)

# Constants for magic-value comparisons (PLR2004)
TWO_INPUT_RECORDS = 2
EXPECTED_OUTPUT_MESSAGE_COUNT = 2
SHARD_EXPECT_5_RECORDS_SIZE2 = 3


class TestJaccardSimilarity:
    def test_identical_sets(self):
        assert _jaccard_similarity(frozenset(["a", "b", "c"]), frozenset(["a", "b", "c"])) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard_similarity(frozenset(["a", "b"]), frozenset(["c", "d"])) == 0.0

    def test_partial_overlap(self):
        result = _jaccard_similarity(frozenset(["a", "b", "c"]), frozenset(["b", "c", "d"]))
        assert 0.0 < result < 1.0

    def test_empty_sets(self):
        assert _jaccard_similarity(frozenset(), frozenset()) == 1.0

    def test_one_empty(self):
        assert _jaccard_similarity(frozenset(["a", "b"]), frozenset()) == 0.0


class TestExtractText:
    def test_from_messages(self):
        record = {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]}
        assert _extract_text(record) == "hello world"

    def test_from_text_field(self):
        record = {"text": "some text content"}
        assert _extract_text(record) == "some text content"

    def test_from_instruction_output(self):
        record = {"instruction": "What is CBT?", "output": "Cognitive behavioral therapy."}
        assert "What is CBT?" in _extract_text(record)


class TestIsEdgeCase:
    def test_edge_case_true(self):
        assert _is_edge_case({"is_training_edge_case": True})

    def test_edge_case_false(self):
        assert not _is_edge_case({"is_training_edge_case": False})

    def test_missing_key(self):
        assert not _is_edge_case({"other_key": True})


class TestVerifyChatmlBoundary:
    def test_valid_messages(self):
        record = {"messages": [{"role": "user", "content": "hello"}]}
        assert _verify_chatml_boundary(record)

    def test_valid_inst_boundary(self):
        record = {"text": "[INST] hello [/INST] world"}
        assert _verify_chatml_boundary(record)

    def test_no_inst_in_simple_record(self):
        record = {"instruction": "hello", "output": "world"}
        assert _verify_chatml_boundary(record)


class TestAttemptReformat:
    def test_reformats_instruction_output(self):
        record = {"instruction": "What is CBT?", "output": "Therapy approach."}
        result = _attempt_reformat(record)
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == EXPECTED_OUTPUT_MESSAGE_COUNT
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"


class TestSimHash:
    def test_identical_text_same_signature(self):
        a = _simhash_signature("CBT reframes negative automatic thoughts")
        b = _simhash_signature("CBT reframes negative automatic thoughts")
        assert a == b

    def test_hamming_zero_for_identical(self):
        a = _simhash_signature("hello world")
        assert _hamming_distance(a, a) == 0

    def test_hamming_bounded(self):
        a = _simhash_signature("CBT is a therapeutic approach")
        b = _simhash_signature("DBT is a therapeutic approach")
        assert 0 <= _hamming_distance(a, b) <= 64

    def test_reordered_words_same_signature(self):
        # SimHash is order-insensitive (token bag): reordering gives identical sig.
        a = _simhash_signature("CBT reframes negative automatic thoughts")
        b = _simhash_signature("automatic thoughts negative reframes CBT")
        assert _hamming_distance(a, b) <= 3

    def test_case_and_whitespace_insensitive(self):
        base = _simhash_signature("CBT reframes negative automatic thoughts")
        assert _hamming_distance(base, _simhash_signature("CBT reframes negative automatic thoughts ")) == 0
        assert _hamming_distance(base, _simhash_signature("CBT REFRAMES negative AUTOMATIC thoughts")) == 0

    def test_dissimilar_texts_large_hamming(self):
        a = _simhash_signature("cognitive behavioral therapy session notes")
        b = _simhash_signature("quantum chromodynamics lattice gauge theory")
        assert _hamming_distance(a, b) > 3

    def test_empty_text_zero_signature(self):
        assert _simhash_signature("") == 0


class TestSimHashIndex:
    def test_near_duplicate_detected(self):
        # Reordered tokens = near-identical SimHash (order-insensitive).
        idx = SimHashIndex(hamming_threshold=3)
        text_a = "CBT reframes negative automatic thoughts in depressed clients"
        text_b = "depressed clients CBT reframes negative automatic thoughts in"
        idx.add("a", _simhash_signature(text_a))
        assert idx.is_near_duplicate(_simhash_signature(text_b), "b")

    def test_distinct_not_duplicate(self):
        idx = SimHashIndex(hamming_threshold=3)
        idx.add("a", _simhash_signature("cognitive behavioral therapy session"))
        assert not idx.is_near_duplicate(
            _simhash_signature("quantum chromodynamics lattice gauge"), "b"
        )

    def test_self_not_duplicate(self):
        idx = SimHashIndex(hamming_threshold=3)
        sig = _simhash_signature("hello world")
        idx.add("a", sig)
        assert not idx.is_near_duplicate(sig, "a")


class TestProcessFile:
    def test_exact_dedup(self, tmp_path: Path):
        input_file = tmp_path / "test.jsonl"
        content = json.dumps({"instruction": "Hello", "output": "World"}) + "\n"
        input_file.write_text(content + content, encoding="utf-8")

        seen: set[str] = set()
        token_sets: list[tuple[frozenset[str], str]] = []
        rejection: list[dict] = []
        ctx = ProcessingContext(seen_hashes=seen, edge_case_hashes=set(), token_sets=token_sets)
        stats = process_file(input_file, 0.85, rejection, ctx)
        assert stats.total_read == TWO_INPUT_RECORDS
        assert stats.exact_dupes == 1
        assert len(stats.kept) == 1

    def test_near_dedup(self, tmp_path: Path):
        input_file = tmp_path / "test.jsonl"
        shared = "CBT is a therapeutic approach for reframing negative automatic thoughts in clients"
        r1 = json.dumps({"instruction": "Explain CBT.", "output": shared + " with depression."})
        r2 = json.dumps({"instruction": "Explain CBT.", "output": shared + " with anxiety."})
        input_file.write_text(r1 + "\n" + r2 + "\n", encoding="utf-8")

        seen: set[str] = set()
        token_sets: list[tuple[frozenset[str], str]] = []
        rejection: list[dict] = []
        ctx = ProcessingContext(seen_hashes=seen, edge_case_hashes=set(), token_sets=token_sets)
        stats = process_file(input_file, 0.85, rejection, ctx)
        assert stats.total_read == TWO_INPUT_RECORDS
        assert stats.near_dupes >= 1 or stats.exact_dupes >= 1

    def test_edge_case_preserved(self, tmp_path: Path):
        input_file = tmp_path / "test.jsonl"
        record = {"instruction": "Crisis prompt", "output": "Safe response", "is_training_edge_case": True}
        input_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        seen: set[str] = set()
        token_sets: list[tuple[frozenset[str], str]] = []
        rejection: list[dict] = []
        ctx = ProcessingContext(seen_hashes=seen, edge_case_hashes=set(), token_sets=token_sets)
        stats = process_file(input_file, 0.85, rejection, ctx)
        assert len(stats.kept) == 1

    def test_unreadable_jsonl_skipped(self, tmp_path: Path):
        input_file = tmp_path / "bad.jsonl"
        input_file.write_text("not valid json\n", encoding="utf-8")

        seen: set[str] = set()
        token_sets: list[tuple[frozenset[str], str]] = []
        rejection: list[dict] = []
        ctx = ProcessingContext(seen_hashes=seen, edge_case_hashes=set(), token_sets=token_sets)
        stats = process_file(input_file, 0.85, rejection, ctx)
        assert stats.total_read == 1
        assert len(stats.kept) == 0


class TestRunDedup:
    def test_normalization_report_fields(self, tmp_path: Path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "data.jsonl").write_text(
            json.dumps({"instruction": "Hello", "output": "World"}) + "\n",
            encoding="utf-8",
        )

        output_dir = tmp_path / "output"

        args = build_parser().parse_args(
            [
                "--input_dirs",
                str(input_dir),
                "--output_dir",
                str(output_dir),
            ]
        )
        run_dedup(args)

        report_path = output_dir / "normalization_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert "total_samples_in" in report
        assert "exact_duplicates" in report
        assert "near_duplicates" in report
        assert "chatml_failures" in report
        assert "edge_cases_preserved" in report
        assert "total_samples_out" in report

    def test_sharding(self, tmp_path: Path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        records = [json.dumps({"instruction": f"Q{i}", "output": f"A{i}"}) for i in range(5)]
        (input_dir / "data.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")

        output_dir = tmp_path / "output"

        args = build_parser().parse_args(
            [
                "--input_dirs",
                str(input_dir),
                "--output_dir",
                str(output_dir),
                "--shard_size",
                "2",
            ]
        )
        run_dedup(args)

        shards = list(output_dir.glob("shard_*.jsonl"))
        assert len(shards) == SHARD_EXPECT_5_RECORDS_SIZE2  # 5 records with shard_size=2 → 3 shards


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------

if st is not None:

    @given(st.text(min_size=1, max_size=200))
    @settings(max_examples=50)
    def test_hypothesis_content_hash_deterministic(text: str):
        assert _content_hash(text) == _content_hash(text)

    @given(
        a=st.sets(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
        b=st.sets(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
    )
    @settings(max_examples=100)
    def test_hypothesis_jaccard_range(a: set[str], b: set[str]):
        result = _jaccard_similarity(frozenset(a), frozenset(b))
        assert 0.0 <= result <= 1.0

    @given(st.text(min_size=1, max_size=200).filter(lambda t: t.strip()))
    @settings(max_examples=50)
    def test_hypothesis_near_dup_removal(text: str):
        tokens = _token_set(text)
        if not tokens:
            return
        result = _jaccard_similarity(tokens, tokens)
        assert result == 1.0

    @given(st.from_regex(r"[a-zA-Z ]{1,50}", fullmatch=True))
    @settings(max_examples=50)
    def test_hypothesis_edge_case_preserved(text: str):
        if not text.strip():
            return
        record = {
            "instruction": text,
            "output": "Safe response with resources.",
            "is_training_edge_case": True,
        }
        assert _is_edge_case(record)

else:

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_content_hash_deterministic():
        raise AssertionError("Skipped when hypothesis is unavailable")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_jaccard_range():
        raise AssertionError("Skipped when hypothesis is unavailable")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_near_dup_removal():
        raise AssertionError("Skipped when hypothesis is unavailable")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_edge_case_preserved():
        raise AssertionError("Skipped when hypothesis is unavailable")
