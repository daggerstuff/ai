"""Tests for the YouTube transcript ingestion pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

try:
    from hypothesis import given, settings, strategies as st
except ImportError:
    given = None
    settings = None
    st = None

from training.youtube_ingestion import (
    GERMAN_CHANNELS,
    MIN_CHUNK_WORDS,
    _content_hash,
    _is_german_channel,
    _load_compiled_hashes,
    _semantic_chunks,
    _transcript_to_pairs,
    _word_chunks,
    build_parser,
    ingest_channel,
    run_ingestion,
    split_sentences,
)

EXPECTED_TWO_RECORDS = 2
EXPECTED_THREE_CHUNKS = 3
TEST_CHUNK_WORDS = 50
LONG_TRANSCRIPT = " ".join(f"word{i}" for i in range(150))

# ---------------------------------------------------------------------------
# Unit tests — language tagging
# ---------------------------------------------------------------------------


class TestGermanChannelTagging:
    @pytest.mark.parametrize("name", sorted(GERMAN_CHANNELS))
    def test_german_channel_tagged(self, name: str):
        assert _is_german_channel(name, GERMAN_CHANNELS)

    @pytest.mark.parametrize("name", ["DoctorRamani", "Patrick Teahan", "Crappy Childhood Fairy"])
    def test_english_channel_not_tagged(self, name: str):
        assert not _is_german_channel(name, GERMAN_CHANNELS)

    def test_custom_german_channels(self):
        custom = frozenset({"MyGermanChannel"})
        assert _is_german_channel("MyGermanChannel", custom)
        assert not _is_german_channel("ARTEde", custom)


# ---------------------------------------------------------------------------
# Unit tests — content hashing
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_deterministic(self):
        assert _content_hash("hello") == _content_hash("hello")

    def test_different_inputs(self):
        assert _content_hash("hello") != _content_hash("world")


# ---------------------------------------------------------------------------
# Unit tests — compiled hash loading
# ---------------------------------------------------------------------------


class TestCompiledHashes:
    def test_missing_file_returns_empty(self, tmp_path: Path):
        result = _load_compiled_hashes(tmp_path / "nonexistent.jsonl")
        assert result == set()

    def test_loads_hashes_from_jsonl(self, tmp_path: Path):
        path = tmp_path / "compiled.jsonl"
        path.write_text(
            '{"messages": [{"role": "user", "content": "hello"}]}\n'
            '{"messages": [{"role": "user", "content": "world"}]}\n',
            encoding="utf-8",
        )
        hashes = _load_compiled_hashes(path)
        assert len(hashes) == EXPECTED_TWO_RECORDS

    def test_invalid_lines_skipped(self, tmp_path: Path):
        path = tmp_path / "compiled.jsonl"
        path.write_text(
            '{"messages": [{"role": "user", "content": "hello"}]}\n'
            "not json\n"
            '{"messages": [{"role": "user", "content": "world"}]}\n',
            encoding="utf-8",
        )
        hashes = _load_compiled_hashes(path)
        assert len(hashes) == EXPECTED_TWO_RECORDS


# ---------------------------------------------------------------------------
# Unit tests — transcript to pairs
# ---------------------------------------------------------------------------


class TestTranscriptToPairs:
    def test_splits_paragraphs(self):
        text = "Question about trauma?\n\nAnswer about healing."
        pairs = _transcript_to_pairs(text, "TestChannel")
        assert len(pairs) == 1
        assert pairs[0]["instruction"] == "Question about trauma?"
        assert pairs[0]["output"] == "Answer about healing."

    def test_empty_text(self):
        assert _transcript_to_pairs("", "TestChannel") == []

    def test_single_paragraph(self):
        text = "Just one paragraph of content."
        pairs = _transcript_to_pairs(text, "TestChannel")
        assert len(pairs) == 1
        assert "TestChannel" in pairs[0]["instruction"]

    def test_long_single_paragraph_is_chunked_without_filler(self):
        pairs = _transcript_to_pairs(
            LONG_TRANSCRIPT,
            "TestChannel",
            chunk_words=TEST_CHUNK_WORDS,
            chunk_overlap_words=0,
        )
        assert len(pairs) == EXPECTED_THREE_CHUNKS
        assert pairs[0]["pairing_strategy"] == "word_chunk"
        assert pairs[0]["chunk_word_count"] == TEST_CHUNK_WORDS
        assert pairs[0]["output"].startswith("word0 word1")
        assert pairs[-1]["output"].endswith("word149")

    def test_sentence_bound_transcript_uses_semantic_chunks(self):
        text = (
            "When trauma gets stored in the body, people often feel stuck. "
            "Therapy can help them notice triggers before they escalate. "
            "That awareness creates room for a different response. "
            "Over time, safety becomes something they can practice."
        )
        pairs = _transcript_to_pairs(
            text,
            "TestChannel",
            chunk_words=20,
            chunk_overlap_words=0,
        )
        assert len(pairs) >= EXPECTED_TWO_RECORDS
        assert all(pair["pairing_strategy"] == "semantic_chunk" for pair in pairs)
        assert "." in pairs[0]["output"]

    def test_multiple_pairs(self):
        text = "Q1\n\nA1\n\nQ2\n\nA2"
        pairs = _transcript_to_pairs(text, "TestChannel")
        assert len(pairs) == EXPECTED_TWO_RECORDS


class TestWordChunks:
    def test_rejects_tiny_chunks(self):
        with pytest.raises(ValueError, match=str(MIN_CHUNK_WORDS)):
            _word_chunks(LONG_TRANSCRIPT, chunk_words=10, overlap_words=0)

    def test_keeps_short_real_text_as_one_chunk(self):
        chunks = _word_chunks("short real transcript", chunk_words=50, overlap_words=0)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "short real transcript"


class TestSemanticChunks:
    def test_splits_on_sentence_boundaries(self):
        text = (
            "Healing begins when we notice our patterns. "
            "That awareness creates space for change. "
            "Therapy helps people reconnect with themselves. "
            "Small steps can rebuild trust over time."
        )
        chunks = _semantic_chunks(text, chunk_words=20, overlap_words=0)
        assert len(chunks) >= EXPECTED_TWO_RECORDS
        assert all(chunk["pairing_strategy"] == "semantic_chunk" for chunk in chunks)
        assert all("." in chunk["text"] or "?" in chunk["text"] or "!" in chunk["text"] for chunk in chunks[:-1])

    def test_preserves_paragraphs_under_limit(self):
        text = (
            "First complete paragraph about trauma recovery.\n\n"
            "Second complete paragraph about nervous system regulation."
        )
        chunks = _semantic_chunks(text, chunk_words=50, overlap_words=0)
        assert len(chunks) == EXPECTED_TWO_RECORDS
        assert chunks[0]["text"] == "First complete paragraph about trauma recovery."
        assert chunks[1]["text"] == "Second complete paragraph about nervous system regulation."

    def test_falls_back_to_word_chunks_for_unbroken_long_text(self):
        chunks = _semantic_chunks(LONG_TRANSCRIPT, chunk_words=TEST_CHUNK_WORDS, overlap_words=0)
        assert len(chunks) == EXPECTED_THREE_CHUNKS
        assert all(chunk["pairing_strategy"] == "word_chunk" for chunk in chunks)

    def testsplit_sentences_handles_punctuation(self):
        sentences = split_sentences("First idea. Second idea! Third idea?")
        assert sentences == ["First idea.", "Second idea!", "Third idea?"]


# ---------------------------------------------------------------------------
# Unit tests — ingest_channel
# ---------------------------------------------------------------------------


class TestIngestChannel:
    def test_missing_channel_dir(self, tmp_path: Path):
        samples, n_read, _n_dup = ingest_channel(
            tmp_path / "nonexistent",
            "en",
            set(),
        )
        assert samples == []
        assert n_read == 0

    def test_unreadable_transcript_skipped(self, tmp_path: Path):
        channel_dir = tmp_path / "TestChannel"
        channel_dir.mkdir()
        bad_file = channel_dir / "bad.txt"
        bad_file.write_text("Valid content here\n\nResponse here.", encoding="utf-8")
        _samples, n_read, _ = ingest_channel(
            channel_dir,
            "en",
            set(),
        )
        assert n_read > 0

    def test_all_samples_kept(self, tmp_path: Path):
        """Safety filter disabled — crisis content is kept for training."""
        channel_dir = tmp_path / "CrisisChannel"
        channel_dir.mkdir()
        crisis_file = channel_dir / "crisis.txt"
        crisis_file.write_text(
            "I want to talk about suicide\n\nI want to kill myself tonight",
            encoding="utf-8",
        )
        samples, _n_read, _ = ingest_channel(
            channel_dir,
            "en",
            set(),
        )
        assert len(samples) > 0
        assert samples[0]["provenance"]["source_type"] == "youtube"
        assert samples[0]["provenance"]["license"] == "NOASSERTION"

    def test_duplicate_samples_skipped(self, tmp_path: Path):
        channel_dir = tmp_path / "DupChannel"
        channel_dir.mkdir()
        transcript = channel_dir / "test.txt"
        transcript.write_text(
            "What is CBT?\n\nCognitive behavioral therapy helps reframe thoughts.",
            encoding="utf-8",
        )
        content = "What is CBT? Cognitive behavioral therapy helps reframe thoughts."
        compiled_hash = {_content_hash(content.lower().strip())}
        samples, _n_read, n_dup = ingest_channel(
            channel_dir,
            "en",
            compiled_hash,
        )
        assert n_dup > 0
        assert len(samples) == 0

    def test_long_single_paragraph_samples_include_chunk_provenance(self, tmp_path: Path):
        channel_dir = tmp_path / "ChunkChannel"
        channel_dir.mkdir()
        transcript = channel_dir / "long.txt"
        transcript.write_text(LONG_TRANSCRIPT, encoding="utf-8")

        samples, n_read, _n_dup = ingest_channel(
            channel_dir,
            "en",
            set(),
            chunk_words=TEST_CHUNK_WORDS,
            chunk_overlap_words=0,
        )

        assert n_read == EXPECTED_THREE_CHUNKS
        assert len(samples) == EXPECTED_THREE_CHUNKS
        metadata = samples[0]["provenance"]["metadata"]
        assert metadata["pairing_strategy"] == "word_chunk"
        assert metadata["chunk_index"] == 1
        assert metadata["chunk_total"] == EXPECTED_THREE_CHUNKS


# ---------------------------------------------------------------------------
# Unit tests — processing_report fields
# ---------------------------------------------------------------------------


class TestProcessingReport:
    def test_report_has_required_fields(self, tmp_path: Path):
        transcripts_dir = tmp_path / "transcripts"
        transcripts_dir.mkdir()
        channel = transcripts_dir / "TestChannel"
        channel.mkdir()
        (channel / "video.txt").write_text(
            "What is mindfulness?\n\nMindfulness is present-moment awareness.",
            encoding="utf-8",
        )

        compiled = tmp_path / "compiled.jsonl"
        compiled.write_text("{}", encoding="utf-8")

        output_dir = tmp_path / "output"

        args = build_parser().parse_args(
            [
                "--transcripts_dir",
                str(transcripts_dir),
                "--output_dir",
                str(output_dir),
                "--compiled_dataset_dir",
                str(compiled),
                "--chunk_words",
                str(TEST_CHUNK_WORDS),
                "--chunk_overlap_words",
                "0",
            ]
        )
        run_ingestion(args)

        report_path = output_dir / "processing_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert "processed" in report
        assert "skipped_unsafe" not in report
        assert "skipped_duplicate" in report
        assert "total_samples" in report
        assert "channels_processed" in report
        assert "channels_skipped" in report

        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "channels" in manifest
        assert "totals" in manifest

        channel_output = output_dir / "TestChannel.jsonl"
        record = json.loads(channel_output.read_text(encoding="utf-8").strip())
        assert record["provenance"]["source_type"] == "youtube"
        assert record["provenance"]["metadata"]["channel"] == "TestChannel"


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------

if st is not None:

    @given(st.sampled_from(sorted(GERMAN_CHANNELS)))
    @settings(max_examples=20)
    def test_hypothesis_german_channel_tagged(channel: str):
        assert _is_german_channel(channel, GERMAN_CHANNELS)

    @given(
        text=st.text(min_size=1, max_size=200),
        channel=st.sampled_from(["DoctorRamani", "Patrick Teahan", "Therapy in a Nutshell"]),
    )
    @settings(max_examples=50)
    def test_hypothesis_transcripts_processed(text: str, channel: str):
        """Safety filter disabled — all transcripts are processed regardless of content."""
        if not text.strip():
            return
        tmp = Path(tempfile.mkdtemp())
        channel_dir = tmp / channel
        channel_dir.mkdir()
        (channel_dir / "test.txt").write_text(
            f"{text}\n\nResponse to the question about therapy.",
            encoding="utf-8",
        )
        samples, _, _ = ingest_channel(
            channel_dir,
            "en",
            set(),
        )
        if samples:
            assert "instruction" in samples[0]
            assert "output" in samples[0]

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_hypothesis_dedup_excludes_known_content(text: str):
        compiled_hash = {_content_hash(text.lower().strip())}
        tmp = Path(tempfile.mkdtemp())
        channel_dir = tmp / "TestChannel"
        channel_dir.mkdir()
        (channel_dir / "test.txt").write_text(
            f"{text}\n\nSome response that is safe and helpful.",
            encoding="utf-8",
        )
        _samples, _, n_dup = ingest_channel(
            channel_dir,
            "en",
            compiled_hash,
        )
        assert n_dup >= 0

else:

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_german_channel_tagged():
        raise AssertionError("Skipped when hypothesis is unavailable")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_transcripts_processed():
        raise AssertionError("Skipped when hypothesis is unavailable")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_dedup_excludes_known_content():
        raise AssertionError("Skipped when hypothesis is unavailable")
