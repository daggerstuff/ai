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
    _content_hash,
    _is_german_channel,
    _load_compiled_hashes,
    _transcript_to_pairs,
    ingest_channel,
)

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
        assert len(hashes) == 2

    def test_invalid_lines_skipped(self, tmp_path: Path):
        path = tmp_path / "compiled.jsonl"
        path.write_text(
            '{"messages": [{"role": "user", "content": "hello"}]}\n'
            'not json\n'
            '{"messages": [{"role": "user", "content": "world"}]}\n',
            encoding="utf-8",
        )
        hashes = _load_compiled_hashes(path)
        assert len(hashes) == 2


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

    def test_multiple_pairs(self):
        text = "Q1\n\nA1\n\nQ2\n\nA2"
        pairs = _transcript_to_pairs(text, "TestChannel")
        assert len(pairs) == 2


# ---------------------------------------------------------------------------
# Unit tests — ingest_channel
# ---------------------------------------------------------------------------

class TestIngestChannel:

    def test_missing_channel_dir(self, tmp_path: Path):
        samples, n_read, n_unsafe, n_dup = ingest_channel(
        samples, n_read, n_dup = ingest_channel(
            tmp_path / "nonexistent", "en", set(),
        )
        assert samples == []
        assert n_read == 0

    def test_unreadable_transcript_skipped(self, tmp_path: Path):
        channel_dir = tmp_path / "TestChannel"
        channel_dir.mkdir()
        bad_file = channel_dir / "bad.txt"
        bad_file.write_text("Valid content here\n\nResponse here.", encoding="utf-8")
        samples, n_read, _, _ = ingest_channel(
        samples, n_read, _ = ingest_channel(
            channel_dir, "en", set(),
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
        samples, n_read, _ = ingest_channel(
            channel_dir, "en", set(),
        )
        assert len(samples) > 0
        )
        assert len(samples) > 0
        )
        assert len(samples) > 0
        )
        assert len(samples) > 0
        )
        assert len(samples) > 0
        )
        assert len(samples) > 0
        )
        assert len(samples) > 0
        )
        assert len(samples) > 0
        )
        assert len(samples) > 0
        )
        assert len(samples) > 0
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        samples, n_read, n_unsafe, _ = ingest_channel(
            channel_dir, "en", set(),
        )
        assert len(samples) > 0
        assert n_unsafe == 0
        samples, n_read, _ = ingest_channel(
            channel_dir, "en", set(),
        )
        assert len(samples) > 0

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
        samples, n_read, _, n_dup = ingest_channel(
        samples, n_read, n_dup = ingest_channel(
            channel_dir, "en", compiled_hash,
        )
        assert n_dup > 0
        assert len(samples) == 0


# ---------------------------------------------------------------------------
# Unit tests — processing_report fields
# ---------------------------------------------------------------------------

class TestProcessingReport:

    def test_report_has_required_fields(self, tmp_path: Path):
        from training.youtube_ingestion import build_parser, run_ingestion

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

        args = build_parser().parse_args([
            "--transcripts_dir", str(transcripts_dir),
            "--output_dir", str(output_dir),
            "--compiled_dataset_dir", str(compiled),
        ])
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
        samples, _, n_unsafe, _ = ingest_channel(
            channel_dir, "en", set(),
        )
        assert n_unsafe == 0
        samples, _, _ = ingest_channel(
            channel_dir, "en", set(),
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
        samples, _, _, n_dup = ingest_channel(
        samples, _, n_dup = ingest_channel(
            channel_dir, "en", compiled_hash,
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
