"""Tests for Stage 4 voice persona processing pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai.pipelines.voice.pipeline import DialogueTurn, TranscriptParser
from ai.pipelines.voice.persona_blender import (
    DialoguePair,
    PersonaBlender,
    VoiceSignatureToken,
)


class TestTranscriptParser:
    """Test transcript parsing functionality."""

    def test_parse_speaker_pattern(self, tmp_path):
        """Test parsing with Speaker: pattern."""
        transcript = "Speaker 1: Hello there\nSpeaker 2: Hi, how are you?"
        transcript_file = tmp_path / "test.txt"
        transcript_file.write_text(transcript)

        parser = TranscriptParser(tmp_path, tmp_path / "output")
        turns = parser.parse_transcript(transcript_file)

        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 1"
        assert turns[0].text == "Hello there"
        assert turns[1].speaker == "Speaker 2"
        assert turns[1].text == "Hi, how are you?"

    def test_parse_multiline_turn(self, tmp_path):
        """Test parsing multi-line dialogue turns."""
        transcript = "Client: I've been feeling\nreally down lately\nTherapist: I hear that"
        transcript_file = tmp_path / "test.txt"
        transcript_file.write_text(transcript)

        parser = TranscriptParser(tmp_path, tmp_path / "output")
        turns = parser.parse_transcript(transcript_file)

        assert len(turns) == 2
        assert turns[0].text == "I've been feeling really down lately"

    def test_process_all(self, tmp_path):
        """Test processing all transcript files."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        (input_dir / "file1.txt").write_text("A: First\nB: Second")
        (input_dir / "file2.txt").write_text("X: Third\nY: Fourth")

        parser = TranscriptParser(input_dir, output_dir)
        stats = parser.process_all()

        assert stats["files_processed"] == 2
        assert stats["total_turns"] == 4
        assert stats["errors"] == 0

        output_files = list(output_dir.glob("*_turns.jsonl"))
        assert len(output_files) == 2


class TestPersonaBlender:
    """Test persona blending functionality."""

    def test_detect_therapeutic_tone(self):
        """Test therapeutic tone detection."""
        blender = PersonaBlender(Path("/tmp"), Path("/tmp"))

        text = "What I'm hearing is that you feel overwhelmed. That makes sense given your situation."
        tone = blender.detect_therapeutic_tone(text)

        assert tone["reflective_listening"] > 0
        assert tone["validation"] > 0

    def test_extract_voice_tokens(self):
        """Test voice signature token extraction."""
        blender = PersonaBlender(Path("/tmp"), Path("/tmp"))

        text = "I appreciate your gentle and compassionate approach to this difficult situation."
        tokens = blender.extract_voice_tokens(text)

        assert len(tokens) > 0
        token_words = [t.token for t in tokens]
        assert "gentle" in token_words
        assert "compassionate" in token_words

    def test_create_dialogue_pairs(self, tmp_path):
        """Test dialogue pair creation from turns."""
        turns_file = tmp_path / "test_turns.jsonl"
        turns = [
            {"speaker": "Client", "text": "I'm struggling", "source_file": "test.txt", "turn_index": 0},
            {"speaker": "Therapist", "text": "I hear that", "source_file": "test.txt", "turn_index": 1},
            {"speaker": "Client", "text": "Thank you", "source_file": "test.txt", "turn_index": 2},
        ]

        with open(turns_file, "w") as f:
            for turn in turns:
                f.write(json.dumps(turn) + "\n")

        blender = PersonaBlender(tmp_path, tmp_path / "output")
        pairs = blender.create_dialogue_pairs(turns_file)

        assert len(pairs) == 2
        assert pairs[0].prompt == "I'm struggling"
        assert pairs[0].response == "I hear that"

    def test_process_all(self, tmp_path):
        """Test processing all turns files."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        turns = [
            {"speaker": "A", "text": "Hello", "source_file": "test.txt", "turn_index": 0},
            {"speaker": "B", "text": "Hi there", "source_file": "test.txt", "turn_index": 1},
        ]

        with open(input_dir / "file1_turns.jsonl", "w") as f:
            for turn in turns:
                f.write(json.dumps(turn) + "\n")

        blender = PersonaBlender(input_dir, output_dir)
        stats = blender.process_all()

        assert stats["files_processed"] == 1
        assert stats["total_pairs"] == 1
        assert stats["errors"] == 0

        output_files = list(output_dir.glob("*_voice_pairs.jsonl"))
        assert len(output_files) == 1


class TestDataclasses:
    """Test dataclass structures."""

    def test_dialogue_turn(self):
        """Test DialogueTurn dataclass."""
        turn = DialogueTurn(
            speaker="Client",
            text="I feel sad",
            source_file="test.txt",
            turn_index=0,
            metadata={"emotion": "sadness"},
        )

        assert turn.speaker == "Client"
        assert turn.text == "I feel sad"
        assert turn.metadata["emotion"] == "sadness"

    def test_voice_signature_token(self):
        """Test VoiceSignatureToken dataclass."""
        token = VoiceSignatureToken(
            token="compassionate",
            tone_category="warmth",
            intensity=0.9,
            therapeutic_technique="empathy",
        )

        assert token.token == "compassionate"
        assert token.tone_category == "warmth"
        assert token.intensity == 0.9

    def test_dialogue_pair(self):
        """Test DialoguePair dataclass."""
        pair = DialoguePair(
            prompt="I'm struggling",
            response="I hear you",
            voice_tokens=[],
            therapeutic_tone={"empathy": 1.0},
            source_file="test.txt",
            pair_index=0,
        )

        assert pair.prompt == "I'm struggling"
        assert pair.response == "I hear you"
        assert pair.therapeutic_tone["empathy"] == 1.0
