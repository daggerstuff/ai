"""Tests for the YouTube clinical practitioner persona distiller."""

from __future__ import annotations

import pytest

from training.youtube_persona_distiller import (
    DEFAULT_MODEL,
    PERSONA_PROFILES,
    _chunk_transcript,
    _strip_speaker_labels,
    get_persona_profile,
)


class TestPersonaKeyNormalization:
    def test_all_profile_keys_are_canonical(self):
        """No key may carry leading/trailing whitespace (regression for the
        ``"Patrick Teahan "`` trailing-space bug)."""
        for key in PERSONA_PROFILES:
            assert key == key.strip(), f"non-canonical persona key: {key!r}"

    def test_patrick_teahan_key_has_no_trailing_space(self):
        assert "Patrick Teahan" in PERSONA_PROFILES
        assert "Patrick Teahan " not in PERSONA_PROFILES

    def test_lookup_with_trailing_space_resolves(self):
        profile = get_persona_profile("Patrick Teahan ")
        assert profile["name"] == "Patrick Teahan, LICSW"

    def test_lookup_is_case_insensitive(self):
        profile = get_persona_profile("patrick teahan")
        assert profile["name"] == "Patrick Teahan, LICSW"

    def test_unknown_channel_returns_generic_fallback(self):
        profile = get_persona_profile("Nonexistent Channel")
        assert profile["domain"] == "general_mental_health"
        assert profile["name"] == "Nonexistent Channel"


class TestDefaultModel:
    def test_default_model_is_glm_flash(self):
        assert DEFAULT_MODEL == "@cf/zai-org/glm-5.3-flash"

    def test_default_model_is_not_llama(self):
        assert "llama" not in DEFAULT_MODEL.lower()


class TestSpeakerLabelStripping:
    def test_strips_named_speaker_labels(self):
        out = _strip_speaker_labels("Client: I feel stuck\n\nSpeaker 1: Tell me more")
        assert "Client:" not in out
        assert "Speaker 1:" not in out

    def test_strips_bracket_speaker_labels(self):
        out = _strip_speaker_labels("[Patrick] Let us begin")
        assert "[Patrick]" not in out

    def test_preserves_plain_prose(self):
        prose = "Therapy can help people notice triggers before they escalate."
        assert _strip_speaker_labels(prose) == prose

    def test_chunking_strips_labels_before_splitting(self):
        chunks = _chunk_transcript(
            "Therapist: hello world\n\nClient: I am hurting", chunk_size=100
        )
        assert chunks
        for chunk in chunks:
            assert "Therapist:" not in chunk
            assert "Client:" not in chunk