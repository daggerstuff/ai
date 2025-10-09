"""Unit tests for validation.py module.

Tests Pydantic models, sanitization, quality scoring, and format converters.
Uses pytest with fixtures for isolation.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from ai.dataset_pipeline.validation import (
    ConversationRecord,
    QualityScore,
    SpeakerTurn,
    validate_record,
)


@pytest.fixture
def sample_speaker_turn():
    return SpeakerTurn(
        speaker_id="therapist",
        content="This is a sample therapeutic dialogue turn.",
        timestamp=datetime(2023, 1, 1),
        metadata={"session_id": "sess_001"},
    )


def test_speaker_turn_sanitization(sample_speaker_turn):
    # Test basic sanitization passes
    turn = SpeakerTurn(**sample_speaker_turn.dict())
    assert turn.content == "This is a sample therapeutic dialogue turn."

    # Test malicious input gets sanitized
    malicious = SpeakerTurn(
        speaker_id="therapist",
        content="<script>alert('xss')</script> Benign text",
        timestamp=datetime(2023, 1, 1),
    )
    assert "<script>" not in malicious.content
    assert "Benign text" in malicious.content

    # Test length limits
    long_content = "A" * 6000
    with pytest.raises(ValueError, match="max_length"):
        SpeakerTurn(speaker_id="test", content=long_content)


def test_conversation_record_validation(sample_speaker_turn):
    conv = ConversationRecord(
        id="conv_001",
        title="Sample Session",
        turns=[sample_speaker_turn, SpeakerTurn(speaker_id="client", content="Response")],
        source_type="json",
        metadata={},
    )
    assert conv.id == "conv_001"
    assert len(conv.turns) == 2

    # Test invalid: single turn
    invalid_turns = [sample_speaker_turn]
    with pytest.raises(ValueError, match="at least two turns"):
        ConversationRecord(
            id="invalid",
            turns=invalid_turns,
            source_type="json",
            metadata={},
        )

    # Test invalid: same speaker consecutive
    same_speaker = [sample_speaker_turn, SpeakerTurn(speaker_id="therapist", content="Same")]
    with pytest.raises(ValueError, match="Consecutive turns"):
        ConversationRecord(
            id="invalid",
            turns=same_speaker,
            source_type="json",
            metadata={},
        )


def test_quality_score_computation():
    # Simple test data
    rec = ConversationRecord(
        id="test",
        turns=[
            SpeakerTurn(speaker_id="therapist", content="Mental health discussion"),
            SpeakerTurn(speaker_id="client", content="Response about feelings"),
        ],
        source_type="json",
        metadata={},
    )
    score = QualityScore.compute_initial(rec)
    assert 0.0 <= score.completeness <= 1.0
    assert 0.0 <= score.coherence <= 1.0
    assert score.relevance > 0.5  # Contains mental health keywords


def test_validate_record_json():
    raw_rec = MagicMock()
    raw_rec.id = "test_id"
    raw_rec.payload = {"turns": [{"speaker_id": "t", "content": "ok"}], "source_type": "json"}
    validated = validate_record(raw_rec)
    assert isinstance(validated, ConversationRecord)
    assert "quality_score" in validated.metadata


def test_validate_record_csv():
    raw_rec = MagicMock()
    raw_rec.id = "test_csv"
    raw_rec.payload = {
        "source_type": "csv",
        "rows": [
            {"speaker": "t", "content": "ok1"},
            {"speaker": "c", "content": "ok2"},
        ],
    }
    validated = validate_record(raw_rec)
    assert len(validated.turns) == 2
    assert validated.source_type == "csv"


def test_validate_record_invalid():
    raw_rec = MagicMock()
    raw_rec.id = "invalid"
    raw_rec.payload = {"turns": [{"speaker_id": "same", "content": "only one"}]}
    with pytest.raises(ValueError, match="at least two turns"):
        validate_record(raw_rec)
