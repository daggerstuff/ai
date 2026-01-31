"""Unit and integration tests for validation.py components.

Tests for SpeakerTurn, ConversationRecord, QualityScore, and validate_record function.
"""

import pytest
from datetime import datetime
from ai.pipelines.orchestrator.validation import (
    SpeakerTurn, 
    ConversationRecord, 
    QualityScore, 
    validate_record
)
from ai.pipelines.orchestrator.ingestion_interface import IngestRecord


class TestSpeakerTurn:
    """Test SpeakerTurn validation and sanitization."""
    
    def test_speaker_turn_creation_valid(self):
        """Test valid SpeakerTurn creation."""
        turn = SpeakerTurn(
            speaker_id="client",
            content="Hello, I need help.",
            timestamp=datetime.now(),
            metadata={"source": "test"}
        )
        assert turn.speaker_id == "client"
        assert turn.content == "Hello, I need help."
    
    def test_speaker_turn_sanitization_removes_xss(self):
        """Test that content sanitization removes potential XSS."""
        dangerous_content = "<script>alert('xss')</script>Hello, World!"
        turn = SpeakerTurn(
            speaker_id="client",
            content=dangerous_content,
            metadata={}
        )
        # Should be sanitized
        assert "<script>" not in turn.content
        assert "Hello, World!" in turn.content
    
    def test_speaker_turn_safe_html_preserved(self):
        """Test that safe HTML tags are preserved."""
        safe_content = "Hello <strong>World</strong> and <em>test</em> content."
        turn = SpeakerTurn(
            speaker_id="client",
            content=safe_content,
            metadata={}
        )
        assert "<strong>World</strong>" in turn.content
        assert "<em>test</em>" in turn.content
    
    def test_invalid_speaker_id(self):
        """Test that invalid speaker IDs are rejected."""
        with pytest.raises(ValueError):
            SpeakerTurn(
                speaker_id="<script>hacked</script>",
                content="test content",
                metadata={}
            )
    
    def test_content_length_validation(self):
        """Test content length validation."""
        long_content = "a" * 6000  # Exceeds max length of 5000
        with pytest.raises(ValueError):
            SpeakerTurn(
                speaker_id="client",
                content=long_content,
                metadata={}
            )


class TestConversationRecord:
    """Test ConversationRecord validation and constraints."""
    
    def test_conversation_record_creation(self):
        """Test valid ConversationRecord creation."""
        turn = SpeakerTurn(
            speaker_id="client",
            content="Hello",
            metadata={}
        )
        conv = ConversationRecord(
            id="test-conv-001",
            title="Test Conversation",
            turns=[turn],
            source_type="json",
            source_id="test-source",
            metadata={}
        )
        assert conv.id == "test-conv-001"
        assert len(conv.turns) == 1
    
    def test_conversation_minimum_turns(self):
        """Test that conversations must have at least 2 turns."""
        single_turn = SpeakerTurn(
            speaker_id="client",
            content="Single message",
            metadata={}
        )
        with pytest.raises(ValueError, match="at least two turns"):
            ConversationRecord(
                id="test-conv-002",
                title="Test", 
                turns=[single_turn],  # Only one turn
                source_type="json",
                source_id="test-source",
                metadata={}
            )
    
    def test_conversation_minimum_speakers(self):
        """Test that conversations must have at least 2 distinct speakers."""
        same_speaker_turns = [
            SpeakerTurn(speaker_id="client", content="First", metadata={}),
            SpeakerTurn(speaker_id="client", content="Second", metadata={})  # Same speaker
        ]
        with pytest.raises(ValueError, match="at least two distinct speakers"):
            ConversationRecord(
                id="test-conv-003",
                title="Test",
                turns=same_speaker_turns,
                source_type="json", 
                source_id="test-source",
                metadata={}
            )
    
    def test_conversation_alternating_speakers(self):
        """Test conversation with alternating speakers."""
        turns = [
            SpeakerTurn(speaker_id="client", content="Hello", metadata={}),
            SpeakerTurn(speaker_id="therapist", content="Hi", metadata={})
        ]
        conv = ConversationRecord(
            id="test-conv-004",
            title="Test",
            turns=turns,
            source_type="json",
            source_id="test-source", 
            metadata={}
        )
        assert len(conv.turns) == 2
        assert conv.turns[0].speaker_id != conv.turns[1].speaker_id
    
    def test_invalid_source_type(self):
        """Test that invalid source types are rejected."""
        turns = [
            SpeakerTurn(speaker_id="client", content="Hello", metadata={}),
            SpeakerTurn(speaker_id="therapist", content="Hi", metadata={})
        ]
        with pytest.raises(ValueError, match="Unknown source_type"):
            ConversationRecord(
                id="test-conv-005",
                title="Test",
                turns=turns,
                source_type="invalid_type",
                source_id="test-source",
                metadata={}
            )


class TestQualityScore:
    """Test QualityScore computation and validation."""
    
    def test_quality_score_normalization(self):
        """Test that quality scores are normalized."""
        score = QualityScore(
            completeness=1.5,  # Exceeds max
            coherence=0.5,
            relevance=-0.2,   # Below min
            raw_score=8.0
        )
        # Should be normalized to 0-1 range
        assert score.completeness == 1.0  # Max normalized
        assert score.coherence == 0.5     # Unchanged
        assert score.relevance == 0.0     # Min normalized
    
    def test_compute_initial_quality(self):
        """Test initial quality score computation."""
        # Create a conversation with 20+ turns for high completeness
        turns = [
            SpeakerTurn(speaker_id="client" if i % 2 == 0 else "therapist", 
                       content=f"This is turn {i} with some meaningful content to test coherence",
                       metadata={})
            for i in range(25)
        ]
        conv = ConversationRecord(
            id="test-conv-006",
            title="Test",
            turns=turns,
            source_type="json",
            source_id="test-source",
            metadata={}
        )
        
        score = QualityScore.compute_initial(conv)
        # Should have high completeness due to 25 turns (25/20 = 1.25 -> 1.0 max)
        assert score.completeness == 1.0
        # Should have reasonable coherence
        assert 0.0 <= score.coherence <= 1.0
        # Relevance is 0.3 since no "mental" or "health" in content
        assert score.relevance == 0.3


class TestValidateRecord:
    """Test the validate_record function with source mapping."""
    
    def test_validate_csv_source(self):
        """Test validation with CSV source mapping."""
        raw_record = IngestRecord(
            id="csv-test-001",
            payload={
                "source_type": "csv",
                "rows": [
                    {"speaker": "client", "content": "I'm feeling anxious", "timestamp": "2023-01-01T10:00:00"},
                    {"speaker": "therapist", "content": "Can you tell me more?", "timestamp": "2023-01-01T10:01:00"}
                ]
            },
            metadata={"source": "csv_test"}
        )
        
        validated = validate_record(raw_record)
        assert validated.id == "csv-test-001"
        assert len(validated.turns) == 2
        assert validated.turns[0].speaker_id == "client"
        assert validated.turns[1].speaker_id == "therapist"
        assert "anxious" in validated.turns[0].content
    
    def test_validate_json_source(self):
        """Test validation with JSON source mapping."""
        raw_record = IngestRecord(
            id="json-test-001",
            payload={
                "source_type": "json",
                "turns": [
                    {"speaker_id": "client", "content": "How are you?", "timestamp": "2023-01-01T10:00:00"},
                    {"speaker_id": "therapist", "content": "I'm well, thank you", "timestamp": "2023-01-01T10:01:00"}
                ]
            },
            metadata={"source": "json_test"}
        )
        
        validated = validate_record(raw_record)
        assert validated.id == "json-test-001"
        assert len(validated.turns) == 2
        assert validated.turns[0].content == "How are you?"
        assert validated.turns[1].content == "I'm well, thank you"
    
    def test_validate_audio_transcript_source(self):
        """Test validation with audio transcript source mapping."""
        raw_record = IngestRecord(
            id="audio-test-001",
            payload={
                "source_type": "audio_transcript",
                "segments": [
                    {"speaker": "client", "text": "I need help with my anxiety", "start_time": 0.0},
                    {"speaker": "therapist", "text": "I understand. Can you describe your anxiety?", "start_time": 5.5}
                ]
            },
            metadata={"source": "audio_test"}
        )
        
        validated = validate_record(raw_record)
        assert validated.id == "audio-test-001"
        assert len(validated.turns) == 2
        assert "anxiety" in validated.turns[0].content
        assert "describe" in validated.turns[1].content
    
    def test_validate_default_source(self):
        """Test validation with default source mapping."""
        raw_record = IngestRecord(
            id="default-test-001",
            payload={
                "source_type": "unknown",
                "turns": [
                    {"speaker_id": "client", "content": "Test content", "timestamp": "2023-01-01T10:00:00"}
                ]
            },
            metadata={"source": "default_test"}
        )
        
        validated = validate_record(raw_record)
        assert validated.id == "default-test-001"
        assert len(validated.turns) == 1
        assert validated.turns[0].content == "Test content"
    
    def test_validate_record_xss_sanitization(self):
        """Test that XSS sanitization works during validation."""
        raw_record = IngestRecord(
            id="xss-test-001",
            payload={
                "source_type": "json",
                "turns": [
                    {"speaker_id": "client", "content": "<script>alert('xss')</script>Safe content", "timestamp": "2023-01-01T10:00:00"},
                    {"speaker_id": "therapist", "content": "Normal response", "timestamp": "2023-01-01T10:01:00"}
                ]
            },
            metadata={"source": "xss_test"}
        )
        
        validated = validate_record(raw_record)
        # XSS content should be sanitized
        assert "<script>" not in validated.turns[0].content
        assert "Safe content" in validated.turns[0].content
        assert "Normal response" in validated.turns[1].content
    
    def test_validate_record_payload_not_dict(self):
        """Test that non-dict payloads are rejected."""
        raw_record = IngestRecord(
            id="invalid-test-001",
            payload="not a dict",  # Invalid payload type
            metadata={"source": "invalid_test"}
        )
        
        with pytest.raises(Exception, match="Payload must be dict"):
            validate_record(raw_record)
    
    def test_validate_mental_health_relevance(self):
        """Test that mental health relevance affects quality scoring."""
        raw_record = IngestRecord(
            id="relevance-test-001",
            payload={
                "source_type": "json",
                "turns": [
                    {"speaker_id": "client", "content": "I'm having mental health issues", "timestamp": "2023-01-01T10:00:00"},
                    {"speaker_id": "therapist", "content": "Let's address your health concerns", "timestamp": "2023-01-01T10:01:00"}
                ]
            },
            metadata={"source": "relevance_test"}
        )
        
        validated = validate_record(raw_record)
        # Quality score should be higher due to mental/health keywords
        quality = validated.metadata['quality_score']
        assert quality['relevance'] == 0.7  # Higher relevance due to keywords


# Integration tests
class TestValidationIntegration:
    """Integration tests for validation components."""
    
    def test_full_validation_pipeline(self):
        """Test the complete validation pipeline flow."""
        # Create raw ingestion record with CSV data
        raw_record = IngestRecord(
            id="integration-test-001",
            payload={
                "source_type": "csv",
                "rows": [
                    {"speaker": "client", "content": "I need help with my depression", "timestamp": "2023-01-01T10:00:00"},
                    {"speaker": "therapist", "content": "I'm here to help. Can you tell me more?", "timestamp": "2023-01-01T10:01:00"}
                ]
            },
            metadata={"source": "integration_test", "original_file": "test.csv"}
        )
        
        # Validate the record
        validated = validate_record(raw_record)
        
        # Verify all aspects of the pipeline worked
        assert validated.id == "integration-test-001"
        assert len(validated.turns) == 2
        assert validated.turns[0].speaker_id == "client"
        assert validated.turns[1].speaker_id == "therapist"
        
        # Verify sanitization worked
        assert "<script>" not in validated.turns[0].content
        
        # Verify quality scoring worked
        assert 'quality_score' in validated.metadata
        quality = validated.metadata['quality_score']
        assert 0.0 <= quality['completeness'] <= 1.0
        assert 0.0 <= quality['coherence'] <= 1.0
        assert 0.0 <= quality['relevance'] <= 1.0
        
        # Verify provenance metadata added
        assert 'ingestion_timestamp' in validated.metadata
        assert validated.metadata['source_type'] == 'json'  # Mapped from CSV


if __name__ == "__main__":
    pytest.main([__file__])