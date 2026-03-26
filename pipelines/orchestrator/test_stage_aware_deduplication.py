"""
Tests for stage-aware deduplication per MasterTrainingPlan.md.

Tests verify:
1. Hash consistency (same input = same hash)
2. Stage priority resolution (stage4 wins over stage3, etc.)
3. Crisis intensity preserved in secondary hash
4. Target dedup rate < 1%
"""

import pytest
from unittest.mock import Mock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage_aware_deduplication import (
    compute_primary_hash,
    compute_secondary_hash,
    get_stage_priority,
    get_stage_name,
    resolve_conflict,
    deduplicate_conversations,
    deduplicate_with_secondary_hash,
    validate_deduplication,
    STAGE_PRIORITY,
)


def create_mock_message(role: str = "user", content: str = "test"):
    """Create a mock message for testing."""
    msg = Mock()
    msg.role = role
    msg.content = content
    return msg


def create_mock_conversation(
    conversation_id: str = "test-conv-1",
    messages=None,
    metadata: dict = None
):
    """Create a mock conversation for testing."""
    conv = Mock()
    conv.conversation_id = conversation_id
    conv.messages = messages or [
        create_mock_message("user", "Hello"),
        create_mock_message("assistant", "Hi there"),
    ]
    conv.metadata = metadata or {}
    return conv


class TestPrimaryHash:
    """Test primary hash computation."""

    def test_same_input_same_hash(self):
        """Hash consistency: same input should produce same hash."""
        conv1 = create_mock_conversation(
            messages=[create_mock_message("user", "Hello")]
        )
        conv2 = create_mock_conversation(
            messages=[create_mock_message("user", "Hello")]
        )

        hash1 = compute_primary_hash(conv1)
        hash2 = compute_primary_hash(conv2)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_different_content_different_hash(self):
        """Different content should produce different hash."""
        conv1 = create_mock_conversation(
            messages=[create_mock_message("user", "Hello")]
        )
        conv2 = create_mock_conversation(
            messages=[create_mock_message("user", "Goodbye")]
        )

        hash1 = compute_primary_hash(conv1)
        hash2 = compute_primary_hash(conv2)

        assert hash1 != hash2

    def test_case_insensitive(self):
        """Hash should be case-insensitive (lowercase)."""
        conv1 = create_mock_conversation(
            messages=[create_mock_message("user", "Hello")]
        )
        conv2 = create_mock_conversation(
            messages=[create_mock_message("user", "HELLO")]
        )

        hash1 = compute_primary_hash(conv1)
        hash2 = compute_primary_hash(conv2)

        assert hash1 == hash2

    def test_multiple_messages(self):
        """Should handle multiple messages correctly."""
        conv = create_mock_conversation(
            messages=[
                create_mock_message("user", "First"),
                create_mock_message("assistant", "Second"),
                create_mock_message("user", "Third"),
            ]
        )

        hash = compute_primary_hash(conv)
        assert len(hash) == 64


class TestSecondaryHash:
    """Test secondary hash computation."""

    def test_includes_crisis_intensity(self):
        """Secondary hash should include crisis_intensity."""
        conv1 = create_mock_conversation(
            conversation_id="test-1",
            metadata={"stage": "stage1_foundation", "source": "test", "crisis_intensity": "extreme"}
        )
        conv2 = create_mock_conversation(
            conversation_id="test-1",
            metadata={"stage": "stage1_foundation", "source": "test", "crisis_intensity": "high"}
        )

        hash1 = compute_secondary_hash(conv1)
        hash2 = compute_secondary_hash(conv2)

        assert hash1 != hash2  # Different crisis_intensity = different hash

    def test_includes_stage(self):
        """Secondary hash should include stage."""
        conv1 = create_mock_conversation(
            conversation_id="test-1",
            metadata={"stage": "stage1_foundation", "source": "test", "crisis_intensity": "high"}
        )
        conv2 = create_mock_conversation(
            conversation_id="test-1",
            metadata={"stage": "stage4_voice_persona", "source": "test", "crisis_intensity": "high"}
        )

        hash1 = compute_secondary_hash(conv1)
        hash2 = compute_secondary_hash(conv2)

        assert hash1 != hash2

    def test_includes_conversation_id(self):
        """Secondary hash should include conversation_id."""
        conv1 = create_mock_conversation(
            conversation_id="test-1",
            metadata={"stage": "stage1_foundation", "source": "test", "crisis_intensity": "high"}
        )
        conv2 = create_mock_conversation(
            conversation_id="test-2",
            metadata={"stage": "stage1_foundation", "source": "test", "crisis_intensity": "high"}
        )

        hash1 = compute_secondary_hash(conv1)
        hash2 = compute_secondary_hash(conv2)

        assert hash1 != hash2

    def test_sha1_format(self):
        """Secondary hash should be SHA1 (40 hex chars)."""
        conv = create_mock_conversation()
        hash = compute_secondary_hash(conv)

        assert len(hash) == 40  # SHA1 hex length


class TestStagePriority:
    """Test stage priority resolution."""

    def test_stage4_highest_priority(self):
        """Stage 4 should have highest priority."""
        stage4_conv = create_mock_conversation(
            metadata={"stage": "stage4_voice_persona"}
        )
        stage3_conv = create_mock_conversation(
            metadata={"stage": "stage3_edge_stress_test"}
        )

        priority_4 = get_stage_priority(stage4_conv)
        priority_3 = get_stage_priority(stage3_conv)

        assert priority_4 > priority_3
        assert priority_4 == 5  # Highest

    def test_stage_order(self):
        """Verify stage order: stage4 > stage3 > stage2 > stage1 > supplementary."""
        stages = [
            ("stage4_voice_persona", 5),
            ("stage3_edge_stress_test", 4),
            ("stage2_therapeutic_expertise", 3),
            ("stage1_foundation", 2),
            ("supplementary", 1),
        ]

        for stage_name, expected_priority in stages:
            conv = create_mock_conversation(metadata={"stage": stage_name})
            assert get_stage_priority(conv) == expected_priority

    def test_unknown_stage_default(self):
        """Unknown stages should get default priority."""
        conv = create_mock_conversation(
            metadata={"stage": "unknown_stage"}
        )

        priority = get_stage_priority(conv)
        assert priority == 1  # Default


class TestConflictResolution:
    """Test conflict resolution by stage priority."""

    def test_stage4_wins(self):
        """Stage 4 should win over lower stages."""
        stage3_conv = create_mock_conversation(
            conversation_id="stage3-conv",
            metadata={"stage": "stage3_edge_stress_test"}
        )
        stage4_conv = create_mock_conversation(
            conversation_id="stage4-conv",
            metadata={"stage": "stage4_voice_persona"}
        )

        winner = resolve_conflict([stage3_conv, stage4_conv])
        assert winner.conversation_id == "stage4-conv"

    def test_stage3_wins_over_stage2(self):
        """Stage 3 should win over stage 2."""
        stage2_conv = create_mock_conversation(
            conversation_id="stage2-conv",
            metadata={"stage": "stage2_therapeutic_expertise"}
        )
        stage3_conv = create_mock_conversation(
            conversation_id="stage3-conv",
            metadata={"stage": "stage3_edge_stress_test"}
        )

        winner = resolve_conflict([stage2_conv, stage3_conv])
        assert winner.conversation_id == "stage3-conv"

    def test_single_conversation(self):
        """Single conversation should be returned as-is."""
        conv = create_mock_conversation(
            conversation_id="single-conv",
            metadata={"stage": "stage1_foundation"}
        )

        winner = resolve_conflict([conv])
        assert winner.conversation_id == "single-conv"


class TestDeduplication:
    """Test full deduplication workflow."""

    def test_exact_duplicates_removed(self):
        """Exact duplicates should be removed."""
        conv1 = create_mock_conversation(
            conversation_id="conv-1",
            messages=[create_mock_message("user", "Hello")]
        )
        conv2 = create_mock_conversation(
            conversation_id="conv-2",
            messages=[create_mock_message("user", "Hello")]
        )

        unique, result = deduplicate_conversations([conv1, conv2])

        assert result.duplicates_removed == 1
        assert result.original_count == 2
        assert result.unique_count == 1

    def test_stage_priority_preserved(self):
        """Higher priority stage should be preserved."""
        stage2_conv = create_mock_conversation(
            conversation_id="stage2-conv",
            messages=[create_mock_message("user", "Hello")],
            metadata={"stage": "stage2_therapeutic_expertise"}
        )
        stage4_conv = create_mock_conversation(
            conversation_id="stage4-conv",
            messages=[create_mock_message("user", "Hello")],
            metadata={"stage": "stage4_voice_persona"}
        )

        unique, result = deduplicate_conversations([stage2_conv, stage4_conv])

        assert len(unique) == 1
        assert unique[0].conversation_id == "stage4-conv"

    def test_dedup_rate_under_target(self):
        """Deduplication rate should be under 1% target for clean data."""
        # Create 100 unique conversations
        conversations = [
            create_mock_conversation(
                conversation_id=f"conv-{i}",
                messages=[create_mock_message("user", f"Message {i}")]
            )
            for i in range(100)
        ]

        _, result = deduplicate_conversations(conversations)

        # No duplicates, so dedup rate should be 0%
        assert result.dedup_rate == 0.0

    def test_empty_input(self):
        """Empty input should return empty result."""
        unique, result = deduplicate_conversations([])

        assert result.original_count == 0
        assert result.unique_count == 0
        assert result.duplicates_removed == 0


class TestValidation:
    """Test deduplication validation."""

    def test_validate_under_target(self):
        """Validation should pass when under target rate."""
        conversations = [
            create_mock_conversation(
                conversation_id=f"conv-{i}",
                messages=[create_mock_message("user", f"Unique message {i}")]
            )
            for i in range(10)
        ]

        result = validate_deduplication(conversations, target_dedup_rate=0.01)
        assert result is True  # Should pass (0% dedup rate)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
