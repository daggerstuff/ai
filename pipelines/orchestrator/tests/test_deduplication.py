"""
Unit tests for conversation deduplication and similarity detection system.
"""

import datetime

from .conversation_schema import Conversation, Message
from .deduplication import (
    ConversationDeduplicator,
    DuplicationResult,
    SimilarityMetrics,
    conversation_text,
    deduplicate_conversations,
    deduplicate_conversations_simple,
    jaccard_similarity,
)


def make_conv(messages):
    return Conversation(
        id="dc", messages=messages, source="testset", created_at=datetime.datetime.now()
    )


def test_deduplicate_exact_duplicates():
    convs = [
        make_conv(
            [
                Message(role="user", content="Hello, how are you?", timestamp=None),
                Message(role="assistant", content="I'm good, thanks!", timestamp=None),
            ]
        ),
        make_conv(
            [
                Message(role="user", content="Hello, how are you?", timestamp=None),
                Message(role="assistant", content="I'm good, thanks!", timestamp=None),
            ]
        ),
    ]
    unique, duplicates = deduplicate_conversations(convs, similarity_threshold=0.95)
    assert len(unique) == 1
    assert len(duplicates) == 1
    assert duplicates[0][2] >= 0.95


def test_deduplicate_similar_but_not_duplicate():
    convs = [
        make_conv(
            [
                Message(role="user", content="Hello, how are you?", timestamp=None),
                Message(role="assistant", content="I'm good, thanks!", timestamp=None),
            ]
        ),
        make_conv(
            [
                Message(role="user", content="Hi, how are you doing?", timestamp=None),
                Message(role="assistant", content="I'm well, thank you!", timestamp=None),
            ]
        ),
    ]
    unique, duplicates = deduplicate_conversations(convs, similarity_threshold=0.8)
    assert len(unique) == 2
    assert len(duplicates) == 0


def test_deduplicate_empty_input():
    unique, duplicates = deduplicate_conversations([], similarity_threshold=0.9)
    assert unique == []
    assert duplicates == []


class TestJaccardSimilarity:
    """Test basic Jaccard similarity function."""

    def test_identical_strings(self):
        """Test Jaccard similarity with identical strings."""
        result = jaccard_similarity("hello world", "hello world")
        assert result == 1.0

    def test_completely_different_strings(self):
        """Test Jaccard similarity with completely different strings."""
        result = jaccard_similarity("hello world", "foo bar")
        assert result == 0.0

    def test_partial_overlap(self):
        """Test Jaccard similarity with partial overlap."""
        result = jaccard_similarity("hello world", "hello universe")
        expected = 1 / 3  # "hello" is common, "world", "universe" are different
        assert abs(result - expected) < 0.001

    def test_empty_strings(self):
        """Test Jaccard similarity with empty strings."""
        assert jaccard_similarity("", "") == 1.0
        assert jaccard_similarity("hello", "") == 0.0
        assert jaccard_similarity("", "world") == 0.0

    def test_case_insensitive(self):
        """Test that Jaccard similarity is case insensitive."""
        result = jaccard_similarity("Hello World", "hello world")
        assert result == 1.0


class TestConversationText:
    """Test conversation text extraction function."""

    def test_single_message(self):
        """Test text extraction from single message conversation."""
        conv = Conversation(
            id="test1",
            messages=[Message(role="user", content="Hello there")],
            source="test"
        )
        result = conversation_text(conv)
        assert result == "Hello there"

    def test_multiple_messages(self):
        """Test text extraction from multi-message conversation."""
        conv = Conversation(
            id="test2",
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
                Message(role="user", content="How are you?")
            ],
            source="test"
        )
        result = conversation_text(conv)
        assert result == "Hello Hi there How are you?"


class TestConversationDeduplicator:
    """Test the comprehensive conversation deduplication system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.deduplicator = ConversationDeduplicator(similarity_threshold=0.8)

        # Create test conversations
        self.conv1 = Conversation(
            id="conv1",
            messages=[
                Message(role="user", content="I'm feeling anxious about work."),
                Message(role="assistant", content="I understand. Can you tell me more about what's causing this anxiety?")
            ],
            source="test"
        )

        self.conv2 = Conversation(
            id="conv2",
            messages=[
                Message(role="user", content="I'm feeling anxious about work."),
                Message(role="assistant", content="I understand. Can you tell me more about what's causing this anxiety?")
            ],
            source="test"
        )

        self.conv3 = Conversation(
            id="conv3",
            messages=[
                Message(role="user", content="I'm worried about my job performance."),
                Message(role="assistant", content="Work-related stress is common. What specific aspects concern you?")
            ],
            source="test"
        )

        self.conv4 = Conversation(
            id="conv4",
            messages=[
                Message(role="user", content="I love sunny days and ice cream."),
                Message(role="assistant", content="That sounds wonderful! What's your favorite flavor?")
            ],
            source="test"
        )

    def test_initialization(self):
        """Test deduplicator initialization."""
        dedup = ConversationDeduplicator(
            similarity_threshold=0.9,
            content_weight=0.5,
            semantic_weight=0.3,
            structural_weight=0.2
        )

        assert dedup.similarity_threshold == 0.9
        # Weights should be normalized
        assert abs(dedup.content_weight + dedup.semantic_weight + dedup.structural_weight - 1.0) < 0.001

    def test_calculate_similarity_identical(self):
        """Test similarity calculation for identical conversations."""
        similarity = self.deduplicator.calculate_similarity(self.conv1, self.conv2)

        assert isinstance(similarity, SimilarityMetrics)
        assert similarity.content_similarity == 1.0
        assert similarity.semantic_similarity == 1.0
        assert similarity.structural_similarity == 1.0
        assert similarity.overall_similarity == 1.0

    def test_calculate_similarity_different(self):
        """Test similarity calculation for different conversations."""
        similarity = self.deduplicator.calculate_similarity(self.conv1, self.conv4)

        assert isinstance(similarity, SimilarityMetrics)
        assert similarity.content_similarity < 0.5
        assert similarity.semantic_similarity < 0.5
        assert similarity.overall_similarity < 0.5

    def test_calculate_similarity_similar(self):
        """Test similarity calculation for similar conversations."""
        similarity = self.deduplicator.calculate_similarity(self.conv1, self.conv3)

        assert isinstance(similarity, SimilarityMetrics)
        # Should have some similarity due to work/anxiety theme
        assert 0.2 < similarity.semantic_similarity < 0.8
        assert similarity.structural_similarity > 0.8  # Same structure

    def test_exact_duplicate_detection(self):
        """Test exact duplicate detection using content hashing."""
        conversations = [self.conv1, self.conv2, self.conv3]  # conv1 and conv2 are identical

        exact_duplicates = self.deduplicator._find_exact_duplicates(conversations)

        assert len(exact_duplicates) == 1
        duplicate_group = next(iter(exact_duplicates.values()))
        assert len(duplicate_group) == 2
        assert "conv1" in duplicate_group
        assert "conv2" in duplicate_group

    def test_remove_exact_duplicates(self):
        """Test removal of exact duplicates."""
        conversations = [self.conv1, self.conv2, self.conv3]
        exact_duplicates = self.deduplicator._find_exact_duplicates(conversations)

        unique = self.deduplicator._remove_exact_duplicates(conversations, exact_duplicates)

        assert len(unique) == 2
        unique_ids = [conv.id for conv in unique]
        assert "conv1" in unique_ids  # First occurrence kept
        assert "conv2" not in unique_ids  # Duplicate removed
        assert "conv3" in unique_ids

    def test_deduplicate_conversations_full(self):
        """Test full deduplication process."""
        conversations = [self.conv1, self.conv2, self.conv3, self.conv4]

        unique, result = self.deduplicator.deduplicate_conversations(conversations)

        assert isinstance(result, DuplicationResult)
        assert result.original_count == 4
        assert result.unique_count == 3  # conv1 and conv2 are duplicates
        assert result.duplicates_removed == 1
        assert len(result.duplicate_groups) >= 1

        unique_ids = [conv.id for conv in unique]
        assert len(unique_ids) == 3
        assert "conv1" in unique_ids
        assert "conv2" not in unique_ids  # Should be removed as duplicate

    def test_similarity_distribution_calculation(self):
        """Test similarity distribution calculation."""
        conversations = [self.conv1, self.conv2, self.conv3, self.conv4]

        distribution = self.deduplicator._calculate_similarity_distribution(conversations)

        assert isinstance(distribution, dict)
        expected_keys = ["very_high", "high", "medium", "low", "very_low"]
        assert all(key in distribution for key in expected_keys)
        assert all(isinstance(count, int) for count in distribution.values())


class TestBackwardCompatibility:
    """Test backward compatibility functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.conv1 = Conversation(
            id="conv1",
            messages=[Message(role="user", content="Hello world")],
            source="test"
        )

        self.conv2 = Conversation(
            id="conv2",
            messages=[Message(role="user", content="Hello world")],
            source="test"
        )

        self.conv3 = Conversation(
            id="conv3",
            messages=[Message(role="user", content="Goodbye universe")],
            source="test"
        )

    def test_deduplicate_conversations_enhanced(self):
        """Test enhanced deduplicate_conversations function."""
        conversations = [self.conv1, self.conv2, self.conv3]

        unique, duplicates = deduplicate_conversations(conversations, similarity_threshold=0.9)

        assert len(unique) == 2  # conv1 and conv2 are duplicates
        assert len(duplicates) >= 1

        # Check duplicate format
        for dup in duplicates:
            assert len(dup) == 3  # (idx1, idx2, similarity)
            assert isinstance(dup[0], int)
            assert isinstance(dup[1], int)
            assert isinstance(dup[2], float)

    def test_deduplicate_conversations_simple(self):
        """Test simple deduplicate_conversations function."""
        conversations = [self.conv1, self.conv2, self.conv3]

        unique, duplicates = deduplicate_conversations_simple(conversations, similarity_threshold=0.9)

        assert len(unique) == 2  # Should remove one duplicate
        assert len(duplicates) >= 1

    def test_no_duplicates(self):
        """Test deduplication when no duplicates exist."""
        conversations = [self.conv1, self.conv3]  # Different conversations

        unique, duplicates = deduplicate_conversations(conversations, similarity_threshold=0.9)

        assert len(unique) == 2
        assert len(duplicates) == 0

    def test_all_duplicates(self):
        """Test deduplication when all conversations are duplicates."""
        conversations = [self.conv1, self.conv2, self.conv2]  # All same content

        unique, duplicates = deduplicate_conversations(conversations, similarity_threshold=0.9)

        assert len(unique) == 1  # Only one unique conversation
        assert len(duplicates) >= 1  # At least 1 duplicate removed


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_conversation_list(self):
        """Test deduplication with empty conversation list."""
        deduplicator = ConversationDeduplicator()

        unique, result = deduplicator.deduplicate_conversations([])

        assert len(unique) == 0
        assert result.original_count == 0
        assert result.unique_count == 0
        assert result.duplicates_removed == 0

    def test_single_conversation(self):
        """Test deduplication with single conversation."""
        conv = Conversation(
            id="single",
            messages=[Message(role="user", content="Hello")],
            source="test"
        )

        deduplicator = ConversationDeduplicator()
        unique, result = deduplicator.deduplicate_conversations([conv])

        assert len(unique) == 1
        assert result.original_count == 1
        assert result.unique_count == 1
        assert result.duplicates_removed == 0

    def test_empty_messages(self):
        """Test handling of conversations with empty messages."""
        conv1 = Conversation(id="empty1", messages=[], source="test")
        conv2 = Conversation(id="empty2", messages=[], source="test")

        deduplicator = ConversationDeduplicator()
        unique, result = deduplicator.deduplicate_conversations([conv1, conv2])

        # Should handle empty conversations gracefully
        assert len(unique) <= 2
        assert result.original_count == 2

    def test_very_low_threshold(self):
        """Test deduplication with very low similarity threshold."""
        conv1 = Conversation(
            id="conv1",
            messages=[Message(role="user", content="Hello")],
            source="test"
        )
        conv2 = Conversation(
            id="conv2",
            messages=[Message(role="user", content="Goodbye")],
            source="test"
        )

        deduplicator = ConversationDeduplicator(similarity_threshold=0.1)
        unique, result = deduplicator.deduplicate_conversations([conv1, conv2])

        # With very low threshold, even different conversations might be considered duplicates
        assert len(unique) >= 1
        assert result.original_count == 2

    def test_very_high_threshold(self):
        """Test deduplication with very high similarity threshold."""
        conv1 = Conversation(
            id="conv1",
            messages=[Message(role="user", content="Hello world")],
            source="test"
        )
        conv2 = Conversation(
            id="conv2",
            messages=[Message(role="user", content="Hello world!")],  # Slight difference
            source="test"
        )

        deduplicator = ConversationDeduplicator(similarity_threshold=0.99)
        unique, result = deduplicator.deduplicate_conversations([conv1, conv2])

        # With very high threshold, even similar conversations should be kept
        assert len(unique) == 2
        assert result.duplicates_removed == 0
