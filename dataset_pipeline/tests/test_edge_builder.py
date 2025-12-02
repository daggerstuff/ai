#!/usr/bin/env python3
"""
Tests for EdgeDatasetBuilder, focusing on conversation normalization and strict mode.
"""

import pytest
from ai.dataset_pipeline.edge.edge_builder import (
    EdgeDatasetBuilder,
    RawEdgeExample,
    EdgeExample,
)
from ai.dataset_pipeline.types.edge_categories import (
    EdgeCategory,
    IntensityLevel,
)


class TestConversationNormalization:
    """Test conversation normalization in strict and non-strict modes."""

    def test_strict_mode_raises_on_invalid_message_format(self):
        """Test that strict mode (default) raises ValueError on invalid messages."""
        builder = EdgeDatasetBuilder(strict_conversation_format=True)

        invalid_conversation = [
            {"role": "user", "content": "Valid message"},
            {"invalid": "format"},  # Missing role/content
            {"role": "assistant"},  # Missing content
        ]

        with pytest.raises(ValueError, match="Invalid message format"):
            builder._normalize_conversation(invalid_conversation)

    def test_strict_mode_raises_on_non_dict_message(self):
        """Test that strict mode raises on non-dict messages."""
        builder = EdgeDatasetBuilder(strict_conversation_format=True)

        invalid_conversation = [
            {"role": "user", "content": "Valid"},
            "not a dict",  # Invalid type
        ]

        with pytest.raises(ValueError, match="Invalid message format"):
            builder._normalize_conversation(invalid_conversation)

    def test_non_strict_mode_skips_invalid_messages(self):
        """Test that non-strict mode logs warnings and skips invalid messages."""
        builder = EdgeDatasetBuilder(strict_conversation_format=False)

        conversation = [
            {"role": "user", "content": "First valid"},
            {"invalid": "format"},  # Should be skipped
            {"role": "assistant", "content": "Second valid"},
            "not a dict",  # Should be skipped
        ]

        normalized = builder._normalize_conversation(conversation)

        assert len(normalized) == 2
        assert normalized[0]["content"] == "First valid"
        assert normalized[1]["content"] == "Second valid"

    def test_strict_mode_validates_all_valid_messages(self):
        """Test that strict mode accepts all valid messages."""
        builder = EdgeDatasetBuilder(strict_conversation_format=True)

        conversation = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]

        normalized = builder._normalize_conversation(conversation)

        assert len(normalized) == 3
        assert normalized[0]["role"] == "user"
        assert normalized[0]["content"] == "First"
        assert normalized[1]["role"] == "assistant"
        assert normalized[1]["content"] == "Second"

    def test_strict_mode_defaults_to_true(self):
        """Test that strict mode defaults to True for safety."""
        builder = EdgeDatasetBuilder()  # No explicit flag

        invalid_conversation = [
            {"role": "user", "content": "Valid"},
            {"invalid": "format"},
        ]

        with pytest.raises(ValueError, match="Invalid message format"):
            builder._normalize_conversation(invalid_conversation)

    def test_empty_conversation_list_returns_fallback(self):
        """Test that empty conversation list returns fallback message (consistent with string inputs)."""
        builder_strict = EdgeDatasetBuilder(strict_conversation_format=True)
        builder_non_strict = EdgeDatasetBuilder(strict_conversation_format=False)

        # Empty list returns fallback in both modes (consistent with string inputs)
        # Strict mode only raises on invalid message formats, not on empty lists
        for builder in [builder_strict, builder_non_strict]:
            result = builder._normalize_conversation([])
            assert len(result) == 1
            assert result[0]["role"] == "user"
            assert result[0]["content"] == ""

    def test_all_invalid_messages_in_non_strict_mode_returns_fallback(self):
        """Test that if all messages are invalid in non-strict mode, it returns fallback message."""
        builder = EdgeDatasetBuilder(strict_conversation_format=False)

        invalid_conversation = [
            {"invalid": "format"},
            "not a dict",
        ]

        result = builder._normalize_conversation(invalid_conversation)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == ""


class TestBuildEdgeExampleWithStrictMode:
    """Test building edge examples with strict mode enabled."""

    def test_build_example_with_valid_conversation(self):
        """Test building example with valid conversation succeeds."""
        builder = EdgeDatasetBuilder(strict_conversation_format=True)

        raw_example = RawEdgeExample(
            conversation=[
                {"role": "user", "content": "I'm struggling"},
                {"role": "assistant", "content": "I'm here to help"},
            ],
            category=EdgeCategory.SUICIDAL_IDEATION,
            intensity=IntensityLevel.HIGH,
        )

        example = builder.build_edge_example(raw_example)

        assert isinstance(example, EdgeExample)
        assert len(example.conversation) == 2

    def test_build_example_with_invalid_conversation_raises(self):
        """Test that building example with invalid conversation raises in strict mode."""
        builder = EdgeDatasetBuilder(strict_conversation_format=True)

        raw_example = RawEdgeExample(
            conversation=[
                {"role": "user", "content": "Valid"},
                {"invalid": "format"},
            ],
            category=EdgeCategory.SUICIDAL_IDEATION,
            intensity=IntensityLevel.HIGH,
        )

        with pytest.raises(ValueError, match="Invalid message format"):
            builder.build_edge_example(raw_example)

    def test_build_example_with_invalid_conversation_non_strict(self):
        """Test that non-strict mode allows building with invalid messages skipped."""
        builder = EdgeDatasetBuilder(strict_conversation_format=False)

        raw_example = RawEdgeExample(
            conversation=[
                {"role": "user", "content": "Valid"},
                {"invalid": "format"},  # Will be skipped
            ],
            category=EdgeCategory.SUICIDAL_IDEATION,
            intensity=IntensityLevel.HIGH,
        )

        example = builder.build_edge_example(raw_example)

        assert isinstance(example, EdgeExample)
        assert len(example.conversation) == 1  # Only valid message kept

    def test_build_example_with_all_invalid_messages_uses_fallback(self):
        """Test that building with all invalid messages uses fallback in non-strict mode."""
        builder = EdgeDatasetBuilder(strict_conversation_format=False)

        raw_example = RawEdgeExample(
            conversation=[
                {"invalid": "format"},
                "not a dict",
            ],
            category=EdgeCategory.SUICIDAL_IDEATION,
            intensity=IntensityLevel.HIGH,
        )

        example = builder.build_edge_example(raw_example)

        assert isinstance(example, EdgeExample)
        assert len(example.conversation) == 1
        assert example.conversation[0]["role"] == "user"
        assert example.conversation[0]["content"] == ""


class TestStringConversationParsing:
    """Test string conversation parsing (should work in both modes)."""

    def test_string_conversation_parsing(self):
        """Test that string conversations are parsed correctly."""
        builder = EdgeDatasetBuilder(strict_conversation_format=True)

        conversation_str = "Therapist: How are you?\nClient: I'm struggling"
        normalized = builder._normalize_conversation(conversation_str)

        assert len(normalized) == 2
        assert normalized[0]["role"] == "therapist"
        assert normalized[1]["role"] == "client"

    def test_string_conversation_without_role_markers(self):
        """Test string conversation without role markers."""
        builder = EdgeDatasetBuilder(strict_conversation_format=True)

        conversation_str = "This is a simple message"
        normalized = builder._normalize_conversation(conversation_str)

        assert len(normalized) == 1
        assert normalized[0]["role"] == "user"
        assert normalized[0]["content"] == conversation_str


class TestBuildEdgeDatasetWithErrors:
    """Test build_edge_dataset_with_errors method for error handling."""

    def test_build_dataset_with_errors_returns_both_examples_and_errors(self):
        """Test that build_edge_dataset_with_errors returns examples and errors."""
        builder = EdgeDatasetBuilder(strict_conversation_format=False)

        raw_examples = [
            RawEdgeExample(
                conversation=[{"role": "user", "content": "Valid message"}],
                category=EdgeCategory.SUICIDAL_IDEATION,
                intensity=IntensityLevel.HIGH,
            ),
            RawEdgeExample(
                conversation=[{"invalid": "format"}],  # Will fail in strict mode
                category=EdgeCategory.SUICIDAL_IDEATION,
                intensity=IntensityLevel.HIGH,
            ),
            RawEdgeExample(
                conversation=[{"role": "user", "content": "Another valid"}],
                category=EdgeCategory.DOMESTIC_VIOLENCE,
                intensity=IntensityLevel.MODERATE,
            ),
        ]

        examples, errors = builder.build_edge_dataset_with_errors(raw_examples)

        assert len(examples) == 3  # All succeed in non-strict mode
        assert len(errors) == 0

    def test_build_dataset_with_errors_captures_failures(self):
        """Test that build_edge_dataset_with_errors captures and returns errors."""
        builder = EdgeDatasetBuilder(strict_conversation_format=True)

        raw_examples = [
            RawEdgeExample(
                conversation=[{"role": "user", "content": "Valid"}],
                category=EdgeCategory.SUICIDAL_IDEATION,
                intensity=IntensityLevel.HIGH,
            ),
            RawEdgeExample(
                conversation=[{"invalid": "format"}],  # Will fail
                category=EdgeCategory.SUICIDAL_IDEATION,
                intensity=IntensityLevel.HIGH,
            ),
            RawEdgeExample(
                conversation=[{"role": "user", "content": "Also valid"}],
                category=EdgeCategory.DOMESTIC_VIOLENCE,
                intensity=IntensityLevel.MODERATE,
            ),
        ]

        examples, errors = builder.build_edge_dataset_with_errors(raw_examples)

        assert len(examples) == 2  # Two succeed
        assert len(errors) == 1  # One fails
        assert "Error building example 1" in errors[0]
        assert "Invalid message format" in errors[0]

    def test_build_dataset_with_errors_threshold_raises_when_exceeded(self):
        """Test that error_threshold raises ValueError when exceeded."""
        builder = EdgeDatasetBuilder(strict_conversation_format=True)

        raw_examples = [
            RawEdgeExample(
                conversation=[{"role": "user", "content": "Valid"}],
                category=EdgeCategory.SUICIDAL_IDEATION,
                intensity=IntensityLevel.HIGH,
            ),
            RawEdgeExample(
                conversation=[{"invalid": "format"}],  # Will fail
                category=EdgeCategory.SUICIDAL_IDEATION,
                intensity=IntensityLevel.HIGH,
            ),
            RawEdgeExample(
                conversation=[{"invalid2": "format"}],  # Will fail
                category=EdgeCategory.SUICIDAL_IDEATION,
                intensity=IntensityLevel.HIGH,
            ),
        ]

        # Error rate is 2/3 = 66.7%, threshold is 50%
        with pytest.raises(ValueError, match="Error rate.*exceeds threshold"):
            builder.build_edge_dataset_with_errors(raw_examples, error_threshold=0.5)

    def test_build_dataset_with_errors_threshold_allows_below_threshold(self):
        """Test that error_threshold allows builds below threshold."""
        builder = EdgeDatasetBuilder(strict_conversation_format=True)

        raw_examples = [
            RawEdgeExample(
                conversation=[{"role": "user", "content": "Valid"}],
                category=EdgeCategory.SUICIDAL_IDEATION,
                intensity=IntensityLevel.HIGH,
            ),
            RawEdgeExample(
                conversation=[{"role": "user", "content": "Also valid"}],
                category=EdgeCategory.DOMESTIC_VIOLENCE,
                intensity=IntensityLevel.MODERATE,
            ),
            RawEdgeExample(
                conversation=[{"invalid": "format"}],  # Will fail
                category=EdgeCategory.SUICIDAL_IDEATION,
                intensity=IntensityLevel.HIGH,
            ),
        ]

        # Error rate is 1/3 = 33.3%, threshold is 50%
        examples, errors = builder.build_edge_dataset_with_errors(
            raw_examples, error_threshold=0.5
        )

        assert len(examples) == 2
        assert len(errors) == 1

    def test_build_dataset_with_errors_zero_threshold_raises_on_any_error(self):
        """Test that error_threshold=0.0 raises on any error."""
        builder = EdgeDatasetBuilder(strict_conversation_format=True)

        raw_examples = [
            RawEdgeExample(
                conversation=[{"role": "user", "content": "Valid"}],
                category=EdgeCategory.SUICIDAL_IDEATION,
                intensity=IntensityLevel.HIGH,
            ),
            RawEdgeExample(
                conversation=[{"invalid": "format"}],  # Will fail
                category=EdgeCategory.SUICIDAL_IDEATION,
                intensity=IntensityLevel.HIGH,
            ),
        ]

        # Error rate is 1/2 = 50%, threshold is 0%
        with pytest.raises(ValueError, match="Error rate.*exceeds threshold"):
            builder.build_edge_dataset_with_errors(raw_examples, error_threshold=0.0)

    def test_build_dataset_backward_compatibility(self):
        """Test that build_edge_dataset maintains backward compatibility."""
        builder = EdgeDatasetBuilder(strict_conversation_format=True)

        raw_examples = [
            RawEdgeExample(
                conversation=[{"role": "user", "content": "Valid"}],
                category=EdgeCategory.SUICIDAL_IDEATION,
                intensity=IntensityLevel.HIGH,
            ),
            RawEdgeExample(
                conversation=[{"invalid": "format"}],  # Will fail
                category=EdgeCategory.SUICIDAL_IDEATION,
                intensity=IntensityLevel.HIGH,
            ),
        ]

        # Should return only examples, not errors (backward compatible)
        examples = builder.build_edge_dataset(raw_examples)

        assert isinstance(examples, list)
        assert len(examples) == 1  # Only successful example
        assert all(isinstance(ex, EdgeExample) for ex in examples)

    def test_build_dataset_with_errors_empty_input(self):
        """Test build_edge_dataset_with_errors with empty input."""
        builder = EdgeDatasetBuilder()

        examples, errors = builder.build_edge_dataset_with_errors([])

        assert examples == []
        assert errors == []

    def test_build_dataset_with_errors_all_fail(self):
        """Test build_edge_dataset_with_errors when all examples fail."""
        builder = EdgeDatasetBuilder(strict_conversation_format=True)

        raw_examples = [
            RawEdgeExample(
                conversation=[{"invalid": "format"}],
                category=EdgeCategory.SUICIDAL_IDEATION,
                intensity=IntensityLevel.HIGH,
            ),
            RawEdgeExample(
                conversation=[{"invalid2": "format"}],
                category=EdgeCategory.SUICIDAL_IDEATION,
                intensity=IntensityLevel.HIGH,
            ),
        ]

        examples, errors = builder.build_edge_dataset_with_errors(raw_examples)

        assert len(examples) == 0
        assert len(errors) == 2
        assert all("Error building example" in err for err in errors)

