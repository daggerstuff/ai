"""
Tests for DataStandardizer orchestration class.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from ai.dataset_pipeline.conversation_schema import Conversation
from ai.dataset_pipeline.data_standardizer import (
    DataStandardizer,
    StandardizationResult,
    StandardizationStats,
)


class TestDataStandardizer:
    """Test cases for DataStandardizer."""

    @pytest.fixture
    def standardizer(self):
        """Create a DataStandardizer instance for testing."""
        return DataStandardizer(max_workers=2, batch_size=10, enable_monitoring=True)

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return {
            "simple_messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "input_output": {
                "input": "What is AI?",
                "output": "AI is artificial intelligence."
            },
            "huggingface_chat": {
                "conversations": [[
                    {"role": "user", "content": "Test"},
                    {"role": "assistant", "content": "Response"}
                ]]
            },
            "openai_format": {
                "role": "user",
                "content": "Single message"
            },
            "custom_json": {
                "text": "Custom format text"
            }
        }

    def test_initialization(self):
        """Test DataStandardizer initialization."""
        standardizer = DataStandardizer(max_workers=4, batch_size=50)

        assert standardizer.max_workers == 4
        assert standardizer.batch_size == 50
        assert standardizer.enable_monitoring is True
        assert isinstance(standardizer.stats, StandardizationStats)
        assert len(standardizer.converters) >= 5
        assert standardizer.validators == []

    def test_format_detection(self, standardizer, sample_data):
        """Test automatic format detection."""
        assert standardizer.detect_format(sample_data["simple_messages"]) == "simple_messages"
        assert standardizer.detect_format(sample_data["input_output"]) == "input_output"
        assert standardizer.detect_format(sample_data["huggingface_chat"]) == "huggingface_chat"
        assert standardizer.detect_format(sample_data["openai_format"]) == "openai_format"
        assert standardizer.detect_format(sample_data["custom_json"]) == "custom_json"
        assert standardizer.detect_format({"unknown": "format"}) == "custom_json"
        assert standardizer.detect_format("string") == "unknown"

    def test_register_converter(self, standardizer):
        """Test registering custom converter."""
        def custom_converter(data, source=None, conversation_id=None):
            return Mock(spec=Conversation)

        standardizer.register_converter("custom_format", custom_converter)
        assert "custom_format" in standardizer.converters
        assert standardizer.converters["custom_format"] == custom_converter

    def test_register_validator(self, standardizer):
        """Test registering validator."""
        def custom_validator(conversation):
            return {"valid": True}

        standardizer.register_validator(custom_validator)
        assert len(standardizer.validators) == 1
        assert standardizer.validators[0] == custom_validator

    def test_standardize_single_success(self, standardizer, sample_data):
        """Test successful single item standardization."""
        result = standardizer.standardize_single(
            sample_data["simple_messages"],
            source="test_source"
        )

        assert result.success is True
        assert result.conversation is not None
        assert result.error is None
        assert result.processing_time > 0
        assert result.source_format == "simple_messages"
        assert "message_count" in result.metadata
        assert "total_chars" in result.metadata

    def test_standardize_single_with_format_hint(self, standardizer, sample_data):
        """Test standardization with format hint."""
        result = standardizer.standardize_single(
            sample_data["input_output"],
            format_hint="input_output",
            source="test_source"
        )

        assert result.success is True
        assert result.source_format == "input_output"

    def test_standardize_single_unknown_format(self, standardizer):
        """Test standardization with unknown format."""
        result = standardizer.standardize_single(
            {"unknown": "format"},
            format_hint="nonexistent_format"
        )

        assert result.success is False
        assert "No converter found" in result.error
        assert result.source_format == "nonexistent_format"

    def test_standardize_single_with_validator(self, standardizer, sample_data):
        """Test standardization with validator."""
        def failing_validator(conversation):
            return {"valid": False, "error": "Test validation error"}

        standardizer.register_validator(failing_validator)

        result = standardizer.standardize_single(sample_data["simple_messages"])

        assert result.success is False
        assert "Validation failed" in result.error
        assert "Test validation error" in result.error

    def test_standardize_single_exception(self, standardizer):
        """Test standardization with exception."""
        # Mock converter to raise exception
        def failing_converter(data, source=None, conversation_id=None):
            raise ValueError("Test exception")

        standardizer.register_converter("failing_format", failing_converter)

        result = standardizer.standardize_single(
            {"test": "data"},
            format_hint="failing_format"
        )

        assert result.success is False
        assert "Test exception" in result.error

    def test_standardize_batch(self, standardizer, sample_data):
        """Test batch standardization."""
        data_items = [
            sample_data["simple_messages"],
            sample_data["input_output"],
            sample_data["openai_format"]
        ]

        results = standardizer.standardize_batch(data_items, source="batch_test")

        assert len(results) == 3
        assert all(isinstance(r, StandardizationResult) for r in results)
        assert all(r.success for r in results)

        # Check statistics were updated
        assert standardizer.stats.total_processed == 3
        assert standardizer.stats.successful == 3
        assert standardizer.stats.failed == 0

    def test_standardize_batch_mixed_results(self, standardizer, sample_data):
        """Test batch standardization with mixed success/failure."""
        data_items = [
            sample_data["simple_messages"],  # Should succeed
            {"invalid": "data"},  # Should fail
            sample_data["input_output"]  # Should succeed
        ]

        results = standardizer.standardize_batch(data_items)

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True

    def test_standardize_file_json(self, standardizer, sample_data):
        """Test file standardization with JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([sample_data["simple_messages"], sample_data["input_output"]], f)
            temp_path = f.name

        try:
            summary = standardizer.standardize_file(temp_path)

            assert summary["total_items"] == 2
            assert summary["successful"] == 2
            assert summary["failed"] == 0
            assert summary["success_rate"] == 1.0
            assert summary["total_conversations"] == 2
            assert "processing_time" in summary
            assert "format_distribution" in summary
        finally:
            Path(temp_path).unlink()

    def test_standardize_file_jsonl(self, standardizer, sample_data):
        """Test file standardization with JSONL."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            json.dump(sample_data["simple_messages"], f)
            f.write("\n")
            json.dump(sample_data["input_output"], f)
            f.write("\n")
            temp_path = f.name

        try:
            summary = standardizer.standardize_file(temp_path)

            assert summary["total_items"] == 2
            assert summary["successful"] == 2
        finally:
            Path(temp_path).unlink()

    def test_standardize_file_with_output(self, standardizer, sample_data):
        """Test file standardization with output file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([sample_data["simple_messages"]], f)
            input_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            summary = standardizer.standardize_file(input_path, output_path)

            assert summary["successful"] == 1
            assert Path(output_path).exists()

            # Check output file content
            with open(output_path) as f:
                lines = f.readlines()
                assert len(lines) == 1
                conversation_data = json.loads(lines[0])
                assert "messages" in conversation_data
        finally:
            Path(input_path).unlink()
            Path(output_path).unlink()

    def test_standardize_file_not_found(self, standardizer):
        """Test file standardization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            standardizer.standardize_file("nonexistent_file.json")

    def test_get_stats(self, standardizer, sample_data):
        """Test getting statistics."""
        # Process some data to generate stats
        standardizer.standardize_single(sample_data["simple_messages"])
        standardizer.standardize_single({"invalid": "data"})

        stats = standardizer.get_stats()

        assert isinstance(stats, StandardizationStats)
        assert stats.total_processed == 2
        assert stats.successful == 1
        assert stats.failed == 1
        assert stats.success_rate == 0.5
        assert stats.average_time > 0

    def test_reset_stats(self, standardizer, sample_data):
        """Test resetting statistics."""
        # Generate some stats
        standardizer.standardize_single(sample_data["simple_messages"])
        assert standardizer.stats.total_processed == 1

        # Reset stats
        standardizer.reset_stats()
        assert standardizer.stats.total_processed == 0
        assert standardizer.stats.successful == 0
        assert standardizer.stats.failed == 0

    def test_converter_methods(self, standardizer, sample_data):
        """Test individual converter methods."""
        # Test simple messages converter
        conv = standardizer._convert_simple_messages(sample_data["simple_messages"])
        assert isinstance(conv, Conversation)
        assert len(conv.messages) == 2

        # Test input/output converter
        conv = standardizer._convert_input_output(sample_data["input_output"])
        assert isinstance(conv, Conversation)
        assert len(conv.messages) == 2

        # Test HuggingFace converter
        conv = standardizer._convert_huggingface_chat(sample_data["huggingface_chat"])
        assert isinstance(conv, Conversation)

        # Test OpenAI converter
        conv = standardizer._convert_openai_format(sample_data["openai_format"])
        assert isinstance(conv, Conversation)

        # Test custom JSON converter
        conv = standardizer._convert_custom_json(sample_data["custom_json"])
        assert isinstance(conv, Conversation)

    def test_converter_errors(self, standardizer):
        """Test converter error handling."""
        # Test simple messages with invalid data
        with pytest.raises(ValueError):
            standardizer._convert_simple_messages("invalid")

        # Test HuggingFace with no conversations
        with pytest.raises(ValueError):
            standardizer._convert_huggingface_chat({"conversations": []})

        # Test custom JSON with no recognizable fields
        with pytest.raises(ValueError):
            standardizer._convert_custom_json({"unknown": "field"})


class TestStandardizationResult:
    """Test cases for StandardizationResult."""

    def test_initialization(self):
        """Test StandardizationResult initialization."""
        result = StandardizationResult(success=True)

        assert result.success is True
        assert result.conversation is None
        assert result.error is None
        assert result.processing_time == 0.0
        assert result.source_format is None
        assert result.metadata == {}


class TestStandardizationStats:
    """Test cases for StandardizationStats."""

    def test_initialization(self):
        """Test StandardizationStats initialization."""
        stats = StandardizationStats()

        assert stats.total_processed == 0
        assert stats.successful == 0
        assert stats.failed == 0
        assert stats.total_time == 0.0
        assert stats.format_counts == {}
        assert stats.error_counts == {}

    def test_success_rate(self):
        """Test success rate calculation."""
        stats = StandardizationStats()

        # No data
        assert stats.success_rate == 0.0

        # With data
        stats.total_processed = 10
        stats.successful = 8
        assert stats.success_rate == 0.8

    def test_average_time(self):
        """Test average time calculation."""
        stats = StandardizationStats()

        # No data
        assert stats.average_time == 0.0

        # With data
        stats.total_processed = 5
        stats.total_time = 10.0
        assert stats.average_time == 2.0
