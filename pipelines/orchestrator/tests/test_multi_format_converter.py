"""
Tests for MultiFormatConverter.
"""

from unittest.mock import Mock, patch

import pytest

from ai.pipelines.orchestrator.conversation_schema import Conversation
from ai.pipelines.orchestrator.multi_format_converter import (
    ConversionRule,
    FormatDetectionResult,
    FormatType,
    MultiFormatConverter,
)


class TestMultiFormatConverter:
    """Test cases for MultiFormatConverter."""

    @pytest.fixture
    def converter(self):
        """Create a MultiFormatConverter instance for testing."""
        return MultiFormatConverter()

    @pytest.fixture
    def sample_formats(self):
        """Sample data in various formats."""
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
            "sharegpt": {
                "id": "test_123",
                "title": "Test Conversation",
                "conversations": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"}
                ]
            },
            "alpaca": {
                "instruction": "Explain the concept",
                "input": "What is machine learning?",
                "output": "Machine learning is a subset of AI."
            },
            "vicuna": {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi there!"}
                ]
            },
            "dolly": {
                "instruction": "Answer the question",
                "context": "In the field of AI",
                "response": "AI stands for Artificial Intelligence."
            },
            "oasst": {
                "message_id": "msg_123",
                "parent_id": "parent_456",
                "role": "prompter",
                "text": "What is the weather like?"
            },
            "xml_chat": '<conversation><message role="user">Hello</message><message role="assistant">Hi!</message></conversation>',
            "csv_format": "user,assistant\nHello,Hi there!",
            "custom_json": {
                "text": "This is custom format text",
                "metadata": {"source": "custom"}
            }
        }

    def test_initialization(self, converter):
        """Test MultiFormatConverter initialization."""
        assert len(converter.format_detectors) >= 12
        assert len(converter.custom_converters) == 0
        assert len(converter.conversion_rules) == 0

    def test_detect_simple_messages(self, converter, sample_formats):
        """Test detection of simple messages format."""
        result = converter.detect_format(sample_formats["simple_messages"])

        assert result.format_type == FormatType.SIMPLE_MESSAGES
        assert result.confidence >= 0.9
        assert "Has role and content fields" in result.indicators
        assert "Contains standard chat roles" in result.indicators

    def test_detect_input_output(self, converter, sample_formats):
        """Test detection of input/output format."""
        result = converter.detect_format(sample_formats["input_output"])

        assert result.format_type == FormatType.INPUT_OUTPUT
        assert result.confidence >= 0.9
        assert "Has input and output fields" in result.indicators
        assert "Input and output are strings" in result.indicators

    def test_detect_huggingface_chat(self, converter, sample_formats):
        """Test detection of HuggingFace chat format."""
        result = converter.detect_format(sample_formats["huggingface_chat"])

        assert result.format_type == FormatType.HUGGINGFACE_CHAT
        assert result.confidence >= 0.8
        assert "Has conversations field" in result.indicators

    def test_detect_openai_format(self, converter, sample_formats):
        """Test detection of OpenAI format."""
        result = converter.detect_format(sample_formats["openai_format"])

        assert result.format_type == FormatType.OPENAI_FORMAT
        assert result.confidence >= 0.7
        assert "Has role and content fields" in result.indicators

    def test_detect_alpaca(self, converter, sample_formats):
        """Test detection of Alpaca format."""
        result = converter.detect_format(sample_formats["alpaca"])

        assert result.format_type == FormatType.ALPACA
        assert result.confidence >= 0.8
        assert "Has all Alpaca fields" in result.indicators[0] or "Has instruction and output fields" in result.indicators[0]

    def test_detect_vicuna(self, converter, sample_formats):
        """Test detection of Vicuna format."""
        result = converter.detect_format(sample_formats["vicuna"])

        assert result.format_type == FormatType.VICUNA
        assert result.confidence >= 0.7
        assert "Has Vicuna-style from/value structure" in result.indicators

    def test_detect_dolly(self, converter, sample_formats):
        """Test detection of Dolly format."""
        result = converter.detect_format(sample_formats["dolly"])

        assert result.format_type == FormatType.DOLLY
        assert result.confidence >= 0.7

    def test_detect_oasst(self, converter, sample_formats):
        """Test detection of OpenAssistant format."""
        result = converter.detect_format(sample_formats["oasst"])

        assert result.format_type == FormatType.OASST
        assert result.confidence >= 0.7
        assert "Has text and role fields" in result.indicators

    def test_detect_xml_chat(self, converter, sample_formats):
        """Test detection of XML chat format."""
        result = converter.detect_format(sample_formats["xml_chat"])

        assert result.format_type == FormatType.XML_CHAT
        assert result.confidence >= 0.8
        assert "Valid XML structure" in result.indicators

    def test_detect_csv_format(self, converter, sample_formats):
        """Test detection of CSV format."""
        result = converter.detect_format(sample_formats["csv_format"])

        assert result.format_type == FormatType.CSV_FORMAT
        assert result.confidence >= 0.7
        assert "Has chat-related CSV headers" in result.indicators

    def test_detect_unknown_format(self, converter):
        """Test detection of unknown format."""
        result = converter.detect_format("unknown string format")

        assert result.format_type == FormatType.UNKNOWN
        assert result.confidence == 0.0
        assert "No format patterns matched" in result.indicators

    def test_convert_single_with_hint(self, converter, sample_formats):
        """Test single conversion with format hint."""
        conversation = converter.convert_single(
            sample_formats["simple_messages"],
            format_hint=FormatType.SIMPLE_MESSAGES,
            source="test_source"
        )

        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2
        assert conversation.source == "test_source"

    def test_convert_single_auto_detect(self, converter, sample_formats):
        """Test single conversion with automatic detection."""
        conversation = converter.convert_single(
            sample_formats["input_output"],
            source="test_source"
        )

        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2
        assert conversation.messages[0].role == "user"
        assert conversation.messages[1].role == "assistant"

    def test_convert_single_unknown_format(self, converter):
        """Test single conversion with unknown format."""
        with pytest.raises(ValueError, match="Unable to detect format"):
            converter.convert_single("unknown format")

    def test_convert_single_no_converter(self, converter):
        """Test single conversion with no available converter."""
        # Mock a format type that doesn't have a converter
        with patch.object(converter, "detect_format") as mock_detect:
            mock_detect.return_value = FormatDetectionResult(
                format_type=FormatType.UNKNOWN,
                confidence=0.9
            )

            with pytest.raises(ValueError, match="No converter available"):
                converter.convert_single({"test": "data"})

    def test_convert_batch(self, converter, sample_formats):
        """Test batch conversion."""
        data_items = [
            sample_formats["simple_messages"],
            sample_formats["input_output"],
            sample_formats["alpaca"]
        ]

        conversations = converter.convert_batch(data_items, source="batch_test")

        assert len(conversations) == 3
        assert all(isinstance(conv, Conversation) for conv in conversations)
        assert all(conv.source == "batch_test" for conv in conversations)

    def test_convert_batch_with_hint(self, converter, sample_formats):
        """Test batch conversion with format hint."""
        data_items = [
            sample_formats["simple_messages"],
            sample_formats["simple_messages"]  # Same format
        ]

        conversations = converter.convert_batch(
            data_items,
            format_hint=FormatType.SIMPLE_MESSAGES,
            source="batch_test"
        )

        assert len(conversations) == 2
        assert all(isinstance(conv, Conversation) for conv in conversations)

    def test_convert_batch_with_failures(self, converter, sample_formats):
        """Test batch conversion with some failures."""
        data_items = [
            sample_formats["simple_messages"],  # Should succeed
            "invalid data",  # Should fail
            sample_formats["input_output"]  # Should succeed
        ]

        conversations = converter.convert_batch(data_items)

        # Should only return successful conversions
        assert len(conversations) == 2
        assert all(isinstance(conv, Conversation) for conv in conversations)

    def test_register_custom_converter(self, converter):
        """Test registering custom converter."""
        def custom_detector(data, context=None):
            return 0.9, ["Custom format detected"], {}

        def custom_converter(data, source=None, conversation_id=None):
            return Mock(spec=Conversation)

        converter.register_custom_converter(
            "custom_format",
            custom_detector,
            custom_converter,
            "Test custom format"
        )

        assert "custom_format" in converter.custom_converters

    def test_convert_alpaca_format(self, converter, sample_formats):
        """Test Alpaca format conversion."""
        conversation = converter.convert_single(
            sample_formats["alpaca"],
            format_hint=FormatType.ALPACA
        )

        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2
        assert "Explain the concept" in conversation.messages[0].content
        assert "What is machine learning?" in conversation.messages[0].content
        assert conversation.messages[1].content == "Machine learning is a subset of AI."

    def test_convert_vicuna_format(self, converter, sample_formats):
        """Test Vicuna format conversion."""
        conversation = converter.convert_single(
            sample_formats["vicuna"],
            format_hint=FormatType.VICUNA
        )

        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2
        assert conversation.messages[0].role == "user"
        assert conversation.messages[1].role == "assistant"

    def test_convert_dolly_format(self, converter, sample_formats):
        """Test Dolly format conversion."""
        conversation = converter.convert_single(
            sample_formats["dolly"],
            format_hint=FormatType.DOLLY
        )

        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2
        assert "Answer the question" in conversation.messages[0].content
        assert "In the field of AI" in conversation.messages[0].content

    def test_convert_oasst_format(self, converter, sample_formats):
        """Test OpenAssistant format conversion."""
        conversation = converter.convert_single(
            sample_formats["oasst"],
            format_hint=FormatType.OASST
        )

        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 1
        assert conversation.messages[0].role == "user"  # prompter -> user
        assert conversation.messages[0].content == "What is the weather like?"

    def test_convert_xml_chat_format(self, converter, sample_formats):
        """Test XML chat format conversion."""
        conversation = converter.convert_single(
            sample_formats["xml_chat"],
            format_hint=FormatType.XML_CHAT
        )

        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2
        assert conversation.messages[0].role == "user"
        assert conversation.messages[1].role == "assistant"

    def test_convert_csv_format(self, converter, sample_formats):
        """Test CSV format conversion."""
        conversation = converter.convert_single(
            sample_formats["csv_format"],
            format_hint=FormatType.CSV_FORMAT
        )

        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2
        assert conversation.messages[0].content == "Hello"
        assert conversation.messages[1].content == "Hi there!"

    def test_convert_custom_json_format(self, converter, sample_formats):
        """Test custom JSON format conversion."""
        conversation = converter.convert_single(
            sample_formats["custom_json"],
            format_hint=FormatType.CUSTOM_JSON
        )

        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2
        assert conversation.messages[1].content == "This is custom format text"

    def test_format_consistency_validation(self, converter, sample_formats):
        """Test format consistency validation in batch processing."""
        # Mix different formats - should trigger warning
        data_items = [
            sample_formats["simple_messages"],  # Different format
            sample_formats["input_output"]      # Different format
        ]

        with patch.object(converter.logger, "warning") as mock_warning:
            conversations = converter.convert_batch(
                data_items,
                validate_consistency=True
            )

            # Should still convert successfully but log warning
            assert len(conversations) == 2
            mock_warning.assert_called_once()
            assert "Inconsistent formats detected" in mock_warning.call_args[0][0]


class TestFormatDetectionResult:
    """Test cases for FormatDetectionResult."""

    def test_initialization(self):
        """Test FormatDetectionResult initialization."""
        result = FormatDetectionResult(
            format_type=FormatType.SIMPLE_MESSAGES,
            confidence=0.9,
            indicators=["test indicator"],
            metadata={"test": "data"}
        )

        assert result.format_type == FormatType.SIMPLE_MESSAGES
        assert result.confidence == 0.9
        assert result.indicators == ["test indicator"]
        assert result.metadata == {"test": "data"}


class TestConversionRule:
    """Test cases for ConversionRule."""

    def test_initialization(self):
        """Test ConversionRule initialization."""
        def dummy_converter(data):
            return data

        rule = ConversionRule(
            name="test_rule",
            pattern={"test": "pattern"},
            converter=dummy_converter,
            priority=5,
            description="Test rule"
        )

        assert rule.name == "test_rule"
        assert rule.pattern == {"test": "pattern"}
        assert rule.converter == dummy_converter
        assert rule.priority == 5
        assert rule.description == "Test rule"
