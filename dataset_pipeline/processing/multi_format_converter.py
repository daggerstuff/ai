"""
Multi-format conversion pipeline with automatic format detection.
Provides advanced format detection and conversion capabilities for various data sources.
"""

import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from conversation_schema import Conversation
from logger import get_logger
from standardizer import from_input_output_pair, from_simple_message_list


class FormatType(Enum):
    """Enumeration of supported format types."""
    SIMPLE_MESSAGES = "simple_messages"
    INPUT_OUTPUT = "input_output"
    HUGGINGFACE_CHAT = "huggingface_chat"
    OPENAI_FORMAT = "openai_format"
    CUSTOM_JSON = "custom_json"
    SHAREGPT = "sharegpt"
    ALPACA = "alpaca"
    VICUNA = "vicuna"
    DOLLY = "dolly"
    OASST = "oasst"
    XML_CHAT = "xml_chat"
    CSV_FORMAT = "csv_format"
    UNKNOWN = "unknown"


@dataclass
class FormatDetectionResult:
    """Result of format detection analysis."""
    format_type: FormatType
    confidence: float
    indicators: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversionRule:
    """Rule for converting specific format patterns."""
    name: str
    pattern: dict[str, Any]
    converter: Callable
    priority: int = 0
    description: str = ""


class MultiFormatConverter:
    """
    Advanced multi-format conversion pipeline with automatic format detection.

    Features:
    - Intelligent format detection with confidence scoring
    - Support for 12+ common conversation formats
    - Extensible rule-based conversion system
    - Batch processing with format consistency validation
    - Custom format registration
    """

    def __init__(self):
        """Initialize MultiFormatConverter."""
        self.logger = get_logger(__name__)
        self.conversion_rules: list[ConversionRule] = []
        self.format_detectors: dict[FormatType, Callable] = {}
        self.custom_converters: dict[str, Callable] = {}

        # Initialize built-in detectors and converters
        self._initialize_detectors()
        self._initialize_converters()

        self.logger.info("MultiFormatConverter initialized with built-in format support")

    def detect_format(self, data: Any, context: dict[str, Any] | None = None) -> FormatDetectionResult:
        """
        Automatically detect the format of input data with confidence scoring.

        Args:
            data: Input data to analyze
            context: Optional context information (file extension, source, etc.)

        Returns:
            FormatDetectionResult with detected format and confidence
        """
        detection_results = []

        # Run all format detectors
        for format_type, detector in self.format_detectors.items():
            try:
                confidence, indicators, metadata = detector(data, context)
                if confidence > 0:
                    detection_results.append(FormatDetectionResult(
                        format_type=format_type,
                        confidence=confidence,
                        indicators=indicators,
                        metadata=metadata
                    ))
            except Exception as e:
                self.logger.debug(f"Detector {format_type} failed: {e}")

        # Sort by confidence and return best match
        if detection_results:
            detection_results.sort(key=lambda x: x.confidence, reverse=True)
            return detection_results[0]

        return FormatDetectionResult(
            format_type=FormatType.UNKNOWN,
            confidence=0.0,
            indicators=["No format patterns matched"]
        )

    def convert_single(
        self,
        data: Any,
        format_hint: FormatType | None = None,
        source: str | None = None,
        conversation_id: str | None = None
    ) -> Conversation:
        """
        Convert a single data item to standard conversation format.

        Args:
            data: Input data to convert
            format_hint: Optional format hint to skip detection
            source: Source identifier
            conversation_id: Optional conversation ID

        Returns:
            Converted Conversation object
        """
        # Detect format if not provided
        if format_hint:
            detected_format = format_hint
        else:
            detection_result = self.detect_format(data)
            detected_format = detection_result.format_type

            if detected_format == FormatType.UNKNOWN:
                raise ValueError(f"Unable to detect format for data: {type(data)}")

        # Get converter for detected format
        converter_name = f"_convert_{detected_format.value}"
        converter = getattr(self, converter_name, None)

        if not converter:
            raise ValueError(f"No converter available for format: {detected_format}")

        # Convert data
        return converter(data, source=source, conversation_id=conversation_id)

    def convert_batch(
        self,
        data_items: list[Any],
        format_hint: FormatType | None = None,
        source: str | None = None,
        validate_consistency: bool = True
    ) -> list[Conversation]:
        """
        Convert a batch of data items with format consistency validation.

        Args:
            data_items: List of data items to convert
            format_hint: Optional format hint for all items
            source: Source identifier
            validate_consistency: Whether to validate format consistency

        Returns:
            List of converted Conversation objects
        """
        conversations = []
        detected_formats = []

        for i, item in enumerate(data_items):
            try:
                # Convert item
                conversation = self.convert_single(
                    item,
                    format_hint=format_hint,
                    source=source,
                    conversation_id=f"{source}_{i}" if source else None
                )
                conversations.append(conversation)

                # Track format for consistency validation
                if not format_hint:
                    detection_result = self.detect_format(item)
                    detected_formats.append(detection_result.format_type)

            except Exception as e:
                self.logger.warning(f"Failed to convert item {i}: {e}")
                continue

        # Validate format consistency if requested
        if validate_consistency and detected_formats:
            self._validate_format_consistency(detected_formats)

        return conversations

    def register_custom_converter(
        self,
        format_name: str,
        detector: Callable,
        converter: Callable,
        description: str = ""
    ) -> None:
        """
        Register a custom format converter.

        Args:
            format_name: Name of the custom format
            detector: Function to detect this format
            converter: Function to convert this format
            description: Optional description
        """
        # Register detector
        custom_format = FormatType(format_name) if format_name not in [f.value for f in FormatType] else FormatType(format_name)
        self.format_detectors[custom_format] = detector

        # Register converter
        self.custom_converters[format_name] = converter

        self.logger.info(f"Registered custom converter: {format_name}")

    # Built-in format detectors

    def _initialize_detectors(self) -> None:
        """Initialize built-in format detectors."""
        self.format_detectors = {
            FormatType.SIMPLE_MESSAGES: self._detect_simple_messages,
            FormatType.INPUT_OUTPUT: self._detect_input_output,
            FormatType.HUGGINGFACE_CHAT: self._detect_huggingface_chat,
            FormatType.OPENAI_FORMAT: self._detect_openai_format,
            FormatType.SHAREGPT: self._detect_sharegpt,
            FormatType.ALPACA: self._detect_alpaca,
            FormatType.VICUNA: self._detect_vicuna,
            FormatType.DOLLY: self._detect_dolly,
            FormatType.OASST: self._detect_oasst,
            FormatType.XML_CHAT: self._detect_xml_chat,
            FormatType.CSV_FORMAT: self._detect_csv_format,
            FormatType.CUSTOM_JSON: self._detect_custom_json
        }

    def _initialize_converters(self) -> None:
        """Initialize built-in converters."""
        # Converters are implemented as methods with naming convention _convert_{format_name}

    def _detect_simple_messages(self, data: Any, context: dict | None = None) -> tuple[float, list[str], dict]:
        """Detect simple messages format."""
        indicators = []
        confidence = 0.0

        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            if isinstance(first_item, dict):
                if "role" in first_item and "content" in first_item:
                    confidence = 0.9
                    indicators.append("Has role and content fields")

                    # Check for common roles
                    roles = [item.get("role", "") for item in data if isinstance(item, dict)]
                    if any(role in ["user", "assistant", "system"] for role in roles):
                        confidence = 0.95
                        indicators.append("Contains standard chat roles")

        return confidence, indicators, {}

    def _detect_input_output(self, data: Any, context: dict | None = None) -> tuple[float, list[str], dict]:
        """Detect input/output format."""
        indicators = []
        confidence = 0.0

        if isinstance(data, dict) and "input" in data and "output" in data:
            confidence = 0.9
            indicators.append("Has input and output fields")

            if isinstance(data["input"], str) and isinstance(data["output"], str):
                confidence = 0.95
                indicators.append("Input and output are strings")

        return confidence, indicators, {}

    def _detect_huggingface_chat(self, data: Any, context: dict | None = None) -> tuple[float, list[str], dict]:
        """Detect HuggingFace chat format."""
        indicators = []
        confidence = 0.0

        if isinstance(data, dict) and "conversations" in data:
            confidence = 0.8
            indicators.append("Has conversations field")

            conversations = data["conversations"]
            if isinstance(conversations, list) and len(conversations) > 0:
                if isinstance(conversations[0], list):
                    confidence = 0.9
                    indicators.append("Conversations is list of lists")

        return confidence, indicators, {}

    def _detect_openai_format(self, data: Any, context: dict | None = None) -> tuple[float, list[str], dict]:
        """Detect OpenAI format."""
        indicators = []
        confidence = 0.0

        if isinstance(data, dict) and "role" in data and "content" in data:
            confidence = 0.7
            indicators.append("Has role and content fields")

            if data.get("role") in ["user", "assistant", "system"]:
                confidence = 0.8
                indicators.append("Has standard OpenAI role")

        return confidence, indicators, {}

    def _detect_sharegpt(self, data: Any, context: dict | None = None) -> tuple[float, list[str], dict]:
        """Detect ShareGPT format."""
        indicators = []
        confidence = 0.0

        if isinstance(data, dict) and ("conversations" in data or "items" in data):
            confidence = 0.6
            indicators.append("Has conversations or items field")

            # Look for ShareGPT-specific fields
            if "id" in data and "title" in data:
                confidence = 0.8
                indicators.append("Has ShareGPT-style id and title")

        return confidence, indicators, {}

    def _detect_alpaca(self, data: Any, context: dict | None = None) -> tuple[float, list[str], dict]:
        """Detect Alpaca format."""
        indicators = []
        confidence = 0.0

        if isinstance(data, dict):
            alpaca_fields = ["instruction", "input", "output"]
            if all(field in data for field in alpaca_fields):
                confidence = 0.9
                indicators.append("Has all Alpaca fields (instruction, input, output)")
            elif "instruction" in data and "output" in data:
                confidence = 0.8
                indicators.append("Has instruction and output fields")

        return confidence, indicators, {}

    def _detect_vicuna(self, data: Any, context: dict | None = None) -> tuple[float, list[str], dict]:
        """Detect Vicuna format."""
        indicators = []
        confidence = 0.0

        if isinstance(data, dict) and "conversations" in data:
            conversations = data["conversations"]
            if isinstance(conversations, list):
                # Check for Vicuna-style conversation structure
                for conv in conversations:
                    if isinstance(conv, dict) and "from" in conv and "value" in conv:
                        confidence = 0.8
                        indicators.append("Has Vicuna-style from/value structure")
                        break

        return confidence, indicators, {}

    def _detect_dolly(self, data: Any, context: dict | None = None) -> tuple[float, list[str], dict]:
        """Detect Dolly format."""
        indicators = []
        confidence = 0.0

        if isinstance(data, dict):
            dolly_fields = ["instruction", "context", "response"]
            if all(field in data for field in dolly_fields):
                confidence = 0.9
                indicators.append("Has all Dolly fields (instruction, context, response)")
            elif "instruction" in data and "response" in data:
                confidence = 0.7
                indicators.append("Has instruction and response fields")

        return confidence, indicators, {}

    def _detect_oasst(self, data: Any, context: dict | None = None) -> tuple[float, list[str], dict]:
        """Detect OpenAssistant format."""
        indicators = []
        confidence = 0.0

        if isinstance(data, dict) and "text" in data and "role" in data:
            confidence = 0.7
            indicators.append("Has text and role fields")

            if "message_id" in data and "parent_id" in data:
                confidence = 0.9
                indicators.append("Has OASST-style message and parent IDs")

        return confidence, indicators, {}

    def _detect_xml_chat(self, data: Any, context: dict | None = None) -> tuple[float, list[str], dict]:
        """Detect XML chat format."""
        indicators = []
        confidence = 0.0

        if isinstance(data, str):
            if data.strip().startswith("<") and data.strip().endswith(">"):
                try:
                    ET.fromstring(data)
                    confidence = 0.8
                    indicators.append("Valid XML structure")

                    if "<message>" in data or "<conversation>" in data:
                        confidence = 0.9
                        indicators.append("Contains chat-related XML tags")
                except ET.ParseError:
                    pass

        return confidence, indicators, {}

    def _detect_csv_format(self, data: Any, context: dict | None = None) -> tuple[float, list[str], dict]:
        """Detect CSV format."""
        indicators = []
        confidence = 0.0

        if isinstance(data, str):
            lines = data.strip().split("\n")
            if len(lines) > 1:
                # Check for CSV-like structure
                first_line = lines[0]
                if "," in first_line:
                    headers = [h.strip().lower() for h in first_line.split(",")]
                    chat_headers = ["user", "assistant", "input", "output", "question", "answer"]

                    if any(header in chat_headers for header in headers):
                        confidence = 0.8
                        indicators.append("Has chat-related CSV headers")

        return confidence, indicators, {}

    def _detect_custom_json(self, data: Any, context: dict | None = None) -> tuple[float, list[str], dict]:
        """Detect custom JSON format (fallback)."""
        indicators = []
        confidence = 0.0

        if isinstance(data, dict):
            # This is a fallback detector with low confidence
            confidence = 0.1
            indicators.append("Generic JSON object")

            # Look for text-like fields
            text_fields = ["text", "content", "message", "response", "answer"]
            if any(field in data for field in text_fields):
                confidence = 0.3
                indicators.append("Contains text-like fields")

        return confidence, indicators, {}

    # Built-in format converters

    def _convert_simple_messages(self, data: Any, source: str | None = None, conversation_id: str | None = None) -> Conversation:
        """Convert simple messages format."""
        if isinstance(data, dict) and "messages" in data:
            messages = data["messages"]
        elif isinstance(data, list):
            messages = data
        else:
            raise ValueError("Invalid simple messages format")

        return from_simple_message_list(messages, conversation_id, source)

    def _convert_input_output(self, data: dict[str, Any], source: str | None = None, conversation_id: str | None = None) -> Conversation:
        """Convert input/output format."""
        return from_input_output_pair(
            data["input"],
            data["output"],
            conversation_id=conversation_id,
            source=source
        )

    def _convert_huggingface_chat(self, data: dict[str, Any], source: str | None = None, conversation_id: str | None = None) -> Conversation:
        """Convert HuggingFace chat format."""
        conversations = data.get("conversations", [])
        if conversations and isinstance(conversations[0], list):
            return from_simple_message_list(conversations[0], conversation_id, source)
        raise ValueError("Invalid HuggingFace chat format")

    def _convert_openai_format(self, data: dict[str, Any], source: str | None = None, conversation_id: str | None = None) -> Conversation:
        """Convert OpenAI format."""
        return from_simple_message_list([data], conversation_id, source)

    def _convert_sharegpt(self, data: dict[str, Any], source: str | None = None, conversation_id: str | None = None) -> Conversation:
        """Convert ShareGPT format."""
        conversations = data.get("conversations", data.get("items", []))
        if conversations:
            return from_simple_message_list(conversations, conversation_id, source)
        raise ValueError("Invalid ShareGPT format")

    def _convert_alpaca(self, data: dict[str, Any], source: str | None = None, conversation_id: str | None = None) -> Conversation:
        """Convert Alpaca format."""
        instruction = data.get("instruction", "")
        input_text = data.get("input", "")
        output_text = data.get("output", "")

        # Combine instruction and input
        user_message = f"{instruction}\n{input_text}".strip()

        return from_input_output_pair(user_message, output_text, conversation_id=conversation_id, source=source)

    def _convert_vicuna(self, data: dict[str, Any], source: str | None = None, conversation_id: str | None = None) -> Conversation:
        """Convert Vicuna format."""
        conversations = data.get("conversations", [])
        messages = []

        for conv in conversations:
            role = "user" if conv.get("from") == "human" else "assistant"
            messages.append({
                "role": role,
                "content": conv.get("value", "")
            })

        return from_simple_message_list(messages, conversation_id, source)

    def _convert_dolly(self, data: dict[str, Any], source: str | None = None, conversation_id: str | None = None) -> Conversation:
        """Convert Dolly format."""
        instruction = data.get("instruction", "")
        context = data.get("context", "")
        response = data.get("response", "")

        # Combine instruction and context
        user_message = f"{instruction}\n{context}".strip()

        return from_input_output_pair(user_message, response, conversation_id=conversation_id, source=source)

    def _convert_oasst(self, data: dict[str, Any], source: str | None = None, conversation_id: str | None = None) -> Conversation:
        """Convert OpenAssistant format."""
        role = data.get("role", "user")
        text = data.get("text", "")

        # Convert OASST roles to standard roles
        if role == "prompter":
            role = "user"
        elif role == "assistant":
            role = "assistant"

        message = {"role": role, "content": text}
        return from_simple_message_list([message], conversation_id, source)

    def _convert_xml_chat(self, data: str, source: str | None = None, conversation_id: str | None = None) -> Conversation:
        """Convert XML chat format."""
        try:
            root = ET.fromstring(data)
            messages = []

            for message_elem in root.findall(".//message"):
                role = message_elem.get("role", "user")
                content = message_elem.text or ""
                messages.append({"role": role, "content": content})

            return from_simple_message_list(messages, conversation_id, source)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format: {e}")

    def _convert_csv_format(self, data: str, source: str | None = None, conversation_id: str | None = None) -> Conversation:
        """Convert CSV format."""
        lines = data.strip().split("\n")
        if len(lines) < 2:
            raise ValueError("CSV must have at least header and one data row")

        headers = [h.strip().lower() for h in lines[0].split(",")]
        data_row = lines[1].split(",")

        # Map CSV columns to conversation format
        user_content = ""
        assistant_content = ""

        for i, header in enumerate(headers):
            if i < len(data_row):
                value = data_row[i].strip()
                if header in ["user", "input", "question"]:
                    user_content = value
                elif header in ["assistant", "output", "answer", "response"]:
                    assistant_content = value

        return from_input_output_pair(user_content, assistant_content, conversation_id=conversation_id, source=source)

    def _convert_custom_json(self, data: dict[str, Any], source: str | None = None, conversation_id: str | None = None) -> Conversation:
        """Convert custom JSON format."""
        # Try to extract content from various possible fields
        content = ""
        for field in ["text", "content", "message", "response", "answer"]:
            if field in data:
                content = data[field]
                break

        if not content:
            raise ValueError("Unable to extract content from custom JSON format")

        return from_input_output_pair("", content, conversation_id=conversation_id, source=source)

    # Utility methods

    def _validate_format_consistency(self, detected_formats: list[FormatType]) -> None:
        """Validate that all items in a batch have consistent formats."""
        if len(set(detected_formats)) > 1:
            format_counts = {}
            for fmt in detected_formats:
                format_counts[fmt] = format_counts.get(fmt, 0) + 1

            self.logger.warning(f"Inconsistent formats detected in batch: {format_counts}")
            # Could raise an exception here if strict consistency is required
