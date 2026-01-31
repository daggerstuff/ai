"""
DataStandardizer orchestration class for unified format conversion.
Provides centralized coordination of all data standardization operations.
"""

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .conversation_schema import Conversation
from .logger import get_logger
from .standardizer import from_input_output_pair, from_simple_message_list

# from config import Config  # Comment out if config not available


@dataclass
class StandardizationResult:
    """Result of a standardization operation."""

    success: bool
    conversation: Conversation | None = None
    error: str | None = None
    processing_time: float = 0.0
    source_format: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StandardizationStats:
    """Statistics for standardization operations."""

    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    total_time: float = 0.0
    format_counts: dict[str, int] = field(default_factory=dict)
    error_counts: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return (
            self.successful / self.total_processed if self.total_processed > 0 else 0.0
        )

    @property
    def average_time(self) -> float:
        """Calculate average processing time."""
        return (
            self.total_time / self.total_processed if self.total_processed > 0 else 0.0
        )


class DataStandardizer:
    """
    Orchestration class for unified format conversion.

    Coordinates all data standardization operations with:
    - Multi-format support with automatic detection
    - Batch processing with concurrency control
    - Quality monitoring and validation
    - Performance optimization
    - Error handling and recovery
    """

    def __init__(
        self,
        config: dict | None = None,
        max_workers: int = 4,
        batch_size: int = 100,
        enable_monitoring: bool = True,
    ):
        """
        Initialize DataStandardizer.

        Args:
            config: Configuration object
            max_workers: Maximum number of worker threads
            batch_size: Size of processing batches
            enable_monitoring: Whether to enable monitoring
        """
        self.config = config or {}
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.enable_monitoring = enable_monitoring

        self.logger = get_logger(__name__)
        self.stats = StandardizationStats()

        # Format converters registry
        self.converters: dict[str, Callable] = {
            "simple_messages": self._convert_simple_messages,
            "input_output": self._convert_input_output,
            "huggingface_chat": self._convert_huggingface_chat,
            "openai_format": self._convert_openai_format,
            "custom_json": self._convert_custom_json,
        }

        # Quality validators
        self.validators: list[Callable] = []

        self.logger.info(
            f"DataStandardizer initialized with {max_workers} workers, batch size {batch_size}"
        )

    def register_converter(self, format_name: str, converter_func: Callable) -> None:
        """Register a custom format converter."""
        self.converters[format_name] = converter_func
        self.logger.info(f"Registered converter for format: {format_name}")

    def register_validator(self, validator_func: Callable) -> None:
        """Register a quality validator."""
        self.validators.append(validator_func)
        self.logger.info("Registered quality validator")

    def detect_format(self, data: Any) -> str:
        """
        Automatically detect the format of input data.

        Args:
            data: Input data to analyze

        Returns:
            Detected format name
        """
        if isinstance(data, dict):
            if "messages" in data and isinstance(data["messages"], list):
                return "simple_messages"
            if "input" in data and "output" in data:
                return "input_output"
            if "conversations" in data:
                return "huggingface_chat"
            if "role" in data and "content" in data:
                return "openai_format"
            return "custom_json"
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            if "role" in data[0] and "content" in data[0]:
                return "simple_messages"

        return "unknown"

    def standardize_single(
        self,
        data: Any,
        format_hint: str | None = None,
        source: str | None = None,
        conversation_id: str | None = None,
    ) -> StandardizationResult:
        """
        Standardize a single data item.

        Args:
            data: Input data to standardize
            format_hint: Optional format hint to skip detection
            source: Source identifier
            conversation_id: Optional conversation ID

        Returns:
            StandardizationResult object
        """
        result = self._standardize_single_internal(data, format_hint, source, conversation_id)

        # Update stats if monitoring is enabled
        if self.enable_monitoring:
            self._update_stats([result])

        return result

    def _standardize_single_internal(
        self,
        data: Any,
        format_hint: str | None = None,
        source: str | None = None,
        conversation_id: str | None = None,
    ) -> StandardizationResult:
        """
        Internal method to standardize a single data item without updating stats.

        Args:
            data: Input data to standardize
            format_hint: Optional format hint to skip detection
            source: Source identifier
            conversation_id: Optional conversation ID

        Returns:
            StandardizationResult object
        """
        start_time = time.time()

        try:
            # Detect format if not provided
            detected_format = format_hint or self.detect_format(data)

            # Get converter
            converter = self.converters.get(detected_format)
            if not converter:
                return StandardizationResult(
                    success=False,
                    error=f"No converter found for format: {detected_format}",
                    processing_time=time.time() - start_time,
                    source_format=detected_format,
                )

            # Convert data
            conversation = converter(
                data, source=source, conversation_id=conversation_id
            )

            # Validate if validators are registered
            for validator in self.validators:
                validation_result = validator(conversation)
                if not validation_result.get("valid", True):
                    return StandardizationResult(
                        success=False,
                        error=f"Validation failed: {validation_result.get('error', 'Unknown validation error')}",
                        processing_time=time.time() - start_time,
                        source_format=detected_format,
                    )

            processing_time = time.time() - start_time

            return StandardizationResult(
                success=True,
                conversation=conversation,
                processing_time=processing_time,
                source_format=detected_format,
                metadata={
                    "message_count": len(conversation.messages),
                    "total_chars": sum(
                        len(msg.content) for msg in conversation.messages
                    ),
                },
            )

        except Exception as e:
            return StandardizationResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
                source_format=format_hint or "unknown",
            )

    def standardize_batch(
        self,
        data_items: list[Any],
        format_hint: str | None = None,
        source: str | None = None,
    ) -> list[StandardizationResult]:
        """
        Standardize a batch of data items with parallel processing.

        Args:
            data_items: List of data items to standardize
            format_hint: Optional format hint for all items
            source: Source identifier

        Returns:
            List of StandardizationResult objects
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(
                    self._standardize_single_internal,
                    item,
                    format_hint,
                    source,
                    f"{source}_{i}" if source else None,
                ): i
                for i, item in enumerate(data_items)
            }

            # Collect results in order
            results = [None] * len(data_items)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()

        # Update statistics
        if self.enable_monitoring:
            self._update_stats(results)

        return results

    def standardize_file(
        self,
        file_path: str | Path,
        output_path: str | Path | None = None,
        format_hint: str | None = None,
    ) -> dict[str, Any]:
        """
        Standardize data from a file.

        Args:
            file_path: Path to input file
            output_path: Optional path to save standardized data
            format_hint: Optional format hint

        Returns:
            Processing summary
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.logger.info(f"Standardizing file: {file_path}")

        # Load data
        with open(file_path, encoding="utf-8") as f:
            if file_path.suffix == ".jsonl":
                data_items = [json.loads(line) for line in f if line.strip()]
            else:
                data_items = json.load(f)
                if not isinstance(data_items, list):
                    data_items = [data_items]

        # Process in batches
        all_results = []
        for i in range(0, len(data_items), self.batch_size):
            batch = data_items[i : i + self.batch_size]
            batch_results = self.standardize_batch(
                batch, format_hint=format_hint, source=str(file_path)
            )
            all_results.extend(batch_results)

            if self.enable_monitoring:
                self.logger.info(
                    f"Processed batch {i//self.batch_size + 1}/{(len(data_items) + self.batch_size - 1)//self.batch_size}"
                )

        # Save results if output path provided
        if output_path:
            self._save_results(all_results, output_path)

        # Generate summary
        successful_results = [r for r in all_results if r.success]
        summary = {
            "total_items": len(data_items),
            "successful": len(successful_results),
            "failed": len(all_results) - len(successful_results),
            "success_rate": (
                len(successful_results) / len(data_items) if data_items else 0
            ),
            "total_conversations": len(successful_results),
            "processing_time": sum(r.processing_time for r in all_results),
            "format_distribution": self._get_format_distribution(all_results),
        }

        self.logger.info(f"File standardization complete: {summary}")
        return summary

    def get_stats(self) -> StandardizationStats:
        """Get current standardization statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = StandardizationStats()
        self.logger.info("Statistics reset")

    # Private methods

    def _convert_simple_messages(
        self,
        data: Any,
        source: str | None = None,
        conversation_id: str | None = None,
    ) -> Conversation:
        """Convert simple messages format."""
        if isinstance(data, dict) and "messages" in data:
            messages = data["messages"]
        elif isinstance(data, list):
            messages = data
        else:
            raise ValueError("Invalid simple messages format")

        return from_simple_message_list(messages, conversation_id, source)

    def _convert_input_output(
        self,
        data: dict[str, Any],
        source: str | None = None,
        conversation_id: str | None = None,
    ) -> Conversation:
        """Convert input/output format."""
        return from_input_output_pair(
            data["input"],
            data["output"],
            conversation_id=conversation_id,
            source=source,
        )

    def _convert_huggingface_chat(
        self,
        data: dict[str, Any],
        source: str | None = None,
        conversation_id: str | None = None,
    ) -> Conversation:
        """Convert HuggingFace chat format."""
        conversations = data.get("conversations", [])
        if conversations:
            return from_simple_message_list(conversations[0], conversation_id, source)
        raise ValueError("No conversations found in HuggingFace format")

    def _convert_openai_format(
        self,
        data: dict[str, Any],
        source: str | None = None,
        conversation_id: str | None = None,
    ) -> Conversation:
        """Convert OpenAI format."""
        return from_simple_message_list([data], conversation_id, source)

    def _convert_custom_json(
        self,
        data: dict[str, Any],
        source: str | None = None,
        conversation_id: str | None = None,
    ) -> Conversation:
        """Convert custom JSON format."""
        # Try to extract messages from various possible structures
        if "text" in data:
            return from_input_output_pair(
                "", data["text"], conversation_id=conversation_id, source=source
            )
        if "content" in data:
            return from_input_output_pair(
                "", data["content"], conversation_id=conversation_id, source=source
            )
        raise ValueError("Unable to convert custom JSON format")

    def _update_stats(self, results: list[StandardizationResult]) -> None:
        """Update statistics with batch results."""
        for result in results:
            self.stats.total_processed += 1
            self.stats.total_time += result.processing_time

            if result.success:
                self.stats.successful += 1
            else:
                self.stats.failed += 1
                error_type = type(result.error).__name__ if result.error else "Unknown"
                self.stats.error_counts[error_type] = (
                    self.stats.error_counts.get(error_type, 0) + 1
                )

            if result.source_format:
                self.stats.format_counts[result.source_format] = (
                    self.stats.format_counts.get(result.source_format, 0) + 1
                )

    def _save_results(
        self, results: list[StandardizationResult], output_path: str | Path
    ) -> None:
        """Save standardization results to file."""
        output_path = Path(output_path)
        successful_conversations = [
            r.conversation for r in results if r.success and r.conversation
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            for conv in successful_conversations:
                json.dump(conv.to_dict(), f)
                f.write("\n")

        self.logger.info(
            f"Saved {len(successful_conversations)} conversations to {output_path}"
        )

    def _get_format_distribution(
        self, results: list[StandardizationResult]
    ) -> dict[str, int]:
        """Get format distribution from results."""
        distribution = {}
        for result in results:
            if result.source_format:
                distribution[result.source_format] = (
                    distribution.get(result.source_format, 0) + 1
                )
        return distribution
