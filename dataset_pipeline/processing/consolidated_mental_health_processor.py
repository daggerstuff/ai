"""
Consolidated Mental Health Dataset Processor for Task 5.1
Processes the existing 86MB merged mental health dataset (43,683 conversations).
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from emotional_authenticity_assessment import EmotionalAuthenticityAssessor
from logger import get_logger
from quality_filter import QualityFilter
from standardizer import from_input_output_pair
from therapeutic_accuracy_assessment import TherapeuticAccuracyAssessor

logger = get_logger("dataset_pipeline.consolidated_mental_health_processor")


@dataclass
class ConsolidatedProcessingConfig:
    """Configuration for consolidated mental health dataset processing."""

    input_file: str = "ai/merged_mental_health_dataset.jsonl"
    output_dir: str = "data/processed/mental_health"
    target_conversations: int = 20000  # 20% of 100K target
    quality_threshold: float = 0.75
    therapeutic_accuracy_threshold: float = 0.8
    emotional_authenticity_threshold: float = 0.75
    batch_size: int = 1000


class ConsolidatedMentalHealthProcessor:
    """
    Processes the existing consolidated mental health dataset (86MB JSONL file).
    Handles quality assessment, therapeutic validation, and format standardization.
    """

    def __init__(self, config: ConsolidatedProcessingConfig | None = None):
        self.config = config or ConsolidatedProcessingConfig()
        self.quality_filter = QualityFilter()
        self.emotional_assessor = EmotionalAuthenticityAssessor()
        self.therapeutic_assessor = TherapeuticAccuracyAssessor()

        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)

        self.processing_stats = {
            "total_processed": 0,
            "total_accepted": 0,
            "quality_filtered": 0,
            "therapeutic_filtered": 0,
            "emotional_filtered": 0,
            "format_errors": 0,
            "processing_errors": 0,
        }

    def process_consolidated_dataset(self) -> dict[str, Any]:
        """
        Process the consolidated mental health dataset.

        Returns:
            Dict containing processing results and statistics
        """
        logger.info(
            f"Starting consolidated mental health dataset processing from {self.config.input_file}"
        )
        logger.info(f"Target conversations: {self.config.target_conversations}")
        logger.info(f"Quality threshold: {self.config.quality_threshold}")
        logger.info(
            f"Therapeutic accuracy threshold: {self.config.therapeutic_accuracy_threshold}"
        )
        start_time = datetime.now(tz=datetime.timezone.utc)

        if not os.path.exists(self.config.input_file):
            raise FileNotFoundError(
                f"Consolidated dataset not found: {self.config.input_file}"
            )

        processed_conversations = []

        try:
            with open(self.config.input_file, encoding="utf-8") as f:
                batch = []

                for line_num, line in enumerate(f, 1):
                    try:
                        # Parse JSON line
                        item = json.loads(line.strip())
                        batch.append((line_num, item))

                        # Process in batches
                        if len(batch) >= self.config.batch_size:
                            batch_results = self._process_batch(batch)
                            processed_conversations.extend(batch_results)
                            batch = []

                            # Check if we've reached target
                            if (
                                len(processed_conversations)
                                >= self.config.target_conversations
                            ):
                                logger.info(
                                    "Reached target of %d conversations",
                                    self.config.target_conversations,
                                )
                                break

                        # Progress logging
                        if line_num % 5000 == 0:
                            logger.info(
                                "Processed %d lines, accepted %d conversations",
                                line_num,
                                len(processed_conversations),
                            )

                    except json.JSONDecodeError as e:
                        logger.warning("JSON decode error at line %d: %s", line_num, e)
                        self.processing_stats["format_errors"] += 1
                        continue
                    except Exception as e:
                        logger.warning("Error processing line %d: %s", line_num, e)
                        self.processing_stats["processing_errors"] += 1
                        continue

                # Process remaining batch
                if batch:
                    batch_results = self._process_batch(batch)
                    processed_conversations.extend(batch_results)

        except Exception as e:
            logger.error("Error reading consolidated dataset: %s", e)
            raise

        # Save processed conversations
        output_path = os.path.join(
            self.config.output_dir, "consolidated_mental_health_conversations.jsonl"
        )
        self._save_conversations(processed_conversations, output_path)

        # Generate processing report
        processing_time = (
            datetime.now(tz=datetime.timezone.utc) - start_time
        ).total_seconds()
        processing_report = self._generate_processing_report(
            processed_conversations, processing_time
        )

        # Save processing report
        report_path = os.path.join(
            self.config.output_dir, "consolidated_processing_report.json"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(processing_report, f, indent=2)

        logger.info(
            "Consolidated processing completed. Processed %d conversations",
            len(processed_conversations),
        )
        return processing_report

    def _process_batch(
        self, batch: list[tuple[int, dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Process a batch of conversations."""
        batch_results = []

        for line_num, item in batch:
            try:
                # Standardize conversation format
                standardized = self._standardize_consolidated_conversation(
                    item, line_num
                )
                if not standardized:
                    continue

                # Quality assessment
                if not self._assess_quality(standardized):
                    self.processing_stats["quality_filtered"] += 1
                    continue

                # Therapeutic accuracy assessment
                if not self._assess_therapeutic_accuracy(standardized):
                    self.processing_stats["therapeutic_filtered"] += 1
                    continue

                # Emotional authenticity assessment
                if not self._assess_emotional_authenticity(standardized):
                    self.processing_stats["emotional_filtered"] += 1
                    continue

                # Add processing metadata
                standardized["metadata"].update(
                    {
                        "source_dataset": "consolidated_mental_health",
                        "source_file": self.config.input_file,
                        "original_line_number": line_num,
                        "processing_timestamp": datetime.now(
                            tz=datetime.timezone.utc
                        ).isoformat(),
                    }
                )

                batch_results.append(standardized)
                self.processing_stats["total_accepted"] += 1

            except Exception as e:
                # Broad exception caught for batch item processing; log full details for debugging
                logger.warning(
                    "Error processing item at line %d: %s", line_num, e, exc_info=True
                )
                self.processing_stats["processing_errors"] += 1
                continue

            self.processing_stats["total_processed"] += 1

        return batch_results

    def _standardize_consolidated_conversation(
        self, item: dict[str, Any], line_num: int
    ) -> dict[str, Any] | None:
        """
        Standardize consolidated conversation format.

        Args:
            item: Raw data item from consolidated dataset
            line_num: Line number for tracking

        Returns:
            Standardized conversation or None if conversion fails
        """
        try:
            # Extract prompt and response
            prompt = item.get("prompt", "")
            response = item.get("response", "")

            if not prompt or not response:
                logger.warning("Missing prompt or response at line %d", line_num)
                return None

            # Clean the prompt (remove system instructions)
            client_content = self._extract_client_content(prompt)

            # Convert to standard conversation format
            conversation = from_input_output_pair(
                client_content,
                response,
                input_role="client",
                output_role="therapist",
                source="consolidated_mental_health",
            )

            return {
                "conversation": [
                    {"role": msg.role, "content": msg.content}
                    for msg in conversation.messages
                ],
                "metadata": {
                    "category": "mental_health",
                    "subcategory": "consolidated_counseling",
                    "therapeutic_approach": self._determine_therapeutic_approach(
                        prompt, response
                    ),
                    "emotional_intensity": self._estimate_emotional_intensity(
                        prompt, response
                    ),
                    "conversation_length": len(conversation.messages),
                    "client_content_length": len(client_content),
                    "therapist_response_length": len(response),
                },
            }
        except Exception as e:
            # Broad exception caught for standardization; log full details for debugging
            logger.warning(
                "Failed to standardize conversation at line %d: %s",
                line_num,
                e,
                exc_info=True,
            )
            return None

    def _extract_client_content(self, prompt: str) -> str:
        """Extract actual client content from prompt, removing system instructions."""
        # Common system instruction patterns to remove
        system_patterns = [
            "You are a helpful mental health counselling assistant",
            "please answer the mental health questions based on the patient's description",
            "The assistant gives helpful, comprehensive, and appropriate answers to the user's questions",
            "You are a mental health counselor",
            "You are a therapist",
        ]

        # Remove system instructions
        client_content = prompt
        for pattern in system_patterns:
            if pattern in client_content:
                # Find the end of system instructions and extract client content
                parts = client_content.split(pattern)
                if len(parts) > 1:
                    client_content = parts[-1].strip()
                    # Remove any remaining instruction markers
                    client_content = client_content.lstrip(". ").strip()

        return client_content

    def _determine_therapeutic_approach(self, prompt: str, response: str) -> str:
        """Determine therapeutic approach from conversation content."""
        content = (prompt + " " + response).lower()

        # Approach detection based on keywords
        if any(
            term in content
            for term in ["cognitive", "thought", "thinking", "belief", "cbt"]
        ):
            return "cognitive_behavioral"
        if any(
            term in content
            for term in ["mindful", "present", "awareness", "meditation"]
        ):
            return "mindfulness_based"
        if any(
            term in content
            for term in ["behavior", "action", "activity", "goal", "exposure"]
        ):
            return "behavioral"
        if any(
            term in content for term in ["emotion", "feeling", "empathy", "validation"]
        ):
            return "emotion_focused"
        if any(
            term in content
            for term in ["relationship", "family", "communication", "interpersonal"]
        ):
            return "interpersonal"
        return "integrative"

    def _estimate_emotional_intensity(self, prompt: str, response: str) -> float:
        """Estimate emotional intensity of conversation (0.0 to 1.0)."""
        content = (prompt + " " + response).lower()

        # High intensity indicators
        high_intensity_terms = [
            "crisis",
            "suicide",
            "emergency",
            "severe",
            "panic",
            "trauma",
            "abuse",
        ]
        medium_intensity_terms = [
            "anxious",
            "depressed",
            "worried",
            "stressed",
            "upset",
            "hurt",
            "angry",
        ]
        low_intensity_terms = [
            "concerned",
            "uncertain",
            "confused",
            "tired",
            "frustrated",
        ]

        high_count = sum(1 for term in high_intensity_terms if term in content)
        medium_count = sum(1 for term in medium_intensity_terms if term in content)
        low_count = sum(1 for term in low_intensity_terms if term in content)

        if high_count > 0:
            return min(0.8 + (high_count * 0.1), 1.0)
        if medium_count > 0:
            return min(0.5 + (medium_count * 0.05), 0.8)
        if low_count > 0:
            return min(0.3 + (low_count * 0.05), 0.5)
        return 0.4  # Default moderate intensity

    self, conversation: dict[str, Any]) -> bool:
        """Assess conversation quality using quality filter."""
        try:
            # Convert dict to Conversation object for assessment
            conv_obj = self._dict_to_conversation(conversation)
            filter_result = self.quality_filter.filter_conversation(conv_obj)
            logger.info(
                "QualityFilter: conversation_id=%s, overall_score=%.3f, passed=%s",
                getattr(conv_obj, "id", "unknown"),
                filter_result.overall_score,
                filter_result.passed,
            )
            return filter_result.overall_score >= self.config.quality_threshold
        except Exception as e:
            logger.warning("Quality assessment failed: %s", e, exc_info=True)
            return False

    def _assess_therapeutic_accuracy(self, conversation: dict[str, Any]) -> bool:
        """Assess therapeutic accuracy of conversation."""
        try:
            # Convert dict to Conversation object for assessment
            conv_obj = self._dict_to_conversation(conversation)
            accuracy_metrics = self.therapeutic_assessor.assess_therapeutic_accuracy(
                conv_obj
            )
            return (
                accuracy_metrics.overall_score
                >= self.config.therapeutic_accuracy_threshold
            )
        except Exception as e:
            # Broad exception caught for therapeutic accuracy; log full details for debugging
            logger.warning(
                "Therapeutic accuracy assessment failed: %s", e, exc_info=True
            )
            return False

    def _assess_emotional_authenticity(self, conversation: dict[str, Any]) -> bool:
        """Assess emotional authenticity of conversation."""
        try:
            # Convert dict to Conversation object for assessment
            conv_obj = self._dict_to_conversation(conversation)
            authenticity_metrics = (
                self.emotional_assessor.assess_emotional_authenticity(conv_obj)
            )
            return (
                authenticity_metrics.overall_score
                >= self.config.emotional_authenticity_threshold
            )
        except Exception as e:
            # Broad exception caught for emotional authenticity; log full details for debugging
            logger.warning(
                "Emotional authenticity assessment failed: %s", e, exc_info=True
            )
            return False

    def _dict_to_conversation(self, conversation_dict: dict[str, Any]):
        """Convert conversation dict to Conversation object for assessments."""
        from conversation_schema import Conversation, Message

        messages = []
        for msg_dict in conversation_dict.get("conversation", []):
            messages.append(
                Message(
                    role=msg_dict["role"],
                    content=msg_dict["content"],
                    timestamp=None,
                    meta={},
                )
            )

        return Conversation(
            id=None,
            messages=messages,
            context=conversation_dict.get("metadata", {}),
            source="consolidated_mental_health",
            created_at=None,
            meta={},
        )

    def _save_conversations(
        self, conversations: list[dict[str, Any]], output_path: str
    ) -> None:
        """Save conversations to JSONL file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for conversation in conversations:
                f.write(json.dumps(conversation) + "\n")

        logger.info(
            "Saved %d consolidated conversations to %s", len(conversations), output_path
        )

    def _generate_processing_report(
self, conversations: list[dict[str, Any]], processing_time: float) -> dict[str, Any]:
        """Generate comprehensive processing report."""
        return {
            "processing_summary": {
                "source_file": self.config.input_file,
                "total_conversations": len(conversations),
                "target_conversations": self.config.target_conversations,
                "completion_percentage": (
                    len(conversations) / self.config.target_conversations
                )
                * 100,
                "processing_time_seconds": processing_time,
                "average_processing_rate": (
                    len(conversations) / processing_time if processing_time > 0 else 0
                ),
            },
            "quality_metrics": {
                "total_processed": self.processing_stats["total_processed"],
                "total_accepted": self.processing_stats["total_accepted"],
                "acceptance_rate": (
                    self.processing_stats["total_accepted"]
                    / max(self.processing_stats["total_processed"], 1)
                )
                * 100,
                "quality_filtered": self.processing_stats["quality_filtered"],
                "therapeutic_filtered": self.processing_stats["therapeutic_filtered"],
                "emotional_filtered": self.processing_stats["emotional_filtered"],
                "format_errors": self.processing_stats["format_errors"],
                "processing_errors": self.processing_stats["processing_errors"],
            },
            "conversation_analysis": {
                "therapeutic_approaches": self._analyze_therapeutic_approaches(
                    conversations
                ),
                "emotional_intensity_distribution": self._analyze_emotional_intensity(
                    conversations
                ),
                "conversation_length_stats": self._analyze_conversation_lengths(
                    conversations
                ),
                "content_length_stats": self._analyze_content_lengths(conversations),
            },
            "configuration": {
                "quality_threshold": self.config.quality_threshold,
                "therapeutic_accuracy_threshold": self.config.therapeutic_accuracy_threshold,
                "emotional_authenticity_threshold": self.config.emotional_authenticity_threshold,
                "batch_size": self.config.batch_size,
            },
            "timestamp": datetime.now(tz=datetime.timezone.utc).isoformat(),
            "task": "5.1: Process existing consolidated mental health dataset (86MB JSONL)",
        }

    def _analyze_therapeutic_approaches(
        self, conversations: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Analyze distribution of therapeutic approaches."""
        approaches = {}
        for conv in conversations:
            approach = conv.get("metadata", {}).get("therapeutic_approach", "unknown")
            approaches[approach] = approaches.get(approach, 0) + 1
        return approaches

    def _analyze_emotional_intensity(
        self, conversations: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Analyze emotional intensity distribution."""
        intensity_ranges = {"low": 0, "medium": 0, "high": 0}
        for conv in conversations:
            intensity = conv.get("metadata", {}).get("emotional_intensity", 0.0)
            if intensity < 0.4:
                intensity_ranges["low"] += 1
            elif intensity < 0.7:
                intensity_ranges["medium"] += 1
            else:
                intensity_ranges["high"] += 1
        return intensity_ranges

    def _analyze_conversation_lengths(
        self, conversations: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Analyze conversation length statistics."""
        lengths = [
            conv.get("metadata", {}).get("conversation_length", 0)
            for conv in conversations
        ]
        if not lengths:
            return {"min": 0, "max": 0, "average": 0}

        return {
            "min": min(lengths),
            "max": max(lengths),
            "average": sum(lengths) / len(lengths),
        }

    def _analyze_content_lengths(
        self, conversations: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """Analyze content length statistics."""
        client_lengths = [
            conv.get("metadata", {}).get("client_content_length", 0)
            for conv in conversations
        ]
        therapist_lengths = [
            conv.get("metadata", {}).get("therapist_response_length", 0)
            for conv in conversations
        ]

        def get_stats(lengths):
            if not lengths:
                return {"min": 0, "max": 0, "average": 0}
            return {
                "min": min(lengths),
                "max": max(lengths),
                "average": sum(lengths) / len(lengths),
            }

        return {
            "client_content": get_stats(client_lengths),
            "therapist_response": get_stats(therapist_lengths),
        }
