"""
Voice-to-Conversation Format Converter for Voice Training Data.

This module converts voice-derived data (transcriptions + personality markers)
into standard conversation format compatible with the dataset pipeline,
enabling seamless integration of voice training data.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from conversation_schema import Conversation, Message
from logger import setup_logger
from personality_extractor import PersonalityExtractor
from voice_transcriber import TranscriptionResult, TranscriptionSegment
from voice_types import PersonalityProfile


@dataclass
class VoiceConversationMetadata:
    """Metadata for voice-derived conversations."""

    original_audio_file: str
    transcription_model: str
    transcription_confidence: float
    personality_confidence: float
    speaker_count: int
    total_duration: float
    processing_timestamp: str
    voice_quality_score: float
    personality_profile: dict | None = None
    transcription_segments: list[dict] = field(default_factory=list)


@dataclass
class ConversionResult:
    """Result of voice-to-conversation conversion."""

    success: bool
    conversation: Conversation | None = None
    metadata: VoiceConversationMetadata | None = None
    error_message: str | None = None
    processing_time: float = 0.0
    quality_score: float = 0.0


class VoiceConversationConverter:
    """
    Converts voice-derived data into standard conversation format.

    Features:
    - Multi-speaker conversation detection and separation
    - Personality-aware role assignment
    - Temporal conversation flow reconstruction
    - Quality assessment and filtering
    - Metadata preservation and enrichment
    """

    def __init__(
        self,
        min_conversation_length: int = 3,
        max_gap_seconds: float = 30.0,
        min_speaker_turns: int = 2,
        personality_weight: float = 0.3,
        transcription_weight: float = 0.7,
    ):
        self.min_conversation_length = min_conversation_length
        self.max_gap_seconds = max_gap_seconds
        self.min_speaker_turns = min_speaker_turns
        self.personality_weight = personality_weight
        self.transcription_weight = transcription_weight

        self.logger = setup_logger("voice_conversation_converter")
        self.personality_extractor = PersonalityExtractor()

    def convert_transcription_to_conversation(
        self,
        transcription: TranscriptionResult,
        personality_profile: PersonalityProfile | None = None,
        speaker_mapping: dict[str, str] | None = None,
    ) -> ConversionResult:
        """Convert a transcription result to conversation format."""
        start_time = datetime.now()

        try:
            self.logger.info(
                f"Converting transcription to conversation: {transcription.file_path}"
            )

            # Extract personality profile if not provided
            if personality_profile is None:
                personality_profile = (
                    self.personality_extractor.extract_personality_profile(
                        transcription.full_text,
                        source_info={
                            "source": transcription.file_path,
                            "type": "voice_transcription",
                        },
                    )
                )

            # Detect speakers and conversation structure
            conversation_segments = self._detect_conversation_structure(
                transcription.segments
            )

            # Convert segments to messages
            messages = self._segments_to_messages(
                conversation_segments, personality_profile, speaker_mapping
            )

            # Validate conversation quality
            if len(messages) < self.min_conversation_length:
                return ConversionResult(
                    success=False,
                    error_message=f"Conversation too short: {len(messages)} messages < {self.min_conversation_length}",
                    processing_time=(datetime.now() - start_time).total_seconds(),
                )

            # Create conversation object
            conversation = Conversation(
                messages=messages,
                metadata={
                    "source": "voice_transcription",
                    "original_file": transcription.file_path,
                    "transcription_model": transcription.model_used,
                    "language": transcription.language,
                    "personality_analysis": True,
                    "conversion_timestamp": datetime.now().isoformat(),
                },
            )

            # Create metadata
            metadata = VoiceConversationMetadata(
                original_audio_file=transcription.file_path,
                transcription_model=transcription.model_used,
                transcription_confidence=transcription.confidence_score,
                personality_confidence=personality_profile.confidence_score,
                speaker_count=len({msg.role for msg in messages}),
                total_duration=sum(
                    seg.end_time - seg.start_time for seg in transcription.segments
                ),
                processing_timestamp=datetime.now().isoformat(),
                voice_quality_score=transcription.quality_score,
                personality_profile=self._serialize_personality_profile(
                    personality_profile
                ),
                transcription_segments=[
                    {
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "text": seg.text,
                        "confidence": seg.confidence,
                    }
                    for seg in transcription.segments
                ],
            )

            # Calculate overall quality score
            quality_score = self._calculate_conversion_quality(
                transcription, personality_profile, conversation
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            result = ConversionResult(
                success=True,
                conversation=conversation,
                metadata=metadata,
                processing_time=processing_time,
                quality_score=quality_score,
            )

            self.logger.info(
                f"Conversion successful: {len(messages)} messages, quality: {quality_score:.2f}"
            )
            return result

        except Exception as e:
            error_msg = f"Conversion failed: {e}"
            self.logger.error(error_msg)

            return ConversionResult(
                success=False,
                error_message=error_msg,
                processing_time=(datetime.now() - start_time).total_seconds(),
            )

    def _detect_conversation_structure(
        self, segments: list[TranscriptionSegment]
    ) -> list[list[TranscriptionSegment]]:
        """Detect conversation structure and group segments by speaker/turn."""
        if not segments:
            return []

        conversation_groups = []
        current_group = [segments[0]]

        for i in range(1, len(segments)):
            current_seg = segments[i]
            prev_seg = segments[i - 1]

            # Check for speaker change indicators
            gap_duration = current_seg.start_time - prev_seg.end_time

            # Heuristics for speaker change:
            # 1. Long pause (> max_gap_seconds)
            # 2. Significant change in speaking pattern
            # 3. Content analysis (questions followed by answers, etc.)

            speaker_change = False

            # Gap-based detection
            if gap_duration > self.max_gap_seconds:
                speaker_change = True

            # Content-based detection
            if self._detect_speaker_change_by_content(prev_seg.text, current_seg.text):
                speaker_change = True

            if speaker_change and len(current_group) >= 1:
                conversation_groups.append(current_group)
                current_group = [current_seg]
            else:
                current_group.append(current_seg)

        # Add the last group
        if current_group:
            conversation_groups.append(current_group)

        return conversation_groups

    def _detect_speaker_change_by_content(
        self, prev_text: str, current_text: str
    ) -> bool:
        """Detect speaker change based on content analysis."""
        prev_text = prev_text.strip().lower()
        current_text = current_text.strip().lower()

        # Question-answer pattern
        if prev_text.endswith("?") and not current_text.endswith("?"):
            return True

        # Greeting patterns
        greetings = [
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        ]
        if any(current_text.startswith(greeting) for greeting in greetings):
            return True

        # Response patterns
        response_starters = ["yes", "no", "well", "actually", "i think", "i believe"]
        return bool(any(current_text.startswith(starter) for starter in response_starters))

    def _segments_to_messages(
        self,
        conversation_groups: list[list[TranscriptionSegment]],
        personality_profile: PersonalityProfile,
        speaker_mapping: dict[str, str] | None = None,
    ) -> list[Message]:
        """Convert conversation segments to messages with role assignment."""
        messages = []

        # Assign roles based on conversation structure and personality
        roles = self._assign_speaker_roles(
            conversation_groups, personality_profile, speaker_mapping
        )

        for group_idx, segment_group in enumerate(conversation_groups):
            # Combine segments in the same group into a single message
            combined_text = " ".join(seg.text.strip() for seg in segment_group)

            # Clean up the text
            combined_text = self._clean_message_text(combined_text)

            if combined_text:  # Only add non-empty messages
                role = roles.get(group_idx, f"speaker_{group_idx % 2}")

                # Calculate timestamp from first segment
                timestamp = datetime.now() + timedelta(
                    seconds=segment_group[0].start_time
                )

                message = Message(
                    role=role,
                    content=combined_text,
                    timestamp=timestamp,
                    meta={
                        "source": "voice_transcription",
                        "segment_group": group_idx,
                        "start_time": segment_group[0].start_time,
                        "end_time": segment_group[-1].end_time,
                        "confidence": sum(seg.confidence for seg in segment_group)
                        / len(segment_group),
                        "segment_count": len(segment_group),
                    },
                )

                messages.append(message)

        return messages

    def _assign_speaker_roles(
        self,
        conversation_groups: list[list[TranscriptionSegment]],
        personality_profile: PersonalityProfile,
        speaker_mapping: dict[str, str] | None = None,
    ) -> dict[int, str]:
        """Assign speaker roles based on conversation analysis and personality."""
        roles = {}

        if speaker_mapping:
            # Use provided mapping
            for i, group in enumerate(conversation_groups):
                speaker_id = f"speaker_{i % len(speaker_mapping)}"
                roles[i] = speaker_mapping.get(speaker_id, f"speaker_{i % 2}")
        else:
            # Automatic role assignment based on conversation patterns

            # Simple alternating pattern for now
            # TODO: Enhance with more sophisticated speaker identification

            # Analyze conversation for therapeutic context
            full_text = " ".join(
                " ".join(seg.text for seg in group) for group in conversation_groups
            ).lower()

            is_therapeutic = any(
                word in full_text
                for word in [
                    "therapy",
                    "therapist",
                    "counseling",
                    "session",
                    "feelings",
                    "how are you feeling",
                    "tell me about",
                    "what brings you",
                ]
            )

            if is_therapeutic:
                # Assign therapist/client roles based on question patterns
                for i, group in enumerate(conversation_groups):
                    group_text = " ".join(seg.text for seg in group).lower()

                    # Therapist indicators: questions, professional language
                    therapist_indicators = [
                        "how do you feel",
                        "tell me about",
                        "what brings you",
                        "can you describe",
                        "how has that been",
                        "what would you like",
                    ]

                    if any(
                        indicator in group_text for indicator in therapist_indicators
                    ) or group_text.count("?") > 0:
                        roles[i] = "therapist"
                    else:
                        roles[i] = "client"
            else:
                # General conversation - use personality-based assignment
                for i, group in enumerate(conversation_groups):
                    if i % 2 == 0:
                        roles[i] = "user"
                    else:
                        roles[i] = "assistant"

        return roles

    def _clean_message_text(self, text: str) -> str:
        """Clean and normalize message text."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove filler words and transcription artifacts
        filler_words = ["um", "uh", "er", "ah", "like", "you know"]
        for filler in filler_words:
            text = re.sub(rf"\b{filler}\b", "", text, flags=re.IGNORECASE)

        # Clean up punctuation
        text = re.sub(r"\s+([.!?])", r"\1", text)  # Remove space before punctuation
        text = re.sub(r"([.!?])\s*([.!?])", r"\1", text)  # Remove duplicate punctuation

        # Capitalize first letter
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]

        return text

    def _serialize_personality_profile(self, profile: PersonalityProfile) -> dict:
        """Serialize personality profile for metadata storage."""
        return {
            "confidence_score": profile.confidence_score,
            "personality_scores": [
                {
                    "dimension": score.dimension.value,
                    "score": score.score,
                    "confidence": score.confidence,
                }
                for score in profile.personality_scores
            ],
            "communication_patterns": [
                {"style": pattern.style.value, "strength": pattern.strength}
                for pattern in profile.communication_patterns[:3]  # Top 3 patterns
            ],
            "emotional_profile": {
                "dominant_emotions": profile.emotional_profile.dominant_emotions,
                "emotional_range": profile.emotional_profile.emotional_range,
                "emotional_stability": profile.emotional_profile.emotional_stability,
            },
        }

    def _calculate_conversion_quality(
        self,
        transcription: TranscriptionResult,
        personality_profile: PersonalityProfile,
        conversation: Conversation,
    ) -> float:
        """Calculate overall quality score for the conversion."""

        # Transcription quality component
        transcription_quality = (
            transcription.confidence_score * 0.6 + transcription.quality_score * 0.4
        )

        # Personality analysis quality
        personality_quality = personality_profile.confidence_score

        # Conversation structure quality
        message_count = len(conversation.messages)
        speaker_variety = len({msg.role for msg in conversation.messages})

        structure_quality = min(
            1.0,
            (message_count / 10.0) * 0.5  # Reward longer conversations
            + (speaker_variety / 2.0) * 0.5,  # Reward multi-speaker conversations
        )

        # Weighted overall quality
        overall_quality = (
            transcription_quality * self.transcription_weight
            + personality_quality * self.personality_weight * 0.5
            + structure_quality * 0.2
        )

        return min(1.0, max(0.0, overall_quality))

    def convert_batch(
        self,
        transcription_results: list[TranscriptionResult],
        output_dir: str | None = None,
    ) -> list[ConversionResult]:
        """Convert multiple transcription results to conversations."""
        results = []

        self.logger.info(
            f"Converting {len(transcription_results)} transcriptions to conversations"
        )

        for i, transcription in enumerate(transcription_results):
            self.logger.info(
                f"Converting {i+1}/{len(transcription_results)}: {transcription.file_path}"
            )

            try:
                result = self.convert_transcription_to_conversation(transcription)
                results.append(result)

                # Save conversation if output directory specified and conversion successful
                if output_dir and result.success and result.conversation:
                    self._save_conversation(
                        result.conversation, result.metadata, output_dir
                    )

            except Exception as e:
                error_msg = f"Conversion failed for {transcription.file_path}: {e}"
                self.logger.error(error_msg)

                results.append(ConversionResult(success=False, error_message=error_msg))

        # Summary
        successful = sum(1 for r in results if r.success)
        self.logger.info(
            f"Batch conversion complete: {successful}/{len(transcription_results)} successful"
        )

        return results

    def _save_conversation(
        self,
        conversation: Conversation,
        metadata: VoiceConversationMetadata,
        output_dir: str,
    ):
        """Save conversation and metadata to files."""
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Generate filename from original audio file
            audio_file = Path(metadata.original_audio_file)
            base_name = audio_file.stem

            # Save conversation as JSON
            conversation_path = Path(output_dir) / f"{base_name}_conversation.json"
            conversation_data = {
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": (
                            msg.timestamp.isoformat() if msg.timestamp else None
                        ),
                        "meta": msg.meta,
                    }
                    for msg in conversation.messages
                ],
                "metadata": conversation.metadata,
            }

            with open(conversation_path, "w", encoding="utf-8") as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)

            # Save detailed metadata
            metadata_path = Path(output_dir) / f"{base_name}_voice_metadata.json"
            metadata_data = {
                "original_audio_file": metadata.original_audio_file,
                "transcription_model": metadata.transcription_model,
                "transcription_confidence": metadata.transcription_confidence,
                "personality_confidence": metadata.personality_confidence,
                "speaker_count": metadata.speaker_count,
                "total_duration": metadata.total_duration,
                "processing_timestamp": metadata.processing_timestamp,
                "voice_quality_score": metadata.voice_quality_score,
                "personality_profile": metadata.personality_profile,
                "transcription_segments": metadata.transcription_segments,
            }

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_data, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Conversation saved: {conversation_path}")

        except Exception as e:
            self.logger.error(f"Failed to save conversation: {e}")

    def generate_conversion_report(self, results: list[ConversionResult]) -> str:
        """Generate a detailed conversion report."""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        report = []
        report.append("=" * 60)
        report.append("VOICE-TO-CONVERSATION CONVERSION REPORT")
        report.append("=" * 60)
        report.append(f"Total Conversions: {len(results)}")
        report.append(f"Successful: {len(successful_results)}")
        report.append(f"Failed: {len(failed_results)}")

        if successful_results:
            avg_quality = sum(r.quality_score for r in successful_results) / len(
                successful_results
            )
            avg_messages = sum(
                len(r.conversation.messages) for r in successful_results
            ) / len(successful_results)
            avg_speakers = sum(
                r.metadata.speaker_count for r in successful_results
            ) / len(successful_results)

            report.append(f"Average Quality Score: {avg_quality:.2f}")
            report.append(f"Average Messages per Conversation: {avg_messages:.1f}")
            report.append(f"Average Speakers per Conversation: {avg_speakers:.1f}")

        report.append(
            f"Success Rate: {(len(successful_results)/len(results)*100):.1f}%"
        )
        report.append("")

        if successful_results:
            report.append("SUCCESSFUL CONVERSIONS:")
            report.append("-" * 40)
            for result in successful_results[:10]:  # Show first 10
                filename = Path(result.metadata.original_audio_file).name
                msg_count = len(result.conversation.messages)
                quality = result.quality_score
                speakers = result.metadata.speaker_count
                report.append(
                    f"✅ {filename} | {msg_count} messages | {speakers} speakers | quality: {quality:.2f}"
                )

        if failed_results:
            report.append("")
            report.append("FAILED CONVERSIONS:")
            report.append("-" * 40)
            for result in failed_results[:10]:  # Show first 10 failures
                error = result.error_message or "Unknown error"
                report.append(f"❌ {error[:100]}...")

        report.append("=" * 60)
        return "\n".join(report)


# Backward compatibility function
def convert_voice_to_conversations(
    transcription_results: list[TranscriptionResult],
    output_dir: str | None = None,
    min_conversation_length: int = 3,
) -> list[ConversionResult]:
    """
    Convert voice transcriptions to conversation format with enhanced capabilities.

    This function provides backward compatibility while offering
    the enhanced features of the new VoiceConversationConverter.
    """
    converter = VoiceConversationConverter(
        min_conversation_length=min_conversation_length
    )

    return converter.convert_batch(transcription_results, output_dir)
