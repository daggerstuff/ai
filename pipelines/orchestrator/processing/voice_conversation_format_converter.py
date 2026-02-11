"""
Voice Conversation Format Converter

Builds conversation format converter for voice data (Task 3.5).
Converts voice-derived transcriptions into standardized conversation formats.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from logger import get_logger

logger = get_logger(__name__)

@dataclass
class VoiceConversationData:
    """Voice-derived conversation data."""
    conversation_id: str
    original_audio_path: str
    transcription_text: str
    speaker_segments: list[dict[str, Any]] = field(default_factory=list)
    conversation_format: dict[str, Any] = field(default_factory=dict)
    quality_metrics: dict[str, float] = field(default_factory=dict)

class VoiceConversationFormatConverter:
    """Builds conversation format converter for voice data."""

    def __init__(self, voice_data_path: str = "./voice_data", output_dir: str = "./converted_conversations"):
        self.voice_data_path = Path(voice_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = get_logger(__name__)

        # Conversation format templates
        self.conversation_formats = {
            "therapeutic": {
                "roles": ["client", "therapist"],
                "structure": "alternating_dialogue",
                "metadata_fields": ["session_type", "therapeutic_approach", "client_demographics"]
            },
            "counseling": {
                "roles": ["counselee", "counselor"],
                "structure": "guided_conversation",
                "metadata_fields": ["counseling_type", "session_phase", "intervention_used"]
            },
            "interview": {
                "roles": ["interviewer", "interviewee"],
                "structure": "question_answer",
                "metadata_fields": ["interview_type", "topic", "duration"]
            },
            "support_group": {
                "roles": ["facilitator", "participant"],
                "structure": "group_discussion",
                "metadata_fields": ["group_type", "session_number", "participant_count"]
            }
        }

        # Speaker identification patterns
        self.speaker_patterns = {
            "therapist_indicators": ["let's explore", "how does that make you feel", "what comes to mind", "i hear you saying"],
            "client_indicators": ["i feel", "i think", "i'm struggling", "i don't know"],
            "counselor_indicators": ["what would you like to work on", "let's focus on", "that's a great insight"],
            "facilitator_indicators": ["let's go around", "who would like to share", "thank you for sharing"]
        }

        logger.info("VoiceConversationFormatConverter initialized")

    def convert_voice_to_conversation_format(self, voice_file_path: str, target_format: str = "therapeutic") -> dict[str, Any]:
        """Convert voice data to standardized conversation format."""
        start_time = datetime.now()

        result = {
            "success": False,
            "conversation_id": "",
            "original_file": voice_file_path,
            "target_format": target_format,
            "segments_processed": 0,
            "quality_score": 0.0,
            "issues": [],
            "output_path": None
        }

        try:
            # Load voice data (mock implementation)
            voice_data = self._load_voice_data(voice_file_path)

            # Extract speaker segments
            speaker_segments = self._extract_speaker_segments(voice_data)

            # Identify speakers and roles
            role_assignments = self._identify_speaker_roles(speaker_segments, target_format)

            # Convert to conversation format
            conversation_data = self._convert_to_conversation_format(
                speaker_segments, role_assignments, target_format
            )

            # Assess conversion quality
            quality_metrics = self._assess_conversion_quality(conversation_data)

            # Save converted conversation
            output_path = self._save_converted_conversation(conversation_data, quality_metrics)

            # Update result
            result.update({
                "success": True,
                "conversation_id": conversation_data.conversation_id,
                "segments_processed": len(speaker_segments),
                "quality_score": quality_metrics.get("overall_quality", 0.0),
                "output_path": str(output_path),
                "processing_time": (datetime.now() - start_time).total_seconds()
            })

            logger.info(f"Successfully converted voice to conversation format: {conversation_data.conversation_id}")

        except Exception as e:
            result["issues"].append(f"Conversion failed: {e!s}")
            logger.error(f"Voice conversation format conversion failed: {e}")

        return result

    def _load_voice_data(self, voice_file_path: str) -> dict[str, Any]:
        """Load voice data (mock implementation for testing)."""
        # Mock voice data with transcription
        return {
            "file_path": voice_file_path,
            "transcription": "Speaker 1: I've been feeling really anxious lately and I don't know what to do about it. Speaker 2: I hear you saying that you're experiencing anxiety. Can you tell me more about when these feelings started? Speaker 1: It started about a month ago when I changed jobs. Everything feels overwhelming now. Speaker 2: Job transitions can be really challenging. Let's explore what specific aspects feel most overwhelming to you.",
            "speaker_timestamps": [
                {"speaker": "Speaker 1", "start": 0.0, "end": 8.5, "text": "I've been feeling really anxious lately and I don't know what to do about it."},
                {"speaker": "Speaker 2", "start": 9.0, "end": 16.2, "text": "I hear you saying that you're experiencing anxiety. Can you tell me more about when these feelings started?"},
                {"speaker": "Speaker 1", "start": 17.0, "end": 24.8, "text": "It started about a month ago when I changed jobs. Everything feels overwhelming now."},
                {"speaker": "Speaker 2", "start": 25.5, "end": 33.1, "text": "Job transitions can be really challenging. Let's explore what specific aspects feel most overwhelming to you."}
            ],
            "audio_quality": 0.85,
            "duration": 35.0
        }

    def _extract_speaker_segments(self, voice_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract speaker segments from voice data."""
        segments = []

        for timestamp_data in voice_data.get("speaker_timestamps", []):
            segment = {
                "speaker_id": timestamp_data["speaker"],
                "start_time": timestamp_data["start"],
                "end_time": timestamp_data["end"],
                "text": timestamp_data["text"],
                "duration": timestamp_data["end"] - timestamp_data["start"],
                "word_count": len(timestamp_data["text"].split())
            }
            segments.append(segment)

        return segments

    def _identify_speaker_roles(self, segments: list[dict[str, Any]], target_format: str) -> dict[str, str]:
        """Identify speaker roles based on conversation patterns."""
        format_config = self.conversation_formats.get(target_format, self.conversation_formats["therapeutic"])
        available_roles = format_config["roles"]

        role_assignments = {}
        speaker_scores = {}

        # Analyze each speaker's language patterns
        for segment in segments:
            speaker_id = segment["speaker_id"]
            text = segment["text"].lower()

            if speaker_id not in speaker_scores:
                speaker_scores[speaker_id] = dict.fromkeys(available_roles, 0)

            # Score based on language patterns
            for role in available_roles:
                if role in ["therapist", "counselor", "facilitator"]:
                    # Professional helper patterns
                    for pattern in self.speaker_patterns.get(f"{role}_indicators", []):
                        if pattern in text:
                            speaker_scores[speaker_id][role] += 1
                elif role in ["client", "counselee", "participant"]:
                    # Help-seeker patterns
                    for pattern in self.speaker_patterns.get("client_indicators", []):
                        if pattern in text:
                            speaker_scores[speaker_id][role] += 1

        # Assign roles based on highest scores
        assigned_roles = set()
        for speaker_id, scores in speaker_scores.items():
            best_role = max(scores.items(), key=lambda x: x[1])[0]
            if best_role not in assigned_roles:
                role_assignments[speaker_id] = best_role
                assigned_roles.add(best_role)
            else:
                # Assign remaining role
                remaining_roles = [r for r in available_roles if r not in assigned_roles]
                if remaining_roles:
                    role_assignments[speaker_id] = remaining_roles[0]
                    assigned_roles.add(remaining_roles[0])

        return role_assignments

    def _convert_to_conversation_format(self, segments: list[dict[str, Any]],
                                      role_assignments: dict[str, str],
                                      target_format: str) -> VoiceConversationData:
        """Convert segments to standardized conversation format."""

        conversation_id = f"voice_conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Build conversation messages
        messages = []
        for segment in segments:
            speaker_id = segment["speaker_id"]
            role = role_assignments.get(speaker_id, "unknown")

            message = {
                "role": role,
                "content": segment["text"],
                "timestamp": segment["start_time"],
                "duration": segment["duration"],
                "metadata": {
                    "speaker_id": speaker_id,
                    "word_count": segment["word_count"],
                    "segment_quality": self._assess_segment_quality(segment)
                }
            }
            messages.append(message)

        # Create conversation format
        format_config = self.conversation_formats[target_format]
        conversation_format = {
            "format_type": target_format,
            "structure": format_config["structure"],
            "messages": messages,
            "metadata": {
                "total_duration": sum(s["duration"] for s in segments),
                "message_count": len(messages),
                "speaker_count": len({s["speaker_id"] for s in segments}),
                "role_assignments": role_assignments,
                "conversion_timestamp": datetime.now().isoformat()
            }
        }

        return VoiceConversationData(
            conversation_id=conversation_id,
            original_audio_path="mock_audio_path.wav",
            transcription_text=" ".join(s["text"] for s in segments),
            speaker_segments=segments,
            conversation_format=conversation_format
        )

    def _assess_segment_quality(self, segment: dict[str, Any]) -> float:
        """Assess quality of individual segment."""
        quality_score = 0.5  # Base score

        # Duration check (not too short, not too long)
        duration = segment["duration"]
        if 2.0 <= duration <= 15.0:
            quality_score += 0.2

        # Word count check
        word_count = segment["word_count"]
        if 5 <= word_count <= 50:
            quality_score += 0.2

        # Text quality (no empty or very short text)
        text = segment["text"].strip()
        if len(text) >= 10:
            quality_score += 0.1

        return min(1.0, quality_score)

    def _assess_conversion_quality(self, conversation_data: VoiceConversationData) -> dict[str, float]:
        """Assess quality of conversation format conversion."""

        messages = conversation_data.conversation_format["messages"]

        if not messages:
            return {"overall_quality": 0.0}

        # Role assignment quality
        role_diversity = len({msg["role"] for msg in messages if msg["role"] != "unknown"})
        role_quality = min(1.0, role_diversity / 2)  # Expect at least 2 roles

        # Message quality
        message_qualities = [msg["metadata"]["segment_quality"] for msg in messages]
        avg_message_quality = sum(message_qualities) / len(message_qualities)

        # Conversation flow quality (alternating speakers)
        flow_quality = 0.5
        if len(messages) > 1:
            alternations = sum(1 for i in range(1, len(messages))
                             if messages[i]["role"] != messages[i-1]["role"])
            flow_quality = min(1.0, alternations / (len(messages) - 1))

        # Duration balance (no speaker dominates too much)
        speaker_durations = {}
        for msg in messages:
            role = msg["role"]
            speaker_durations[role] = speaker_durations.get(role, 0) + msg["duration"]

        if len(speaker_durations) > 1:
            total_duration = sum(speaker_durations.values())
            duration_balance = 1.0 - max(d / total_duration for d in speaker_durations.values())
        else:
            duration_balance = 0.0

        overall_quality = (role_quality + avg_message_quality + flow_quality + duration_balance) / 4

        return {
            "overall_quality": overall_quality,
            "role_assignment_quality": role_quality,
            "message_quality": avg_message_quality,
            "conversation_flow_quality": flow_quality,
            "speaker_balance": duration_balance
        }

    def _save_converted_conversation(self, conversation_data: VoiceConversationData,
                                   quality_metrics: dict[str, float]) -> Path:
        """Save converted conversation format."""
        output_file = self.output_dir / f"{conversation_data.conversation_id}_converted.json"

        output_data = {
            "conversation_info": {
                "conversation_id": conversation_data.conversation_id,
                "original_audio_path": conversation_data.original_audio_path,
                "conversion_type": "voice_to_conversation_format",
                "converted_at": datetime.now().isoformat()
            },
            "transcription": {
                "full_text": conversation_data.transcription_text,
                "speaker_segments": conversation_data.speaker_segments
            },
            "conversation_format": conversation_data.conversation_format,
            "quality_metrics": quality_metrics,
            "conversion_metadata": {
                "converter_version": "1.0",
                "processing_timestamp": datetime.now().isoformat()
            }
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Converted conversation saved: {output_file}")
        return output_file


# Example usage
if __name__ == "__main__":
    # Initialize converter
    converter = VoiceConversationFormatConverter()

    # Convert voice to conversation format
    result = converter.convert_voice_to_conversation_format("mock_audio.wav", "therapeutic")

    # Show results
    if result["success"]:
        pass
    else:
        pass

