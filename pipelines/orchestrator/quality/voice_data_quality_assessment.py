"""
Voice Data Quality Assessment and Filtering

Builds voice data quality assessment and filtering system (Task 3.8).
Comprehensive quality evaluation for voice-derived conversation data.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from logger import get_logger

logger = get_logger(__name__)

@dataclass
class VoiceQualityMetrics:
    """Voice data quality metrics."""
    audio_quality_score: float = 0.0
    transcription_quality_score: float = 0.0
    conversation_quality_score: float = 0.0
    overall_quality_score: float = 0.0
    quality_issues: list[str] = field(default_factory=list)
    quality_recommendations: list[str] = field(default_factory=list)

@dataclass
class VoiceDataAssessment:
    """Complete voice data quality assessment."""
    file_id: str
    file_path: str
    assessment_timestamp: str
    quality_metrics: VoiceQualityMetrics
    filter_decision: str  # "accept", "reject", "conditional"
    filter_reason: str = ""

class VoiceDataQualityAssessment:
    """Builds voice data quality assessment and filtering system."""

    def __init__(self, voice_data_path: str = "./voice_data", output_dir: str = "./quality_assessments"):
        self.voice_data_path = Path(voice_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = get_logger(__name__)

        # Quality thresholds
        self.quality_thresholds = {
            "audio_quality": {
                "excellent": 0.9,
                "good": 0.7,
                "acceptable": 0.5,
                "poor": 0.3
            },
            "transcription_quality": {
                "excellent": 0.95,
                "good": 0.8,
                "acceptable": 0.6,
                "poor": 0.4
            },
            "conversation_quality": {
                "excellent": 0.9,
                "good": 0.75,
                "acceptable": 0.6,
                "poor": 0.4
            },
            "overall_acceptance": 0.6
        }

        # Quality assessment criteria
        self.assessment_criteria = {
            "audio_quality": [
                "signal_to_noise_ratio",
                "clarity_score",
                "volume_consistency",
                "background_noise_level",
                "audio_artifacts"
            ],
            "transcription_quality": [
                "word_recognition_accuracy",
                "punctuation_quality",
                "speaker_identification_accuracy",
                "timestamp_accuracy",
                "transcription_completeness"
            ],
            "conversation_quality": [
                "dialogue_coherence",
                "speaker_turn_quality",
                "content_relevance",
                "therapeutic_value",
                "conversation_flow"
            ]
        }

        # Filter criteria
        self.filter_criteria = {
            "automatic_reject": {
                "min_duration": 30.0,  # seconds
                "max_duration": 3600.0,  # 1 hour
                "min_word_count": 20,
                "min_speaker_count": 2,
                "max_silence_ratio": 0.7
            },
            "conditional_accept": {
                "min_overall_quality": 0.4,
                "min_audio_quality": 0.3,
                "min_transcription_quality": 0.4
            }
        }

        logger.info("VoiceDataQualityAssessment initialized")

    def assess_voice_data_quality(self, voice_file_path: str) -> VoiceDataAssessment:
        """Assess quality of voice data file."""
        datetime.now()

        try:
            # Load voice data
            voice_data = self._load_voice_data(voice_file_path)

            # Assess audio quality
            audio_quality = self._assess_audio_quality(voice_data)

            # Assess transcription quality
            transcription_quality = self._assess_transcription_quality(voice_data)

            # Assess conversation quality
            conversation_quality = self._assess_conversation_quality(voice_data)

            # Calculate overall quality
            overall_quality = self._calculate_overall_quality(
                audio_quality, transcription_quality, conversation_quality
            )

            # Create quality metrics
            quality_metrics = VoiceQualityMetrics(
                audio_quality_score=audio_quality["score"],
                transcription_quality_score=transcription_quality["score"],
                conversation_quality_score=conversation_quality["score"],
                overall_quality_score=overall_quality["score"],
                quality_issues=overall_quality["issues"],
                quality_recommendations=overall_quality["recommendations"]
            )

            # Make filter decision
            filter_decision, filter_reason = self._make_filter_decision(quality_metrics, voice_data)

            # Create assessment
            assessment = VoiceDataAssessment(
                file_id=f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                file_path=voice_file_path,
                assessment_timestamp=datetime.now().isoformat(),
                quality_metrics=quality_metrics,
                filter_decision=filter_decision,
                filter_reason=filter_reason
            )

            logger.info(f"Voice data quality assessment completed: {assessment.file_id} - {filter_decision}")
            return assessment

        except Exception as e:
            logger.error(f"Voice data quality assessment failed: {e}")
            # Return failed assessment
            return VoiceDataAssessment(
                file_id=f"failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                file_path=voice_file_path,
                assessment_timestamp=datetime.now().isoformat(),
                quality_metrics=VoiceQualityMetrics(),
                filter_decision="reject",
                filter_reason=f"Assessment failed: {e!s}"
            )

    def batch_assess_voice_data(self, voice_files: list[str]) -> dict[str, Any]:
        """Assess quality of multiple voice data files."""
        start_time = datetime.now()

        assessments = []
        filter_stats = {"accept": 0, "reject": 0, "conditional": 0}

        for voice_file in voice_files:
            assessment = self.assess_voice_data_quality(voice_file)
            assessments.append(assessment)
            filter_stats[assessment.filter_decision] += 1

        # Calculate batch statistics
        batch_stats = self._calculate_batch_statistics(assessments)

        # Save batch results
        output_path = self._save_batch_assessment_results(assessments, batch_stats)

        return {
            "success": True,
            "files_assessed": len(voice_files),
            "filter_statistics": filter_stats,
            "batch_statistics": batch_stats,
            "assessments": assessments,
            "output_path": str(output_path),
            "processing_time": (datetime.now() - start_time).total_seconds()
        }

    def _load_voice_data(self, voice_file_path: str) -> dict[str, Any]:
        """Load voice data (mock implementation for testing)."""
        # Mock voice data with various quality indicators
        return {
            "file_path": voice_file_path,
            "duration": 180.0,  # 3 minutes
            "audio_metrics": {
                "signal_to_noise_ratio": 15.2,  # dB
                "clarity_score": 0.85,
                "volume_consistency": 0.78,
                "background_noise_level": 0.15,
                "audio_artifacts": 0.05
            },
            "transcription": {
                "text": "This is a sample therapeutic conversation between a client and therapist discussing anxiety management techniques.",
                "word_count": 16,
                "confidence_scores": [0.95, 0.92, 0.88, 0.94, 0.91, 0.89, 0.93, 0.87, 0.96, 0.90, 0.85, 0.92, 0.88, 0.94, 0.89, 0.91],
                "speaker_segments": [
                    {"speaker": "client", "text": "I've been having trouble with anxiety lately.", "confidence": 0.92},
                    {"speaker": "therapist", "text": "Can you tell me more about when you notice these feelings?", "confidence": 0.89}
                ],
                "timestamp_accuracy": 0.94
            },
            "conversation_data": {
                "speaker_count": 2,
                "turn_count": 12,
                "dialogue_coherence": 0.87,
                "therapeutic_indicators": ["anxiety", "feelings", "coping", "support"],
                "silence_ratio": 0.12
            }
        }

    def _assess_audio_quality(self, voice_data: dict[str, Any]) -> dict[str, Any]:
        """Assess audio quality metrics."""
        audio_metrics = voice_data.get("audio_metrics", {})

        # Signal-to-noise ratio assessment
        snr = audio_metrics.get("signal_to_noise_ratio", 0)
        snr_score = min(1.0, max(0.0, (snr - 5) / 20))  # 5-25 dB range

        # Clarity score
        clarity_score = audio_metrics.get("clarity_score", 0.5)

        # Volume consistency
        volume_consistency = audio_metrics.get("volume_consistency", 0.5)

        # Background noise level (lower is better)
        noise_level = audio_metrics.get("background_noise_level", 0.5)
        noise_score = 1.0 - noise_level

        # Audio artifacts (lower is better)
        artifacts = audio_metrics.get("audio_artifacts", 0.5)
        artifacts_score = 1.0 - artifacts

        # Calculate overall audio quality
        audio_quality_score = (snr_score + clarity_score + volume_consistency + noise_score + artifacts_score) / 5

        issues = []
        if snr < 10:
            issues.append("Low signal-to-noise ratio")
        if clarity_score < 0.7:
            issues.append("Poor audio clarity")
        if noise_level > 0.3:
            issues.append("High background noise")
        if artifacts > 0.2:
            issues.append("Audio artifacts detected")

        return {
            "score": audio_quality_score,
            "components": {
                "snr_score": snr_score,
                "clarity_score": clarity_score,
                "volume_consistency": volume_consistency,
                "noise_score": noise_score,
                "artifacts_score": artifacts_score
            },
            "issues": issues
        }

    def _assess_transcription_quality(self, voice_data: dict[str, Any]) -> dict[str, Any]:
        """Assess transcription quality metrics."""
        transcription = voice_data.get("transcription", {})

        # Word recognition accuracy (from confidence scores)
        confidence_scores = transcription.get("confidence_scores", [0.5])
        avg_confidence = sum(confidence_scores) / len(confidence_scores)

        # Transcription completeness
        word_count = transcription.get("word_count", 0)
        duration = voice_data.get("duration", 1)
        words_per_minute = (word_count / duration) * 60
        completeness_score = min(1.0, words_per_minute / 150)  # Expect ~150 WPM

        # Speaker identification accuracy
        speaker_segments = transcription.get("speaker_segments", [])
        speaker_confidence = sum(seg.get("confidence", 0.5) for seg in speaker_segments) / max(1, len(speaker_segments))

        # Timestamp accuracy
        timestamp_accuracy = transcription.get("timestamp_accuracy", 0.5)

        # Text quality (basic checks)
        text = transcription.get("text", "")
        text_quality = self._assess_text_quality(text)

        # Calculate overall transcription quality
        transcription_quality_score = (
            avg_confidence + completeness_score + speaker_confidence +
            timestamp_accuracy + text_quality
        ) / 5

        issues = []
        if avg_confidence < 0.8:
            issues.append("Low transcription confidence")
        if completeness_score < 0.5:
            issues.append("Incomplete transcription")
        if speaker_confidence < 0.7:
            issues.append("Poor speaker identification")
        if text_quality < 0.6:
            issues.append("Poor text quality")

        return {
            "score": transcription_quality_score,
            "components": {
                "word_recognition": avg_confidence,
                "completeness": completeness_score,
                "speaker_identification": speaker_confidence,
                "timestamp_accuracy": timestamp_accuracy,
                "text_quality": text_quality
            },
            "issues": issues
        }

    def _assess_conversation_quality(self, voice_data: dict[str, Any]) -> dict[str, Any]:
        """Assess conversation quality metrics."""
        conversation_data = voice_data.get("conversation_data", {})

        # Speaker count (expect 2+ for dialogue)
        speaker_count = conversation_data.get("speaker_count", 1)
        speaker_score = min(1.0, speaker_count / 2)

        # Turn count (more turns = better dialogue)
        turn_count = conversation_data.get("turn_count", 1)
        duration = voice_data.get("duration", 1)
        turns_per_minute = (turn_count / duration) * 60
        turn_score = min(1.0, turns_per_minute / 10)  # Expect ~10 turns per minute

        # Dialogue coherence
        dialogue_coherence = conversation_data.get("dialogue_coherence", 0.5)

        # Therapeutic value (presence of therapeutic indicators)
        therapeutic_indicators = conversation_data.get("therapeutic_indicators", [])
        therapeutic_score = min(1.0, len(therapeutic_indicators) / 5)  # Expect 5+ indicators

        # Silence ratio (not too much silence)
        silence_ratio = conversation_data.get("silence_ratio", 0.5)
        silence_score = 1.0 - min(1.0, silence_ratio / 0.5)  # Penalize >50% silence

        # Calculate overall conversation quality
        conversation_quality_score = (
            speaker_score + turn_score + dialogue_coherence +
            therapeutic_score + silence_score
        ) / 5

        issues = []
        if speaker_count < 2:
            issues.append("Insufficient speakers for dialogue")
        if turns_per_minute < 3:
            issues.append("Low dialogue interaction")
        if dialogue_coherence < 0.6:
            issues.append("Poor dialogue coherence")
        if len(therapeutic_indicators) < 2:
            issues.append("Limited therapeutic content")
        if silence_ratio > 0.4:
            issues.append("Excessive silence")

        return {
            "score": conversation_quality_score,
            "components": {
                "speaker_diversity": speaker_score,
                "interaction_frequency": turn_score,
                "dialogue_coherence": dialogue_coherence,
                "therapeutic_value": therapeutic_score,
                "silence_management": silence_score
            },
            "issues": issues
        }

    def _assess_text_quality(self, text: str) -> float:
        """Assess basic text quality."""
        if not text:
            return 0.0

        quality_score = 0.5  # Base score

        # Length check
        if len(text) >= 50:
            quality_score += 0.2

        # Word count check
        words = text.split()
        if len(words) >= 10:
            quality_score += 0.2

        # Basic grammar check (simple heuristics)
        if text[0].isupper():  # Starts with capital
            quality_score += 0.05
        if text.endswith("."):  # Ends with punctuation
            quality_score += 0.05

        return min(1.0, quality_score)

    def _calculate_overall_quality(self, audio_quality: dict[str, Any],
                                 transcription_quality: dict[str, Any],
                                 conversation_quality: dict[str, Any]) -> dict[str, Any]:
        """Calculate overall quality score and recommendations."""

        # Weighted average (transcription quality is most important)
        overall_score = (
            audio_quality["score"] * 0.25 +
            transcription_quality["score"] * 0.45 +
            conversation_quality["score"] * 0.30
        )

        # Collect all issues
        all_issues = (
            audio_quality["issues"] +
            transcription_quality["issues"] +
            conversation_quality["issues"]
        )

        # Generate recommendations
        recommendations = []
        if audio_quality["score"] < 0.6:
            recommendations.append("Improve audio recording quality")
        if transcription_quality["score"] < 0.7:
            recommendations.append("Review and correct transcription")
        if conversation_quality["score"] < 0.6:
            recommendations.append("Enhance conversation content quality")

        return {
            "score": overall_score,
            "issues": all_issues,
            "recommendations": recommendations
        }

    def _make_filter_decision(self, quality_metrics: VoiceQualityMetrics,
                            voice_data: dict[str, Any]) -> tuple[str, str]:
        """Make filtering decision based on quality metrics."""

        # Check automatic rejection criteria
        duration = voice_data.get("duration", 0)
        word_count = voice_data.get("transcription", {}).get("word_count", 0)
        speaker_count = voice_data.get("conversation_data", {}).get("speaker_count", 0)
        silence_ratio = voice_data.get("conversation_data", {}).get("silence_ratio", 1.0)

        reject_criteria = self.filter_criteria["automatic_reject"]

        if duration < reject_criteria["min_duration"]:
            return "reject", f"Duration too short: {duration}s < {reject_criteria['min_duration']}s"
        if duration > reject_criteria["max_duration"]:
            return "reject", f"Duration too long: {duration}s > {reject_criteria['max_duration']}s"
        if word_count < reject_criteria["min_word_count"]:
            return "reject", f"Word count too low: {word_count} < {reject_criteria['min_word_count']}"
        if speaker_count < reject_criteria["min_speaker_count"]:
            return "reject", f"Insufficient speakers: {speaker_count} < {reject_criteria['min_speaker_count']}"
        if silence_ratio > reject_criteria["max_silence_ratio"]:
            return "reject", f"Excessive silence: {silence_ratio:.2f} > {reject_criteria['max_silence_ratio']}"

        # Check quality thresholds
        overall_threshold = self.quality_thresholds["overall_acceptance"]

        if quality_metrics.overall_quality_score >= overall_threshold:
            return "accept", f"Quality score {quality_metrics.overall_quality_score:.2f} meets threshold {overall_threshold}"

        # Check conditional acceptance
        conditional_criteria = self.filter_criteria["conditional_accept"]
        if (quality_metrics.overall_quality_score >= conditional_criteria["min_overall_quality"] and
            quality_metrics.audio_quality_score >= conditional_criteria["min_audio_quality"] and
            quality_metrics.transcription_quality_score >= conditional_criteria["min_transcription_quality"]):
            return "conditional", "Meets minimum criteria for conditional acceptance"

        return "reject", f"Quality score {quality_metrics.overall_quality_score:.2f} below threshold {overall_threshold}"

    def _calculate_batch_statistics(self, assessments: list[VoiceDataAssessment]) -> dict[str, Any]:
        """Calculate statistics for batch assessment."""
        if not assessments:
            return {}

        quality_scores = [a.quality_metrics.overall_quality_score for a in assessments]

        return {
            "total_files": len(assessments),
            "average_quality": sum(quality_scores) / len(quality_scores),
            "quality_std": np.std(quality_scores),
            "min_quality": min(quality_scores),
            "max_quality": max(quality_scores),
            "quality_distribution": {
                "excellent": sum(1 for q in quality_scores if q >= 0.9),
                "good": sum(1 for q in quality_scores if 0.7 <= q < 0.9),
                "acceptable": sum(1 for q in quality_scores if 0.5 <= q < 0.7),
                "poor": sum(1 for q in quality_scores if q < 0.5)
            }
        }

    def _save_batch_assessment_results(self, assessments: list[VoiceDataAssessment],
                                     batch_stats: dict[str, Any]) -> Path:
        """Save batch assessment results."""
        output_file = self.output_dir / f"voice_quality_batch_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert assessments to serializable format
        assessments_data = []
        for assessment in assessments:
            assessment_dict = {
                "file_id": assessment.file_id,
                "file_path": assessment.file_path,
                "assessment_timestamp": assessment.assessment_timestamp,
                "quality_metrics": {
                    "audio_quality_score": assessment.quality_metrics.audio_quality_score,
                    "transcription_quality_score": assessment.quality_metrics.transcription_quality_score,
                    "conversation_quality_score": assessment.quality_metrics.conversation_quality_score,
                    "overall_quality_score": assessment.quality_metrics.overall_quality_score,
                    "quality_issues": assessment.quality_metrics.quality_issues,
                    "quality_recommendations": assessment.quality_metrics.quality_recommendations
                },
                "filter_decision": assessment.filter_decision,
                "filter_reason": assessment.filter_reason
            }
            assessments_data.append(assessment_dict)

        output_data = {
            "batch_info": {
                "assessment_type": "voice_data_quality_batch",
                "processed_at": datetime.now().isoformat(),
                "assessor_version": "1.0"
            },
            "batch_statistics": batch_stats,
            "quality_thresholds": self.quality_thresholds,
            "filter_criteria": self.filter_criteria,
            "assessments": assessments_data
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Batch assessment results saved: {output_file}")
        return output_file


# Example usage
if __name__ == "__main__":
    # Initialize assessor
    assessor = VoiceDataQualityAssessment()

    # Assess single voice file
    assessment = assessor.assess_voice_data_quality("mock_voice_file.wav")


    if assessment.quality_metrics.quality_issues:
        pass
    if assessment.quality_metrics.quality_recommendations:
        pass

    # Batch assessment example
    mock_files = ["file1.wav", "file2.wav", "file3.wav"]
    batch_result = assessor.batch_assess_voice_data(mock_files)

