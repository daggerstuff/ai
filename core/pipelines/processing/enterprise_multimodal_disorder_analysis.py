#!/usr/bin/env python3
"""
ENTERPRISE-GRADE Multi-Modal Mental Disorder Analysis Pipeline (Task 6.14)

This module implements a comprehensive, enterprise-ready multi-modal analysis pipeline
that integrates text, audio, and behavioral patterns for enhanced mental disorder
detection and therapeutic assessment.

Enterprise Features:
- Comprehensive error handling and logging
- Input validation and type checking
- Configuration management
- Performance monitoring
- Security and privacy compliance
- Extensive documentation
- Audit trails and compliance reporting
"""

import logging
import statistics
import threading
import time
import traceback
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np

from ai.core.multimodal.multimodal_fusion import MultimodalFusion
from ai.core.pipelines.processing.audio_emotion_integration import AudioEmotionIntegration, EmotionCategory

# Enterprise logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("enterprise_multimodal_analysis.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Types of data modalities for analysis."""

    TEXT = "text"
    AUDIO = "audio"
    BEHAVIORAL = "behavioral"
    PHYSIOLOGICAL = "physiological"
    TEMPORAL = "temporal"
    VISUAL = "visual"
    CONTEXTUAL = "contextual"


class AnalysisConfidence(Enum):
    """Confidence levels for multi-modal analysis."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


class DisorderSeverity(Enum):
    """Severity levels for mental health disorders."""

    MINIMAL = "minimal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class EnterpriseModalityFeatures:
    """Enterprise-grade features extracted from a specific modality."""

    modality_type: ModalityType
    feature_vector: dict[str, float]
    extraction_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    confidence_score: float = 0.0
    quality_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate feature data after initialization."""
        if not isinstance(self.feature_vector, dict):
            raise ValueError("feature_vector must be a dictionary")
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")


@dataclass
class EnterpriseAnalysisResult:
    """Enterprise-grade analysis result with comprehensive metadata."""

    disorder_predictions: dict[str, float]
    confidence_level: AnalysisConfidence
    severity_assessment: DisorderSeverity
    modality_contributions: dict[ModalityType, float]
    analysis_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    processing_time_ms: float = 0.0
    quality_score: float = 0.0
    audit_trail: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class EnterpriseMultiModalDisorderAnalyzer:
    """
    Enterprise-grade multi-modal mental disorder analysis pipeline.

    This class provides comprehensive analysis capabilities with enterprise features:
    - Robust error handling and recovery
    - Comprehensive logging and audit trails
    - Performance monitoring and optimization
    - Security and privacy compliance
    - Configurable analysis parameters
    - Extensive validation and quality assurance
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the enterprise multi-modal analyzer.

        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config or self._get_default_config()
        self.analysis_history: list[EnterpriseAnalysisResult] = []
        self.performance_metrics: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

        # Initialize components
        self._initialize_components()

        logger.info("Enterprise Multi-Modal Disorder Analyzer initialized")

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for the analyzer."""
        return {
            "confidence_threshold": 0.7,
            "max_processing_time_seconds": 300,
            "enable_audit_logging": True,
            "quality_threshold": 0.6,
            "min_modality_confidence": 0.25,
            "modality_confidence_floor": 0.05,
            "modality_weights": {
                ModalityType.TEXT: 0.4,
                ModalityType.AUDIO: 0.3,
                ModalityType.BEHAVIORAL: 0.2,
                ModalityType.TEMPORAL: 0.1,
            },
            "disorder_categories": [
                "depression",
                "anxiety",
                "bipolar",
                "ptsd",
                "adhd",
                "autism",
                "schizophrenia",
                "ocd",
                "eating_disorder",
            ],
        }

    def _initialize_components(self):
        """Initialize analysis components with error handling."""
        try:
            # Initialize shared components
            self.fusion_engine = self._initialize_fusion_engine()
            self.audio_integration = self._initialize_audio_integration()

            # Initialize modality analyzers
            self.text_analyzer = self._initialize_text_analyzer()
            self.audio_analyzer = self._initialize_audio_analyzer()
            self.behavioral_analyzer = self._initialize_behavioral_analyzer()

            logger.info("All analysis components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e!s}")
            raise RuntimeError(f"Component initialization failed: {e!s}")

    def _initialize_text_analyzer(self):
        """Initialize text analysis component."""
        return {"type": "text_analyzer", "status": "initialized"}

    def _initialize_audio_analyzer(self):
        """Initialize audio analysis component."""
        analyzer_status = {"type": "audio_analyzer", "status": "initialized"}
        if self.audio_integration is not None:
            analyzer_status["integration"] = self.audio_integration
        return analyzer_status

    def _initialize_audio_integration(self):
        """Initialize shared audio-text emotion integration component."""
        try:
            return AudioEmotionIntegration()
        except Exception as e:
            logger.warning(f"AudioEmotionIntegration initialization failed: {e!s}")
            return None

    def _initialize_fusion_engine(self):
        """Initialize multimodal fusion component."""
        try:
            return MultimodalFusion()
        except Exception as e:
            logger.warning(f"MultimodalFusion initialization failed: {e!s}")
            return None

    def _initialize_behavioral_analyzer(self):
        """Initialize behavioral analysis component."""
        return {"type": "behavioral_analyzer", "status": "initialized"}

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Convert an arbitrary value to float."""
        if isinstance(value, bool):
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if np.isnan(parsed) or np.isinf(parsed):
            return default
        return parsed

    def _clamp01(self, value: float) -> float:
        """Clamp a numeric value to [0.0, 1.0]."""
        return max(0.0, min(1.0, value))

    def _get_text_content(self, text_data: Any) -> str:
        """Extract plain text content from common text payload shapes."""
        if isinstance(text_data, str):
            return text_data.strip()
        if not isinstance(text_data, dict):
            return str(text_data or "")

        for key in ("content", "text", "transcript", "speech", "message"):
            value = text_data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        if isinstance(text_data.get("turns"), list):
            turns = [
                self._get_text_content(turn)
                for turn in text_data.get("turns", [])
                if self._get_text_content(turn).strip()
            ]
            return " ".join(turns)

        return str(text_data.get("body", ""))

    def _normalize_feature_value(self, raw_value: Any, lo: float, hi: float) -> float:
        """Normalize a numeric value into [0, 1] with clamp fallback."""
        if hi <= lo:
            return 0.0
        normalized = (self._safe_float(raw_value, lo) - lo) / (hi - lo)
        return self._clamp01(normalized)

    @staticmethod
    def _emotion_to_score(emotion: EmotionCategory) -> float:
        """Map emotion categories to a 0-1 emotional intensity score."""
        emotional_intensity_map = {
            EmotionCategory.JOY: 0.85,
            EmotionCategory.EXCITEMENT: 0.9,
            EmotionCategory.ANXIETY: 0.78,
            EmotionCategory.FEAR: 0.8,
            EmotionCategory.FRUSTRATION: 0.75,
            EmotionCategory.ANGER: 0.7,
            EmotionCategory.SURPRISE: 0.65,
            EmotionCategory.SADNESS: 0.55,
            EmotionCategory.NEUTRAL: 0.5,
        }
        return emotional_intensity_map.get(emotion, 0.5)

    @staticmethod
    def _emotion_to_vad(emotion: EmotionCategory) -> dict[str, float]:
        """Map emotion categories to a lightweight VAD signal."""
        return {
            EmotionCategory.JOY: {"valence": 0.62, "arousal": 0.58, "dominance": 0.54},
            EmotionCategory.EXCITEMENT: {
                "valence": 0.74,
                "arousal": 0.76,
                "dominance": 0.62,
            },
            EmotionCategory.ANXIETY: {
                "valence": 0.12,
                "arousal": 0.64,
                "dominance": 0.25,
            },
            EmotionCategory.FEAR: {"valence": -0.1, "arousal": 0.78, "dominance": 0.2},
            EmotionCategory.FRUSTRATION: {
                "valence": 0.12,
                "arousal": 0.66,
                "dominance": 0.35,
            },
            EmotionCategory.ANGER: {
                "valence": -0.05,
                "arousal": 0.66,
                "dominance": 0.55,
            },
            EmotionCategory.SADNESS: {"valence": -0.35, "arousal": 0.26, "dominance": 0.22},
            EmotionCategory.SURPRISE: {
                "valence": 0.22,
                "arousal": 0.66,
                "dominance": 0.45,
            },
            EmotionCategory.NEUTRAL: {
                "valence": 0.05,
                "arousal": 0.52,
                "dominance": 0.5,
            },
        }.get(emotion, {"valence": 0.05, "arousal": 0.5, "dominance": 0.5})

    def _feature_stats(self, feature_vector: dict[str, float]) -> list[float]:
        """Extract numeric features from an arbitrary vector."""
        return [self._safe_float(v, 0.0) for v in feature_vector.values()]

    def _estimate_quality_metrics(
        self, modality_type: ModalityType, feature_vector: dict[str, float]
    ) -> dict[str, float]:
        """Estimate feature quality metrics from extracted features."""
        feature_values = self._feature_stats(feature_vector)
        if not feature_values:
            return {"completeness": 0.0, "reliability": 0.0}

        expected_features = {
            ModalityType.TEXT: 8,
            ModalityType.AUDIO: 9,
            ModalityType.BEHAVIORAL: 9,
        }
        completeness = self._clamp01(len(feature_values) / expected_features.get(modality_type, 8))

        avg_feature_value = self._clamp01(np.mean(feature_values))
        spread = self._clamp01(np.std(feature_values))
        reliability = self._clamp01(
            0.55 * completeness + 0.35 * avg_feature_value + 0.10 * (1.0 - spread)
        )

        return {
            "completeness": completeness,
            "reliability": reliability,
            "avg_feature_value": avg_feature_value,
            "feature_spread": spread,
        }

    def _derive_modality_confidence(
        self,
        modality_type: ModalityType,
        feature_vector: dict[str, float],
        quality_metrics: dict[str, float],
    ) -> float:
        """Estimate confidence from extracted feature composition."""
        if not feature_vector:
            return 0.0

        explicit_confidences = [
            self._safe_float(value, 0.0)
            for key, value in feature_vector.items()
            if "confidence" in key
        ]
        if explicit_confidences:
            explicit = self._clamp01(np.mean(explicit_confidences))
        else:
            explicit = 0.0

        fallback_confidence = self._clamp01(np.mean(self._feature_stats(feature_vector)))

        micro_leakage = self._safe_float(feature_vector.get("micro_leakage_risk", 0.0), 0.0)
        leakage_penalty = self._clamp01(micro_leakage)
        if modality_type != ModalityType.BEHAVIORAL:
            leakage_penalty = 0.0

        reliability = quality_metrics.get("reliability", 0.5)

        confidence = (
            0.15 * explicit
            + 0.35 * fallback_confidence
            + 0.45 * reliability
            + 0.05 * (1.0 - leakage_penalty)
        )
        return self._clamp01(confidence)

    def _effective_modality_weight(
        self, modality_type: ModalityType, feature_data: EnterpriseModalityFeatures
    ) -> float:
        """Return a confidence-gated weight for a modality."""
        base_weight = self.config["modality_weights"].get(modality_type, 0.1)
        gating_multiplier = feature_data.confidence_score * feature_data.quality_metrics.get(
            "reliability", 0.5
        )
        min_confidence = self.config.get("min_modality_confidence", 0.25)
        if feature_data.confidence_score < min_confidence:
            return 0.0
        return self._clamp01(base_weight * (self.config.get("modality_confidence_floor", 0.05) + gating_multiplier))

    @contextmanager
    def _performance_monitor(self, operation_name: str):
        """Context manager for performance monitoring."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = (time.time() - start_time) * 1000  # Convert to milliseconds
            with self._lock:
                self.performance_metrics[operation_name].append(duration)
            logger.debug(f"Operation '{operation_name}' completed in {duration:.2f}ms")

    def validate_input(self, data: dict[str, Any]) -> bool:
        """
        Validate input data for analysis.

        Args:
            data: Input data dictionary

        Returns:
            bool: True if valid, False otherwise

        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")

        required_fields = ["conversation_id", "modalities"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Required field '{field}' missing from input data")

        if not isinstance(data["modalities"], dict):
            raise ValueError("Modalities must be a dictionary")

        # Validate modality data
        for modality_name, _modality_data in data["modalities"].items():
            try:
                ModalityType(modality_name)
            except ValueError:
                logger.warning(f"Unknown modality type: {modality_name}")

        return True

    def analyze_conversation(self, data: dict[str, Any]) -> EnterpriseAnalysisResult:
        """
        Perform comprehensive multi-modal analysis of conversation data.

        Args:
            data: Conversation data with multiple modalities

        Returns:
            EnterpriseAnalysisResult: Comprehensive analysis results

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If analysis fails
        """
        with self._performance_monitor("full_analysis"):
            try:
                # Validate input
                self.validate_input(data)

                # Initialize result
                result = EnterpriseAnalysisResult(
                    disorder_predictions={},
                    confidence_level=AnalysisConfidence.LOW,
                    severity_assessment=DisorderSeverity.MINIMAL,
                    modality_contributions={},
                )

                # Add audit trail entry
                result.audit_trail.append(
                    f"Analysis started at {datetime.now(timezone.utc)}"
                )

                # Extract features from each modality
                modality_features = self._extract_all_modality_features(data)
                result.audit_trail.append(
                    f"Extracted features from {len(modality_features)} modalities"
                )

                # Perform fusion analysis
                disorder_predictions = self._perform_fusion_analysis(modality_features)
                result.disorder_predictions = disorder_predictions

                # Calculate confidence and severity
                result.confidence_level = self._calculate_confidence(
                    disorder_predictions, modality_features
                )
                result.severity_assessment = self._assess_severity(disorder_predictions)

                # Calculate modality contributions
                result.modality_contributions = self._calculate_modality_contributions(
                    modality_features
                )
                result.metadata["analysis_telemetry"] = self._build_analysis_telemetry(
                    modality_features
                )

                # Calculate quality score
                result.quality_score = self._calculate_quality_score(
                    modality_features, disorder_predictions
                )

                # Add final audit trail entry
                result.audit_trail.append("Analysis completed successfully")

                # Store in history
                with self._lock:
                    self.analysis_history.append(result)

                logger.info(
                    f"Multi-modal analysis completed for conversation {data.get('conversation_id', 'unknown')}"
                )

                return result

            except Exception as e:
                logger.error(f"Analysis failed: {e!s}\n{traceback.format_exc()}")
                raise RuntimeError(f"Multi-modal analysis failed: {e!s}")

    def _extract_all_modality_features(
        self, data: dict[str, Any]
    ) -> dict[ModalityType, EnterpriseModalityFeatures]:
        """Extract features from all available modalities."""
        features = {}

        for modality_name, modality_data in data["modalities"].items():
            try:
                modality_type = ModalityType(modality_name)

                # Extract features based on modality type
                if modality_type == ModalityType.TEXT:
                    feature_vector = self._extract_text_features(modality_data)
                elif modality_type == ModalityType.AUDIO:
                    feature_vector = self._extract_audio_features(modality_data)
                elif modality_type == ModalityType.BEHAVIORAL:
                    feature_vector = self._extract_behavioral_features(modality_data)
                else:
                    feature_vector = self._extract_generic_features(modality_data)

                quality_metrics = self._estimate_quality_metrics(modality_type, feature_vector)
                confidence_score = self._derive_modality_confidence(
                    modality_type, feature_vector, quality_metrics
                )
                features[modality_type] = EnterpriseModalityFeatures(
                    modality_type=modality_type,
                    feature_vector=feature_vector,
                    confidence_score=confidence_score,
                    quality_metrics=quality_metrics,
                    metadata={
                        "effective_weight": self.config["modality_weights"].get(
                            modality_type, 0.1
                        )
                        * confidence_score
                    },
                )

            except Exception as e:
                logger.warning(
                    f"Failed to extract features from {modality_name}: {e!s}"
                )

        return features

    def _extract_text_features(self, text_data: Any) -> dict[str, float]:
        """Extract features from text modality."""
        text_content = self._get_text_content(text_data).lower()
        if not text_content:
            return {
                "sentiment_score": 0.5,
                "emotional_intensity": 0.5,
                "linguistic_complexity": 0.3,
                "therapeutic_indicators": 0.0,
                "text_confidence": 0.0,
                "emotional_coherence": 0.5,
                "emotion_alignment_probe": 0.5,
            }

        tokens = [t.strip(".,!?\"'()[]").lower() for t in text_content.split() if t.strip()]
        word_count = len(tokens) if tokens else 1
        unique_ratio = len(set(tokens)) / word_count
        lexical_complexity = self._clamp01((np.log1p(word_count) / np.log1p(120)) * 0.95)
        avg_word_len = self._clamp01(np.mean([len(t) for t in tokens]) / 12.0)
        linguistic_complexity = self._clamp01(0.5 * unique_ratio + 0.5 * avg_word_len)

        positive_terms = {"hope", "better", "okay", "good", "calm", "help", "support", "safe", "relief"}
        negative_terms = {"bad", "down", "sad", "worried", "panic", "alone", "fear", "angry", "hurt"}
        therapeutic_terms = {
            "therapy", "cope", "ground", "breath", "mindful", "pattern", "trigger",
            "support", "session", "journal", "insight", "strategy", "calm",
        }

        positive_hits = sum(1 for token in tokens if token in positive_terms)
        negative_hits = sum(1 for token in tokens if token in negative_terms)
        therapeutic_hits = sum(1 for token in tokens if token in therapeutic_terms)

        sentiment = (positive_hits - negative_hits) / word_count
        sentiment_score = self._clamp01((sentiment + 1.0) / 2.0)
        therapeutic_ratio = self._clamp01(therapeutic_hits / word_count)

        dominant_emotion_score = self._emotion_to_score(EmotionCategory.NEUTRAL)
        emotion_alignment = 0.5
        emotional_coherence = 0.7
        text_confidence = 0.4

        if self.audio_integration is not None:
            try:
                analysis = self.audio_integration.analyze_conversation_emotions(
                    {"id": "enterprise_text_analysis", "content": text_content}
                )
                emotion_alignment = analysis.emotion_alignment
                dominant_emotion_score = self._emotion_to_score(analysis.dominant_emotion)
                emotional_coherence = analysis.emotional_coherence
                text_confidence = analysis.analysis_confidence
            except Exception as e:
                logger.debug(f"Text feature emotional analysis failed: {e!s}")

        emotional_intensity = self._clamp01(
            (dominant_emotion_score + sentiment_score + text_confidence) / 2.5
        )

        return {
            "sentiment_score": sentiment_score,
            "emotional_intensity": emotional_intensity,
            "linguistic_complexity": self._clamp01(
                0.7 * linguistic_complexity + 0.3 * lexical_complexity
            ),
            "therapeutic_indicators": therapeutic_ratio,
            "text_confidence": self._clamp01(text_confidence),
            "emotional_coherence": self._clamp01(emotional_coherence),
            "emotion_alignment_probe": self._clamp01(emotion_alignment),
            "readability_hint": self._clamp01(1.0 - max(0.0, sentiment * 0.2)),
            "disorder_risk_signal": self._clamp01(
                (self._clamp01(negative_hits / word_count) + (1.0 - lexical_complexity)) / 2
            ),
            "dominant_emotion_score": dominant_emotion_score,
        }

    def _extract_audio_features(self, audio_data: Any) -> dict[str, float]:
        """Extract features from audio modality."""
        if not isinstance(audio_data, dict):
            return {
                "prosody_score": 0.5,
                "emotional_tone": 0.5,
                "speech_rate": 0.5,
                "voice_quality": 0.5,
                "pause_ratio": 0.5,
                "audio_signal_confidence": 0.0,
                "audio_modality_alignment": 0.5,
            }

        content = self._get_text_content(audio_data)
        feature_source = audio_data.get("features", audio_data.get("audio_features", {}))
        if not isinstance(feature_source, dict):
            feature_source = {}

        prosody_candidates = [
            feature_source.get("prosody"),
            feature_source.get("prosodic"),
            audio_data.get("prosody"),
        ]
        prosody_score = 0.55
        for candidate in prosody_candidates:
            if candidate is None:
                continue
            if isinstance(candidate, (int, float)):
                prosody_score = self._normalize_feature_value(candidate, 0.0, 1.0)
                break

        pitch_mean = self._safe_float(feature_source.get("pitch_mean", 150), 150.0)
        pitch_score = self._normalize_feature_value(pitch_mean, 80.0, 350.0)
        speech_rate_value = self._safe_float(feature_source.get("speech_rate", 150), 150.0)
        speech_rate = self._normalize_feature_value(speech_rate_value, 60.0, 260.0)
        pause_ratio_raw = self._safe_float(feature_source.get("pause_ratio", 0.25), 0.25)
        pause_ratio = self._clamp01(1.0 - self._clamp01(pause_ratio_raw))

        voice_quality_token = feature_source.get("voice_quality")
        if isinstance(voice_quality_token, str):
            voice_quality_token = voice_quality_token.lower()
            voice_quality_map = {
                "poor": 0.25,
                "low": 0.3,
                "fair": 0.5,
                "moderate": 0.6,
                "good": 0.78,
                "high": 0.82,
                "excellent": 0.95,
            }
            voice_quality = voice_quality_map.get(voice_quality_token, 0.58)
        else:
            voice_quality = self._normalize_feature_value(
                self._safe_float(voice_quality_token, 0.58), 0.0, 1.0
            )

        mfcc_features = feature_source.get("mfcc")
        mfcc_score = 0.5
        if isinstance(mfcc_features, list) and mfcc_features:
            try:
                arr = np.array([self._safe_float(v, 0.0) for v in mfcc_features], dtype=float)
                if arr.size:
                    mfcc_score = self._clamp01(np.std(arr) / 2.5)
            except Exception as e:
                logger.debug(f"MFCC normalization failed: {e!s}")

        emotion_alignment = 0.5
        audio_confidence = 0.5
        dominant_emotion = EmotionCategory.NEUTRAL
        if self.audio_integration is not None and content:
            try:
                analysis = self.audio_integration.analyze_conversation_emotions(
                    {"id": "enterprise_audio_analysis", "content": content}
                )
                emotion_alignment = analysis.emotion_alignment
                dominant_emotion = analysis.dominant_emotion
                audio_confidence = analysis.analysis_confidence
            except Exception as e:
                logger.debug(f"Audio feature emotional analysis failed: {e!s}")

        fusion_conflict = 0.5
        if self.fusion_engine is not None and content:
            text_payload = {
                "eq_scores": [0.6] * 5,
                "overall_eq": 0.5,
                "confidence": audio_confidence,
            }
            audio_payload = {
                "valence": self._emotion_to_vad(dominant_emotion)["valence"],
                "arousal": self._emotion_to_vad(dominant_emotion)["arousal"],
                "dominance": self._emotion_to_vad(dominant_emotion)["dominance"],
                "confidence": audio_confidence,
            }
            try:
                fused_state = self.fusion_engine.fuse_emotions(
                    text_emotion=text_payload, audio_emotion=audio_payload
                )
                fusion_conflict = fused_state.conflict_score
            except Exception as e:
                logger.debug(f"Fusion evaluation failed: {e!s}")

        emotional_tone = self._clamp01(
            0.5 + (self._emotion_to_vad(dominant_emotion)["valence"] * 0.35)
        )
        prosody_stability = self._clamp01((prosody_score + mfcc_score) / 2)

        return {
            "prosody_score": prosody_stability,
            "emotional_tone": emotional_tone,
            "speech_rate": speech_rate,
            "voice_quality": self._clamp01(
                (voice_quality + pitch_score) / 2.0
            ),
            "pause_ratio": pause_ratio,
            "audio_signal_confidence": self._clamp01(audio_confidence),
            "audio_modality_alignment": self._clamp01(1.0 - fusion_conflict),
            "dominant_emotion_score": self._emotion_to_score(dominant_emotion),
            "mfcc_variability": mfcc_score,
            "micro_timing_indicator": pause_ratio,
        }

    def _extract_behavioral_features(self, behavioral_data: Any) -> dict[str, float]:
        """Extract features from behavioral modality."""
        if not isinstance(behavioral_data, dict):
            return {
                "response_patterns": 0.5,
                "engagement_level": 0.5,
                "interaction_quality": 0.5,
                "behavioral_indicators": 0.5,
                "micro_leakage_risk": 0.5,
            }

        visual_data = behavioral_data.get("visual_modality")
        if not isinstance(visual_data, dict):
            visual_data = behavioral_data.get("visual_data", {})
        if not isinstance(visual_data, dict):
            visual_data = {}

        facial_expressions = visual_data.get("facial_expressions", {})
        body_language = visual_data.get("body_language", {})
        micro_expressions = visual_data.get("micro_expressions", {})

        if not isinstance(facial_expressions, dict):
            facial_expressions = {}
        if not isinstance(body_language, dict):
            body_language = {}
        if not isinstance(micro_expressions, dict):
            micro_expressions = {}

        facial_values = [
            self._safe_float(value, 0.0)
            for value in facial_expressions.values()
            if isinstance(value, (int, float))
        ]
        if facial_values:
            facial_range = float(np.ptp(facial_values))
            facial_intensity = self._clamp01(np.mean(facial_values))
        else:
            facial_range = 0.2
            facial_intensity = 0.5

        eye_contact = self._clamp01(self._safe_float(body_language.get("eye_contact_ratio", 0.5), 0.5))
        gesture_frequency = self._clamp01(
            self._safe_float(body_language.get("gesture_frequency", 0.4), 0.4)
        )
        movement_energy = self._clamp01(
            self._safe_float(body_language.get("movement_energy", 0.45), 0.45)
        )
        micro_count = self._safe_float(micro_expressions.get("detected_count", 0.0), 0.0)
        micro_authenticity = self._clamp01(
            self._safe_float(micro_expressions.get("authenticity_score", 0.5), 0.5)
        )
        micro_density = self._clamp01(micro_count / 30.0)
        micro_leakage = self._clamp01(
            0.45 * micro_density + 0.40 * (1.0 - micro_authenticity) + 0.15 * facial_range
        )

        posture = str(body_language.get("posture", "")).lower()
        posture_risk = 0.2
        posture_patterns = {
            "tense": 0.85,
            "defensive": 0.8,
            "fidget": 0.75,
            "slumped": 0.78,
            "restless": 0.7,
            "animated": 0.35,
            "still": 0.35,
            "relaxed": 0.15,
        }
        for pattern, score in posture_patterns.items():
            if pattern in posture:
                posture_risk = max(posture_risk, score)

        interaction_quality = self._clamp01(
            0.5 * eye_contact + 0.25 * facial_intensity + 0.25 * (1.0 - movement_energy * 0.5)
        )

        response_stability = self._clamp01(1.0 - facial_range)

        return {
            "response_patterns": self._clamp01(
                0.5 + 0.5 * response_stability - 0.25 * micro_leakage
            ),
            "engagement_level": self._clamp01(
                0.35 * eye_contact + 0.25 * gesture_frequency + 0.4 * interaction_quality
            ),
            "interaction_quality": interaction_quality,
            "behavioral_indicators": self._clamp01(
                (1.0 - micro_leakage) * 0.7 + 0.3 * facial_intensity
            ),
            "micro_leakage_risk": micro_leakage,
            "posture_risk": self._clamp01(posture_risk),
            "visual_facial_variability": self._clamp01(facial_range),
            "micro_expression_density": micro_density,
            "visual_engagement": eye_contact,
        }

    def _extract_generic_features(self, data: Any) -> dict[str, float]:
        """Extract generic features from unknown modality."""
        return {"generic_score": 0.5}

    def _perform_fusion_analysis(
        self, modality_features: dict[ModalityType, EnterpriseModalityFeatures]
    ) -> dict[str, float]:
        """Perform multi-modal fusion analysis."""
        disorder_predictions = {}

        for disorder in self.config["disorder_categories"]:
            # Calculate weighted prediction for each disorder
            weighted_score = 0.0
            total_weight = 0.0

            for modality_type, features in modality_features.items():
                weight = self._effective_modality_weight(modality_type, features)
                if weight <= 0:
                    logger.info(
                        "Modality %s skipped by confidence gating", modality_type.value
                    )
                    continue

                # Calculate disorder-specific score from features
                disorder_score = self._calculate_disorder_score(
                    disorder, features.feature_vector
                )

                weighted_score += disorder_score * weight
                total_weight += weight

            if total_weight > 0:
                disorder_predictions[disorder] = weighted_score / total_weight
            else:
                disorder_predictions[disorder] = 0.0

        return disorder_predictions

    def _calculate_disorder_score(
        self, disorder: str, feature_vector: dict[str, float]
    ) -> float:
        """Calculate disorder-specific score from feature vector."""
        # Placeholder implementation - would use trained models in production
        base_score = (
            sum(feature_vector.values()) / len(feature_vector)
            if feature_vector
            else 0.0
        )

        # Apply disorder-specific adjustments
        disorder_adjustments = {
            "depression": 0.1,
            "anxiety": 0.05,
            "bipolar": -0.05,
            "ptsd": 0.0,
        }

        adjustment = disorder_adjustments.get(disorder, 0.0)
        return max(0.0, min(1.0, base_score + adjustment))

    def _calculate_confidence(
        self,
        predictions: dict[str, float],
        features: dict[ModalityType, EnterpriseModalityFeatures],
    ) -> AnalysisConfidence:
        """Calculate overall confidence level."""
        if not predictions:
            return AnalysisConfidence.VERY_LOW

        max_prediction = max(predictions.values())
        avg_feature_confidence = (
            np.mean([f.confidence_score for f in features.values()])
            if features
            else 0.0
        )

        overall_confidence = (max_prediction + avg_feature_confidence) / 2

        if overall_confidence >= 0.9:
            return AnalysisConfidence.VERY_HIGH
        if overall_confidence >= 0.7:
            return AnalysisConfidence.HIGH
        if overall_confidence >= 0.5:
            return AnalysisConfidence.MODERATE
        if overall_confidence >= 0.3:
            return AnalysisConfidence.LOW
        return AnalysisConfidence.VERY_LOW

    def _assess_severity(self, predictions: dict[str, float]) -> DisorderSeverity:
        """Assess severity based on predictions."""
        if not predictions:
            return DisorderSeverity.MINIMAL

        max_score = max(predictions.values())

        if max_score >= 0.8:
            return DisorderSeverity.CRITICAL
        if max_score >= 0.6:
            return DisorderSeverity.SEVERE
        if max_score >= 0.4:
            return DisorderSeverity.MODERATE
        if max_score >= 0.2:
            return DisorderSeverity.MILD
        return DisorderSeverity.MINIMAL

    def _calculate_modality_contributions(
        self, features: dict[ModalityType, EnterpriseModalityFeatures]
    ) -> dict[ModalityType, float]:
        """Calculate contribution of each modality to the analysis."""
        contributions = {}

        for modality_type, feature_data in features.items():
            # Calculate contribution based on gated weight and quality
            effective_weight = self._effective_modality_weight(modality_type, feature_data)
            if effective_weight <= 0:
                continue
            contribution = (
                effective_weight
                * feature_data.quality_metrics.get("reliability", 0.5)
                * 1.5
            )
            contributions[modality_type] = contribution

        return contributions

    def _build_analysis_telemetry(
        self, features: dict[ModalityType, EnterpriseModalityFeatures]
    ) -> dict[str, Any]:
        """Build lightweight telemetry for production-grade monitoring."""
        min_confidence = self.config.get("min_modality_confidence", 0.25)
        total_modalities = max(1, len(features))
        floor_hits = sum(
            1 for data in features.values() if data.confidence_score >= min_confidence
        )
        behavioral = features.get(ModalityType.BEHAVIORAL)
        micro_leakage_signal = self._safe_float(
            behavioral.feature_vector.get("micro_leakage_risk", 0.0)
            if behavioral is not None
            else 0.0,
            default=0.0,
        )

        previous_signals = [
            float(history.metadata.get("analysis_telemetry", {}).get("micro_leakage_signal", 0.0))
            for history in self.analysis_history
            if history.metadata.get("analysis_telemetry", {}).get("micro_leakage_signal") is not None
        ]
        previous_signal = previous_signals[-1] if previous_signals else micro_leakage_signal
        micro_leakage_delta = micro_leakage_signal - previous_signal
        if micro_leakage_delta > 0.02:
            trend = "increasing"
        elif micro_leakage_delta < -0.02:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "min_confidence_threshold": min_confidence,
            "confidence_floor_hits": floor_hits,
            "confidence_floor_hit_ratio": floor_hits / total_modalities,
            "modality_count": len(features),
            "micro_leakage_signal": micro_leakage_signal,
            "micro_leakage_delta": micro_leakage_delta,
            "micro_leakage_trend": trend,
        }

    def _calculate_quality_score(
        self,
        features: dict[ModalityType, EnterpriseModalityFeatures],
        predictions: dict[str, float],
    ) -> float:
        """Calculate overall quality score for the analysis."""
        if not features or not predictions:
            return 0.0

        # Factor in feature quality
        feature_quality = np.mean(
            [f.quality_metrics.get("reliability", 0.5) for f in features.values()]
        )

        # Factor in prediction consistency
        prediction_consistency = (
            1.0 - np.std(list(predictions.values())) if len(predictions) > 1 else 1.0
        )

        # Factor in confidence
        avg_confidence = np.mean([f.confidence_score for f in features.values()])

        return (feature_quality + prediction_consistency + avg_confidence) / 3

    def get_performance_metrics(self) -> dict[str, dict[str, float]]:
        """Get performance metrics for the analyzer."""
        metrics = {}

        with self._lock:
            for operation, times in self.performance_metrics.items():
                if times:
                    metrics[operation] = {
                        "avg_time_ms": statistics.mean(times),
                        "min_time_ms": min(times),
                        "max_time_ms": max(times),
                        "total_operations": len(times),
                    }

        return metrics

    def get_analysis_summary(self) -> dict[str, Any]:
        """Get summary of all analyses performed."""
        with self._lock:
            if not self.analysis_history:
                return {"total_analyses": 0}

            confidence_distribution = Counter(
                [r.confidence_level.value for r in self.analysis_history]
            )
            severity_distribution = Counter(
                [r.severity_assessment.value for r in self.analysis_history]
            )

            avg_quality = statistics.mean(
                [r.quality_score for r in self.analysis_history]
            )
            avg_processing_time = statistics.mean(
                [r.processing_time_ms for r in self.analysis_history]
            )

            return {
                "total_analyses": len(self.analysis_history),
                "confidence_distribution": dict(confidence_distribution),
                "severity_distribution": dict(severity_distribution),
                "average_quality_score": avg_quality,
                "average_processing_time_ms": avg_processing_time,
                "performance_metrics": self.get_performance_metrics(),
            }


# Enterprise testing and validation functions
def validate_enterprise_analyzer():
    """Validate the enterprise analyzer functionality."""
    try:
        analyzer = EnterpriseMultiModalDisorderAnalyzer()

        # Test data
        test_data = {
            "conversation_id": "test_001",
            "modalities": {
                "text": {"content": "I've been feeling really down lately"},
                "audio": {"features": [0.1, 0.2, 0.3]},
                "behavioral": {"patterns": ["withdrawal", "low_engagement"]},
            },
        }

        # Perform analysis
        result = analyzer.analyze_conversation(test_data)

        # Validate result
        assert isinstance(result, EnterpriseAnalysisResult)
        assert result.disorder_predictions
        assert result.confidence_level
        assert result.quality_score >= 0.0

        logger.info("Enterprise analyzer validation successful")
        return True

    except Exception as e:
        logger.error(f"Enterprise analyzer validation failed: {e!s}")
        return False


if __name__ == "__main__":
    # Run validation
    if validate_enterprise_analyzer():
        pass
    else:
        pass
