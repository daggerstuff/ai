#!/usr/bin/env python3
"""
Task 6.14: Multi-Modal Mental Disorder Analysis Pipeline (MODMA)

This module implements a comprehensive multi-modal analysis pipeline that
integrates text, audio, and behavioral patterns for enhanced mental disorder
detection and therapeutic assessment.

Strategic Goal: Create the most comprehensive mental health analysis system
by combining multiple data modalities for superior diagnostic accuracy.
"""

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

# Import our existing components
from dataset_pipeline.audio_emotion_integration import (
    AudioEmotionFeatures,
    MultiModalTherapeuticAnalyzer,
)
from dataset_pipeline.ecosystem.condition_pattern_recognition import MentalHealthCondition


class ModalityType(Enum):
    """Types of data modalities for analysis."""
    TEXT = "text"
    AUDIO = "audio"
    BEHAVIORAL = "behavioral"
    PHYSIOLOGICAL = "physiological"
    TEMPORAL = "temporal"


class AnalysisConfidence(Enum):
    """Confidence levels for multi-modal analysis."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ModalityFeatures:
    """Features extracted from a specific modality."""
    modality_type: ModalityType
    feature_vector: dict[str, float]
    confidence_score: float
    quality_indicators: dict[str, float]
    extraction_metadata: dict[str, Any]


@dataclass
class MultiModalDisorderAnalysis:
    """Comprehensive multi-modal disorder analysis result."""
    conversation_id: str
    primary_disorder: MentalHealthCondition
    disorder_confidence: float
    modality_contributions: dict[str, float]
    cross_modal_consistency: float
    feature_fusion_results: dict[str, Any]
    diagnostic_evidence: dict[str, list[str]]
    severity_assessment: dict[str, Any]
    treatment_recommendations: list[str]
    analysis_confidence: AnalysisConfidence
    timestamp: str


class BehavioralPatternAnalyzer:
    """Analyzes behavioral patterns from conversation data."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Behavioral pattern indicators
        self.behavioral_patterns = {
            "communication_style": {
                "verbose": r"(very|really|extremely|quite|pretty)\s+\w+",
                "concise": r"^.{1,50}$",
                "repetitive": r"(\b\w+\b).*\1",
                "tangential": r"(but|however|although|anyway|speaking of)"
            },
            "emotional_regulation": {
                "stable": r"(calm|steady|balanced|consistent)",
                "labile": r"(up and down|back and forth|all over|mood swing)",
                "suppressed": r"(fine|okay|whatever|doesn\'t matter)",
                "intense": r"(overwhelming|intense|extreme|unbearable)"
            },
            "social_interaction": {
                "engaged": r"(we|us|together|relationship|connect)",
                "withdrawn": r"(alone|isolated|nobody|no one|by myself)",
                "conflicted": r"(argue|fight|conflict|disagree|tension)",
                "supportive": r"(help|support|care|understand|there for)"
            },
            "cognitive_patterns": {
                "organized": r"(first|second|then|next|finally|plan)",
                "scattered": r"(confused|mixed up|can\'t think|all over)",
                "ruminating": r"(keep thinking|can\'t stop|over and over)",
                "decisive": r"(decided|sure|certain|clear|definite)"
            }
        }

    def analyze_behavioral_patterns(self, conversation: dict[str, Any]) -> ModalityFeatures:
        """Analyze behavioral patterns from conversation text."""
        text = self._extract_conversation_text(conversation)

        # Extract behavioral features
        feature_vector = {}

        for pattern_category, patterns in self.behavioral_patterns.items():
            category_scores = {}

            for pattern_name, pattern_regex in patterns.items():
                import re
                matches = len(re.findall(pattern_regex, text, re.IGNORECASE))
                category_scores[pattern_name] = min(matches * 0.2, 1.0)

            # Aggregate category score
            feature_vector[pattern_category] = statistics.mean(category_scores.values())

        # Calculate overall confidence
        confidence = min(sum(feature_vector.values()) / len(feature_vector), 1.0)

        # Quality indicators
        quality_indicators = {
            "text_length": len(text),
            "pattern_diversity": len([v for v in feature_vector.values() if v > 0.1]),
            "feature_completeness": len(feature_vector) / len(self.behavioral_patterns)
        }

        return ModalityFeatures(
            modality_type=ModalityType.BEHAVIORAL,
            feature_vector=feature_vector,
            confidence_score=confidence,
            quality_indicators=quality_indicators,
            extraction_metadata={
                "analysis_method": "pattern_matching",
                "text_length": len(text),
                "timestamp": datetime.now().isoformat()
            }
        )

    def _extract_conversation_text(self, conversation: dict[str, Any]) -> str:
        """Extract text content from conversation structure."""
        text_parts = []

        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict) and "content" in message:
                    text_parts.append(message["content"])
        elif "input" in conversation and "output" in conversation:
            text_parts.extend([conversation["input"], conversation["output"]])
        elif "text" in conversation:
            text_parts.append(conversation["text"])

        return " ".join(text_parts).lower()


class TemporalPatternAnalyzer:
    """Analyzes temporal patterns in conversation data."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Temporal pattern indicators
        self.temporal_patterns = {
            "time_references": {
                "past_focused": r"(was|were|used to|before|previously|back then)",
                "present_focused": r"(now|currently|today|right now|at the moment)",
                "future_focused": r"(will|going to|plan to|hope to|want to)",
                "time_distorted": r"(feels like forever|time flies|lost track)"
            },
            "progression_patterns": {
                "improving": r"(better|improving|getting easier|progress)",
                "deteriorating": r"(worse|getting harder|declining|falling apart)",
                "cyclical": r"(comes and goes|up and down|good days bad days)",
                "stable": r"(same|consistent|steady|unchanged)"
            },
            "urgency_indicators": {
                "immediate": r"(right now|urgent|emergency|can\'t wait)",
                "gradual": r"(slowly|gradually|over time|little by little)",
                "episodic": r"(sometimes|occasionally|episodes|attacks)",
                "chronic": r"(always|constantly|all the time|never stops)"
            }
        }

    def analyze_temporal_patterns(self, conversation: dict[str, Any]) -> ModalityFeatures:
        """Analyze temporal patterns from conversation text."""
        text = self._extract_conversation_text(conversation)

        # Extract temporal features
        feature_vector = {}

        for pattern_category, patterns in self.temporal_patterns.items():
            category_scores = {}

            for pattern_name, pattern_regex in patterns.items():
                import re
                matches = len(re.findall(pattern_regex, text, re.IGNORECASE))
                category_scores[pattern_name] = min(matches * 0.25, 1.0)

            # Aggregate category score
            feature_vector[pattern_category] = statistics.mean(category_scores.values())

        # Calculate temporal coherence
        feature_vector["temporal_coherence"] = self._calculate_temporal_coherence(text)

        # Calculate overall confidence
        confidence = min(sum(feature_vector.values()) / len(feature_vector), 1.0)

        # Quality indicators
        quality_indicators = {
            "temporal_markers": sum(1 for v in feature_vector.values() if v > 0.2),
            "coherence_score": feature_vector["temporal_coherence"],
            "pattern_strength": max(feature_vector.values())
        }

        return ModalityFeatures(
            modality_type=ModalityType.TEMPORAL,
            feature_vector=feature_vector,
            confidence_score=confidence,
            quality_indicators=quality_indicators,
            extraction_metadata={
                "analysis_method": "temporal_pattern_matching",
                "coherence_analysis": True,
                "timestamp": datetime.now().isoformat()
            }
        )

    def _extract_conversation_text(self, conversation: dict[str, Any]) -> str:
        """Extract text content from conversation structure."""
        text_parts = []

        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict) and "content" in message:
                    text_parts.append(message["content"])

        return " ".join(text_parts).lower()

    def _calculate_temporal_coherence(self, text: str) -> float:
        """Calculate temporal coherence of the narrative."""
        # Look for temporal connectors and logical flow
        coherence_indicators = [
            r"(then|next|after|before|while|during|since)",
            r"(first|second|finally|last|eventually)",
            r"(when|as|until|once|whenever)"
        ]

        import re
        coherence_score = 0.0

        for pattern in coherence_indicators:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            coherence_score += min(matches * 0.1, 0.3)

        return min(coherence_score, 1.0)


class MultiModalFeatureFusion:
    """Fuses features from multiple modalities for comprehensive analysis."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Fusion weights for different modalities
        self.modality_weights = {
            ModalityType.TEXT: 0.35,
            ModalityType.AUDIO: 0.25,
            ModalityType.BEHAVIORAL: 0.20,
            ModalityType.TEMPORAL: 0.20
        }

        # Disorder-specific modality importance
        self.disorder_modality_importance = {
            MentalHealthCondition.DEPRESSION: {
                ModalityType.AUDIO: 0.4,  # Voice patterns important
                ModalityType.TEMPORAL: 0.3,  # Time perception affected
                ModalityType.TEXT: 0.2,
                ModalityType.BEHAVIORAL: 0.1
            },
            MentalHealthCondition.ANXIETY: {
                ModalityType.AUDIO: 0.35,  # Voice tremor, rate
                ModalityType.BEHAVIORAL: 0.3,  # Avoidance patterns
                ModalityType.TEXT: 0.25,
                ModalityType.TEMPORAL: 0.1
            },
            MentalHealthCondition.BIPOLAR: {
                ModalityType.TEMPORAL: 0.4,  # Episodic nature
                ModalityType.BEHAVIORAL: 0.3,  # Mood-dependent behavior
                ModalityType.AUDIO: 0.2,
                ModalityType.TEXT: 0.1
            }
        }

    def fuse_modality_features(self, modality_features: list[ModalityFeatures],
                              target_disorder: MentalHealthCondition | None = None) -> dict[str, Any]:
        """Fuse features from multiple modalities."""

        # Get appropriate weights
        if target_disorder and target_disorder in self.disorder_modality_importance:
            weights = self.disorder_modality_importance[target_disorder]
        else:
            weights = self.modality_weights

        # Organize features by modality
        features_by_modality = {
            feature.modality_type: feature for feature in modality_features
        }

        # Calculate weighted fusion
        fused_features = {}
        total_confidence = 0.0
        total_weight = 0.0

        for modality_type, weight in weights.items():
            if modality_type in features_by_modality:
                feature = features_by_modality[modality_type]

                # Weight the features
                for feature_name, feature_value in feature.feature_vector.items():
                    weighted_name = f"{modality_type.value}_{feature_name}"
                    fused_features[weighted_name] = feature_value * weight

                # Weight the confidence
                total_confidence += feature.confidence_score * weight
                total_weight += weight

        # Calculate cross-modal consistency
        consistency_score = self._calculate_cross_modal_consistency(modality_features)

        # Calculate overall fusion confidence
        fusion_confidence = total_confidence / total_weight if total_weight > 0 else 0.0

        return {
            "fused_features": fused_features,
            "fusion_confidence": fusion_confidence,
            "cross_modal_consistency": consistency_score,
            "modality_contributions": {
                modality.value: weights.get(modality, 0.0)
                for modality in ModalityType
            },
            "feature_count": len(fused_features),
            "modalities_used": [f.modality_type.value for f in modality_features]
        }

    def _calculate_cross_modal_consistency(self, modality_features: list[ModalityFeatures]) -> float:
        """Calculate consistency across different modalities."""
        if len(modality_features) < 2:
            return 1.0

        # Compare confidence scores across modalities
        confidences = [f.confidence_score for f in modality_features]
        confidence_variance = statistics.variance(confidences) if len(confidences) > 1 else 0.0

        # Lower variance = higher consistency
        return max(0.0, 1.0 - confidence_variance)



class MODMAAnalyzer:
    """Main Multi-Modal Mental Disorder Analysis system."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize component analyzers
        self.multimodal_analyzer = MultiModalTherapeuticAnalyzer()
        self.behavioral_analyzer = BehavioralPatternAnalyzer()
        self.temporal_analyzer = TemporalPatternAnalyzer()
        self.feature_fusion = MultiModalFeatureFusion()

        # Analysis tracking
        self.analysis_stats = {
            "total_analyses": 0,
            "modality_usage": defaultdict(int),
            "disorder_detections": defaultdict(int),
            "confidence_scores": []
        }

    def analyze_conversation(self, conversation: dict[str, Any]) -> MultiModalDisorderAnalysis:
        """Perform comprehensive multi-modal disorder analysis."""
        conversation_id = conversation.get("id", "unknown")

        # Step 1: Multi-modal therapeutic analysis (text + audio)
        multimodal_result = self.multimodal_analyzer.analyze_conversation(conversation)

        # Step 2: Behavioral pattern analysis
        behavioral_features = self.behavioral_analyzer.analyze_behavioral_patterns(conversation)

        # Step 3: Temporal pattern analysis
        temporal_features = self.temporal_analyzer.analyze_temporal_patterns(conversation)

        # Step 4: Extract modality features
        modality_features = [behavioral_features, temporal_features]

        # Add audio features if available
        if hasattr(multimodal_result, "audio_analysis"):
            audio_features = self._convert_audio_to_modality_features(multimodal_result.audio_analysis)
            modality_features.append(audio_features)

        # Step 5: Determine primary disorder from text analysis
        primary_disorder = MentalHealthCondition(
            multimodal_result.text_analysis["condition_recognition"]["primary_condition"]
        )

        # Step 6: Feature fusion
        fusion_results = self.feature_fusion.fuse_modality_features(
            modality_features, primary_disorder
        )

        # Step 7: Calculate disorder confidence
        disorder_confidence = self._calculate_disorder_confidence(
            multimodal_result, fusion_results
        )

        # Step 8: Generate diagnostic evidence
        diagnostic_evidence = self._generate_diagnostic_evidence(
            multimodal_result, modality_features
        )

        # Step 9: Severity assessment
        severity_assessment = self._assess_multimodal_severity(
            multimodal_result, modality_features
        )

        # Step 10: Treatment recommendations
        treatment_recommendations = self._generate_treatment_recommendations(
            primary_disorder, multimodal_result, modality_features
        )

        # Step 11: Determine analysis confidence
        analysis_confidence = self._determine_analysis_confidence(
            disorder_confidence, fusion_results["cross_modal_consistency"]
        )

        analysis = MultiModalDisorderAnalysis(
            conversation_id=conversation_id,
            primary_disorder=primary_disorder,
            disorder_confidence=disorder_confidence,
            modality_contributions=fusion_results["modality_contributions"],
            cross_modal_consistency=fusion_results["cross_modal_consistency"],
            feature_fusion_results=fusion_results,
            diagnostic_evidence=diagnostic_evidence,
            severity_assessment=severity_assessment,
            treatment_recommendations=treatment_recommendations,
            analysis_confidence=analysis_confidence,
            timestamp=datetime.now().isoformat()
        )

        # Update statistics
        self._update_analysis_stats(analysis)

        return analysis

    def _convert_audio_to_modality_features(self, audio_analysis: AudioEmotionFeatures) -> ModalityFeatures:
        """Convert audio analysis to modality features format."""
        feature_vector = {
            "dominant_emotion": audio_analysis.emotion_confidence,
            "emotional_stability": audio_analysis.emotional_stability,
            "speech_rate": audio_analysis.speech_rate,
            "intensity_level": {
                "low": 0.25, "moderate": 0.5, "high": 0.75, "extreme": 1.0
            }.get(audio_analysis.intensity_level.value, 0.5)
        }

        # Add prosodic features
        feature_vector.update(audio_analysis.prosodic_features)

        return ModalityFeatures(
            modality_type=ModalityType.AUDIO,
            feature_vector=feature_vector,
            confidence_score=audio_analysis.emotion_confidence,
            quality_indicators={
                "emotion_confidence": audio_analysis.emotion_confidence,
                "stability_score": audio_analysis.emotional_stability
            },
            extraction_metadata={
                "dominant_emotion": audio_analysis.dominant_emotion.value,
                "intensity": audio_analysis.intensity_level.value
            }
        )

    def _calculate_disorder_confidence(self, multimodal_result, fusion_results: dict[str, Any]) -> float:
        """Calculate confidence in disorder identification."""
        # Base confidence from text analysis
        text_confidence = multimodal_result.text_analysis["condition_recognition"]["quality_score"]

        # Fusion confidence
        fusion_confidence = fusion_results["fusion_confidence"]

        # Cross-modal consistency bonus
        consistency_bonus = fusion_results["cross_modal_consistency"] * 0.2

        # Overall confidence
        disorder_confidence = (text_confidence * 0.5 + fusion_confidence * 0.3 + consistency_bonus)

        return min(disorder_confidence, 1.0)

    def _generate_diagnostic_evidence(self, multimodal_result, modality_features: list[ModalityFeatures]) -> dict[str, list[str]]:
        """Generate diagnostic evidence from all modalities."""
        evidence = defaultdict(list)

        # Text-based evidence
        text_risks = multimodal_result.text_analysis["condition_recognition"]["risk_indicators"]
        evidence["text_indicators"].extend(text_risks)

        # Audio evidence
        if hasattr(multimodal_result, "audio_analysis"):
            audio = multimodal_result.audio_analysis
            evidence["audio_indicators"].append(f"Dominant emotion: {audio.dominant_emotion.value}")
            evidence["audio_indicators"].append(f"Emotional stability: {audio.emotional_stability:.2f}")

        # Behavioral evidence
        for feature in modality_features:
            if feature.modality_type == ModalityType.BEHAVIORAL:
                for pattern, score in feature.feature_vector.items():
                    if score > 0.5:
                        evidence["behavioral_indicators"].append(f"{pattern}: {score:.2f}")

        return dict(evidence)

    def _assess_multimodal_severity(self, multimodal_result, modality_features: list[ModalityFeatures]) -> dict[str, Any]:
        """Assess severity using multiple modalities."""
        # Base severity from text
        text_severity = multimodal_result.text_analysis["condition_recognition"]["severity"]

        # Audio severity indicators
        audio_severity_factor = 1.0
        if hasattr(multimodal_result, "audio_analysis"):
            audio = multimodal_result.audio_analysis
            if audio.intensity_level.value == "high":
                audio_severity_factor = 1.2
            elif audio.intensity_level.value == "extreme":
                audio_severity_factor = 1.4

        # Behavioral severity indicators
        behavioral_severity_factor = 1.0
        for feature in modality_features:
            if feature.modality_type == ModalityType.BEHAVIORAL:
                if feature.feature_vector.get("emotional_regulation", 0) > 0.7:
                    behavioral_severity_factor = 1.1

        # Combined severity assessment
        severity_multiplier = (audio_severity_factor + behavioral_severity_factor) / 2

        return {
            "base_severity": text_severity,
            "audio_severity_factor": audio_severity_factor,
            "behavioral_severity_factor": behavioral_severity_factor,
            "combined_severity_multiplier": severity_multiplier,
            "final_severity_assessment": text_severity if severity_multiplier < 1.2 else "elevated_" + text_severity
        }

    def _generate_treatment_recommendations(self, disorder: MentalHealthCondition,
                                          multimodal_result, modality_features: list[ModalityFeatures]) -> list[str]:
        """Generate treatment recommendations based on multi-modal analysis."""
        recommendations = []

        # Base recommendations from multimodal analysis
        base_recommendations = multimodal_result.intervention_recommendations
        recommendations.extend(base_recommendations)

        # Audio-informed recommendations
        if hasattr(multimodal_result, "audio_analysis"):
            audio = multimodal_result.audio_analysis
            if audio.emotional_stability < 0.5:
                recommendations.append("Voice-based emotional regulation training")
            if audio.intensity_level.value in ["high", "extreme"]:
                recommendations.append("Immediate crisis intervention protocols")

        # Behavioral pattern recommendations
        for feature in modality_features:
            if feature.modality_type == ModalityType.BEHAVIORAL:
                if feature.feature_vector.get("social_interaction", 0) < 0.3:
                    recommendations.append("Social skills training and support group participation")
                if feature.feature_vector.get("cognitive_patterns", 0) < 0.4:
                    recommendations.append("Cognitive restructuring and organization skills")

        return list(set(recommendations))[:6]  # Remove duplicates, limit to 6

    def _determine_analysis_confidence(self, disorder_confidence: float,
                                     consistency_score: float) -> AnalysisConfidence:
        """Determine overall analysis confidence level."""
        combined_score = (disorder_confidence + consistency_score) / 2

        if combined_score >= 0.8:
            return AnalysisConfidence.VERY_HIGH
        if combined_score >= 0.65:
            return AnalysisConfidence.HIGH
        if combined_score >= 0.5:
            return AnalysisConfidence.MODERATE
        return AnalysisConfidence.LOW

    def _update_analysis_stats(self, analysis: MultiModalDisorderAnalysis):
        """Update analysis statistics."""
        self.analysis_stats["total_analyses"] += 1
        self.analysis_stats["disorder_detections"][analysis.primary_disorder.value] += 1
        self.analysis_stats["confidence_scores"].append(analysis.disorder_confidence)

        # Track modality usage
        for modality in analysis.feature_fusion_results["modalities_used"]:
            self.analysis_stats["modality_usage"][modality] += 1

    def get_analysis_statistics(self) -> dict[str, Any]:
        """Get comprehensive analysis statistics."""
        total = self.analysis_stats["total_analyses"]
        if total == 0:
            return {}

        return {
            "total_analyses": total,
            "disorder_distribution": dict(self.analysis_stats["disorder_detections"]),
            "modality_usage": dict(self.analysis_stats["modality_usage"]),
            "average_confidence": statistics.mean(self.analysis_stats["confidence_scores"]) if self.analysis_stats["confidence_scores"] else 0,
            "multimodal_stats": self.multimodal_analyzer.get_analysis_statistics()
        }


# Example usage and testing
def main():
    """Example usage of the MODMA analysis system."""

    # Create MODMA analyzer
    analyzer = MODMAAnalyzer()

    # Example conversation with rich multi-modal content
    test_conversation = {
        "id": "modma_test",
        "messages": [
            {
                "role": "client",
                "content": "I feel so overwhelmed and my voice is shaking. I keep going back and forth between feeling angry and then completely empty. I used to be able to handle things but now I can't focus on anything. I avoid people because I'm afraid they'll see how messed up I am."
            },
            {
                "role": "therapist",
                "content": "I can hear the distress in your voice and I want you to know that what you're experiencing is valid. Let's work together to understand these patterns and develop some coping strategies."
            }
        ]
    }


    # Perform comprehensive analysis
    analysis = analyzer.analyze_conversation(test_conversation)

    # Display results

    for _modality, _contribution in analysis.modality_contributions.items():
        pass

    for _evidence_type, evidence_list in analysis.diagnostic_evidence.items():
        if evidence_list:
            for _evidence in evidence_list[:3]:  # Show top 3
                pass


    for _recommendation in analysis.treatment_recommendations[:5]:
        pass

    # Show system statistics
    analyzer.get_analysis_statistics()


if __name__ == "__main__":
    main()

# Alias for compatibility
MultimodalDisorderAnalyzer = MODMAAnalyzer
