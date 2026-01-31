#!/usr/bin/env python3
"""
Advanced Pattern Recognition Engine

This module provides sophisticated pattern recognition capabilities for therapeutic
conversations, including therapeutic technique identification, conversation flow
analysis, and clinical pattern detection.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

# NLP imports
try:
    import spacy
    import torch
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import AutoModel, AutoTokenizer, pipeline
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False


@dataclass
class TherapeuticPattern:
    """Represents a detected therapeutic pattern."""
    pattern_id: str
    pattern_type: str  # 'technique', 'flow', 'clinical', 'emotional'
    confidence: float
    description: str
    evidence: list[str]
    metadata: dict[str, Any] = None


@dataclass
class ConversationAnalysis:
    """Complete analysis of a therapeutic conversation."""
    conversation_id: str
    patterns: list[TherapeuticPattern]
    complexity_score: float
    therapeutic_quality: float
    flow_coherence: float
    emotional_depth: float
    clinical_accuracy: float
    summary: str
    recommendations: list[str]


class TherapeuticTechniqueDetector:
    """Detects therapeutic techniques in conversations."""

    def __init__(self):
        self.technique_patterns = {
            "active_listening": [
                r"\b(I hear you|I understand|tell me more|go on|that sounds)\b",
                r"\b(reflecting|paraphrasing|summarizing)\b",
                r"\b(what I\'m hearing is|it sounds like|so you\'re saying)\b"
            ],
            "empathic_reflection": [
                r"\b(that must be|I can imagine|it sounds difficult|that\'s challenging)\b",
                r"\b(you\'re feeling|you seem|it appears you)\b",
                r"\b(I sense|I notice|I can see)\b"
            ],
            "cognitive_restructuring": [
                r"\b(what evidence|alternative thought|different perspective)\b",
                r"\b(thought challenging|cognitive distortion|thinking pattern)\b",
                r"\b(is that thought helpful|realistic thinking|balanced view)\b"
            ],
            "behavioral_activation": [
                r"\b(activity scheduling|behavioral experiment|action plan)\b",
                r"\b(what activities|pleasant events|behavioral goals)\b",
                r"\b(homework assignment|between sessions|practice)\b"
            ],
            "mindfulness": [
                r"\b(present moment|mindful|awareness|breathing)\b",
                r"\b(meditation|grounding|body scan|mindful observation)\b",
                r"\b(notice without judgment|accepting|letting go)\b"
            ],
            "solution_focused": [
                r"\b(scaling question|miracle question|exception finding)\b",
                r"\b(what\'s working|strengths|resources|coping skills)\b",
                r"\b(small steps|goals|preferred future)\b"
            ],
            "psychodynamic": [
                r"\b(childhood|early experiences|patterns|unconscious)\b",
                r"\b(transference|defense mechanisms|insight|interpretation)\b",
                r"\b(relationship patterns|attachment|family dynamics)\b"
            ],
            "motivational_interviewing": [
                r"\b(change talk|ambivalence|motivation|readiness)\b",
                r"\b(what would need to change|importance|confidence)\b",
                r"\b(rolling with resistance|eliciting|evoking)\b"
            ]
        }

    def detect_techniques(self, conversation: dict[str, Any]) -> list[TherapeuticPattern]:
        """Detect therapeutic techniques in a conversation."""
        patterns = []

        # Extract text from conversation
        text = self._extract_conversation_text(conversation)

        for technique, pattern_list in self.technique_patterns.items():
            matches = []
            total_confidence = 0

            for pattern in pattern_list:
                found_matches = re.findall(pattern, text, re.IGNORECASE)
                if found_matches:
                    matches.extend(found_matches)
                    total_confidence += len(found_matches)

            if matches:
                confidence = min(total_confidence / 10.0, 1.0)  # Normalize to 0-1

                pattern = TherapeuticPattern(
                    pattern_id=f"technique_{technique}",
                    pattern_type="technique",
                    confidence=confidence,
                    description=f"Detected {technique.replace('_', ' ')} technique",
                    evidence=matches[:5],  # Top 5 evidence pieces
                    metadata={"technique": technique, "match_count": len(matches)}
                )
                patterns.append(pattern)

        return patterns

    def _extract_conversation_text(self, conversation: dict[str, Any]) -> str:
        """Extract text content from conversation."""
        text_parts = []

        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict) and "content" in message:
                    text_parts.append(message["content"])
                elif isinstance(message, str):
                    text_parts.append(message)
        elif "conversations" in conversation:
            for conv in conversation["conversations"]:
                if isinstance(conv, dict):
                    for _key, value in conv.items():
                        if isinstance(value, str):
                            text_parts.append(value)

        return " ".join(text_parts)


class ConversationFlowAnalyzer:
    """Analyzes conversation flow and coherence."""

    def __init__(self):
        if NLP_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.nlp = None
        else:
            self.nlp = None

    def analyze_flow(self, conversation: dict[str, Any]) -> list[TherapeuticPattern]:
        """Analyze conversation flow patterns."""
        patterns = []

        if not self.nlp:
            return patterns

        messages = self._extract_messages(conversation)
        if len(messages) < 2:
            return patterns

        # Analyze turn-taking patterns
        turn_pattern = self._analyze_turn_taking(messages)
        if turn_pattern:
            patterns.append(turn_pattern)

        # Analyze topic coherence
        coherence_pattern = self._analyze_topic_coherence(messages)
        if coherence_pattern:
            patterns.append(coherence_pattern)

        # Analyze conversation progression
        progression_pattern = self._analyze_progression(messages)
        if progression_pattern:
            patterns.append(progression_pattern)

        return patterns

    def _extract_messages(self, conversation: dict[str, Any]) -> list[dict[str, str]]:
        """Extract messages with speaker information."""
        messages = []

        if "messages" in conversation:
            for i, message in enumerate(conversation["messages"]):
                if isinstance(message, dict):
                    messages.append({
                        "speaker": message.get("role", f"speaker_{i % 2}"),
                        "content": message.get("content", ""),
                        "index": i
                    })

        return messages

    def _analyze_turn_taking(self, messages: list[dict[str, str]]) -> TherapeuticPattern | None:
        """Analyze turn-taking patterns."""
        if len(messages) < 4:
            return None

        # Calculate turn lengths
        turn_lengths = [len(msg["content"].split()) for msg in messages]
        avg_length = np.mean(turn_lengths)
        length_variance = np.var(turn_lengths)

        # Analyze speaker balance
        speakers = [msg["speaker"] for msg in messages]
        speaker_counts = Counter(speakers)
        balance_score = min(speaker_counts.values()) / max(speaker_counts.values()) if speaker_counts else 0

        # Calculate overall flow score
        flow_score = balance_score * (1 - min(length_variance / (avg_length ** 2), 1))

        return TherapeuticPattern(
            pattern_id="flow_turn_taking",
            pattern_type="flow",
            confidence=flow_score,
            description=f"Turn-taking balance: {balance_score:.2f}, Length consistency: {1 - min(length_variance / (avg_length ** 2), 1):.2f}",
            evidence=[f"Average turn length: {avg_length:.1f} words", f"Speaker balance: {balance_score:.2f}"],
            metadata={
                "avg_turn_length": avg_length,
                "length_variance": length_variance,
                "speaker_balance": balance_score
            }
        )

    def _analyze_topic_coherence(self, messages: list[dict[str, str]]) -> TherapeuticPattern | None:
        """Analyze topic coherence across conversation."""
        if len(messages) < 3:
            return None

        # Extract content
        contents = [msg["content"] for msg in messages if msg["content"]]
        if len(contents) < 3:
            return None

        # Calculate semantic similarity between adjacent messages
        similarities = []
        for i in range(len(contents) - 1):
            doc1 = self.nlp(contents[i])
            doc2 = self.nlp(contents[i + 1])
            similarity = doc1.similarity(doc2)
            similarities.append(similarity)

        avg_coherence = np.mean(similarities) if similarities else 0

        return TherapeuticPattern(
            pattern_id="flow_coherence",
            pattern_type="flow",
            confidence=avg_coherence,
            description=f"Topic coherence score: {avg_coherence:.3f}",
            evidence=[f"Average adjacent similarity: {avg_coherence:.3f}"],
            metadata={"coherence_score": avg_coherence, "similarities": similarities}
        )

    def _analyze_progression(self, messages: list[dict[str, str]]) -> TherapeuticPattern | None:
        """Analyze conversation progression patterns."""
        if len(messages) < 5:
            return None

        # Analyze emotional progression (simplified)
        emotional_words = {
            "positive": ["good", "better", "happy", "hopeful", "confident", "strong"],
            "negative": ["bad", "worse", "sad", "hopeless", "anxious", "weak"],
            "neutral": ["okay", "fine", "normal", "usual", "same"]
        }

        emotional_scores = []
        for msg in messages:
            content_lower = msg["content"].lower()
            pos_count = sum(1 for word in emotional_words["positive"] if word in content_lower)
            neg_count = sum(1 for word in emotional_words["negative"] if word in content_lower)

            if pos_count + neg_count > 0:
                emotional_scores.append((pos_count - neg_count) / (pos_count + neg_count))
            else:
                emotional_scores.append(0)

        # Calculate progression trend
        if len(emotional_scores) > 2:
            progression_trend = np.polyfit(range(len(emotional_scores)), emotional_scores, 1)[0]

            return TherapeuticPattern(
                pattern_id="flow_progression",
                pattern_type="flow",
                confidence=abs(progression_trend),
                description=f"Emotional progression trend: {'positive' if progression_trend > 0 else 'negative'} ({progression_trend:.3f})",
                evidence=[f"Progression slope: {progression_trend:.3f}"],
                metadata={"progression_trend": progression_trend, "emotional_scores": emotional_scores}
            )

        return None


class ClinicalPatternDetector:
    """Detects clinical patterns and indicators."""

    def __init__(self):
        self.clinical_indicators = {
            "depression": [
                r"\b(depressed|sad|hopeless|worthless|empty|down)\b",
                r"\b(sleep problems|insomnia|fatigue|tired|exhausted)\b",
                r"\b(appetite|weight loss|weight gain|eating)\b",
                r"\b(concentration|focus|memory|decision making)\b"
            ],
            "anxiety": [
                r"\b(anxious|worried|nervous|panic|fear|scared)\b",
                r"\b(racing thoughts|restless|on edge|tense)\b",
                r"\b(physical symptoms|heart racing|sweating|trembling)\b",
                r"\b(avoidance|avoiding|can\'t handle|overwhelming)\b"
            ],
            "trauma": [
                r"\b(flashbacks|nightmares|intrusive thoughts|memories)\b",
                r"\b(triggered|hypervigilant|startled|jumpy)\b",
                r"\b(numb|detached|disconnected|dissociation)\b",
                r"\b(trauma|abuse|assault|accident|violence)\b"
            ],
            "relationship_issues": [
                r"\b(relationship|partner|spouse|marriage|divorce)\b",
                r"\b(communication|conflict|arguing|fighting)\b",
                r"\b(trust|betrayal|infidelity|cheating)\b",
                r"\b(family|parents|children|siblings)\b"
            ]
        }

        self.risk_indicators = [
            r"\b(suicide|kill myself|end it all|not worth living)\b",
            r"\b(self harm|cutting|hurting myself)\b",
            r"\b(substance|drinking|drugs|addiction)\b",
            r"\b(violence|hurt someone|angry|rage)\b"
        ]

    def detect_clinical_patterns(self, conversation: dict[str, Any]) -> list[TherapeuticPattern]:
        """Detect clinical patterns in conversation."""
        patterns = []
        text = self._extract_text(conversation)

        # Detect clinical indicators
        for condition, pattern_list in self.clinical_indicators.items():
            matches = []
            confidence = 0

            for pattern in pattern_list:
                found = re.findall(pattern, text, re.IGNORECASE)
                if found:
                    matches.extend(found)
                    confidence += len(found)

            if matches:
                normalized_confidence = min(confidence / 5.0, 1.0)

                pattern = TherapeuticPattern(
                    pattern_id=f"clinical_{condition}",
                    pattern_type="clinical",
                    confidence=normalized_confidence,
                    description=f"Indicators of {condition.replace('_', ' ')}",
                    evidence=matches[:3],
                    metadata={"condition": condition, "indicator_count": len(matches)}
                )
                patterns.append(pattern)

        # Detect risk indicators
        risk_matches = []
        for pattern in self.risk_indicators:
            found = re.findall(pattern, text, re.IGNORECASE)
            if found:
                risk_matches.extend(found)

        if risk_matches:
            risk_pattern = TherapeuticPattern(
                pattern_id="clinical_risk",
                pattern_type="clinical",
                confidence=min(len(risk_matches) / 2.0, 1.0),
                description="Risk indicators detected",
                evidence=risk_matches[:3],
                metadata={"risk_level": "high" if len(risk_matches) > 2 else "moderate"}
            )
            patterns.append(risk_pattern)

        return patterns

    def _extract_text(self, conversation: dict[str, Any]) -> str:
        """Extract text from conversation."""
        text_parts = []

        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict) and "content" in message:
                    text_parts.append(message["content"])

        return " ".join(text_parts)


class PatternRecognitionEngine:
    """Main pattern recognition engine combining all detectors."""

    def __init__(self):
        self.technique_detector = TherapeuticTechniqueDetector()
        self.flow_analyzer = ConversationFlowAnalyzer()
        self.clinical_detector = ClinicalPatternDetector()

        self.logger = logging.getLogger(__name__)

    def analyze_conversation(self, conversation: dict[str, Any]) -> ConversationAnalysis:
        """Perform complete pattern analysis of a conversation."""
        conversation_id = conversation.get("id", "unknown")

        try:
            # Detect all patterns
            all_patterns = []

            # Therapeutic techniques
            technique_patterns = self.technique_detector.detect_techniques(conversation)
            all_patterns.extend(technique_patterns)

            # Conversation flow
            flow_patterns = self.flow_analyzer.analyze_flow(conversation)
            all_patterns.extend(flow_patterns)

            # Clinical patterns
            clinical_patterns = self.clinical_detector.detect_clinical_patterns(conversation)
            all_patterns.extend(clinical_patterns)

            # Calculate overall scores
            complexity_score = self._calculate_complexity_score(conversation, all_patterns)
            therapeutic_quality = self._calculate_therapeutic_quality(all_patterns)
            flow_coherence = self._calculate_flow_coherence(flow_patterns)
            emotional_depth = self._calculate_emotional_depth(all_patterns)
            clinical_accuracy = self._calculate_clinical_accuracy(clinical_patterns)

            # Generate summary and recommendations
            summary = self._generate_summary(all_patterns)
            recommendations = self._generate_recommendations(all_patterns)

            return ConversationAnalysis(
                conversation_id=conversation_id,
                patterns=all_patterns,
                complexity_score=complexity_score,
                therapeutic_quality=therapeutic_quality,
                flow_coherence=flow_coherence,
                emotional_depth=emotional_depth,
                clinical_accuracy=clinical_accuracy,
                summary=summary,
                recommendations=recommendations
            )

        except Exception as e:
            self.logger.error(f"Error analyzing conversation {conversation_id}: {e}")
            return ConversationAnalysis(
                conversation_id=conversation_id,
                patterns=[],
                complexity_score=0.0,
                therapeutic_quality=0.0,
                flow_coherence=0.0,
                emotional_depth=0.0,
                clinical_accuracy=0.0,
                summary="Analysis failed",
                recommendations=["Review conversation format"]
            )

    def _calculate_complexity_score(self, conversation: dict[str, Any], patterns: list[TherapeuticPattern]) -> float:
        """Calculate conversation complexity score."""
        # Base complexity on message count, length, and pattern diversity
        message_count = len(conversation.get("messages", []))

        # Calculate text complexity
        text = " ".join([
            msg.get("content", "") for msg in conversation.get("messages", [])
            if isinstance(msg, dict)
        ])

        word_count = len(text.split())
        unique_words = len(set(text.lower().split()))
        lexical_diversity = unique_words / max(word_count, 1)

        # Pattern diversity
        pattern_types = {p.pattern_type for p in patterns}
        pattern_diversity = len(pattern_types) / 4.0  # 4 possible types

        # Combine factors
        complexity = (
            min(message_count / 20.0, 1.0) * 0.3 +  # Message count factor
            min(word_count / 1000.0, 1.0) * 0.3 +    # Length factor
            lexical_diversity * 0.2 +                 # Vocabulary diversity
            pattern_diversity * 0.2                   # Pattern diversity
        )

        return min(complexity, 1.0)

    def _calculate_therapeutic_quality(self, patterns: list[TherapeuticPattern]) -> float:
        """Calculate therapeutic quality score."""
        technique_patterns = [p for p in patterns if p.pattern_type == "technique"]

        if not technique_patterns:
            return 0.0

        # Average confidence of therapeutic techniques
        avg_confidence = np.mean([p.confidence for p in technique_patterns])

        # Bonus for technique diversity
        unique_techniques = len({p.metadata.get("technique", "") for p in technique_patterns})
        diversity_bonus = min(unique_techniques / 5.0, 0.2)  # Max 20% bonus

        return min(avg_confidence + diversity_bonus, 1.0)

    def _calculate_flow_coherence(self, flow_patterns: list[TherapeuticPattern]) -> float:
        """Calculate flow coherence score."""
        if not flow_patterns:
            return 0.5  # Neutral score if no flow analysis

        return np.mean([p.confidence for p in flow_patterns])

    def _calculate_emotional_depth(self, patterns: list[TherapeuticPattern]) -> float:
        """Calculate emotional depth score."""
        # Look for emotional indicators in patterns
        emotional_indicators = 0
        total_confidence = 0

        for pattern in patterns:
            if any(keyword in pattern.description.lower()
                   for keyword in ["emotional", "feeling", "empathic", "depth"]):
                emotional_indicators += 1
                total_confidence += pattern.confidence

        if emotional_indicators == 0:
            return 0.3  # Low baseline

        return min(total_confidence / emotional_indicators, 1.0)

    def _calculate_clinical_accuracy(self, clinical_patterns: list[TherapeuticPattern]) -> float:
        """Calculate clinical accuracy score."""
        if not clinical_patterns:
            return 0.7  # Neutral score if no clinical patterns detected

        # Higher confidence in clinical patterns indicates better accuracy
        return np.mean([p.confidence for p in clinical_patterns])

    def _generate_summary(self, patterns: list[TherapeuticPattern]) -> str:
        """Generate analysis summary."""
        if not patterns:
            return "No significant patterns detected in conversation."

        technique_count = len([p for p in patterns if p.pattern_type == "technique"])
        clinical_count = len([p for p in patterns if p.pattern_type == "clinical"])
        flow_count = len([p for p in patterns if p.pattern_type == "flow"])

        summary_parts = []

        if technique_count > 0:
            summary_parts.append(f"{technique_count} therapeutic techniques identified")

        if clinical_count > 0:
            summary_parts.append(f"{clinical_count} clinical indicators detected")

        if flow_count > 0:
            summary_parts.append("conversation flow analysis completed")

        return "Analysis found: " + ", ".join(summary_parts) + "."

    def _generate_recommendations(self, patterns: list[TherapeuticPattern]) -> list[str]:
        """Generate recommendations based on patterns."""
        recommendations = []

        # Check for missing therapeutic techniques
        technique_patterns = [p for p in patterns if p.pattern_type == "technique"]
        detected_techniques = {p.metadata.get("technique", "") for p in technique_patterns}

        if "empathic_reflection" not in detected_techniques:
            recommendations.append("Consider incorporating more empathic reflection")

        if "active_listening" not in detected_techniques:
            recommendations.append("Enhance active listening responses")

        # Check for clinical concerns
        clinical_patterns = [p for p in patterns if p.pattern_type == "clinical"]
        for pattern in clinical_patterns:
            if pattern.pattern_id == "clinical_risk":
                recommendations.append("Risk indicators present - consider safety assessment")

        # Check flow issues
        flow_patterns = [p for p in patterns if p.pattern_type == "flow"]
        for pattern in flow_patterns:
            if pattern.confidence < 0.5:
                if "coherence" in pattern.pattern_id:
                    recommendations.append("Improve topic coherence and continuity")
                elif "turn_taking" in pattern.pattern_id:
                    recommendations.append("Balance conversation participation")

        if not recommendations:
            recommendations.append("Conversation shows good therapeutic patterns")

        return recommendations


# Example usage
if __name__ == "__main__":
    # Create pattern recognition engine
    engine = PatternRecognitionEngine()

    # Example conversation
    example_conversation = {
        "id": "example_001",
        "messages": [
            {"role": "client", "content": "I've been feeling really depressed lately. I can't sleep and I have no energy."},
            {"role": "therapist", "content": "I hear you saying you're feeling depressed and experiencing sleep problems and low energy. That sounds really difficult. Can you tell me more about when these feelings started?"},
            {"role": "client", "content": "It started about a month ago when I lost my job. I feel worthless and like nothing will ever get better."},
            {"role": "therapist", "content": "Losing a job can be a significant life stressor, and it makes sense that you're struggling. When you say nothing will ever get better, what evidence do you have for that thought?"},
            {"role": "client", "content": "I guess... I don't really have evidence. It just feels that way."},
            {"role": "therapist", "content": "That's a really important insight. Sometimes our feelings can make thoughts seem like facts. What are some things that have gotten better in your life before, even during difficult times?"}
        ]
    }

    # Analyze conversation
    analysis = engine.analyze_conversation(example_conversation)

    # Print results

    for _pattern in analysis.patterns:
        pass

    for _rec in analysis.recommendations:
        pass
