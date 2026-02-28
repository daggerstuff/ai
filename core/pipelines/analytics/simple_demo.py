#!/usr/bin/env python3
"""
Simplified Advanced Analytics Demonstration

This script demonstrates the core analytics capabilities without requiring
additional dependencies, focusing on pattern recognition and complexity scoring.
"""

import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class SimplePattern:
    """Simple pattern detection result."""
    pattern_type: str
    description: str
    confidence: float
    evidence: list[str]


@dataclass
class SimpleComplexity:
    """Simple complexity metrics."""
    overall_complexity: float
    word_count: int
    unique_words: int
    vocabulary_diversity: float
    therapeutic_concepts: int
    emotional_words: int


@dataclass
class SimpleAnalysis:
    """Simple conversation analysis result."""
    conversation_id: str
    quality_score: float
    sophistication_level: str
    patterns: list[SimplePattern]
    complexity: SimpleComplexity
    recommendations: list[str]


class SimplePatternDetector:
    """Simplified pattern detection for therapeutic conversations."""

    def __init__(self):
        self.therapeutic_patterns = {
            "empathic_reflection": [
                r"\b(I understand|I hear you|that sounds|I can imagine)\b",
                r"\b(you\'re feeling|it seems like|I sense)\b"
            ],
            "active_listening": [
                r"\b(tell me more|go on|what else|help me understand)\b",
                r"\b(I\'m listening|I\'m here|continue)\b"
            ],
            "cognitive_restructuring": [
                r"\b(what evidence|alternative thought|different perspective)\b",
                r"\b(is that helpful|realistic|balanced view)\b"
            ],
            "validation": [
                r"\b(that\'s valid|makes sense|understandable|legitimate)\b",
                r"\b(anyone would feel|normal to feel|reasonable)\b"
            ],
            "questioning": [
                r"\b(what|how|when|where|why|can you tell me)\b",
                r"\b(what do you think|how does that feel)\b"
            ]
        }

        self.emotional_words = [
            "anxious", "depressed", "sad", "happy", "angry", "frustrated",
            "hopeful", "worried", "scared", "excited", "overwhelmed",
            "grateful", "ashamed", "guilty", "proud", "lonely", "confused"
        ]

        self.therapeutic_concepts = [
            "therapy", "counseling", "coping", "stress", "anxiety", "depression",
            "trauma", "healing", "growth", "insight", "awareness", "mindfulness",
            "relationship", "attachment", "boundaries", "self-care", "resilience"
        ]

    def detect_patterns(self, conversation: dict[str, Any]) -> list[SimplePattern]:
        """Detect therapeutic patterns in conversation."""
        patterns = []
        text = self._extract_text(conversation)

        for pattern_type, pattern_list in self.therapeutic_patterns.items():
            matches = []
            for pattern in pattern_list:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)

            if matches:
                confidence = min(len(matches) / 3.0, 1.0)  # Normalize
                patterns.append(SimplePattern(
                    pattern_type=pattern_type,
                    description=f"Detected {pattern_type.replace('_', ' ')}",
                    confidence=confidence,
                    evidence=matches[:3]
                ))

        return patterns

    def _extract_text(self, conversation: dict[str, Any]) -> str:
        """Extract text from conversation."""
        text_parts = []
        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict) and "content" in message:
                    text_parts.append(message["content"])
        return " ".join(text_parts)


class SimpleComplexityScorer:
    """Simplified complexity scoring for conversations."""

    def __init__(self):
        self.detector = SimplePatternDetector()

    def score_complexity(self, conversation: dict[str, Any]) -> SimpleComplexity:
        """Score conversation complexity."""
        text = self._extract_text(conversation)
        words = text.lower().split()

        # Basic metrics
        word_count = len(words)
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / max(word_count, 1)

        # Therapeutic concepts
        therapeutic_count = sum(1 for concept in self.detector.therapeutic_concepts
                              if concept in text.lower())

        # Emotional words
        emotional_count = sum(1 for emotion in self.detector.emotional_words
                            if emotion in text.lower())

        # Calculate overall complexity
        length_factor = min(word_count / 200.0, 1.0)
        diversity_factor = vocabulary_diversity
        concept_factor = min(therapeutic_count / 5.0, 1.0)
        emotional_factor = min(emotional_count / 3.0, 1.0)

        overall_complexity = (
            length_factor * 0.3 +
            diversity_factor * 0.3 +
            concept_factor * 0.2 +
            emotional_factor * 0.2
        )

        return SimpleComplexity(
            overall_complexity=overall_complexity,
            word_count=word_count,
            unique_words=unique_words,
            vocabulary_diversity=vocabulary_diversity,
            therapeutic_concepts=therapeutic_count,
            emotional_words=emotional_count
        )

    def _extract_text(self, conversation: dict[str, Any]) -> str:
        """Extract text from conversation."""
        text_parts = []
        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict) and "content" in message:
                    text_parts.append(message["content"])
        return " ".join(text_parts)


class SimpleAnalyticsEngine:
    """Simplified analytics engine combining pattern detection and complexity scoring."""

    def __init__(self):
        self.pattern_detector = SimplePatternDetector()
        self.complexity_scorer = SimpleComplexityScorer()
        self.results_history = []

    def analyze_conversation(self, conversation: dict[str, Any]) -> SimpleAnalysis:
        """Analyze a conversation comprehensively."""
        conversation_id = conversation.get("id", f"conv_{int(time.time())}")

        # Detect patterns
        patterns = self.pattern_detector.detect_patterns(conversation)

        # Score complexity
        complexity = self.complexity_scorer.score_complexity(conversation)

        # Calculate quality score
        quality_score = self._calculate_quality_score(patterns, complexity)

        # Determine sophistication level
        sophistication_level = self._determine_sophistication_level(complexity, patterns)

        # Generate recommendations
        recommendations = self._generate_recommendations(patterns, complexity)

        # Create analysis result
        analysis = SimpleAnalysis(
            conversation_id=conversation_id,
            quality_score=quality_score,
            sophistication_level=sophistication_level,
            patterns=patterns,
            complexity=complexity,
            recommendations=recommendations
        )

        # Store result
        self.results_history.append(analysis)

        return analysis

    def _calculate_quality_score(self, patterns: list[SimplePattern],
                                complexity: SimpleComplexity) -> float:
        """Calculate overall quality score."""
        pattern_score = 0.0 if not patterns else statistics.mean([p.confidence for p in patterns])

        # Combine pattern quality with complexity (higher complexity can indicate quality)
        quality_score = pattern_score * 0.7 + complexity.overall_complexity * 0.3
        return min(quality_score, 1.0)

    def _determine_sophistication_level(self, complexity: SimpleComplexity,
                                      patterns: list[SimplePattern]) -> str:
        """Determine sophistication level."""
        # Base on complexity and pattern diversity
        pattern_diversity = len({p.pattern_type for p in patterns})

        sophistication_score = (
            complexity.overall_complexity * 0.6 +
            min(pattern_diversity / 5.0, 1.0) * 0.4
        )

        if sophistication_score >= 0.8:
            return "expert"
        if sophistication_score >= 0.6:
            return "advanced"
        if sophistication_score >= 0.4:
            return "intermediate"
        return "basic"

    def _generate_recommendations(self, patterns: list[SimplePattern],
                                complexity: SimpleComplexity) -> list[str]:
        """Generate improvement recommendations."""
        recommendations = []

        # Pattern-based recommendations
        pattern_types = {p.pattern_type for p in patterns}

        if "empathic_reflection" not in pattern_types:
            recommendations.append("Consider adding more empathic reflection")

        if "active_listening" not in pattern_types:
            recommendations.append("Enhance active listening responses")

        if "validation" not in pattern_types:
            recommendations.append("Include more validation statements")

        # Complexity-based recommendations
        if complexity.therapeutic_concepts < 2:
            recommendations.append("Incorporate more therapeutic concepts")

        if complexity.emotional_words < 2:
            recommendations.append("Explore emotional depth more thoroughly")

        if complexity.vocabulary_diversity < 0.5:
            recommendations.append("Vary vocabulary for richer expression")

        # Default recommendation
        if not recommendations:
            recommendations.append("Conversation shows good therapeutic patterns")

        return recommendations[:5]  # Limit to top 5

    def get_summary(self) -> dict[str, Any]:
        """Get analytics summary."""
        if not self.results_history:
            return {"message": "No conversations analyzed yet"}

        # Calculate summary statistics
        quality_scores = [r.quality_score for r in self.results_history]
        complexity_scores = [r.complexity.overall_complexity for r in self.results_history]
        sophistication_counts = Counter(r.sophistication_level for r in self.results_history)

        return {
            "total_conversations": len(self.results_history),
            "average_quality": statistics.mean(quality_scores),
            "average_complexity": statistics.mean(complexity_scores),
            "sophistication_distribution": dict(sophistication_counts),
            "quality_distribution": {
                "high": len([q for q in quality_scores if q >= 0.7]),
                "medium": len([q for q in quality_scores if 0.4 <= q < 0.7]),
                "low": len([q for q in quality_scores if q < 0.4])
            }
        }


def create_sample_conversations():
    """Create sample conversations for demonstration."""
    return [
        {
            "id": "basic_conversation",
            "messages": [
                {"role": "client", "content": "I feel sad today."},
                {"role": "therapist", "content": "I understand. Can you tell me more?"},
                {"role": "client", "content": "Work is stressful."},
                {"role": "therapist", "content": "What about work is stressful?"}
            ]
        },
        {
            "id": "intermediate_conversation",
            "messages": [
                {"role": "client", "content": "I've been having anxiety attacks. They come out of nowhere and I can't breathe."},
                {"role": "therapist", "content": "I hear you saying you're experiencing panic attacks, and that sounds really frightening. When you say they come out of nowhere, have you noticed any patterns or triggers?"},
                {"role": "client", "content": "Now that you mention it, they often happen in crowded places or before meetings."},
                {"role": "therapist", "content": "That's a really important insight. It sounds like social situations might be triggering your anxiety. Let's explore some coping strategies you can use."}
            ]
        },
        {
            "id": "advanced_conversation",
            "messages": [
                {"role": "client", "content": "I've been reflecting on my attachment patterns. I think my childhood experiences with my emotionally unavailable father are affecting my relationships."},
                {"role": "therapist", "content": "That's a profound insight. Making connections between early attachment experiences and current patterns shows significant growth. What specific patterns are you noticing?"},
                {"role": "client", "content": "I have this anxious attachment where I seek reassurance but also push people away. I'm terrified of both abandonment and intimacy."},
                {"role": "therapist", "content": "You've articulated something many struggle with - that paradox of craving connection while fearing it. This awareness is the first step toward developing more secure attachment behaviors."}
            ]
        },
        {
            "id": "expert_conversation",
            "messages": [
                {"role": "client", "content": "I've been practicing mindfulness and noticing my dissociative episodes. There's a somatic component - I feel disconnection starting in my chest before my mind goes blank."},
                {"role": "therapist", "content": "That's remarkable somatic awareness. Tracking embodied precursors to dissociation suggests your window of tolerance is expanding. You're catching the nervous system response before full dissociation."},
                {"role": "client", "content": "Yes, and when I notice that chest sensation, I use bilateral stimulation. It helps me stay present and connected to my body."},
                {"role": "therapist", "content": "This is beautiful integration of somatic awareness with self-regulation. You're rewiring your nervous system's trauma response. This neuroplasticity-informed healing is exactly what we're aiming for."}
            ]
        }
    ]


def main():
    """Main demonstration function."""

    # Create analytics engine
    engine = SimpleAnalyticsEngine()

    # Get sample conversations
    conversations = create_sample_conversations()


    # Analyze each conversation
    for conv in conversations:
        analysis = engine.analyze_conversation(conv)


        # Show detected patterns
        if analysis.patterns:
            for _pattern in analysis.patterns[:3]:  # Show top 3
                pass

        # Show complexity details

        # Show recommendations
        for _rec in analysis.recommendations[:2]:  # Show top 2
            pass

    # Show system summary

    summary = engine.get_summary()

    for _level, _count in summary["sophistication_distribution"].items():
        pass

    for _level, _count in summary["quality_distribution"].items():
        pass




if __name__ == "__main__":
    main()
