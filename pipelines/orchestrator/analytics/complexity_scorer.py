#!/usr/bin/env python3
"""
Advanced Complexity Scoring System

This module provides sophisticated complexity analysis for therapeutic conversations,
including linguistic complexity, therapeutic depth, emotional complexity, and
clinical sophistication scoring.
"""

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# NLP imports
try:
    import spacy
    from textstat import (
        automated_readability_index,
        flesch_kincaid_grade,
        flesch_reading_ease,
    )
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False


@dataclass
class ComplexityMetrics:
    """Comprehensive complexity metrics for a conversation."""
    overall_complexity: float
    linguistic_complexity: float
    therapeutic_depth: float
    emotional_complexity: float
    clinical_sophistication: float
    cognitive_load: float
    interaction_complexity: float

    # Detailed sub-metrics
    readability_score: float
    vocabulary_diversity: float
    syntactic_complexity: float
    semantic_density: float
    therapeutic_technique_count: int
    emotional_range: float
    clinical_terminology_density: float
    turn_taking_complexity: float

    # Metadata
    word_count: int
    sentence_count: int
    turn_count: int
    unique_words: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class LinguisticComplexityAnalyzer:
    """Analyzes linguistic complexity of conversations."""

    def __init__(self):
        if NLP_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.nlp = None
        else:
            self.nlp = None

    def analyze_linguistic_complexity(self, text: str) -> dict[str, float]:
        """Analyze linguistic complexity of text."""
        if not text.strip():
            return self._empty_metrics()

        metrics = {}

        # Basic readability metrics
        try:
            metrics["flesch_reading_ease"] = flesch_reading_ease(text)
            metrics["flesch_kincaid_grade"] = flesch_kincaid_grade(text)
            metrics["automated_readability"] = automated_readability_index(text)
        except Exception as e:
            self.logger.warning(f"Error calculating readability metrics: {e}")
            metrics["flesch_reading_ease"] = 50.0
            metrics["flesch_kincaid_grade"] = 8.0
            metrics["automated_readability"] = 8.0

        # Vocabulary diversity
        words = text.lower().split()
        unique_words = set(words)
        metrics["vocabulary_diversity"] = len(unique_words) / max(len(words), 1)

        # Lexical density
        if self.nlp:
            doc = self.nlp(text)
            content_words = [token for token in doc if not token.is_stop and not token.is_punct]
            metrics["lexical_density"] = len(content_words) / max(len(doc), 1)
        else:
            metrics["lexical_density"] = 0.5

        # Syntactic complexity
        metrics["syntactic_complexity"] = self._calculate_syntactic_complexity(text)

        # Semantic density
        metrics["semantic_density"] = self._calculate_semantic_density(text)

        return metrics

    def _empty_metrics(self) -> dict[str, float]:
        """Return empty metrics for invalid input."""
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "automated_readability": 0.0,
            "vocabulary_diversity": 0.0,
            "lexical_density": 0.0,
            "syntactic_complexity": 0.0,
            "semantic_density": 0.0
        }

    def _calculate_syntactic_complexity(self, text: str) -> float:
        """Calculate syntactic complexity based on sentence structure."""
        if not self.nlp:
            return 0.5

        doc = self.nlp(text)
        sentences = list(doc.sents)

        if not sentences:
            return 0.0

        complexity_scores = []

        for sent in sentences:
            # Count subordinate clauses
            subordinate_count = sum(1 for token in sent if token.dep_ in ["advcl", "ccomp", "xcomp", "acl"])

            # Count complex structures
            complex_structures = sum(1 for token in sent if token.dep_ in ["conj", "appos", "nmod"])

            # Sentence length factor
            length_factor = min(len(sent) / 20.0, 1.0)

            # Combine factors
            sent_complexity = (subordinate_count * 0.4 + complex_structures * 0.3 + length_factor * 0.3)
            complexity_scores.append(sent_complexity)

        return min(np.mean(complexity_scores), 1.0)

    def _calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic density based on content word distribution."""
        if not self.nlp:
            return 0.5

        doc = self.nlp(text)

        if len(doc) == 0:
            return 0.0

        # Count different types of semantic content
        nouns = sum(1 for token in doc if token.pos_ == "NOUN")
        verbs = sum(1 for token in doc if token.pos_ == "VERB")
        adjectives = sum(1 for token in doc if token.pos_ == "ADJ")
        adverbs = sum(1 for token in doc if token.pos_ == "ADV")

        content_words = nouns + verbs + adjectives + adverbs
        semantic_density = content_words / len(doc)

        return min(semantic_density, 1.0)


class TherapeuticDepthAnalyzer:
    """Analyzes therapeutic depth and sophistication."""

    def __init__(self):
        self.therapeutic_concepts = {
            "basic": [
                "feeling", "think", "problem", "help", "better", "difficult", "understand"
            ],
            "intermediate": [
                "emotion", "behavior", "pattern", "relationship", "coping", "stress", "anxiety",
                "depression", "therapy", "counseling", "support", "insight", "awareness"
            ],
            "advanced": [
                "transference", "countertransference", "unconscious", "defense mechanism",
                "attachment", "trauma", "dissociation", "cognitive distortion", "schema",
                "mindfulness", "dialectical", "psychodynamic", "behavioral activation",
                "exposure therapy", "cognitive restructuring", "therapeutic alliance"
            ],
            "expert": [
                "mentalization", "affect regulation", "interpersonal neurobiology",
                "somatic experiencing", "EMDR", "internal family systems", "polyvagal",
                "neuroplasticity", "epigenetics", "attachment theory", "object relations",
                "self psychology", "relational psychoanalysis", "existential therapy"
            ]
        }

        self.therapeutic_techniques = {
            "reflection": [r"\b(reflect|mirror|paraphrase|summarize)\b"],
            "questioning": [r"\b(what|how|when|where|why|tell me about)\b"],
            "interpretation": [r"\b(it seems|perhaps|maybe|could it be|I wonder)\b"],
            "validation": [r"\b(understand|makes sense|valid|legitimate)\b"],
            "challenge": [r"\b(evidence|alternative|different perspective|consider)\b"],
            "psychoeducation": [r"\b(research shows|studies indicate|common|typical)\b"]
        }

    def analyze_therapeutic_depth(self, conversation: dict[str, Any]) -> dict[str, float]:
        """Analyze therapeutic depth of conversation."""
        text = self._extract_text(conversation)

        if not text.strip():
            return self._empty_depth_metrics()

        metrics = {}

        # Analyze concept sophistication
        concept_scores = self._analyze_concept_sophistication(text)
        metrics.update(concept_scores)

        # Analyze therapeutic techniques
        technique_scores = self._analyze_therapeutic_techniques(text)
        metrics.update(technique_scores)

        # Calculate overall therapeutic depth
        metrics["therapeutic_depth"] = self._calculate_overall_depth(metrics)

        return metrics

    def _extract_text(self, conversation: dict[str, Any]) -> str:
        """Extract text from conversation."""
        text_parts = []

        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict) and "content" in message:
                    text_parts.append(message["content"])

        return " ".join(text_parts)

    def _empty_depth_metrics(self) -> dict[str, float]:
        """Return empty depth metrics."""
        return {
            "basic_concepts": 0.0,
            "intermediate_concepts": 0.0,
            "advanced_concepts": 0.0,
            "expert_concepts": 0.0,
            "technique_diversity": 0.0,
            "therapeutic_depth": 0.0
        }

    def _analyze_concept_sophistication(self, text: str) -> dict[str, float]:
        """Analyze sophistication of therapeutic concepts used."""
        text_lower = text.lower()
        word_count = len(text.split())

        concept_scores = {}

        for level, concepts in self.therapeutic_concepts.items():
            count = sum(1 for concept in concepts if concept in text_lower)
            # Normalize by text length and concept list size
            normalized_score = (count / max(word_count / 100, 1)) * (1 / len(concepts))
            concept_scores[f"{level}_concepts"] = min(normalized_score, 1.0)

        return concept_scores

    def _analyze_therapeutic_techniques(self, text: str) -> dict[str, float]:
        """Analyze therapeutic techniques present in text."""
        technique_counts = {}

        for technique, patterns in self.therapeutic_techniques.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                count += len(matches)
            technique_counts[technique] = count

        # Calculate technique diversity
        used_techniques = sum(1 for count in technique_counts.values() if count > 0)
        technique_diversity = used_techniques / len(self.therapeutic_techniques)

        return {"technique_diversity": technique_diversity}

    def _calculate_overall_depth(self, metrics: dict[str, float]) -> float:
        """Calculate overall therapeutic depth score."""
        # Weight different levels of concepts
        concept_score = (
            metrics.get("basic_concepts", 0) * 0.1 +
            metrics.get("intermediate_concepts", 0) * 0.3 +
            metrics.get("advanced_concepts", 0) * 0.4 +
            metrics.get("expert_concepts", 0) * 0.2
        )

        technique_score = metrics.get("technique_diversity", 0)

        # Combine concept sophistication and technique diversity
        overall_depth = (concept_score * 0.7 + technique_score * 0.3)

        return min(overall_depth, 1.0)


class EmotionalComplexityAnalyzer:
    """Analyzes emotional complexity and range."""

    def __init__(self):
        self.emotion_categories = {
            "basic_emotions": [
                "happy", "sad", "angry", "afraid", "surprised", "disgusted"
            ],
            "complex_emotions": [
                "anxious", "depressed", "frustrated", "overwhelmed", "hopeful",
                "grateful", "ashamed", "guilty", "proud", "envious", "jealous",
                "contempt", "admiration", "compassion", "empathy"
            ],
            "emotional_states": [
                "numb", "empty", "conflicted", "ambivalent", "vulnerable",
                "resilient", "fragile", "stable", "volatile", "balanced"
            ],
            "emotional_processes": [
                "processing", "working through", "struggling with", "coming to terms",
                "letting go", "holding onto", "suppressing", "expressing",
                "regulating", "managing", "coping with"
            ]
        }

        self.intensity_modifiers = [
            "very", "extremely", "incredibly", "somewhat", "slightly",
            "deeply", "profoundly", "mildly", "intensely", "overwhelmingly"
        ]

    def analyze_emotional_complexity(self, conversation: dict[str, Any]) -> dict[str, float]:
        """Analyze emotional complexity of conversation."""
        text = self._extract_text(conversation)

        if not text.strip():
            return self._empty_emotional_metrics()

        metrics = {}

        # Analyze emotional vocabulary
        emotional_vocab = self._analyze_emotional_vocabulary(text)
        metrics.update(emotional_vocab)

        # Analyze emotional range
        metrics["emotional_range"] = self._calculate_emotional_range(text)

        # Analyze emotional intensity
        metrics["emotional_intensity"] = self._analyze_emotional_intensity(text)

        # Analyze emotional progression
        metrics["emotional_progression"] = self._analyze_emotional_progression(conversation)

        # Calculate overall emotional complexity
        metrics["emotional_complexity"] = self._calculate_emotional_complexity(metrics)

        return metrics

    def _extract_text(self, conversation: dict[str, Any]) -> str:
        """Extract text from conversation."""
        text_parts = []

        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict) and "content" in message:
                    text_parts.append(message["content"])

        return " ".join(text_parts)

    def _empty_emotional_metrics(self) -> dict[str, float]:
        """Return empty emotional metrics."""
        return {
            "basic_emotions": 0.0,
            "complex_emotions": 0.0,
            "emotional_states": 0.0,
            "emotional_processes": 0.0,
            "emotional_range": 0.0,
            "emotional_intensity": 0.0,
            "emotional_progression": 0.0,
            "emotional_complexity": 0.0
        }

    def _analyze_emotional_vocabulary(self, text: str) -> dict[str, float]:
        """Analyze emotional vocabulary sophistication."""
        text_lower = text.lower()
        word_count = len(text.split())

        vocab_scores = {}

        for category, emotions in self.emotion_categories.items():
            count = sum(1 for emotion in emotions if emotion in text_lower)
            # Normalize by text length
            normalized_score = count / max(word_count / 100, 1)
            vocab_scores[category] = min(normalized_score, 1.0)

        return vocab_scores

    def _calculate_emotional_range(self, text: str) -> float:
        """Calculate emotional range (diversity of emotions)."""
        text_lower = text.lower()

        total_emotions_found = 0
        unique_emotions = set()

        for _category, emotions in self.emotion_categories.items():
            for emotion in emotions:
                if emotion in text_lower:
                    unique_emotions.add(emotion)
                    total_emotions_found += text_lower.count(emotion)

        # Calculate range as ratio of unique emotions to total emotion words
        if total_emotions_found == 0:
            return 0.0

        emotional_range = len(unique_emotions) / max(total_emotions_found, 1)
        return min(emotional_range, 1.0)

    def _analyze_emotional_intensity(self, text: str) -> float:
        """Analyze emotional intensity based on modifiers."""
        text_lower = text.lower()

        intensity_count = sum(1 for modifier in self.intensity_modifiers if modifier in text_lower)
        word_count = len(text.split())

        # Normalize by text length
        intensity_score = intensity_count / max(word_count / 50, 1)
        return min(intensity_score, 1.0)

    def _analyze_emotional_progression(self, conversation: dict[str, Any]) -> float:
        """Analyze emotional progression throughout conversation."""
        messages = conversation.get("messages", [])

        if len(messages) < 3:
            return 0.0

        # Simple emotional progression analysis
        emotional_scores = []

        for message in messages:
            if isinstance(message, dict) and "content" in message:
                content = message["content"].lower()

                # Count positive vs negative emotional words
                positive_words = ["happy", "good", "better", "hopeful", "grateful", "proud"]
                negative_words = ["sad", "bad", "worse", "hopeless", "angry", "frustrated"]

                pos_count = sum(1 for word in positive_words if word in content)
                neg_count = sum(1 for word in negative_words if word in content)

                if pos_count + neg_count > 0:
                    emotional_scores.append((pos_count - neg_count) / (pos_count + neg_count))
                else:
                    emotional_scores.append(0)

        if len(emotional_scores) < 2:
            return 0.0

        # Calculate variance in emotional scores (progression complexity)
        progression_score = np.var(emotional_scores) if emotional_scores else 0
        return min(progression_score, 1.0)

    def _calculate_emotional_complexity(self, metrics: dict[str, float]) -> float:
        """Calculate overall emotional complexity."""
        # Weight different aspects of emotional complexity
        complexity = (
            metrics.get("basic_emotions", 0) * 0.1 +
            metrics.get("complex_emotions", 0) * 0.3 +
            metrics.get("emotional_states", 0) * 0.2 +
            metrics.get("emotional_processes", 0) * 0.2 +
            metrics.get("emotional_range", 0) * 0.1 +
            metrics.get("emotional_intensity", 0) * 0.05 +
            metrics.get("emotional_progression", 0) * 0.05
        )

        return min(complexity, 1.0)


class ComplexityScorer:
    """Main complexity scoring system."""

    def __init__(self):
        self.linguistic_analyzer = LinguisticComplexityAnalyzer()
        self.therapeutic_analyzer = TherapeuticDepthAnalyzer()
        self.emotional_analyzer = EmotionalComplexityAnalyzer()

        self.logger = logging.getLogger(__name__)

    def score_conversation_complexity(self, conversation: dict[str, Any]) -> ComplexityMetrics:
        """Score overall complexity of a conversation."""
        try:
            # Extract basic statistics
            text = self._extract_full_text(conversation)
            messages = conversation.get("messages", [])

            word_count = len(text.split())
            sentence_count = text.count(".") + text.count("!") + text.count("?")
            turn_count = len(messages)
            unique_words = len(set(text.lower().split()))

            # Analyze different complexity dimensions
            linguistic_metrics = self.linguistic_analyzer.analyze_linguistic_complexity(text)
            therapeutic_metrics = self.therapeutic_analyzer.analyze_therapeutic_depth(conversation)
            emotional_metrics = self.emotional_analyzer.analyze_emotional_complexity(conversation)

            # Calculate main complexity scores
            linguistic_complexity = self._calculate_linguistic_complexity(linguistic_metrics)
            therapeutic_depth = therapeutic_metrics.get("therapeutic_depth", 0.0)
            emotional_complexity = emotional_metrics.get("emotional_complexity", 0.0)

            # Calculate additional complexity dimensions
            clinical_sophistication = self._calculate_clinical_sophistication(text)
            cognitive_load = self._calculate_cognitive_load(linguistic_metrics, word_count)
            interaction_complexity = self._calculate_interaction_complexity(messages)

            # Create a temporary ComplexityMetrics object for overall calculation
            temp_metrics = ComplexityMetrics(
                overall_complexity=0.0, # Placeholder, will be overwritten
                linguistic_complexity=linguistic_complexity,
                therapeutic_depth=therapeutic_depth,
                emotional_complexity=emotional_complexity,
                clinical_sophistication=clinical_sophistication,
                cognitive_load=cognitive_load,
                interaction_complexity=interaction_complexity,
                readability_score=0.0, vocabulary_diversity=0.0, syntactic_complexity=0.0,
                semantic_density=0.0, therapeutic_technique_count=0, emotional_range=0.0,
                clinical_terminology_density=0.0, turn_taking_complexity=0.0,
                word_count=0, sentence_count=0, turn_count=0, unique_words=0
            )

            # Calculate overall complexity
            overall_complexity = self._calculate_overall_complexity(temp_metrics)

            return ComplexityMetrics(
                overall_complexity=overall_complexity,
                linguistic_complexity=linguistic_complexity,
                therapeutic_depth=therapeutic_depth,
                emotional_complexity=emotional_complexity,
                clinical_sophistication=clinical_sophistication,
                cognitive_load=cognitive_load,
                interaction_complexity=interaction_complexity,

                # Detailed sub-metrics
                readability_score=self._normalize_readability(linguistic_metrics.get("flesch_reading_ease", 50)),
                vocabulary_diversity=linguistic_metrics.get("vocabulary_diversity", 0.0),
                syntactic_complexity=linguistic_metrics.get("syntactic_complexity", 0.0),
                semantic_density=linguistic_metrics.get("semantic_density", 0.0),
                therapeutic_technique_count=int(therapeutic_metrics.get("technique_diversity", 0) * 6),
                emotional_range=emotional_metrics.get("emotional_range", 0.0),
                clinical_terminology_density=clinical_sophistication,
                turn_taking_complexity=interaction_complexity,

                # Basic statistics
                word_count=word_count,
                sentence_count=max(sentence_count, 1),
                turn_count=turn_count,
                unique_words=unique_words
            )

        except Exception as e:
            self.logger.error(f"Error scoring conversation complexity: {e}")
            return self._empty_complexity_metrics()

    def _extract_full_text(self, conversation: dict[str, Any]) -> str:
        """Extract all text from conversation."""
        text_parts = []

        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict) and "content" in message:
                    text_parts.append(message["content"])

        return " ".join(text_parts)

    def _calculate_linguistic_complexity(self, metrics: dict[str, float]) -> float:
        """Calculate overall linguistic complexity."""
        # Combine different linguistic metrics
        readability = self._normalize_readability(metrics.get("flesch_reading_ease", 50))
        vocabulary = metrics.get("vocabulary_diversity", 0.0)
        syntactic = metrics.get("syntactic_complexity", 0.0)
        semantic = metrics.get("semantic_density", 0.0)

        # Weight the components
        complexity = (
            readability * 0.3 +
            vocabulary * 0.3 +
            syntactic * 0.2 +
            semantic * 0.2
        )

        return min(complexity, 1.0)

    def _normalize_readability(self, flesch_score: float) -> float:
        """Normalize Flesch reading ease to 0-1 complexity scale."""
        # Flesch: 100 = very easy, 0 = very hard
        # Convert to complexity: 0 = simple, 1 = complex
        normalized = (100 - flesch_score) / 100
        return max(0, min(normalized, 1))

    def _calculate_clinical_sophistication(self, text: str) -> float:
        """Calculate clinical sophistication based on terminology."""
        clinical_terms = [
            "diagnosis", "symptoms", "treatment", "therapy", "intervention",
            "assessment", "evaluation", "disorder", "syndrome", "pathology",
            "etiology", "prognosis", "comorbidity", "differential", "criteria"
        ]

        text_lower = text.lower()
        word_count = len(text.split())

        clinical_count = sum(1 for term in clinical_terms if term in text_lower)

        # Normalize by text length
        sophistication = clinical_count / max(word_count / 100, 1)
        return min(sophistication, 1.0)

    def _calculate_cognitive_load(self, linguistic_metrics: dict[str, float], word_count: int) -> float:
        """Calculate cognitive load based on text complexity."""
        # Factors that increase cognitive load
        readability_load = self._normalize_readability(linguistic_metrics.get("flesch_reading_ease", 50))
        length_load = min(word_count / 1000, 1.0)  # Longer texts = higher load
        vocabulary_load = linguistic_metrics.get("vocabulary_diversity", 0.0)

        # Combine factors
        cognitive_load = (readability_load * 0.4 + length_load * 0.3 + vocabulary_load * 0.3)
        return min(cognitive_load, 1.0)

    def _calculate_interaction_complexity(self, messages: list[dict[str, Any]]) -> float:
        """Calculate interaction complexity based on conversation structure."""
        if len(messages) < 2:
            return 0.0

        # Analyze turn lengths
        turn_lengths = []
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                turn_lengths.append(len(message["content"].split()))

        if not turn_lengths:
            return 0.0

        # Calculate turn length variance (higher variance = more complex interaction)
        length_variance = np.var(turn_lengths) if len(turn_lengths) > 1 else 0
        avg_length = np.mean(turn_lengths)

        # Normalize variance by average length
        normalized_variance = length_variance / max(avg_length ** 2, 1)

        # Factor in number of turns
        turn_factor = min(len(messages) / 20, 1.0)

        interaction_complexity = (normalized_variance * 0.6 + turn_factor * 0.4)
        return min(interaction_complexity, 1.0)

    def _calculate_overall_complexity(self, metrics: ComplexityMetrics) -> float:
        """Calculate overall complexity score."""
        # Weight different complexity dimensions
        overall = (
            metrics.linguistic_complexity * 0.25 +
            metrics.therapeutic_depth * 0.25 +
            metrics.emotional_complexity * 0.20 +
            metrics.clinical_sophistication * 0.15 +
            metrics.cognitive_load * 0.10 +
            metrics.interaction_complexity * 0.05
        )

        return min(overall, 1.0)

    def _empty_complexity_metrics(self) -> ComplexityMetrics:
        """Return empty complexity metrics for error cases."""
        return ComplexityMetrics(
            overall_complexity=0.0,
            linguistic_complexity=0.0,
            therapeutic_depth=0.0,
            emotional_complexity=0.0,
            clinical_sophistication=0.0,
            cognitive_load=0.0,
            interaction_complexity=0.0,
            readability_score=0.0,
            vocabulary_diversity=0.0,
            syntactic_complexity=0.0,
            semantic_density=0.0,
            therapeutic_technique_count=0,
            emotional_range=0.0,
            clinical_terminology_density=0.0,
            turn_taking_complexity=0.0,
            word_count=0,
            sentence_count=0,
            turn_count=0,
            unique_words=0
        )


# Example usage
if __name__ == "__main__":
    # Create complexity scorer
    scorer = ComplexityScorer()

    # Example conversation
    example_conversation = {
        "id": "complexity_example",
        "messages": [
            {
                "role": "client",
                "content": "I've been experiencing persistent depressive symptoms for several months now. The cognitive distortions seem to be getting worse, and I'm struggling with rumination patterns that feel overwhelming."
            },
            {
                "role": "therapist",
                "content": "I appreciate you sharing that with me. When you mention cognitive distortions and rumination patterns, it sounds like you have some awareness of your thought processes. Can you help me understand what specific types of distorted thinking you've noticed? For instance, are you experiencing catastrophic thinking, all-or-nothing patterns, or perhaps mind reading?"
            },
            {
                "role": "client",
                "content": "Definitely catastrophic thinking. Every small setback feels like evidence that I'm fundamentally flawed. I also notice a lot of emotional reasoning - if I feel worthless, then I must be worthless. It's like my emotions are dictating my reality."
            },
            {
                "role": "therapist",
                "content": "That's a really insightful observation about emotional reasoning. The fact that you can identify these patterns suggests you're developing metacognitive awareness, which is actually a significant therapeutic asset. Let's explore this further - when you notice these catastrophic thoughts arising, what happens in your body? Are there somatic markers that accompany these cognitive patterns?"
            }
        ]
    }

    # Score complexity
    complexity = scorer.score_conversation_complexity(example_conversation)

    # Print results

