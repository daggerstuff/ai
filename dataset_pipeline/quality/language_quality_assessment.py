"""
Language quality assessment system for conversation datasets.

This module provides comprehensive linguistic analysis including readability,
lexical diversity, grammar quality, vocabulary appropriateness, and other
linguistic metrics to assess the quality of conversational text.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from conversation_schema import Conversation, Message

# Set up logging
logger = logging.getLogger(__name__)


class LanguageComplexity(Enum):
    """Language complexity levels."""
    VERY_SIMPLE = "very_simple"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class LanguageQualityMetrics:
    """Metrics for language quality assessment."""
    overall_score: float
    readability_score: float
    lexical_diversity_score: float
    grammar_quality_score: float
    vocabulary_appropriateness_score: float
    sentence_complexity_score: float
    coherence_score: float
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    complexity_level: LanguageComplexity = LanguageComplexity.MODERATE
    quality_level: str = ""


class LanguageQualityAssessor:
    """
    Comprehensive language quality assessment system.

    Evaluates conversations across multiple linguistic dimensions:
    1. Readability - ease of reading and comprehension
    2. Lexical diversity - vocabulary richness and variety
    3. Grammar quality - grammatical correctness and structure
    4. Vocabulary appropriateness - suitable word choice for context
    5. Sentence complexity - structural sophistication
    6. Coherence - linguistic flow and connection
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the assessor with configuration."""
        self.config = config or {}

        # Default weights for different assessment dimensions
        self.weights = self.config.get("weights", {
            "readability": 0.20,
            "lexical_diversity": 0.18,
            "grammar_quality": 0.18,
            "vocabulary_appropriateness": 0.16,
            "sentence_complexity": 0.14,
            "coherence": 0.14
        })

        # Quality thresholds
        self.thresholds = self.config.get("thresholds", {
            "excellent": 0.85,
            "good": 0.70,
            "acceptable": 0.55,
            "poor": 0.40
        })

        # Initialize linguistic knowledge bases
        self._initialize_linguistic_resources()

    def _initialize_linguistic_resources(self):
        """Initialize linguistic resources and patterns."""
        # Common words for lexical diversity calculation
        self.common_words = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their",
            "what", "so", "up", "out", "if", "about", "who", "get", "which", "go"
        }

        # Grammar error patterns (simplified)
        self.grammar_error_patterns = [
            r"\bi\s+am\s+went\b",  # "I am went"
            r"\bhe\s+don\'t\b",    # "he don't"
            r"\bshe\s+don\'t\b",   # "she don't"
            r"\bit\s+don\'t\b",    # "it don't"
            r"\bwould\s+of\b",     # "would of"
            r"\bcould\s+of\b",     # "could of"
            r"\bshould\s+of\b",    # "should of"
            r"\bthere\s+is\s+\w+s\b",  # "there is cats" (simplified)
            r"\ba\s+\w*s\b",       # "a cats" (simplified)
        ]

        # Sophisticated vocabulary indicators
        self.sophisticated_words = {
            "furthermore", "nevertheless", "consequently", "therefore", "however",
            "moreover", "specifically", "particularly", "essentially", "fundamentally",
            "comprehensive", "substantial", "significant", "considerable", "extensive",
            "demonstrate", "illustrate", "emphasize", "acknowledge", "recognize",
            "facilitate", "implement", "establish", "maintain", "contribute",
            "perspective", "approach", "methodology", "framework", "concept",
            "analyze", "evaluate", "assess", "examine", "investigate"
        }

        # Informal/inappropriate words for certain contexts
        self.informal_words = {
            "gonna", "wanna", "gotta", "kinda", "sorta", "dunno", "yeah", "nah",
            "ok", "okay", "cool", "awesome", "dude", "guy", "stuff", "things",
            "like", "totally", "really", "super", "pretty", "quite", "very"
        }

        # Sentence starters for variety assessment
        self.sentence_starters = {
            "question": ["what", "how", "why", "when", "where", "who", "which"],
            "conditional": ["if", "unless", "provided", "assuming"],
            "temporal": ["when", "while", "after", "before", "during", "since"],
            "causal": ["because", "since", "as", "due to", "owing to"],
            "contrast": ["although", "though", "while", "whereas", "despite"]
        }

        # Transition words for coherence
        self.transition_words = {
            "addition": ["also", "furthermore", "moreover", "additionally", "besides"],
            "contrast": ["however", "nevertheless", "nonetheless", "conversely", "on the other hand"],
            "sequence": ["first", "second", "then", "next", "finally", "subsequently"],
            "example": ["for example", "for instance", "such as", "namely", "specifically"],
            "conclusion": ["therefore", "thus", "consequently", "as a result", "in conclusion"]
        }

    def assess_language_quality(self, conversation: Conversation) -> LanguageQualityMetrics:
        """
        Assess the language quality of a conversation.

        Args:
            conversation: The conversation to assess

        Returns:
            LanguageQualityMetrics with detailed assessment results
        """
        logger.info(f"Assessing language quality for conversation {conversation.id}")

        if len(conversation.messages) < 2:
            return LanguageQualityMetrics(
                overall_score=0.0,
                readability_score=0.0,
                lexical_diversity_score=0.0,
                grammar_quality_score=0.0,
                vocabulary_appropriateness_score=0.0,
                sentence_complexity_score=0.0,
                coherence_score=0.0,
                issues=["Insufficient messages for language quality assessment"],
                details={"conversation_length": len(conversation.messages)},
                complexity_level=LanguageComplexity.VERY_SIMPLE,
                quality_level="very_poor"
            )

        # Extract text content
        text_content = self._extract_text_content(conversation.messages)

        # Assess different dimensions
        readability = self._assess_readability(text_content)
        lexical_diversity = self._assess_lexical_diversity(text_content)
        grammar_quality = self._assess_grammar_quality(text_content)
        vocabulary_appropriateness = self._assess_vocabulary_appropriateness(text_content)
        sentence_complexity = self._assess_sentence_complexity(text_content)
        coherence = self._assess_coherence(conversation.messages)

        # Calculate weighted overall score
        overall_score = (
            readability["score"] * self.weights["readability"] +
            lexical_diversity["score"] * self.weights["lexical_diversity"] +
            grammar_quality["score"] * self.weights["grammar_quality"] +
            vocabulary_appropriateness["score"] * self.weights["vocabulary_appropriateness"] +
            sentence_complexity["score"] * self.weights["sentence_complexity"] +
            coherence["score"] * self.weights["coherence"]
        )

        # Compile all issues and warnings
        all_issues = []
        all_warnings = []

        for assessment in [readability, lexical_diversity, grammar_quality,
                          vocabulary_appropriateness, sentence_complexity, coherence]:
            all_issues.extend(assessment.get("issues", []))
            all_warnings.extend(assessment.get("warnings", []))

        # Determine complexity level
        complexity_level = self._determine_complexity_level(
            readability["details"], lexical_diversity["details"],
            sentence_complexity["details"]
        )

        # Compile detailed results
        details = {
            "conversation_length": len(conversation.messages),
            "total_words": len(text_content.split()),
            "total_sentences": len([s for s in text_content.split(".") if s.strip()]),
            "readability_details": readability.get("details", {}),
            "lexical_diversity_details": lexical_diversity.get("details", {}),
            "grammar_quality_details": grammar_quality.get("details", {}),
            "vocabulary_appropriateness_details": vocabulary_appropriateness.get("details", {}),
            "sentence_complexity_details": sentence_complexity.get("details", {}),
            "coherence_details": coherence.get("details", {}),
            "quality_level": self._determine_quality_level(overall_score)
        }

        return LanguageQualityMetrics(
            overall_score=overall_score,
            readability_score=readability["score"],
            lexical_diversity_score=lexical_diversity["score"],
            grammar_quality_score=grammar_quality["score"],
            vocabulary_appropriateness_score=vocabulary_appropriateness["score"],
            sentence_complexity_score=sentence_complexity["score"],
            coherence_score=coherence["score"],
            issues=all_issues,
            warnings=all_warnings,
            details=details,
            complexity_level=complexity_level,
            quality_level=self._determine_quality_level(overall_score)
        )

    def _extract_text_content(self, messages: list[Message]) -> str:
        """Extract all text content from messages."""
        return " ".join(message.content for message in messages)

    def _determine_complexity_level(self, readability_details: dict,
                                   lexical_details: dict,
                                   complexity_details: dict) -> LanguageComplexity:
        """Determine overall language complexity level."""
        # Simple scoring based on various metrics
        complexity_score = 0

        # Readability contribution
        avg_sentence_length = readability_details.get("avg_sentence_length", 10)
        if avg_sentence_length > 25:
            complexity_score += 2
        elif avg_sentence_length > 15:
            complexity_score += 1

        # Lexical diversity contribution
        ttr = lexical_details.get("type_token_ratio", 0.5)
        if ttr > 0.8:
            complexity_score += 2
        elif ttr > 0.6:
            complexity_score += 1

        # Sentence complexity contribution
        complex_sentences = complexity_details.get("complex_sentence_ratio", 0.3)
        if complex_sentences > 0.6:
            complexity_score += 2
        elif complex_sentences > 0.4:
            complexity_score += 1

        # Map score to complexity level
        if complexity_score >= 5:
            return LanguageComplexity.VERY_COMPLEX
        if complexity_score >= 4:
            return LanguageComplexity.COMPLEX
        if complexity_score >= 2:
            return LanguageComplexity.MODERATE
        if complexity_score >= 1:
            return LanguageComplexity.SIMPLE
        return LanguageComplexity.VERY_SIMPLE

    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on score."""
        if score >= self.thresholds["excellent"]:
            return "excellent"
        if score >= self.thresholds["good"]:
            return "good"
        if score >= self.thresholds["acceptable"]:
            return "acceptable"
        if score >= self.thresholds["poor"]:
            return "poor"
        return "very_poor"

    def _assess_readability(self, text: str) -> dict[str, Any]:
        """Assess readability using multiple metrics."""
        issues = []
        warnings = []

        # Basic text statistics
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        words = text.split()

        if not sentences or not words:
            return {
                "score": 0.0,
                "issues": ["No readable content found"],
                "details": {"avg_sentence_length": 0, "avg_word_length": 0}
            }

        # Calculate basic metrics
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word.strip('.,!?;:"()[]')) for word in words) / len(words)

        # Flesch Reading Ease approximation
        syllable_count = sum(self._count_syllables(word) for word in words)
        if len(sentences) > 0 and len(words) > 0:
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / len(words)))
        else:
            flesch_score = 0

        # Normalize Flesch score to 0-1 range
        readability_score = max(0.0, min(1.0, flesch_score / 100))

        # Assess readability issues
        if avg_sentence_length > 30:
            issues.append("Very long sentences may reduce readability")
        elif avg_sentence_length > 20:
            warnings.append("Long sentences detected - consider breaking them up")

        if avg_word_length > 6:
            warnings.append("Complex vocabulary may affect readability")

        if flesch_score < 30:
            issues.append("Text is very difficult to read")
        elif flesch_score < 50:
            warnings.append("Text may be difficult for some readers")

        return {
            "score": readability_score,
            "issues": issues,
            "warnings": warnings,
            "details": {
                "avg_sentence_length": avg_sentence_length,
                "avg_word_length": avg_word_length,
                "flesch_score": flesch_score,
                "total_sentences": len(sentences),
                "total_words": len(words),
                "syllable_count": syllable_count
            }
        }

    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for a word."""
        word = word.lower().strip('.,!?;:"()[]')
        if not word:
            return 0

        # Simple syllable counting heuristic
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel

        # Handle silent 'e'
        if word.endswith("e") and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)

    def _assess_lexical_diversity(self, text: str) -> dict[str, Any]:
        """Assess lexical diversity and vocabulary richness."""
        issues = []
        warnings = []

        words = [word.lower().strip('.,!?;:"()[]') for word in text.split() if word.strip()]

        if len(words) < 10:
            return {
                "score": 0.0,
                "issues": ["Insufficient text for lexical diversity analysis"],
                "details": {"type_token_ratio": 0, "unique_words": 0, "total_words": len(words)}
            }

        # Calculate Type-Token Ratio (TTR)
        unique_words = set(words)
        ttr = len(unique_words) / len(words)

        # Calculate sophisticated vocabulary ratio
        sophisticated_count = sum(1 for word in words if word in self.sophisticated_words)
        sophisticated_ratio = sophisticated_count / len(words)

        # Calculate common word ratio
        common_count = sum(1 for word in words if word in self.common_words)
        common_ratio = common_count / len(words)

        # Calculate lexical diversity score
        diversity_score = ttr * 0.6 + sophisticated_ratio * 0.3 + (1 - common_ratio) * 0.1
        diversity_score = max(0.0, min(1.0, diversity_score))

        # Assess diversity issues
        if ttr < 0.3:
            issues.append("Very low lexical diversity - repetitive vocabulary")
        elif ttr < 0.5:
            warnings.append("Low lexical diversity - consider varying vocabulary")

        if sophisticated_ratio < 0.02:
            warnings.append("Limited use of sophisticated vocabulary")

        if common_ratio > 0.7:
            warnings.append("High proportion of common words")

        return {
            "score": diversity_score,
            "issues": issues,
            "warnings": warnings,
            "details": {
                "type_token_ratio": ttr,
                "unique_words": len(unique_words),
                "total_words": len(words),
                "sophisticated_word_ratio": sophisticated_ratio,
                "common_word_ratio": common_ratio,
                "sophisticated_words_used": sophisticated_count
            }
        }

    def _assess_grammar_quality(self, text: str) -> dict[str, Any]:
        """Assess grammar quality using pattern matching."""
        issues = []
        warnings = []

        # Count potential grammar errors
        error_count = 0
        detected_errors = []

        for pattern in self.grammar_error_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                error_count += len(matches)
                detected_errors.extend(matches)

        # Calculate grammar score
        words = text.split()
        if len(words) == 0:
            return {
                "score": 0.0,
                "issues": ["No text to analyze for grammar"],
                "details": {"error_count": 0, "error_rate": 0}
            }

        error_rate = error_count / len(words)
        grammar_score = max(0.0, 1.0 - (error_rate * 10))  # Penalize errors heavily

        # Assess grammar issues
        if error_rate > 0.05:
            issues.append("High frequency of grammar errors detected")
        elif error_rate > 0.02:
            warnings.append("Some grammar errors detected")

        # Check for basic punctuation
        has_periods = "." in text
        has_proper_capitalization = any(c.isupper() for c in text)

        if not has_periods and len(text) > 50:
            warnings.append("Missing sentence-ending punctuation")

        if not has_proper_capitalization:
            warnings.append("Inconsistent capitalization")

        return {
            "score": grammar_score,
            "issues": issues,
            "warnings": warnings,
            "details": {
                "error_count": error_count,
                "error_rate": error_rate,
                "detected_errors": detected_errors[:5],  # Limit to first 5
                "has_proper_punctuation": has_periods,
                "has_proper_capitalization": has_proper_capitalization
            }
        }

    def _assess_vocabulary_appropriateness(self, text: str) -> dict[str, Any]:
        """Assess vocabulary appropriateness for the context."""
        issues = []
        warnings = []

        words = [word.lower().strip('.,!?;:"()[]') for word in text.split() if word.strip()]

        if not words:
            return {
                "score": 0.0,
                "issues": ["No words to analyze"],
                "details": {"informal_word_ratio": 0, "appropriate_vocabulary_score": 0}
            }

        # Count informal words
        informal_count = sum(1 for word in words if word in self.informal_words)
        informal_ratio = informal_count / len(words)

        # Count sophisticated words
        sophisticated_count = sum(1 for word in words if word in self.sophisticated_words)
        sophisticated_ratio = sophisticated_count / len(words)

        # Calculate appropriateness score
        appropriateness_score = max(0.0, 1.0 - informal_ratio * 2 + sophisticated_ratio * 0.5)
        appropriateness_score = min(1.0, appropriateness_score)

        # Assess vocabulary issues
        if informal_ratio > 0.15:
            issues.append("High use of informal language")
        elif informal_ratio > 0.08:
            warnings.append("Some informal language detected")

        if sophisticated_ratio < 0.01 and len(words) > 50:
            warnings.append("Limited sophisticated vocabulary usage")

        return {
            "score": appropriateness_score,
            "issues": issues,
            "warnings": warnings,
            "details": {
                "informal_word_ratio": informal_ratio,
                "sophisticated_word_ratio": sophisticated_ratio,
                "informal_words_count": informal_count,
                "sophisticated_words_count": sophisticated_count,
                "appropriate_vocabulary_score": appropriateness_score
            }
        }

    def _assess_sentence_complexity(self, text: str) -> dict[str, Any]:
        """Assess sentence complexity and structure variety."""
        issues = []
        warnings = []

        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        if not sentences:
            return {
                "score": 0.0,
                "issues": ["No sentences to analyze"],
                "details": {"complex_sentence_ratio": 0, "sentence_variety_score": 0}
            }

        # Analyze sentence complexity
        complex_sentences = 0
        sentence_starters = []

        for sentence in sentences:
            words = sentence.split()
            if not words:
                continue

            # Count complex sentences (with conjunctions, relative clauses, etc.)
            complexity_indicators = ["because", "although", "since", "while", "whereas",
                                   "however", "therefore", "moreover", "furthermore",
                                   "which", "that", "who", "whom", "whose"]

            if any(indicator in sentence.lower() for indicator in complexity_indicators):
                complex_sentences += 1

            # Track sentence starters for variety
            first_word = words[0].lower().strip('.,!?;:"()[]')
            sentence_starters.append(first_word)

        complex_sentence_ratio = complex_sentences / len(sentences)

        # Calculate sentence variety
        unique_starters = len(set(sentence_starters))
        variety_score = unique_starters / len(sentences) if sentences else 0

        # Overall complexity score
        complexity_score = complex_sentence_ratio * 0.6 + variety_score * 0.4

        # Assess complexity issues
        if complex_sentence_ratio < 0.1:
            warnings.append("Very simple sentence structures - consider adding complexity")
        elif complex_sentence_ratio > 0.8:
            warnings.append("Very complex sentences - may affect readability")

        if variety_score < 0.3:
            warnings.append("Limited sentence structure variety")

        return {
            "score": complexity_score,
            "issues": issues,
            "warnings": warnings,
            "details": {
                "complex_sentence_ratio": complex_sentence_ratio,
                "sentence_variety_score": variety_score,
                "total_sentences": len(sentences),
                "complex_sentences": complex_sentences,
                "unique_sentence_starters": unique_starters
            }
        }

    def _assess_coherence(self, messages: list[Message]) -> dict[str, Any]:
        """Assess linguistic coherence and flow between messages."""
        issues = []
        warnings = []

        if len(messages) < 2:
            return {
                "score": 0.0,
                "issues": ["Insufficient messages for coherence analysis"],
                "details": {"transition_usage": 0, "pronoun_reference_score": 0}
            }

        # Analyze transition word usage
        transition_count = 0
        total_transitions_possible = 0

        # Analyze pronoun references and coherence
        pronoun_references = 0
        unclear_references = 0

        for i in range(1, len(messages)):
            current_content = messages[i].content.lower()
            previous_content = messages[i-1].content.lower()

            # Count transition words
            total_transitions_possible += 1
            has_transition = any(
                any(word in current_content for word in words)
                for words in self.transition_words.values()
            )
            if has_transition:
                transition_count += 1

            # Analyze pronoun references
            pronouns = ["it", "this", "that", "these", "those", "they", "them"]
            for pronoun in pronouns:
                if pronoun in current_content:
                    pronoun_references += 1
                    # Simple heuristic: if pronoun appears but no clear antecedent in previous message
                    if len(previous_content.split()) < 3:
                        unclear_references += 1

        # Calculate coherence scores
        transition_score = transition_count / max(1, total_transitions_possible)
        pronoun_score = 1.0 - (unclear_references / max(1, pronoun_references))

        # Overall coherence score
        coherence_score = transition_score * 0.4 + pronoun_score * 0.6

        # Assess coherence issues
        if transition_score < 0.2:
            warnings.append("Limited use of transition words - may affect flow")

        if unclear_references > pronoun_references * 0.3:
            warnings.append("Some unclear pronoun references detected")

        # Check for topic continuity (simplified)
        topic_shifts = 0
        for i in range(1, len(messages)):
            current_words = set(messages[i].content.lower().split())
            previous_words = set(messages[i-1].content.lower().split())

            # Simple overlap check
            overlap = len(current_words.intersection(previous_words))
            if overlap < 2 and len(current_words) > 5 and len(previous_words) > 5:
                topic_shifts += 1

        if topic_shifts > len(messages) * 0.5:
            warnings.append("Frequent topic shifts may affect coherence")

        return {
            "score": coherence_score,
            "issues": issues,
            "warnings": warnings,
            "details": {
                "transition_usage": transition_score,
                "pronoun_reference_score": pronoun_score,
                "transition_words_used": transition_count,
                "unclear_pronoun_references": unclear_references,
                "topic_shifts": topic_shifts,
                "messages_analyzed": len(messages) - 1
            }
        }


# Backward compatibility function
def assess_language_quality(conversation: Conversation, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Backward compatibility function for language quality assessment.

    Args:
        conversation: The conversation to assess
        config: Optional configuration dictionary

    Returns:
        Dictionary with 'score', 'issues', 'warnings', 'complexity_level', and 'details' keys
    """
    assessor = LanguageQualityAssessor(config)
    metrics = assessor.assess_language_quality(conversation)

    return {
        "score": metrics.overall_score,
        "issues": metrics.issues,
        "warnings": metrics.warnings,
        "complexity_level": metrics.complexity_level.value,
        "details": metrics.details
    }
