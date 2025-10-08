"""
Advanced personality marker extraction system for voice-derived data.
Provides comprehensive personality analysis using multiple psychological frameworks.
"""

import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from logger import get_logger


class PersonalityFramework(Enum):
    """Supported personality analysis frameworks."""

    BIG_FIVE = "big_five"
    MBTI = "mbti"
    DISC = "disc"
    ENNEAGRAM = "enneagram"
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"


@dataclass
class PersonalityMarker:
    """Individual personality marker with context."""

    text: str
    category: str
    framework: PersonalityFramework
    confidence: float
    context: str = ""
    position: int = 0
    weight: float = 1.0


@dataclass
class PersonalityAnalysis:
    """Comprehensive personality analysis result."""

    big_five_scores: dict[str, float] = field(default_factory=dict)
    mbti_indicators: dict[str, float] = field(default_factory=dict)
    disc_profile: dict[str, float] = field(default_factory=dict)
    enneagram_type: int | None = None
    emotional_intelligence: dict[str, float] = field(default_factory=dict)
    communication_patterns: dict[str, float] = field(default_factory=dict)
    empathy_markers: list[str] = field(default_factory=list)
    authenticity_indicators: list[str] = field(default_factory=list)
    confidence: float = 0.0
    markers_found: list[PersonalityMarker] = field(default_factory=list)


class PersonalityExtractor:
    """
    Advanced personality marker extraction system.

    Features:
    - Multi-framework personality analysis (Big Five, MBTI, DISC, Enneagram)
    - Context-aware marker detection
    - Weighted scoring based on marker reliability
    - Communication pattern analysis
    - Empathy and authenticity detection
    - Confidence scoring and validation
    """

    def __init__(self):
        """Initialize PersonalityExtractor."""
        self.logger = get_logger(__name__)

        # Initialize marker databases
        self.big_five_markers = self._initialize_big_five_markers()
        self.mbti_markers = self._initialize_mbti_markers()
        self.disc_markers = self._initialize_disc_markers()
        self.enneagram_markers = self._initialize_enneagram_markers()
        self.emotional_intelligence_markers = self._initialize_ei_markers()
        self.empathy_markers = self._initialize_empathy_markers()
        self.authenticity_markers = self._initialize_authenticity_markers()

        # Communication pattern analyzers
        self.communication_analyzers = self._initialize_communication_analyzers()

        self.logger.info(
            "PersonalityExtractor initialized with comprehensive marker databases"
        )

    def extract_personality(
        self,
        text: str,
        frameworks: list[PersonalityFramework] | None = None,
        include_markers: bool = True,
    ) -> PersonalityAnalysis:
        """
        Extract comprehensive personality analysis from text.

        Args:
            text: Input text to analyze
            frameworks: Optional list of frameworks to use (default: all)
            include_markers: Whether to include detailed markers in result

        Returns:
            PersonalityAnalysis with comprehensive results
        """
        if frameworks is None:
            frameworks = list(PersonalityFramework)

        analysis = PersonalityAnalysis()
        all_markers = []

        # Extract markers for each framework
        if PersonalityFramework.BIG_FIVE in frameworks:
            big_five_markers = self._extract_big_five_markers(text)
            analysis.big_five_scores = self._calculate_big_five_scores(big_five_markers)
            all_markers.extend(big_five_markers)

        if PersonalityFramework.MBTI in frameworks:
            mbti_markers = self._extract_mbti_markers(text)
            analysis.mbti_indicators = self._calculate_mbti_indicators(mbti_markers)
            all_markers.extend(mbti_markers)

        if PersonalityFramework.DISC in frameworks:
            disc_markers = self._extract_disc_markers(text)
            analysis.disc_profile = self._calculate_disc_profile(disc_markers)
            all_markers.extend(disc_markers)

        if PersonalityFramework.ENNEAGRAM in frameworks:
            enneagram_markers = self._extract_enneagram_markers(text)
            analysis.enneagram_type = self._determine_enneagram_type(enneagram_markers)
            all_markers.extend(enneagram_markers)

        if PersonalityFramework.EMOTIONAL_INTELLIGENCE in frameworks:
            ei_markers = self._extract_ei_markers(text)
            analysis.emotional_intelligence = self._calculate_ei_scores(ei_markers)
            all_markers.extend(ei_markers)

        # Extract communication patterns
        analysis.communication_patterns = self._analyze_communication_patterns(text)

        # Extract empathy and authenticity markers
        analysis.empathy_markers = self._extract_empathy_phrases(text)
        analysis.authenticity_indicators = self._extract_authenticity_indicators(text)

        # Calculate overall confidence
        analysis.confidence = self._calculate_overall_confidence(all_markers, text)

        # Include detailed markers if requested
        if include_markers:
            analysis.markers_found = all_markers

        self.logger.debug(
            f"Extracted personality analysis with {len(all_markers)} markers "
            f"(confidence: {analysis.confidence:.3f})"
        )

        return analysis

    def extract_empathy_profile(self, text: str) -> dict[str, Any]:
        """Extract detailed empathy profile from text."""
        empathy_phrases = self._extract_empathy_phrases(text)
        empathy_categories = self._categorize_empathy_markers(empathy_phrases, text)
        empathy_score = self._calculate_empathy_score(empathy_categories)

        return {
            "empathy_score": empathy_score,
            "empathy_phrases": empathy_phrases,
            "categories": empathy_categories,
            "empathy_consistency": self._calculate_empathy_consistency(text),
            "emotional_responsiveness": self._analyze_emotional_responsiveness(text),
        }

    def extract_authenticity_profile(self, text: str) -> dict[str, Any]:
        """Extract detailed authenticity profile from text."""
        authenticity_indicators = self._extract_authenticity_indicators(text)
        naturalness_score = self._calculate_naturalness_score(text)
        personal_disclosure = self._analyze_personal_disclosure(text)

        return {
            "authenticity_score": self._calculate_authenticity_score(text),
            "authenticity_indicators": authenticity_indicators,
            "naturalness_score": naturalness_score,
            "personal_disclosure": personal_disclosure,
            "conversational_flow": self._analyze_conversational_flow(text),
            "emotional_authenticity": self._analyze_emotional_authenticity(text),
        }

    # Private methods - Framework-specific marker extraction

    def _initialize_big_five_markers(self) -> dict[str, list[tuple[str, float]]]:
        """Initialize Big Five personality markers."""
        return {
            "openness": [
                ("creative", 0.8),
                ("imaginative", 0.9),
                ("curious", 0.7),
                ("artistic", 0.8),
                ("innovative", 0.8),
                ("explore", 0.6),
                ("new ideas", 0.7),
                ("abstract", 0.8),
                ("philosophical", 0.9),
            ],
            "conscientiousness": [
                ("organized", 0.8),
                ("responsible", 0.9),
                ("reliable", 0.8),
                ("disciplined", 0.9),
                ("systematic", 0.8),
                ("thorough", 0.7),
                ("careful", 0.6),
                ("punctual", 0.8),
                ("methodical", 0.8),
            ],
            "extraversion": [
                ("outgoing", 0.8),
                ("social", 0.7),
                ("energetic", 0.7),
                ("talkative", 0.8),
                ("assertive", 0.7),
                ("enthusiastic", 0.8),
                ("people person", 0.9),
                ("party", 0.6),
                ("leadership", 0.7),
            ],
            "agreeableness": [
                ("kind", 0.8),
                ("helpful", 0.8),
                ("cooperative", 0.8),
                ("trusting", 0.7),
                ("compassionate", 0.9),
                ("understanding", 0.8),
                ("supportive", 0.8),
                ("caring", 0.9),
                ("empathetic", 0.9),
            ],
            "neuroticism": [
                ("anxious", 0.8),
                ("worried", 0.7),
                ("stressed", 0.7),
                ("emotional", 0.6),
                ("sensitive", 0.5),
                ("nervous", 0.8),
                ("overwhelmed", 0.8),
                ("insecure", 0.7),
                ("moody", 0.7),
            ],
        }

    def _initialize_mbti_markers(self) -> dict[str, list[tuple[str, float]]]:
        """Initialize MBTI personality markers."""
        return {
            "extraversion": [
                ("love meeting people", 0.8),
                ("energized by others", 0.9),
                ("think out loud", 0.7),
                ("broad interests", 0.6),
            ],
            "introversion": [
                ("need quiet time", 0.8),
                ("prefer small groups", 0.7),
                ("think before speaking", 0.8),
                ("deep focus", 0.7),
            ],
            "sensing": [
                ("practical", 0.8),
                ("concrete", 0.7),
                ("step by step", 0.8),
                ("details matter", 0.8),
                ("experience", 0.6),
            ],
            "intuition": [
                ("possibilities", 0.8),
                ("big picture", 0.8),
                ("patterns", 0.7),
                ("future focused", 0.7),
                ("theoretical", 0.8),
            ],
            "thinking": [
                ("logical", 0.8),
                ("objective", 0.8),
                ("analyze", 0.8),
                ("rational", 0.8),
                ("critique", 0.7),
            ],
            "feeling": [
                ("values", 0.8),
                ("harmony", 0.8),
                ("personal", 0.7),
                ("empathy", 0.9),
                ("consider feelings", 0.8),
            ],
            "judging": [
                ("planned", 0.8),
                ("organized", 0.8),
                ("decided", 0.7),
                ("structured", 0.8),
                ("closure", 0.7),
            ],
            "perceiving": [
                ("flexible", 0.8),
                ("spontaneous", 0.8),
                ("adaptable", 0.8),
                ("open ended", 0.7),
                ("go with flow", 0.8),
            ],
        }

    def _initialize_disc_markers(self) -> dict[str, list[tuple[str, float]]]:
        """Initialize DISC personality markers."""
        return {
            "dominance": [
                ("results oriented", 0.8),
                ("direct", 0.8),
                ("decisive", 0.8),
                ("competitive", 0.7),
                ("challenge", 0.7),
                ("control", 0.6),
            ],
            "influence": [
                ("enthusiastic", 0.8),
                ("optimistic", 0.8),
                ("persuasive", 0.8),
                ("people oriented", 0.8),
                ("inspiring", 0.7),
                ("expressive", 0.7),
            ],
            "steadiness": [
                ("patient", 0.8),
                ("reliable", 0.8),
                ("supportive", 0.8),
                ("stable", 0.7),
                ("consistent", 0.8),
                ("loyal", 0.8),
            ],
            "conscientiousness": [
                ("accurate", 0.8),
                ("analytical", 0.8),
                ("systematic", 0.8),
                ("quality", 0.7),
                ("precise", 0.8),
                ("cautious", 0.7),
            ],
        }

    def _initialize_enneagram_markers(self) -> dict[int, list[tuple[str, float]]]:
        """Initialize Enneagram personality markers."""
        return {
            1: [("perfectionist", 0.9), ("principled", 0.8), ("right way", 0.8)],
            2: [("helper", 0.9), ("caring", 0.8), ("others first", 0.8)],
            3: [("achiever", 0.9), ("success", 0.8), ("image", 0.7)],
            4: [("individualist", 0.9), ("unique", 0.8), ("authentic", 0.8)],
            5: [("investigator", 0.9), ("knowledge", 0.8), ("private", 0.7)],
            6: [("loyalist", 0.9), ("security", 0.8), ("guidance", 0.7)],
            7: [("enthusiast", 0.9), ("possibilities", 0.8), ("adventure", 0.8)],
            8: [("challenger", 0.9), ("control", 0.8), ("justice", 0.8)],
            9: [("peacemaker", 0.9), ("harmony", 0.8), ("avoid conflict", 0.8)],
        }

    def _initialize_ei_markers(self) -> dict[str, list[tuple[str, float]]]:
        """Initialize Emotional Intelligence markers."""
        return {
            "self_awareness": [
                ("know myself", 0.8),
                ("my feelings", 0.8),
                ("self reflection", 0.9),
                ("aware of", 0.6),
                ("recognize", 0.7),
            ],
            "self_regulation": [
                ("manage emotions", 0.9),
                ("stay calm", 0.8),
                ("control", 0.7),
                ("adapt", 0.7),
                ("flexible", 0.6),
            ],
            "motivation": [
                ("driven", 0.8),
                ("passionate", 0.8),
                ("goals", 0.7),
                ("committed", 0.8),
                ("persevere", 0.8),
            ],
            "empathy": [
                ("understand others", 0.9),
                ("feel for", 0.8),
                ("perspective", 0.8),
                ("compassionate", 0.9),
                ("sensitive to", 0.7),
            ],
            "social_skills": [
                ("communicate well", 0.8),
                ("build relationships", 0.8),
                ("team player", 0.7),
                ("influence", 0.7),
                ("collaborate", 0.8),
            ],
        }

    def _initialize_empathy_markers(self) -> list[tuple[str, float]]:
        """Initialize empathy markers."""
        return [
            ("I understand", 0.9),
            ("I can imagine", 0.8),
            ("that must be", 0.8),
            ("I feel for you", 0.9),
            ("sounds difficult", 0.7),
            ("I hear you", 0.8),
            ("that's really hard", 0.8),
            ("you're not alone", 0.9),
            ("I'm here for you", 0.9),
            ("that sounds tough", 0.7),
            ("I can see why", 0.7),
            ("makes sense", 0.6),
            ("validate", 0.8),
        ]

    def _initialize_authenticity_markers(self) -> list[tuple[str, float]]:
        """Initialize authenticity markers."""
        return [
            ("honestly", 0.7),
            ("to be honest", 0.8),
            ("personally", 0.7),
            ("in my experience", 0.8),
            ("I feel", 0.6),
            ("I think", 0.5),
            ("from my heart", 0.9),
            ("genuinely", 0.8),
            ("truly", 0.6),
            ("actually", 0.5),
            ("really", 0.4),
            ("you know", 0.6),
        ]

    def _initialize_communication_analyzers(self) -> dict[str, Callable]:
        """Initialize communication pattern analyzers."""
        return {
            "formality": self._analyze_formality,
            "directness": self._analyze_directness,
            "emotional_expression": self._analyze_emotional_expression,
            "question_asking": self._analyze_question_asking,
            "supportiveness": self._analyze_supportiveness,
        }

    def _extract_big_five_markers(self, text: str) -> list[PersonalityMarker]:
        """Extract Big Five personality markers from text."""
        markers = []
        text_lower = text.lower()

        for trait, trait_markers in self.big_five_markers.items():
            for marker_text, confidence in trait_markers:
                if marker_text in text_lower:
                    position = text_lower.find(marker_text)
                    context = self._extract_context(text, position, marker_text)

                    marker = PersonalityMarker(
                        text=marker_text,
                        category=trait,
                        framework=PersonalityFramework.BIG_FIVE,
                        confidence=confidence,
                        context=context,
                        position=position,
                    )
                    markers.append(marker)

        return markers

    def _extract_mbti_markers(self, text: str) -> list[PersonalityMarker]:
        """Extract MBTI personality markers from text."""
        markers = []
        text_lower = text.lower()

        for dimension, dimension_markers in self.mbti_markers.items():
            for marker_text, confidence in dimension_markers:
                if marker_text in text_lower:
                    position = text_lower.find(marker_text)
                    context = self._extract_context(text, position, marker_text)

                    marker = PersonalityMarker(
                        text=marker_text,
                        category=dimension,
                        framework=PersonalityFramework.MBTI,
                        confidence=confidence,
                        context=context,
                        position=position,
                    )
                    markers.append(marker)

        return markers

    def _extract_disc_markers(self, text: str) -> list[PersonalityMarker]:
        """Extract DISC personality markers from text."""
        markers = []
        text_lower = text.lower()

        for style, style_markers in self.disc_markers.items():
            for marker_text, confidence in style_markers:
                if marker_text in text_lower:
                    position = text_lower.find(marker_text)
                    context = self._extract_context(text, position, marker_text)

                    marker = PersonalityMarker(
                        text=marker_text,
                        category=style,
                        framework=PersonalityFramework.DISC,
                        confidence=confidence,
                        context=context,
                        position=position,
                    )
                    markers.append(marker)

        return markers

    def _extract_enneagram_markers(self, text: str) -> list[PersonalityMarker]:
        """Extract Enneagram personality markers from text."""
        markers = []
        text_lower = text.lower()

        for type_num, type_markers in self.enneagram_markers.items():
            for marker_text, confidence in type_markers:
                if marker_text in text_lower:
                    position = text_lower.find(marker_text)
                    context = self._extract_context(text, position, marker_text)

                    marker = PersonalityMarker(
                        text=marker_text,
                        category=str(type_num),
                        framework=PersonalityFramework.ENNEAGRAM,
                        confidence=confidence,
                        context=context,
                        position=position,
                    )
                    markers.append(marker)

        return markers

    def _extract_ei_markers(self, text: str) -> list[PersonalityMarker]:
        """Extract Emotional Intelligence markers from text."""
        markers = []
        text_lower = text.lower()

        for component, component_markers in self.emotional_intelligence_markers.items():
            for marker_text, confidence in component_markers:
                if marker_text in text_lower:
                    position = text_lower.find(marker_text)
                    context = self._extract_context(text, position, marker_text)

                    marker = PersonalityMarker(
                        text=marker_text,
                        category=component,
                        framework=PersonalityFramework.EMOTIONAL_INTELLIGENCE,
                        confidence=confidence,
                        context=context,
                        position=position,
                    )
                    markers.append(marker)

        return markers

    def _extract_empathy_phrases(self, text: str) -> list[str]:
        """Extract empathy phrases from text."""
        found_phrases = []
        text_lower = text.lower()

        for phrase, _confidence in self.empathy_markers:
            if phrase in text_lower:
                found_phrases.append(phrase)

        return found_phrases

    def _extract_authenticity_indicators(self, text: str) -> list[str]:
        """Extract authenticity indicators from text."""
        found_indicators = []
        text_lower = text.lower()

        for indicator, _confidence in self.authenticity_markers:
            if indicator in text_lower:
                found_indicators.append(indicator)

        # Add additional authenticity checks
        if self._has_personal_pronouns(text):
            found_indicators.append("personal_pronouns")

        if self._has_emotional_expressions(text):
            found_indicators.append("emotional_expressions")

        if self._has_conversational_markers(text):
            found_indicators.append("conversational_markers")

        return found_indicators

    def _extract_context(self, text: str, position: int, marker: str) -> str:
        """Extract context around a marker."""
        context_size = 50
        start = max(0, position - context_size)
        end = min(len(text), position + len(marker) + context_size)
        return text[start:end].strip()

    # Scoring and calculation methods

    def _calculate_big_five_scores(
        self, markers: list[PersonalityMarker]
    ) -> dict[str, float]:
        """Calculate Big Five scores from markers."""
        trait_scores = defaultdict(list)

        for marker in markers:
            if marker.framework == PersonalityFramework.BIG_FIVE:
                weighted_score = marker.confidence * marker.weight
                trait_scores[marker.category].append(weighted_score)

        # Calculate average scores for each trait
        final_scores = {}
        for trait, scores in trait_scores.items():
            final_scores[trait] = min(1.0, statistics.mean(scores)) if scores else 0.0

        return final_scores

    def _calculate_mbti_indicators(
        self, markers: list[PersonalityMarker]
    ) -> dict[str, float]:
        """Calculate MBTI indicators from markers."""
        dimension_scores = defaultdict(list)

        for marker in markers:
            if marker.framework == PersonalityFramework.MBTI:
                weighted_score = marker.confidence * marker.weight
                dimension_scores[marker.category].append(weighted_score)

        # Calculate average scores
        final_scores = {}
        for dimension, scores in dimension_scores.items():
            final_scores[dimension] = (
                min(1.0, statistics.mean(scores)) if scores else 0.0
            )

        return final_scores

    def _calculate_disc_profile(
        self, markers: list[PersonalityMarker]
    ) -> dict[str, float]:
        """Calculate DISC profile from markers."""
        style_scores = defaultdict(list)

        for marker in markers:
            if marker.framework == PersonalityFramework.DISC:
                weighted_score = marker.confidence * marker.weight
                style_scores[marker.category].append(weighted_score)

        # Calculate average scores
        final_scores = {}
        for style, scores in style_scores.items():
            final_scores[style] = min(1.0, statistics.mean(scores)) if scores else 0.0

        return final_scores

    def _determine_enneagram_type(
        self, markers: list[PersonalityMarker]
    ) -> int | None:
        """Determine most likely Enneagram type from markers."""
        type_scores = defaultdict(float)

        for marker in markers:
            if marker.framework == PersonalityFramework.ENNEAGRAM:
                type_num = int(marker.category)
                type_scores[type_num] += marker.confidence * marker.weight

        if type_scores:
            return max(type_scores.keys(), key=lambda k: type_scores[k])

        return None

    def _calculate_ei_scores(
        self, markers: list[PersonalityMarker]
    ) -> dict[str, float]:
        """Calculate Emotional Intelligence scores from markers."""
        component_scores = defaultdict(list)

        for marker in markers:
            if marker.framework == PersonalityFramework.EMOTIONAL_INTELLIGENCE:
                weighted_score = marker.confidence * marker.weight
                component_scores[marker.category].append(weighted_score)

        # Calculate average scores
        final_scores = {}
        for component, scores in component_scores.items():
            final_scores[component] = (
                min(1.0, statistics.mean(scores)) if scores else 0.0
            )

        return final_scores

    def _analyze_communication_patterns(self, text: str) -> dict[str, float]:
        """Analyze communication patterns in text."""
        patterns = {}

        for pattern_name, analyzer in self.communication_analyzers.items():
            try:
                patterns[pattern_name] = analyzer(text)
            except Exception as e:
                self.logger.warning(
                    f"Communication analyzer {pattern_name} failed: {e}"
                )
                patterns[pattern_name] = 0.0

        return patterns

    def _calculate_overall_confidence(
        self, markers: list[PersonalityMarker], text: str
    ) -> float:
        """Calculate overall confidence in personality analysis."""
        if not markers:
            return 0.0

        # Base confidence from marker quality
        marker_confidences = [m.confidence for m in markers]
        base_confidence = statistics.mean(marker_confidences)

        # Adjust for text length (more text = higher confidence)
        text_length_factor = min(1.0, len(text) / 1000)  # Normalize to 1000 chars

        # Adjust for marker diversity (more diverse markers = higher confidence)
        unique_categories = len({m.category for m in markers})
        diversity_factor = min(
            1.0, unique_categories / 10
        )  # Normalize to 10 categories

        # Combined confidence
        overall_confidence = (
            base_confidence * 0.6 + text_length_factor * 0.2 + diversity_factor * 0.2
        )

        return min(1.0, overall_confidence)

    # Communication pattern analyzers

    def _analyze_formality(self, text: str) -> float:
        """Analyze formality level in text."""
        formal_indicators = ["please", "thank you", "would you", "could you", "may I"]
        informal_indicators = ["yeah", "ok", "cool", "awesome", "hey"]

        formal_count = sum(
            1 for indicator in formal_indicators if indicator in text.lower()
        )
        informal_count = sum(
            1 for indicator in informal_indicators if indicator in text.lower()
        )

        total_indicators = formal_count + informal_count
        if total_indicators == 0:
            return 0.5  # Neutral

        return formal_count / total_indicators

    def _analyze_directness(self, text: str) -> float:
        """Analyze directness in communication."""
        direct_indicators = [
            "I think",
            "I believe",
            "clearly",
            "obviously",
            "definitely",
        ]
        indirect_indicators = ["maybe", "perhaps", "might", "could be", "I guess"]

        direct_count = sum(
            1 for indicator in direct_indicators if indicator in text.lower()
        )
        indirect_count = sum(
            1 for indicator in indirect_indicators if indicator in text.lower()
        )

        total_indicators = direct_count + indirect_count
        if total_indicators == 0:
            return 0.5  # Neutral

        return direct_count / total_indicators

    def _analyze_emotional_expression(self, text: str) -> float:
        """Analyze level of emotional expression."""
        emotional_words = [
            "feel",
            "emotion",
            "heart",
            "love",
            "hate",
            "excited",
            "sad",
            "happy",
        ]

        emotional_count = sum(1 for word in emotional_words if word in text.lower())
        word_count = len(text.split())

        if word_count == 0:
            return 0.0

        return min(1.0, emotional_count / word_count * 10)  # Scale up for visibility

    def _analyze_question_asking(self, text: str) -> float:
        """Analyze frequency of question asking."""
        question_count = text.count("?")
        sentence_count = max(1, text.count(".") + text.count("!") + question_count)

        return question_count / sentence_count

    def _analyze_supportiveness(self, text: str) -> float:
        """Analyze supportiveness in communication."""
        supportive_phrases = [
            "you can do it",
            "I believe in you",
            "support",
            "help",
            "here for you",
        ]

        supportive_count = sum(
            1 for phrase in supportive_phrases if phrase in text.lower()
        )

        return min(1.0, supportive_count / 5)  # Normalize to max 5 phrases

    # Authenticity analysis helpers

    def _has_personal_pronouns(self, text: str) -> bool:
        """Check if text has personal pronouns."""
        personal_pronouns = ["I", "me", "my", "myself", "we", "us", "our"]
        return any(pronoun in text for pronoun in personal_pronouns)

    def _has_emotional_expressions(self, text: str) -> bool:
        """Check if text has emotional expressions."""
        emotional_expressions = ["feel", "feeling", "emotion", "heart", "soul"]
        return any(expr in text.lower() for expr in emotional_expressions)

    def _has_conversational_markers(self, text: str) -> bool:
        """Check if text has conversational markers."""
        conversational_markers = ["you know", "I mean", "like", "well", "so"]
        return any(marker in text.lower() for marker in conversational_markers)

    # Additional analysis methods

    def _categorize_empathy_markers(
        self, phrases: list[str], text: str
    ) -> dict[str, list[str]]:
        """Categorize empathy markers by type."""
        categories = {
            "cognitive_empathy": [],
            "affective_empathy": [],
            "compassionate_empathy": [],
        }

        cognitive_indicators = ["I understand", "I can see", "makes sense"]
        affective_indicators = ["I feel for you", "that must hurt", "I can imagine"]
        compassionate_indicators = [
            "I'm here for you",
            "let me help",
            "you're not alone",
        ]

        for phrase in phrases:
            if any(indicator in phrase for indicator in cognitive_indicators):
                categories["cognitive_empathy"].append(phrase)
            elif any(indicator in phrase for indicator in affective_indicators):
                categories["affective_empathy"].append(phrase)
            elif any(indicator in phrase for indicator in compassionate_indicators):
                categories["compassionate_empathy"].append(phrase)

        return categories

    def _calculate_empathy_score(self, categories: dict[str, list[str]]) -> float:
        """Calculate overall empathy score from categorized markers."""
        total_markers = sum(len(markers) for markers in categories.values())

        if total_markers == 0:
            return 0.0

        # Weight different types of empathy
        cognitive_weight = 0.3
        affective_weight = 0.4
        compassionate_weight = 0.3

        score = (
            len(categories["cognitive_empathy"]) * cognitive_weight
            + len(categories["affective_empathy"]) * affective_weight
            + len(categories["compassionate_empathy"]) * compassionate_weight
        )

        return min(1.0, score / 3)  # Normalize to max 3 markers per category

    def _calculate_empathy_consistency(self, text: str) -> float:
        """Calculate consistency of empathy throughout text."""
        sentences = text.split(".")
        empathy_per_sentence = []

        for sentence in sentences:
            empathy_phrases = self._extract_empathy_phrases(sentence)
            empathy_per_sentence.append(len(empathy_phrases))

        if len(empathy_per_sentence) <= 1:
            return 1.0

        # Calculate coefficient of variation (lower = more consistent)
        mean_empathy = statistics.mean(empathy_per_sentence)
        if mean_empathy == 0:
            return 1.0

        std_empathy = statistics.stdev(empathy_per_sentence)
        cv = std_empathy / mean_empathy

        # Convert to consistency score (higher = more consistent)
        return max(0.0, 1.0 - cv)

    def _analyze_emotional_responsiveness(self, text: str) -> float:
        """Analyze emotional responsiveness in text."""
        emotional_response_indicators = [
            "that sounds",
            "I can hear",
            "seems like",
            "appears that",
            "I notice",
            "I sense",
            "picking up on",
        ]

        response_count = sum(
            1
            for indicator in emotional_response_indicators
            if indicator in text.lower()
        )

        return min(1.0, response_count / 3)  # Normalize to max 3 indicators

    def _calculate_naturalness_score(self, text: str) -> float:
        """Calculate naturalness score based on conversational markers."""
        natural_markers = [
            "um",
            "uh",
            "you know",
            "I mean",
            "like",
            "well",
            "so",
            "actually",
            "really",
            "kind of",
            "sort of",
        ]

        marker_count = sum(1 for marker in natural_markers if marker in text.lower())
        word_count = len(text.split())

        if word_count == 0:
            return 0.0

        # Optimal naturalness is around 2-5% of words being natural markers
        naturalness_ratio = marker_count / word_count

        if 0.02 <= naturalness_ratio <= 0.05:
            return 1.0
        if naturalness_ratio < 0.02:
            return naturalness_ratio / 0.02
        return max(0.0, 1.0 - (naturalness_ratio - 0.05) / 0.05)

    def _analyze_personal_disclosure(self, text: str) -> dict[str, float]:
        """Analyze level of personal disclosure in text."""
        disclosure_indicators = {
            "experiences": ["I experienced", "I went through", "happened to me"],
            "feelings": ["I feel", "I felt", "makes me feel"],
            "opinions": ["I think", "I believe", "in my opinion"],
            "values": ["I value", "important to me", "I care about"],
        }

        disclosure_scores = {}

        for category, indicators in disclosure_indicators.items():
            count = sum(1 for indicator in indicators if indicator in text.lower())
            disclosure_scores[category] = min(
                1.0, count / 2
            )  # Normalize to max 2 per category

        return disclosure_scores

    def _analyze_conversational_flow(self, text: str) -> float:
        """Analyze naturalness of conversational flow."""
        flow_indicators = [
            "and then",
            "so",
            "but",
            "however",
            "also",
            "besides",
            "meanwhile",
            "afterwards",
            "by the way",
            "speaking of",
        ]

        flow_count = sum(
            1 for indicator in flow_indicators if indicator in text.lower()
        )
        sentence_count = max(1, text.count(".") + text.count("!") + text.count("?"))

        # Good flow has about 1 connector per 2-3 sentences
        optimal_ratio = 0.4
        actual_ratio = flow_count / sentence_count

        if actual_ratio <= optimal_ratio:
            return actual_ratio / optimal_ratio
        return max(0.0, 1.0 - (actual_ratio - optimal_ratio) / optimal_ratio)

    def _analyze_emotional_authenticity(self, text: str) -> float:
        """Analyze authenticity of emotional expressions."""
        authentic_emotional_expressions = [
            "from my heart",
            "deeply feel",
            "truly believe",
            "genuinely",
            "honestly feel",
            "really means",
            "touches me",
        ]

        authentic_count = sum(
            1 for expr in authentic_emotional_expressions if expr in text.lower()
        )

        return min(1.0, authentic_count / 3)  # Normalize to max 3 expressions

    def _calculate_authenticity_score(self, text: str) -> float:
        """Calculate overall authenticity score."""
        authenticity_indicators = self._extract_authenticity_indicators(text)
        naturalness_score = self._calculate_naturalness_score(text)
        personal_disclosure = self._analyze_personal_disclosure(text)
        conversational_flow = self._analyze_conversational_flow(text)
        emotional_authenticity = self._analyze_emotional_authenticity(text)

        # Weight different components
        indicator_score = min(1.0, len(authenticity_indicators) / 5)
        disclosure_score = (
            statistics.mean(personal_disclosure.values())
            if personal_disclosure
            else 0.0
        )

        return (
            indicator_score * 0.25
            + naturalness_score * 0.25
            + disclosure_score * 0.2
            + conversational_flow * 0.15
            + emotional_authenticity * 0.15
        )

