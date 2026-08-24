"""
Crisis Keyword Expansion Module for Mental Health Safety Systems.

This module expands crisis-related keywords and phrases to improve detection
coverage for self-harm, suicide, abuse, and other mental health crisis indicators.
"""

import os
from dataclasses import dataclass, field
from enum import StrEnum

import yaml


class CrisisCategory(StrEnum):
    """Categories of crisis indicators."""

    SELF_HARM = "self_harm"
    SUICIDAL_IDEATION = "suicidal_ideation"
    SUBSTANCE_ABUSE = "substance_abuse"
    DOMESTIC_VIOLENCE = "domestic_violence"
    CHILD_ABUSE = "child_abuse"
    EATING_DISORDERS = "eating_disorders"
    PSYCHOTIC_EPISODES = "psychotic_episodes"


class Language(StrEnum):
    """Supported languages."""

    ENGLISH = "en"
    SPANISH = "es"


@dataclass(frozen=True)
class CrisisTerm:
    """Represents a crisis term with metadata."""

    term: str
    category: CrisisCategory
    intensity: float  # 0.0 to 1.0, where 1.0 is highest intensity
    synonyms: set[str] = field(default_factory=set)
    language: Language = Language.ENGLISH

    def __post_init__(self):
        # Ensure intensity is within valid range
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"Intensity must be between 0.0 and 1.0, got {self.intensity}")


@dataclass
class CrisisExpansionConfig:
    """Configuration for CrisisExpansion."""

    # Language preferences
    languages: set[Language] = field(default_factory=lambda: {Language.ENGLISH})

    # Intensity thresholds
    low_intensity_threshold: float = 0.3
    medium_intensity_threshold: float = 0.6
    high_intensity_threshold: float = 0.8

    # Whether to include negation handling
    handle_negation: bool = True

    # Whether to generate phrase variants
    generate_phrase_variants: bool = True

    # Whether to include synonyms
    include_synonyms: bool = True

    # Custom term files to load
    custom_term_files: list[str] = field(default_factory=list)

    # Minimum term length to consider
    min_term_length: int = 3

    # Maximum expansion terms per input term
    max_expansion_terms: int = 20


class CrisisExpansion:
    """
    Expands crisis-related keywords and phrases to improve detection coverage.

    This module provides functionality to:
    - Expand crisis terms with synonyms and phrase variants
    - Score intensity of crisis terms
    - Categorize terms by crisis type
    - Support multiple languages (English + Spanish)
    - Handle negation to avoid false positives
    """

    def __init__(self, config: CrisisExpansionConfig | None = None):
        """
        Initialize the CrisisExpansion module.

        Args:
            config: Optional configuration. If None, uses default configuration.
        """
        self.config = config or CrisisExpansionConfig()
        self._terms: dict[str, CrisisTerm] = {}
        self._term_by_category: dict[CrisisCategory, set[str]] = {category: set() for category in CrisisCategory}
        self._initialize_base_terms()

        # Load custom term files if specified
        for filepath in self.config.custom_term_files:
            self.load_from_file(filepath)

    def _initialize_base_terms(self):
        """Initialize base crisis terms for all supported categories and languages."""
        # English terms
        self._add_english_terms()

        # Spanish terms
        if Language.SPANISH in self.config.languages:
            self._add_spanish_terms()

    def _add_english_terms(self):
        """Add English crisis terms."""
        # Self-harm terms
        self._add_term(
            term="cut myself",
            category=CrisisCategory.SELF_HARM,
            intensity=0.9,
            synonyms={
                "cutting myself",
                "self-cut",
                "wrist cutting",
                "arm cutting",
                "cutting my wrists",
                "cutting my arms",
            },
        )

        self._add_term(
            term="burn myself",
            category=CrisisCategory.SELF_HARM,
            intensity=0.85,
            synonyms={
                "burning myself",
                "self-burn",
                "burning my skin",
                "chemical burn",
                "self-inflicted burn",
            },
        )

        self._add_term(
            term="hit myself",
            category=CrisisCategory.SELF_HARM,
            intensity=0.8,
            synonyms={
                "hitting myself",
                "self-hitting",
                "punching myself",
                "banging head",
                "head banging",
            },
        )

        # Suicidal ideation terms
        self._add_term(
            term="kill myself",
            category=CrisisCategory.SUICIDAL_IDEATION,
            intensity=1.0,
            synonyms={
                "end it all",
                "not worth living",
                "better off dead",
                "want to die",
                "wish I was dead",
                "take my own life",
                "end my life",
                "commit suicide",
                "suicide",
            },
        )

        self._add_term(
            term="suicidal",
            category=CrisisCategory.SUICIDAL_IDEATION,
            intensity=0.95,
            synonyms={
                "suicide",
                "suicidal thoughts",
                "suicidal ideation",
                "thinking about suicide",
                "planning suicide",
            },
        )

        self._add_term(
            term="overdose",
            category=CrisisCategory.SUICIDAL_IDEATION,
            intensity=0.9,
            synonyms={
                "overdosing",
                "pill overdose",
                "drug overdose",
                "intentional overdose",
                "overdose on pills",
            },
        )

        # Substance abuse terms
        self._add_term(
            term="can't stop drinking",
            category=CrisisCategory.SUBSTANCE_ABUSE,
            intensity=0.8,
            synonyms={
                "alcohol addiction",
                "drinking problem",
                "alcoholic",
                "need a drink",
                "craving alcohol",
                "withdrawal",
            },
        )

        self._add_term(
            term="using drugs",
            category=CrisisCategory.SUBSTANCE_ABUSE,
            intensity=0.85,
            synonyms={
                "drug use",
                "substance abuse",
                "getting high",
                "using substances",
                "drug addiction",
                "relapse",
            },
        )

        # Domestic violence terms
        self._add_term(
            term="hit my partner",
            category=CrisisCategory.DOMESTIC_VIOLENCE,
            intensity=0.9,
            synonyms={
                "abuse my partner",
                "domestic violence",
                "spouse abuse",
                "intimate partner violence",
                "beat my wife",
                "beat my husband",
            },
        )

        self._add_term(
            term="threaten family",
            category=CrisisCategory.DOMESTIC_VIOLENCE,
            intensity=0.85,
            synonyms={
                "threaten spouse",
                "threaten children",
                "family violence",
                "domestic threat",
                "threaten loved ones",
            },
        )

        # Child abuse terms
        self._add_term(
            term="hurt my child",
            category=CrisisCategory.CHILD_ABUSE,
            intensity=0.95,
            synonyms={
                "child abuse",
                "abuse my child",
                "harm my child",
                "neglect my child",
                "endanger my child",
            },
        )

        self._add_term(
            term="yell at child",
            category=CrisisCategory.CHILD_ABUSE,
            intensity=0.7,
            synonyms={
                "verbal abuse child",
                "emotional abuse child",
                "shouting at child",
                "scaring child",
            },
        )

        # Eating disorders terms
        self._add_term(
            term="throwing up after eating",
            category=CrisisCategory.EATING_DISORDERS,
            intensity=0.85,
            synonyms={
                "purging",
                "bulimia",
                "vomiting after meals",
                "self-induced vomiting",
                "laxative abuse",
            },
        )

        self._add_term(
            term="not eating enough",
            category=CrisisCategory.EATING_DISORDERS,
            intensity=0.8,
            synonyms={
                "anorexia",
                "starving myself",
                "food restriction",
                "not eating",
                "eating disorder",
                "weight loss",
            },
        )

        # Psychotic episodes terms
        self._add_term(
            term="hearing voices",
            category=CrisisCategory.PSYCHOTIC_EPISODES,
            intensity=0.9,
            synonyms={
                "auditory hallucinations",
                "voices in my head",
                "hearing things",
                "hearing people talk",
            },
        )

        self._add_term(
            term="seeing things",
            category=CrisisCategory.PSYCHOTIC_EPISODES,
            intensity=0.85,
            synonyms={
                "visual hallucinations",
                "seeing people",
                "seeing things that aren't there",
                "delusions",
            },
        )

    def _add_spanish_terms(self):
        """Add Spanish crisis terms."""
        # Self-harm terms
        self._add_term(
            term="cortarme",
            category=CrisisCategory.SELF_HARM,
            intensity=0.9,
            synonyms={
                "cortándome",
                "autolesión",
                "herirme",
                "lastimarme",
                "hacerme daño",
            },
            language=Language.SPANISH,
        )

        self._add_term(
            term="quitarme la vida",
            category=CrisisCategory.SUICIDAL_IDEATION,
            intensity=1.0,
            synonyms={
                "suicidarme",
                "morir",
                "no quiero vivir",
                "mejor muerto",
                "terminar con todo",
            },
            language=Language.SPANISH,
        )

        # Substance abuse terms
        self._add_term(
            term="no puedo dejar de beber",
            category=CrisisCategory.SUBSTANCE_ABUSE,
            intensity=0.8,
            synonyms={
                "adicción al alcohol",
                "problema con el alcohol",
                "necesito beber",
                "craving alcohol",
            },
            language=Language.SPANISH,
        )

        # Domestic violence terms
        self._add_term(
            term="golpear a mi pareja",
            category=CrisisCategory.DOMESTIC_VIOLENCE,
            intensity=0.9,
            synonyms={
                "violencia doméstica",
                "abuso de pareja",
                "maltrato familiar",
                "agredir a mi pareja",
            },
            language=Language.SPANISH,
        )

    def _add_term(
        self,
        term: str,
        category: CrisisCategory,
        intensity: float,
        synonyms: set[str] | None = None,
        language: Language = Language.ENGLISH,
    ):
        """
        Add a term to the crisis terms database.

        Args:
            term: The base term
            category: Crisis category
            intensity: Intensity score (0.0-1.0)
            synonyms: Optional set of synonyms
            language: Language of the term
        """
        if synonyms is None:
            synonyms = set()

        crisis_term = CrisisTerm(
            term=term.lower(),
            category=category,
            intensity=intensity,
            synonyms={s.lower() for s in synonyms},
            language=language,
        )

        self._terms[term.lower()] = crisis_term
        self._term_by_category[category].add(term.lower())

        # Also add synonyms to the index
        for synonym in synonyms:
            self._terms[synonym.lower()] = crisis_term
            self._term_by_category[category].add(synonym.lower())

    def expand_term(self, term: str, category: str) -> list[str]:
        """
        Expand a crisis term with synonyms and related phrases.

        Args:
            term: The term to expand
            category: The crisis category

        Returns:
            List of expanded terms including synonyms and variants
        """
        term_lower = term.lower()

        # Check if term exists in our database
        if term_lower not in self._terms:
            return [term]  # Return original term if not found

        crisis_term = self._terms[term_lower]

        # Verify category matches (if specified)
        if category and crisis_term.category.value != category.lower():
            # Term exists but category doesn't match - still return expansions
            pass

        expansions = set()

        # Add original term
        expansions.add(term_lower)

        # Add synonyms if enabled
        if self.config.include_synonyms:
            expansions.update(crisis_term.synonyms)

        # Generate phrase variants if enabled
        if self.config.generate_phrase_variants:
            variants = self._generate_phrase_variants(term_lower, crisis_term.category)
            expansions.update(variants)

        # Convert to list, filter out any negated forms, and limit size.
        # Negated phrases must never appear in crisis expansions — they
        # represent safe language and would cause false positives.
        result = list(expansions)
        result = [
            expansion
            for expansion in result
            if not (
                expansion.startswith(("not ", "don't ", "do not ", "never ", "would never ", "could never ", "should never "))
                or " not " in expansion
            )
        ]
        result = [term_lower, *sorted(e for e in result if e != term_lower)]
        if len(result) > self.config.max_expansion_terms:
            result = result[: self.config.max_expansion_terms]

        return result

    def _generate_phrase_variants(self, term: str, _category: CrisisCategory) -> set[str]:
        """
        Generate phrase variants for a term based on common expressions.

        Args:
            term: The base term
            category: The crisis category

        Returns:
            Set of phrase variants
        """
        variants = set()

        # Common intensity modifiers
        intensity_modifiers = [
            "sometimes I think about",
            "I've been thinking about",
            "I keep thinking about",
            "I can't stop thinking about",
            "I'm thinking about",
            "I want to",
            "I need to",
            "I'm going to",
            "I plan to",
            "I intend to",
            "I've been",
            "lately I've been",
            "recently I've",
        ]

        # Context modifiers that reduce immediacy
        context_modifiers = [
            "sometimes I",
            "I've been",
            "lately I",
            "recently I",
            "I used to",
            "I sometimes",
            "occasionally I",
        ]

        # Add intensity modifiers
        for modifier in intensity_modifiers:
            variants.add(f"{modifier} {term}")

        # Add context modifiers
        for modifier in context_modifiers:
            variants.add(f"{modifier} {term}")

        # NOTE: Negation forms (e.g. "not kill myself", "never cut myself") are
        # deliberately NOT generated as crisis expansions.  They represent safe
        # language, not crisis indicators, and including them causes false
        # positives when scanning text.  Use the is_negated() method to detect
        # negated crisis language in real text instead.

        # Add past tense forms
        if term.endswith("ing"):
            base = term[:-3]
            variants.add(f"{base}ed")
            variants.add(f"have {base}ed")
            variants.add(f"has {base}ed")
        else:
            variants.add(f"{term}ed")
            variants.add(f"have {term}ed")
            variants.add(f"has {term}ed")

        return variants

    def get_all_crisis_terms(self) -> set[str]:
        """
        Get all crisis terms in the database.

        Returns:
            Set of all crisis terms
        """
        return set(self._terms.keys())

    def categorize_term(self, term: str) -> str | None:
        """
        Categorize a term by crisis type.

        Args:
            term: The term to categorize

        Returns:
            Crisis category string if found, None otherwise
        """
        term_lower = term.lower()
        if term_lower in self._terms:
            return self._terms[term_lower].category.value
        return None

    def get_intensity(self, term: str) -> float:
        """
        Get the intensity score for a term.

        Args:
            term: The term to score

        Returns:
            Intensity score (0.0-1.0), 0.0 if term not found
        """
        term_lower = term.lower()
        if term_lower in self._terms:
            return self._terms[term_lower].intensity
        return 0.0

    def load_from_file(self, filepath: str) -> None:
        """
        Load crisis terms from a YAML file.

        Expected format:
        ```yaml
        - term: "example term"
          category: "self_harm"
          intensity: 0.8
          synonyms:
            - "synonym 1"
            - "synonym 2"
          language: "en"
        ```

        Args:
            filepath: Path to the YAML file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, list):
            raise ValueError("YAML file must contain a list of term definitions")

        for item in data:
            if not isinstance(item, dict):
                continue

            term = item.get("term")
            category_str = item.get("category")
            intensity = item.get("intensity")
            synonyms = item.get("synonyms", [])
            language_str = item.get("language", "en")

            if not all([term, category_str, intensity is not None]):
                continue

            try:
                category = CrisisCategory(category_str)
                language = Language(language_str)

                self._add_term(
                    term=term,
                    category=category,
                    intensity=float(intensity),
                    synonyms=set(synonyms) if synonyms else None,
                    language=language,
                )
            except (ValueError, KeyError):
                # Skip invalid entries
                continue

    def get_terms_by_category(self, category: CrisisCategory) -> set[str]:
        """
        Get all terms for a specific crisis category.

        Args:
            category: The crisis category

        Returns:
            Set of terms in the category
        """
        return self._term_by_category.get(category, set()).copy()

    def is_crisis_term(self, term: str) -> bool:
        """
        Check if a term is a crisis term.

        Args:
            term: The term to check

        Returns:
            True if term is a crisis term, False otherwise
        """
        return term.lower() in self._terms

    def is_negated(self, text: str, term: str) -> bool:
        """
        Check if a crisis term appears in a negated context in the given text.

        This should be called when a crisis term match is found to verify it
        is not actually a negation (e.g. "I would never kill myself" should
        NOT be flagged as a crisis).

        Args:
            text: The full text to check
            term: The crisis term that was matched

        Returns:
            True if the term is negated in the text, False otherwise
        """
        text_lower = text.lower()
        term_lower = term.lower()

        negation_prefixes = [
            "not ", "don't ", "do not ", "never ",
            "would never ", "could never ", "should never ",
            "won't ", "doesn't ", "isn't ", "aren't ",
        ]

        for prefix in negation_prefixes:
            if f"{prefix}{term_lower}" in text_lower:
                return True

        # Check for negation within a few tokens before the term
        tokens = text_lower.split()
        term_tokens = term_lower.split()
        if not term_tokens:
            return False

        # Check negation before both the first and last token of the term
        for check_token in (term_tokens[0], term_tokens[-1]):
            for i, token in enumerate(tokens):
                if token == check_token and i > 0:
                    prev_tokens = tokens[max(0, i - 4):i]
                    negation_words = {
                        "not", "don't", "never", "no", "without",
                        "doesn't", "isn't", "aren't", "won't", "can't",
                    }
                    if any(neg in prev_tokens for neg in negation_words):
                        return True

        return False

    def get_term_info(self, term: str) -> CrisisTerm | None:
        """
        Get detailed information about a term.

        Args:
            term: The term to get information for

        Returns:
            CrisisTerm object if found, None otherwise
        """
        return self._terms.get(term.lower())


# Convenience function for easy access
def create_crisis_expansion(
    languages: set[Language] | None = None,
    custom_term_files: list[str] | None = None,
) -> CrisisExpansion:
    """
    Create a CrisisExpansion instance with common configurations.

    Args:
        languages: Optional set of languages to support
        custom_term_files: Optional list of custom term files to load

    Returns:
        Configured CrisisExpansion instance
    """
    config = CrisisExpansionConfig(
        languages=languages or {Language.ENGLISH, Language.SPANISH},
        custom_term_files=custom_term_files or [],
    )
    return CrisisExpansion(config)


if __name__ == "__main__":
    # Example usage and basic tests
    expansion = CrisisExpansion()

    # Test expand_term
    expansions = expansion.expand_term("kill myself", "suicidal_ideation")

    # Test categorize_term
    category = expansion.categorize_term("cut myself")

    # Test get_intensity
    intensity = expansion.get_intensity("suicidal")

    # Test get_all_crisis_terms
    all_terms = expansion.get_all_crisis_terms()

    # Test is_crisis_term

    # Test negation handling
    neg_expansions = expansion.expand_term("kill myself", "suicidal_ideation")
    neg_terms = [t for t in neg_expansions if "not" in t or "don't" in t]
