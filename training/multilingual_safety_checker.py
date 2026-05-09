"""Multilingual content analyzer extending ClinicalContentAnalyzer.

Adds crisis pattern definitions for Spanish, French, Portuguese, and German
with per-language negation patterns. These patterns are used for content
analysis and measurement — they do not filter or remove any content.
"""

from __future__ import annotations

import re
from typing import ClassVar

try:
    from .clinical_safety_checker import ClinicalContentAnalyzer
except ModuleNotFoundError:
    from ai.training.clinical_safety_checker import ClinicalContentAnalyzer


class MultilingualContentChecker(ClinicalContentAnalyzer):
    """Extended content analyzer with multilingual crisis pattern support.

    Adds crisis and negation patterns for es, fr, pt, and de. Inherits the
    English patterns and negation-window logic from ClinicalContentAnalyzer.
    When the language is unknown, all pattern sets are analyzed.
    """

    VERSION: ClassVar[str] = "1.0.0"

    supported_languages: ClassVar[list[str]] = ["en", "es", "fr", "pt", "de"]

    _EN_CRISIS_EXTRA_RAW: ClassVar[tuple[str, ...]] = (
        r"(?<!\w)(took|take|taking|took|swallowed|swallow|ingested)\s+(too\s+many|all|every|all the|many)\s+(pill|pills|tablet|tablets|medication|medications)(?!\w)",
    )

    _ES_CRISIS_RAW: ClassVar[tuple[str, ...]] = (
        r"(?<!\w)(suicidio|suicidarme|quitarme la vida|acabar con mi vida)(?!\w)",
        r"(?<!\w)(matarme|quiero morir|mejor muerto)(?!\w)",
        r"(?<!\w)(autolesi[oó]n|cortarme|hacerme da[nñ]o)(?!\w)",
        r"(?<!\w)(no tengo raz[oó]n para vivir|sin raz[oó]n para vivir)(?!\w)",
        r"(?<!\w)(quiero morirme|no quiero seguir viviendo)(?!\w)",
        r"(?<!\w)(sobredosis de medicamento|sobredosis|pensamientos suicidas|tomar todas las pastillas)(?!\w)",
    )

    _ES_NEGATION_RAW: ClassVar[tuple[str, ...]] = (
        r"\b(no|nunca|nadie|nada|ning[uú]n|ninguna|tampoco|jam[aá]s)\b",
    )

    _FR_CRISIS_RAW: ClassVar[tuple[str, ...]] = (
        r"(?<!\w)(suicide|me suicider|mettre fin [aà] ma vie)(?!\w)",
        r"(?<!\w)(me tuer|vouloir mourir|mieux mort)(?!\w)",
        r"(?<!\w)(auto[- ]?blessure|me couper|me faire du mal)(?!\w)",
        r"(?<!\w)(aucune raison de vivre|plus rien [aà] vivre)(?!\w)",
        r"(?<!\w)(je veux mourir|envie de mourir)(?!\w)",
        r"(?<!\w)(surdosage de m[eé]dicament|surdosage|une id[eé]e de mourir|avaler tous les m[eé]dicaments)(?!\w)",
    )

    _FR_NEGATION_RAW: ClassVar[tuple[str, ...]] = (
        r"\b(ne|pas|jamais|rien|personne|aucun|aucune|non plus)\b",
    )

    _PT_CRISIS_RAW: ClassVar[tuple[str, ...]] = (
        r"(?<!\w)(suic[ií]dio|me suicidar|acabar com a minha vida)(?!\w)",
        r"(?<!\w)(me matar|quero morrer|melhor morto)(?!\w)",
        r"(?<!\w)(autoles[aã]o|me cortar|me machucar)(?!\w)",
        r"(?<!\w)(sem raz[aã]o para viver|nenhuma raz[aã]o para viver)(?!\w)",
        r"(?<!\w)(quero morrer|n[aã]o quero continuar vivendo)(?!\w)",
        r"(?<!\w)(overdose de rem[eé]dio|overdose|fazer uma overdose|tomar todos os rem[eé]dios)(?!\w)",
    )

    _PT_NEGATION_RAW: ClassVar[tuple[str, ...]] = (
        r"\b(n[aã]o|nunca|ningu[eé]m|nada|nenhum|nenhuma|tampouco|jamais)\b",
    )

    _DE_CRISIS_RAW: ClassVar[tuple[str, ...]] = (
        r"(?<!\w)(selbstmord|suizid|mir das leben nehmen)(?!\w)",
        r"(?<!\w)(mich umbringen|mich t[oö]ten|will sterben|lieber tot)(?!\w)",
        r"(?<!\w)(selbstverletzung|mich schneiden|mich verletzen)(?!\w)",
        r"(?<!\w)(kein grund zu leben|keinen grund mehr zu leben)(?!\w)",
        r"(?<!\w)(ich will sterben|m[oö]chte sterben)(?!\w)",
        r"(?<!\w)(medikamenten[eü]berdosis|alle tabletten nehmen|suizidale gedanken)(?!\w)",
    )

    _DE_NEGATION_RAW: ClassVar[tuple[str, ...]] = (
        r"\b(nicht|kein|keine|keiner|niemals|niemand|nichts|nirgends|nie)\b",
    )

    _ES_CRISIS: ClassVar[list[re.Pattern[str]]] = [re.compile(p) for p in _ES_CRISIS_RAW]
    _ES_NEGATION: ClassVar[list[re.Pattern[str]]] = [re.compile(p) for p in _ES_NEGATION_RAW]
    _FR_CRISIS: ClassVar[list[re.Pattern[str]]] = [re.compile(p) for p in _FR_CRISIS_RAW]
    _FR_NEGATION: ClassVar[list[re.Pattern[str]]] = [re.compile(p) for p in _FR_NEGATION_RAW]
    _PT_CRISIS: ClassVar[list[re.Pattern[str]]] = [re.compile(p) for p in _PT_CRISIS_RAW]
    _PT_NEGATION: ClassVar[list[re.Pattern[str]]] = [re.compile(p) for p in _PT_NEGATION_RAW]
    _DE_CRISIS: ClassVar[list[re.Pattern[str]]] = [re.compile(p) for p in _DE_CRISIS_RAW]
    _DE_NEGATION: ClassVar[list[re.Pattern[str]]] = [re.compile(p) for p in _DE_NEGATION_RAW]
    _EN_CRISIS: ClassVar[list[re.Pattern[str]]] = [re.compile(p) for p in _EN_CRISIS_EXTRA_RAW]

    _LANG_PATTERNS: ClassVar[dict[str, tuple[list[re.Pattern[str]], list[re.Pattern[str]]]]] = {
        "en": (ClinicalContentAnalyzer.CRISIS_PATTERNS + _EN_CRISIS, ClinicalContentAnalyzer.NEGATION_PATTERNS),
        "es": (_ES_CRISIS, _ES_NEGATION),
        "fr": (_FR_CRISIS, _FR_NEGATION),
        "pt": (_PT_CRISIS, _PT_NEGATION),
        "de": (_DE_CRISIS, _DE_NEGATION),
    }

    @classmethod
    def contains_crisis_keywords(cls, text: str, language: str | None = None) -> bool:
        """Check text for crisis content across one or all languages.

        Args:
            text: Input text to check.
            language: ISO 639-1 code (en/es/fr/pt/de). When None or
                unrecognized, all language pattern sets are checked.

        Returns:
            True if any non-negated crisis match is found.
        """
        if not text or not isinstance(text, str):
            return False

        if language is not None and language in cls._LANG_PATTERNS:
            crisis_pats, neg_pats = cls._LANG_PATTERNS[language]
            return cls._check_patterns(text, crisis_pats, neg_pats)

        for crisis_pats, neg_pats in cls._LANG_PATTERNS.values():
            if cls._check_patterns(text, crisis_pats, neg_pats):
                return True
        return False

    @classmethod
    def _check_patterns(
        cls,
        text: str,
        crisis_patterns: list[re.Pattern[str]],
        negation_patterns: list[re.Pattern[str]],
    ) -> bool:
        """Run crisis pattern matching with negation-window suppression."""
        text_lower = text.lower()
        for pattern in crisis_patterns:
            for match in pattern.finditer(text_lower):
                pre_start = max(0, match.start() - cls.NEGATION_WINDOW)
                pre_context = text_lower[pre_start:match.start()]
                if any(neg.search(pre_context) for neg in negation_patterns):
                    continue

                post_end = min(len(text_lower), match.end() + cls.NEGATION_WINDOW_AFTER)
                post_raw = text_lower[match.end():post_end]
                sent_boundary = cls._SENTENCE_END.search(post_raw)
                post_context = post_raw[:sent_boundary.start()] if sent_boundary else post_raw
                if any(neg.search(post_context) for neg in negation_patterns):
                    continue

                return True
        return False
