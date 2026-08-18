"""English-only clinical content analysis utilities for training data.

Provides crisis pattern definitions, negation detection, and context window
helpers used by the multilingual content analyzer. This module provides
pattern-matching infrastructure — it does not filter or remove content.
"""

from __future__ import annotations

import re
from typing import ClassVar


class ClinicalContentAnalyzer:
    """Pattern definitions and utilities for analyzing clinical content.

    Attributes:
        VERSION: Bump when patterns change to track dataset provenance.
        CRISIS_PATTERNS: Pre-compiled crisis expression patterns.
        NEGATION_PATTERNS: Pre-compiled syntactic negation patterns.
        NEGATION_WINDOW: Context window (chars) before match for negation check.
        NEGATION_WINDOW_AFTER: Context window (chars) after match for negation check.

    Note:
        These patterns are used for analysis and measurement only — they do not
        gate or filter content. All content is preserved for training.
    """

    VERSION: ClassVar[str] = "1.2.0"

    _CRISIS_PATTERNS_RAW: ClassVar[tuple[str, ...]] = (
        r"(?<!\w)(suicide|kill myself|end my life|want to die|better off dead)(?!\w)",
        r"(?<!\w)(self[- ]?harm|cutting|hurting myself)(?!\w)",
        r"(?<!\w)(goodbye|farewell)\s+(world|everyone|all)(?!\w)",
        r"(?<!\w)(take|swallow|overdose on)\s+(all|every|my)\s+(pill|tablet|medication)(?!\w)",
        r"(?<!\w)(can't go on|cannot go on|no reason to live|no point in living)(?!\w)",
        r"(?<!\w)(everyone would be better (off )?without me)(?!\w)",
        r"(?<!\w)(i('ve| have) made my (final )?decision)(?!\w)",
        r"(?<!\w)(nothing left (to live for|for me))(?!\w)",
        r"(?<!\w)(don't want to be here anymore|do not want to be here anymore)(?!\w)",
        r"(?<!\w)(i('m| am) done fighting)(?!\w)",
        r"(?<!\w)(just want (it|the pain) to stop)(?!\w)",
        r"(?<!\w)(i('ve| have) been stockpiling)(?!\w)",
        r"(?<!\w)(said my goodbyes)(?!\w)",
        r"(?<!\w)(i have a (plan|method|way))(?!\w)",
        r"(?<!\w)(i know how i('ll| will) do it)(?!\w)",
        r"(?<!\w)(i('ve| have) written (a |my )?(note|letter))(?!\w)",
        r"(?<!\w)(i('ve| have) given away my (things|stuff|belongings|possessions))(?!\w)",
        r"(?<!\w)(i('ve| have) already decided)(?!\w)",
    )

    _NEGATION_PATTERNS_RAW: ClassVar[tuple[str, ...]] = (
        r"\b(not|don't|doesn't|didn't|won't|wouldn't|shouldn't|can't|cannot)\b",
        r"\b(no|never|nobody|nothing|nowhere)\b",
    )

    CRISIS_PATTERNS: ClassVar[list[re.Pattern[str]]] = [re.compile(p) for p in _CRISIS_PATTERNS_RAW]
    NEGATION_PATTERNS: ClassVar[list[re.Pattern[str]]] = [re.compile(p) for p in _NEGATION_PATTERNS_RAW]
    NEGATION_WINDOW: ClassVar[int] = 100
    NEGATION_WINDOW_AFTER: ClassVar[int] = 30
    _SENTENCE_END: ClassVar[re.Pattern[str]] = re.compile(r"[.!?]")

    @classmethod
    def contains_crisis_keywords(cls, text: str) -> bool:
        """Check text for non-negated crisis content.

        Returns True if any non-negated crisis match is found.
        Returns False for empty strings, non-string input, or fully negated matches.
        """
        if not text or not isinstance(text, str):
            return False
        text_lower = text.lower()
        for pattern in cls.CRISIS_PATTERNS:
            for match in pattern.finditer(text_lower):
                pre_start = max(0, match.start() - cls.NEGATION_WINDOW)
                pre_context = text_lower[pre_start:match.start()]
                if any(neg.search(pre_context) for neg in cls.NEGATION_PATTERNS):
                    continue

                post_end = min(len(text_lower), match.end() + cls.NEGATION_WINDOW_AFTER)
                post_raw = text_lower[match.end():post_end]
                sent_boundary = cls._SENTENCE_END.search(post_raw)
                post_context = post_raw[:sent_boundary.start()] if sent_boundary else post_raw
                if any(neg.search(post_context) for neg in cls.NEGATION_PATTERNS):
                    continue

                return True
        return False
