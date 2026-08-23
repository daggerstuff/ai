"""
Heuristic toxicity detector for hackathon-sourced training data.

Distinct from the existing safety_ethics_validator.py (which is thin — only 3 self-harm
patterns), this module extends the safety surface for PIX-4240 with five categories the
brief requires:

  1. self_harm_suicide   — graphic ideation/references (NOT clinical discussion)
  2. graphic_violence     — violence/abuse descriptions
  3. hate_discrimination  — slurs, discriminatory dehumanization
  4. inappropriate_sexual — explicit content inappropriate for clinical training
  5. manipulative_coercive — manipulation/coercion tactics

The detector is a hazard assessor — it returns a structured ToxicityResult attached
to the record, NOT a deletion. Disposition (route to toxic output shard vs leave for
review) lives in the HackathonSafetyProcessor.

Clinical content distinction (critical): each category pairs toxic-trigger patterns
with clinical-context cues. A trigger matches alone -> toxic. A trigger that occurs
with a clinical cue nearby (within CLINICAL_CONTEXT_WINDOW chars) is treated as
clinical discussion and NOT flagged. This handles false-negative-vs-over-filtering
tension explicitly per the PIX-4240 brief: clinical mental health data naturally
references trauma, substance use, difficult emotions, and self-harm language for
clinical purposes — that is training signal, not toxic content.

Pure-regrex + deterministic + no ML model dependency (works in AI_DISABLE_SAFETY_ML_MODELS=1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Window around a trigger to look for clinical-context cues (chars on each side)
CLINICAL_CONTEXT_WINDOW = 200


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------
# Each category is a list of (pattern, weight) tuples. Patterns are compiled
# regex; weight is 1.0 by default. A category is "matched" if at least one
# pattern matches without a clinical cue nearby.
#
# CLINICAL_CUES are phrases that indicate clinical framing (research/therapy
# context). When a clinical cue is found within CLINICAL_CONTEXT_WINDOW chars
# of a trigger match, the trigger is downgraded to "clinical" and does NOT
# raise the category.
# ---------------------------------------------------------------------------


# Clinical context cues — indicate professional / therapeutic / research framing.
# These exist for the SAME topics the triggers flag, so we can distinguish.
CLINICAL_CUES: list[str] = [
    "patient",
    "patients",
    "client",
    "clients",
    "therapist",
    "clinician",
    "clinical",
    "clinically",
    "therapeutic",
    "therapy",
    "counselling",
    "counseling",
    "counsel",
    "diagnosis",
    "diagnosed",
    "diagnostic",
    "dsm",
    "icd",
    "treatment",
    "intervention",
    "practitioner",
    "psychiatrist",
    "psychologist",
    "psychiatric",
    "psychiatry",
    "psychology",
    "social worker",
    "counselor",
    "session",
    "triage",
    "screening",
    "screened",
    "symptom",
    "symptoms",
    "ideation",
    "suicidal ideation",
    "risk assessment",
    "risk factor",
    "protective factor",
    "safety plan",
    "safety planning",
    "crisis",
    "crisis intervention",
    "presenting",
    "presents with",
    "presented",
    "history of",
    "hx of",
    "reports of",
    "reported",
    "disclosed",
    "disclosure",
    "case study",
    "case report",
    "study",
    "studies",
    "research",
    "findings",
    "cohort",
    "sample",
    "survey",
    "questionnaire",
    "scale",
    "instrument",
    "protocol",
    "guideline",
    "guidelines",
    "contraindicated",
    "cbt",
    "dbt",
    "emdr",
    "exposure",
    "relapse prevention",
    "inpatient",
    "outpatient",
    "in-patient",
    "outpatient",
    "hospitalized",
    "hospitalization",
    "admitted",
    "discharge",
    "discharged",
    "substance",
    "substance",
    "substance use",
    "substance abuse",
    "addiction",
    "addict",
    "dependence",
    "withdrawal",
    "relapse",
    "medication",
    "medications",
    "prescribed",
    "prescription",
    "dose",
    "dosage",
    "side effect",
    "adverse",
    "veteran",
    "veterans",
    "military",
    "combat",
    "ptsd",
    "trauma",
    "traumatic",
    "treatment plan",
    "care plan",
    "training data",
    "training example",
    "training sample",
    "evidence-based",
    "evidence based",
    "best practice",
]


# Self-harm / suicide ideation — graphic and direct, NOT clinical
# e.g. "I want to kill myself" → trigger
# But NOT: "patients may experience suicidal ideation"
SELF_HARM_SUICIDE_PATTERNS: list[tuple[str, float]] = [
    # Direct first-person intent
    (
        r"\bi\s+(?:want(?:\s+to|na)|wanna|gonna|going\s+to)\s+(?:kill|end|take)\s+(?:myself|my\s+own\s+life|my\s+life)\b",
        1.0,
    ),
    (r"\bi\s+(?:want|need)\s+to\s+die\b", 1.0),
    (r"\bkill\s+me\b", 1.0),
    (r"\bi(?:'m| am)\s+(?:going\s+to|gonna)\s+(?:kill\s+myself|end\s+(?:it|everything)|commit\s+suicide)\b", 1.0),
    (r"\bcommit(?:ting)?\s+suicide\b", 1.0),
    (r"\b(\d+)\s+(?:ways|methods)\s+to\s+(?:kill|end|take)\s+(?:yourself|oneself|myself)\b", 1.0),
    # Specific means
    (r"\bslit\s+(?:my|your|their)\s+(own\s+)?wrists?\b", 1.0),
    (r"\b(?:overdose|od)\s+(?:on|with)\s+\w+\s+(?:to\s+(?:kill|end)\s+(?:myself|my\s+life))\b", 1.0),
    (r"\b(?:jump|hanging|hang\s+myself|hang\s+himself|hang\s+herself)\b\s*(?:off|from|by|to)\s+", 1.0),
    # Intent with method mention (no clinical framing nearby)
    (
        r"\b(?:how\s+to|best\s+way\s+to|easiest\s+way\s+to|quickest\s+way\s+to)\s+(?:kill\s+myself|end\s+(?:my\s+life|it\s+all)|commit\s+suicide|die)\b",
        1.0,
    ),
    # Encouraging others — high severity
    (r"\byou\s+should\s+(?:kill\s+yourself|end\s+your\s+life|commit\s+suicide|die)\b", 1.0),
    (
        r"\b(?:everybody|everyone|no\s+one)\s+would\s+be\s+better\s+(?:off|up)\s+with(?:out|out)\s+(?:me|you|them|him|her)\b",
        0.85,
    ),
    # Self-harm intent (cutting, etc.) — first-person active
    (r"\bi\s+(?:cut|cutting|self[\s-]?harm(?:ed|ing)?)\s+(?:myself|my|because|x?to)\b", 0.85),
    (r"\bi\s+am\s+(?:cutting|self[\s-]?harming)\s+myself\b", 0.85),
]


# Graphic violence / abuse descriptions — direct violent or assaultive content
GRAPHIC_VIOLENCE_PATTERNS: list[tuple[str, float]] = [
    # Detailed infliction imagery (not abstract "violence" word)
    (
        r"\b(?:stabbed|stabbing|slit\s+(?:his|her|their)\s+throat|beat\s+(?:him|her|them)\s+(?:to\s+a\s+pulp|senseless|unconscious))\b",
        1.0,
    ),
    (
        r"\b(?:rape|raped|raping|molest|molested|molesting)\s+(?:(?:a|an|the|his|her|their|this)\s+)?(?:child|kid|minor|girl|boy|baby|infant|toddler)\b",
        1.0,
    ),
    (r"\b(?:gore|gory|brains|skull)\s+(?:splattered|everywhere|on\s+(?:the\s+)?(?:wall|ceiling))\b", 1.0),
    (r"\bdismember(?:ed|ing|ment)\b", 1.0),
    (r"\b(?:beheaded|behead(?:ing)?|decapitate[ds]?|decapitating|throat\s+slit)\b", 1.0),
    # Instructive violence
    (r"\bhow\s+to\s+(?:kill|murder|torture|rape|abuse)\s+(?:someone|a\s+person|him|her|them)\b", 1.0),
    # Graphic threats toward specific people
    (
        r"\bi(?:'ll| will|'m\s+going\s+to)\s+(?:kill|murder|stab|shoot|decapitate)\s+(?:you|him|her|them|that\s+bitch|that\s+(?:guy|fuck))\b",
        1.0,
    ),
]


# Hate speech / discriminatory — slurs and dehumanizing language
HATE_DISCRIMINATION_PATTERNS: list[tuple[str, float]] = [
    # Common slurs — direct references (word boundary anchored). These are by
    # nature explicit; include them so the heuristic catches them when spaCy
    # is unavailable. They are in clinical-training-ground NOT because they
    # should appear in clinical data, but because hackathon sources often
    # contain raw patient/Internet text that must be flagged.
    (r"\b(?:n[i1]gg(?:er|a|ah)\s*g?|n[i1]glets?)\b", 1.0),
    (r"\bf[a@1i]gg?ot\b", 1.0),
    (r"\bk[i1]ke\s*s?\b", 1.0),
    (r"\bsp[i1]c\s*s?\b", 1.0),
    (r"\bch[i1]nk\s*s?\b", 1.0),
    (r"\bwetback\s*s?\b", 1.0),
    (r"\btrann(?:y|ie)\s*s?\b", 1.0),
    (r"\bshemale\s*s?\b", 1.0),
    # Dehumanization
    (
        r"\b(?:all|these|those|the)\s+(?:niggers|jews|muslims|blacks|whites|spics|chinks|mexicans)\s+(?:are|should|deserve|must|need|ought)\b",
        0.9,
    ),
    (r"\b(?:subhuman|vermin|inferior\s+race|master\s+race|ethnic\s+cleansing|racial\s+purity)\b", 1.0),
    # Calls to discriminate
    (
        r"\b(?:kill|die|hang|lynch|gas|deport)\s+all\s+(?:niggers|jews|muslims|blacks|spics|chinks|wetbacks|trannies|faggots|gays|immigrants)\b",
        1.0,
    ),
]


# Inappropriate sexual content — explicit/pornographic/age-inappropriate
INAPPROPRIATE_SEXUAL_PATTERNS: list[tuple[str, float]] = [
    # Explicit pornographic/pedophilic
    (r"\b(?:child\s+porn|cp|kiddie\s+porn|pedo|pedophile|pedophilia|paedophile|paedophilia)\b", 1.0),
    (r"\b(?:underage\s+(?:girl|boy|teen|kid)|prepubescent|loli\s*con?|shota\s*con?)\b", 1.0),
    # Explicit anatomical/sex-act phrases (high specificity)
    (r"\b(?:penetrate|penetrated|penetration)\s+(?:anus|vagina|rectum|mouth)\b", 0.9),
    (r"\b(?:anal|oral|vaginal)\s+(?:sex|penetration|intercourse|rape)\b", 0.85),
    # Indecent exposure / explicit solicitation of minors
    (r"\bexposed?\s+(?:myself|himself|herself)\s+to\s+(?:a|an|the|some|young)?\s*(?:child|kid|minor|girl|boy)\b", 1.0),
    # Step-by-step sexual content directed at "you"
    (
        r"\bi\s+(?:want|am\s+going\s+to|'ll)\s+(?:suck|lick|kiss|fuck|bang|pound|pump)\s+(?:your|her|his)\s+(?:pussy|dick|cock|vagina|breasts?|ass|butt|anus)\b",
        0.85,
    ),
    # Ethnic-rape framing
    (r"\b(?:rape|raped)\s+(?:the|that|a)?\s*(?:bitch|slut|whore|cunt|nigger|faggot)\b", 1.0),
]


# Manipulative / coercive — tactics, NOT abstract "persuasion"
MANIPULATIVE_COERCIVE_PATTERNS: list[tuple[str, float]] = [
    # Gaslighting / reality-denial tactics
    (
        r"\byou(?:'re| are)\s+(?:crazy|insane|psycho|nuts|imagining\s+things|making\s+(?:this|it)\s+up|delusional)\b",
        0.85,
    ),
    # Soft-coercion / victim-blaming patterns
    (r"\bif\s+you\s+(?:really\s+)?loved\s+me\s*,?\s+you(?:'d| would)\b", 0.85),
    (r"\bnobody\s+(?:else|else)\s+will\s+ever\s+love\s+(?:you|him|her|them)\b", 0.85),
    (r"\bit(?:'s| is)\s+(?:your|all\s+your)\s+fault\s+(?:i|we|this|everything)\b", 0.85),
    # Threat of self-harm as manipulation. Accept variants like:
    #   "If you leave, I'll kill myself"
    #   "If you leave me, I'll kill myself"
    #   "If you leave him, I will hurt myself"
    (
        r"\bif\s+you\s+(?:leave|don't|do\s+not|break\s+up(?:\s+with\s+(?:me|him|her|them))?|end\s+this)"
        r"(?:\s+(?:me|him|her|them))?"
        r"\s*,?\s*i(?:'ll| will)\s+(?:kill\s+myself|end\s+it|die|hurt\s+myself)\b",
        1.0,
    ),
    # Isolation tactics
    (
        r"\b(?:don't|do\s+not|stop)\s+(?:talk|talking|see|seeing|hang|hangning)\s+to\s+(?:your|your\s+friends|family|mom|dad|parents|sister|brother)\b",
        0.85,
    ),
    (
        r"\b(?:they(?:'re| are)|your\s+(?:friends|family))\s+(?:don't|do\s+not)\s+(?:really\s+)?care\s+about\s+you\b",
        0.85,
    ),
    # Financial / log-coercion
    (r"\bgive\s+me\s+(?:your|the)\s+(?:password|pin|ssn|social\s+security|bank\s+(?:account|info|routing))\b", 1.0),
    (
        r"\b(?:wire|send|transfer)\s+(?:me|us)\s+\$?\s*\d{3,}\s+(?:and\s+i(?:'ll| will)\s+(?:love\s+you|forgive\s+you|be\s+yours))\b",
        0.9,
    ),
]


# Map category name -> (patterns, default weight) for runtime iteration
_CATEGORY_DEFS: dict[str, list[tuple[str, float]]] = {
    "self_harm_suicide": SELF_HARM_SUICIDE_PATTERNS,
    "graphic_violence": GRAPHIC_VIOLENCE_PATTERNS,
    "hate_discrimination": HATE_DISCRIMINATION_PATTERNS,
    "inappropriate_sexual": INAPPROPRIATE_SEXUAL_PATTERNS,
    "manipulative_coercive": MANIPULATIVE_COERCIVE_PATTERNS,
}


# Precompile clinical-cue regex once for speed
_CLINICAL_CUE_RE = re.compile(
    r"|".join(r"\b" + re.escape(cue) + r"\b" for cue in CLINICAL_CUES),
    re.IGNORECASE | re.UNICODE,
)


@dataclass
class ToxicityFinding:
    """One toxic trigger match (already classified toxic vs clinical-surrounded)."""

    category: str
    pattern: str  # The pattern source (regex source string)
    matched_text: str
    start: int
    end: int
    weight: float
    flagged_clinical: bool  # True if a clinical cue was found nearby → NOT raised as toxic


@dataclass
class CategoryResult:
    """Per-category detection result."""

    name: str
    triggered: bool  # True if at least one toxic (non-clinical) match
    score: float  # Sum of weights of toxic matches (clinical ones don't add)
    findings: list[ToxicityFinding] = field(default_factory=list)
    clinical_matches: list[ToxicityFinding] = field(default_factory=list)  # downgraded


@dataclass
class ToxicityResult:
    """Aggregate result of toxicity detection across one record's text."""

    is_toxic: bool
    score: float
    categories: dict[str, CategoryResult] = field(default_factory=dict)
    findings: list[ToxicityFinding] = field(default_factory=list)
    text_length: int = 0

    def __post_init__(self) -> None:
        # Defensive: total score is sum of triggered category scores.
        if not self.categories:
            object.__setattr__(self, "score", 0.0)
            object.__setattr__(self, "is_toxic", False)
            return
        total = sum(cr.score for cr in self.categories.values() if cr.triggered)
        object.__setattr__(self, "score", round(total, 4))
        object.__setattr__(self, "is_toxic", total > 0.0)


class HeuristicToxicityDetector:
    """
    Heuristic toxicity detector for hackathon-sourced clinical training data.

    Distinguishes toxic content from legitimate clinical discussion via
    paired trigger/clinical-cue pattern sets. Returns a structured
    ToxicityResult — never deletes or modifies input text.

    Pure regex, no ML model dependency. Deterministic and reproducible.
    """

    # Categories are fixed to match PIX-4240 brief requirements.
    CATEGORY_NAMES: tuple[str, ...] = (
        "self_harm_suicide",
        "graphic_violence",
        "hate_discrimination",
        "inappropriate_sexual",
        "manipulative_coercive",
    )

    def __init__(self, clinical_context_window: int = CLINICAL_CONTEXT_WINDOW):
        self._window = clinical_context_window
        self._compiled: dict[str, list[re.Pattern]] = {}
        for cat_name, patterns in _CATEGORY_DEFS.items():
            self._compiled[cat_name] = [re.compile(p, re.IGNORECASE | re.UNICODE) for p, _ in patterns]
        # Also keep (source_string, weight) for reporting
        self._pattern_meta: dict[str, list[tuple[str, float]]] = {
            cat: [(p, w) for p, w in patterns] for cat, patterns in _CATEGORY_DEFS.items()
        }

    # -- public API ---------------------------------------------------------

    def detect(self, text: str) -> ToxicityResult:
        """Detect toxicity in a single text string."""
        if not text or not isinstance(text, str):
            return ToxicityResult(is_toxic=False, score=0.0, categories={}, findings=[], text_length=0)

        category_results: dict[str, CategoryResult] = {}
        all_findings: list[ToxicityFinding] = []

        for cat_name in self.CATEGORY_NAMES:
            compiled_patterns = self._compiled[cat_name]
            meta = self._pattern_meta[cat_name]
            cat = CategoryResult(name=cat_name, triggered=False, score=0.0)

            for compiled, (source, weight) in zip(compiled_patterns, meta):
                for match in compiled.finditer(text):
                    start, end = match.start(), match.end()
                    matched_text = match.group()
                    # Look for clinical cue within ±window chars
                    window_start = max(0, start - self._window)
                    window_end = min(len(text), end + self._window)
                    window = text[window_start:window_end]
                    has_clinical = bool(_CLINICAL_CUE_RE.search(window))

                    finding = ToxicityFinding(
                        category=cat_name,
                        pattern=source,
                        matched_text=matched_text,
                        start=start,
                        end=end,
                        weight=weight,
                        flagged_clinical=has_clinical,
                    )
                    if has_clinical:
                        cat.clinical_matches.append(finding)
                    else:
                        cat.findings.append(finding)
                        cat.triggered = True
                        cat.score = round(cat.score + weight, 4)
                        all_findings.append(finding)

            category_results[cat_name] = cat

        result = ToxicityResult(
            is_toxic=False,  # __post_init__ will recompute from categories
            score=0.0,  # __post_init__ will recompute
            categories=category_results,
            findings=all_findings,
            text_length=len(text),
        )
        return result

    def detect_record(self, chatml_record: dict[str, Any]) -> ToxicityResult:
        """
        Detect toxicity across all string content of a ChatML record.

        Concatenates role-tagged message bodies with position tracking so that
        findings point to span offsets in the *concatenated* representation.
        System messages are excluded from cue scanning — system prompts carry
        pipeline boilerplate ("clinical assistant", "empathetic therapist")
        that would leak clinical cues and downgrade real patient-generated
        toxicity. Callers using per-message detection should call detect()
        per message instead.
        """
        if not isinstance(chatml_record, dict):
            return ToxicityResult(is_toxic=False, score=0.0, categories={}, findings=[], text_length=0)

        parts: list[str] = []
        for msg in chatml_record.get("messages", []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                continue
            parts.append(f"[{role}] {content}")
        meta = chatml_record.get("metadata", {})
        if isinstance(meta, dict):
            for k, v in meta.items():
                if isinstance(v, str):
                    parts.append(f"[meta:{k}] {v}")

        # Insert a boundary pad wider than CLINICAL_CONTEXT_WINDOW between
        # messages so the ±window check in detect() cannot bleed a clinical
        # cue from the tail of one message into the head of the next.
        _boundary = "\n" + ("#" * (self._window + 10)) + "\n"
        combined = _boundary.join(parts) if parts else ""
        return self.detect(combined)

    # -- introspection ------------------------------------------------------

    def category_names(self) -> tuple[str, ...]:
        """Return the tuple of category names (for tests / reporting)."""
        return self.CATEGORY_NAMES

    def clinical_cues(self) -> tuple[str, ...]:
        """Return the tuple of clinical-cue strings the detector uses."""
        return tuple(CLINICAL_CUES)


__all__ = [
    "CLINICAL_CONTEXT_WINDOW",
    "CLINICAL_CUES",
    "CategoryResult",
    "HeuristicToxicityDetector",
    "ToxicityFinding",
    "ToxicityResult",
]
