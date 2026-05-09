"""Clinical validity scoring for therapeutic training data.

Measures the clinical quality of therapeutic responses across multiple
dimensions: therapeutic technique usage, therapeutic alliance, clinical
structure, cultural competence, and evidence-based practice.

This is a measurement and quality-assessment tool — it does not filter
or remove content from the training pipeline.
"""

from __future__ import annotations

import re
from typing import ClassVar


class ClinicalValidityScorer:
    """Score therapeutic responses on clinical validity dimensions.

    Attributes:
        VERSION: Bump when patterns change to track dataset provenance.
        TECHNIQUE_PATTERNS: Evidence-based therapeutic technique markers.
        ALLIANCE_PATTERNS: Therapeutic alliance and rapport-building markers.
        STRUCTURE_PATTERNS: Clinical session structure markers.
        CULTURAL_PATTERNS: Cultural competence and awareness markers.
        EBP_PATTERNS: Evidence-based practice markers.
    """

    VERSION: ClassVar[str] = "1.0.0"

    THERAPY_MODALITIES: ClassVar[dict[str, tuple[str, ...]]] = {
        "cbt": (
            r"\b(cognitive (restructuring|reframing|distortion|pattern|thought))\b",
            r"\b(thought record|thinking trap|automatic thought|core belief)\b",
            r"\b(behavioral activation|exposure|response prevention|homework)\b",
            r"\b(cbt|cognitive behavioral|behavioral experiment|activity scheduling)\b",
            r"\b(challeng(e|ing) (that )?thought|alternative perspective|evidence for)\b",
            r"\b(thought (diary|log|monitoring)|cognitive shift|reframe)\b",
        ),
        "dbt": (
            r"\b(dbt|dialectical|mindfulness (exercise|practice|skill))\b",
            r"\b(distress tolerance|wise mind|radical acceptance|TIPP)\b",
            r"\b(emotional regulation|emotion regulation|opposite action)\b",
            r"\b(interpersonal effectiveness|DEAR MAN|GIVE|FAST)\b",
            r"\b(chain analysis|behavior chain|check the facts)\b",
        ),
        "mi": (
            r"\b(motivational interviewing|change talk|sustain talk)\b",
            r"\b(decisional balance|importance (and )?confidence|readiness ruler)\b",
            r"\b(discrepancy|ambivalence|rolling with resistance)\b",
            r"\b(open(-| )ended question|affirm|reflective listen|summarize)\b",
            r"\b(OARS|elicit(-| )provide(-| )elicit)\b",
            r"\b(on a scale (from|of)|readiness|how important|how confident)\b",
            r"\b(pro(-| )con|benefit (of|to)|disadvantage|what (would|do) you (gain|lose))\b",
            r"\b(you (said|mention) (that )?you('re| are) (feeling|thinking|considering))\b",
            r"\b(ambivalent|mixed (feelings|emotions)|two (parts|sides)|part of you)\b",
        ),
        "act": (
            r"\b(act|acceptance and commitment|acceptance (of|that)|committed action)\b",
            r"\b(cognitive defusion|defusion (exercise|technique)|notice (that|your))\b",
            r"\b(values(-| )based (action|living)|value(s)? clarification)\b",
            r"\b(self as context|observing self|present moment awareness)\b",
            r"\b(experiential avoidance|willingness|creative hopelessness)\b",
        ),
        "psychodynamic": (
            r"\b(psychodynamic|psychoanalytic|transference|countertransference)\b",
            r"\b(early (experience|relationship|attachment)|unconscious pattern)\b",
            r"\b(defense mechanism|repetition compulsion|working through)\b",
            r"\b(attachment (style|pattern|theory)|object relations)\b",
            r"\b(here(-| )and(-| )now|therapeutic (alliance|relationship))\b",
        ),
        "person_centered": (
            r"\b(person(-| )centered|client(-| )centered|Rogerian)\b",
            r"\b(unconditional positive regard|UPR|prizing|warmth)\b",
            r"\b(congruence|genuineness|realness|authenticity)\b",
            r"\b(empathic understand|empathic reflection|accurate empathy)\b",
            r"\b(non(-| )judgmental|accepting|safe (space|environment))\b",
        ),
        "solution_focused": (
            r"\b(solution(-| )focused|brief therapy|SFBT)\b",
            r"\b(miracle question|exception (-| )finding|scaling question)\b",
            r"\b(goal setting|coping question|resources|strengths)\b",
            r"\b(what('s| is) different|what worked|progress (so far|made))\b",
            r"\b(next (small )?step|action plan|concrete goal)\b",
        ),
        "crisis_intervention": (
            r"\b(crisis (intervention|management|stabilization|plan))\b",
            r"\b(safety (plan|assessment|contract)|risk assessment)\b",
            r"\b(de-escalat|stabilize|immediate (support|safety))\b",
            r"\b(988|crisis line|emergency (contact|services)|hotline)\b",
            r"\b(warning sign|triggers|coping strateg|support network)\b",
        ),
    }

    ALLIANCE_PATTERNS: ClassVar[dict[str, tuple[str, ...]]] = {
        "collaboration": (
            r"\b(work together|collaborat|partner with|we can|let's)\b",
            r"\b(what do you think|how does that (feel|sound|land)|your input)\b",
            r"\b(together|shared (understanding|goal)|agenda)\b",
        ),
        "validation": (
            r"\b(that makes sense|understandably|it's understandable|of course)\b",
            r"\b(valid|legitimate|reasonable|makes (perfect )?sense)\b",
            r"\b(I hear you|I can see|that sounds|that must be)\b",
        ),
        "empowerment": (
            r"\b(you (have|are) (the )?(strength|resilience|courage|capable))\b",
            r"\b(you can|you get to decide|your choice|your (own )?pace)\b",
            r"\b(agency|autonomy|self(-| )efficacy|build(ing)? on)\b",
        ),
    }

    STRUCTURE_PATTERNS: ClassVar[dict[str, tuple[str, ...]]] = {
        "assessment": (
            r"\b(assessment|intake|evaluation|history|presenting (problem|concern))\b",
            r"\b(symptom|duration|onset|frequency|severity|intensity)\b",
            r"\b(previous (treatment|therapy|hospitalization)|medication)\b",
        ),
        "intervention": (
            r"\b(intervention|technique|strategy|approach|skill|tool)\b",
            r"\b(practice|exercise|worksheet|try|attempt)\b",
            r"\b(explore|examine|look at|delve|unpack)\b",
        ),
        "planning": (
            r"\b(goal|objective|outcome|treatment plan|therapeutic goal)\b",
            r"\b(between sessions|homework|practice|follow(-| )up|next step)\b",
            r"\b(progress|track|monitor|review|adjust)\b",
        ),
        "closure": (
            r"\b(summarize|wrap up|close|recap|reflect back)\b",
            r"\b(key (takeaway|point)|main (theme|idea)|what stood out)\b",
            r"\b(next session|check in|continue (working|exploring))\b",
        ),
    }

    CULTURAL_PATTERNS: ClassVar[dict[str, tuple[str, ...]]] = {
        "cultural_awareness": (
            r"\b(cultural|culture|diverse|background|identity)\b",
            r"\b(ethnic|racial|socioeconomic|faith|spiritual)\b",
            r"\b(community|cultural (context|factor|consideration|humility))\b",
        ),
        "inclusive_language": (
            r"\b(they|them|partner|folks|all (people|identities))\b",
            r"\b(LGBTQ|gender|pronoun|orientation|neurodivergen)\b",
            r"\b(access|barrier|systemic|equity|inclusive)\b",
        ),
    }

    EBP_PATTERNS: ClassVar[dict[str, tuple[str, ...]]] = {
        "research_informed": (
            r"\b(research|study|evidence|literature|finding)\b",
            r"\b(efficacy|effectiveness|outcome|meta-analysis|RCT)\b",
            r"\b(best practice|standard of care|clinical (guideline|practice))\b",
        ),
        "clinical_reasoning": (
            r"\b(conceptualiz|cased? formulation|clinical (judgment|reasoning))\b",
            r"\b(differential|diagnos|presentation|etiology|prognosis)\b",
            r"\b(indication|contraindication|referral|consultation)\b",
        ),
        "therapeutic_framing": (
            r"\b(one (way|approach) is|a (helpful|useful) way|some people find)\b",
            r"\b(in (my|our) experience|from a clinical perspective|professionally)\b",
            r"\b(evidence suggests|research (shows|indicates)|studies (show|suggest))\b",
        ),
    }

    WEIGHTS: ClassVar[dict[str, float]] = {
        "technique": 0.35,
        "alliance": 0.25,
        "structure": 0.15,
        "cultural": 0.10,
        "ebp": 0.15,
    }

    DIMENSION_PATTERNS: ClassVar[dict[str, dict[str, tuple[str, ...]]]] = {
        "technique": THERAPY_MODALITIES,
        "alliance": ALLIANCE_PATTERNS,
        "structure": STRUCTURE_PATTERNS,
        "cultural": CULTURAL_PATTERNS,
        "ebp": EBP_PATTERNS,
    }

    # Flatten per dimension for scoring
    _DIMENSION_REGEX: ClassVar[dict[str, re.Pattern[str]]] = {}

    @classmethod
    def _get_dimension_pattern(cls, dimension: str) -> re.Pattern[str]:
        if dimension not in cls._DIMENSION_REGEX:
            all_patterns: list[str] = []
            for sub_patterns in cls.DIMENSION_PATTERNS[dimension].values():
                all_patterns.extend(sub_patterns)
            cls._DIMENSION_REGEX[dimension] = re.compile("|".join(all_patterns), re.IGNORECASE)
        return cls._DIMENSION_REGEX[dimension]

    @classmethod
    def _score_dimension(cls, text: str, dimension: str) -> float:
        if not text or not isinstance(text, str):
            return 0.0
        pattern = cls._get_dimension_pattern(dimension)
        matches = list(pattern.finditer(text))
        if not matches:
            return 0.0
        sub_dimensions = len(cls.DIMENSION_PATTERNS[dimension])
        match_texts = {m.group(0).lower() for m in matches}
        diversity_bonus = min(1.0, len(match_texts) / 10.0)
        raw = min(len(matches) / (sub_dimensions * 2), 1.0)
        return min(raw + diversity_bonus * 0.15, 1.0)

    @classmethod
    def score(cls, response: str) -> float:
        """Compute overall clinical validity score in [0.0, 1.0]."""
        if not response or not isinstance(response, str):
            return 0.0
        total = 0.0
        for dimension, weight in cls.WEIGHTS.items():
            total += cls._score_dimension(response, dimension) * weight
        return total

    @classmethod
    def score_detail(cls, response: str) -> dict:
        """Compute per-dimension clinical validity scores."""
        if not response or not isinstance(response, str):
            return {d: 0.0 for d in cls.WEIGHTS}
        return {
            dimension: cls._score_dimension(response, dimension)
            for dimension in cls.WEIGHTS
        }
