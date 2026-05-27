"""Clinical validity scoring for therapeutic training data.

Measures the *keyword density* of therapeutic responses across multiple
dimensions: therapeutic technique usage, therapeutic alliance, clinical
structure, cultural competence, and evidence-based practice.

LIMITATION: This scorer uses regex keyword matching — it measures how densely
a response uses therapeutic vocabulary, NOT how clinically sound the content
is. A long, generic transcript dump with therapeutic buzzwords may score
higher than a concise, clinically precise response. Use the per-message
averaging approach (see score_conversations in extract_therapist_voice.py)
to mitigate verbosity inflation.

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

    VERSION: ClassVar[str] = "2.0.0"

    THERAPY_MODALITIES: ClassVar[dict[str, tuple[str, ...]]] = {
        "cbt": (
            r"\b(cognitive (restructuring|reframing|distortions?|patterns?|thoughts?))\b",
            r"\b(thought records?|thinking traps?|automatic thoughts?|core beliefs?)\b",
            r"\b(behavioral activation|exposure|response prevention|homework)\b",
            r"\b(cbt|cognitive behavioral|behavioral experiments?|activity scheduling)\b",
            r"\b(challeng(e|ing) (that )?thoughts?|alternative perspectives?|evidence for)\b",
            r"\b(thought (diaries?|logs?|monitoring)|cognitive shifts?|reframes?)\b",
        ),
        "dbt": (
            r"\b(dbt|dialectical|mindfulness (exercises?|practices?|skills?))\b",
            r"\b(distress tolerance|wise mind|radical acceptance|TIPP)\b",
            r"\b(emotional regulation|emotion regulation|opposite action)\b",
            r"\b(interpersonal effectiveness|DEAR MAN|GIVE|FAST)\b",
            r"\b(chain analysis|behavior chain|check the facts)\b",
        ),
        "mi": (
            r"\b(motivational interviewing|change talk|sustain talk)\b",
            r"\b(decisional balance|importance (and )?confidence|readiness rulers?)\b",
            r"\b(discrepancy|ambivalence|rolling with resistance)\b",
            r"\b(open(-| )ended questions?|affirm\w*|reflective listen\w*|summariz\w*)\b",
            r"\b(OARS|elicit(-| )provide(-| )elicit)\b",
            r"\b(on a scale (from|of)|readiness|how important|how confident)\b",
            r"\b(pro(-| )con|benefits? (of|to)|disadvantages?|what (would|do) you (gain|lose))\b",
            r"\b(you (said|mention) (that )?you('re| are) (feeling|thinking|considering))\b",
            r"\b(ambivalent|mixed (feelings|emotions)|two (parts|sides)|part of you)\b",
        ),
        "act": (
            r"\b(act|acceptance and commitment|acceptance (of|that)|committed actions?)\b",
            r"\b(cognitive defusion|defusion (exercises?|techniques?)|notice (that|your))\b",
            r"\b(values?(-| )based (action|living)|value(s)? clarification)\b",
            r"\b(self as context|observing self|present moment awareness)\b",
            r"\b(experiential avoidance|willingness|creative hopelessness)\b",
        ),
        "psychodynamic": (
            r"\b(psychodynamic|psychoanalytic|transference|countertransference)\b",
            r"\b(early (experiences?|relationships?|attachments?)|unconscious patterns?)\b",
            r"\b(defense mechanisms?|repetition compulsion|working through)\b",
            r"\b(attachment (styles?|patterns?|theory)|object relations)\b",
            r"\b(here(-| )and(-| )now|therapeutic (alliance|relationship))\b",
        ),
        "person_centered": (
            r"\b(person(-| )centered|client(-| )centered|Rogerian)\b",
            r"\b(unconditional positive regard|UPR|prizing|warmth)\b",
            r"\b(congruence|genuineness|realness|authenticity)\b",
            r"\b(empathic understand\w*|empathic reflection|accurate empathy)\b",
            r"\b(non(-| )judgmental|accepting|safe (space|environment))\b",
        ),
        "solution_focused": (
            r"\b(solution(-| )focused|brief therapy|SFBT)\b",
            r"\b(miracle questions?|exception (-| )finding|scaling questions?)\b",
            r"\b(goal setting|coping questions?|resources|strengths)\b",
            r"\b(what('s| is) different|what worked|progress (so far|made))\b",
            r"\b(next (small )?steps?|action plans?|concrete goals?)\b",
        ),
        "crisis_intervention": (
            r"\b(crisis (intervention|management|stabilization|plan))\b",
            r"\b(safety (plan|assessment|contract)|risk assessment)\b",
            r"\b(de-escalat\w*|stabiliz\w*|immediate (support|safety))\b",
            r"\b(988|crisis line|emergency (contact|services)|hotline)\b",
            r"\b(warning signs?|triggers?|coping strateg\w*|support network)\b",
        ),
        "supportive_counseling": (
            r"\b(it (sounds|seems) like (you|you're)|sounds like you (are|have|need))\b",
            r"\b(I hear (you|that|what you)|I can (hear|see) (that|how))\b",
            r"\b(encourage you to|it('s| is) (important|helpful) (to|that))\b",
            r"\b(you are not alone|that's (completely|perfectly) (normal|understandable))\b",
            r"\b(reach out to|speak with a|talk to a|consult a)\b",
            r"\b(professional help|mental health professional|therapist (who|that) (specializes|can))\b",
            r"\b(coping (skills?|strateg\w*|mechanisms?|tools?)|self(-| )care|support (system|network))\b",
            r"\b(what you are (experiencing|feeling|going through)|you're (experiencing|feeling|going through))\b",
            r"\b(first step|next step|small (step|changes)|take (step|action))\b",
            r"\b(it('s| is) (okay|normal|common) to (feel|have|experience))\b",
            r"\b(writing in a journal|journaling|keep a (journal|diary)|write down)\b",
            r"\b(you deserve|your feelings are (valid|real|important)|it's okay to feel)\b",
        ),
        "trauma_informed": (
            r"\btrauma(-| )?(informed|specific|focused|sensitive|aware)\b",
            r"\b(triggers?|triggering|grounding|grounding (techniques?|exercises?))\b",
            r"\b(emotional safety|safe (spaces?|environments?|relationships?)|felt safety)\b",
            r"\b(hypervigilance|hyperarousal|flashbacks?|intrusive (thoughts?|memories?))\b",
            r"\b(window of tolerance|nervous system|fight or flight|freeze responses?)\b",
            r"\b(PTSD|complex trauma|developmental trauma|childhood trauma)\b",
            r"\b(trauma (recovery|healing|therap\w*|treatment\w*)|re(-| )traumatiz\w*)\b",
            r"\b(emotional (regulation|dysregulation)|self(-| )regulation|co(-| )regulation)\b",
            r"\b(safety (plans?|contracts?|assessments?)|stabilization|containment)\b",
        ),
        "somatic_therapy": (
            r"\bsomatic\b|\bsomatic (experiencing|therapy|practice)\b",
            r"\b(body (awareness|sensations?|scan|based)|(awareness|scan) of (the )?(body|physical))\b",
            r"\b(breathwork|breathing (exercises?|techniques?|practices?)|deep breaths?)\b",
            r"\b(mindful (movements?|breathing)|yoga|gentle stretch\w*)\b",
            r"\b(physical (sensations?|feelings?|experiences?)|embodiment|embodied)\b",
            r"\b(progressive muscle relaxation|relaxation (techniques?|exercises?|responses?))\b",
            r"\b(vagus nerve|polyvagal|nervous system (regulation|health))\b",
            r"\b(tension (release\w*|reduction\w*)|relax\w* (muscles?|body)|bodywork)\b",
        ),
    }

    ALLIANCE_PATTERNS: ClassVar[dict[str, tuple[str, ...]]] = {
        "collaboration": (
            r"\b(work together|collaborat\w*|partner with|we can|let's)\b",
            r"\b(what do you think|how does that (feel|sound|land)|your input)\b",
            r"\b(together|shared (understanding|goal)|agenda)\b",
            r"\b(we ('re| are) (in this|here) together|both of you)\b",
        ),
        "validation": (
            r"\b(that makes sense|understandably|it's understandable|of course)\b",
            r"\b(valid|legitimate|reasonable|makes (perfect )?sense)\b",
            r"\b(I hear you|I can see|that sounds|that must be)\b",
            r"\b(you are not alone|it('s| is) (okay|normal|understandable) (to|that))\b",
            r"\b(courage (to|in)|brave|strength (in|to)|resilience?|resilient)\b",
            r"\b(thank you for (sharing|trusting|reaching|opening))\b",
            r"\b(it takes (a lot of |so much )?courage|I('m| am) (glad|grateful) (you|that))\b",
        ),
        "empowerment": (
            r"\b(you (have|are) (the )?(strength|resilience|courage|capable))\b",
            r"\b(you can|you get to decide|your choice|your (own )?pace)\b",
            r"\b(agency|autonomy|self(-| )efficacy|build(ing)? on)\b",
            r"\b(you (are|were) able to|you (have|found) the (strength|resources))\b",
            r"\b(you (deserve|matter)|worth (it|your time|the effort))\b",
        ),
    }

    STRUCTURE_PATTERNS: ClassVar[dict[str, tuple[str, ...]]] = {
        "assessment": (
            r"\b(assessment|intake|evaluation|history|presenting (problems?|concerns?))\b",
            r"\b(symptoms?|duration|onset|frequency|severity|intensity)\b",
            r"\b(previous (treatment|therapy|hospitalization)|medication)\b",
            r"\b(how long (have|does)|when did (you|it)|how often (do|does))\b",
            r"\b(it sounds like you (are|have|may be|might be) (experiencing|dealing|struggling))\b",
        ),
        "intervention": (
            r"\b(interventions?|techniques?|strategies?|approach\w*|skills?|tools?)\b",
            r"\b(practice|exercises?|worksheets?|try|attempt)\b",
            r"\b(explore|examine|look at|delve|unpack)\b",
            r"\b(suggest (that|you|trying|considering)|recommend (that|you|trying|seeking))\b",
            r"\b(one (thing|approach|strategy) (that|you|is)|a (good|helpful) (way|idea) (to|is))\b",
            r"\b(work on|focus on|address (the|this|these|your))\b",
        ),
        "planning": (
            r"\b(goals?|objectives?|outcomes?|treatment plans?|therapeutic goals?)\b",
            r"\b(between sessions|homework|practice|follow(-| )up|next steps?)\b",
            r"\b(progress|track|monitor|review|adjust)\b",
            r"\b(develop (a|an|strateg\w*|plans?|skills?)|build(ing)? (on|upon|skills?|coping))\b",
        ),
        "closure": (
            r"\b(summarize|wrap up|close|recap|reflect back)\b",
            r"\b(key (takeaways?|points?)|main (themes?|ideas?)|what stood out)\b",
            r"\b(next session|check in|continue (working|exploring))\b",
            r"\b(I (hope|wish) (this|that|you)|wishing you|best of luck|hope this (helps|is))\b",
        ),
    }

    CULTURAL_PATTERNS: ClassVar[dict[str, tuple[str, ...]]] = {
        "cultural_awareness": (
            r"\b(cultural|culture|diverse|background|identity)\b",
            r"\b(ethnic|racial|socioeconomic|faith|spiritual)\b",
            r"\b(community|cultural (context|factors?|considerations?|humility))\b",
        ),
        "inclusive_language": (
            r"\b(they|them|partner|folks|all (people|identities))\b",
            r"\b(LGBTQ|gender|pronouns?|orientations?|neurodivergent|neurodivergence)\b",
            r"\b(access|barrier|systemic|equity|inclusive)\b",
        ),
    }

    EBP_PATTERNS: ClassVar[dict[str, tuple[str, ...]]] = {
        "research_informed": (
            r"\b(research|study|evidence|literature|findings?)\b",
            r"\b(efficacy|effectiveness|outcomes?|meta-analysis|RCT)\b",
            r"\b(best practice|standards? of care|clinical (guidelines?|practices?))\b",
            r"\b(therapy (has been )?(shown|found|demonstrated)|treatment (options?|approaches?))\b",
        ),
        "clinical_reasoning": (
            r"\b(conceptualiz\w*|cased? formulation|clinical (judgment|reasoning))\b",
            r"\b(differential|diagnos\w*|presentations?|etiology|prognosis)\b",
            r"\b(indications?|contraindications?|referrals?|consultations?)\b",
            r"\b(symptoms? (of|suggest\w*|indicat\w*)|consistent with|present with)\b",
            r"\b(treatable|responsive to|management (of|strateg\w*)|interventions?)\b",
        ),
        "therapeutic_framing": (
            r"\b(one (way|approach) is|a (helpful|useful) way|some people find)\b",
            r"\b(in (my|our) experience|from a clinical perspective|professionally)\b",
            r"\b(evidence suggests|research (shows|indicates)|studies (show|suggest))\b",
            r"\b(consider (trying|exploring|working with)|worth (considering|exploring|trying))\b",
            r"\b(goals? (is|would be) to|aims? (is|would be) to|approaches? (is|involves))\b",
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
        raw = min(len(matches) / sub_dimensions, 1.0)
        return min(raw + diversity_bonus * 0.15, 1.0)

    @classmethod
    def score(cls, response: str) -> float:
        """Compute overall clinical validity score in [0.0, 1.0].

        Weight normalization: when used in GRPO training with clinical_weight > 0,
        all three weights (empathy, crisis, clinical) are normalized to sum 1.0.
        The ClinicalValidityScorer's internal WEIGHTS always sum to 1.0 independently.

        Note: This measures keyword *density*, not true clinical quality.
        Very long responses may inflate match counts. Prefer scoring individual
        messages and averaging (see extract_therapist_voice.py score_conversations).
        """
        if not response or not isinstance(response, str):
            return 0.0
        total = 0.0
        for dimension, weight in cls.WEIGHTS.items():
            total += cls._score_dimension(response, dimension) * weight
        return total

    @classmethod
    def score_detail(cls, response: str) -> dict:
        """Compute per-dimension clinical validity scores.

        For a density-normalized alternative see score_density_detail().
        """
        if not response or not isinstance(response, str):
            return dict.fromkeys(cls.WEIGHTS, 0.0)
        return {
            dimension: cls._score_dimension(response, dimension)
            for dimension in cls.WEIGHTS
        }

    @classmethod
    def score_density_detail(cls, response: str) -> dict:
        """Per-dimension scores normalized by response length (density instead of raw count).

        Returns scores penalized by verbosity: a short, dense clinical response
        scores higher than a long rambling one with the same absolute matches.
        """
        if not response or not isinstance(response, str):
            return dict.fromkeys(cls.WEIGHTS, 0.0)
        results = {}
        token_count = max(1, len(response.split()))
        for dimension in cls.WEIGHTS:
            raw = cls._score_dimension(response, dimension)
            density_penalty = min(token_count / 250, 1.0)
            density_factor = 1.0 - density_penalty * 0.5
            results[dimension] = round(raw * density_factor, 4)
        return results

    @classmethod
    def score_density(cls, response: str) -> float:
        """Overall score normalized by response length (density instead of raw count)."""
        if not response or not isinstance(response, str):
            return 0.0
        total = 0.0
        detail = cls.score_density_detail(response)
        for dimension, weight in cls.WEIGHTS.items():
            total += detail[dimension] * weight
        return total
