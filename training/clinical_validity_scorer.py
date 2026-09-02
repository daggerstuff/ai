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

import argparse
import json
import re
import sys
from typing import ClassVar

# Threshold constants for three-tier routing (PIX-3773)
EXCLUDE_THRESHOLD = 0.4
ACCEPT_THRESHOLD = 0.6


# Non-English script detection ranges
_NON_ENGLISH_RE = re.compile(
    "["
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\u3040-\u309f"  # Hiragana
    "\u30a0-\u30ff"  # Katakana
    "\uac00-\ud7af"  # Hangul Syllables
    "\u0400-\u04ff"  # Cyrillic
    "\u0600-\u06ff"  # Arabic
    "\u0e00-\u0e7f"  # Thai
    "\u0f00-\u0fff"  # Tibetan
    "]"
)

# Lower-bound ratio of non-ASCII characters to flag as non-English
_NON_ENGLISH_RATIO = 0.30

# Minimum per-dimension score for a dimension to be flagged as present
DIMENSION_PRESENT_THRESHOLD = 0.3


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

    VERSION: ClassVar[str] = "4.0.0"

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
        "supportive_counseling": (
            r"\b(it (sounds|seems) like (you|you're)|sounds like you (are|have|need))\b",
            r"\b(I hear (you|that|what you)|I can (hear|see) (that|how))\b",
            r"\b(encourage you to|it('s| is) (important|helpful) (to|that))\b",
            r"\b(you are not alone|that's (completely|perfectly) (normal|understandable))\b",
            r"\b(reach out to|speak with a|talk to a|consult a)\b",
            r"\b(professional help|mental health professional|therapist (who|that) (specializes|can))\b",
            r"\b(coping (skill|strateg|mechanism|tool)|self(-| )care|support (system|network))\b",
            r"\b(what you are (experiencing|feeling|going through)|you're (experiencing|feeling|going through))\b",
            r"\b(first step|next step|small (step|changes)|take (step|action))\b",
            r"\b(it('s| is) (okay|normal|common) to (feel|have|experience))\b",
            r"\b(writing in a journal|journaling|keep a (journal|diary)|write down)\b",
            r"\b(you deserve|your feelings are (valid|real|important)|it's okay to feel)\b",
        ),
        "trauma_informed": (
            r"\btrauma(-| )?(informed|specific|focused|sensitive|aware)\b",
            r"\b(triggers?|triggering|grounding|grounding (technique|exercise))\b",
            r"\b(emotional safety|safe (space|environment|relationship)|felt safety)\b",
            r"\b(hypervigilance|hyperarousal|flashback|intrusive (thought|memory))\b",
            r"\b(window of tolerance|nervous system|fight or flight|freeze response)\b",
            r"\b(PTSD|complex trauma|developmental trauma|childhood trauma)\b",
            r"\b(trauma (recovery|healing|therapy|treatment)|re(-| )traumatiz)\b",
            r"\b(emotional (regulation|dysregulation)|self(-| )regulation|co(-| )regulation)\b",
            r"\b(safety (plan|contract|assessment)|stabilization|containment)\b",
        ),
        "somatic_therapy": (
            r"\bsomatic|somatic (experiencing|therapy|practice)\b",
            r"\b(body (awareness|sensation|scan|based)|(awareness|scan) of (the )?(body|physical))\b",
            r"\b(breathwork|breathing (exercise|technique|practice)|deep breath)\b",
            r"\b(mindful (movement|breathing)|yoga|gentle stretch)\b",
            r"\b(physical (sensation|feeling|experience)|embodiment|embodied)\b",
            r"\b(progressive muscle relaxation|relaxation (technique|exercise|response))\b",
            r"\b(vagus nerve|polyvagal|nervous system (regulation|health))\b",
            r"\b(tension (release|reduction)|relax(ation)? (muscle|body)|bodywork)\b",
            r"\b(where do you feel that|where in your body|notice (what|that) in your body)\b",
            r"\b(what happens in your body|what do you notice (in|about) your body)\b",
            r"\b(where does that (live|show up|sit) (in your body|))\b",
            r"\b(take a (breath|pause)|slow down|slow (it )?down)\b",
        ),
    }

    ALLIANCE_PATTERNS: ClassVar[dict[str, tuple[str, ...]]] = {
        "collaboration": (
            r"\b(work together|collaborat|partner with|we can|let's)\b",
            r"\b(what do you think|how does that (feel|sound|land)|your input)\b",
            r"\b(together|shared (understanding|goal)|agenda)\b",
            r"\b(we ('re| are) (in this|here) together|both of you)\b",
        ),
        "validation": (
            r"\b(that makes sense|understandably|it's understandable|of course)\b",
            r"\b(valid|legitimate|reasonable|makes (perfect )?sense)\b",
            r"\b(I hear you|I can see|that sounds|that must be)\b",
            r"\b(you are not alone|it('s| is) (okay|normal|understandable) (to|that))\b",
            r"\b(courage (to|in)|brave|strength (in|to)|resilien)\b",
            r"\b(thank you for (sharing|trusting|reaching|opening))\b",
            r"\b(it takes (a lot of |so much )?courage|I('m| am) (glad|grateful) (you|that))\b",
            r"\b(no wonder (you|that)|anyone would (feel|be) )\b",
            r"\b(that\'s (important|significant|meaningful)|it matters (that|because))\b",
        ),
        "empowerment": (
            r"\b(you (have|are) (the )?(strength|resilience|courage|capable))\b",
            r"\b(you can|you get to decide|your choice|your (own )?pace)\b",
            r"\b(agency|autonomy|self(-| )efficacy|build(ing)? on)\b",
            r"\b(you (are|were) able to|you (have|found) the (strength|resources))\b",
            r"\b(you (deserve|matter)|worth (it|your time|the effort))\b",
            r"\b(what do you need (right now|)|what would help (right now|))\b",
            r"\b(what would (help|support) (you|)|what might (help|be useful))\b",
            r"\b(take your time|no rush|there\'s no hurry)\b",
        ),
        "exploration": (
            r"\b(can you tell me more|tell me more|say more about)\b",
            r"\b(what (is|was) that like (for you|)|how does that feel)\b",
            r"\b(can we (explore|look at|try) (that|this|it))\b",
            r"\b(what\'s (that|it) like (for you|)|how (is|does) that (feel|show up))\b",
        ),
    }

    STRUCTURE_PATTERNS: ClassVar[dict[str, tuple[str, ...]]] = {
        "assessment": (
            r"\b(assessment|intake|evaluation|history|presenting (problem|concern))\b",
            r"\b(symptom|duration|onset|frequency|severity|intensity)\b",
            r"\b(previous (treatment|therapy|hospitalization)|medication)\b",
            r"\b(how long (have|does)|when did (you|it)|how often (do|does))\b",
            r"\b(it sounds like you (are|have|may be|might be) (experiencing|dealing|struggling))\b",
        ),
        "intervention": (
            r"\b(intervention|technique|strategy|approach|skill|tool)\b",
            r"\b(practice|exercise|worksheet|try|attempt)\b",
            r"\b(explore|examine|look at|delve|unpack)\b",
            r"\b(suggest (that|you|trying|considering)|recommend (that|you|trying|seeking))\b",
            r"\b(one (thing|approach|strategy) (that|you|is)|a (good|helpful) (way|idea) (to|is))\b",
            r"\b(work on|focus on|address (the|this|these|your))\b",
            r"\b(stay with that|sit with that|be with that|notice that)\b",
            r"\b(what happens (when|next)|what comes up (for you|)|what comes next)\b",
        ),
        "planning": (
            r"\b(goal|objective|outcome|treatment plan|therapeutic goal)\b",
            r"\b(between sessions|homework|practice|follow(-| )up|next step)\b",
            r"\b(progress|track|monitor|review|adjust)\b",
            r"\b(develop (a|an|strateg|plan|skill)|build(ing)? (on|upon|skills|coping))\b",
        ),
        "closure": (
            r"\b(summarize|wrap up|close|recap|reflect back)\b",
            r"\b(key (takeaway|point)|main (theme|idea)|what stood out)\b",
            r"\b(next session|check in|continue (working|exploring))\b",
            r"\b(I (hope|wish) (this|that|you)|wishing you|best of luck|hope this (helps|is))\b",
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
            r"\b(therapy (has been )?(shown|found|demonstrated)|treatment (option|approach))\b",
        ),
        "clinical_reasoning": (
            r"\b(conceptualiz|cased? formulation|clinical (judgment|reasoning))\b",
            r"\b(differential|diagnos|presentation|etiology|prognosis)\b",
            r"\b(indication|contraindication|referral|consultation)\b",
            r"\b(symptom (of|suggest|indicat)|consistent with|present with)\b",
            r"\b(treatable|responsive to|management (of|strateg)|intervention)\b",
        ),
        "therapeutic_framing": (
            r"\b(one (way|approach) is|a (helpful|useful) way|some people find)\b",
            r"\b(in (my|our) experience|from a clinical perspective|professionally)\b",
            r"\b(evidence suggests|research (shows|indicates)|studies (show|suggest))\b",
            r"\b(consider (trying|exploring|working with)|worth (considering|exploring|trying))\b",
            r"\b(goal (is|would be) to|aim (is|would be) to|approach (is|involves))\b",
        ),
    }

    DSM5_PATTERNS: ClassVar[dict[str, tuple[str, ...]]] = {
        "mood_disorders": (
            r"\b(major depressive|persistent depressive|dysthymia|cyclothymia)\b",
            r"\b(depressed mood|anhedonia|loss of interest|hopelessness|worthlessness)\b",
            r"\b(sleep (disturbance|insomnia|hypersomnia)|appetite (change|loss|increase))\b",
            r"\b(fatigue|loss of energy|psychomotor (agitation|retardation))\b",
            r"\b(concentration (difficulty|problem)|indecisiveness)\b",
            r"\b(death (thoughts|ideation)|suicidal (ideation|thoughts|intent))\b",
            r"\b(bipolar|manic (episode|symptoms)|hypomanic|mood (swing|episode))\b",
            r"\b(elevated (mood|energy)|grandiose|pressured speech|decreased need for sleep)\b",
            r"\b(flight of ideas|distractibility|increased (goal|activity|pleasure))\b",
        ),
        "anxiety_disorders": (
            r"\b(generalized anxiety|GAD|excessive worry|anxiety disorder)\b",
            r"\b(panic (attack|disorder)|agoraphobia|social anxiety|specific phobia)\b",
            r"\b(restlessness|fatigue|difficulty concentrating|irritability)\b",
            r"\b(muscle tension|sleep (difficulty|disturbance)|avoidance behavior)\b",
            r"\b(racing heart|shortness of breath|chest pain|dizziness|trembling)\b",
            r"\b(fear of (losing control|dying|going crazy)|numbness|tingling)\b",
            r"\b(separation anxiety|selective mutism|anxiety (about|over|regarding))\b",
            r"\b(hypervigilance|exaggerated startle|always on (edge|guard))\b",
        ),
        "trauma_disorders": (
            r"\b(PTSD|post-traumatic|c(omplex )?trauma|acute stress disorder)\b",
            r"\b(traumatic (event|experience)|exposure to (death|injury|violence))\b",
            r"\b(intrusive (memory|thought|image)|flashback|nightmare|distressing dream)\b",
            r"\b(avoidance of (reminder|trigger|memory)|emotional numbing)\b",
            r"\b(negative (belief|emotion|mood)|distorted (blame|cognition))\b",
            r"\b(detachment|estrangement|loss of interest|dissociative (amnesia|symptom))\b",
            r"\b(irritable|angry (outburst|behavior)|reckless (behavior|self-destructive))\b",
            r"\b(sleep (disturbance|problem)|concentration (problem|difficulty)|hypervigilance)\b",
            r"\b(dissociative identity|depersonalization|derealization|identity (confusion|disturbance))\b",
            r"\b(adjustment disorder|stress,? (not|other)|bereavement|prolonged grief)\b",
        ),
        "psychotic_disorders": (
            r"\b(schizophrenia|schizoaffective|delusional disorder|brief psychotic)\b",
            r"\b(hallucination|delusion|disorganized (thinking|speech|behavior))\b",
            r"\b(paranoid (ideation|delusion)|persecutory|reference|grandiose)\b",
            r"\b(auditory (hallucination|voice)|hearing (voice|thing)|visual (hallucination|disturbance))\b",
            r"\b(voices? (tell|speak|comment|whisper)|people are (plotting|against|following))\b",
            r"\b(negative symptom|flat affect|alogia|avolition|asociality)\b",
            r"\b(catatonic|catalepsy|waxy flexibility|mutism|stupor)\b",
            r"\b(thought (broadcast|insertion|withdrawal|blocking)|ideas of reference)\b",
            r"\b(schizotypal|schizoid|paranoid personality|psychotic (symptom|episode))\b",
        ),
        "ocd_related": (
            r"\b(obsessive compulsive|OCD|body dysmorphic|hoarding|trichotillomania)\b",
            r"\b(intrusive (thought|image|urge)|obsession|compulsion|ritual)\b",
            r"\b(checking|washing|ordering|counting|repeating|hoarding)\b",
            r"\b(contamination (fear|anxiety)|magical thinking|just(-| )right)\b",
            r"\b(excoriation|skin picking|body focused repetitive|dermatillomania)\b",
            r"\b(cleanliness|symmetry|exactness|perfectionism (intrusive|related))\b",
        ),
        "eating_disorders": (
            r"\b(anorexia nervosa|bulimia nervosa|binge eating|OSFED|ARFID)\b",
            r"\b(restrict (intake|eating)|purge|binge|over-exercise|fasting)\b",
            r"\b(body (image|dissatisfaction|weight (preoccupation|concern)))\b",
            r"\b(fear of (weight gain|getting fat)|drive for thinness|weight (loss|suppression))\b",
            r"\b(self-(induced )?vomiting|laxative (abuse|use)|diuretic (use|abuse))\b",
            r"\b(calorie (counting|restriction)|food (avoidance|refusal|ritual))\b",
            r"\b(eating disorder|disordered eating|binge-purge|weight restoration)\b",
        ),
        "neurodevelopmental": (
            r"\b(ADHD|attention deficit|hyperactivity|impulsivity|inattention)\b",
            r"\b(autism (spectrum|disorder)|ASD|asperger|neurodivergen)\b",
            r"\b(specific learning|dyslexia|dyscalculia|dysgraphia|developmental)\b",
            r"\b(executive function|organization|time management|task initiation)\b",
            r"\b(sensory (processing|sensitivity|overload)|stim(ming|ulation))\b",
            r"\b(masking|social (cue|skill|pragmatic)|communication (challenge|difficulty))\b",
            r"\b(hyperfocus|distractible|restless|fidget|cannot (sit|stay) still)\b",
            r"\b(tic disorder|tourette|motor (tic|vocalization)|brief (vocal|motor))\b",
        ),
        "personality_disorders": (
            r"\b(borderline personality|BPD|emotionally (unstable|dysregulated))\b",
            r"\b(narcissistic|antisocial|avoidant|dependent|histrionic|schizoid)\b",
            r"\b(identity (disturbance|diffusion|instability)|chronic emptiness)\b",
            r"\b(abandonment (fear|anxiety)|idealization|devaluation|splitting)\b",
            r"\b(affective (instability|dysregulation)|intense (relationship|emotion))\b",
            r"\b(self(-| )harm|suicidal (behavior|gesture|threat)|impulsive (behavior|action))\b",
            r"\b(grandios|superiority|entitlement|lack of empathy|exploitative)\b",
            r"\b(paranoid (ideation|personality)|suspicious|mistrust|persecutory)\b",
            r"\b(obsessive(-| )compulsive (personality)|perfectionism (interferes|interfering))\b",
        ),
        "sleep_wake_disorders": (
            r"\b(insomnia|hypersomnia|narcolepsy|sleep (apnea|disorder))\b",
            r"\b(difficulty (falling|staying) asleep|early morning awakening)\b",
            r"\b(circadian (rhythm|disorder)|delayed sleep|shift work)\b",
            r"\b(restless leg|periodic limb|night terror|sleepwalking|somnambuli)\b",
            r"\b(parasomnia|nightmare (disorder)|REM (behavior|sleep))\b",
            r"\b(sleep (maintenance|onset)|non-restorative sleep|sleep (hygiene|quality))\b",
        ),
        "substance_related": (
            r"\b(alcohol (use|abuse|depend)|substance (use|abuse|depend))\b",
            r"\b(opioid|stimulant|cannabis|sedative|hallucinogen|inhalant)\b",
            r"\b(tolerance|withdrawal|craving|addiction|substance use disorder)\b",
            r"\b(intoxication|overdose|detox|rehab|relapse|abstinence)\b",
            r"\b(cocaine|heroin|methamphetamine|fentanyl|marijuana|THC|CBD)\b",
            r"\b(prescription (drug|misuse)|polysubstance|poly(-| )drug)\b",
            r"\b(reduced (use|consumption)|quit (drinking|using|smoking)|harm reduction)\b",
        ),
    }

    WEIGHTS: ClassVar[dict[str, float]] = {
        "technique": 0.25,
        "alliance": 0.20,
        "structure": 0.10,
        "cultural": 0.05,
        "ebp": 0.10,
        "dsm5": 0.30,
    }

    DIMENSION_PATTERNS: ClassVar[dict[str, dict[str, tuple[str, ...]]]] = {
        "technique": THERAPY_MODALITIES,
        "alliance": ALLIANCE_PATTERNS,
        "structure": STRUCTURE_PATTERNS,
        "cultural": CULTURAL_PATTERNS,
        "ebp": EBP_PATTERNS,
        "dsm5": DSM5_PATTERNS,
    }

    # Three-tier routing thresholds (PIX-3773)
    EXCLUDE_THRESHOLD: ClassVar[float] = 0.4
    ACCEPT_THRESHOLD: ClassVar[float] = 0.6

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
    def _score_dimension_density(cls, text: str, dimension: str) -> tuple:
        """Compute density-specific scoring components.

        Returns (matches, density_score) where density_score is properly
        normalized to be <= the raw score and penalizes verbosity.
        """
        if not text or not isinstance(text, str):
            return (0, 0.0)
        pattern = cls._get_dimension_pattern(dimension)
        matches = list(pattern.finditer(text))
        sub_dimensions = len(cls.DIMENSION_PATTERNS[dimension])
        if not matches:
            return (0, 0.0)
        match_texts = {m.group(0).lower() for m in matches}
        diversity_bonus = min(1.0, len(match_texts) / 10.0)
        token_count = max(1, len(text.split()))

        # Compute raw score WITHOUT capping at 1.0 (to allow proper density calc)
        # This is the "uncapped match rate" - how many match groups per sub-dimension
        uncapped_match_rate = len(matches) / sub_dimensions

        # Density penalty: longer text gets penalized more
        # density_factor ranges from 1.0 (very short) to 0.5 (250+ tokens)
        density_penalty = min(token_count / 250, 1.0)
        density_factor = 1.0 - density_penalty * 0.5

        # Density score: uncapped match rate normalized by token count and density factor
        # This ensures:
        # 1. density <= raw for all inputs (because we divide by token_count which is >= 1)
        # 2. Verbose text gets penalized (higher token_count reduces score)
        density_score = (uncapped_match_rate / token_count) * density_factor + diversity_bonus * 0.15 * density_factor

        return (len(matches), min(density_score, 1.0))

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
        return {dimension: cls._score_dimension(response, dimension) for dimension in cls.WEIGHTS}

    @classmethod
    def score_density_detail(cls, response: str) -> dict:
        """Per-dimension scores normalized by response length (density instead of raw count).

        Returns scores penalized by verbosity: a short, dense clinical response
        scores higher than a long rambling one with the same absolute matches.

        The density score is computed as:
            density = (matches / sub_dims) / max(1, tokens) * scale_factor

        This is a true "clinical terms per token" measure, which properly penalizes
        verbose text by reducing score as tokens increase.
        """
        if not response or not isinstance(response, str):
            return dict.fromkeys(cls.WEIGHTS, 0.0)
        results = {}
        for dimension in cls.WEIGHTS:
            _, density_score = cls._score_dimension_density(response, dimension)
            results[dimension] = round(density_score, 4)
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

    @classmethod
    def _detect_non_english(cls, text: str) -> bool:
        """Check if text contains a significant proportion of non-English scripts."""
        if not text or not isinstance(text, str):
            return False
        non_english_chars = len(_NON_ENGLISH_RE.findall(text))
        total_chars = max(1, len(text.strip()))
        return (non_english_chars / total_chars) > _NON_ENGLISH_RATIO

    @classmethod
    def _determine_category(cls, detail_scores: dict[str, float]) -> str:
        """Return the dimension name with the highest score as the category."""
        if not detail_scores:
            return "unknown"
        best = max(detail_scores, key=detail_scores.get)  # type: ignore[arg-type]
        return best if detail_scores[best] > 0.0 else "unknown"

    @classmethod
    def _build_flags(cls, response: str, detail_scores: dict[str, float], overall: float) -> list[str]:
        """Build a list of diagnostic flags for the score result."""
        flags: list[str] = []
        if not response or not isinstance(response, str):
            flags.append("empty_input")
            return flags
        if cls._detect_non_english(response):
            flags.append("non_english_content")
        flags.extend(
            f"{dimension}_present" for dimension, score in detail_scores.items() if score >= DIMENSION_PRESENT_THRESHOLD
        )
        if overall < cls.EXCLUDE_THRESHOLD:
            flags.append("below_exclude_threshold")
        elif overall < cls.ACCEPT_THRESHOLD:
            flags.append("annotation_needed")
        return flags

    @classmethod
    def score_with_flags(cls, response: str) -> dict:
        """Compute score with structured output including flags and dominant category.

        Returns:
            dict with keys: validity_score, flags (list[str]), category (str), detail (dict)
        """
        if not response or not isinstance(response, str):
            return {
                "validity_score": 0.0,
                "flags": ["empty_input"],
                "category": "unknown",
                "detail": dict.fromkeys(cls.WEIGHTS, 0.0),
            }
        overall = cls.score(response)
        detail = cls.score_detail(response)
        return {
            "validity_score": overall,
            "flags": cls._build_flags(response, detail, overall),
            "category": cls._determine_category(detail),
            "detail": detail,
        }

    @classmethod
    def batch_score(cls, responses: list[str]) -> list[float]:
        """Compute clinical validity scores for multiple responses.

        Args:
            responses: List of text responses to score

        Returns:
            List of scores in [0.0, 1.0], preserving the order of input responses
        """
        return [cls.score(response) for response in responses]

    @classmethod
    def modality_coverage(cls, text: str) -> dict[str, dict]:
        """Count matches per therapy modality.

        Args:
            text: Text to analyze

        Returns:
            Dictionary mapping modality names (cbt, dbt, mi, etc.) to dicts
            containing 'count' (match count) and 'patterns' (list of matched patterns)
        """
        if not text or not isinstance(text, str):
            return {modality: {"count": 0, "patterns": []} for modality in cls.THERAPY_MODALITIES}

        results = {}
        for modality, patterns in cls.THERAPY_MODALITIES.items():
            matches: list[str] = []
            for pattern in patterns:
                matches.extend(m.group(0) for m in re.finditer(pattern, text, re.IGNORECASE))
            results[modality] = {"count": len(matches), "patterns": matches}
        return results

    @classmethod
    def classify_score(cls, score: float) -> str:
        """Three-tier routing for clinical validity scores.

        Returns one of:
            "excluded" — score below EXCLUDE_THRESHOLD, sample should be excluded
            "annotation_needed" — score between thresholds, sample needs expert review
            "accepted" — score at or above ACCEPT_THRESHOLD, sample can be used
        """
        if score < cls.EXCLUDE_THRESHOLD:
            return "excluded"
        return "annotation_needed" if score < cls.ACCEPT_THRESHOLD else "accepted"


def main() -> None:
    """CLI entry point for the clinical validity scorer.

    Usage:
        uv run python -m training.clinical_validity_scorer --text "Your text here"
        echo "Your text here" | uv run python -m training.clinical_validity_scorer
    """
    parser = argparse.ArgumentParser(description="Score clinical validity of therapeutic training text")
    parser.add_argument("--text", type=str, default=None, help="Text to score")
    parser.add_argument("--detail", action="store_true", help="Include per-dimension detail in output")
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read().strip()

    if not text:
        result = ClinicalValidityScorer.score_with_flags("")
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        sys.exit(0)

    if args.detail:
        result = ClinicalValidityScorer.score_with_flags(text)
    else:
        score = ClinicalValidityScorer.score(text)
        result = {
            "validity_score": score,
            "classification": ClinicalValidityScorer.classify_score(score),
        }

    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
