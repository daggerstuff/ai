"""PATIENT-Ψ clinical profile definitions and registry.

Provides 20 clinically-informed patient profiles based on DSM-5
diagnoses, each with a unique CCD configuration mapped to the
8-component cognitive conceptualization model (arXiv 2405.19660 §3).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ai.pkg_mera.platform.patient_psi.styles import ConversationalStyle


class ClinicalProfile(BaseModel):
    """A complete clinical patient profile for PATIENT-Ψ simulation."""

    name: str
    display_name: str
    description: str
    diagnoses: list[str]
    typical_symptoms: list[str]
    default_style: ConversationalStyle
    ccd_config: dict[str, list[dict]]
    linguistic_features: dict[str, float]
    severity_range: tuple[float, float]
    common_triggers: list[str]
    treatment_history: str
    model_config = ConfigDict(arbitrary_types_allowed=True)


# ──────────────────────────────────────────────
# Style mapping shorthand
# ──────────────────────────────────────────────
_N = ConversationalStyle.NEUTRAL
_F = ConversationalStyle.FRIENDLY
_H = ConversationalStyle.HOSTILE
_A = ConversationalStyle.ANXIOUS
_MC = ConversationalStyle.MELANCHOLIC
_MN = ConversationalStyle.MANIC


# ──────────────────────────────────────────────
# Profile data
# ──────────────────────────────────────────────

_PROFILES: dict[str, ClinicalProfile] = {}


def _register(p: ClinicalProfile) -> ClinicalProfile:
    _PROFILES[p.name] = p
    return p


# ── 1. Major Depressive Disorder ──
_register(
    ClinicalProfile(
        name="major_depressive_disorder",
        display_name="Major Depressive Disorder",
        description="Persistent low mood, anhedonia, worthlessness beliefs, and somatic disturbance.",
        diagnoses=["F32.2", "F32.3", "F33.2"],
        typical_symptoms=[
            "persistent sad mood",
            "loss of interest or pleasure",
            "significant weight change",
            "psychomotor retardation",
            "fatigue or loss of energy",
            "feelings of worthlessness",
        ],
        default_style=_MC,
        ccd_config={
            "core_beliefs": [
                {"content": "I am fundamentally worthless", "domain": "self", "conviction": 0.92},
                {"content": "The world is empty and meaningless", "domain": "world", "conviction": 0.88},
                {"content": "There is no hope for improvement", "domain": "future", "conviction": 0.85},
            ],
            "intermediate_beliefs": [
                {"content": "If I am not perfect, I am a failure", "rule_type": "rule", "conviction": 0.85},
                {"content": "It is better to expect nothing", "rule_type": "attitude", "conviction": 0.8},
                {
                    "content": "Others will reject me if they see the real me",
                    "rule_type": "assumption",
                    "conviction": 0.75,
                },
            ],
            "coping_strategies": [
                {"content": "Social withdrawal", "strategy_type": "avoidance", "effectiveness": 0.2},
                {"content": "Rumination about past failures", "strategy_type": "compensation", "effectiveness": 0.1},
                {"content": "Oversleeping to escape distress", "strategy_type": "avoidance", "effectiveness": 0.3},
            ],
            "compensatory_strategies": [
                {
                    "content": "Extreme perfectionism in work output",
                    "behavior": "overworking to avoid criticism",
                    "overcompensation_for": "worthlessness feelings",
                },
                {
                    "content": "Excessive reassurance seeking from therapist",
                    "behavior": "seeking validation before decisions",
                    "overcompensation_for": "decision paralysis",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Receiving constructive feedback",
                    "interpretation": "They think I am incompetent",
                    "distortion_type": "mind-reading",
                },
                {
                    "situation": "Making a minor mistake",
                    "interpretation": "I always ruin everything",
                    "distortion_type": "overgeneralization",
                },
                {
                    "situation": "Someone cancels plans",
                    "interpretation": "They do not want to be around me",
                    "distortion_type": "personalization",
                },
            ],
            "emotional_responses": [
                {"emotion": "sadness", "intensity": 0.88, "valence": "negative"},
                {"emotion": "guilt", "intensity": 0.75, "valence": "negative"},
                {"emotion": "emptiness", "intensity": 0.8, "valence": "negative"},
                {"emotion": "irritability", "intensity": 0.5, "valence": "negative"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Stays in bed for extended periods",
                    "triggered_by": "waking up to a new day",
                    "consequence": "Avoids daily responsibilities",
                },
                {
                    "behavior": "Cancels social engagements",
                    "triggered_by": "anticipating social interaction",
                    "consequence": "Increasing isolation",
                },
                {
                    "behavior": "Procrastinates important tasks",
                    "triggered_by": "fear of failure",
                    "consequence": "Increased guilt and self-criticism",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.15, "world_views": 0.2, "future_views": 0.1},
            ],
        },
        linguistic_features={
            "hedging": 0.4,
            "negation_density": 0.75,
            "first_person_singular": 0.85,
            "cause_words": 0.6,
            "absolutist_words": 0.7,
        },
        severity_range=(0.4, 0.9),
        common_triggers=[
            "interpersonal rejection",
            "work or academic failure",
            "anniversary of loss",
            "seasonal change (winter)",
        ],
        treatment_history="Partial response to SSRIs; MECT considered for severe episodes.",
    )
)

# ── 2. Generalized Anxiety Disorder ──
_register(
    ClinicalProfile(
        name="generalized_anxiety",
        display_name="Generalized Anxiety Disorder",
        description="Chronic excessive worry across multiple domains with physical tension and hypervigilance.",
        diagnoses=["F41.1"],
        typical_symptoms=[
            "excessive worry about multiple topics",
            "muscle tension",
            "sleep disturbance",
            "fatigue",
            "restlessness",
            "difficulty concentrating",
        ],
        default_style=_A,
        ccd_config={
            "core_beliefs": [
                {"content": "The world is a dangerous place", "domain": "world", "conviction": 0.85},
                {"content": "I am not capable of handling threats", "domain": "self", "conviction": 0.8},
                {"content": "Something bad is always about to happen", "domain": "future", "conviction": 0.9},
            ],
            "intermediate_beliefs": [
                {"content": "I must be prepared for every possible disaster", "rule_type": "rule", "conviction": 0.88},
                {"content": "Worrying keeps me safe", "rule_type": "attitude", "conviction": 0.82},
                {
                    "content": "If I relax, something terrible will happen",
                    "rule_type": "assumption",
                    "conviction": 0.78,
                },
            ],
            "coping_strategies": [
                {
                    "content": "Excessive information seeking",
                    "strategy_type": "overcompensation",
                    "effectiveness": 0.15,
                },
                {"content": "Avoiding news or triggering content", "strategy_type": "avoidance", "effectiveness": 0.25},
                {"content": "Verbal rumination with loved ones", "strategy_type": "compensation", "effectiveness": 0.3},
            ],
            "compensatory_strategies": [
                {
                    "content": "Overpreparation for routine activities",
                    "behavior": "packing hours before leaving",
                    "overcompensation_for": "fear of being unprepared",
                },
                {
                    "content": "Multiple contingency planning",
                    "behavior": "imagining worst-case scenarios for every decision",
                    "overcompensation_for": "intolerance of uncertainty",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Boss sends an email requesting a meeting",
                    "interpretation": "I am about to be fired",
                    "distortion_type": "catastrophizing",
                },
                {
                    "situation": "Feeling minor chest discomfort",
                    "interpretation": "I am having a heart attack",
                    "distortion_type": "catastrophizing",
                },
                {
                    "situation": "Partner is late coming home",
                    "interpretation": "They were in a serious accident",
                    "distortion_type": "catastrophizing",
                },
            ],
            "emotional_responses": [
                {"emotion": "anxiety", "intensity": 0.85, "valence": "negative"},
                {"emotion": "apprehension", "intensity": 0.78, "valence": "negative"},
                {"emotion": "tension", "intensity": 0.82, "valence": "negative"},
                {"emotion": "irritability", "intensity": 0.55, "valence": "negative"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Checks locks repeatedly",
                    "triggered_by": "leaving the house",
                    "consequence": "Temporary relief followed by doubt",
                },
                {
                    "behavior": "Seeks reassurance from family",
                    "triggered_by": "uncertainty about a decision",
                    "consequence": "Short-term anxiety reduction",
                },
                {
                    "behavior": "Scans internet for symptom causes",
                    "triggered_by": "minor physical sensation",
                    "consequence": "Increased health anxiety",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.4, "world_views": 0.25, "future_views": 0.3},
            ],
        },
        linguistic_features={
            "hedging": 0.8,
            "negation_density": 0.4,
            "first_person_singular": 0.6,
            "cause_words": 0.7,
            "absolutist_words": 0.5,
        },
        severity_range=(0.3, 0.8),
        common_triggers=[
            "uncertainty or ambiguity",
            "health concerns",
            "work deadlines",
            "family responsibilities",
        ],
        treatment_history="Responds partially to SNRIs and CBT; ongoing metacognitive therapy.",
    )
)

# ── 3. Social Anxiety Disorder ──
_register(
    ClinicalProfile(
        name="social_anxiety",
        display_name="Social Anxiety Disorder",
        description="Intense fear of negative evaluation in social or performance situations.",
        diagnoses=["F40.10", "F40.11"],
        typical_symptoms=[
            "fear of social situations",
            "fear of being judged negatively",
            "avoidance of social events",
            "physical symptoms in social settings",
            "intense self-consciousness",
        ],
        default_style=_A,
        ccd_config={
            "core_beliefs": [
                {"content": "Others are harshly judgmental", "domain": "others", "conviction": 0.92},
                {"content": "I am socially inept and unlikeable", "domain": "self", "conviction": 0.88},
                {"content": "Social failure will destroy my reputation", "domain": "future", "conviction": 0.82},
            ],
            "intermediate_beliefs": [
                {"content": "I must never appear anxious or awkward", "rule_type": "rule", "conviction": 0.9},
                {"content": "Showing vulnerability is dangerous", "rule_type": "attitude", "conviction": 0.85},
                {
                    "content": "If I say something stupid, everyone will remember it forever",
                    "rule_type": "assumption",
                    "conviction": 0.88,
                },
            ],
            "coping_strategies": [
                {"content": "Avoiding parties and gatherings", "strategy_type": "avoidance", "effectiveness": 0.4},
                {"content": "Safety behaviors in conversations", "strategy_type": "compensation", "effectiveness": 0.2},
                {"content": "Substance use before social events", "strategy_type": "avoidance", "effectiveness": 0.35},
            ],
            "compensatory_strategies": [
                {
                    "content": "Over-preparing conversation topics",
                    "behavior": "rehearsing scripts before social events",
                    "overcompensation_for": "fear of awkward silence",
                },
                {
                    "content": "Avoiding eye contact",
                    "behavior": "looking away to reduce perceived scrutiny",
                    "overcompensation_for": "fear of being judged",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Someone yawns during conversation",
                    "interpretation": "I am boring them unbearably",
                    "distortion_type": "mind-reading",
                },
                {
                    "situation": "Group of people laughs nearby",
                    "interpretation": "They are laughing at me",
                    "distortion_type": "personalization",
                },
                {
                    "situation": "Pausing to think before speaking",
                    "interpretation": "They think I am stupid",
                    "distortion_type": "fortune-telling",
                },
            ],
            "emotional_responses": [
                {"emotion": "fear", "intensity": 0.88, "valence": "negative"},
                {"emotion": "shame", "intensity": 0.78, "valence": "negative"},
                {"emotion": "embarrassment", "intensity": 0.85, "valence": "negative"},
                {"emotion": "relief", "intensity": 0.7, "valence": "positive"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Leaves social events early",
                    "triggered_by": "rising anxiety during interaction",
                    "consequence": "Reinforces avoidance cycle",
                },
                {
                    "behavior": "Speaks very quietly in groups",
                    "triggered_by": "fear of being judged",
                    "consequence": "Others cannot hear, leading to repeated questions",
                },
                {
                    "behavior": "Declines invitations with excuses",
                    "triggered_by": "receiving a social invitation",
                    "consequence": "Growing social isolation",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.3, "world_views": 0.35, "future_views": 0.25},
            ],
        },
        linguistic_features={
            "hedging": 0.85,
            "negation_density": 0.35,
            "first_person_singular": 0.75,
            "cause_words": 0.4,
            "absolutist_words": 0.45,
        },
        severity_range=(0.3, 0.8),
        common_triggers=[
            "public speaking",
            "eating in front of others",
            "being the center of attention",
            "meeting new people",
        ],
        treatment_history="Good response to CBT and group therapy; SSRI augmentation helpful.",
    )
)

# ── 4. Panic Disorder ──
_register(
    ClinicalProfile(
        name="panic_disorder",
        display_name="Panic Disorder",
        description="Recurrent unexpected panic attacks with fear of future attacks and bodily sensations.",
        diagnoses=["F41.0"],
        typical_symptoms=[
            "recurrent panic attacks",
            "fear of future attacks",
            "avoidance of physical exertion",
            "hypervigilance to bodily sensations",
            "sense of impending doom",
        ],
        default_style=_A,
        ccd_config={
            "core_beliefs": [
                {"content": "My bodily sensations are dangerous", "domain": "self", "conviction": 0.92},
                {"content": "I am going to lose control completely", "domain": "self", "conviction": 0.88},
                {"content": "Intense fear means something is terribly wrong", "domain": "world", "conviction": 0.8},
            ],
            "intermediate_beliefs": [
                {"content": "I must monitor my body for any sign of danger", "rule_type": "rule", "conviction": 0.9},
                {
                    "content": "Physical symptoms are catastrophic until proven otherwise",
                    "rule_type": "attitude",
                    "conviction": 0.85,
                },
                {
                    "content": "If I have a panic attack, I might die or go crazy",
                    "rule_type": "assumption",
                    "conviction": 0.88,
                },
            ],
            "coping_strategies": [
                {
                    "content": "Avoiding exercise to prevent symptoms",
                    "strategy_type": "avoidance",
                    "effectiveness": 0.2,
                },
                {
                    "content": "Carrying calming medications at all times",
                    "strategy_type": "compensation",
                    "effectiveness": 0.45,
                },
                {"content": "Controlled breathing techniques", "strategy_type": "compensation", "effectiveness": 0.6},
            ],
            "compensatory_strategies": [
                {
                    "content": "Never leaving home without an escape plan",
                    "behavior": "mapping exits and hospitals everywhere",
                    "overcompensation_for": "fear of being trapped during a panic attack",
                },
                {
                    "content": "Frequent vital sign checking",
                    "behavior": "taking pulse and blood pressure multiple times daily",
                    "overcompensation_for": "fear of catastrophic medical event",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Heart races after climbing stairs",
                    "interpretation": "I might be having a heart attack",
                    "distortion_type": "catastrophizing",
                },
                {
                    "situation": "Feeling slightly dizzy",
                    "interpretation": "I am about to faint",
                    "distortion_type": "catastrophizing",
                },
                {
                    "situation": "Shortness of breath during stress",
                    "interpretation": "I am suffocating",
                    "distortion_type": "catastrophizing",
                },
            ],
            "emotional_responses": [
                {"emotion": "terror", "intensity": 0.95, "valence": "negative"},
                {"emotion": "dread", "intensity": 0.85, "valence": "negative"},
                {"emotion": "helplessness", "intensity": 0.78, "valence": "negative"},
                {"emotion": "exhaustion", "intensity": 0.7, "valence": "negative"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Sits down immediately when heart rate increases",
                    "triggered_by": "noticing heartbeat",
                    "consequence": "Reinforces danger association with normal physiology",
                },
                {
                    "behavior": "Leaves public places abruptly",
                    "triggered_by": "sudden wave of panic",
                    "consequence": "Avoidance generalizes to more locations",
                },
                {
                    "behavior": "Calls emergency contacts when panicking",
                    "triggered_by": "sensation of losing control",
                    "consequence": "Immediate reassurance but increased dependency",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.35, "world_views": 0.3, "future_views": 0.25},
            ],
        },
        linguistic_features={
            "hedging": 0.6,
            "negation_density": 0.25,
            "first_person_singular": 0.8,
            "cause_words": 0.75,
            "absolutist_words": 0.65,
        },
        severity_range=(0.4, 0.9),
        common_triggers=[
            "physical exertion",
            "caffeine",
            "enclosed spaces",
            "interpersonal conflict",
        ],
        treatment_history="Good response to CBT with interoceptive exposure; SSRI maintenance.",
    )
)

# ── 5. PTSD ──
_register(
    ClinicalProfile(
        name="ptsd",
        display_name="Post-Traumatic Stress Disorder",
        description="Re-experiencing, avoidance, and hyperarousal following a traumatic event.",
        diagnoses=["F43.10"],
        typical_symptoms=[
            "intrusive memories or flashbacks",
            "avoidance of trauma reminders",
            "hypervigilance",
            "exaggerated startle response",
            "emotional numbing",
        ],
        default_style=_H,
        ccd_config={
            "core_beliefs": [
                {"content": "The world is fundamentally unsafe", "domain": "world", "conviction": 0.95},
                {"content": "I am permanently damaged by what happened", "domain": "self", "conviction": 0.9},
                {"content": "No one can understand or protect me", "domain": "others", "conviction": 0.88},
            ],
            "intermediate_beliefs": [
                {"content": "I must always be on guard for danger", "rule_type": "rule", "conviction": 0.92},
                {"content": "Relaxation is dangerous and vulnerable", "rule_type": "attitude", "conviction": 0.85},
                {
                    "content": "If I let my guard down, I will be hurt again",
                    "rule_type": "assumption",
                    "conviction": 0.9,
                },
            ],
            "coping_strategies": [
                {"content": "Avoiding trauma-related stimuli", "strategy_type": "avoidance", "effectiveness": 0.35},
                {
                    "content": "Hypervigilant scanning of environment",
                    "strategy_type": "overcompensation",
                    "effectiveness": 0.2,
                },
                {"content": "Substance use to suppress memories", "strategy_type": "avoidance", "effectiveness": 0.3},
            ],
            "compensatory_strategies": [
                {
                    "content": "Sitting with back to wall in public",
                    "behavior": "positioning to see all exits",
                    "overcompensation_for": "fear of being ambushed",
                },
                {
                    "content": "Avoiding intimacy and trust",
                    "behavior": "keeping relationships superficial",
                    "overcompensation_for": "fear of vulnerability",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Hearing a loud noise",
                    "interpretation": "Danger is here immediately",
                    "distortion_type": "emotional reasoning",
                },
                {
                    "situation": "Someone walks up behind quickly",
                    "interpretation": "I am about to be attacked",
                    "distortion_type": "catastrophizing",
                },
                {
                    "situation": "Feeling vulnerable emotionally",
                    "interpretation": "I am being weak and this is dangerous",
                    "distortion_type": "labeling",
                },
            ],
            "emotional_responses": [
                {"emotion": "fear", "intensity": 0.92, "valence": "negative"},
                {"emotion": "anger", "intensity": 0.8, "valence": "negative"},
                {"emotion": "numbness", "intensity": 0.7, "valence": "negative"},
                {"emotion": "shame", "intensity": 0.75, "valence": "negative"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Flinches at sudden noises",
                    "triggered_by": "loud unexpected sound",
                    "consequence": "Increased hypervigilance",
                },
                {
                    "behavior": "Avoids locations resembling trauma context",
                    "triggered_by": "reminders of traumatic event",
                    "consequence": "Narrowing of safe spaces",
                },
                {
                    "behavior": "Sleeps with lights on",
                    "triggered_by": "darkness triggering memories",
                    "consequence": "Chronic sleep deprivation",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.2, "world_views": 0.1, "future_views": 0.15},
            ],
        },
        linguistic_features={
            "hedging": 0.3,
            "negation_density": 0.5,
            "first_person_singular": 0.75,
            "cause_words": 0.65,
            "absolutist_words": 0.8,
        },
        severity_range=(0.5, 0.95),
        common_triggers=[
            "anniversary of trauma",
            "sensory reminders",
            "feeling trapped or vulnerable",
            "media depictions of similar events",
        ],
        treatment_history="Prolonged exposure and EMGP show partial response; requires stabilization phase.",
    )
)

# ── 6. OCD ──
_register(
    ClinicalProfile(
        name="ocd",
        display_name="Obsessive-Compulsive Disorder",
        description="Intrusive thoughts and repetitive behaviors aimed at preventing perceived harm.",
        diagnoses=["F42.2"],
        typical_symptoms=[
            "intrusive unwanted thoughts",
            "compulsive rituals",
            "fear of contamination",
            "need for symmetry or exactness",
            "excessive doubt and checking",
        ],
        default_style=_N,
        ccd_config={
            "core_beliefs": [
                {"content": "My thoughts are dangerous and meaningful", "domain": "self", "conviction": 0.88},
                {"content": "I am responsible for preventing all possible harm", "domain": "self", "conviction": 0.92},
                {"content": "The world is full of invisible threats", "domain": "world", "conviction": 0.78},
            ],
            "intermediate_beliefs": [
                {"content": "I must be 100 percent certain about everything", "rule_type": "rule", "conviction": 0.95},
                {
                    "content": "Having a bad thought is morally equivalent to acting on it",
                    "rule_type": "attitude",
                    "conviction": 0.85,
                },
                {
                    "content": "If I do not perform this ritual, something terrible will happen",
                    "rule_type": "assumption",
                    "conviction": 0.93,
                },
            ],
            "coping_strategies": [
                {"content": "Ritualistic checking behaviors", "strategy_type": "compensation", "effectiveness": 0.25},
                {"content": "Avoiding triggers for obsessions", "strategy_type": "avoidance", "effectiveness": 0.2},
                {"content": "Mental neutralization rituals", "strategy_type": "compensation", "effectiveness": 0.15},
            ],
            "compensatory_strategies": [
                {
                    "content": "Excessive hand washing",
                    "behavior": "washing until skin is raw",
                    "overcompensation_for": "fear of contamination",
                },
                {
                    "content": "Counting rituals during routine tasks",
                    "behavior": "counting steps while walking",
                    "overcompensation_for": "fear of losing control",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Having a violent intrusive thought",
                    "interpretation": "I am secretly dangerous",
                    "distortion_type": "thought-action fusion",
                },
                {
                    "situation": "Touching a door handle in public",
                    "interpretation": "I am now contaminated",
                    "distortion_type": "magical thinking",
                },
                {
                    "situation": "Leaving the house without checking",
                    "interpretation": "The house will burn down because of me",
                    "distortion_type": "catastrophizing",
                },
            ],
            "emotional_responses": [
                {"emotion": "anxiety", "intensity": 0.88, "valence": "negative"},
                {"emotion": "disgust", "intensity": 0.75, "valence": "negative"},
                {"emotion": "guilt", "intensity": 0.8, "valence": "negative"},
                {"emotion": "temporary relief after ritual", "intensity": 0.65, "valence": "positive"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Repeats actions until they feel right",
                    "triggered_by": "intrusive doubt while performing task",
                    "consequence": "Significant time loss and exhaustion",
                },
                {
                    "behavior": "Asks for repeated reassurance",
                    "triggered_by": "uncertainty about having caused harm",
                    "consequence": "Strained relationships",
                },
                {
                    "behavior": "Arranges objects symmetrically",
                    "triggered_by": "visual imbalance causing discomfort",
                    "consequence": "Temporary anxiety reduction",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.4, "world_views": 0.3, "future_views": 0.35},
            ],
        },
        linguistic_features={
            "hedging": 0.5,
            "negation_density": 0.45,
            "first_person_singular": 0.65,
            "cause_words": 0.7,
            "absolutist_words": 0.85,
        },
        severity_range=(0.3, 0.85),
        common_triggers=[
            "uncertainty or ambiguity",
            "perceived contamination",
            "moral or religious scrupulosity",
            "asymmetry or disorder",
        ],
        treatment_history="Gold-standard treatment: ERP + SSRI (fluoxetine or fluvoxamine).",
    )
)

# ── 7. Bipolar I Disorder ──
_register(
    ClinicalProfile(
        name="bipolar_i",
        display_name="Bipolar I Disorder",
        description="Full manic episodes with possible depressive episodes, cycling between poles.",
        diagnoses=["F31.1", "F31.3", "F31.6"],
        typical_symptoms=[
            "manic episodes lasting 7+ days",
            "grandiose beliefs",
            "decreased need for sleep",
            "pressured speech",
            "impulsive decision-making",
            "possible depressive episodes",
        ],
        default_style=_MN,
        ccd_config={
            "core_beliefs": [
                {"content": "I am destined for greatness", "domain": "self", "conviction": 0.9},
                {"content": "The universe is aligned around me", "domain": "world", "conviction": 0.85},
                {"content": "Rules and limits do not apply to me", "domain": "self", "conviction": 0.88},
            ],
            "intermediate_beliefs": [
                {"content": "I must seize every opportunity immediately", "rule_type": "rule", "conviction": 0.85},
                {
                    "content": "Slowing down means missing out on something important",
                    "rule_type": "attitude",
                    "conviction": 0.82,
                },
                {
                    "content": "If I have a creative idea, I must act on it right away",
                    "rule_type": "assumption",
                    "conviction": 0.9,
                },
            ],
            "coping_strategies": [
                {"content": "High-risk novelty seeking", "strategy_type": "overcompensation", "effectiveness": 0.1},
                {"content": "Minimizing sleep for productivity", "strategy_type": "compensation", "effectiveness": 0.3},
                {"content": "Multiple concurrent projects", "strategy_type": "compensation", "effectiveness": 0.15},
            ],
            "compensatory_strategies": [
                {
                    "content": "Grandiose financial spending sprees",
                    "behavior": "impulsive purchases and investments",
                    "overcompensation_for": "fear of insignificance",
                },
                {
                    "content": "Excessive social engagement",
                    "behavior": "constant need for audience and validation",
                    "overcompensation_for": "underlying emptiness",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Receiving a compliment",
                    "interpretation": "I am clearly the most talented person here",
                    "distortion_type": "grandiosity",
                },
                {
                    "situation": "Someone disagrees with an idea",
                    "interpretation": "They are jealous of my brilliance",
                    "distortion_type": "mind-reading",
                },
                {
                    "situation": "Feeling energetic after little sleep",
                    "interpretation": "I am more productive than everyone else",
                    "distortion_type": "selective abstraction",
                },
            ],
            "emotional_responses": [
                {"emotion": "euphoria", "intensity": 0.9, "valence": "positive"},
                {"emotion": "irritability", "intensity": 0.7, "valence": "negative"},
                {"emotion": "excitement", "intensity": 0.92, "valence": "positive"},
                {"emotion": "grandiosity", "intensity": 0.85, "valence": "positive"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Starts multiple projects without finishing any",
                    "triggered_by": "surge of creative energy",
                    "consequence": "Incomplete commitments and chaos",
                },
                {
                    "behavior": "Speaks rapidly with pressured speech",
                    "triggered_by": "racing thoughts",
                    "consequence": "Others cannot follow or respond",
                },
                {
                    "behavior": "Impulsive travel or spending",
                    "triggered_by": "grandiose plan",
                    "consequence": "Financial and relational consequences",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.95, "world_views": 0.9, "future_views": 0.98},
            ],
        },
        linguistic_features={
            "hedging": 0.1,
            "negation_density": 0.15,
            "first_person_singular": 0.8,
            "cause_words": 0.5,
            "absolutist_words": 0.7,
        },
        severity_range=(0.4, 0.95),
        common_triggers=[
            "sleep disruption",
            "stressful life events",
            "seasonal changes (spring)",
            "antidepressant monotherapy",
        ],
        treatment_history="Mood stabilizer (lithium) maintenance; atypical antipsychotics for acute mania.",
    )
)

# ── 8. Bipolar II Disorder ──
_register(
    ClinicalProfile(
        name="bipolar_ii",
        display_name="Bipolar II Disorder",
        description="Hypomanic episodes alternating with major depressive episodes. Depressive predominance.",
        diagnoses=["F31.81"],
        typical_symptoms=[
            "hypomanic episodes (4+ days)",
            "major depressive episodes",
            "mixed features possible",
            "anxiety and rumination",
            "interpersonal sensitivity",
        ],
        default_style=_A,
        ccd_config={
            "core_beliefs": [
                {"content": "I can accomplish anything during good periods", "domain": "self", "conviction": 0.82},
                {"content": "I am fundamentally broken when depressed", "domain": "self", "conviction": 0.85},
                {"content": "My moods are uncontrollable", "domain": "self", "conviction": 0.78},
            ],
            "intermediate_beliefs": [
                {
                    "content": "I must capitalize on good moods before they disappear",
                    "rule_type": "rule",
                    "conviction": 0.85,
                },
                {"content": "Feeling good now means a crash is inevitable", "rule_type": "attitude", "conviction": 0.8},
                {
                    "content": "If I am not productive, I am wasting my potential",
                    "rule_type": "assumption",
                    "conviction": 0.82,
                },
            ],
            "coping_strategies": [
                {
                    "content": "Overcommitting during hypomania",
                    "strategy_type": "overcompensation",
                    "effectiveness": 0.1,
                },
                {"content": "Isolating during depressive phases", "strategy_type": "avoidance", "effectiveness": 0.2},
                {
                    "content": "Creative expression to stabilize mood",
                    "strategy_type": "compensation",
                    "effectiveness": 0.5,
                },
            ],
            "compensatory_strategies": [
                {
                    "content": "Intensive career pushes during hypomania",
                    "behavior": "working 16-hour days to catch up",
                    "overcompensation_for": "anticipated depressive downtime",
                },
                {
                    "content": "Preemptive apology for mood fluctuations",
                    "behavior": "warning others about capacity changes",
                    "overcompensation_for": "fear of being unreliable",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Feeling productive and energetic",
                    "interpretation": "I must be becoming manic, this will not last",
                    "distortion_type": "emotional reasoning",
                },
                {
                    "situation": "Feeling sad after productive week",
                    "interpretation": "I have failed again, the depression is back",
                    "distortion_type": "all-or-nothing",
                },
                {
                    "situation": "Others seem stable and consistent",
                    "interpretation": "Everyone else has it together except me",
                    "distortion_type": "comparison trap",
                },
            ],
            "emotional_responses": [
                {"emotion": "optimism during hypomania", "intensity": 0.85, "valence": "positive"},
                {"emotion": "despair during depression", "intensity": 0.88, "valence": "negative"},
                {"emotion": "anxiety about cycling", "intensity": 0.75, "valence": "negative"},
                {"emotion": "hopelessness about stability", "intensity": 0.7, "valence": "negative"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Takes on ambitious projects impulsively",
                    "triggered_by": "hypomanic phase",
                    "consequence": "Overwhelmed when depressive phase returns",
                },
                {
                    "behavior": "Withdraws from social commitments",
                    "triggered_by": "depressive episode onset",
                    "consequence": "Damaged relationships and guilt",
                },
                {
                    "behavior": "Frequently checks mood tracking apps",
                    "triggered_by": "uncertainty about episode status",
                    "consequence": "Hypervigilance to mood changes",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.5, "world_views": 0.55, "future_views": 0.4},
            ],
        },
        linguistic_features={
            "hedging": 0.55,
            "negation_density": 0.3,
            "first_person_singular": 0.7,
            "cause_words": 0.65,
            "absolutist_words": 0.5,
        },
        severity_range=(0.3, 0.85),
        common_triggers=[
            "sleep disruption",
            "interpersonal stress",
            "failure or criticism",
            "seasonal changes",
        ],
        treatment_history="Lamotrigine for maintenance; quetiapine for acute episodes. Avoid SSRI monotherapy.",
    )
)

# ── 9. Paranoid Schizophrenia ──
_register(
    ClinicalProfile(
        name="paranoid_schizophrenia",
        display_name="Paranoid Schizophrenia",
        description="Prominent delusions and hallucinations with paranoid or persecutory themes.",
        diagnoses=["F20.0"],
        typical_symptoms=[
            "persecutory delusions",
            "auditory hallucinations",
            "referential thinking",
            "suspiciousness",
            "blunted affect possible",
        ],
        default_style=_H,
        ccd_config={
            "core_beliefs": [
                {"content": "People are conspiring against me", "domain": "others", "conviction": 0.95},
                {"content": "My thoughts are being controlled or monitored", "domain": "self", "conviction": 0.92},
                {"content": "There are hidden messages intended for me", "domain": "world", "conviction": 0.88},
            ],
            "intermediate_beliefs": [
                {"content": "I must never let my guard down with anyone", "rule_type": "rule", "conviction": 0.95},
                {"content": "Trusting others is a fatal mistake", "rule_type": "attitude", "conviction": 0.92},
                {
                    "content": "If someone seems kind, they are manipulating me",
                    "rule_type": "assumption",
                    "conviction": 0.9,
                },
            ],
            "coping_strategies": [
                {"content": "Social withdrawal and isolation", "strategy_type": "avoidance", "effectiveness": 0.3},
                {"content": "Checking for surveillance devices", "strategy_type": "compensation", "effectiveness": 0.1},
                {"content": "Speaking in coded language", "strategy_type": "compensation", "effectiveness": 0.15},
            ],
            "compensatory_strategies": [
                {
                    "content": "Vigilant scanning of media for hidden messages",
                    "behavior": "interpreting news and TV as personally directed",
                    "overcompensation_for": "feeling of being targeted",
                },
                {
                    "content": "Testing loyalty of others",
                    "behavior": "setting traps to reveal true intentions",
                    "overcompensation_for": "paranoid mistrust",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Two strangers whisper in public",
                    "interpretation": "They are talking about my secret mission",
                    "distortion_type": "referential thinking",
                },
                {
                    "situation": "Car parked across the street",
                    "interpretation": "I am being watched by intelligence agents",
                    "distortion_type": "persecutory delusion",
                },
                {
                    "situation": "Neighbor asks a casual question",
                    "interpretation": "They are probing for information",
                    "distortion_type": "suspiciousness",
                },
            ],
            "emotional_responses": [
                {"emotion": "suspicion", "intensity": 0.95, "valence": "negative"},
                {"emotion": "hypervigilance", "intensity": 0.92, "valence": "negative"},
                {"emotion": "anger", "intensity": 0.8, "valence": "negative"},
                {"emotion": "confusion", "intensity": 0.7, "valence": "negative"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Covers camera on devices",
                    "triggered_by": "belief of being watched",
                    "consequence": "Temporary reduction in paranoia",
                },
                {
                    "behavior": "Refuses to answer questions directly",
                    "triggered_by": "perception of interrogation",
                    "consequence": "Frustrates caregivers",
                },
                {
                    "behavior": "Moves frequently to avoid detection",
                    "triggered_by": "feeling that current location is compromised",
                    "consequence": "Chronic instability",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.3, "world_views": 0.05, "future_views": 0.2},
            ],
        },
        linguistic_features={
            "hedging": 0.2,
            "negation_density": 0.6,
            "first_person_singular": 0.85,
            "cause_words": 0.6,
            "absolutist_words": 0.9,
        },
        severity_range=(0.6, 0.95),
        common_triggers=[
            "perceived social threat",
            "sensory overload",
            "medication non-adherence",
            "substance use",
        ],
        treatment_history="Long-term atypical antipsychotic (clozapine for refractory cases).",
    )
)

# ── 10. Borderline Personality Disorder ──
_register(
    ClinicalProfile(
        name="borderline_personality",
        display_name="Borderline Personality Disorder",
        description="Emotional dysregulation, unstable relationships, identity disturbance, and fear of abandonment.",
        diagnoses=["F60.3"],
        typical_symptoms=[
            "fear of abandonment",
            "unstable relationships",
            "identity disturbance",
            "impulsive behavior",
            "emotional instability",
            "chronic emptiness",
        ],
        default_style=_H,
        ccd_config={
            "core_beliefs": [
                {"content": "People will inevitably abandon me", "domain": "others", "conviction": 0.92},
                {"content": "I am bad and unworthy of love", "domain": "self", "conviction": 0.88},
                {"content": "My feelings define reality", "domain": "world", "conviction": 0.85},
            ],
            "intermediate_beliefs": [
                {"content": "I must test people to see if they really care", "rule_type": "rule", "conviction": 0.85},
                {"content": "If someone leaves me, I am nothing", "rule_type": "assumption", "conviction": 0.92},
                {
                    "content": "Intense relationships are the only meaningful ones",
                    "rule_type": "attitude",
                    "conviction": 0.82,
                },
            ],
            "coping_strategies": [
                {
                    "content": "Non-suicidal self-injury for emotional relief",
                    "strategy_type": "compensation",
                    "effectiveness": 0.35,
                },
                {
                    "content": "Idealizing and devaluing in relationships",
                    "strategy_type": "overcompensation",
                    "effectiveness": 0.1,
                },
                {
                    "content": "Impulsive spending or substance use",
                    "strategy_type": "compensation",
                    "effectiveness": 0.15,
                },
            ],
            "compensatory_strategies": [
                {
                    "content": "Desperate attempts to prevent abandonment",
                    "behavior": "pleading, calling repeatedly, threatening leaving",
                    "overcompensation_for": "fear of being alone",
                },
                {
                    "content": "Rapid idealization of new partners",
                    "behavior": "intense enmeshment within days",
                    "overcompensation_for": "chronic emptiness",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Therapist reschedules an appointment",
                    "interpretation": "They are abandoning me just like everyone else",
                    "distortion_type": "mental filtering",
                },
                {
                    "situation": "Partner does not reply immediately",
                    "interpretation": "They do not care about me anymore",
                    "distortion_type": "jumping to conclusions",
                },
                {
                    "situation": "Feeling angry at someone close",
                    "interpretation": "They are entirely bad and always have been",
                    "distortion_type": "splitting",
                },
            ],
            "emotional_responses": [
                {"emotion": "abandonment panic", "intensity": 0.95, "valence": "negative"},
                {"emotion": "intense anger", "intensity": 0.88, "valence": "negative"},
                {"emotion": "emptiness", "intensity": 0.85, "valence": "negative"},
                {"emotion": "shame", "intensity": 0.8, "valence": "negative"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Self-harm when overwhelmed",
                    "triggered_by": "intense emotional pain or fear of abandonment",
                    "consequence": "Temporary relief followed by shame",
                },
                {
                    "behavior": "Intense calls and texts after perceived rejection",
                    "triggered_by": "minor relational trigger",
                    "consequence": "Often pushes others away",
                },
                {
                    "behavior": "Quits therapy or job impulsively",
                    "triggered_by": "feeling criticized or rejected",
                    "consequence": "Disrupted support and stability",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.2, "world_views": 0.3, "future_views": 0.25},
            ],
        },
        linguistic_features={
            "hedging": 0.3,
            "negation_density": 0.5,
            "first_person_singular": 0.9,
            "cause_words": 0.7,
            "absolutist_words": 0.75,
        },
        severity_range=(0.5, 0.95),
        common_triggers=[
            "perceived abandonment or rejection",
            "criticism or feedback",
            "relationship conflict",
            "feeling alone",
        ],
        treatment_history="DBT is gold standard; MBT has strong evidence. SSRI for comorbid symptoms.",
    )
)

# ── 11. Narcissistic Personality Disorder ──
_register(
    ClinicalProfile(
        name="narcissistic_personality",
        display_name="Narcissistic Personality Disorder",
        description="Grandiose self-importance, need for admiration, and lack of empathy.",
        diagnoses=["F60.81"],
        typical_symptoms=[
            "grandiose sense of self-importance",
            "preoccupation with success fantasies",
            "belief of being special",
            "need for excessive admiration",
            "sense of entitlement",
            "interpersonal exploitation",
        ],
        default_style=_F,
        ccd_config={
            "core_beliefs": [
                {"content": "I am superior and deserve special treatment", "domain": "self", "conviction": 0.92},
                {"content": "Ordinary people are beneath me", "domain": "others", "conviction": 0.85},
                {"content": "Rules exist for others, not for me", "domain": "world", "conviction": 0.8},
            ],
            "intermediate_beliefs": [
                {"content": "I must be admired to feel worthy", "rule_type": "rule", "conviction": 0.88},
                {"content": "Criticism is an attack on my superiority", "rule_type": "attitude", "conviction": 0.85},
                {"content": "If I am not the best, I am nothing", "rule_type": "assumption", "conviction": 0.92},
            ],
            "coping_strategies": [
                {
                    "content": "Devaluing others to maintain self-esteem",
                    "strategy_type": "overcompensation",
                    "effectiveness": 0.5,
                },
                {
                    "content": "Fantasy withdrawal into imagined success",
                    "strategy_type": "compensation",
                    "effectiveness": 0.4,
                },
                {
                    "content": "Avoiding situations that reveal inadequacy",
                    "strategy_type": "avoidance",
                    "effectiveness": 0.45,
                },
            ],
            "compensatory_strategies": [
                {
                    "content": "Name-dropping and status signaling",
                    "behavior": "frequently mentioning important connections",
                    "overcompensation_for": "underlying inadequacy",
                },
                {
                    "content": "Monopolizing conversations in groups",
                    "behavior": "dominating discussions with achievements",
                    "overcompensation_for": "fear of being ordinary",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Not receiving a promotion",
                    "interpretation": "My boss is incompetent and threatened by me",
                    "distortion_type": "blaming",
                },
                {
                    "situation": "Someone else receives recognition",
                    "interpretation": "They do not deserve it; I am more qualified",
                    "distortion_type": "discounting the positive",
                },
                {
                    "situation": "Receiving constructive feedback",
                    "interpretation": "This person is trying to bring me down",
                    "distortion_type": "hostile attribution bias",
                },
            ],
            "emotional_responses": [
                {"emotion": "grandiose pride", "intensity": 0.85, "valence": "positive"},
                {"emotion": "rage when criticized", "intensity": 0.9, "valence": "negative"},
                {"emotion": "envy of others", "intensity": 0.7, "valence": "negative"},
                {"emotion": "emptiness beneath the surface", "intensity": 0.3, "valence": "negative"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Dismisses other viewpoints in team settings",
                    "triggered_by": "someone else's idea being accepted",
                    "consequence": "Alienates colleagues",
                },
                {
                    "behavior": "Name-drops excessively",
                    "triggered_by": "feeling undervalued in a conversation",
                    "consequence": "Seen as arrogant by others",
                },
                {
                    "behavior": "Refuses to admit mistakes",
                    "triggered_by": "being confronted with an error",
                    "consequence": "Repeated patterns of blame externalization",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.95, "world_views": 0.6, "future_views": 0.85},
            ],
        },
        linguistic_features={
            "hedging": 0.1,
            "negation_density": 0.2,
            "first_person_singular": 0.9,
            "cause_words": 0.3,
            "absolutist_words": 0.8,
        },
        severity_range=(0.3, 0.8),
        common_triggers=[
            "criticism or challenge to authority",
            "being ignored or outshined",
            "failure or setback",
            "exposure of limitations",
        ],
        treatment_history="Limited insight and low treatment engagement. Transference-focused psychotherapy.",
    )
)

# ── 12. Avoidant Personality Disorder ──
_register(
    ClinicalProfile(
        name="avoidant_personality",
        display_name="Avoidant Personality Disorder",
        description="Pervasive social inhibition, feelings of inadequacy, and hypersensitivity to negative evaluation.",
        diagnoses=["F60.6"],
        typical_symptoms=[
            "avoidance of occupational activities involving contact",
            "fear of disapproval or rejection",
            "feeling inadequate and inferior",
            "extreme shyness",
            "reluctance to try new activities",
        ],
        default_style=_A,
        ccd_config={
            "core_beliefs": [
                {"content": "I am socially inept and inferior to others", "domain": "self", "conviction": 0.92},
                {
                    "content": "Others will inevitably reject me if they get close",
                    "domain": "others",
                    "conviction": 0.9,
                },
                {"content": "The social world is full of painful evaluations", "domain": "world", "conviction": 0.8},
            ],
            "intermediate_beliefs": [
                {"content": "I must avoid any risk of social rejection", "rule_type": "rule", "conviction": 0.92},
                {"content": "Being myself is not good enough for others", "rule_type": "attitude", "conviction": 0.88},
                {
                    "content": "If I try to connect and am rejected, I will be destroyed",
                    "rule_type": "assumption",
                    "conviction": 0.85,
                },
            ],
            "coping_strategies": [
                {
                    "content": "Avoiding new social situations entirely",
                    "strategy_type": "avoidance",
                    "effectiveness": 0.45,
                },
                {"content": "Keeping relationships superficial", "strategy_type": "avoidance", "effectiveness": 0.4},
                {"content": "Imaginary social rehearsals", "strategy_type": "compensation", "effectiveness": 0.2},
            ],
            "compensatory_strategies": [
                {
                    "content": "Avoiding career advancement opportunities",
                    "behavior": "turning down promotions requiring more interaction",
                    "overcompensation_for": "fear of exposure as inadequate",
                },
                {
                    "content": "Imaginary friendships to fill social void",
                    "behavior": "fantasizing about ideal relationships",
                    "overcompensation_for": "chronic loneliness",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Invited to a colleague's party",
                    "interpretation": "They only invited me out of obligation",
                    "distortion_type": "mind-reading",
                },
                {
                    "situation": "Working on a team project",
                    "interpretation": "Everyone else is more competent than me",
                    "distortion_type": "comparison trap",
                },
                {
                    "situation": "Someone glances during conversation",
                    "interpretation": "They noticed how awkward I am",
                    "distortion_type": "self-referential thinking",
                },
            ],
            "emotional_responses": [
                {"emotion": "anxiety", "intensity": 0.85, "valence": "negative"},
                {"emotion": "shame", "intensity": 0.82, "valence": "negative"},
                {"emotion": "loneliness", "intensity": 0.78, "valence": "negative"},
                {"emotion": "longing for connection", "intensity": 0.7, "valence": "mixed"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Avoids work meetings or sits silently",
                    "triggered_by": "fear of saying something stupid",
                    "consequence": "Perceived as unfriendly or disengaged",
                },
                {
                    "behavior": "Declines social invitations routinely",
                    "triggered_by": "anticipatory anxiety about social interaction",
                    "consequence": "Growing social isolation",
                },
                {
                    "behavior": "Leaves group situations early",
                    "triggered_by": "feeling exposed or judged",
                    "consequence": "Reinforces self-concept as social failure",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.2, "world_views": 0.35, "future_views": 0.3},
            ],
        },
        linguistic_features={
            "hedging": 0.88,
            "negation_density": 0.3,
            "first_person_singular": 0.78,
            "cause_words": 0.35,
            "absolutist_words": 0.4,
        },
        severity_range=(0.4, 0.85),
        common_triggers=[
            "social evaluation",
            "performance expectations",
            "intimacy or closeness",
            "change or novelty",
        ],
        treatment_history="CBT and group therapy with gradual exposure. SSRI for anxiety.",
    )
)

# ── 13. Dependent Personality Disorder ──
_register(
    ClinicalProfile(
        name="dependent_personality",
        display_name="Dependent Personality Disorder",
        description="Excessive need to be taken care of, leading to submissive and clinging behavior.",
        diagnoses=["F60.7"],
        typical_symptoms=[
            "difficulty making decisions alone",
            "need for others to assume responsibility",
            "fear of disagreeing with others",
            "difficulty initiating projects",
            "urgent search for replacement relationships",
        ],
        default_style=_F,
        ccd_config={
            "core_beliefs": [
                {"content": "I am unable to function alone", "domain": "self", "conviction": 0.92},
                {"content": "I need someone stronger to guide me", "domain": "self", "conviction": 0.88},
                {
                    "content": "The world is too complex for me to handle independently",
                    "domain": "world",
                    "conviction": 0.82,
                },
            ],
            "intermediate_beliefs": [
                {
                    "content": "I must never make important decisions without advice",
                    "rule_type": "rule",
                    "conviction": 0.92,
                },
                {
                    "content": "Disagreeing with someone means losing their support",
                    "rule_type": "attitude",
                    "conviction": 0.85,
                },
                {
                    "content": "If I am alone, I will fall apart completely",
                    "rule_type": "assumption",
                    "conviction": 0.9,
                },
            ],
            "coping_strategies": [
                {
                    "content": "Seeking excessive guidance before decisions",
                    "strategy_type": "compensation",
                    "effectiveness": 0.4,
                },
                {"content": "Deferring to stronger personalities", "strategy_type": "avoidance", "effectiveness": 0.5},
                {
                    "content": "Rapid attachment to new caregivers",
                    "strategy_type": "compensation",
                    "effectiveness": 0.35,
                },
            ],
            "compensatory_strategies": [
                {
                    "content": "Deferring all major life decisions",
                    "behavior": "consulting others for even minor choices",
                    "overcompensation_for": "fear of making wrong decision",
                },
                {
                    "content": "Over-compliance in relationships",
                    "behavior": "agreeing to things against own interests",
                    "overcompensation_for": "fear of abandonment",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Partner goes out of town",
                    "interpretation": "I cannot manage alone for even a few days",
                    "distortion_type": "catastrophizing",
                },
                {
                    "situation": "Needing to make a career decision",
                    "interpretation": "I am not equipped to choose wisely",
                    "distortion_type": "labeling",
                },
                {
                    "situation": "Therapist suggests greater independence",
                    "interpretation": "They are trying to abandon me",
                    "distortion_type": "personalization",
                },
            ],
            "emotional_responses": [
                {"emotion": "anxiety when alone", "intensity": 0.9, "valence": "negative"},
                {"emotion": "relief when advice is given", "intensity": 0.85, "valence": "positive"},
                {"emotion": "helplessness", "intensity": 0.82, "valence": "negative"},
                {"emotion": "gratitude toward caregivers", "intensity": 0.8, "valence": "positive"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Phone calls to check-in during workday",
                    "triggered_by": "needing reassurance while alone",
                    "consequence": "Interruption of daily flow",
                },
                {
                    "behavior": "Seeks advice for routine decisions",
                    "triggered_by": "deciding what to eat or wear",
                    "consequence": "Frustrates those around them",
                },
                {
                    "behavior": "Stays in unhealthy relationships",
                    "triggered_by": "fear of being alone",
                    "consequence": "Sustains exploitative dynamics",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.15, "world_views": 0.3, "future_views": 0.35},
            ],
        },
        linguistic_features={
            "hedging": 0.85,
            "negation_density": 0.25,
            "first_person_singular": 0.7,
            "cause_words": 0.3,
            "absolutist_words": 0.35,
        },
        severity_range=(0.3, 0.8),
        common_triggers=[
            "being alone",
            "decision-making pressure",
            "relationship rupture or change",
            "criticism or disapproval",
        ],
        treatment_history="Gradual autonomy training in therapy. Responds well to CBT with behavioral activation.",
    )
)

# ── 14. Substance Use Disorder ──
_register(
    ClinicalProfile(
        name="substance_use_disorder",
        display_name="Substance Use Disorder",
        description="Compulsive substance use despite negative consequences, with craving and loss of control.",
        diagnoses=["F10.20", "F11.20", "F12.20"],
        typical_symptoms=[
            "impaired control over use",
            "craving or strong desire",
            "continued use despite consequences",
            "tolerance and withdrawal",
            "neglect of important activities",
        ],
        default_style=_N,
        ccd_config={
            "core_beliefs": [
                {"content": "I cannot cope with life without substances", "domain": "self", "conviction": 0.88},
                {"content": "I am fundamentally weak for needing substances", "domain": "self", "conviction": 0.85},
                {"content": "The world is too stressful to face sober", "domain": "world", "conviction": 0.78},
            ],
            "intermediate_beliefs": [
                {
                    "content": "I deserve to use because of what I have been through",
                    "rule_type": "rule",
                    "conviction": 0.82,
                },
                {"content": "Using is the only way to feel okay", "rule_type": "attitude", "conviction": 0.88},
                {"content": "If I feel a craving, I have already failed", "rule_type": "assumption", "conviction": 0.7},
            ],
            "coping_strategies": [
                {"content": "Substance use to manage emotions", "strategy_type": "compensation", "effectiveness": 0.4},
                {"content": "Avoiding sobriety triggers", "strategy_type": "avoidance", "effectiveness": 0.25},
                {
                    "content": "Minimizing severity of use to others",
                    "strategy_type": "compensation",
                    "effectiveness": 0.15,
                },
            ],
            "compensatory_strategies": [
                {
                    "content": "Hiding substance use from loved ones",
                    "behavior": "secretive storage and consumption",
                    "overcompensation_for": "shame about use",
                },
                {
                    "content": "Self-justification for relapse",
                    "behavior": "constructing elaborate reasons for using again",
                    "overcompensation_for": "guilt about loss of control",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Feeling stressed after work",
                    "interpretation": "I need a drink to unwind and relax",
                    "distortion_type": "emotional reasoning",
                },
                {
                    "situation": "Someone expresses concern about use",
                    "interpretation": "They are judging me and do not understand",
                    "distortion_type": "minimization",
                },
                {
                    "situation": "Experiencing a craving",
                    "interpretation": "My body needs it, I cannot resist",
                    "distortion_type": "chaining",
                },
            ],
            "emotional_responses": [
                {"emotion": "shame", "intensity": 0.85, "valence": "negative"},
                {"emotion": "guilt", "intensity": 0.78, "valence": "negative"},
                {"emotion": "craving or desire", "intensity": 0.92, "valence": "mixed"},
                {"emotion": "numbness", "intensity": 0.7, "valence": "negative"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Secretly consumes substances during the day",
                    "triggered_by": "daily stressors",
                    "consequence": "Increased tolerance and dependency",
                },
                {
                    "behavior": "Lies about quantity of use",
                    "triggered_by": "being asked about consumption",
                    "consequence": "Broken trust in relationships",
                },
                {
                    "behavior": "Prioritizes substance access over obligations",
                    "triggered_by": "craving or withdrawal onset",
                    "consequence": "Job loss and relationship damage",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.35, "world_views": 0.35, "future_views": 0.3},
            ],
        },
        linguistic_features={
            "hedging": 0.5,
            "negation_density": 0.35,
            "first_person_singular": 0.8,
            "cause_words": 0.6,
            "absolutist_words": 0.4,
        },
        severity_range=(0.4, 0.85),
        common_triggers=[
            "stressful situations",
            "social pressure",
            "exposure to substance cues",
            "negative emotional states",
        ],
        treatment_history="MAT (buprenorphine/naltrexone) + CBT/MI. High relapse rate in first year.",
    )
)

# ── 15. Anorexia Nervosa ──
_register(
    ClinicalProfile(
        name="anorexia_nervosa",
        display_name="Anorexia Nervosa",
        description="Restrictive eating, intense fear of weight gain, and distorted body image.",
        diagnoses=["F50.01", "F50.02"],
        typical_symptoms=[
            "restriction of energy intake",
            "intense fear of weight gain",
            "distorted body image",
            "low body weight",
            "preoccupation with food and weight",
        ],
        default_style=_A,
        ccd_config={
            "core_beliefs": [
                {"content": "I must control my body completely to be worthy", "domain": "self", "conviction": 0.92},
                {"content": "Being thin is the only source of value", "domain": "self", "conviction": 0.88},
                {"content": "My body is unacceptable as it is", "domain": "self", "conviction": 0.9},
            ],
            "intermediate_beliefs": [
                {"content": "I must never lose control over eating", "rule_type": "rule", "conviction": 0.95},
                {"content": "Feeling hungry means I am succeeding", "rule_type": "attitude", "conviction": 0.85},
                {
                    "content": "If I gain any weight, I will become disgusting",
                    "rule_type": "assumption",
                    "conviction": 0.92,
                },
            ],
            "coping_strategies": [
                {"content": "Strict caloric restriction", "strategy_type": "compensation", "effectiveness": 0.4},
                {
                    "content": "Excessive exercise to burn calories",
                    "strategy_type": "overcompensation",
                    "effectiveness": 0.3,
                },
                {"content": "Avoiding social eating situations", "strategy_type": "avoidance", "effectiveness": 0.5},
            ],
            "compensatory_strategies": [
                {
                    "content": "Rigid food rituals",
                    "behavior": "eating foods in exact order, cutting into tiny pieces",
                    "overcompensation_for": "fear of consuming too much",
                },
                {
                    "content": "Body checking multiple times daily",
                    "behavior": "weighing, pinching, measuring repeatedly",
                    "overcompensation_for": "distorted body image",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Being offered food by a friend",
                    "interpretation": "They are trying to sabotage my control",
                    "distortion_type": "hostile attribution bias",
                },
                {
                    "situation": "Feeling slightly full after a meal",
                    "interpretation": "I have already lost control completely",
                    "distortion_type": "all-or-nothing",
                },
                {
                    "situation": "Looking in the mirror",
                    "interpretation": "I still see so much fat despite being underweight",
                    "distortion_type": "body dysmorphic perception",
                },
            ],
            "emotional_responses": [
                {"emotion": "anxiety about eating", "intensity": 0.92, "valence": "negative"},
                {"emotion": "sense of accomplishment when restricting", "intensity": 0.85, "valence": "positive"},
                {"emotion": "guilt after eating", "intensity": 0.9, "valence": "negative"},
                {"emotion": "shame about body", "intensity": 0.88, "valence": "negative"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Weighs self multiple times daily",
                    "triggered_by": "uncertainty about body state",
                    "consequence": "Fluctuations cause extreme distress",
                },
                {
                    "behavior": "Hides food or claims to have eaten",
                    "triggered_by": "meal times with others",
                    "consequence": "Strained family relationships",
                },
                {
                    "behavior": "Exercises excessively despite physical weakness",
                    "triggered_by": "consuming any calories",
                    "consequence": "Physical health deterioration",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.25, "world_views": 0.4, "future_views": 0.35},
            ],
        },
        linguistic_features={
            "hedging": 0.4,
            "negation_density": 0.55,
            "first_person_singular": 0.85,
            "cause_words": 0.5,
            "absolutist_words": 0.8,
        },
        severity_range=(0.4, 0.9),
        common_triggers=[
            "meal times",
            "weigh-ins",
            "seeing body in mirror or photos",
            "comments about weight or food",
        ],
        treatment_history="FBT for adolescents; CBT-E for adults. Medical stabilization first if severely underweight.",
    )
)

# ── 16. Bulimia Nervosa ──
_register(
    ClinicalProfile(
        name="bulimia_nervosa",
        display_name="Bulimia Nervosa",
        description="Binge-eating followed by compensatory behaviors to prevent weight gain.",
        diagnoses=["F50.2"],
        typical_symptoms=[
            "recurrent binge eating episodes",
            "inappropriate compensatory behaviors",
            "self-evaluation dominated by weight",
            "alternating dieting and bingeing",
            "dental erosion and GI issues",
        ],
        default_style=_MC,
        ccd_config={
            "core_beliefs": [
                {"content": "I am out of control around food", "domain": "self", "conviction": 0.88},
                {"content": "My weight determines my worth as a person", "domain": "self", "conviction": 0.9},
                {"content": "I am disgusting when I binge", "domain": "self", "conviction": 0.92},
            ],
            "intermediate_beliefs": [
                {"content": "I must purge after eating anything substantial", "rule_type": "rule", "conviction": 0.88},
                {"content": "Bingeing is a failure, and I must undo it", "rule_type": "attitude", "conviction": 0.92},
                {
                    "content": "If I eat normally, I will gain weight uncontrollably",
                    "rule_type": "assumption",
                    "conviction": 0.82,
                },
            ],
            "coping_strategies": [
                {"content": "Self-induced vomiting after meals", "strategy_type": "compensation", "effectiveness": 0.3},
                {"content": "Laxative or diuretic misuse", "strategy_type": "compensation", "effectiveness": 0.2},
                {"content": "Strict dieting between binges", "strategy_type": "compensation", "effectiveness": 0.25},
            ],
            "compensatory_strategies": [
                {
                    "content": "Binge planning and secrecy",
                    "behavior": "stockpiling food for planned binges",
                    "overcompensation_for": "feelings of deprivation",
                },
                {
                    "content": "Intense exercise as self-punishment",
                    "behavior": "exercising for hours after bingeing",
                    "overcompensation_for": "guilt about loss of control",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Slight feeling of emptiness after meal",
                    "interpretation": "I have already ruined my diet, I might as well binge",
                    "distortion_type": "all-or-nothing",
                },
                {
                    "situation": "Seeing high-calorie food",
                    "interpretation": "I want it so badly and I cannot resist",
                    "distortion_type": "emotional reasoning",
                },
                {
                    "situation": "Feeling bloated or full",
                    "interpretation": "I am becoming fat this instant",
                    "distortion_type": "catastrophizing",
                },
            ],
            "emotional_responses": [
                {"emotion": "anxiety before binge", "intensity": 0.8, "valence": "negative"},
                {"emotion": "temporary relief during binge", "intensity": 0.7, "valence": "mixed"},
                {"emotion": "intense shame after binge", "intensity": 0.92, "valence": "negative"},
                {"emotion": "guilt and self-disgust", "intensity": 0.88, "valence": "negative"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Binges in secret, consuming large quantities quickly",
                    "triggered_by": "dietary restriction violation",
                    "consequence": "Cycle of shame and purging",
                },
                {
                    "behavior": "Vomits immediately after eating",
                    "triggered_by": "feeling full or having broken a food rule",
                    "consequence": "Electrolyte imbalance and dental damage",
                },
                {
                    "behavior": "Frequently checks body in reflective surfaces",
                    "triggered_by": "anxiety about weight fluctuation",
                    "consequence": "Reinforces body dissatisfaction",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.2, "world_views": 0.35, "future_views": 0.25},
            ],
        },
        linguistic_features={
            "hedging": 0.35,
            "negation_density": 0.5,
            "first_person_singular": 0.88,
            "cause_words": 0.55,
            "absolutist_words": 0.7,
        },
        severity_range=(0.4, 0.85),
        common_triggers=[
            "violation of dietary rules",
            "negative emotions",
            "interpersonal stress",
            "body dissatisfaction",
        ],
        treatment_history="CBT-E is first-line; SSRI (fluoxetine 60 mg) reduces binge-purge frequency.",
    )
)

# ── 17. Somatic Symptom Disorder ──
_register(
    ClinicalProfile(
        name="somatic_symptom_disorder",
        display_name="Somatic Symptom Disorder",
        description="Distressing physical symptoms with excessive thoughts, feelings, or behaviors related to them.",
        diagnoses=["F45.1"],
        typical_symptoms=[
            "one or more distressing somatic symptoms",
            "excessive thoughts about symptom seriousness",
            "high health anxiety",
            "frequent medical visits",
            "functional impairment",
        ],
        default_style=_A,
        ccd_config={
            "core_beliefs": [
                {"content": "My body is unreliable and dangerous", "domain": "self", "conviction": 0.88},
                {"content": "Doctors are missing something serious", "domain": "others", "conviction": 0.82},
                {"content": "Physical symptoms always signal disease", "domain": "world", "conviction": 0.85},
            ],
            "intermediate_beliefs": [
                {"content": "I must monitor every physical sensation closely", "rule_type": "rule", "conviction": 0.9},
                {
                    "content": "Feeling something in my body means something is wrong",
                    "rule_type": "attitude",
                    "conviction": 0.85,
                },
                {
                    "content": "If I ignore a symptom, it will become untreatable",
                    "rule_type": "assumption",
                    "conviction": 0.82,
                },
            ],
            "coping_strategies": [
                {"content": "Frequent medical appointments", "strategy_type": "compensation", "effectiveness": 0.2},
                {"content": "Online symptom checking", "strategy_type": "compensation", "effectiveness": 0.05},
                {
                    "content": "Avoiding activities that cause symptoms",
                    "strategy_type": "avoidance",
                    "effectiveness": 0.3,
                },
            ],
            "compensatory_strategies": [
                {
                    "content": "Documenting all symptoms meticulously",
                    "behavior": "keeping detailed symptom journals and photos",
                    "overcompensation_for": "fear of being disbelieved",
                },
                {
                    "content": "Seeking multiple medical opinions",
                    "behavior": "doctor shopping until one test supports concern",
                    "overcompensation_for": "distrust of negative test results",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Mild headache",
                    "interpretation": "This could be a brain tumor",
                    "distortion_type": "catastrophizing",
                },
                {
                    "situation": "Occasional palpitations",
                    "interpretation": "My heart is failing",
                    "distortion_type": "catastrophizing",
                },
                {
                    "situation": "Feeling tired",
                    "interpretation": "There must be an undiagnosed medical condition",
                    "distortion_type": "emotional reasoning",
                },
            ],
            "emotional_responses": [
                {"emotion": "health anxiety", "intensity": 0.9, "valence": "negative"},
                {"emotion": "fear of death or disease", "intensity": 0.88, "valence": "negative"},
                {"emotion": "frustration with doctors", "intensity": 0.75, "valence": "negative"},
                {"emotion": "temporary relief after test", "intensity": 0.7, "valence": "positive"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Goes to ER for non-emergency symptoms",
                    "triggered_by": "novel physical sensation",
                    "consequence": "Negative tests reinforce temporary relief",
                },
                {
                    "behavior": "Consults multiple specialists simultaneously",
                    "triggered_by": "dissatisfaction with benign explanation",
                    "consequence": "Fragmented care and polypharmacy risk",
                },
                {
                    "behavior": "Avoids exercise or exertion",
                    "triggered_by": "fear of triggering symptoms",
                    "consequence": "Physical deconditioning",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.35, "world_views": 0.3, "future_views": 0.3},
            ],
        },
        linguistic_features={
            "hedging": 0.6,
            "negation_density": 0.3,
            "first_person_singular": 0.7,
            "cause_words": 0.75,
            "absolutist_words": 0.5,
        },
        severity_range=(0.3, 0.8),
        common_triggers=[
            "normal bodily sensations",
            "media about diseases",
            "medical appointment reminders",
            "illness or death of others",
        ],
        treatment_history="CBT focused on health anxiety. Gradual reduction of medical utilization. SSRI.",
    )
)

# ── 18. Insomnia Disorder ──
_register(
    ClinicalProfile(
        name="insomnia_disorder",
        display_name="Insomnia Disorder",
        description="Persistent difficulty initiating or maintaining sleep with significant daytime impairment.",
        diagnoses=["G47.00", "F51.01"],
        typical_symptoms=[
            "difficulty falling asleep",
            "frequent night wakings",
            "early morning awakening",
            "daytime fatigue",
            "irritability and mood disturbance",
            "impaired concentration",
        ],
        default_style=_A,
        ccd_config={
            "core_beliefs": [
                {"content": "I cannot function without perfect sleep", "domain": "self", "conviction": 0.85},
                {"content": "My sleep is completely out of my control", "domain": "self", "conviction": 0.82},
                {"content": "A bad night always predicts a terrible day", "domain": "world", "conviction": 0.78},
            ],
            "intermediate_beliefs": [
                {"content": "I must try harder to fall asleep", "rule_type": "rule", "conviction": 0.88},
                {"content": "Lying in bed awake is productive worry time", "rule_type": "attitude", "conviction": 0.75},
                {
                    "content": "If I do not sleep eight hours, I will fail tomorrow",
                    "rule_type": "assumption",
                    "conviction": 0.85,
                },
            ],
            "coping_strategies": [
                {
                    "content": "Spending extra hours in bed trying to sleep",
                    "strategy_type": "compensation",
                    "effectiveness": 0.1,
                },
                {
                    "content": "Napping during the day to catch up",
                    "strategy_type": "compensation",
                    "effectiveness": 0.2,
                },
                {
                    "content": "Avoiding bedtime to reduce frustration",
                    "strategy_type": "avoidance",
                    "effectiveness": 0.15,
                },
            ],
            "compensatory_strategies": [
                {
                    "content": "Excessive sleep hygiene rituals",
                    "behavior": "checking temperature, light, sound repeatedly before bed",
                    "overcompensation_for": "performance anxiety about sleeping",
                },
                {
                    "content": "Constantly checking the clock at night",
                    "behavior": "tracking time spent awake",
                    "overcompensation_for": "fear of not getting enough sleep",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Looking at the clock at 3 AM",
                    "interpretation": "This is going to be another horrible night",
                    "distortion_type": "fortune-telling",
                },
                {
                    "situation": "Feeling tired after poor sleep",
                    "interpretation": "My entire day is ruined now",
                    "distortion_type": "all-or-nothing",
                },
                {
                    "situation": "Going to bed without feeling sleepy",
                    "interpretation": "I will definitely not sleep tonight",
                    "distortion_type": "emotional reasoning",
                },
            ],
            "emotional_responses": [
                {"emotion": "frustration about sleep", "intensity": 0.85, "valence": "negative"},
                {"emotion": "anxiety at bedtime", "intensity": 0.8, "valence": "negative"},
                {"emotion": "despair after another sleepless night", "intensity": 0.78, "valence": "negative"},
                {"emotion": "irritability during the day", "intensity": 0.7, "valence": "negative"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Stays in bed awake for hours worrying",
                    "triggered_by": "inability to fall asleep",
                    "consequence": "Strengthens conditioned arousal",
                },
                {
                    "behavior": "Uses phone in bed to distract from frustration",
                    "triggered_by": "lying awake in bed",
                    "consequence": "Blue light worsens sleep onset",
                },
                {
                    "behavior": "Cancels daytime plans due to exhaustion",
                    "triggered_by": "poor night of sleep",
                    "consequence": "Social and occupational impairment",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.4, "world_views": 0.45, "future_views": 0.35},
            ],
        },
        linguistic_features={
            "hedging": 0.55,
            "negation_density": 0.35,
            "first_person_singular": 0.7,
            "cause_words": 0.6,
            "absolutist_words": 0.55,
        },
        severity_range=(0.2, 0.7),
        common_triggers=[
            "bedtime",
            "stressful events",
            "travel or schedule changes",
            "caffeine or alcohol",
        ],
        treatment_history=(
            "CBT-I is first-line treatment. Sleep restriction and stimulus control. Avoid chronic hypnotic use."
        ),
    )
)

# ── 19. Adult ADHD ──
_register(
    ClinicalProfile(
        name="adhd_adult",
        display_name="Attention-Deficit/Hyperactivity Disorder (Adult)",
        description="Persistent inattention, hyperactivity, and impulsivity interfering with adult functioning.",
        diagnoses=["F90.0", "F90.1", "F90.2"],
        typical_symptoms=[
            "difficulty sustaining attention",
            "distractibility",
            "organization challenges",
            "forgetfulness in daily activities",
            "impulsive decision-making",
            "restlessness or inner agitation",
        ],
        default_style=_F,
        ccd_config={
            "core_beliefs": [
                {"content": "I am lazy and disorganized by nature", "domain": "self", "conviction": 0.8},
                {"content": "I cannot trust myself to follow through", "domain": "self", "conviction": 0.78},
                {"content": "The neurotypical world was not made for me", "domain": "world", "conviction": 0.75},
            ],
            "intermediate_beliefs": [
                {"content": "I must try harder than everyone to be normal", "rule_type": "rule", "conviction": 0.78},
                {
                    "content": "If I forget something, it proves I am broken",
                    "rule_type": "attitude",
                    "conviction": 0.72,
                },
                {
                    "content": "Others are succeeding effortlessly while I struggle",
                    "rule_type": "assumption",
                    "conviction": 0.8,
                },
            ],
            "coping_strategies": [
                {
                    "content": "Multiple reminder systems and alarms",
                    "strategy_type": "compensation",
                    "effectiveness": 0.5,
                },
                {
                    "content": "Procrastination followed by hyperfocus",
                    "strategy_type": "compensation",
                    "effectiveness": 0.4,
                },
                {"content": "Avoiding boring tasks until crisis", "strategy_type": "avoidance", "effectiveness": 0.2},
            ],
            "compensatory_strategies": [
                {
                    "content": "Over-committing to deadlines to create urgency",
                    "behavior": "waiting until the last minute to generate adrenaline",
                    "overcompensation_for": "difficulty initiating without pressure",
                },
                {
                    "content": "Extreme task-switching throughout the day",
                    "behavior": "starting many tasks, finishing few",
                    "overcompensation_for": "boredom intolerance",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Receiving complex instructions",
                    "interpretation": "I will forget most of this by the time I walk away",
                    "distortion_type": "fortune-telling",
                },
                {
                    "situation": "Zoning out during conversation",
                    "interpretation": "I am rude and disrespectful for not listening",
                    "distortion_type": "labeling",
                },
                {
                    "situation": "Losing an important item",
                    "interpretation": "I cannot manage my own life",
                    "distortion_type": "overgeneralization",
                },
            ],
            "emotional_responses": [
                {"emotion": "frustration with inattention", "intensity": 0.78, "valence": "negative"},
                {"emotion": "intense interest when hyperfocused", "intensity": 0.9, "valence": "positive"},
                {"emotion": "shame about missed deadlines", "intensity": 0.75, "valence": "negative"},
                {"emotion": "overwhelm with multiple inputs", "intensity": 0.82, "valence": "negative"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Frequently checks phone during tasks",
                    "triggered_by": "boredom or low stimulation",
                    "consequence": "Cross-task interference and delays",
                },
                {
                    "behavior": "Loses important documents and items",
                    "triggered_by": "distraction during routine activities",
                    "consequence": "Practical life complications",
                },
                {
                    "behavior": "Interrupts others during conversation",
                    "triggered_by": "fear of forgetting the thought",
                    "consequence": "Social friction",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.45, "world_views": 0.55, "future_views": 0.5},
            ],
        },
        linguistic_features={
            "hedging": 0.4,
            "negation_density": 0.3,
            "first_person_singular": 0.7,
            "cause_words": 0.45,
            "absolutist_words": 0.45,
        },
        severity_range=(0.2, 0.7),
        common_triggers=[
            "boring or repetitive tasks",
            "multiple competing demands",
            "sitting still for extended periods",
            "late afternoon fatigue",
        ],
        treatment_history="Stimulant medication (methylphenidate or amphetamine) with CBT for executive function.",
    )
)

# ── 20. Autism Spectrum Disorder ──
_register(
    ClinicalProfile(
        name="autism_spectrum",
        display_name="Autism Spectrum Disorder",
        description="Persistent differences in social communication, restricted interests, and sensory sensitivities.",
        diagnoses=["F84.0"],
        typical_symptoms=[
            "difficulty with social reciprocity",
            "restricted or intense interests",
            "sensory hyper- or hypo-reactivity",
            "preference for routine and sameness",
            "literal interpretation of language",
            "difficulty understanding non-verbal cues",
        ],
        default_style=_N,
        ccd_config={
            "core_beliefs": [
                {"content": "I am fundamentally different from others", "domain": "self", "conviction": 0.88},
                {"content": "Social rules are confusing and arbitrary", "domain": "world", "conviction": 0.85},
                {"content": "Others often misunderstand my intentions", "domain": "others", "conviction": 0.8},
            ],
            "intermediate_beliefs": [
                {"content": "I must follow rules precisely and consistently", "rule_type": "rule", "conviction": 0.92},
                {"content": "Direct communication is always best", "rule_type": "attitude", "conviction": 0.88},
                {
                    "content": "If I do not understand social expectations, I will fail",
                    "rule_type": "assumption",
                    "conviction": 0.78,
                },
            ],
            "coping_strategies": [
                {"content": "Social masking and scripting", "strategy_type": "compensation", "effectiveness": 0.35},
                {"content": "Stimming for sensory regulation", "strategy_type": "compensation", "effectiveness": 0.7},
                {
                    "content": "Avoiding unfamiliar social situations",
                    "strategy_type": "avoidance",
                    "effectiveness": 0.4,
                },
            ],
            "compensatory_strategies": [
                {
                    "content": "Exhaustive routine planning",
                    "behavior": "scheduling every hour to prevent surprises",
                    "overcompensation_for": "discomfort with unpredictability",
                },
                {
                    "content": "Researching social scripts in advance",
                    "behavior": "reading about expected social behavior before events",
                    "overcompensation_for": "uncertainty about social norms",
                },
            ],
            "situation_interpretations": [
                {
                    "situation": "Sarcastic comment from colleague",
                    "interpretation": "I do not understand what they actually mean",
                    "distortion_type": "literal interpretation",
                },
                {
                    "situation": "Sudden change in meeting schedule",
                    "interpretation": "My carefully prepared plan is destroyed",
                    "distortion_type": "catastrophizing",
                },
                {
                    "situation": "Loud workplace environment",
                    "interpretation": "I cannot think or function with all this noise",
                    "distortion_type": "sensory overwhelm",
                },
            ],
            "emotional_responses": [
                {"emotion": "anxiety about change", "intensity": 0.85, "valence": "negative"},
                {"emotion": "joy during special interest engagement", "intensity": 0.92, "valence": "positive"},
                {"emotion": "overwhelm from sensory input", "intensity": 0.88, "valence": "negative"},
                {"emotion": "confusion in social situations", "intensity": 0.78, "valence": "negative"},
            ],
            "behavioral_responses": [
                {
                    "behavior": "Repeats routines exactly the same way",
                    "triggered_by": "daily transition points",
                    "consequence": "Provides stability but reduces flexibility",
                },
                {
                    "behavior": "Spends hours on special interest topics",
                    "triggered_by": "encountering an information gap in interest area",
                    "consequence": "Deep expertise but neglects other domains",
                },
                {
                    "behavior": "Needs downtime after social interaction",
                    "triggered_by": "prolonged social demands",
                    "consequence": "Limits social stamina",
                },
            ],
            "cognitive_triads": [
                {"self_views": 0.5, "world_views": 0.4, "future_views": 0.5},
            ],
        },
        linguistic_features={
            "hedging": 0.3,
            "negation_density": 0.25,
            "first_person_singular": 0.65,
            "cause_words": 0.55,
            "absolutist_words": 0.6,
        },
        severity_range=(0.2, 0.7),
        common_triggers=[
            "routine disruption",
            "sensory overload",
            "unexpected social demands",
            "miscommunication experiences",
        ],
        treatment_history=(
            "No pharmacological treatment for core symptoms. CBT for comorbid anxiety. Sensory accommodations."
        ),
    )
)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────


class ProfileRegistry:
    """Registry of all clinical profiles for PATIENT-Ψ simulation."""

    def __init__(self) -> None:
        self._profiles = _PROFILES

    def get_profile(self, name: str) -> ClinicalProfile:
        """Return the profile with *name*.

        Raises:
            KeyError: If *name* is not found.
        """
        if name not in self._profiles:
            msg = f"Unknown profile: {name!r}"
            raise KeyError(msg)
        return self._profiles[name]

    def list_profiles(self) -> list[str]:
        """Return sorted list of all registered profile names."""
        return sorted(self._profiles.keys())

    def get_profiles_by_diagnosis(self, diagnosis: str) -> list[ClinicalProfile]:
        """Return all profiles that include the given diagnosis code."""
        return [p for p in self._profiles.values() if diagnosis in p.diagnoses]

    def get_default_profile(self) -> ClinicalProfile:
        """Return the default profile (major depressive disorder)."""
        return self._profiles["major_depressive_disorder"]
