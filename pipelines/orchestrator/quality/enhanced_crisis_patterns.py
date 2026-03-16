#!/usr/bin/env python3
"""
Enhanced Crisis Detection Patterns
Improved patterns to achieve >95% accuracy for production safety validation.
"""

from dataclasses import dataclass
from enum import Enum


class CrisisType(Enum):
    """Types of crisis situations."""
    SUICIDE_IDEATION = "suicide_ideation"
    SELF_HARM = "self_harm"
    VIOLENCE_THREAT = "violence_threat"
    PSYCHOTIC_EPISODE = "psychotic_episode"
    SUBSTANCE_OVERDOSE = "substance_overdose"
    DOMESTIC_VIOLENCE = "domestic_violence"
    CHILD_ABUSE = "child_abuse"
    SEVERE_DEPRESSION = "severe_depression"
    PANIC_ATTACK = "panic_attack"
    EATING_DISORDER_CRISIS = "eating_disorder_crisis"

@dataclass
class CrisisIndicator:
    """Individual crisis indicator."""
    indicator_type: str
    severity_weight: float
    pattern: str
    context_required: bool = False
    immediate_flag: bool = False

class EnhancedCrisisPatterns:
    """Enhanced crisis detection patterns with improved accuracy."""

    def __init__(self):
        """Initialize enhanced patterns."""
        self.patterns = self._load_enhanced_patterns()

        # Improved thresholds for better clinical sensitivity
        self.config = {
            "emergency_threshold": 0.7,  # Lowered for better sensitivity
            "critical_threshold": 0.5,   # Catch more moderate-risk cases
            "elevated_threshold": 0.3,   # Better early warning system
            "response_time_target_minutes": 5,
            "mandatory_reporting_enabled": True,
            "emergency_services_contact": "911",
            "crisis_hotline": "988"
        }

    def _load_enhanced_patterns(self) -> dict[CrisisType, list[CrisisIndicator]]:
        """Load enhanced crisis detection patterns."""
        return {
            CrisisType.SUICIDE_IDEATION: [
                # Explicit intent (highest priority)
                CrisisIndicator("explicit_intent", 1.0, r"(?:want to|going to|plan to) (?:kill myself|end my life|commit suicide|take my own life)", immediate_flag=True),
                CrisisIndicator("direct_statement", 1.0, r"(?:i want to die|i'm going to kill myself|i don't want to be alive)", immediate_flag=True),

                # Method and planning
                CrisisIndicator("method_mention", 0.9, r"(?:pills|gun|rope|bridge|overdose|hanging|jumping|carbon monoxide|knife|razor)", context_required=True),
                CrisisIndicator("method_access", 0.95, r"(?:have a gun|got the pills|found a rope|know where|easy access to)", context_required=True),
                CrisisIndicator("plan_details", 0.9, r"(?:have a plan|know how|figured out how|written (?:a )?note|set a date)"),

                # Timeline and urgency
                CrisisIndicator("timeline_specific", 0.95, r"(?:tonight|today|tomorrow|this week|soon|right now|can't wait)", context_required=True),
                CrisisIndicator("imminent_action", 1.0, r"(?:doing it now|can't stop myself|it's time|this is it)", immediate_flag=True),

                # Passive ideation (CRITICAL - was missing)
                CrisisIndicator("passive_ideation", 0.6, r"(?:sometimes think about|wish i was|wouldn't mind if|better off dead|not being here)"),
                CrisisIndicator("death_wishes", 0.7, r"(?:wish i would die|hope i don't wake up|want to disappear|cease to exist)"),

                # Indirect expressions (CRITICAL - was missing)
                CrisisIndicator("indirect_burden", 0.8, r"(?:everyone would be better off without me|burden to everyone|world without me|they'd be happier)"),
                CrisisIndicator("indirect_planning", 0.8, r"(?:getting affairs in order|writing letters|giving things away|saying goodbye)"),
                CrisisIndicator("no_future", 0.7, r"(?:can't see a future|no tomorrow|no point in planning|nothing to live for)"),

                # Emotional indicators
                CrisisIndicator("hopelessness", 0.6, r"(?:no point|nothing left|can't go on|no hope|completely hopeless)"),
                CrisisIndicator("ideation", 0.7, r"(?:thoughts of|thinking about) (?:suicide|killing myself|dying|ending it)"),
                CrisisIndicator("goodbye_messages", 0.9, r"(?:goodbye|farewell|last time|won't see you again|sorry for everything)"),
                CrisisIndicator("giving_away", 0.8, r"(?:giving away|don't need|take my|want you to have)")
            ],

            CrisisType.SELF_HARM: [
                # Cutting behaviors
                CrisisIndicator("cutting_behavior", 0.8, r"(?:cutting|cut|razor|blade|self-harm|hurt myself|slice|carve)"),
                CrisisIndicator("cutting_escalation", 0.9, r"(?:deeper cuts|cutting more|worse cuts|can't stop cutting)"),
                CrisisIndicator("cutting_tools", 0.8, r"(?:razor blade|kitchen knife|broken glass|scissors|sharp object)"),

                # Burning (CRITICAL - was missing patterns)
                CrisisIndicator("burning_behavior", 0.8, r"(?:burn myself|burning my|cigarette burns|lighter burns)"),
                CrisisIndicator("burning_methods", 0.8, r"(?:cigarette|lighter|hot objects|matches|candle) (?:to burn|for burning|on my)"),
                CrisisIndicator("burning_escalation", 0.9, r"(?:bigger burns|more burns|hotter objects|longer burns)"),

                # Physical self-harm
                CrisisIndicator("hitting", 0.7, r"(?:hitting|punching|banging|slamming) (?:myself|my head|the wall)"),
                CrisisIndicator("other_harm", 0.7, r"(?:scratching|biting|pinching|pulling hair) (?:myself|until it bleeds)"),

                # Urges and compulsions
                CrisisIndicator("urges", 0.6, r"(?:urge to|want to|need to|compelled to) (?:cut|hurt|harm|injure) myself"),
                CrisisIndicator("compulsive_harm", 0.8, r"(?:can't stop|have to hurt|must cut|compelled to harm)"),

                # Tool acquisition and preparation
                CrisisIndicator("tools_mention", 0.7, r"(?:knife|scissors|glass|needle|pin) (?:to hurt|for cutting)"),
                CrisisIndicator("tool_acquisition", 0.8, r"(?:bought razors|hiding sharp objects|collecting knives|found something sharp)"),

                # Escalation patterns
                CrisisIndicator("escalation", 0.8, r"(?:deeper|harder|more|worse) (?:cuts|harm|pain|cutting)"),
                CrisisIndicator("frequency_increase", 0.8, r"(?:every day|multiple times|more often|can't go without)")
            ],

            CrisisType.VIOLENCE_THREAT: [
                CrisisIndicator("threat_explicit", 1.0, r"(?:going to|will|plan to) (?:kill|hurt|attack|shoot|stab) (?:him|her|them|you)", immediate_flag=True),
                CrisisIndicator("weapon_mention", 0.9, r"(?:gun|knife|weapon|baseball bat|hammer|axe)"),
                CrisisIndicator("violence_fantasy", 0.7, r"(?:imagine|fantasize about|dream of) (?:hurting|killing|attacking|beating)"),
                CrisisIndicator("anger_escalation", 0.6, r"(?:so angry|rage|furious|can't control|losing it)"),
                CrisisIndicator("specific_target", 0.8, r"(?:going to get|find|hunt down) (?:him|her|them|that person)"),
                CrisisIndicator("violence_planning", 0.9, r"(?:know where|waiting for|following|stalking)")
            ],

            CrisisType.PSYCHOTIC_EPISODE: [
                # Hallucinations
                CrisisIndicator("hallucinations", 0.8, r"(?:hearing voices|seeing things|voices telling me|shadows moving|people who aren't there)"),
                CrisisIndicator("visual_hallucinations", 0.8, r"(?:seeing things|shadows|people who aren't there|things moving|faces in walls)"),
                CrisisIndicator("auditory_hallucinations", 0.8, r"(?:hearing voices|voices in my head|someone talking|whispers|sounds)"),

                # Command hallucinations (CRITICAL - was missing)
                CrisisIndicator("command_voices", 0.95, r"(?:voices tell me to|commanded to|must obey|voices say (?:to|i should)|telling me to)", immediate_flag=True),
                CrisisIndicator("command_harm", 1.0, r"(?:voices tell me to hurt|command me to kill|must hurt|voices say hurt)", immediate_flag=True),

                # Delusions
                CrisisIndicator("delusions", 0.8, r"(?:they're watching|conspiracy|following me|reading my mind|controlling my thoughts)"),
                CrisisIndicator("paranoia", 0.7, r"(?:everyone is against me|they want to hurt me|can't trust anyone|being poisoned)"),
                CrisisIndicator("persecution", 0.8, r"(?:out to get me|plotting against|trying to harm|being hunted)"),

                # Disorganization
                CrisisIndicator("disorganized", 0.7, r"(?:can't think straight|thoughts are jumbled|mind is racing|confused|lost)"),
                CrisisIndicator("reality_loss", 0.8, r"(?:not real|dream|simulation|matrix|different dimension)"),
                CrisisIndicator("thought_disorder", 0.7, r"(?:thoughts scattered|can't focus|mind jumping|thoughts racing)")
            ],

            CrisisType.SUBSTANCE_OVERDOSE: [
                CrisisIndicator("overdose_explicit", 1.0, r"(?:overdosed|took too many|drank too much|took (?:the )?whole bottle)", immediate_flag=True),
                CrisisIndicator("substance_excess", 0.8, r"(?:whole bottle|all the pills|can't stop drinking)"),
                CrisisIndicator("medical_symptoms", 0.9, r"(?:can't breathe|chest pain|heart racing|passing out) (?:from|after|because) (?:pills|drugs|drinking)")
            ],

            CrisisType.DOMESTIC_VIOLENCE: [
                CrisisIndicator("physical_abuse", 0.9, r"(?:hit me|beat me|choked me|threw me|slapped|punched|kicked)"),
                CrisisIndicator("threat_from_partner", 0.8, r"(?:threatened to kill me|said he'd hurt me|going to get me|will find me)"),
                CrisisIndicator("fear_for_safety", 0.7, r"(?:afraid for my life|scared he'll|hiding from|safe house)"),
                CrisisIndicator("escalating_violence", 0.8, r"(?:getting worse|more violent|angrier|losing control)"),
                CrisisIndicator("isolation_control", 0.6, r"(?:won't let me|controls my|monitors my|isolated me)"),
                CrisisIndicator("children_danger", 0.9, r"(?:hurt the kids|children in danger|threatening my children)")
            ],

            CrisisType.CHILD_ABUSE: [
                CrisisIndicator("child_harm", 1.0, r"(?:hurt my child|hit my kid|abusing my)", immediate_flag=True),
                CrisisIndicator("child_neglect", 0.8, r"(?:can't take care|left alone|not feeding)"),
                CrisisIndicator("child_danger", 0.9, r"(?:child is in danger|unsafe for kids|might hurt them)")
            ],

            CrisisType.SEVERE_DEPRESSION: [
                CrisisIndicator("severe_hopelessness", 0.8, r"(?:completely hopeless|no reason to live|nothing matters anymore)"),
                CrisisIndicator("isolation_extreme", 0.7, r"(?:completely alone|nobody cares|isolated from everyone)"),
                CrisisIndicator("functioning_loss", 0.6, r"(?:can't get out of bed|stopped eating|can't function)"),
                CrisisIndicator("worthlessness", 0.7, r"(?:worthless|useless|burden to everyone|hate myself)"),
                CrisisIndicator("sleep_disruption", 0.5, r"(?:haven't slept|can't sleep|sleeping all day)")
            ],

            CrisisType.PANIC_ATTACK: [
                # Physical symptoms (CRITICAL - was completely missing)
                CrisisIndicator("panic_breathing", 0.8, r"(?:can'?t breathe|can'?t catch my breath|suffocating|gasping for air)"),
                CrisisIndicator("panic_heart", 0.8, r"(?:heart racing|heart pounding|chest pounding|heart beating fast)"),
                CrisisIndicator("panic_chest", 0.8, r"(?:chest tight|chest crushing|chest pain|pressure on chest)"),
                CrisisIndicator("panic_physical", 0.7, r"(?:dizzy|shaking|trembling|sweating|nauseous|tingling)"),

                # Cognitive symptoms (CRITICAL - was missing)
                CrisisIndicator("panic_death_fear", 0.8, r"(?:going to die|think i'?m dying|having a heart attack|this is it)"),
                CrisisIndicator("panic_control_fear", 0.8, r"(?:losing control|going crazy|losing my mind|can'?t control)"),
                CrisisIndicator("panic_escape", 0.7, r"(?:need to get out|can'?t escape|trapped|have to leave)"),

                # Intensity and behavioral symptoms
                CrisisIndicator("panic_intensity", 0.7, r"(?:panic attack|overwhelming fear|terror|can'?t calm down|worst feeling)"),
                CrisisIndicator("panic_behavioral", 0.6, r"(?:can'?t sit still|pacing|running away|hiding|need help)"),

                # Panic-specific language
                CrisisIndicator("panic_language", 0.7, r"(?:panic|panicking|freaking out|losing it|can'?t handle)")
            ],

            CrisisType.EATING_DISORDER_CRISIS: [
                # Medical emergency symptoms (CRITICAL - was completely missing)
                CrisisIndicator("ed_medical_emergency", 0.9, r"(?:fainting|dizzy|weak|heart problems|electrolyte|medical emergency)"),
                CrisisIndicator("ed_physical_symptoms", 0.8, r"(?:losing hair|teeth problems|can'?t stand|chest pain|irregular heartbeat)"),

                # Behavioral escalation (CRITICAL - was missing)
                CrisisIndicator("purging_escalation", 0.8, r"(?:purging multiple times|throwing up all day|can'?t stop purging|vomiting blood)"),
                CrisisIndicator("restriction_extreme", 0.8, r"(?:haven't eaten for days|refusing all food|starving myself|nothing for weeks)"),
                CrisisIndicator("binge_escalation", 0.7, r"(?:can'?t stop eating|ate everything|binge all day|out of control eating)"),

                # Psychological distress (CRITICAL - was missing)
                CrisisIndicator("ed_psychological", 0.7, r"(?:hate my body|disgusting|can'?t look in mirror|want to disappear)"),
                CrisisIndicator("ed_control_loss", 0.8, r"(?:can'?t control|completely out of control|food controls me|obsessed)"),

                # Traditional patterns (enhanced)
                CrisisIndicator("purging_behavior", 0.7, r"(?:throwing up|vomiting|laxatives|purging)"),
                CrisisIndicator("restriction_behavior", 0.6, r"(?:not eating|skipping meals|avoiding food)"),
                CrisisIndicator("binge_behavior", 0.6, r"(?:binge eating|eating too much|can'?t stop eating)"),
                CrisisIndicator("body_dysmorphia", 0.5, r"(?:so fat|too big|body image|weight obsession)")
            ]
        }

    def get_patterns(self) -> dict[CrisisType, list[CrisisIndicator]]:
        """Get enhanced crisis detection patterns."""
        return self.patterns

    def get_config(self) -> dict[str, any]:
        """Get enhanced configuration."""
        return self.config

# Test the enhanced patterns
if __name__ == "__main__":
    patterns = EnhancedCrisisPatterns()


    for _crisis_type, indicators in patterns.get_patterns().items():
        for indicator in indicators:
            flag = " [IMMEDIATE]" if indicator.immediate_flag else ""

    for _key, _value in patterns.get_config().items():
        pass
