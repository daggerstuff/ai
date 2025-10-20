"""
Crisis Detection System for Pixel LLM Training

Implements pattern-based crisis detection with multiple severity levels,
risk assessment, and response protocols integrated with safety gates.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CrisisSeverity(Enum):
    """Crisis severity levels"""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    IMMINENT = "imminent"


class CrisisType(Enum):
    """Types of crisis indicators"""

    SUICIDAL_IDEATION = "suicidal_ideation"
    SELF_HARM = "self_harm"
    PSYCHOSIS = "psychosis"
    AGITATION = "agitation"
    SUBSTANCE_ABUSE = "substance_abuse"
    SEVERE_DEPRESSION = "severe_depression"
    PANIC_ATTACK = "panic_attack"


@dataclass
class CrisisIndicator:
    """Represents a detected crisis indicator"""

    type: CrisisType
    severity: CrisisSeverity
    matches: list[str]
    confidence: float
    context: str = ""


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment result"""

    overall_severity: CrisisSeverity
    indicators: list[CrisisIndicator]
    protective_factors: list[str]
    timeline: str  # "immediate", "short_term", "ongoing"
    confidence_score: float
    requires_intervention: bool
    recommended_actions: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class CrisisDetector:
    """Pattern-based crisis detection system"""

    # Crisis indicator patterns
    CRISIS_PATTERNS = {
        CrisisType.SUICIDAL_IDEATION: {
            "high": [
                r"kill\s+myself",
                r"suicide",
                r"end\s+my\s+life",
                r"want\s+to\s+die",
                r"plan\s+to\s+die",
                r"ending\s+it\s+all",
                r"not\s+worth\s+living",
                r"better\s+off\s+dead",
                r"can't\s+go\s+on",
                r"no\s+reason\s+to\s+live",
            ],
            "moderate": [
                r"wish\s+i\s+was\s+dead",
                r"tired\s+of\s+living",
                r"what's\s+the\s+point",
                r"no\s+hope",
                r"give\s+up",
                r"can't\s+take\s+it",
                r"escape\s+this",
                r"don't\s+want\s+to\s+live",
                r"life\s+is\s+meaningless",
            ],
        },
        CrisisType.SELF_HARM: {
            "high": [
                r"cut\s+myself",
                r"cutting",
                r"hurt\s+myself",
                r"self\s+harm",
                r"self-harm",
                r"burning\s+myself",
                r"punish\s+myself",
                r"deserve\s+pain",
                r"slash\s+my\s+wrists",
                r"carving",
            ],
            "moderate": [
                r"hurting",
                r"pain",
                r"numb",
                r"feel\s+something",
                r"control",
                r"scratch",
                r"pick\s+at",
                r"bang\s+my\s+head",
            ],
        },
        CrisisType.PSYCHOSIS: {
            "high": [
                r"hearing\s+voices",
                r"voices\s+tell\s+me",
                r"seeing\s+things",
                r"not\s+real",
                r"paranoid",
                r"conspiracy",
                r"following\s+me",
                r"watching\s+me",
                r"command\s+voices",
                r"hallucination",
            ],
            "moderate": [r"strange\s+thoughts", r"confused", r"reality", r"delusion"],
        },
        CrisisType.AGITATION: {
            "high": [
                r"rage",
                r"furious",
                r"explosive",
                r"violent",
                r"hurt\s+someone",
                r"angry",
                r"losing\s+control",
                r"can't\s+sit\s+still",
                r"aggressive",
            ],
            "moderate": [r"frustrated", r"irritable", r"tense", r"restless"],
        },
        CrisisType.SUBSTANCE_ABUSE: {
            "high": [
                r"overdose",
                r"overdosing",
                r"too\s+much",
                r"poisoned",
                r"can't\s+stop\s+using",
                r"addiction\s+crisis",
            ],
            "moderate": [
                r"drunk",
                r"high",
                r"using",
                r"pills",
                r"drinking",
                r"drugs",
                r"substance",
                r"intoxicated",
                r"substance\s+use",
            ],
        },
        CrisisType.SEVERE_DEPRESSION: {
            "high": [
                r"completely\s+hopeless",
                r"nothing\s+matters",
                r"can't\s+function",
                r"total\s+despair",
                r"unbearable\s+pain",
            ],
            "moderate": [
                r"depressed",
                r"sad",
                r"empty",
                r"worthless",
                r"ashamed",
                r"guilty",
                r"failure",
                r"alone",
            ],
        },
        CrisisType.PANIC_ATTACK: {
            "high": [
                r"can't\s+breathe",
                r"chest\s+pain",
                r"dying",
                r"heart\s+attack",
                r"losing\s+control",
                r"going\s+crazy",
            ],
            "moderate": [r"panic", r"anxious", r"scared", r"terrified", r"overwhelmed"],
        },
    }

    # Timeline indicators
    TIMELINE_PATTERNS = {
        "immediate": [
            r"right\s+now",
            r"tonight",
            r"today",
            r"this\s+moment",
            r"immediately",
            r"can't\s+wait",
            r"going\s+to",
            r"about\s+to",
            r"in\s+the\s+next\s+hour",
        ],
        "short_term": [
            r"this\s+week",
            r"soon",
            r"tomorrow",
            r"next\s+few\s+days",
            r"planning",
            r"next\s+week",
            r"next\s+month",
        ],
    }

    # Protective factors
    PROTECTIVE_FACTORS = [
        r"family",
        r"friends",
        r"support",
        r"loved\s+ones",
        r"people\s+care",
        r"not\s+alone",
        r"therapist",
        r"counselor",
        r"help",
        r"support\s+group",
        r"coping",
        r"manage",
        r"strategies",
        r"meditation",
        r"breathing",
        r"exercise",
        r"therapy",
        r"skills",
        r"techniques",
        r"grounding",
        r"hope",
        r"future",
        r"goals",
        r"dreams",
        r"plans",
        r"better",
        r"improve",
        r"recovery",
        r"healing",
        r"progress",
        r"children",
        r"kids",
        r"pets",
        r"job",
        r"responsibility",
        r"needed",
        r"care\s+for",
    ]

    def __init__(self):
        """Initialize crisis detector"""
        self.compiled_patterns = self._compile_patterns()
        logger.info("CrisisDetector initialized")

    def _compile_patterns(self) -> dict:
        """Compile regex patterns for efficiency"""
        compiled = {}
        for crisis_type, severities in self.CRISIS_PATTERNS.items():
            compiled[crisis_type] = {}
            for severity, patterns in severities.items():
                compiled[crisis_type][severity] = [
                    re.compile(pattern, re.IGNORECASE) for pattern in patterns
                ]
        return compiled

    def detect_crisis(self, text: str) -> RiskAssessment:
        """
        Detect crisis indicators in text and assess risk level

        Args:
            text: Input text to analyze

        Returns:
            RiskAssessment with detected indicators and recommendations
        """
        if not text or not isinstance(text, str):
            return RiskAssessment(
                overall_severity=CrisisSeverity.NONE,
                indicators=[],
                protective_factors=[],
                timeline="ongoing",
                confidence_score=0.0,
                requires_intervention=False,
            )

        indicators = self._detect_indicators(text)
        protective_factors = self._detect_protective_factors(text)
        timeline = self._detect_timeline(text)

        # Calculate overall severity
        overall_severity = self._calculate_severity(indicators, timeline)
        confidence_score = self._calculate_confidence(indicators)
        requires_intervention = overall_severity in [CrisisSeverity.HIGH, CrisisSeverity.IMMINENT]

        # Generate recommendations
        recommended_actions = self._generate_actions(overall_severity, indicators, timeline)

        return RiskAssessment(
            overall_severity=overall_severity,
            indicators=indicators,
            protective_factors=protective_factors,
            timeline=timeline,
            confidence_score=confidence_score,
            requires_intervention=requires_intervention,
            recommended_actions=recommended_actions,
            metadata={
                "detected_at": datetime.now().isoformat(),
                "text_length": len(text),
                "indicator_count": len(indicators),
            },
        )

    def _detect_indicators(self, text: str) -> list[CrisisIndicator]:
        """Detect crisis indicators in text"""
        indicators = []
        text_lower = text.lower()

        for crisis_type, severities in self.compiled_patterns.items():
            for severity_str, patterns in severities.items():
                severity = (
                    CrisisSeverity.HIGH if severity_str == "high" else CrisisSeverity.MODERATE
                )
                matches = []

                for pattern in patterns:
                    found = pattern.findall(text_lower)
                    if found:
                        matches.extend(found)

                if matches:
                    confidence = min(len(matches) * 0.2 + 0.6, 1.0)
                    indicators.append(
                        CrisisIndicator(
                            type=crisis_type,
                            severity=severity,
                            matches=list(set(matches)),
                            confidence=confidence,
                            context=text[:200],
                        )
                    )

        return indicators

    def _detect_protective_factors(self, text: str) -> list[str]:
        """Detect protective factors in text"""
        factors = []
        text_lower = text.lower()

        for pattern_str in self.PROTECTIVE_FACTORS:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(text_lower):
                factors.append(pattern_str.replace(r"\s+", " "))

        return list(set(factors))

    def _detect_timeline(self, text: str) -> str:
        """Detect crisis timeline"""
        text_lower = text.lower()

        for pattern_str in self.TIMELINE_PATTERNS["immediate"]:
            if re.search(pattern_str, text_lower, re.IGNORECASE):
                return "immediate"

        for pattern_str in self.TIMELINE_PATTERNS["short_term"]:
            if re.search(pattern_str, text_lower, re.IGNORECASE):
                return "short_term"

        return "ongoing"

    def _calculate_severity(
        self, indicators: list[CrisisIndicator], timeline: str = "ongoing"
    ) -> CrisisSeverity:
        """Calculate overall crisis severity"""
        if not indicators:
            return CrisisSeverity.NONE

        high_count = sum(1 for i in indicators if i.severity == CrisisSeverity.HIGH)
        moderate_count = sum(1 for i in indicators if i.severity == CrisisSeverity.MODERATE)

        # Escalate to IMMINENT if immediate timeline with high indicators
        if timeline == "immediate" and high_count >= 1:
            return CrisisSeverity.IMMINENT

        if high_count >= 2:
            return CrisisSeverity.IMMINENT
        if high_count >= 1 or moderate_count >= 2:
            return CrisisSeverity.HIGH
        if moderate_count >= 1:
            return CrisisSeverity.MODERATE
        return CrisisSeverity.LOW

    def _calculate_confidence(self, indicators: list[CrisisIndicator]) -> float:
        """Calculate confidence score"""
        if not indicators:
            return 0.0
        return sum(i.confidence for i in indicators) / len(indicators)

    def _generate_actions(
        self, severity: CrisisSeverity, indicators: list[CrisisIndicator], timeline: str
    ) -> list[str]:
        """Generate recommended actions based on assessment"""
        actions = []

        if severity == CrisisSeverity.IMMINENT:
            actions.extend(
                [
                    "IMMEDIATE: Contact emergency services (911)",
                    "Activate crisis hotline (988)",
                    "Notify emergency contacts",
                    "Implement safety plan immediately",
                ]
            )
        elif severity == CrisisSeverity.HIGH:
            actions.extend(
                [
                    "Contact crisis counselor within 1 hour",
                    "Implement safety plan",
                    "Increase monitoring frequency",
                    "Schedule emergency appointment",
                ]
            )
        elif severity == CrisisSeverity.MODERATE:
            actions.extend(
                [
                    "Schedule therapy appointment within 24-48 hours",
                    "Increase self-monitoring",
                    "Activate coping strategies",
                    "Reach out to support network",
                ]
            )

        if timeline == "immediate":
            actions.insert(0, "URGENT: Immediate intervention required")

        return actions
