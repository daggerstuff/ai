"""
Bias Detection & Mitigation System for Pixel LLM Training

Implements demographic bias detection, fairness metrics, and mitigation strategies
for mental health conversations across diverse populations.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class BiasType(Enum):
    """Types of demographic bias"""

    GENDER_BIAS = "gender_bias"
    RACIAL_BIAS = "racial_bias"
    AGE_BIAS = "age_bias"
    CULTURAL_BIAS = "cultural_bias"
    SOCIOECONOMIC_BIAS = "socioeconomic_bias"
    ABILITY_BIAS = "ability_bias"
    LANGUAGE_BIAS = "language_bias"


class AlertLevel(Enum):
    """Bias alert severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BiasIndicator:
    """Represents a detected bias indicator"""

    type: BiasType
    severity: AlertLevel
    pattern: str
    matches: list[str]
    confidence: float
    context: str = ""


@dataclass
class FairnessMetrics:
    """Fairness metrics for bias assessment"""

    demographic_parity: float  # 0.0-1.0 (1.0 = perfect parity)
    equalized_odds: float  # 0.0-1.0 (1.0 = equal odds)
    calibration: float  # 0.0-1.0 (1.0 = perfectly calibrated)
    representation_balance: float  # 0.0-1.0 (1.0 = balanced)


@dataclass
class BiasAssessment:
    """Comprehensive bias assessment result"""

    overall_bias_score: float  # 0.0-1.0
    alert_level: AlertLevel
    indicators: list[BiasIndicator]
    fairness_metrics: FairnessMetrics
    confidence_score: float
    requires_mitigation: bool
    mitigation_strategies: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BiasDetector:
    """Demographic bias detection system"""

    # Bias patterns by type
    BIAS_PATTERNS = {
        BiasType.GENDER_BIAS: {
            "high": [
                r"(?:he|she)\s+(?:is|was|will|can|should)",
                r"(?:man|woman)(?:ly|hood)",
                r"(?:boy|girl)s?\s+(?:are|will)",
                r"(?:strong|weak)\s+(?:man|woman)",
                r"(?:emotional|rational)\s+(?:man|woman)",
                r"too\s+(?:emotional|weak|strong)\s+for",
            ],
            "medium": [
                r"(?:typical|normal)\s+(?:man|woman)",
                r"(?:mothering|fathering)",
                r"(?:masculine|feminine)\s+(?:traits|qualities)",
            ],
        },
        BiasType.RACIAL_BIAS: {
            "high": [
                r"(?:black|white|asian|hispanic|latino)\s+(?:people|person|community)",
                r"(?:stereotype|stereotypical)",
                r"(?:articulate|well-spoken)\s+(?:for|despite)",
                r"(?:urban|suburban|rural)\s+(?:community|area)",
            ],
            "medium": [
                r"(?:diverse|multicultural)\s+(?:background|community)",
                r"(?:minority|underrepresented)\s+(?:group|population)",
            ],
        },
        BiasType.AGE_BIAS: {
            "high": [
                r"(?:old|young|elderly|senior)\s+(?:people|person)",
                r"(?:tech-savvy|out-of-touch)",
                r"(?:millennial|boomer|gen\s+z)\s+(?:are|will)",
                r"(?:too\s+)?(?:old|young)\s+(?:to|for)",
                r"(?:lazy|stupid|useless)",
            ],
            "medium": [
                r"(?:age-appropriate|age-related)",
                r"(?:generational|age)\s+(?:differences|gaps)",
            ],
        },
        BiasType.CULTURAL_BIAS: {
            "high": [
                r"(?:western|eastern|american|foreign)\s+(?:values|culture)",
                r"(?:traditional|modern)\s+(?:family|values)",
                r"(?:collectivist|individualist)",
                r"(?:primitive|civilized)\s+(?:culture|society)",
                r"superior|inferior",
            ],
            "medium": [
                r"(?:cultural|religious)\s+(?:practices|beliefs)",
                r"(?:different|unusual)\s+(?:customs|traditions)",
            ],
        },
        BiasType.SOCIOECONOMIC_BIAS: {
            "high": [
                r"(?:poor|rich|wealthy|underprivileged)\s+(?:people|person)",
                r"(?:lower|upper)\s+(?:class|income)",
                r"(?:uneducated|educated)\s+(?:people|person)",
                r"(?:lazy|hardworking)\s+(?:poor|rich)",
                r"(?:poor|rich)\s+(?:are|is)",
            ],
            "medium": [
                r"(?:financial|economic)\s+(?:status|situation)",
                r"(?:socioeconomic|income)\s+(?:level|bracket)",
            ],
        },
        BiasType.ABILITY_BIAS: {
            "high": [
                r"(?:disabled|handicapped|crippled)\s+(?:people|person)",
                r"(?:normal|abnormal)\s+(?:ability|function)",
                r"(?:mentally|physically)\s+(?:challenged|disabled)",
                r"inspiration\s+porn|inspiring.*despite",
            ],
            "medium": [
                r"(?:accessibility|accommodation)\s+(?:needs|requirements)",
                r"(?:disability|impairment)\s+(?:status|condition)",
            ],
        },
        BiasType.LANGUAGE_BIAS: {
            "high": [
                r"(?:accent|dialect)\s+(?:is|was)\s+(?:bad|wrong|inferior)",
                r"(?:english|native)\s+(?:speaker|language)",
                r"(?:articulate|intelligent)\s+(?:for|despite)",
                r"(?:broken|poor)\s+(?:english|language)",
            ],
            "medium": [
                r"(?:multilingual|bilingual)\s+(?:background|ability)",
                r"(?:language|linguistic)\s+(?:barrier|difference)",
            ],
        },
    }

    # Demographic representation keywords
    DEMOGRAPHIC_KEYWORDS = {
        "gender": ["man", "woman", "male", "female", "boy", "girl", "he", "she"],
        "race": ["black", "white", "asian", "hispanic", "latino", "african", "caucasian"],
        "age": ["young", "old", "elderly", "senior", "millennial", "boomer", "gen z"],
        "culture": ["american", "asian", "european", "african", "western", "eastern"],
        "socioeconomic": ["poor", "rich", "wealthy", "underprivileged", "homeless", "affluent"],
        "ability": ["disabled", "blind", "deaf", "wheelchair", "mental", "physical"],
    }

    def __init__(self):
        """Initialize bias detector"""
        self.compiled_patterns = self._compile_patterns()
        logger.info("BiasDetector initialized")

    def _compile_patterns(self) -> dict:
        """Compile regex patterns for efficiency"""
        compiled = {}
        for bias_type, severities in self.BIAS_PATTERNS.items():
            compiled[bias_type] = {}
            for severity, patterns in severities.items():
                compiled[bias_type][severity] = [
                    re.compile(pattern, re.IGNORECASE) for pattern in patterns
                ]
        return compiled

    def detect_bias(self, text: str, demographics: dict | None = None) -> BiasAssessment:
        """
        Detect bias in text and assess fairness

        Args:
            text: Input text to analyze
            demographics: Optional demographic context (age, gender, race, etc.)

        Returns:
            BiasAssessment with detected biases and mitigation strategies
        """
        if not text or not isinstance(text, str):
            return BiasAssessment(
                overall_bias_score=0.0,
                alert_level=AlertLevel.LOW,
                indicators=[],
                fairness_metrics=FairnessMetrics(1.0, 1.0, 1.0, 1.0),
                confidence_score=0.0,
                requires_mitigation=False,
            )

        indicators = self._detect_indicators(text)
        fairness_metrics = self._calculate_fairness_metrics(text, demographics)
        overall_bias_score = self._calculate_bias_score(indicators, fairness_metrics)
        alert_level = self._determine_alert_level(overall_bias_score)
        confidence_score = self._calculate_confidence(indicators)
        requires_mitigation = alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]

        mitigation_strategies = self._generate_mitigation_strategies(
            indicators, alert_level, demographics
        )

        return BiasAssessment(
            overall_bias_score=overall_bias_score,
            alert_level=alert_level,
            indicators=indicators,
            fairness_metrics=fairness_metrics,
            confidence_score=confidence_score,
            requires_mitigation=requires_mitigation,
            mitigation_strategies=mitigation_strategies,
            metadata={
                "detected_at": datetime.now().isoformat(),
                "text_length": len(text),
                "indicator_count": len(indicators),
                "demographics_provided": demographics is not None,
            },
        )

    def _detect_indicators(self, text: str) -> list[BiasIndicator]:
        """Detect bias indicators in text"""
        indicators = []
        text_lower = text.lower()

        for bias_type, severities in self.compiled_patterns.items():
            for severity_str, patterns in severities.items():
                severity = AlertLevel.HIGH if severity_str == "high" else AlertLevel.MEDIUM
                matches = []
                matched_patterns = []

                for pattern in patterns:
                    found = pattern.findall(text_lower)
                    if found:
                        matches.extend(found)
                        matched_patterns.append(pattern)

                if matches:
                    confidence = min(len(matches) * 0.15 + 0.5, 1.0)
                    indicators.append(
                        BiasIndicator(
                            type=bias_type,
                            severity=severity,
                            pattern=matched_patterns[0].pattern if matched_patterns else "",
                            matches=list(set(matches)),
                            confidence=confidence,
                            context=text[:200],
                        )
                    )

        # Add keyword-based fallback detection for common bias patterns
        indicators.extend(self._detect_keyword_bias(text_lower))

        return indicators

    def _detect_keyword_bias(self, text_lower: str) -> list[BiasIndicator]:
        """Fallback keyword-based bias detection"""
        indicators = []

        # Gender bias keywords
        if any(kw in text_lower for kw in ["too emotional", "too weak", "too strong for"]):
            if "woman" in text_lower or "women" in text_lower or "female" in text_lower:
                indicators.append(
                    BiasIndicator(
                        type=BiasType.GENDER_BIAS,
                        severity=AlertLevel.HIGH,
                        pattern="gender_stereotype_keywords",
                        matches=["gender_stereotype"],
                        confidence=0.8,
                        context=text_lower[:200],
                    )
                )

        # Age bias keywords
        if any(kw in text_lower for kw in ["lazy", "stupid", "useless"]):
            if any(kw in text_lower for kw in ["old", "elderly", "senior", "millennial", "boomer"]):
                indicators.append(
                    BiasIndicator(
                        type=BiasType.AGE_BIAS,
                        severity=AlertLevel.HIGH,
                        pattern="age_stereotype_keywords",
                        matches=["age_stereotype"],
                        confidence=0.8,
                        context=text_lower[:200],
                    )
                )

        # Socioeconomic bias keywords
        if any(kw in text_lower for kw in ["poor", "rich", "wealthy", "underprivileged"]):
            if any(kw in text_lower for kw in ["lazy", "hardworking", "intelligent", "stupid"]):
                indicators.append(
                    BiasIndicator(
                        type=BiasType.SOCIOECONOMIC_BIAS,
                        severity=AlertLevel.HIGH,
                        pattern="socioeconomic_stereotype_keywords",
                        matches=["socioeconomic_stereotype"],
                        confidence=0.8,
                        context=text_lower[:200],
                    )
                )

        return indicators

    def _calculate_fairness_metrics(self, text: str, demographics: dict | None) -> FairnessMetrics:
        """Calculate fairness metrics"""
        text_lower = text.lower()

        # Count demographic mentions
        demographic_counts = {}
        for category, keywords in self.DEMOGRAPHIC_KEYWORDS.items():
            count = sum(text_lower.count(kw) for kw in keywords)
            demographic_counts[category] = count

        # Calculate representation balance (0.0-1.0, 1.0 = balanced)
        total_mentions = sum(demographic_counts.values())
        if total_mentions == 0:
            representation_balance = 1.0
        else:
            avg_mentions = total_mentions / len(demographic_counts)
            max_deviation = max(abs(count - avg_mentions) for count in demographic_counts.values())
            representation_balance = max(0.0, 1.0 - (max_deviation / (avg_mentions + 1)))

        # Demographic parity (presence of diverse mentions)
        diverse_categories = sum(1 for count in demographic_counts.values() if count > 0)
        demographic_parity = diverse_categories / len(demographic_counts)

        # Equalized odds (balanced treatment across groups)
        equalized_odds = 1.0 - (len([i for i in self._detect_indicators(text)]) * 0.1)
        equalized_odds = max(0.0, min(1.0, equalized_odds))

        # Calibration (consistency in treatment)
        calibration = 1.0 - (
            len([i for i in self._detect_indicators(text) if i.severity == AlertLevel.HIGH]) * 0.15
        )
        calibration = max(0.0, min(1.0, calibration))

        return FairnessMetrics(
            demographic_parity=demographic_parity,
            equalized_odds=equalized_odds,
            calibration=calibration,
            representation_balance=representation_balance,
        )

    def _calculate_bias_score(
        self, indicators: list[BiasIndicator], metrics: FairnessMetrics
    ) -> float:
        """Calculate overall bias score"""
        if not indicators:
            return 0.0

        # Weight indicators by severity
        indicator_score = sum(
            0.8 if i.severity == AlertLevel.HIGH else 0.4 for i in indicators
        ) / max(len(indicators), 1)

        # Weight fairness metrics
        fairness_score = 1.0 - (
            (1.0 - metrics.demographic_parity) * 0.25
            + (1.0 - metrics.equalized_odds) * 0.25
            + (1.0 - metrics.calibration) * 0.25
            + (1.0 - metrics.representation_balance) * 0.25
        )

        # Combine scores
        overall_score = (indicator_score * 0.6) + ((1.0 - fairness_score) * 0.4)
        return min(1.0, max(0.0, overall_score))

    def _determine_alert_level(self, bias_score: float) -> AlertLevel:
        """Determine alert level based on bias score"""
        if bias_score >= 0.7:
            return AlertLevel.CRITICAL
        if bias_score >= 0.5:
            return AlertLevel.HIGH
        if bias_score >= 0.3:
            return AlertLevel.MEDIUM
        return AlertLevel.LOW

    def _calculate_confidence(self, indicators: list[BiasIndicator]) -> float:
        """Calculate confidence score"""
        if not indicators:
            return 0.0
        return sum(i.confidence for i in indicators) / len(indicators)

    def _generate_mitigation_strategies(
        self, indicators: list[BiasIndicator], alert_level: AlertLevel, demographics: dict | None
    ) -> list[str]:
        """Generate mitigation strategies"""
        strategies = []

        if alert_level == AlertLevel.CRITICAL:
            strategies.extend(
                [
                    "URGENT: Immediate review and revision required",
                    "Consult with diversity and inclusion specialist",
                    "Implement comprehensive bias training",
                    "Establish oversight committee for content review",
                ]
            )
        elif alert_level == AlertLevel.HIGH:
            strategies.extend(
                [
                    "Schedule bias review meeting",
                    "Revise content to remove biased language",
                    "Provide cultural competency training",
                    "Implement peer review process",
                ]
            )
        elif alert_level == AlertLevel.MEDIUM:
            strategies.extend(
                [
                    "Monitor for patterns in future content",
                    "Provide targeted bias awareness training",
                    "Review and update guidelines",
                    "Increase diversity in review team",
                ]
            )

        # Type-specific strategies
        for indicator in indicators:
            if indicator.type == BiasType.GENDER_BIAS:
                strategies.append("Use gender-neutral language and pronouns")
            elif indicator.type == BiasType.RACIAL_BIAS:
                strategies.append("Avoid racial stereotypes and generalizations")
            elif indicator.type == BiasType.AGE_BIAS:
                strategies.append("Recognize diverse capabilities across age groups")
            elif indicator.type == BiasType.CULTURAL_BIAS:
                strategies.append("Respect diverse cultural values and practices")
            elif indicator.type == BiasType.SOCIOECONOMIC_BIAS:
                strategies.append("Avoid assumptions about economic status")
            elif indicator.type == BiasType.ABILITY_BIAS:
                strategies.append("Use person-first or identity-first language appropriately")
            elif indicator.type == BiasType.LANGUAGE_BIAS:
                strategies.append("Respect linguistic diversity and non-native speakers")

        return list(set(strategies))  # Remove duplicates
