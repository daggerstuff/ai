"""
Memory Ingestion Configuration Module.

Provides therapeutic memory ingestion controls following the repository's
shared-memory best practices:
- Custom instructions for what to store/ignore
- Confidence thresholds for high-stakes therapeutic data
- PII filtering patterns for HIPAA compliance
- Inference mode configuration

The current implementation is local-service-first and does not depend on mem0.
"""

import re
from dataclasses import dataclass, field
from enum import StrEnum


class InferenceMode(StrEnum):
    """Memory inference modes from Foresight."""

    DEFAULT = "default"
    SPEED = "speed"
    QUALITY = "quality"


class MemoryCategory(StrEnum):
    """Categories for therapeutic memories."""

    THERAPEUTIC_INSIGHT = "therapeutic_insight"
    EMOTIONAL_STATE = "emotional_state"
    TREATMENT_PROGRESS = "treatment_progress"
    SESSION_SUMMARY = "session_summary"
    CRISIS_CONTEXT = "crisis_context"
    PREFERENCE = "preference"
    GENERAL = "general"


# Default PII patterns to never store (HIPAA-compliant)
DEFAULT_PII_PATTERNS: list[str] = [
    r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",  # SSN
    r"\b\d{9,18}\b",  # Insurance/credit card numbers
    r"\b\d{10,}\b",  # Long numeric identifiers
    r"\b[A-Z]{2}\d{6,8}\b",  # License/ID numbers
    r"\b\d{1,5}\s+[\w\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|court|ct)\.?\b",  # Addresses
    r"\b\d{5}(?:-\d{4})?\b",  # ZIP codes
    r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b",  # Phone numbers
]

# Default therapeutic custom instructions for Foresight
DEFAULT_THERAPEUTIC_INSTRUCTIONS = """
Therapeutic Memory Guidelines for Mental Health Context.

STORE (High Confidence):
- Confirmed emotional patterns and triggers
- Therapeutic progress and milestones achieved
- Verified coping strategies that work for the patient
- Established treatment goals and preferences
- Crisis indicators and safety planning details
- Learning style and communication preferences
- Session insights confirmed by both parties

STORE WITH CAUTION (Moderate Confidence):
- Self-reported emotional states (tag as self-reported)
- Mentioned life events affecting mental health
- Expressed goals and aspirations

NEVER STORE:
- Social Security Numbers or government IDs
- Insurance policy numbers or financial details
- Credit card or banking information
- Full addresses or location details beyond city/region
- Phone numbers or contact details
- Names of family members or third parties (anonymize as "family member", "partner", etc.)
- Speculative diagnoses not confirmed by professionals
- Casual mentions without therapeutic relevance

IGNORE (Do Not Extract):
- Speculation (words like "might", "maybe", "I think I could have")
- Unverified symptoms without professional assessment
- Gossip or information about third parties
- Technical small-talk unrelated to therapy
- Duplicate information already stored

SPECIAL HANDLING:
- Crisis signals: Always flag and store with high priority
- Medication mentions: Store only if confirmed as currently prescribed
- Childhood trauma: Store with sensitivity, require explicit disclosure consent
"""


@dataclass
class TherapeuticMemoryConfig:
    """
    Configuration for therapeutic memory ingestion.

    Attributes:
        custom_instructions: Instructions for Foresight on what to store/ignore
        confidence_threshold: Minimum confidence (0.0-1.0) for storing memories
        pii_patterns: Regex patterns for PII that should never be stored
        inference_mode: Speed vs quality tradeoff for memory extraction
        categories: Allowed memory categories
        enable_crisis_detection: Whether to flag crisis-related memories
        max_memory_length: Maximum characters per memory entry
    """

    custom_instructions: str = field(default=DEFAULT_THERAPEUTIC_INSTRUCTIONS)
    confidence_threshold: float = field(default=0.8)
    pii_patterns: list[str] = field(default_factory=DEFAULT_PII_PATTERNS.copy)
    inference_mode: InferenceMode = field(default=InferenceMode.QUALITY)
    categories: list[MemoryCategory] = field(default_factory=lambda: list(MemoryCategory))
    enable_crisis_detection: bool = field(default=True)
    max_memory_length: int = field(default=2000)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be 0.0-1.0, got {self.confidence_threshold}")
        if self.max_memory_length < 100:
            raise ValueError(f"max_memory_length must be >= 100, got {self.max_memory_length}")


class PIIFilter:
    """
    Filters PII from text before memory storage.

    Uses regex patterns to identify and redact sensitive information
    to maintain HIPAA compliance.
    """

    def __init__(self, patterns: list[str] | None = None):
        """
        Initialize PII filter with patterns.

        Args:
            patterns: List of regex patterns. Defaults to HIPAA-compliant patterns.
        """
        self.patterns = patterns or DEFAULT_PII_PATTERNS
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def contains_pii(self, text: str) -> bool:
        """
        Check if text contains any PII.

        Args:
            text: Text to check

        Returns:
            True if PII is detected, False otherwise
        """
        return any(pattern.search(text) for pattern in self._compiled_patterns)

    def redact_pii(self, text: str, replacement: str = "[REDACTED]") -> str:
        """
        Redact PII from text.

        Args:
            text: Text to redact
            replacement: String to replace PII with

        Returns:
            Text with PII replaced
        """
        result = text
        for pattern in self._compiled_patterns:
            result = pattern.sub(replacement, result)
        return result

    def filter_for_storage(self, text: str) -> str | None:
        """
        Filter text for safe memory storage.

        If PII is detected, redacts it. If text becomes meaningless after
        redaction (e.g., mostly redacted), returns None to prevent storage.

        Args:
            text: Text to filter

        Returns:
            Filtered text safe for storage, or None if should not be stored
        """
        if not text or not text.strip():
            return None

        redacted = self.redact_pii(text)

        # Check if too much was redacted (more than 50% redacted)
        redaction_count = redacted.count("[REDACTED]")
        if redaction_count > 0:
            original_words = len(text.split())
            if redaction_count / max(original_words, 1) > 0.5:
                return None

        return redacted


class SpeculationFilter:
    """
    Filters speculative statements from memory storage.

    Detects hedging language that indicates uncertainty, preventing
    speculation from being stored as facts.
    """

    SPECULATION_INDICATORS = [
        "i think",
        "i might",
        "maybe",
        "perhaps",
        "possibly",
        "could be",
        "might be",
        "not sure",
        "i guess",
        "i wonder",
        "i believe",
        "seems like",
        "feels like",
        "probably",
        "i suspect",
        "don't know if",
        "not certain",
    ]

    CONFIRMATION_INDICATORS = [
        "diagnosed",
        "confirmed",
        "doctor said",
        "therapist noted",
        "definitely",
        "certainly",
        "always",
        "documented",
        "prescribed",
        "verified",
    ]

    @classmethod
    def is_speculative(cls, text: str) -> bool:
        """
        Check if text is speculative.

        Args:
            text: Text to analyze

        Returns:
            True if text appears speculative, False if confirmed
        """
        text_lower = text.lower()

        # Check for confirmation indicators (overrides speculation)
        for indicator in cls.CONFIRMATION_INDICATORS:
            if indicator in text_lower:
                return False

        # Check for speculation indicators
        return any(indicator in text_lower for indicator in cls.SPECULATION_INDICATORS)

    @classmethod
    def get_confidence_adjustment(cls, text: str) -> float:
        """
        Get a confidence adjustment factor based on language certainty.

        Args:
            text: Text to analyze

        Returns:
            Float from 0.5 to 1.0 representing confidence adjustment
        """
        text_lower = text.lower()

        # High confidence for confirmed statements
        for indicator in cls.CONFIRMATION_INDICATORS:
            if indicator in text_lower:
                return 1.0

        # Low confidence for speculative statements
        speculation_count = sum(1 for ind in cls.SPECULATION_INDICATORS if ind in text_lower)

        if speculation_count >= 2:
            return 0.5
        if speculation_count == 1:
            return 0.7

        return 0.9  # Neutral statements


class CrisisDetector:
    """
    Detects crisis signals in therapeutic conversations.

    Flags messages that indicate potential crisis situations requiring
    immediate attention and special memory handling.
    """

    CRISIS_KEYWORDS = [
        "suicide",
        "suicidal",
        "kill myself",
        "end my life",
        "want to die",
        "don't want to live",
        "self-harm",
        "cutting",
        "hurting myself",
        "overdose",
        "no reason to live",
        "better off dead",
        "giving up",
        "can't go on",
        "ending it all",
        "goodbye forever",
        "final goodbye",
    ]

    CRISIS_PATTERNS = [
        r"(?:have|had|made)\s+(?:a\s+)?(?:plan|plans)\s+to\s+(?:kill|hurt|harm)",
        r"(?:wrote|writing)\s+(?:a\s+)?(?:note|letter|goodbye)",
        r"(?:gave|giving)\s+away\s+(?:my|all)\s+(?:things|possessions|stuff)",
    ]

    def __init__(self):
        """Initialize crisis detector with compiled patterns."""
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.CRISIS_PATTERNS]

    def detect_crisis(self, text: str) -> bool:
        """
        Detect if text contains crisis indicators.

        Args:
            text: Text to analyze

        Returns:
            True if crisis indicators detected
        """
        text_lower = text.lower()

        # Check keywords
        for keyword in self.CRISIS_KEYWORDS:
            if keyword in text_lower:
                return True

        # Check patterns
        return any(pattern.search(text) for pattern in self._compiled_patterns)

    def get_crisis_severity(self, text: str) -> str:
        """
        Get severity level of detected crisis.

        Args:
            text: Text to analyze

        Returns:
            Severity level: "none", "low", "medium", "high", "critical"
        """
        if not self.detect_crisis(text):
            return "none"

        text_lower = text.lower()
        high_severity_keywords = [
            "plan to",
            "going to",
            "will",
            "tonight",
            "today",
            "now",
        ]
        critical_keywords = ["goodbye", "final", "last", "never see"]

        # Check for critical indicators
        for keyword in critical_keywords:
            if keyword in text_lower:
                return "critical"

        # Check for high severity indicators
        for keyword in high_severity_keywords:
            if keyword in text_lower:
                return "high"

        # Check for action patterns
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                return "high"

        return "medium"
