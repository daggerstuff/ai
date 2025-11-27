"""
Safety Alignment Validator

Integrates toxic-dpo datasets for safety alignment training and validation.
Ensures therapeutic AI responses are safe, appropriate, and non-harmful.

Supported safety datasets:
- PJMixers/unalignment_toxic-dpo-v0.2-ShareGPT: Toxic input handling
- Custom safety rule patterns for mental health contexts

Safety Alignment Features:
- Crisis content detection
- Toxic language identification
- Harmful response prevention
- Professional boundary enforcement
- HIPAA-compliant response validation

Part of the Pixelated Empathy AI dataset pipeline.
"""

import logging
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# Handle imports
quality_path = Path(__file__).parent
pipeline_root = quality_path.parent.parent
sys.path.insert(0, str(pipeline_root))

try:
    from schemas.conversation_schema import Conversation, Message
except ImportError:
    try:
        from ai.dataset_pipeline.schemas.conversation_schema import Conversation, Message
    except ImportError:
        from conversation_schema import Conversation, Message

try:
    from logger import get_logger

    logger = get_logger("dataset_pipeline.safety_alignment")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class SafetyViolationType(Enum):
    """Types of safety violations in therapeutic conversations."""

    CRISIS_CONTENT = "crisis_content"
    TOXIC_LANGUAGE = "toxic_language"
    HARMFUL_ADVICE = "harmful_advice"
    PROFESSIONAL_BOUNDARY = "professional_boundary"
    PII_EXPOSURE = "pii_exposure"
    MEDICAL_MISINFORMATION = "medical_misinformation"
    DISCRIMINATION = "discrimination"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    SUICIDE_ENCOURAGEMENT = "suicide_encouragement"
    SELF_HARM_ENCOURAGEMENT = "self_harm_encouragement"


class SafetySeverity(Enum):
    """Severity levels for safety violations."""

    CRITICAL = 4  # Immediate action required, block response
    HIGH = 3  # Serious concern, flag for review
    MEDIUM = 2  # Moderate concern, add warning
    LOW = 1  # Minor concern, note for improvement
    NONE = 0  # No safety concerns


@dataclass
class SafetyRule:
    """A safety rule for content validation."""

    rule_id: str
    name: str
    violation_type: SafetyViolationType
    severity: SafetySeverity
    patterns: list[str]  # Regex patterns
    description: str
    required_action: str
    exceptions: list[str] = field(default_factory=list)


@dataclass
class SafetyViolation:
    """A detected safety violation."""

    rule_id: str
    violation_type: SafetyViolationType
    severity: SafetySeverity
    matched_content: str
    context: str
    message_index: int
    recommended_action: str


@dataclass
class SafetyAssessment:
    """Complete safety assessment for a conversation."""

    conversation_id: str
    is_safe: bool
    overall_severity: SafetySeverity
    violations: list[SafetyViolation]
    safety_score: float  # 0.0 (unsafe) to 1.0 (safe)
    requires_human_review: bool
    recommended_actions: list[str]
    safe_response_suggestion: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SafetyAlignmentValidator:
    """
    Comprehensive safety alignment validator for therapeutic conversations.

    Validates conversations against safety rules to ensure:
    - No harmful content or advice
    - Appropriate crisis handling
    - Professional therapeutic boundaries
    - HIPAA compliance considerations
    - Non-discriminatory language

    Integrates with DPO training for preference learning on safe vs. unsafe responses.
    """

    def __init__(
        self,
        safety_threshold: float = 0.85,
        enable_toxic_detection: bool = True,
        enable_crisis_detection: bool = True,
        enable_pii_detection: bool = True,
    ):
        """
        Initialize safety alignment validator.

        Args:
            safety_threshold: Minimum safety score for passing validation
            enable_toxic_detection: Enable toxic language detection
            enable_crisis_detection: Enable crisis content detection
            enable_pii_detection: Enable PII exposure detection
        """
        self.safety_threshold = safety_threshold
        self.enable_toxic_detection = enable_toxic_detection
        self.enable_crisis_detection = enable_crisis_detection
        self.enable_pii_detection = enable_pii_detection

        # Initialize safety rules
        self.rules = self._initialize_safety_rules()

        # Compile patterns
        self._compile_patterns()

        # Initialize safe response templates
        self._initialize_safe_responses()

        logger.info(
            f"SafetyAlignmentValidator initialized: "
            f"threshold={safety_threshold}, rules={len(self.rules)}"
        )

    def _initialize_safety_rules(self) -> list[SafetyRule]:
        """Initialize comprehensive safety rules."""
        rules = [
            # Crisis Content Rules
            SafetyRule(
                rule_id="CRISIS_001",
                name="Suicide ideation detection",
                violation_type=SafetyViolationType.CRISIS_CONTENT,
                severity=SafetySeverity.CRITICAL,
                patterns=[
                    r"\b(want(?:s|ed)?\s+to\s+(?:die|kill\s+(?:my)?self|end\s+(?:it|my\s+life)))\b",
                    r"\b(suicid(?:e|al)|take\s+my\s+(?:own\s+)?life)\b",
                    r"\b(no\s+reason\s+to\s+live|better\s+off\s+dead)\b",
                ],
                description="Detects expressions of suicidal ideation",
                required_action="Provide crisis resources, do not dismiss, assess safety",
            ),
            SafetyRule(
                rule_id="CRISIS_002",
                name="Self-harm detection",
                violation_type=SafetyViolationType.SELF_HARM_ENCOURAGEMENT,
                severity=SafetySeverity.CRITICAL,
                patterns=[
                    r"\b(cut(?:ting)?\s+(?:my)?self|self[- ]?harm|hurt\s+(?:my)?self)\b",
                    r"\b(burn(?:ing)?\s+(?:my)?self|starv(?:e|ing)\s+(?:my)?self)\b",
                ],
                description="Detects self-harm mentions or encouragement",
                required_action="Assess safety, provide resources, show concern",
            ),
            SafetyRule(
                rule_id="CRISIS_003",
                name="Homicidal ideation detection",
                violation_type=SafetyViolationType.CRISIS_CONTENT,
                severity=SafetySeverity.CRITICAL,
                patterns=[
                    r"\b(want\s+to\s+(?:kill|hurt|harm)\s+(?:someone|them|him|her))\b",
                    r"\b(homicid(?:e|al)|murder)\b",
                ],
                description="Detects expressions of violence toward others",
                required_action="Assess danger, duty to warn considerations",
            ),
            # Toxic Language Rules
            SafetyRule(
                rule_id="TOXIC_001",
                name="Slurs and hate speech",
                violation_type=SafetyViolationType.TOXIC_LANGUAGE,
                severity=SafetySeverity.HIGH,
                patterns=[
                    r"\b(retard(?:ed)?|psycho|crazy|insane|lunatic)\b",  # Mental health stigma
                ],
                description="Detects stigmatizing language about mental health",
                required_action="Replace with appropriate terminology",
                exceptions=["client quoting others", "discussing stigma"],
            ),
            SafetyRule(
                rule_id="TOXIC_002",
                name="Discriminatory language",
                violation_type=SafetyViolationType.DISCRIMINATION,
                severity=SafetySeverity.HIGH,
                patterns=[
                    r"\b(you\s+(?:people|types)|all\s+(?:men|women)\s+are)\b",
                    r"\b(because\s+(?:you're|your)\s+(?:a\s+)?(?:man|woman|black|white|gay))\b",
                ],
                description="Detects discriminatory generalizations",
                required_action="Remove discriminatory content, use inclusive language",
            ),
            # Harmful Advice Rules
            SafetyRule(
                rule_id="HARMFUL_001",
                name="Medical advice without qualification",
                violation_type=SafetyViolationType.MEDICAL_MISINFORMATION,
                severity=SafetySeverity.HIGH,
                patterns=[
                    r"\b(stop\s+(?:taking|your)\s+medication)\b",
                    r"\b(you\s+(?:don't\s+need|should\s+stop)\s+(?:therapy|treatment))\b",
                    r"\b((?:take|try)\s+(?:this|these)\s+(?:drug|pill|medication))\b",
                ],
                description="Detects potentially harmful medical advice",
                required_action="Remove medical advice, refer to healthcare provider",
            ),
            SafetyRule(
                rule_id="HARMFUL_002",
                name="Dismissive responses",
                violation_type=SafetyViolationType.HARMFUL_ADVICE,
                severity=SafetySeverity.MEDIUM,
                patterns=[
                    r"\b(just\s+(?:get\s+over|move\s+on|forget|stop\s+(?:thinking|feeling)))\b",
                    r"\b(it(?:'s)?\s+(?:not\s+that\s+bad|all\s+in\s+your\s+head))\b",
                    r"\b(you(?:'re)?\s+(?:just|being)\s+(?:dramatic|sensitive|weak))\b",
                ],
                description="Detects dismissive or invalidating responses",
                required_action="Replace with validating, empathetic response",
            ),
            # Professional Boundary Rules
            SafetyRule(
                rule_id="BOUNDARY_001",
                name="Romantic/sexual content",
                violation_type=SafetyViolationType.PROFESSIONAL_BOUNDARY,
                severity=SafetySeverity.CRITICAL,
                patterns=[
                    r"\b(i\s+(?:love|am\s+attracted\s+to)\s+you)\b",
                    r"\b((?:kiss|date|sleep\s+with)\s+(?:me|you))\b",
                    r"\b(sexy|romantic\s+(?:feelings|relationship))\b",
                ],
                description="Detects inappropriate romantic/sexual content",
                required_action="Maintain professional boundaries, redirect conversation",
            ),
            SafetyRule(
                rule_id="BOUNDARY_002",
                name="Personal information sharing",
                violation_type=SafetyViolationType.PROFESSIONAL_BOUNDARY,
                severity=SafetySeverity.MEDIUM,
                patterns=[
                    r"\b(my\s+(?:phone\s+number|address|social\s+media))\b",
                    r"\b(meet\s+(?:outside|in\s+person|privately))\b",
                ],
                description="Detects inappropriate personal info sharing",
                required_action="Remove personal information, maintain boundaries",
            ),
            # PII Exposure Rules
            SafetyRule(
                rule_id="PII_001",
                name="Social Security Number",
                violation_type=SafetyViolationType.PII_EXPOSURE,
                severity=SafetySeverity.CRITICAL,
                patterns=[
                    r"\b(\d{3}[- ]?\d{2}[- ]?\d{4})\b",  # SSN format
                ],
                description="Detects potential SSN exposure",
                required_action="Redact immediately, security review",
            ),
            SafetyRule(
                rule_id="PII_002",
                name="Phone number exposure",
                violation_type=SafetyViolationType.PII_EXPOSURE,
                severity=SafetySeverity.HIGH,
                patterns=[
                    r"\b(\+?1?[- ]?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4})\b",  # Phone format
                ],
                description="Detects phone number exposure",
                required_action="Redact phone number",
            ),
            SafetyRule(
                rule_id="PII_003",
                name="Email exposure",
                violation_type=SafetyViolationType.PII_EXPOSURE,
                severity=SafetySeverity.MEDIUM,
                patterns=[
                    r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b",
                ],
                description="Detects email address exposure",
                required_action="Redact or mask email address",
            ),
        ]

        return rules

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching."""
        self.compiled_rules: dict[str, list[re.Pattern]] = {}

        for rule in self.rules:
            patterns = [re.compile(p, re.IGNORECASE) for p in rule.patterns]
            self.compiled_rules[rule.rule_id] = patterns

    def _initialize_safe_responses(self) -> None:
        """Initialize templates for safe response suggestions."""
        self.safe_response_templates = {
            SafetyViolationType.CRISIS_CONTENT: (
                "I hear that you're going through an incredibly difficult time, and I'm concerned "
                "about your safety. Your feelings are valid, and you deserve support. "
                "If you're having thoughts of harming yourself, please reach out to the "
                "988 Suicide & Crisis Lifeline by calling or texting 988. They're available 24/7. "
                "I'm here to support you, and we can work through this together."
            ),
            SafetyViolationType.HARMFUL_ADVICE: (
                "I want to make sure I'm supporting you in the most helpful way. "
                "What you're experiencing sounds really challenging, and your feelings "
                "make sense given what you're going through. Let's explore this together."
            ),
            SafetyViolationType.PROFESSIONAL_BOUNDARY: (
                "I appreciate you sharing that with me. As your therapist, it's important "
                "that we maintain our professional relationship so I can best support your "
                "therapeutic goals. Let's focus on what brought you to therapy today."
            ),
        }

    def validate_conversation(self, conversation: Conversation) -> SafetyAssessment:
        """
        Validate a complete conversation for safety.

        Args:
            conversation: Conversation to validate

        Returns:
            SafetyAssessment with detailed results
        """
        violations = []
        recommended_actions = []

        # Check each message
        for idx, message in enumerate(conversation.messages):
            message_violations = self._check_message(message.content, idx)
            violations.extend(message_violations)

        # Calculate overall severity and safety score
        if not violations:
            overall_severity = SafetySeverity.NONE
            safety_score = 1.0
            is_safe = True
        else:
            max_severity = max(v.severity.value for v in violations)
            overall_severity = SafetySeverity(max_severity)

            # Calculate safety score (inverse of violation severity)
            severity_impact = sum(v.severity.value for v in violations) / (
                len(violations) * SafetySeverity.CRITICAL.value
            )
            safety_score = max(0.0, 1.0 - severity_impact)
            is_safe = safety_score >= self.safety_threshold

        # Compile recommended actions
        for violation in violations:
            if violation.recommended_action not in recommended_actions:
                recommended_actions.append(violation.recommended_action)

        # Generate safe response suggestion if needed
        safe_response = None
        if not is_safe and violations:
            primary_violation_type = violations[0].violation_type
            safe_response = self.safe_response_templates.get(primary_violation_type)

        requires_human_review = (
            overall_severity.value >= SafetySeverity.HIGH.value
            or any(
                v.violation_type in [SafetyViolationType.CRISIS_CONTENT, SafetyViolationType.PII_EXPOSURE]
                for v in violations
            )
        )

        return SafetyAssessment(
            conversation_id=conversation.conversation_id,
            is_safe=is_safe,
            overall_severity=overall_severity,
            violations=violations,
            safety_score=round(safety_score, 3),
            requires_human_review=requires_human_review,
            recommended_actions=recommended_actions,
            safe_response_suggestion=safe_response,
            metadata={
                "num_messages": len(conversation.messages),
                "num_violations": len(violations),
                "violation_types": list(set(v.violation_type.value for v in violations)),
            },
        )

    def _check_message(self, content: str, message_index: int) -> list[SafetyViolation]:
        """Check a single message for safety violations."""
        violations = []

        for rule in self.rules:
            # Skip disabled checks
            if not self.enable_toxic_detection and rule.violation_type == SafetyViolationType.TOXIC_LANGUAGE:
                continue
            if not self.enable_crisis_detection and rule.violation_type == SafetyViolationType.CRISIS_CONTENT:
                continue
            if not self.enable_pii_detection and rule.violation_type == SafetyViolationType.PII_EXPOSURE:
                continue

            patterns = self.compiled_rules[rule.rule_id]

            for pattern in patterns:
                matches = pattern.findall(content)
                if matches:
                    # Get context around match
                    for match in matches:
                        match_str = match if isinstance(match, str) else match[0]
                        context = self._get_context(content, match_str)

                        violation = SafetyViolation(
                            rule_id=rule.rule_id,
                            violation_type=rule.violation_type,
                            severity=rule.severity,
                            matched_content=match_str,
                            context=context,
                            message_index=message_index,
                            recommended_action=rule.required_action,
                        )
                        violations.append(violation)

        return violations

    def _get_context(self, content: str, match: str, context_chars: int = 50) -> str:
        """Get context around a match."""
        try:
            idx = content.lower().find(match.lower())
            if idx == -1:
                return content[:100]

            start = max(0, idx - context_chars)
            end = min(len(content), idx + len(match) + context_chars)
            context = content[start:end]

            if start > 0:
                context = "..." + context
            if end < len(content):
                context = context + "..."

            return context
        except Exception:
            return content[:100]

    def batch_validate(
        self, conversations: list[Conversation]
    ) -> list[SafetyAssessment]:
        """
        Validate multiple conversations.

        Args:
            conversations: List of conversations

        Returns:
            List of SafetyAssessment objects
        """
        assessments = []
        for conv in conversations:
            assessment = self.validate_conversation(conv)
            assessments.append(assessment)

        safe_count = sum(1 for a in assessments if a.is_safe)
        logger.info(
            f"Batch validation: {safe_count}/{len(assessments)} passed safety check"
        )

        return assessments

    def filter_safe_conversations(
        self, conversations: list[Conversation]
    ) -> tuple[list[Conversation], list[Conversation]]:
        """
        Filter conversations into safe and unsafe groups.

        Args:
            conversations: List of conversations to filter

        Returns:
            Tuple of (safe_conversations, unsafe_conversations)
        """
        safe = []
        unsafe = []

        for conv in conversations:
            assessment = self.validate_conversation(conv)
            if assessment.is_safe:
                safe.append(conv)
            else:
                unsafe.append(conv)

        logger.info(
            f"Filtered conversations: {len(safe)} safe, {len(unsafe)} unsafe"
        )

        return safe, unsafe

    def get_safety_statistics(
        self, assessments: list[SafetyAssessment]
    ) -> dict[str, Any]:
        """
        Calculate aggregate safety statistics.

        Args:
            assessments: List of safety assessments

        Returns:
            Dictionary with statistics
        """
        if not assessments:
            return {"error": "No assessments provided"}

        total = len(assessments)
        safe_count = sum(1 for a in assessments if a.is_safe)
        review_needed = sum(1 for a in assessments if a.requires_human_review)

        # Violation type distribution
        violation_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {}

        for assessment in assessments:
            for violation in assessment.violations:
                vtype = violation.violation_type.value
                violation_counts[vtype] = violation_counts.get(vtype, 0) + 1

                sev = violation.severity.name
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "total_conversations": total,
            "safe_count": safe_count,
            "unsafe_count": total - safe_count,
            "safety_rate": safe_count / total if total > 0 else 0,
            "average_safety_score": sum(a.safety_score for a in assessments) / total,
            "requires_human_review": review_needed,
            "violation_type_distribution": violation_counts,
            "severity_distribution": severity_counts,
        }

    def create_dpo_training_pair(
        self,
        conversation: Conversation,
        assessment: SafetyAssessment,
    ) -> Optional[dict[str, Any]]:
        """
        Create a DPO training pair from an unsafe conversation.

        The unsafe response becomes 'rejected' and a safe alternative
        becomes 'chosen'.

        Args:
            conversation: Original conversation
            assessment: Safety assessment

        Returns:
            DPO training pair dict or None if not applicable
        """
        if assessment.is_safe or not assessment.violations:
            return None

        # Find the problematic assistant response
        for violation in assessment.violations:
            msg_idx = violation.message_index
            if msg_idx < len(conversation.messages):
                msg = conversation.messages[msg_idx]
                if msg.role in ["assistant", "therapist"]:
                    # Get the prompt (previous user message)
                    prompt = ""
                    if msg_idx > 0:
                        prev_msg = conversation.messages[msg_idx - 1]
                        if prev_msg.role in ["user", "client"]:
                            prompt = prev_msg.content

                    return {
                        "prompt": prompt,
                        "rejected": msg.content,
                        "chosen": assessment.safe_response_suggestion
                        or self.safe_response_templates.get(
                            violation.violation_type,
                            "I understand this is difficult. Let me help you in a supportive way.",
                        ),
                        "violation_type": violation.violation_type.value,
                        "original_conversation_id": conversation.conversation_id,
                    }

        return None


# Integration with DPO loader
def create_safety_dpo_dataset(
    conversations: list[Conversation],
    validator: Optional[SafetyAlignmentValidator] = None,
) -> list[dict[str, Any]]:
    """
    Create DPO training dataset from conversations with safety violations.

    Args:
        conversations: List of conversations to process
        validator: SafetyAlignmentValidator instance

    Returns:
        List of DPO training pairs
    """
    validator = validator or SafetyAlignmentValidator()
    dpo_pairs = []

    for conv in conversations:
        assessment = validator.validate_conversation(conv)
        if not assessment.is_safe:
            pair = validator.create_dpo_training_pair(conv, assessment)
            if pair:
                dpo_pairs.append(pair)

    logger.info(f"Created {len(dpo_pairs)} DPO pairs from safety violations")
    return dpo_pairs


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Safety Alignment Validator")
    print("=" * 50)

    # Create validator
    validator = SafetyAlignmentValidator()

    print(f"\nInitialized with {len(validator.rules)} safety rules:")
    for rule in validator.rules[:5]:
        print(f"  - {rule.rule_id}: {rule.name} ({rule.severity.name})")
    print("  ...")

    # Test conversations
    test_conversations = [
        # Safe conversation
        Conversation(
            conversation_id="safe_001",
            source="test",
            messages=[
                Message(role="user", content="I've been feeling anxious about my presentation."),
                Message(
                    role="assistant",
                    content="I understand that presentations can be anxiety-provoking. "
                    "What specific aspects are causing you the most concern?",
                ),
            ],
        ),
        # Unsafe - crisis content
        Conversation(
            conversation_id="unsafe_001",
            source="test",
            messages=[
                Message(role="user", content="I don't want to live anymore."),
                Message(
                    role="assistant",
                    content="Just get over it, things aren't that bad.",  # Harmful response
                ),
            ],
        ),
    ]

    print("\n" + "=" * 50)
    print("Validating test conversations:")

    for conv in test_conversations:
        assessment = validator.validate_conversation(conv)
        print(f"\n  {conv.conversation_id}:")
        print(f"    Safe: {assessment.is_safe}")
        print(f"    Score: {assessment.safety_score}")
        print(f"    Severity: {assessment.overall_severity.name}")
        if assessment.violations:
            print(f"    Violations: {len(assessment.violations)}")
            for v in assessment.violations[:2]:
                print(f"      - {v.rule_id}: {v.violation_type.value}")

    # Get statistics
    assessments = validator.batch_validate(test_conversations)
    stats = validator.get_safety_statistics(assessments)

    print("\n" + "=" * 50)
    print("Safety Statistics:")
    print(f"  Total: {stats['total_conversations']}")
    print(f"  Safe: {stats['safe_count']}")
    print(f"  Safety Rate: {stats['safety_rate']:.1%}")

