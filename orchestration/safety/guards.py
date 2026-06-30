import re
from typing import Any

# Centralized PHI patterns for reuse across guards
PHI_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b\d{3}-\d{3}-\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    # Add more as needed
]

def _apply_phi_scrubbing(text: str) -> str:
    """Apply PHI scrubbing patterns to the given text."""
    sanitized = text
    for pattern, replacement in PHI_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized

class SafetyGuardResult:
    def __init__(self, passed: bool, sanitized_text: str, message: str | None = None, metadata: dict[str, Any] | None = None):
        self.passed = passed
        self.sanitized_text = sanitized_text
        self.message = message
        self.metadata = metadata or {}

class InputGuard:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def run(self, user_input: str) -> SafetyGuardResult:
        sanitized = _apply_phi_scrubbing(user_input)

        # Simple intent recognition
        intent = self._detect_intent(sanitized)

        passed = True # For now, always pass unless critical PHI found that can't be sanitized?

        return SafetyGuardResult(
            passed=passed,
            sanitized_text=sanitized,
            metadata={"intent": intent}
        )

    def _detect_intent(self, text: str) -> str:
        text = text.lower()
        if any(kw in text for kw in ["history", "past", "previously", "ever had"]):
            return "ask_history"
        if any(kw in text for kw in ["exam", "physical", "look at", "check your"]):
            return "perform_exam"
        if any(kw in text for kw in ["give", "medication", "dose", "treat", "intervention"]):
            return "perform_intervention"
        if any(kw in text for kw in ["pain", "hurt", "discomfort"]):
            return "address_pain"
        return "general_talk"

class OutputGuard:
    def __init__(self, persona_definition: dict[str, Any], config: dict[str, Any] | None = None):
        self.persona_definition = persona_definition
        self.config = config or {}

    def run(self, llm_output: str, current_state: str) -> SafetyGuardResult:
        # 1. PHI Scrubbing (last check)
        sanitized = _apply_phi_scrubbing(llm_output)

        # 2. Persona Alignment
        alignment_passed, alignment_msg = self._check_persona_alignment(llm_output)

        # 3. Medical Accuracy (Simulated)
        accuracy_passed, accuracy_msg = self._check_medical_accuracy(llm_output, current_state)

        passed = alignment_passed and accuracy_passed
        message = None
        if not passed:
            message = f"Alignment: {alignment_msg}. Accuracy: {accuracy_msg}"

        return SafetyGuardResult(
            passed=passed,
            sanitized_text=sanitized,
            message=message,
            metadata={
                "alignment_passed": alignment_passed,
                "accuracy_passed": accuracy_passed
            }
        )

    def _check_persona_alignment(self, text: str) -> tuple[bool, str]:
        # Basic check for tone/verbosity if possible
        # e.g. if tone is "scared", and text is too formal
        return True, "Passed"

    def _check_medical_accuracy(self, text: str, state: str) -> tuple[bool, str]:
        # Simulated check
        # e.g. "I have no pain" when state is ESCALATION
        if state == "escalation" and "no pain" in text.lower():
            return False, "Output contradicts escalation state (patient should be in pain)"
        return True, "Passed"
