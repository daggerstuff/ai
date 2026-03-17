import logging
from typing import Optional

from ai.memory.mem0_nvidia.memory_ingestion_config import (
    CrisisDetector,
    PIIFilter,
    SpeculationFilter,
    TherapeuticMemoryConfig,
)

logger = logging.getLogger("therapeutic_processor")


class TherapeuticProcessor:
    """
    Handles PII filtering, crisis detection, and speculation analysis
    for therapeutic memory interactions.
    """

    def __init__(self, config: Optional[TherapeuticMemoryConfig] = None):
        self.config = config or TherapeuticMemoryConfig()
        self.pii_filter = PIIFilter(self.config.pii_patterns)
        self.crisis_detector = CrisisDetector()

    def detect_crisis(self, text: str) -> str:
        """Analyze text for crisis signals."""
        return self.crisis_detector.get_crisis_severity(text)

    def filter_for_storage(self, content: str) -> Optional[str]:
        """
        Filter content before memory storage.
        Checks for PII and speculation.
        """
        # 1. PII Filtering
        filtered = self.pii_filter.filter_for_storage(content)
        if filtered is None:
            logger.debug("Content rejected: too much PII")
            return None

        # 2. Speculation Filtering
        if SpeculationFilter.is_speculative(filtered):
            confidence = SpeculationFilter.get_confidence_adjustment(filtered)
            if confidence < self.config.confidence_threshold:
                logger.debug(f"Content rejected: speculation confidence {confidence}")
                return None

        # 3. Size constraints
        if len(filtered) > self.config.max_memory_length:
            filtered = f"{filtered[: self.config.max_memory_length]}..."

        return filtered

    def build_system_prompt(
        self,
        base_instructions: str,
        memory_context: str,
        fixed_context: Optional[str] = None,
        crisis_severity: str = "none",
    ) -> str:
        """Assemble the system prompt with memory and safety instructions."""
        prompt = [
            base_instructions,
            "\n### CURRENT EMOTIONAL CARTOGRAPHY (RELEVANT MEMORIES)",
            (memory_context or "No prior emotional anchors for this specific context."),
        ]

        if fixed_context:
            prompt.append(f"\n### CLINICAL CONTEXT\n{fixed_context}")

        if crisis_severity != "none":
            prompt.append(
                f"\n!!! SAFETY ALERT !!!\n"
                f"Detected possible {crisis_severity} signal. "
                "Prioritize psychological safety, validation, and de-escalation."
            )

        return "\n".join(prompt)
