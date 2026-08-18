from __future__ import annotations

"""
Therapeutic interaction service for NVIDIA-backed agents using local memory.
"""

import logging

from ai.memory.local_foresight_manager import LocalForesightMemoryManager

from .memory_ingestion_config import MemoryCategory

logger = logging.getLogger("foresight_nvidia.interactions")


class NvidiaTherapeuticInteractionService:
    """Owns therapeutic filtering, categorization, and memory persistence."""

    def __init__(self, *, memory: LocalForesightMemoryManager, processor: object) -> None:
        self.memory = memory
        self.processor = processor

    def filter_for_storage(self, content: str) -> str | None:
        filtered, _memory_type = self.processor.filter_for_storage(content)
        return filtered

    def categorize_message(self, message: str, crisis_severity: str) -> str:
        if crisis_severity != "none":
            return MemoryCategory.CRISIS_CONTEXT.value
        message_lower = message.lower()
        distress_indicators = ("anxious", "depressed", "overwhelmed", "struggling")
        if any(indicator in message_lower for indicator in distress_indicators):
            return "emotional_state"
        progress_indicators = ("milestone", "breakthrough", "improved", "better", "progress")
        if any(indicator in message_lower for indicator in progress_indicators):
            return "treatment_progress"
        return "general"

    def store_interaction(
        self,
        *,
        user_id: str,
        query: str,
        response: str,
        session_id: str | None,
        provider_metadata: dict[str, str],
        crisis_severity: str,
    ) -> None:
        filtered_query = self.filter_for_storage(f"User shared: {query}")
        if filtered_query:
            metadata: dict[str, object] = {"role": "user", **provider_metadata}
            if session_id:
                metadata["session_id"] = session_id
            if crisis_severity != "none":
                metadata["crisis_flag"] = True
                metadata["crisis_severity"] = crisis_severity
                metadata["category"] = MemoryCategory.CRISIS_CONTEXT.value
            self.memory.add_memory(filtered_query, user_id=user_id, metadata=metadata)

        response_summary = response[:500] if len(response) > 500 else response
        filtered_response = self.filter_for_storage(f"Assistant provided: {response_summary}")
        if filtered_response:
            metadata = {"role": "assistant", **provider_metadata}
            if session_id:
                metadata["session_id"] = session_id
            self.memory.add_memory(filtered_response, user_id=user_id, metadata=metadata)
