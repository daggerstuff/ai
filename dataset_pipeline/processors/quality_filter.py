from typing import Any

MIN_MESSAGES = 3
MIN_CONTENT_LENGTH = 2


class QualityFilter:
    """Filters out malformed, duplicated, or low-quality ChatML records."""

    def __init__(self) -> None:
        self.seen_hashes: set[int] = set()

    @staticmethod
    def _format_is_valid(messages: list[dict[str, Any]]) -> bool:
        if len(messages) < MIN_MESSAGES:
            return False
        if any(messages[i]["role"] == messages[i + 1]["role"] for i in range(1, len(messages) - 1)):
            return False
        return all(msg.get("content") and len(str(msg["content"]).strip()) >= MIN_CONTENT_LENGTH for msg in messages)

    def passes_filter(self, chatml_record: dict[str, Any]) -> bool:
        """Returns True if the record meets all quality criteria."""
        messages = chatml_record.get("messages", [])
        if not self._format_is_valid(messages):
            return False
        # P0-1: Edge-case training records keep deliberate clinical signal; still format-validated,
        # but they bypass quality deduplication.
        if chatml_record.get("is_training_edge_case") is True:
            return True
        content_hash = hash("".join([m["content"] for m in messages]))
        if content_hash in self.seen_hashes:
            return False
        self.seen_hashes.add(content_hash)
        return True
