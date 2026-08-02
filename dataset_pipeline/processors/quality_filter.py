class QualityFilter:
    """Filters out malformed, duplicated, or low-quality ChatML records."""

    def __init__(self):
        self.seen_hashes = set()

    def passes_filter(self, chatml_record):
        """Returns True if the record meets all quality criteria."""
        # P0-1: Edge-case training record bypass — preserve deliberate clinical edge-case signal
        # Edge cases are intentionally difficult/scary/horrible — guardrails filtering them out makes training worthless
        if chatml_record.get("is_training_edge_case") is True:
            # Still run format validation (length, alternating roles, non-empty) but skip quality deduplication
            messages = chatml_record.get("messages", [])
            if len(messages) < 3:
                return False
            for i in range(1, len(messages) - 1):
                if messages[i]["role"] == messages[i + 1]["role"]:
                    return False
            for msg in messages:
                if not msg.get("content") or len(str(msg["content"]).strip()) < 2:
                    return False
            return True

        messages = chatml_record.get("messages", [])

        # 1. Length check: Must have at least a system, user, and assistant message
        if len(messages) < 3:
            return False

        # 2. Alternating roles validation (ignoring the first system prompt)
        # Typically: user, assistant, user, assistant
        for i in range(1, len(messages) - 1):
            if messages[i]["role"] == messages[i + 1]["role"]:
                return False

        # 3. Content check: No empty messages
        for msg in messages:
            if not msg.get("content") or len(str(msg["content"]).strip()) < 2:
                return False

        # 4. Exact Deduplication
        # Hash the concatenated content of the conversation
        concat_content = "".join([m["content"] for m in messages])
        content_hash = hash(concat_content)
        if content_hash in self.seen_hashes:
            return False
        self.seen_hashes.add(content_hash)

        return True
