class ChatMLConverter:
    """Normalizes various extracted formats into strict ChatML."""

    def __init__(self):
        self.default_system_prompt = (
            "You are Pixel, a highly empathetic and clinically precise AI "
            "therapist. You balance deep psychological insight with a warm, "
            "authentic conversational style."
        )

    def convert(self, extracted_record):
        """
        Takes an extracted record containing 'raw_data' and 'metadata',
        and returns a standard ChatML dict.
        """
        metadata = extracted_record.get("metadata", {})
        source_family = metadata.get("source_family", "unknown")

        raw = extracted_record.get("raw_data", {})

        messages = []
        # Always inject system prompt first
        messages.append({
            "role": "system",
            "content": self.default_system_prompt
        })

        if source_family == "voice_training":
            # Input format is usually {'conversation': [...]} or {'messages': [...]}
            conv = raw.get("conversation", [])
            if not conv:
                conv = raw.get("messages", [])

            for turn in conv:
                role = turn.get("role", "user")
                content = turn.get("content", "")

                # Normalize roles
                if role in ["client", "user", "human"]:
                    normalized_role = "user"
                elif role in ["therapist", "assistant", "ai", "system"]:
                    # if they have an internal system prompt, we just make it assistant or user
                    # wait, if it's 'system', we should skip it since we already prepended one,
                    # or map it to user/assistant. We will skip it.
                    if role == "system":
                        continue
                    normalized_role = "assistant"
                else:
                    normalized_role = "user"

                messages.append({"role": normalized_role, "content": content})

        elif source_family == "psychology_knowledge":
            # Input format is {'text': 'chunk of text'}
            text = raw.get("text", "")
            book_name = metadata.get("book_name", "Clinical Text")

            messages.append({
                "role": "user",
                "content": f"Please process and store this clinical knowledge from {book_name}:\n\n{text}"
            })
            messages.append({
                "role": "assistant",
                "content": "I have integrated this clinical knowledge and will apply it in therapeutic contexts."
            })

        elif source_family == "reasoning_enhancement":
            # Formats: {"Context": "...", "Response": "..."} or {"text": [{"role": "user", "content": "..."}, ...]}

            if "Context" in raw and "Response" in raw:
                messages.append({"role": "user", "content": raw["Context"]})
                messages.append({"role": "assistant", "content": raw["Response"]})
            elif "text" in raw and isinstance(raw["text"], list):
                for turn in raw["text"]:
                    role = turn.get("role", "user")
                    content = turn.get("content", "")
                    messages.append({"role": role, "content": content})
            else:
                # Generic fallback
                messages.append({"role": "user", "content": str(raw)})
                messages.append({"role": "assistant", "content": "Acknowledged."})

        # Smart fallback for mental_health and personality sets

        # 1. Look for 'Context' and 'Response'
        elif "Context" in raw and "Response" in raw:
            messages.append({"role": "user", "content": str(raw["Context"])})
            messages.append({"role": "assistant", "content": str(raw["Response"])})

        # 2. Look for 'instruction' and 'output' (Alpaca style)
        elif "instruction" in raw and "output" in raw:
            user_msg = raw["instruction"]
            if raw.get("input"): user_msg += "\n" + raw["input"]
            messages.append({"role": "user", "content": str(user_msg)})
            messages.append({"role": "assistant", "content": str(raw["output"])})

        # 3. Look for 'messages' or 'conversation' or 'text' list
        elif "messages" in raw or "conversation" in raw or ("text" in raw and isinstance(raw["text"], list)):
            conv = raw.get("messages", []) or raw.get("conversation", []) or (raw.get("text", []) if isinstance(raw.get("text"), list) else [])
            for turn in conv:
                role = turn.get("role", turn.get("from", "user"))
                content = turn.get("content", turn.get("value", ""))

                if role in ["client", "user", "human"]: role = "user"
                elif role in ["therapist", "assistant", "ai", "system", "gpt"]:
                    if role == "system": continue
                    role = "assistant"
                else: role = "user"

                messages.append({"role": role, "content": str(content)})

        # 4. Total fallback stringification
        else:
            messages.append({
                "role": "user",
                "content": str(raw)
            })
            messages.append({
                "role": "assistant",
                "content": "Acknowledged."
            })

        return {
            "messages": messages,
            "metadata": metadata
        }
