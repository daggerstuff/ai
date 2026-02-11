"""
Pixelated Empathy: The Anti-Echo Hook

This service filters model outputs to remove the 'AI-echo' pattern.
It ensures the first sentence doesn't start with common validation clichés.
"""

import logging
import re
from typing import ClassVar

logger = logging.getLogger(__name__)


class AntiEchoHook:
    """
    Acts as a 'pre-push' (or post-generation) filter to ensure human-like starts.
    """

    ECHO_PATTERNS: ClassVar[tuple[str, ...]] = (
        r"^i hear (that )?you",
        r"^it sounds like",
        r"^i understand (that )?you",
        r"^it seems (like )?you",
        r"^i can see (that )?you",
        r"^i realize (that )?you",
        r"^you're saying",
        r"^so, you're feel",
        r"^it's (completely )?normal to",
        r"^i appreciate (you )?sharing",
        r"^thank you for sharing",
        r"^i'm here to help",
        r"^i'm sorry to hear",
    )

    def __init__(self, fallback_mode: str = "strip"):
        """
        fallback_mode:
            'strip' -> Removes the first sentence if it's an echo.
            'flag'  -> Returns a warning (for RL training).
        """
        self.compiled_echoes = [re.compile(p, re.IGNORECASE) for p in self.ECHO_PATTERNS]
        if fallback_mode not in {"strip", "flag"}:
            raise ValueError(f"Unsupported fallback_mode: {fallback_mode}")
        self.fallback_mode = fallback_mode

    def process_response(self, text: str) -> str:
        """
        Analyzes the response and enforces 'The Human Pivot'.
        Iteratively strips echo patterns from the start.
        """
        current_text = text

        while True:
            # More robust sentence splitting: look for punctuation followed by space or end of string
            # Handle multiple punctuations like '...' or '!!!'
            sentences = re.split(r"(?<=[.!?])\s+", current_text.strip())
            if not sentences or not sentences[0]:
                break

            first_sentence = sentences[0].strip()
            rest = " ".join(sentences[1:]).strip() if len(sentences) > 1 else ""

            if not first_sentence:
                break

            is_echo = any(pattern.match(first_sentence) for pattern in self.compiled_echoes)

            if not is_echo:
                break

            # Safe logging: mask user content
            redacted_start = f"{first_sentence[:10]}...[REDACTED]"

            if self.fallback_mode == "strip" and rest:
                logger.info(f"Anti-Echo Hook: Stripping echo start: '{redacted_start}'")
                current_text = rest
            else:
                logger.warning(
                    f"Response failed Human-Pivot check (cannot strip further): '{redacted_start}'"
                )
                break
        return current_text


# Example Usage
if __name__ == "__main__":
    hook = AntiEchoHook()
    test_1 = "I hear that you are feeling sad. Let's talk about your father."
    test_2 = "My own father used to say that silence was a weapon. I wonder if yours felt the same?"

    print(f"Test 1 Before: {test_1}")
    print(f"Test 1 After:  {hook.process_response(test_1)}")

    print(f"Test 2 Before: {test_2}")
    print(f"Test 2 After:  {hook.process_response(test_2)}")
