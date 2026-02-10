"""
Pixelated Empathy: The Anti-Echo Hook

This service filters model outputs to remove the 'AI-echo' pattern.
It ensures the first sentence doesn't start with common validation clichés.
"""

import logging
import re

logger = logging.getLogger(__name__)


class AntiEchoHook:
    """
    Acts as a 'pre-push' (or post-generation) filter to ensure human-like starts.
    """

    ECHO_PATTERNS = [
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
    ]

    def __init__(self, fallback_mode: str = "strip"):
        """
        fallback_mode:
            'strip' -> Removes the first sentence if it's an echo.
            'flag'  -> Returns a warning (for RL training).
        """
        self.compiled_echoes = [re.compile(p, re.IGNORECASE) for p in self.ECHO_PATTERNS]
        self.fallback_mode = fallback_mode

    def process_response(self, text: str) -> str:
        """
        Analyzes the response and enforces 'The Human Pivot'.
        Iteratively strips echo patterns from the start.
        """
        current_text = text

        while True:
            sentences = current_text.split(". ", 1)
            if not sentences:
                break

            first_sentence = sentences[0].strip()
            is_echo = any(pattern.match(first_sentence) for pattern in self.compiled_echoes)

            if not is_echo:
                break

            if self.fallback_mode == "strip" and len(sentences) > 1:
                logger.info(f"Anti-Echo Hook: Stripping echo start: '{first_sentence}'")
                current_text = sentences[1]
            else:
                logger.warning(
                    f"Response failed Human-Pivot check (cannot strip further): '{first_sentence}'"
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
