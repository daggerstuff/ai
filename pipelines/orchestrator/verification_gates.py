import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class EmpathySafetyGate:
    """
    Implements quality gates based on EMPATHY_RESPONSE_STYLE.md.
    Enforces EARS framework: Empathize, Acknowledge, Reflect, Support.
    """

    # Heuristics for EARS components (Keyword-based for fast gating)
    # Ideally this uses a classifier, but we start with strict keyword/pattern matching
    EMPATHY_MARKERS = [
        r"I hear how",
        r"it sounds like",
        r"It makes sense",
        r"I can imagine",
        r"feeling",
        r"heavy",
        r"painful",
        r"hard",
    ]

    SUPPORT_MARKERS = [
        r"we can",
        r"let's",
        r"together",
        r"would you be open",
        r"what if",
        r"small step",
    ]

    PROHIBITED_PATTERNS = [
        r"As an AI language model",
        r"I cannot have feelings",
        r"I am just a computer",
        r"Please consult a professional",  # Only allowed in strict crisis contexts, but flag for review
    ]

    @classmethod
    def check_ears_compliance(cls, record: dict[str, Any]) -> dict[str, Any]:
        """
        Check if a record complies with EARS response style.
        Returns result dict with 'passed' boolean and 'score'.
        """
        messages = record.get("messages", [])
        if not messages:
            return {"passed": False, "reason": "No messages"}

        # Analyze the last assistant message (the response we train on)
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        if not assistant_msgs:
            return {"passed": False, "reason": "No assistant response"}

        last_response = assistant_msgs[-1].get("content", "")

        # 1. Prohibited Content Check
        for pattern in cls.PROHIBITED_PATTERNS:
            if re.search(pattern, last_response, re.IGNORECASE):
                # Allow strictly if crisis? For now, hard fail to force high quality.
                return {
                    "passed": False,
                    "reason": f"Contains prohibited phrase: '{pattern}'",
                    "score": 0.0,
                }

        # 2. EARS heuristic score
        empathy_hits = sum(
            1 for p in cls.EMPATHY_MARKERS if re.search(p, last_response, re.IGNORECASE)
        )
        support_hits = sum(
            1 for p in cls.SUPPORT_MARKERS if re.search(p, last_response, re.IGNORECASE)
        )

        # Simple scoring: needs at least minimal empathy + support markers
        # Pass if > 0 hits combined, or if very long (context dependent).
        # This is a loose gate to avoid false positives on valid unique responses.
        score = min(1.0, (empathy_hits * 0.3 + support_hits * 0.3))

        passed = score >= 0.3 or len(last_response.split()) > 20  # Length heuristic as fallback

        return {
            "passed": passed,
            "score": score,
            "details": {"empathy_hits": empathy_hits, "support_hits": support_hits},
        }
