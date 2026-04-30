#!/usr/bin/env python3
"""CPTSD topic tagger using regex detection patterns.

Loads the topic taxonomy from cptsd_voice_profiles.json and
detects CPTSD topics + crisis signals in plain text.

Usage:
    from cptsd_topic_tagger import CPTSDTopicTagger
    tagger = CPTSDTopicTagger()
    result = tagger.tag(text)
    # result = {
    #   "cptsd_topics": ["shame_cycles", "dissociation"],
    #   "topic_scores": {"shame_cycles": 0.8, ...},
    #   "crisis_detected": False,
    #   "crisis_severity": None,
    #   "crisis_types": [],
    #   "is_training_edge_case": False,
    # }
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILES_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "cptsd_voice_profiles.json"
)

CRISIS_SEVERITY_WEIGHTS = {
    "high_severity": 1.0,
    "medium_severity": 0.6,
    "cptsd_specific": 0.4,
}

CRISIS_THRESHOLD_HIGH = 0.7
CRISIS_THRESHOLD_MEDIUM = 0.4


class CPTSDTopicTagger:
    """Detect CPTSD topics and crisis signals in text."""

    def __init__(self, profiles_path: Path | None = None) -> None:
        path = profiles_path or PROFILES_PATH
        if not path.exists():
            logger.warning(
                "Voice profiles not found at %s, topic tagging disabled",
                path,
            )
            self._taxonomy: dict = {}
            self._crisis_patterns: dict = {}
            return

        with open(path) as fh:
            data = json.load(fh)

        self._taxonomy = data.get("cptsd_topic_taxonomy", {})
        self._crisis_patterns = data.get("crisis_detection_patterns", {})

        # Pre-compile all detection patterns
        self._compiled_topics: dict[str, list[re.Pattern]] = {}
        for topic_key, topic in self._taxonomy.items():
            patterns = topic.get("detection_patterns", [])
            compiled = []
            for p in patterns:
                try:
                    compiled.append(
                        re.compile(
                            p if _is_regex(p) else re.escape(p),
                            re.IGNORECASE,
                        )
                    )
                except re.error:
                    compiled.append(re.compile(re.escape(p), re.IGNORECASE))
            self._compiled_topics[topic_key] = compiled

        self._compiled_crisis: dict[str, list[re.Pattern]] = {}
        for severity, phrases in self._crisis_patterns.items():
            compiled = [re.compile(re.escape(p), re.IGNORECASE) for p in phrases]
            self._compiled_crisis[severity] = compiled

    def tag(self, text: str) -> dict:
        """Tag text with CPTSD topics and crisis signals."""
        if not text or not self._taxonomy:
            return _empty_result()

        text_lower = text.lower()

        # Topic detection
        topic_scores: dict[str, float] = {}
        for topic_key, patterns in self._compiled_topics.items():
            hits = sum(bool(p.search(text_lower)) for p in patterns)
            if hits > 0:
                score = min(1.0, hits / max(1, len(patterns)))
                topic_scores[topic_key] = round(score, 2)

        detected_topics = sorted(
            topic_scores.keys(),
            key=lambda k: topic_scores[k],
            reverse=True,
        )

        # Crisis detection
        crisis_score = 0.0
        crisis_types: list[str] = []
        for severity, patterns in self._compiled_crisis.items():
            weight = CRISIS_SEVERITY_WEIGHTS.get(severity, 0.5)
            hits = sum(bool(p.search(text_lower)) for p in patterns)
            if hits > 0:
                crisis_score += weight * (hits / max(1, len(patterns)))
                crisis_types.append(severity)

        crisis_score = min(1.0, crisis_score)
        crisis_detected = crisis_score >= CRISIS_THRESHOLD_MEDIUM

        if crisis_score >= CRISIS_THRESHOLD_HIGH:
            crisis_severity = "HIGH"
        elif crisis_score >= CRISIS_THRESHOLD_MEDIUM:
            crisis_severity = "MEDIUM"
        else:
            crisis_severity = None

        is_edge_case = (
            crisis_detected
            or "dissociation" in detected_topics
            or "trauma_reenactment" in topic_scores
        )

        return {
            "cptsd_topics": detected_topics,
            "topic_scores": topic_scores,
            "crisis_detected": crisis_detected,
            "crisis_severity": crisis_severity,
            "crisis_score": round(crisis_score, 3),
            "crisis_types": crisis_types,
            "is_training_edge_case": is_edge_case,
        }


def _is_regex(pattern: str) -> bool:
    """Heuristic: does the string look like a regex."""
    return any(c in pattern for c in r"[]\^$.|?*+()")


def _empty_result() -> dict:
    return {
        "cptsd_topics": [],
        "topic_scores": {},
        "crisis_detected": False,
        "crisis_severity": None,
        "crisis_score": 0.0,
        "crisis_types": [],
        "is_training_edge_case": False,
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    tagger = CPTSDTopicTagger()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = (
            "I feel like I'm not really here. "
            "Everything feels unreal. I can't "
            "control my emotions. I'm so ashamed "
            "of myself."
        )

    result = tagger.tag(text)
    print(json.dumps(result, indent=2))
