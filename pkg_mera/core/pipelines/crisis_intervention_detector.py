"""Crisis intervention detection and escalation support.

Extends the original CrisisInterventionDetector with heuristic toxicity detection
that distinguishes clinical mental health discussion from genuinely toxic content
(PIX-4240). Toxicity categories are kept separate from crisis patterns so that
clinical training data referencing trauma, substance use, or difficult emotions
is not over-filtered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class CrisisInterventionResult:
    """Structured crisis assessment output."""

    flagged: bool
    crisis_type: str
    severity: str
    score: float
    matches: list[str]
    recommendations: list[str] = field(default_factory=list)


@dataclass
class ToxicityResult:
    """Heuristic toxicity assessment output.

    ``flagged`` is True when the text genuinely matches a toxicity pattern. Clinical
    discussion (trauma, substance use, difficult emotions) is NOT flagged — only
    graphic/instructional/hateful/exploitative content is.
    """

    flagged: bool
    categories: list[str]
    score: float
    matches: list[str]


class CrisisInterventionDetector:
    """Production-oriented heuristic crisis detector with escalation actions."""

    CRISIS_KEYWORDS: dict[str, tuple[float, list[str]]] = {
        "suicidal_ideation": (
            0.95,
            [
                r"\bi want to die\b",
                r"\bi want to kill myself\b",
                r"\bend my life\b",
                r"\bno reason to live\b",
                r"\bcan't go on\b",
            ],
        ),
        "self_harm": (
            0.85,
            [r"\bcut\b", r"\bhurt myself\b", r"\bself.?harm\b", r"\bhurt\s+myself\b"],
        ),
        "violence": (0.75, [r"\bhurt\s+.*others\b", r"\bkill\b", r"\battack\b"]),
        "substance_abuse": (0.55, [r"\boverdose\b", r"\bintoxicated\b", r"\bwithdraw\b"]),
        "panic": (0.45, [r"\bcan't breathe\b", r"\bpanic\b", r"\bheart is racing\b"]),
    }
    SEVERITY_BANDS = {
        "critical": 0.8,
        "high": 0.6,
        "elevated": 0.4,
        "moderate": 0.25,
        "low": 0.0,
    }

    TOXICITY_KEYWORDS: dict[str, tuple[float, list[str]]] = {
        "graphic_violence": (
            0.85,
            [
                r"\bhow\s+to\s+(beat|stab|strangl|drown|burn\s+alive)\b",
                r"\bstep.?by.?step\s+(kill|murder|tortur)\w*\b",
                r"\bdisembowel\w*\b",
                r"\bslit\s+(their|her|his)\s+throat\b",
                r"\bchoking\s+.*unconscious\b",
                r"\bseverely\s+beaten\s+.*how\s+to\b",
                r"\bmethod\s+(for|to)\s+(kill|murder|tortur)\w*\b",
                r"\bhow\s+to\s+disfigure\b",
                r"\bblood\s+(pouring|gushing|pooling)\b.*\bbody\b",
            ],
        ),
        "hate_speech": (
            0.85,
            [
                r"\b\w*hate\s+(all|every)\s+(gay|black|jewish|muslim|women|trans|immigrant)\w*\b",
                r"\b(genocide|extermination|ethnic\s+cleansing)\s+.*\b(is\s+good|necessary|justified)\b",
                r"\bdestroy\s+(all|every)\s+(gay|black|jewish|muslim|trans)\w*\b",
                r"\b(vermin|sub.?human|apes|savages)\s+referring\s+to\b",
                r"\bkill\s+(all|every)\s+(gay|black|jewish|muslim|trans|women)\w*\b",
                r"\bdeport\s+(all|every)\s+(gay|black|jewish|muslim|immigrant)\w*\s+now\b",
            ],
        ),
        "sexual_content": (
            0.80,
            [
                r"\b(how\s+to|instructions\s+for)\s+(rape|sexually\s+assault|molest)\b",
                r"\bdetailed\s+(sex|pornographic|erotic)\s+(act|scene|depiction)\b",
                r"\bstep.?by.?step\s+(rape|sexual\s+assault|molestation)\b",
                r"\bexplicit\s+sexual\s+intercourse\s+description\b",
                r"\b\w*secretly\s+record\w*\s+.*\b(naked|undress|shower|bathroom)\b",
                r"\bhidden\s+camera\s+(in|at)\s+(shower|bathroom|bedroom)\b.*\binstructions\b",
                r"\bcoerce\w*\s+(into|to)\s+(sex|intercourse|oral|anal)\b.*\bhow\s+to\b",
            ],
        ),
        "manipulative_patterns": (
            0.75,
            [
                r"\bhow\s+to\s+(groom|manipulat\w+|gaslight|coerce)\s+(a|some|target|victim)\b",
                r"\bstep.?by.?step\s+(gaslighting|manipulation|grooming)\b",
                r"\bhow\s+to\s+(stalk|track\s+without\s+consent|monitor\s+without\s+consent)\b",
                r"\bcoercive\s+control\s+tactics\b.*\binstructions?\b",
                r"\bcut\s+(them|her|him)\s+off\s+from\s+(friends|family|support)\b.*\bhow\s+to\b",
                r"\bmake\s+(them|her|him)\s+doubt\s+(their|her|his)\s+(memory|sanity)\b.*\bhow\s+to\b",
                r"\bforcing\s+.*\b(financial|economic)\s+dependency\b.*\bhow\s+to\b",
            ],
        ),
        "dangerous_instructions": (
            0.90,
            [
                r"\bhow\s+to\s+(make|build|manufacture)\b.*\b(bomb|explosive|grenade|molotov)\b",
                r"\bsynthesis?\s+(for|of)\s+(ricin|sarin|anthrax| VX| VX\s+nerve)\b",
                r"\bstep.?by.?step\s+(poison|poisoning|toxin)\s+recipes?\b",
                r"\brecipes?\s+(for|to\s+make)\s+(meth|methamphetamine|crystal\s+meth|fentanyl)\b",
                r"\bhow\s+to\s+(extract|synthesize|produce)\s+(heroin|cocaine|cyanide|arsenic)\b",
                r"\binstructions?\s+(for|to)\s+(build|construct)\s+(silencer|machine\s+gun|automatic\s+weapon)\b",
                r"\bhow\s+to\s+cut\s+(drugs|heroin|fentanyl)\s+with\s+(lactose|ratanil|fentalyl)\b",
            ],
        ),
    }
    TOXICITY_SEVERITY_BANDS: dict[str, float] = {
        "critical": 0.8,
        "high": 0.6,
        "elevated": 0.4,
        "moderate": 0.25,
        "low": 0.0,
    }

    # Patterns describing legitimate clinical discussion that must NOT be toxic-flagged
    # even if similar keywords appear. These are subtractive: if a clinical context
    # marker is present, the toxicity flag for the overlapping category is suppressed
    # unless a genuinely toxic pattern (with how-to/instructional/explicit framing) still
    # matches independently. The suppression only applies to weak matches.
    CLINICAL_CONTEXT_MARKERS: list[str] = [
        r"\b(patient|client|therapist|counselor|clinician|psychiatrist|psychologist)\b",
        r"\b(discussing|reflecting|processing|exploring|narrating|describing)\s+(trauma|abuse|addiction|substance|crisis)\b",
        r"\b(treatment|therapy|intervention|recovery|support\s+group|relapse\s+prevention)\b",
        r"\b(trauma.informed|clinical|comorbid|diagnosis|symptom|medication|prescription)\b",
        r"\b(rape\s+counseling|sexual\s+assault\s+survivor|abuse\s+survivor)\b",
        r"\b(processing\s+(my|a|their)\s+(assault|abuse|attack|experience))\b",
        r"\b(I|my|their|her|his)\s+(abuse|assault|trauma|rape)\s+(was|happened|survivor|recovery)\w*\b",
        r"\b(dissociat\w+|flashbacks?|ptsd|hypoarousal|hyperarousal|c.?ptsd|bpd|borderline)\b",
        r"\b(trigger\s+warning|content\s+warning|tw:|cw:)\b",
        r"\b(narrative\s+exposure\s+therapy|net\s+session|trauma\s+processing)\b",
        r"\b(suicid\w*|depress\w*|anxiety|panic\s+attack)\s+(ideation|assessment|risk\s+screen|risk\s+assessment)\b",
        r"\b(safety\s+plan|crisis\s+line|crisis\s+hotline|988|coping\s+strategy|coping\s+skill)\b",
    ]

    def __init__(self) -> None:
        self.escalation_contacts = {
            "critical": ["911", "988"],
            "high": ["988"],
        }
        self._clinical_marker_re: list[re.Pattern[str]] = [
            re.compile(p, flags=re.IGNORECASE) for p in self.CLINICAL_CONTEXT_MARKERS
        ]

    def process(self, data: dict[str, Any] | str) -> CrisisInterventionResult:
        if data is None:
            raise ValueError("data must not be None")
        text = self._extract_text(data).lower()
        if not text:
            return CrisisInterventionResult(False, "none", "none", 0.0, [])

        matches, score, crisis_type = self._detect(text)
        severity = self._severity_label(score)

        recommendations = []
        if score >= self.SEVERITY_BANDS["critical"]:
            recommendations.append("Immediate human escalation required")
            contacts = self.escalation_contacts.get("critical", [])
            recommendations.append(f"Escalate to: {', '.join(contacts)}")
        elif score >= self.SEVERITY_BANDS["high"]:
            recommendations.append("Queue for clinical review within 10 minutes")
        elif score >= self.SEVERITY_BANDS["elevated"]:
            recommendations.append("Increase monitoring and add safety check")
        else:
            recommendations.append("No immediate intervention required")

        return CrisisInterventionResult(
            flagged=score >= self.SEVERITY_BANDS["elevated"],
            crisis_type=crisis_type,
            severity=severity,
            score=score,
            matches=matches,
            recommendations=recommendations,
        )

    def detect_toxicity(self, data: dict[str, Any] | str) -> ToxicityResult:
        """Run heuristic toxicity detection separate from crisis detection.

        Clinical mental health discussion (trauma, substance use, abuse survivorship,
        difficult emotions, diagnostic mentions) is deliberately NOT flagged here. Only
        graphic/instructional/hateful/exploitative content is flagged.

        Suppression rule: when a clinical context marker is present, weak toxicity
        matches (score < 0.85) for the overlapping category are suppressed — but high-
        confidence toxicity matches (score >= 0.85, e.g. "how to make a bomb" or slurs
        explicitly aimed at a protected group) still flag, regardless of context.
        """
        text = self._extract_text(data).lower()
        if not text:
            return ToxicityResult(False, [], 0.0, [])

        has_clinical_context = any(p.search(text) for p in self._clinical_marker_re)

        found: list[str] = []
        max_score = 0.0
        matches: list[str] = []

        for label, (base_score, patterns) in self.TOXICITY_KEYWORDS.items():
            for pattern in patterns:
                m = re.search(pattern, text, flags=re.IGNORECASE)
                if m:
                    # Clinical context suppresses weak matches only. Strong matches
                    # (>= 0.85) flag regardless — they are unambiguous.
                    if has_clinical_context and base_score < 0.85:
                        continue
                    found.append(label)
                    max_score = max(max_score, base_score)
                    matches.append(m.group(0))
                    break

        return ToxicityResult(
            flagged=bool(found and max_score >= 0.4),
            categories=sorted(set(found)),
            score=float(max_score),
            matches=matches,
        )

    def _extract_text(self, data: dict[str, Any] | str) -> str:
        if isinstance(data, str):
            return data
        if not isinstance(data, dict):
            raise ValueError("data must be mapping or text")

        for key in ("text", "content", "message", "query", "input"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value

        # Fallback flatten conversation-like payloads
        if isinstance(data.get("messages"), list):
            parts: list[str] = []
            for message in data["messages"]:
                if isinstance(message, dict):
                    role_content = message.get("content")
                    if isinstance(role_content, str):
                        parts.append(role_content)
            return " ".join(parts)

        return ""

    def _detect(self, text: str) -> tuple[list[str], float, str]:
        found: list[str] = []
        max_score = 0.0
        crisis_type = "none"

        for label, (base_score, patterns) in self.CRISIS_KEYWORDS.items():
            for pattern in patterns:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    found.append(label)
                    max_score = max(max_score, base_score)
                    crisis_type = label
                    break

        return sorted(set(found)), float(max_score), crisis_type if found else "none"

    def _severity_label(self, score: float) -> str:
        if score >= self.SEVERITY_BANDS["critical"]:
            return "critical"
        if score >= self.SEVERITY_BANDS["high"]:
            return "high"
        if score >= self.SEVERITY_BANDS["elevated"]:
            return "elevated"
        if score >= self.SEVERITY_BANDS["moderate"]:
            return "moderate"
        return "low"

    def audit_event(self, result: CrisisInterventionResult) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "crisis_type": result.crisis_type,
            "severity": result.severity,
            "score": result.score,
            "matches": result.matches,
            "flagged": result.flagged,
            "recommendations": list(result.recommendations),
        }


__all__ = ["CrisisInterventionDetector", "CrisisInterventionResult", "ToxicityResult"]
