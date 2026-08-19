"""PII redaction gate for therapeutic memory ingestion.

Composes the production ``PiiScrubber`` with stricter defaults for memory
blocks: high-confidence identifiers are redacted while therapeutic context
terms, dates, and addresses are preserved in conservative mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ai.memory.gates import GateDecision, GateResult
from ai.pkg_mera.core.pipelines.processing.pii_scrubber import (
    PiiScrubber,
    PiiScrubberConfig,
    ScrubResult,
)

THERAPY_ALLOWLIST = [
    "therapist",
    "session",
    "client",
    "patient",
    "treatment",
    "diagnosis",
    "medication",
    "anxiety",
    "depression",
    "trauma",
    "coping",
    "boundaries",
    "trigger",
    "healing",
]

CONSERVATIVE_PII_TYPES = [
    "email",
    "phone",
    "ssn",
    "medical_record_number",
    "ip_address",
    "credit_card",
]

# Constants for PII redaction gate
MAX_PII_COUNT_FOR_BLOCK = 3
HIGH_CONFIDENCE_THRESHOLD = 0.9

PHI_TYPES = {"ssn", "medical_record_number"}
NAME_TYPE = "name"
GATE_NAME = "gate0_pii_redaction"


@dataclass(frozen=True)
class PiiRedactorConfig:
    """Configuration for the memory PII redactor."""

    redaction_style: str = "[TYPE]"
    use_spacy_for_names: bool = True
    conservative_mode: bool = True
    therapy_allowlist: list[str] = field(default_factory=THERAPY_ALLOWLIST.copy)
    spacy_confidence_threshold: float = 0.7
    pii_types: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "redaction_style": self.redaction_style,
            "use_spacy_for_names": self.use_spacy_for_names,
            "conservative_mode": self.conservative_mode,
            "therapy_allowlist": list(self.therapy_allowlist),
            "spacy_confidence_threshold": self.spacy_confidence_threshold,
            "pii_types": list(self.pii_types) if self.pii_types is not None else None,
        }


@dataclass
class PiiRedactionResult:
    """Result of a memory PII redaction operation."""

    scrubbed_text: str
    pii_types_found: list[str] = field(default_factory=list)
    pii_counts: dict[str, int] = field(default_factory=dict)
    confidence: float = 0.0
    was_redacted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "scrubbed_text": self.scrubbed_text,
            "pii_types_found": self.pii_types_found,
            "pii_counts": self.pii_counts,
            "confidence": round(self.confidence, 4),
            "was_redacted": self.was_redacted,
        }


@dataclass(frozen=True)
class _PiiMatch:
    pii_type: str
    matched_text: str
    start: int
    end: int
    confidence: float


class PiiRedactor:
    """Production PII redactor for PIX-511 memory ingestion gates."""

    def __init__(self, config: PiiRedactorConfig | None = None):
        self.config = config or PiiRedactorConfig()
        self._allowed_terms = {term.casefold() for term in self.config.therapy_allowlist}
        self._scrubber = PiiScrubber(self._build_scrubber_config())
        self._drift_counts: dict[str, int] = {}
        self._drift_types: set[str] = set()

    def redact(self, content: str) -> PiiRedactionResult:
        """Redact PII from content while preserving therapeutic context."""
        if not content or not isinstance(content, str):
            return PiiRedactionResult(scrubbed_text=content)

        if not self.config.conservative_mode:
            scrub_result = self._scrubber.scrub(content)
            return self._result_from_scrub_result(scrub_result)

        matches = self._detect_conservative_matches(content)
        scrubbed_text, pii_counts = self._apply_matches(content, matches)
        confidence = max((match.confidence for match in matches), default=0.0)
        pii_types_found = list(pii_counts.keys())

        self._record_drift(pii_counts)

        return PiiRedactionResult(
            scrubbed_text=scrubbed_text,
            pii_types_found=pii_types_found,
            pii_counts=pii_counts,
            confidence=confidence,
            was_redacted=bool(pii_counts),
        )

    def evaluate(self, content: str) -> GateResult:
        """Evaluate content for the memory PII gate."""
        result = self.redact(content)
        total_pii_count = sum(result.pii_counts.values())

        if self._has_high_confidence_phi(result):
            return GateResult(
                gate=GATE_NAME,
                decision=GateDecision.BLOCK,
                reason="High-confidence PHI detected in memory content",
                confidence=result.confidence,
                details=self._details_for(result),
            )

        if total_pii_count > MAX_PII_COUNT_FOR_BLOCK:
            return GateResult(
                gate=GATE_NAME,
                decision=GateDecision.ESCALATE,
                reason="Multiple PII instances redacted; human review recommended",
                confidence=result.confidence,
                details=self._details_for(result),
            )

        reason = "No PII detected" if total_pii_count == 0 else "PII detected and scrubbed"

        return GateResult(
            gate=GATE_NAME,
            decision=GateDecision.PASS,
            reason=reason,
            confidence=result.confidence,
            details=self._details_for(result),
        )

    def get_pii_drift_report(self) -> dict[str, Any]:
        """Return cumulative PII drift tracking for this redactor instance."""
        return {
            "types_found": sorted(self._drift_types),
            "counts": dict(sorted(self._drift_counts.items())),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _build_scrubber_config(self) -> PiiScrubberConfig:
        pii_types = self.config.pii_types
        if self.config.conservative_mode:
            conservative_types = CONSERVATIVE_PII_TYPES.copy()
            if self.config.use_spacy_for_names:
                conservative_types.append(NAME_TYPE)
            pii_types = self._intersect_pii_types(pii_types, conservative_types)

        return PiiScrubberConfig(
            redaction_style=self.config.redaction_style,
            pii_types=pii_types,
            spacy_confidence_threshold=self.config.spacy_confidence_threshold,
            use_spacy_for_names=self.config.use_spacy_for_names,
            log_findings=False,
        )

    def _detect_conservative_matches(self, content: str) -> list[_PiiMatch]:
        raw_matches = self._scrubber._detect_pii_with_regex(content)
        raw_matches.extend(self._scrubber._detect_pii_with_spacy(content))

        matches = [
            _PiiMatch(
                pii_type=pii_type,
                matched_text=matched_text,
                start=start,
                end=end,
                confidence=self._confidence_for(pii_type),
            )
            for pii_type, matched_text, start, end in raw_matches
            if self._should_redact(pii_type, matched_text)
        ]
        return self._filter_overlaps(matches)

    def _should_redact(self, pii_type: str, matched_text: str) -> bool:
        if self.config.pii_types is not None and pii_type not in self.config.pii_types:
            return False
        if self._is_allowlisted(matched_text):
            return False
        if pii_type in CONSERVATIVE_PII_TYPES:
            return True
        if pii_type == NAME_TYPE:
            return len(matched_text.strip()) > 1
        return False

    def _apply_matches(self, content: str, matches: list[_PiiMatch]) -> tuple[str, dict[str, int]]:
        scrubbed_text = content
        pii_counts: dict[str, int] = {}

        for match in reversed(matches):
            redaction = self._scrubber._redact_match(match.matched_text, match.pii_type)
            scrubbed_text = scrubbed_text[: match.start] + redaction + scrubbed_text[match.end :]
            pii_counts[match.pii_type] = pii_counts.get(match.pii_type, 0) + 1

        return scrubbed_text, pii_counts

    def _result_from_scrub_result(self, scrub_result: ScrubResult) -> PiiRedactionResult:
        pii_counts = dict(scrub_result.pii_counts)
        self._record_drift(pii_counts)
        return PiiRedactionResult(
            scrubbed_text=scrub_result.scrubbed_text,
            pii_types_found=list(pii_counts.keys()),
            pii_counts=pii_counts,
            confidence=max((self._confidence_for(pii_type) for pii_type in pii_counts), default=0.0),
            was_redacted=scrub_result.total_pii_count > 0,
        )

    def _record_drift(self, pii_counts: dict[str, int]) -> None:
        for pii_type, count in pii_counts.items():
            self._drift_types.add(pii_type)
            self._drift_counts[pii_type] = self._drift_counts.get(pii_type, 0) + count

    def _has_high_confidence_phi(self, result: PiiRedactionResult) -> bool:
        return result.confidence > HIGH_CONFIDENCE_THRESHOLD and any(
            pii_type in PHI_TYPES for pii_type in result.pii_counts
        )

    def _details_for(self, result: PiiRedactionResult) -> list[str]:
        if not result.pii_counts:
            return []
        return [f"{pii_type}: {count}" for pii_type, count in sorted(result.pii_counts.items())]

    def _is_allowlisted(self, text: str) -> bool:
        normalized = text.strip().casefold()
        return normalized in self._allowed_terms

    def _confidence_for(self, pii_type: str) -> float:
        if pii_type in PHI_TYPES:
            return 0.98
        if pii_type == NAME_TYPE:
            return self.config.spacy_confidence_threshold
        if pii_type in CONSERVATIVE_PII_TYPES:
            return 0.95
        return 0.0

    def _filter_overlaps(self, matches: list[_PiiMatch]) -> list[_PiiMatch]:
        filtered_matches: list[_PiiMatch] = []
        previous_end = 0

        for match in sorted(matches, key=lambda item: item.start):
            if match.start >= previous_end:
                filtered_matches.append(match)
                previous_end = match.end
                continue

            if not filtered_matches:
                continue

            previous = filtered_matches[-1]
            previous_length = previous.end - previous.start
            current_length = match.end - match.start
            if current_length > previous_length:
                filtered_matches[-1] = match
                previous_end = match.end

        return filtered_matches

    def _intersect_pii_types(
        self,
        requested_types: list[str] | None,
        allowed_types: list[str],
    ) -> list[str]:
        if requested_types is None:
            return allowed_types
        allowed_type_set = set(allowed_types)
        return [pii_type for pii_type in requested_types if pii_type in allowed_type_set]


__all__ = [
    "PiiRedactionResult",
    "PiiRedactor",
    "PiiRedactorConfig",
]
