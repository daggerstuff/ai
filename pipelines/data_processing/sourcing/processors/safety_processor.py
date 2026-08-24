"""Safety processor combining PII stripping and toxicity filtering.

Orchestrates PII redaction and heuristic toxicity scoring across ChatML
records. Edge-case training records (``is_training_edge_case=True``) are
exempt from toxicity filtering but still receive PII stripping — PII must
be removed from all records regardless of training purpose.

Usage:
    from sourcing.processors.safety_processor import SafetyProcessor

    processor = SafetyProcessor()
    cleaned, report = processor.process_batch(records)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sourcing.processors.pii_stripper import PIIStripReport, PIIStripResult, strip_pii_from_record
from sourcing.processors.toxicity_filter import ToxicityReport, ToxicityResult, score_record


@dataclass
class SafetyReport:
    """Aggregate safety report for a batch of records."""

    pii_report: PIIStripReport = field(default_factory=PIIStripReport)
    toxicity_report: ToxicityReport = field(default_factory=ToxicityReport)
    edge_case_bypassed: int = 0
    records_dropped_toxic: int = 0

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for logging/serialization."""
        return {
            "total_records": self.pii_report.total_records,
            "pii": {
                "records_with_pii": self.pii_report.records_with_pii,
                "total_redactions": self.pii_report.total_redactions,
                "by_type": self.pii_report.by_type,
            },
            "toxicity": {
                "flagged_records": self.toxicity_report.flagged_records,
                "total_hits": self.toxicity_report.total_hits,
                "by_category": self.toxicity_report.by_category,
            },
            "edge_case_bypassed": self.edge_case_bypassed,
            "records_dropped_toxic": self.records_dropped_toxic,
        }


@dataclass
class SafetyRecordResult:
    """Per-record safety result."""

    record: dict[str, Any]
    pii_result: PIIStripResult
    toxicity_result: ToxicityResult
    kept: bool
    edge_case_bypassed: bool


class SafetyProcessor:
    """Combined PII stripping + toxicity filtering processor.

    Args:
        drop_toxic: If True, records that score above the toxicity
            threshold are dropped from the output. If False, they
            are kept but flagged with ``safety_flag`` metadata.
    """

    def __init__(self, *, drop_toxic: bool = False) -> None:
        self.drop_toxic = drop_toxic

    def process_record(self, record: dict[str, Any]) -> SafetyRecordResult:
        """Process a single ChatML record through PII + toxicity checks."""
        is_edge_case = record.get("is_training_edge_case") is True

        # PII stripping always runs — PII must be removed from all records
        pii_result = strip_pii_from_record(record)
        cleaned = pii_result.record

        # Toxicity scoring — edge cases bypass the drop logic but are still scored
        toxicity_result = score_record(cleaned)
        edge_case_bypassed = False
        kept = True

        if is_edge_case:
            edge_case_bypassed = True
            # Add safety metadata but keep the record
            cleaned["safety_flag"] = "edge_case_bypass"
        elif toxicity_result.is_toxic:
            if self.drop_toxic:
                kept = False
            else:
                cleaned["safety_flag"] = "toxic"
        else:
            cleaned.pop("safety_flag", None)

        if kept:
            cleaned["pii_redacted"] = pii_result.had_redactions
            cleaned["toxicity_score"] = round(toxicity_result.score, 4)

        return SafetyRecordResult(
            record=cleaned,
            pii_result=pii_result,
            toxicity_result=toxicity_result,
            kept=kept,
            edge_case_bypassed=edge_case_bypassed,
        )

    def process_batch(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], SafetyReport]:
        """Process a batch of ChatML records.

        Returns the list of cleaned/kept records and an aggregate report.
        """
        report = SafetyReport()
        cleaned: list[dict[str, Any]] = []

        for record in records:
            result = self.process_record(record)
            report.pii_report.add(result.pii_result)
            report.toxicity_report.add(result.toxicity_result)
            if result.edge_case_bypassed:
                report.edge_case_bypassed += 1
            if not result.kept:
                report.records_dropped_toxic += 1
            if result.kept:
                cleaned.append(result.record)

        return cleaned, report
