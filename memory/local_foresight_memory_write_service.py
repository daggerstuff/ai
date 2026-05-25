from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .gates import GateDecision, GateResult, GatingReport
from .gates.consent_gate import ConsentGateChecker
from .gates.crisis_detector import CrisisDetector
from .gates.pii_redactor import PiiRedactionResult, PiiRedactor
from .gates.trauma_filter import TraumaFilter
from .local_foresight_protocol_adapter import LocalForesightProtocolAdapter


class LocalForesightMemoryWriteService:
    """Create normalized memory payloads before they are retained."""

    def __init__(
        self,
        *,
        protocol: LocalForesightProtocolAdapter,
        default_bank_id: str,
        pii_redactor: PiiRedactor | None = None,
        crisis_detector: CrisisDetector | None = None,
        trauma_filter: TraumaFilter | None = None,
        consent_gate: ConsentGateChecker | None = None,
    ) -> None:
        self.protocol = protocol
        self.default_bank_id = default_bank_id
        self.pii_redactor = pii_redactor or PiiRedactor()
        self.crisis_detector = crisis_detector or CrisisDetector()
        self.trauma_filter = trauma_filter or TraumaFilter()
        self.consent_gate = consent_gate or ConsentGateChecker()

    @staticmethod
    def _metadata_dict(metadata: Any | None) -> dict[str, Any]:
        if metadata is None:
            return {}
        if isinstance(metadata, dict):
            return dict(metadata)
        if hasattr(metadata, "to_dict") and callable(metadata.to_dict):
            data = metadata.to_dict()
            return dict(data) if isinstance(data, dict) else {}
        raise TypeError("metadata must be a mapping or expose to_dict()")

    def coerce_metadata(self, metadata: Any | None) -> dict[str, Any]:
        return self._metadata_dict(metadata)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat()

    def prepare_metadata(
        self,
        *,
        metadata: Any | None,
        category: str | None,
        scope_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged = self._metadata_dict(metadata)
        if scope_metadata:
            merged.update(scope_metadata)
        if category:
            merged["category"] = category
        merged.setdefault("timestamp", self._utc_now())
        return merged

    def add_memory(
        self,
        *,
        content: str,
        user_id: str,
        metadata: Any | None = None,
        category: str | None = None,
        scope_metadata: dict[str, Any] | None = None,
    ) -> str:
        merged = self.prepare_metadata(
            metadata=metadata,
            category=category,
            scope_metadata=scope_metadata,
        )
        retained = self.protocol.retain_items(
            self.default_bank_id,
            [self.protocol.build_add_memory_item(user_id=user_id, content=content, metadata=merged)],
        )
        results = retained.get("results")
        if not isinstance(results, list) or not results:
            raise RuntimeError("Retain operation returned no document identifiers")
        first = results[0]
        if not isinstance(first, dict):
            raise RuntimeError("Retain operation returned an invalid document payload")
        document_id = first.get("id")
        if not isinstance(document_id, str) or not document_id:
            raise RuntimeError("Retain operation did not provide a valid document identifier")
        return document_id

    def evaluate_gates(
        self,
        *,
        content: str,
        user_id: str,
        memory_id: str | None = None,
    ) -> tuple[GatingReport, PiiRedactionResult | None]:
        """Run all 4 ingestion gates on content before retention.

        Returns (report, cached_pii_result) so caller can reuse redaction.
        """
        if not content or not content.strip():
            report = GatingReport(source_id=memory_id or "unknown", content=content)
            report.gate0_pii = GateResult(
                gate="gate0_pii",
                decision=GateDecision.BLOCK,
                reason="empty or whitespace-only content",
            )
            return report, None

        report = GatingReport(source_id=memory_id or "unknown", content=content)

        pii_result = self.pii_redactor.redact(content)
        report.gate0_pii = self.pii_redactor.evaluate(content)
        report.pii_types_found = list(pii_result.pii_types_found)

        if report.gate0_pii.decision == GateDecision.BLOCK:
            return report, pii_result

        crisis_result = self.crisis_detector.detect(content)
        report.gate1_crisis = self.crisis_detector.evaluate(content)
        report.crisis_tier = crisis_result.tier.value

        if report.gate1_crisis.decision == GateDecision.BLOCK:
            return report, pii_result

        trauma_result = self.trauma_filter.filter(content, user_id)
        report.gate2_trauma = self.trauma_filter.evaluate(content, user_id)
        report.trauma_indicators = list(trauma_result.indicators)

        consent_result = self.consent_gate.check_consent(user_id, memory_id)
        report.gate3_consent = GateResult(
            gate="gate3_consent",
            decision=GateDecision.BLOCK if not consent_result.allowed else GateDecision.PASS,
            reason=consent_result.reason,
            details=[f"consent_tier: {consent_result.consent_tier.value}"],
        )
        ct = consent_result.consent_tier
        report.consent_gate_value = ct.value if hasattr(ct, "value") else ct

        return report, pii_result

    def gated_add_memory(
        self,
        *,
        content: str,
        user_id: str,
        metadata: Any | None = None,
        category: str | None = None,
        scope_metadata: dict[str, Any] | None = None,
    ) -> tuple[str | None, GatingReport]:
        """Evaluate gates, then retain only if all gates pass and no review is needed.

        Returns (document_id, gating_report). document_id is None if blocked or needs review.
        """
        try:
            report, pii_result = self.evaluate_gates(content=content, user_id=user_id)
        except Exception as exc:
            gate_id = getattr(exc, "gate", getattr(exc, "failed_gate", None))
            if gate_id:
                report = GatingReport(source_id="unknown", content=content)
                setattr(
                    report,
                    gate_id,
                    GateResult(
                        gate=gate_id,
                        decision=GateDecision.BLOCK,
                        reason=str(exc),
                    ),
                )
                return None, report
            raise

        if not report.can_retain:
            return None, report

        scrubbed = pii_result.scrubbed_text if pii_result else content
        merged = self.prepare_metadata(
            metadata=metadata,
            category=category,
            scope_metadata=scope_metadata,
        )
        merged["gating_report"] = report.to_dict()
        merged["crisis_flag"] = report.crisis_tier in ("critical", "high")
        merged["trauma_indicators"] = report.trauma_indicators

        doc_id = self.add_memory(
            content=scrubbed,
            user_id=user_id,
            metadata=merged,
            category=category,
            scope_metadata=scope_metadata,
        )
        report.source_id = doc_id
        return doc_id, report
