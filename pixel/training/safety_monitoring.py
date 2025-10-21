"""
Safety Monitoring System (Tier 1.14)

Model-free, content-based safety monitoring utilities that:
- Detect safety issues using ContentFilter (PII, safety gates, validations)
- Aggregate counters and severities
- Emit alerts when thresholds are exceeded
- Export metrics in Prometheus text format or JSON
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional
from time import time
import json

# Reuse the resilient shim path from evaluation metrics for ContentFilter
try:
    from ai.pixel.training.content_filtering import ContentFilter, ValidationSeverity
except Exception:  # pragma: no cover
    from enum import Enum
    from dataclasses import dataclass

    class ValidationSeverity(Enum):
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"
        CRITICAL = "critical"

    @dataclass
    class _Validation:
        validation_rule: str
        is_valid: bool
        severity: ValidationSeverity
        details: dict

    @dataclass
    class _GateResult:
        gate_name: str
        passed: bool
        severity: ValidationSeverity
        details: dict

    class ContentFilter:  # type: ignore
        def __init__(self, enable_crisis_integration: bool = True) -> None:
            self.enable_crisis_integration = enable_crisis_integration

        def validate_content(self, text: str):
            t = text.lower()
            invalid = []
            if any(k in t for k in ["take 50mg", "prescribe", "diagnose you"]):
                invalid.append(_Validation("medical_advice", False, ValidationSeverity.ERROR, {"reason": "medical advice heuristic"}))
            return invalid

        def enforce_safety_gates(self, text: str):
            t = text.lower()
            violations = []
            if any(k in t for k in ["hate", "racial slur", "kill them all"]):
                violations.append(_GateResult("hate_speech_gate", False, ValidationSeverity.CRITICAL, {"hit": "hate heuristic"}))
            return violations

        def detect_pii(self, text: str):
            t = text
            hits = []
            if any(sym in t for sym in ["@", "+", "-", "(", ")"]) and any(c.isdigit() for c in t):
                hits.append({"type": "contact_hint"})
            return hits


@dataclass
class SafetyEvent:
    source: str
    timestamp: float
    severity: str
    category: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SafetyMonitor:
    def __init__(self, alert_callback: Optional[Callable[[SafetyEvent], None]] = None) -> None:
        self.filter = ContentFilter(enable_crisis_integration=True)
        self.alert_callback = alert_callback
        self.counters: Dict[str, int] = {
            "pii": 0,
            "gate_violations": 0,
            "invalidations": 0,
            "severity_info": 0,
            "severity_warning": 0,
            "severity_error": 0,
            "severity_critical": 0,
            "events_total": 0,
        }
        # Thresholds for alerting per category
        self.thresholds: Dict[str, int] = {
            "gate_violations": 1,
            "invalidations": 3,
            "pii": 5,
        }

    def record_from_text(self, text: str, source: str = "unknown") -> List[SafetyEvent]:
        events: List[SafetyEvent] = []
        ts = time()
        validations = self.filter.validate_content(text)
        gates = self.filter.enforce_safety_gates(text)
        pii = self.filter.detect_pii(text)

        # Invalidations
        for v in validations:
            if not getattr(v, "is_valid", True):
                sev = getattr(v, "severity", ValidationSeverity.WARNING)
                ev = SafetyEvent(source=source, timestamp=ts, severity=sev.value if hasattr(sev, "value") else str(sev), category="invalidations", details={"rule": getattr(v, "validation_rule", "unknown"), "details": getattr(v, "details", {})})
                events.append(ev)
                self._inc("invalidations")
                self._inc(f"severity_{ev.severity}")
        # Gates
        had_gate_violation = False
        for g in gates:
            if not getattr(g, "passed", True):
                had_gate_violation = True
                sev = getattr(g, "severity", ValidationSeverity.ERROR)
                ev = SafetyEvent(source=source, timestamp=ts, severity=sev.value if hasattr(sev, "value") else str(sev), category="gate_violations", details={"gate": getattr(g, "gate_name", "unknown"), "details": getattr(g, "details", {})})
                events.append(ev)
                self._inc("gate_violations")
                self._inc(f"severity_{ev.severity}")
        # Heuristic fallback for egregious phrases if no gate flagged
        if not had_gate_violation:
            t = text.lower()
            if any(p in t for p in ["kill them all", "racial slur", "hate crime"]):
                ev = SafetyEvent(source=source, timestamp=ts, severity="critical", category="gate_violations", details={"gate": "heuristic_hate_speech", "details": {"hit": True}})
                events.append(ev)
                self._inc("gate_violations")
                self._inc("severity_critical")
        # PII
        if pii:
            ev = SafetyEvent(source=source, timestamp=ts, severity="warning", category="pii", details={"count": len(pii)})
            events.append(ev)
            self._inc("pii", len(pii))
            self._inc("severity_warning")

        # Tally total
        self._inc("events_total", len(events))

        # Alerts
        for e in events:
            if self._should_alert(e.category):
                self._emit_alert(e)

        return events

    def _inc(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def _should_alert(self, category: str) -> bool:
        thr = self.thresholds.get(category)
        return thr is not None and self.counters.get(category, 0) >= thr

    def _emit_alert(self, ev: SafetyEvent) -> None:
        if self.alert_callback:
            self.alert_callback(ev)

    # Metrics exporters
    def export_prometheus(self) -> str:
        lines = []
        for k, v in self.counters.items():
            metric = k.replace("-", "_")
            lines.append(f"pixel_{metric} {v}")
        return "\n".join(lines) + "\n"

    def export_json(self) -> str:
        return json.dumps({"counters": self.counters}, ensure_ascii=False)
