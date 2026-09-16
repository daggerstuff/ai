"""Quadit review orchestration — run the judges and auditors over audit items."""

from __future__ import annotations

import json
import logging
from typing import Protocol

from ai.research.quadit.models import AuditItem, Finding, PersonaVerdict, QuadAuditReport
from ai.research.quadit.personas import (
    AUDITOR_ROLE,
    CLINICAL_ACCURACY_JUDGE,
    JUDGE_ROLE,
    TRAINING_SIGNAL_JUDGE,
    VOICE_FIDELITY_JUDGE,
    AuditorDescriptor,
    JudgePersona,
)
from ai.research.quadit.rubric import PASS_THRESHOLD, severity_weight

logger = logging.getLogger(__name__)

DEFAULT_JUDGES: tuple[JudgePersona, ...] = (
    VOICE_FIDELITY_JUDGE,
    CLINICAL_ACCURACY_JUDGE,
    TRAINING_SIGNAL_JUDGE,
)


class QuaditLLMClient(Protocol):
    """Minimal LLM contract — adapters inject their own client."""

    def chat(self, prompt: str, *, temperature: float = 0.3) -> str:
        """Return the model's raw text response for the prompt."""
        ...


def _parse_verdict_json(raw: str) -> dict[str, object]:
    """Parse a judge response, tolerating markdown code fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    parsed: dict[str, object] = json.loads(text)
    return parsed


def _verdict_from_response(
    persona: str,
    role: str,
    parsed: dict[str, object],
) -> PersonaVerdict:
    score = float(parsed.get("score", 0.0))
    passed = bool(parsed.get("passed", score >= PASS_THRESHOLD))
    flagged_raw = parsed.get("flagged_ids", [])
    findings_raw = parsed.get("findings", [])
    findings: list[Finding] = []
    if isinstance(findings_raw, list):
        for entry in findings_raw:
            if isinstance(entry, dict):
                findings.append(
                    Finding(
                        item_id=str(entry.get("item_id", "")),
                        severity=str(entry.get("severity", "info")),
                        signature=str(entry.get("signature", "")),
                        rationale=str(entry.get("rationale", "")),
                        example_excerpt=str(entry.get("example_excerpt", "")),
                    )
                )
    return PersonaVerdict(
        persona=persona,
        role=role,
        passed=passed,
        score=score,
        notes=str(parsed.get("notes", "")),
        flagged_ids=[str(fid) for fid in flagged_raw] if isinstance(flagged_raw, list) else [],
        findings=findings,
    )


def _run_judge_llm(
    judge: JudgePersona,
    items: list[AuditItem],
    client: QuaditLLMClient,
) -> PersonaVerdict:
    prompt = judge.prompt_builder(items)
    try:
        raw = client.chat(prompt)
        parsed = _parse_verdict_json(raw)
    except Exception as exc:
        logger.error("Judge %s failed: %s", judge.name, exc)
        return PersonaVerdict(
            persona=judge.name,
            role=JUDGE_ROLE,
            passed=False,
            score=0.0,
            notes=f"Judge call failed: {exc}",
        )
    return _verdict_from_response(judge.name, JUDGE_ROLE, parsed)


def _run_judge_deterministic(judge: JudgePersona, items: list[AuditItem]) -> PersonaVerdict:
    score, flagged = judge.deterministic_scorer(items)
    return PersonaVerdict(
        persona=judge.name,
        role=JUDGE_ROLE,
        passed=score >= PASS_THRESHOLD,
        score=score,
        notes=(
            "Deterministic mode: pattern-based scan only, not a clinical judgment."
            if flagged
            else "Deterministic mode: no banned patterns detected."
        ),
        flagged_ids=flagged,
    )


def _run_auditor_deterministic(
    descriptor: AuditorDescriptor,
    items: list[AuditItem],
) -> PersonaVerdict:
    """Pattern-based platitude/anti-signal scan in the auditor's vocabulary."""
    findings: list[Finding] = []
    lowered_signals = [s.lower() for s in descriptor.sample_signature_strings]
    platitude_markers = (
        "treat yourself kindly",
        "be the change",
        "it's ok to not be ok",
        "i'm here for you",
        "i understand this is a challenging time",
    )
    for item in items:
        text_lower = item.content.lower()
        if any(marker in text_lower for marker in platitude_markers):
            findings.append(
                Finding(
                    item_id=item.id,
                    severity="warning",
                    signature="platitude-without-cost",
                    rationale=(
                        "Platitude pattern present without behavioral backing "
                        "(deterministic mode: literal pattern match)."
                    ),
                    example_excerpt=item.content[:400],
                )
            )
        elif any(signal.lower() in text_lower for signal in lowered_signals):
            findings.append(
                Finding(
                    item_id=item.id,
                    severity="info",
                    signature=lowered_signals[next(i for i, s in enumerate(lowered_signals) if s in text_lower)],
                    rationale="Sample signature string present (deterministic mode).",
                    example_excerpt=item.content[:400],
                )
            )
    if not items:
        score, passed = 0.0, False
    else:
        score = 1.0 - (len(findings) / len(items))
        passed = not any(f.severity == "critical" for f in findings)
    return PersonaVerdict(
        persona=descriptor.name,
        role=AUDITOR_ROLE,
        passed=passed,
        score=score,
        notes="Deterministic mode: literal pattern scan against the descriptor's anti-signal vocabulary.",
        flagged_ids=[f.item_id for f in findings],
        findings=findings,
    )


def run_quadit_audit(
    items: list[AuditItem],
    *,
    client: QuaditLLMClient | None = None,
    model: str = "deterministic",
    judges: tuple[JudgePersona, ...] = DEFAULT_JUDGES,
    auditors: tuple[AuditorDescriptor, ...] = (),
) -> QuadAuditReport:
    """Run the quadit: three clinical judges plus adversarial auditors.

    With ``client=None`` the audit runs in deterministic mode — pattern-based
    scans, no LLM. With a client, judges receive their full prompts and the
    auditors audit through their persona lenses.
    """
    verdicts: list[PersonaVerdict] = []

    if not items:
        return QuadAuditReport(
            passed=False,
            mode="deterministic" if client is None else "llm",
            model=model,
            item_count=0,
            verdicts=[
                PersonaVerdict(
                    persona=judge.name,
                    role=JUDGE_ROLE,
                    passed=False,
                    score=0.0,
                    notes="No items to audit.",
                )
                for judge in judges
            ],
            summary="Quadit audit rejected: no items submitted.",
        )

    for judge in judges:
        if client is not None:
            verdicts.append(_run_judge_llm(judge, items, client))
        else:
            verdicts.append(_run_judge_deterministic(judge, items))

    for auditor in auditors:
        if client is not None:
            try:
                raw = client.chat(auditor.audit_prompt(items))
                parsed = _parse_verdict_json(raw)
            except Exception as exc:
                logger.error("Auditor %s failed: %s", auditor.name, exc)
                verdicts.append(
                    PersonaVerdict(
                        persona=auditor.name,
                        role=AUDITOR_ROLE,
                        passed=False,
                        score=0.0,
                        notes=f"Auditor call failed: {exc}",
                    )
                )
                continue
            verdicts.append(_verdict_from_response(auditor.name, AUDITOR_ROLE, parsed))
        else:
            verdicts.append(_run_auditor_deterministic(auditor, items))

    critical_count = sum(1 for v in verdicts for f in v.findings if f.severity == "critical")
    warning_count = sum(
        1 for v in verdicts for f in v.findings if severity_weight(f.severity) > 0 and f.severity != "critical"
    )
    judges_passed = all(v.passed for v in verdicts if v.role == JUDGE_ROLE)
    auditors_ok = not any(f.severity == "critical" for v in verdicts if v.role == AUDITOR_ROLE for f in v.findings)
    passed = judges_passed and auditors_ok

    failed_names = [v.persona for v in verdicts if not v.passed]
    summary = (
        f"Quadit audit passed ({len(verdicts)} personas, {len(items)} items)."
        if passed
        else f"Quadit audit failed: {', '.join(failed_names) or 'critical finding'} "
        f"({critical_count} critical, {warning_count} warnings, {len(items)} items)."
    )

    return QuadAuditReport(
        passed=passed,
        mode="deterministic" if client is None else "llm",
        model=model,
        item_count=len(items),
        verdicts=verdicts,
        critical_count=critical_count,
        warning_count=warning_count,
        summary=summary,
    )
