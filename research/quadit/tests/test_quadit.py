"""Tests for the quadit audit package (deterministic mode + fake-LLM mode)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from ai.research.quadit import (
    AuditItem,
    Finding,
    PersonaVerdict,
    QuadAuditReport,
    load_auditor_descriptor,
    run_quadit_audit,
)
from ai.research.quadit.personas import (
    CLINICAL_ACCURACY_JUDGE,
    TRAINING_SIGNAL_JUDGE,
    VOICE_FIDELITY_JUDGE,
)
from ai.research.quadit.review import DEFAULT_JUDGES, QuaditLLMClient, _parse_verdict_json
from ai.research.quadit.rubric import PASS_THRESHOLD, severity_weight


def make_item(
    item_id: str = "i1",
    content: str = "We reviewed the session recording and adjusted the treatment plan.",
) -> AuditItem:
    return AuditItem(
        id=item_id,
        kind="ai_response",
        author_role="pe-service",
        content=content,
    )


# ---------------------------------------------------------------------------
# Deterministic mode
# ---------------------------------------------------------------------------


def test_deterministic_clean_items_pass() -> None:
    report = run_quadit_audit([make_item("r1"), make_item("r2")])
    assert report.passed
    assert report.mode == "deterministic"
    n_items = 2
    assert report.item_count == n_items
    assert len(report.verdicts) == len(DEFAULT_JUDGES)
    assert all(v.passed for v in report.verdicts)
    assert report.critical_count == 0


def test_deterministic_platitude_flagged_by_brene() -> None:
    brene = load_auditor_descriptor("brene_brown")
    plat = make_item(
        "plat",
        "I'm here for you. It's ok to not be ok. Treat yourself kindly.",
    )
    clean = make_item("clean")
    report = run_quadit_audit([clean, plat], auditors=(brene,))
    brene_verdict = next(v for v in report.verdicts if v.persona == "Brené Brown")
    assert "plat" in brene_verdict.flagged_ids
    assert any(f.severity == "warning" for f in brene_verdict.findings)
    # A warning does not flip the audit
    assert report.passed


def test_deterministic_banned_phrase_fails_voice_fidelity() -> None:
    corporate = make_item("corp", "Great point — let's circle back on that next week.")
    report = run_quadit_audit([corporate])
    voice = next(v for v in report.verdicts if v.persona == VOICE_FIDELITY_JUDGE.name)
    assert not voice.passed
    assert "corp" in voice.flagged_ids
    assert not report.passed


def test_deterministic_phi_pattern_fails_clinical_accuracy() -> None:
    phi = make_item("phi", "Patient name: Jane Doe. SSN is 123-45-6789.")
    report = run_quadit_audit([phi])
    clinical = next(v for v in report.verdicts if v.persona == CLINICAL_ACCURACY_JUDGE.name)
    assert not clinical.passed
    assert "phi" in clinical.flagged_ids


def test_deterministic_hedging_fails_training_signal() -> None:
    hedge = make_item("hedge", "As an AI, I cannot provide medical advice.")
    report = run_quadit_audit([hedge])
    signal = next(v for v in report.verdicts if v.persona == TRAINING_SIGNAL_JUDGE.name)
    assert not signal.passed
    assert "hedge" in signal.flagged_ids


def test_deterministic_empty_items_rejected() -> None:
    report = run_quadit_audit([])
    assert not report.passed
    assert report.item_count == 0
    assert "no items" in report.summary.lower()


def test_empty_content_rejected_by_model() -> None:
    with pytest.raises(ValueError, match="content"):
        AuditItem(id="bad", kind="ai_response", author_role="pe-service", content="")


# ---------------------------------------------------------------------------
# LLM mode (fake client)
# ---------------------------------------------------------------------------


class FakeClient:
    """Scriptable QuaditLLMClient — returns canned JSON per call, or raises."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.temperatures: list[float] = []

    def chat(self, prompt: str, *, temperature: float = 0.3) -> str:
        self.prompts.append(prompt)
        self.temperatures.append(temperature)
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _judge_json(score: float, flagged: list[str] | None = None) -> str:
    return json.dumps(
        {
            "score": score,
            "passed": score >= PASS_THRESHOLD,
            "notes": "Looks fine.",
            "flagged_ids": flagged or [],
        }
    )


def test_llm_mode_all_pass() -> None:
    client: QuaditLLMClient = FakeClient([_judge_json(0.9)] * 3)
    report = run_quadit_audit([make_item()], client=client)
    assert report.mode == "llm"
    assert report.passed
    assert len(report.verdicts) == len(DEFAULT_JUDGES)


def test_llm_mode_parses_code_fences() -> None:
    fenced = f"```json\n{_judge_json(0.85)}\n```"
    client: QuaditLLMClient = FakeClient([fenced] * 3)
    report = run_quadit_audit([make_item()], client=client)
    assert report.passed
    assert all(v.score == pytest.approx(0.85) for v in report.verdicts)


def test_llm_mode_judge_failure_fails_safe() -> None:
    client: QuaditLLMClient = FakeClient([RuntimeError("ollama down"), _judge_json(0.9), _judge_json(0.9)])
    report = run_quadit_audit([make_item()], client=client)
    assert not report.passed
    failed = [v for v in report.verdicts if not v.passed]
    assert len(failed) == 1
    assert "failed" in failed[0].notes.lower()


def test_llm_mode_auditor_critical_finding_flips_fail() -> None:
    brene = load_auditor_descriptor("brene_brown")
    critical = json.dumps(
        {
            "score": 0.3,
            "passed": False,
            "notes": "Platitudes without cost.",
            "flagged_ids": ["i1"],
            "findings": [
                {
                    "item_id": "i1",
                    "severity": "critical",
                    "signature": "armor-of-optimism",
                    "rationale": "All optimism, no practice.",
                    "example_excerpt": "It's ok to not be ok.",
                }
            ],
        }
    )
    client: QuaditLLMClient = FakeClient([_judge_json(0.9)] * 3 + [critical])
    report = run_quadit_audit([make_item()], client=client, auditors=(brene,))
    assert not report.passed
    assert report.critical_count == 1
    assert "critical" in report.summary.lower()


def test_parse_verdict_json_strips_fences() -> None:
    assert _parse_verdict_json('```json\n{"score": 1}\n```') == {"score": 1}
    assert _parse_verdict_json('  {"score": 1}  ') == {"score": 1}


# ---------------------------------------------------------------------------
# Models / rubric
# ---------------------------------------------------------------------------


def test_finding_severity_weights() -> None:

    assert severity_weight("critical") > severity_weight("warning")
    assert severity_weight("warning") > severity_weight("info")
    assert severity_weight("info") == 0


def test_report_is_pydantic_and_serializable() -> None:
    report = QuadAuditReport(
        passed=True,
        mode="deterministic",
        model="deterministic",
        item_count=1,
        verdicts=[
            PersonaVerdict(
                persona="x",
                role="judge",
                passed=True,
                score=0.9,
                notes="ok",
            )
        ],
        summary="ok",
    )
    dumped: dict[str, Any] = report.model_dump()
    assert dumped["passed"] is True
    assert dumped["verdicts"][0]["persona"] == "x"


def test_finding_model_fields() -> None:
    f = Finding(
        item_id="i1",
        severity="warning",
        signature="platitude-without-cost",
        rationale="why",
        example_excerpt="excerpt",
    )
    assert f.severity == "warning"
