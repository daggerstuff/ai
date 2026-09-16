"""Quadit personas — the three clinical judges plus TOML auditor descriptors."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai.research.quadit.models import AuditItem
from ai.research.quadit.rubric import BANNED_PHRASES, DIRECTOR_SUPERLATIVES

DESCRIPTOR_DIR = Path(__file__).parent

JUDGE_ROLE = "judge"
AUDITOR_ROLE = "adversarial_auditor"


@dataclass(frozen=True)
class JudgePersona:
    """A clinical judge: name, mission, prompt builder, and deterministic scorer."""

    name: str
    mission: str
    prompt_builder: Any  # Callable[[list[AuditItem]], str]
    deterministic_scorer: Any  # Callable[[list[AuditItem]], tuple[float, list[str]]]


def _format_items(items: list[AuditItem]) -> str:
    blocks: list[str] = []
    for item in items:
        header = f"[{item.id}] ({item.kind}) {item.author_role}".rstrip()
        blocks.append(f"{header}\n{item.content[:400]}")
    return "\n\n---\n\n".join(blocks) or "(no items)"


# ---------------------------------------------------------------------------
# Judge 1 — Voice Fidelity
# ---------------------------------------------------------------------------


def _voice_fidelity_prompt(items: list[AuditItem]) -> str:
    samples = _format_items(items)
    return f"""You are the "Voice Fidelity Auditor" — a clinical communication expert.
Evaluate whether the content below is consistent with the voice it claims to carry
(author roles are annotated per item).

HARD CONSTRAINTS (flag any violation as CRITICAL):
- Clinical directors never use "amazing", "incredible", "game-changing".
- Nobody uses: "circle back", "double-click", "synergize", "I hope this finds you well".
- Platform/infra roles do not engage in clinical discussion — they redirect to infra/security.
- No job-title signatures in internal communications.
- All patient references must use de-identified case codes, never real names.
- Trainee voice must evolve: deferential early, confident later.

ITEMS TO EVALUATE:
{samples}

Respond in JSON with this exact structure:
{{
  "score": <float 0.0-1.0>,
  "passed": <bool>,
  "notes": "<2-4 sentence assessment>",
  "flagged_ids": ["<id1>", ...]
}}

Score >= 0.7 = pass. Respond ONLY with JSON. No markdown.
"""


def _voice_fidelity_score(items: list[AuditItem]) -> tuple[float, list[str]]:
    """Deterministic fallback: banned-phrase and superlative scan."""
    banned = tuple(p.lower() for p in BANNED_PHRASES)
    flagged: list[str] = []
    for item in items:
        text_lower = item.content.lower()
        if any(phrase in text_lower for phrase in banned) or any(word in text_lower for word in DIRECTOR_SUPERLATIVES):
            flagged.append(item.id)
    score = 1.0 - (len(flagged) / len(items)) if items else 0.0
    return score, flagged


VOICE_FIDELITY_JUDGE = JudgePersona(
    name="Voice Fidelity Auditor",
    mission="Does each voice sound like itself?",
    prompt_builder=_voice_fidelity_prompt,
    deterministic_scorer=_voice_fidelity_score,
)


# ---------------------------------------------------------------------------
# Judge 2 — Clinical Accuracy
# ---------------------------------------------------------------------------


def _clinical_accuracy_prompt(items: list[AuditItem]) -> str:
    samples = _format_items(items)
    return f"""You are the "Clinical Accuracy Reviewer" — a licensed clinical psychologist.
Find clinical errors, safety violations, or misrepresentations of therapeutic practice.

You are looking for:
- Incorrect use of clinical terminology (wrong DSM-5-TR criteria, misnamed techniques, wrong assessment tools)
- Safety violations (patient identifying information, confidentiality breaches, inappropriate clinical advice)
- Misrepresentation of modalities (CBT, DBT, MI, EMDR, TF-CBT described incorrectly)
- Risk assessment errors (wrong protocol, incorrect stratification, missing safety planning)
- Ethical violations (boundary crossings, dual relationships, inappropriate supervisor-trainee dynamics)
- Clinical plausibility (would a real clinical team communicate this way?)

ITEMS TO EVALUATE:
{samples}

Respond in JSON:
{{
  "score": <float 0.0-1.0 — higher is MORE clinically accurate>,
  "passed": <bool — pass if score >= 0.7>,
  "notes": "<sharpest clinical finding + 1-2 secondary observations>",
  "flagged_ids": ["<id of the worst item>"]
}}

Respond ONLY with JSON. No markdown.
"""


def _clinical_accuracy_score(items: list[AuditItem]) -> tuple[float, list[str]]:
    """Deterministic fallback: PHI-shaped and unsafe-advice pattern scan.

    Deliberately narrow — the deterministic mode documents that it is a
    pattern check, not a clinical judgment; the LLM judge carries the
    clinical reasoning.
    """
    flagged: list[str] = []
    phi_patterns = (
        "ssn is",
        "social security number is",
        "date of birth is",
        "patient name:",
        "full name:",
    )
    for item in items:
        text_lower = item.content.lower()
        if any(pattern in text_lower for pattern in phi_patterns):
            flagged.append(item.id)
    score = 1.0 - (len(flagged) / len(items)) if items else 0.0
    return score, flagged


CLINICAL_ACCURACY_JUDGE = JudgePersona(
    name="Clinical Accuracy Reviewer",
    mission="Are there clinical errors, safety violations, or misrepresentations?",
    prompt_builder=_clinical_accuracy_prompt,
    deterministic_scorer=_clinical_accuracy_score,
)


# ---------------------------------------------------------------------------
# Judge 3 — Training Signal
# ---------------------------------------------------------------------------


def _training_signal_prompt(items: list[AuditItem]) -> str:
    samples = _format_items(items)
    return f"""You are the "Training Signal Engineer" — a technical evaluator assessing
content quality as AI clinical-training data. Identify issues that would degrade a
model trained on this content.

You are checking for:
- Training signal pollution (repeated patterns that would overfit a model)
- Label leakage (content that reveals its own generation — LLM-style hedging, over-explanation)
- Semantic diversity (different voices discussing different clinical topics, or topic monoculture)
- Hallucination artifacts (tools, people, or events that don't exist)
- Format consistency (IDs parseable, required fields present)
- De-identification compliance (case codes only, no real patient names)

ITEMS TO EVALUATE:
{samples}

Respond in JSON:
{{
  "score": <float 0.0-1.0>,
  "passed": <bool — pass if score >= 0.7>,
  "notes": "<what would degrade a model trained here?>",
  "flagged_ids": ["<ids of the worst items>"]
}}

Respond ONLY with JSON. No markdown.
"""


def _training_signal_score(items: list[AuditItem]) -> tuple[float, list[str]]:
    """Deterministic fallback: generation-artifact hedging scan."""
    flagged: list[str] = []
    hedging = (
        "as an ai",
        "i'm just an ai",
        "i cannot provide medical advice",
        "consult a licensed professional",
    )
    for item in items:
        text_lower = item.content.lower()
        if any(phrase in text_lower for phrase in hedging):
            flagged.append(item.id)
    score = 1.0 - (len(flagged) / len(items)) if items else 0.0
    return score, flagged


TRAINING_SIGNAL_JUDGE = JudgePersona(
    name="Training Signal Engineer",
    mission="Would training data from this content produce a clinically competent model?",
    prompt_builder=_training_signal_prompt,
    deterministic_scorer=_training_signal_score,
)


# ---------------------------------------------------------------------------
# Adversarial auditor descriptors (TOML)
# ---------------------------------------------------------------------------


class AuditorDescriptor:
    """A loaded adversarial-auditor persona descriptor (e.g. brene_brown.toml)."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.name: str = str(data.get("name", "unnamed-auditor"))
        self.role: str = str(data.get("role", "adversarial auditor"))
        self.auditor_set: str = str(data.get("auditor_set", "quadit"))
        self.tone: str = str(data.get("tone", ""))
        self.quirks: list[str] = [str(q) for q in data.get("quirks", [])]
        self.signature_questions: list[str] = [str(q) for q in data.get("auditor_signature_questions", [])]
        self.vocabulary_signals: list[str] = [str(v) for v in data.get("auditor_vocabulary_signals", [])]
        self.anti_signals: list[str] = [str(a) for a in data.get("auditor_anti_signals", [])]
        self.anti_signal_keys: list[str] = [str(a).split(" (")[0] for a in self.anti_signals]
        self.sample_signature_strings: list[str] = [str(s) for s in data.get("auditor_sample_signature_strings", [])]
        rubric = data.get("auditor_severity_rubric", {})
        self.severity_rubric: dict[str, str] = {str(k): str(v) for k, v in rubric.items()}

    def audit_prompt(self, items: list[AuditItem]) -> str:
        samples = _format_items(items)
        questions = "\n".join(f"- {q}" for q in self.signature_questions)
        anti = "\n".join(f"- {a}" for a in self.anti_signals)
        return f"""You are "{self.name}" ({self.role}) in the {self.auditor_set} adversarial
audit set. You audit content adversarially through the lens of your persona.

Your tone: {self.tone}

Signature questions you ask of every item:
{questions}

ANTI-SIGNALS you hunt (label findings with these):
{anti}

Severity rubric:
- critical: {self.severity_rubric.get("critical", "material falsification of the persona frame")}
- warning: {self.severity_rubric.get("warning", "erosion of specific emotional granularity")}
- info: {self.severity_rubric.get("info", "incidental armor-flavored word choice")}

ITEMS TO AUDIT:
{samples}

Respond in JSON:
{{
  "score": <float 0.0-1.0 — how well the content holds up against your persona>,
  "passed": <bool>,
  "notes": "<2-4 sentence adversarial assessment>",
  "flagged_ids": ["<id1>", ...],
  "findings": [
    {{
      "item_id": "<id>",
      "severity": "info" | "warning" | "critical",
      "signature": "<anti-signal label>",
      "rationale": "<why>",
      "example_excerpt": "<verbatim excerpt, <= 400 chars>"
    }}
  ]
}}

A critical finding flips the audit to FAIL. Respond ONLY with JSON. No markdown.
"""


def load_auditor_descriptor(name: str = "brene_brown") -> AuditorDescriptor:
    """Load a TOML auditor descriptor shipped with the quadit package."""
    path = DESCRIPTOR_DIR / f"{name}.toml"
    with path.open("rb") as f:
        data: dict[str, Any] = tomllib.load(f)
    return AuditorDescriptor(data)
