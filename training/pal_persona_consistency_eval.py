"""PAL persona-consistency evaluation (paper §C.score).

PIX-4078 — Phase 6: Evaluation. Ports the Persona-Aware Alignment evaluation
metric from arxiv:2511.10215v1 (Li et al.) so we can measure whether PAL
training actually improves persona adherence — the one phase the original
implementation plan omitted entirely.

Paper definition (verbatim, §4):
    ``NLI(p_l, r_n) = 1 if p_l entails r_n; 0 if independent; -1 if p_l
    contradicts r_n.`` where ``p_l`` is the persona line and ``r_n`` the
    generated response. The C.score is the mean over the dialogue.

This module scores each (persona, response) pair with an NLI entailment
classifier and reduces to the C.score. It deliberately mirrors the paper's
*direction*: persona entails response ⇒ persona-consistent (+1), neutral
⇈ 0, contradiction ⇒ -1.

Backends
--------
* **CrossEncoder** (default, real): a ``sentence-transformers`` CrossEncoder
  NLI model (e.g. ``cross-encoder/nli-deberta-v3-base``). ``sentence-transformers``
  is already a declared dependency; no new dep is introduced. Loads
  lazily so the module imports without a model download.
* **Heuristic** (fallback / offline / tests): a deterministic keyword-based
  approximation that fires when the NLI backend cannot load (no network,
  missing model). It is NOT a substitute for the real score in a results
  report — it exists so the evaluation runs in CI and the contract is
  exercised. ``score()`` records which backend produced the result.

The CLI reads a PAL-generated JSONL (one ``{persona, response}`` per line, or
a full SFT record where ``persona`` lives in metadata and ``response`` is the
assistant turn) and writes a JSON report with per-example and aggregate scores.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Paper's NLI score mapping. The CrossEncoder NLI family emits a 3-logit
# vector over (contradict, neutral, entail) — we map argmax to the paper's
# {-1, 0, +1} scale.
NLI_LABELS = ("contradiction", "neutral", "entailment")
SCORE_MAP = {"entailment": 1, "neutral": 0, "contradiction": -1}

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


@dataclass
class ConsistencyExample:
    persona: str
    response: str
    label: str  # one of NLI_LABELS
    score: int  # one of {-1, 0, +1}
    backend: str


@dataclass
class ConsistencyReport:
    n: int
    c_score: float  # mean of per-example scores
    entail_rate: float
    neutral_rate: float
    contradict_rate: float
    backend: str
    examples: list[ConsistencyExample] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class _HeuristicNli:
    """Deterministic offline NLI proxy for tests / no-network CI.

    This is a *coarse* approximation, not a real entailment model. It catches
    the most obvious personas-/response- consistencies the PAL corpus is
    built around (health literacy register, healthcare-seeking preference)
    and flags blatant contradictions. It exists only so the evaluation runs
    without a model download; do NOT cite its numbers as results. ``backend``
    is reported as ``"heuristic"`` so downstream reports can filter it out.
    """

    name = "heuristic"

    # Low-literacy persona signals vs. high-register response = contradiction.
    _LOW_LITERACY = ("low health literacy", "low literacy")
    # High-literacy/clinical-jargon response — the PAL *rejected* style.
    _JARGON = (
        "tertiary academic medical center",
        "expedited neuroimaging",
        "multi-disciplinary",
        "differential diagnosis",
        "clinical guidelines",
        "expedited neuro",
    )

    def predict(self, premise: str, hypothesis: str) -> str:
        premise_l = premise.lower()
        hypo_l = hypothesis.lower()
        if not premise.strip() or not hypothesis.strip():
            return "neutral"
        # Persona says low literacy; response is heavy medical jargon ⇒ contradict.
        if any(tok in premise_l for tok in self._LOW_LITERACY) and any(tok in hypo_l for tok in self._JARGON):
            return "contradiction"
        # Persona health-seeking preference echoed in the response ⇒ entail.
        for pref in ("traditional medicine", "modern medicine", "integrated medicine"):
            if pref in premise_l and pref in hypo_l:
                return "entailment"
        # Persona location echoed ⇒ weak entail signal.
        for loc in ("hanoi", "hcmc"):
            if loc in premise_l and loc in hypo_l:
                return "entailment"
        return "neutral"


class _CrossEncoderNli:
    """Real NLI backend backed by ``sentence-transformers.CrossEncoder``."""

    name = "cross-encoder-nli"

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL) -> None:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        self._model = CrossEncoder(model_name)
        self._model_name = model_name

    def predict(self, premise: str, hypothesis: str) -> str:
        # CrossEncoder NLI models return logits over (contradiction, neutral, entailment).
        logits = self._model.predict([(premise, hypothesis)])
        # ``predict`` may return a nested list for batched input; take the first row.
        row = logits[0] if hasattr(logits, "__getitem__") else logits
        idx = int(max(range(len(row)), key=lambda i: row[i]))
        return NLI_LABELS[idx]


def build_nli_backend(
    model_name: str | None = None,
    force_heuristic: bool = False,
) -> Any:
    """Construct the NLI backend, falling back to the heuristic when the
    cross-encoder model cannot be loaded (offline / missing weights).

    Returns an object with ``.predict(premise, hypothesis) -> label`` and a
    ``.name`` attribute identifying the backend used in the report.
    """
    if force_heuristic:
        return _HeuristicNli()
    try:
        return _CrossEncoderNli(model_name or DEFAULT_NLI_MODEL)
    except Exception as exc:
        logger.warning(
            "CrossEncoder NLI model %s could not load (%s); falling back to heuristic. "
            "Scores from this run are NOT publication-quality — set force_heuristic=False "
            "with a reachable model for real results.",
            model_name or DEFAULT_NLI_MODEL,
            exc,
        )
        return _HeuristicNli()


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _label_to_score(label: str) -> int:
    if label not in SCORE_MAP:
        raise ValueError(f"unexpected NLI label {label!r}; expected one of {NLI_LABELS}")
    return SCORE_MAP[label]


def score_example(persona: str, response: str, backend: Any) -> ConsistencyExample:
    if not isinstance(persona, str) or not persona.strip():
        raise ValueError("persona must be a non-empty string")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("response must be a non-empty string")
    label = backend.predict(persona, response)
    if label not in NLI_LABELS:
        raise ValueError(f"backend returned unknown label {label!r}")
    return ConsistencyExample(
        persona=persona,
        response=response,
        label=label,
        score=_label_to_score(label),
        backend=getattr(backend, "name", "unknown"),
    )


def score_pairs(
    pairs: Iterable[tuple[str, str]],
    backend: Any,
) -> ConsistencyReport:
    """Score an iterable of (persona, response) tuples → aggregate C.score."""
    examples: list[ConsistencyExample] = []
    for persona, response in pairs:
        examples.append(score_example(persona, response, backend))
    return _aggregate(examples)


def score_records(
    records: Iterable[dict[str, Any]],
    backend: Any,
) -> ConsistencyReport:
    """Score PAL-style records.

    Accepts either:
      * flat:     {"persona": str, "response": str}
      * SFT:      {"messages": [...], "metadata": {"persona_string": str}}
        where the assistant turn content is the response.
    """
    examples: list[ConsistencyExample] = []
    for rec in records:
        persona, response = _extract_persona_response(rec)
        examples.append(score_example(persona, response, backend))
    return _aggregate(examples)


def _extract_persona_response(record: dict[str, Any]) -> tuple[str, str]:
    persona = record.get("persona")
    response = record.get("response")
    if persona is None or response is None:
        messages = record.get("messages")
        metadata = record.get("metadata", {}) or {}
        if persona is None:
            persona = metadata.get("persona_string")
        if response is None and isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    response = msg.get("content")
                    break
    if not isinstance(persona, str) or not persona.strip():
        raise ValueError("record missing persona (persona / metadata.persona_string)")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("record missing response (response / assistant turn)")
    return persona, response


def _aggregate(examples: list[ConsistencyExample]) -> ConsistencyReport:
    n = len(examples)
    if n == 0:
        return ConsistencyReport(
            n=0, c_score=0.0, entail_rate=0.0, neutral_rate=0.0, contradict_rate=0.0, backend="unknown"
        )
    c_score = sum(e.score for e in examples) / n
    backend = examples[0].backend
    entail = sum(1 for e in examples if e.label == "entailment") / n
    neutral = sum(1 for e in examples if e.label == "neutral") / n
    contradict = sum(1 for e in examples if e.label == "contradiction") / n
    return ConsistencyReport(
        n=n,
        c_score=c_score,
        entail_rate=entail,
        neutral_rate=neutral,
        contradict_rate=contradict,
        backend=backend,
        examples=examples,
    )


# --------------------------------------------------------------------------- #
# I/O + CLI
# --------------------------------------------------------------------------- #


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("skipping malformed JSON at %s:%d: %s", path, lineno, exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PAL persona-consistency evaluation (paper §C.score, NLI entailment).",
    )
    parser.add_argument("input", type=Path, help="PAL JSONL: persona/response or SFT records.")
    parser.add_argument("--model", type=str, default=DEFAULT_NLI_MODEL)
    parser.add_argument(
        "--force-heuristic",
        action="store_true",
        help="Use the offline heuristic NLI (no model download). Numbers are NOT real.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N records.")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON report here.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    if not args.input.exists():
        return 1

    backend = build_nli_backend(args.model, force_heuristic=args.force_heuristic)
    records: Iterable[dict[str, Any]] = _iter_jsonl(args.input)
    if args.limit is not None:
        records = list(records)[: args.limit]

    report = score_records(records, backend)
    summary = {
        "n": report.n,
        "c_score": report.c_score,
        "entail_rate": report.entail_rate,
        "neutral_rate": report.neutral_rate,
        "contradict_rate": report.contradict_rate,
        "backend": report.backend,
        "model": args.model,
    }
    out = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(out + "\n", encoding="utf-8")
    # Exit 2 if the heuristic ran without --force-heuristic: warns the
    # operator the numbers are not publication-quality.
    if report.backend == "heuristic" and not args.force_heuristic:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
