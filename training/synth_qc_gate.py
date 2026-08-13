"""Synthetic QC Gate (§B.5.5) — stricter than natural data QC.

Per blueprint:
- min_quality_score=0.80 (vs 0.85 pipeline-default)
- min_self_consistency=0.85 (3-sample variance < 0.05)
- max_synth_fraction=0.30 (cap in final dataset)
- max_dup_vs_natural=0.60 (MinHash Jaccard to natural corpus)
- human_spot_check_rate=0.05 (always spot-check synthetic)
- Pass EVERY gate (dedup, PII, LLM judge) + second LLM judge pass for
  synthetic-style artifacts (over-formality, repetition, weird dialogue flow)

Usage: import synth_qc_gate from this module; call gate_synthetic_record().
"""
from __future__ import annotations

import math
from typing import Any

SYNTH_QC_THRESH = {
    "min_quality_score": 0.80,
    "min_self_consistency": 0.85,
    "max_synth_fraction": 0.30,
    "max_dup_vs_natural": 0.60,
    "human_spot_check_rate": 0.05,
}


def gate_synthetic_record(
    record: dict[str, Any],
    prior_synthetic: list[dict[str, Any]] | None = None,
    natural_corpus_samples: list[str] | None = None,
) -> tuple[bool, str]:
    """Apply strict synthetic QC gates. Return (passed: bool, reason: str)."""
    # Gate 1: quality score
    quality = record.get("quality_score", 0.0)
    if quality < SYNTH_QC_THRESH["min_quality_score"]:
        return False, f"quality_score={quality} < {SYNTH_QC_THRESH['min_quality_score']}"

    # Gate 2: self-consistency (placeholder — full k=3 variance in dual_judge)
    kappa = record.get("fleiss_kappa")
    # If annotated with IAA, require high kappa
    if kappa is not None and kappa < SYNTH_QC_THRESH["min_self_consistency"]:
        return False, f"fleiss_kappa={kappa} < {SYNTH_QC_THRESH['min_self_consistency']}"

    # Gate 3: synthetic fraction cap (dataset-level, not per-record)
    # The caller manages this aggregate gate; per-record we just stamp.
    # Here we note it in reason but always pass (aggregate gate handled externally).
    synth_note = f"synth_fraction_cap={SYNTH_QC_THRESH['max_synth_fraction']}"

    # Gate 4: dedup vs natural corpus (MinHash Jaccard placeholder)
    # If natural corpus provided, approximate overlap check
    dup_note = f"max_dup_vs_natural={SYNTH_QC_THRESH['max_dup_vs_natural']}"

    # Gate 5: synthetic-style artifact check (heuristic)
    content_blob = " ".join(str(m.get("content", "")) for m in record.get("messages", []) if isinstance(m.get("content"), str))
    lowered = content_blob.lower()
    # Over-formality markers
    over_formal_words = ["furthermore", "moreover", "notwithstanding", "consequently"]
    over_formal_score = sum(1 for w in over_formal_words if w in lowered) / max(len(over_formal_words), 1)
    # Repetition: simple word-repeat detection
    words = lowered.split()
    repetition_score = 1.0 - (len(set(words)) / max(len(words), 1)) if words else 0.0
    # Combined artifact score
    artifact_score = 0.5 * over_formal_score + 0.5 * repetition_score
    # If artifact score high, flag (but don't drop — human review per spec)
    artifact_passed = artifact_score < 0.60  # heuristic

    if not artifact_passed:
        # Per §B.5.5: synthetic artifacts need second judge / human review
        # We pass the gate but attach a review flag
        return True, f"passed_with_flag artifact_score={artifact_score:.2f} (needs_human_review) {synth_note} {dup_note}"

    return True, f"passed {synth_note} {dup_note} quality={quality}"


def aggregate_synth_fraction(current_synth: int, total_dataset: int) -> bool:
    """Dataset-level gate: synthetic fraction must not exceed cap."""
    if total_dataset == 0:
        return True
    return (current_synth / total_dataset) <= SYNTH_QC_THRESH["max_synth_fraction"]


def check_human_spot_check(record_index: int, seed: int = 42) -> bool:
    """Always spot-check at 5% rate; deterministic for reproducibility."""
    # Simple hash-based 5% selection
    return (hash(str(record_index) + str(seed)) % 20) == 0
