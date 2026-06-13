#!/usr/bin/env python3
"""Spot-check SDG outputs for quality after API + scorer fixes (PIX-3879).

Samples 50 items per scenario from existing generated data, validates quality
with automatic checks, and produces a human review JSON report.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training.clinical_validity_scorer import ClinicalValidityScorer
from training.sdg_pipeline import (
    CRISIS_RESOURCES,
    STYLE_EVAL_RULES,
    _count_questions,
    _collect_style_marker_reasons,
    _repetition_reason,
    _safety_reason,
)

logger = logging.getLogger("spot_check_sdg")

# Banned platitudes from task description and extended set
BANNED_PLATITUDE_PHRASES = [
    "you matter",
    "great courage",
    "incredible job",
    "you are not alone",
    "that's completely normal",
    "it makes sense",
    "you deserve",
    "be gentle with yourself",
    "i hear you",
    "i understand",
    "i know that you",
    "it's okay",
    "i can imagine",
    "absolutely right",
    "exactly right",
    "you are very aware",
    "i completely agree",
    "as an ai",
    "let's focus",
    "it's important to know",
]

SPOT_CHECK_SEED = 42
TARGET_PER_SCENARIO = 50


def _sample_from_jsonl(path: Path, target: int) -> list[dict]:
    """Read a JSONL file and deterministically sample `target` records."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if len(records) <= target:
        return records
    random.seed(SPOT_CHECK_SEED)
    return random.sample(records, target)


def _check_human_sounding(text: str) -> tuple[bool, list[str]]:
    """Check for garbled / mashup / non-human signals."""
    issues: list[str] = []
    text_lower = text.lower().strip()
    if not text_lower:
        issues.append("Empty output")
        return False, issues
    # Check for JSON fragments that leaked through
    if text.strip().startswith("{") and ("\"" in text or "'" in text):
        issues.append("Suspected JSON fragment in output")
    # Check for excessive repetition
    rep = _repetition_reason(text_lower)
    if rep:
        issues.append(rep)
    # Check robotic signals
    for marker in STYLE_EVAL_RULES["robotic_signals"]:
        if marker in text_lower:
            issues.append(f"Robotic signal: '{marker}'")
    return len(issues) == 0, issues


def _check_topic_match(instruction: str, output: str) -> tuple[bool, list[str]]:
    """Validate that output addresses the instruction topic (heuristic)."""
    issues: list[str] = []
    # If output is too short it's unlikely to address anything
    if len(output.strip()) < 40:
        issues.append("Output too short to be responsive")
    # Topic drift heuristics could go here; for now we keep it lightweight
    return len(issues) == 0, issues


def _check_banned_platitudes(output: str) -> tuple[bool, list[str]]:
    """Check output for banned affirmation / platitude phrases."""
    issues: list[str] = []
    output_lower = output.lower()
    for phrase in BANNED_PLATITUDE_PHRASES:
        if phrase in output_lower:
            issues.append(f"Banned platitude: '{phrase}'")
    return len(issues) == 0, issues


def _check_crisis_resources(output: str) -> tuple[bool, list[str]]:
    """Nightmare fuel / crisis scenarios MUST contain a crisis resource."""
    issues: list[str] = []
    output_lower = output.lower()
    if any(resource.lower() in output_lower for resource in CRISIS_RESOURCES):
        return True, []
    issues.append("Missing crisis resource (e.g., 988, Crisis Text Line)")
    return False, issues


def _check_dpo_plausible_but_inferior(pair: dict) -> tuple[bool, list[str]]:
    """Rejected DPO response should be clinically plausible but worse than chosen."""
    issues: list[str] = []
    chosen = pair.get("chosen", "").strip()
    rejected = pair.get("rejected", "").strip()
    if not chosen or not rejected:
        issues.append("Missing chosen or rejected text")
        return False, issues
    # Rejected should be shorter or less supportive
    if len(rejected) > len(chosen) * 1.5:
        issues.append("Rejected longer than chosen — suspicious")
    # Run scorer on both
    try:
        chosen_score = ClinicalValidityScorer.score(chosen)
        rejected_score = ClinicalValidityScorer.score(rejected)
        if rejected_score >= chosen_score:
            issues.append(
                f"Rejected score ({rejected_score:.2f}) >= chosen ({chosen_score:.2f})"
            )
    except Exception:
        pass
    return len(issues) == 0, issues


def _spot_check_dpo(samples: list[dict]) -> dict:
    """Run spot-check validation on DPO pairs."""
    results: list[dict] = []
    valid_scores: list[float] = []
    for sample in samples:
        issues: list[str] = []
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")

        ok, hs_issues = _check_human_sounding(chosen)
        issues.extend(hs_issues)

        ok, tm_issues = _check_topic_match(prompt, chosen)
        issues.extend(tm_issues)

        ok, bp_issues = _check_banned_platitudes(chosen)
        issues.extend(bp_issues)

        ok, dpo_issues = _check_dpo_plausible_but_inferior(sample)
        issues.extend(dpo_issues)

        # Score chosen response
        try:
            score = ClinicalValidityScorer.score(chosen)
            detail = ClinicalValidityScorer.score_detail(chosen)
        except Exception:
            score = 0.0
            detail = {}

        if score > 0.0:
            valid_scores.append(score)

        results.append(
            {
                "id": sample.get("metadata", {}).get("pair_type", "unknown"),
                "prompt_preview": prompt[:120] + "..." if len(prompt) > 120 else prompt,
                "issues": issues,
                "clinical_validity_score": round(score, 3),
                "clinical_validity_detail": detail,
            }
        )

    pass_rate = (
        round(sum(1 for r in results if not r["issues"]) / len(results), 3)
        if results
        else 0.0
    )
    score_stats = {}
    if valid_scores:
        score_stats = {
            "mean": round(statistics.mean(valid_scores), 3),
            "median": round(statistics.median(valid_scores), 3),
            "min": round(min(valid_scores), 3),
            "max": round(max(valid_scores), 3),
            "stdev": round(statistics.stdev(valid_scores), 3) if len(valid_scores) > 1 else 0.0,
        }

    return {
        "scenario": "dpo_preference_pairs",
        "sample_count": len(results),
        "pass_rate": pass_rate,
        "clinical_validity_distribution": score_stats,
        "samples": results,
    }


def _spot_check_niche(samples: list[dict]) -> dict:
    """Run spot-check validation on niche category samples (DPO format)."""
    results: list[dict] = []
    valid_scores: list[float] = []
    for sample in samples:
        issues: list[str] = []
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")

        ok, hs_issues = _check_human_sounding(chosen)
        issues.extend(hs_issues)

        ok, tm_issues = _check_topic_match(prompt, chosen)
        issues.extend(tm_issues)

        ok, bp_issues = _check_banned_platitudes(chosen)
        issues.extend(bp_issues)

        ok, dpo_issues = _check_dpo_plausible_but_inferior(sample)
        issues.extend(dpo_issues)

        try:
            score = ClinicalValidityScorer.score(chosen)
            detail = ClinicalValidityScorer.score_detail(chosen)
        except Exception:
            score = 0.0
            detail = {}

        if score > 0.0:
            valid_scores.append(score)

        results.append(
            {
                "id": sample.get("metadata", {}).get("domain", "unknown"),
                "prompt_preview": prompt[:120] + "..." if len(prompt) > 120 else prompt,
                "issues": issues,
                "clinical_validity_score": round(score, 3),
                "clinical_validity_detail": detail,
            }
        )

    pass_rate = (
        round(sum(1 for r in results if not r["issues"]) / len(results), 3)
        if results
        else 0.0
    )
    score_stats = {}
    if valid_scores:
        score_stats = {
            "mean": round(statistics.mean(valid_scores), 3),
            "median": round(statistics.median(valid_scores), 3),
            "min": round(min(valid_scores), 3),
            "max": round(max(valid_scores), 3),
            "stdev": round(statistics.stdev(valid_scores), 3) if len(valid_scores) > 1 else 0.0,
        }

    return {
        "scenario": "niche_category",
        "sample_count": len(results),
        "pass_rate": pass_rate,
        "clinical_validity_distribution": score_stats,
        "samples": results,
    }


def _spot_check_nightmare(samples: list[dict]) -> dict:
    """Run spot-check validation on nightmare fuel samples (DPO format)."""
    results: list[dict] = []
    valid_scores: list[float] = []
    for sample in samples:
        issues: list[str] = []
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")

        ok, hs_issues = _check_human_sounding(chosen)
        issues.extend(hs_issues)

        ok, tm_issues = _check_topic_match(prompt, chosen)
        issues.extend(tm_issues)

        ok, bp_issues = _check_banned_platitudes(chosen)
        issues.extend(bp_issues)

        ok, crisis_issues = _check_crisis_resources(chosen)
        issues.extend(crisis_issues)

        ok, dpo_issues = _check_dpo_plausible_but_inferior(sample)
        issues.extend(dpo_issues)

        try:
            score = ClinicalValidityScorer.score(chosen)
            detail = ClinicalValidityScorer.score_detail(chosen)
        except Exception:
            score = 0.0
            detail = {}

        if score > 0.0:
            valid_scores.append(score)

        results.append(
            {
                "id": sample.get("metadata", {}).get("category", "unknown"),
                "prompt_preview": prompt[:120] + "..." if len(prompt) > 120 else prompt,
                "issues": issues,
                "clinical_validity_score": round(score, 3),
                "clinical_validity_detail": detail,
            }
        )

    pass_rate = (
        round(sum(1 for r in results if not r["issues"]) / len(results), 3)
        if results
        else 0.0
    )
    score_stats = {}
    if valid_scores:
        score_stats = {
            "mean": round(statistics.mean(valid_scores), 3),
            "median": round(statistics.median(valid_scores), 3),
            "min": round(min(valid_scores), 3),
            "max": round(max(valid_scores), 3),
            "stdev": round(statistics.stdev(valid_scores), 3) if len(valid_scores) > 1 else 0.0,
        }

    return {
        "scenario": "nightmare_fuel",
        "sample_count": len(results),
        "pass_rate": pass_rate,
        "clinical_validity_distribution": score_stats,
        "samples": results,
    }


def run_spot_check(
    dpo_path: Path,
    niche_path: Path,
    nightmare_path: Path,
    output_report_path: Path,
    target_per_scenario: int = TARGET_PER_SCENARIO,
) -> dict:
    """Run full spot-check suite and write JSON report."""
    dpo_samples = _sample_from_jsonl(dpo_path, target_per_scenario)
    niche_samples = _sample_from_jsonl(niche_path, target_per_scenario)
    nightmare_samples = _sample_from_jsonl(nightmare_path, target_per_scenario)

    logger.info("Spot-check samples loaded: DPO=%d, niche=%d, nightmare=%d", len(dpo_samples), len(niche_samples), len(nightmare_samples))

    report = {
        "spot_check_version": "1.0",
        "target_per_scenario": target_per_scenario,
        "dpo": _spot_check_dpo(dpo_samples),
        "niche": _spot_check_niche(niche_samples),
        "nightmare": _spot_check_nightmare(nightmare_samples),
    }

    # Overall summary
    overall_passed = sum(
        sum(1 for s in scenario["samples"] if not s["issues"])
        for scenario in [report["dpo"], report["niche"], report["nightmare"]]
    )
    overall_total = sum(scenario["sample_count"] for scenario in [report["dpo"], report["niche"], report["nightmare"]])
    report["overall"] = {
        "total_samples": overall_total,
        "total_passed": overall_passed,
        "overall_pass_rate": round(overall_passed / overall_total, 3) if overall_total else 0.0,
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Spot-check report written to %s", output_report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spot-check SDG outputs for quality")
    parser.add_argument(
        "--dpo_path",
        type=Path,
        default=Path("ai/training/data/synthetic_for_dedup/dpo_preference_pairs_10k.jsonl"),
    )
    parser.add_argument(
        "--niche_path",
        type=Path,
        default=Path("ai/training/data/synthetic_deduped/shard_0000.jsonl"),
    )
    parser.add_argument(
        "--nightmare_path",
        type=Path,
        default=Path("ai/training/data/synthetic_for_dedup/nightmare_fuel_5k.jsonl"),
    )
    parser.add_argument(
        "--output_report",
        type=Path,
        default=Path("ai/training/spot_check_report.json"),
    )
    parser.add_argument("--target", type=int, default=TARGET_PER_SCENARIO)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    run_spot_check(args.dpo_path, args.niche_path, args.nightmare_path, args.output_report, args.target)


if __name__ == "__main__":
    main()
