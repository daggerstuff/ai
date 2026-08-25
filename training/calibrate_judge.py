#!/usr/bin/env python3
"""Calibration runner for the dual-model LLM quality judge (PIX-4345 §B.4 step 2B).

Runs ``DualModelQualityJudge.calibrate()`` against the 200-sample golden set and
emits a report JSON with the Pearson r / Cohen κ release-gate verdict.

Release gate (per blueprint step 2B):
  - Pearson r >= 0.80
  - Cohen κ >= 0.65 (quadratic-weighted)

CRITICAL: ``training/data/golden_judge_calib.jsonl`` is synthetic/placeholder
data (see llm_quality_judge.py:GOLDEN_CALIB_PATH warning). The gate numbers from
this runner are NOT representative of real human ratings until the golden set is
replaced with real human labels. The runner fails closed on placeholder data by
default (``--allow-placeholder`` overrides for dry-runs).

Usage:
  python calibrate_judge.py [--golden PATH] [--out PATH] [--allow-placeholder]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from training.llm_quality_judge import (
    CALIB_KAPPA_MIN,
    CALIB_PEARSON_MIN,
    GOLDEN_CALIB_PATH,
    DualModelQualityJudge,
)

PLACEHOLDER_NOTICE = (
    "golden_judge_calib.jsonl is synthetic/placeholder data — "
    "release-gate metrics are NOT representative of real human ratings. "
    "Replace with real human labels before trusting the gate verdict "
    "(see docs/plans/PIX-4343)."
)


def _is_placeholder(golden_path: Path) -> bool:
    """Heuristic: placeholder golden file is Neon AI Gateway consensus data.

    The placeholder file's records use 'neon-consensus-NNNN' ids and carry a
    ``_neon_consensus_label`` / ``_data_source_note`` marker rather than real
    human labels. Detection matches both the id prefix and the explicit marker.
    """
    if not golden_path.exists():
        return False
    with open(golden_path, encoding="utf-8") as f:
        first = f.readline().strip()
    try:
        rec = json.loads(first)
        rid = str(rec.get("id", ""))
        marked = bool(
            rec.get("_neon_consensus_label")
            or rec.get("_synthetic_golden_calibration")
            or "consensus" in str(rec.get("_data_source_note", "")).lower()
        )
        return rid.startswith(("neon-consensus-", "golden-")) or marked
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM judge calibration runner (PIX-4345 §B.4 step 2B)")
    parser.add_argument("--golden", type=str, default=str(GOLDEN_CALIB_PATH), help="Path to golden JSONL")
    parser.add_argument(
        "--out", type=str, default="ai/training/output/calibration_report.json", help="Path to write report JSON"
    )
    parser.add_argument(
        "--allow-placeholder", action="store_true", help="Run against placeholder golden data (dry-run only)"
    )
    args = parser.parse_args()

    golden_path = Path(args.golden)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not golden_path.exists():
        print(f"FAIL: golden calibration file not found: {golden_path}")
        return 2

    placeholder = _is_placeholder(golden_path)
    if placeholder and not args.allow_placeholder:
        print(f"REFUSING to run: {golden_path} looks like placeholder data.")
        print(f"  {PLACEHOLDER_NOTICE}")
        print("  Pass --allow-placeholder for a dry-run against the synthetic set.")
        return 3

    print(f"[calibrate] golden={golden_path} placeholder={placeholder}")
    if placeholder:
        print(f"[calibrate] WARN: {PLACEHOLDER_NOTICE}")

    # Dry-run on placeholder data: do NOT hit the LLM. Emit a synthetic report
    # so the harness is exercised without spending API calls on fake labels.
    if placeholder:
        print("[calibrate] dry-run: skipping live LLM (placeholder data)")
        sample_count = sum(1 for _ in open(golden_path, encoding="utf-8") if _.strip())
        report = {
            "golden_path": str(golden_path),
            "is_placeholder": True,
            "pearson_r": None,
            "cohens_kappa": None,
            "per_dimension_correlations": {},
            "sample_count": sample_count,
            "thresholds": {"pearson_min": CALIB_PEARSON_MIN, "kappa_min": CALIB_KAPPA_MIN},
            "gate_passed": False,
            "gate_blocked_reason": "placeholder golden data — no real human labels",
            "placeholder_notice": PLACEHOLDER_NOTICE,
        }
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[calibrate] dry-run report written to {out_path}")
        print("[calibrate] gate_passed=False (placeholder — replace golden set with real labels)")
        return 0

    if not (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("FAIL: no LLM_API_KEY or OPENAI_API_KEY set — judge needs a live LLM endpoint.")
        print("  Configure a vLLM/OpenAI-compatible endpoint (Qwen-72B primary, LLaMA-70B secondary).")
        return 4

    try:
        judge = DualModelQualityJudge()
    except Exception as e:
        print(f"FAIL: could not construct judge: {e}")
        return 5

    print("[calibrate] running calibration (this calls the LLM per sample; may take a while)...")
    try:
        result = judge.calibrate(golden_path=golden_path)
    except Exception as e:
        print(f"FAIL: calibration run errored: {e}")
        return 6

    report = {
        "golden_path": str(golden_path),
        "is_placeholder": placeholder,
        "pearson_r": result.get("pearson_r"),
        "cohens_kappa": result.get("cohens_kappa"),
        "per_dimension_correlations": result.get("per_dimension_correlations", {}),
        "sample_count": result.get("sample_count", 0),
        "thresholds": {"pearson_min": CALIB_PEARSON_MIN, "kappa_min": CALIB_KAPPA_MIN},
        "gate_passed": bool(result.get("pass", False)),
        "placeholder_notice": PLACEHOLDER_NOTICE if placeholder else None,
    }

    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[calibrate] report written to {out_path}")
    print(
        f"[calibrate] pearson_r={report['pearson_r']} cohens_kappa={report['cohens_kappa']} "
        f"gate_passed={report['gate_passed']}"
    )
    if placeholder:
        print(f"[calibrate] NOTE: {PLACEHOLDER_NOTICE}")
        return 0  # dry-run; don't fail CI on placeholder

    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
