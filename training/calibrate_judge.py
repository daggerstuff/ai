#!/usr/bin/env python3
"""Calibration runner for the dual-model LLM quality judge (PIX-4345 §B.4 step 2B).

Runs ``DualModelQualityJudge.calibrate()`` against the 200-sample golden set and
emits a report JSON with the Pearson r / Cohen κ release-gate verdict.

Release gate (per blueprint step 2B):
  - Pearson r >= 0.80
  - Cohen κ >= 0.65 (quadratic-weighted)

CRITICAL: ``training/data/golden_judge_calib_v2.jsonl`` is the real golden set
(200 AnnoMI + ESConv clinical samples with expert-rule-based scores).

Usage:
  python calibrate_judge.py [--golden PATH] [--out PATH] [--allow-placeholder]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure ai directory is on sys.path when executed directly
_ai_root = Path(__file__).resolve().parent.parent
if str(_ai_root) not in sys.path:
    sys.path.insert(0, str(_ai_root))

try:
    from training.llm_quality_judge import (
        CALIB_KAPPA_MIN,
        CALIB_PEARSON_MIN,
        GOLDEN_CALIB_PATH,
        DualModelQualityJudge,
    )
except ImportError:  # pragma: no cover - direct execution with unpinned cwd
    sys.path.insert(0, str(_ai_root))
    from training.llm_quality_judge import (
        CALIB_KAPPA_MIN,
        CALIB_PEARSON_MIN,
        GOLDEN_CALIB_PATH,
        DualModelQualityJudge,
    )

PLACEHOLDER_NOTICE = (
    "golden judge set is synthetic/placeholder data — "
    "release-gate metrics are NOT representative of real human ratings. "
    "Regenerate the real golden set (golden_judge_calib_v2.jsonl) before "
    "trusting the gate verdict (see docs/plans/PIX-4343)."
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


def _write_report(out_path: Path, report: dict, kind: str = "report") -> None:
    """Persist a calibration report JSON and log where it landed."""
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[calibrate] {kind} written to {out_path}")


def _run_dry_run(golden_path: Path, out_path: Path) -> int:
    """Emit a synthetic report for placeholder data without hitting the LLM."""
    print("[calibrate] dry-run: skipping live LLM (placeholder data)")
    sample_count = sum(bool(_.strip()) for _ in open(golden_path, encoding="utf-8"))
    report = _build_report(
        golden_path,
        True,
        {
            "pearson_r": None,
            "cohens_kappa": None,
            "per_dimension_correlations": {},
            "sample_count": sample_count,
            "gate_passed": False,
        },
        thresholds={"pearson_min": CALIB_PEARSON_MIN, "kappa_min": CALIB_KAPPA_MIN},
        gate_blocked_reason="placeholder golden data — no real human labels",
    )
    _write_report(out_path, report, kind="dry-run report")
    print("[calibrate] gate_passed=False (placeholder — replace golden set with real labels)")
    return 0


def _build_report(
    golden_path: Path,
    placeholder: bool,
    data: dict,
    *,
    thresholds: dict,
    gate_blocked_reason: str | None = None,
) -> dict:
    """Assemble the calibration report dict (shared by dry-run and live paths)."""
    return {
        "golden_path": str(golden_path),
        "is_placeholder": placeholder,
        "pearson_r": data["pearson_r"],
        "cohens_kappa": data["cohens_kappa"],
        "per_dimension_correlations": data["per_dimension_correlations"],
        "sample_count": data["sample_count"],
        "thresholds": thresholds,
        "gate_passed": data["gate_passed"],
        "gate_blocked_reason": gate_blocked_reason,
        "placeholder_notice": PLACEHOLDER_NOTICE if placeholder else None,
    }


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
        return _run_dry_run(golden_path, out_path)

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

    report = _build_report(
        golden_path,
        placeholder,
        {
            "pearson_r": result.get("pearson_r"),
            "cohens_kappa": result.get("cohens_kappa"),
            "per_dimension_correlations": result.get("per_dimension_correlations", {}),
            "sample_count": result.get("sample_count", 0),
            "gate_passed": bool(result.get("pass", False)),
        },
        thresholds={"pearson_min": CALIB_PEARSON_MIN, "kappa_min": CALIB_KAPPA_MIN},
    )

    _write_report(out_path, report)
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
