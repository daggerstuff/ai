#!/usr/bin/env python3
"""Merge clinician rating JSON files into the golden judge calibration set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PART_KEYS = ("clinical_validity", "therapeutic_alliance", "safety")
PART_WEIGHTS = (0.55, 0.35, 0.10)
BIN_EXCELLENT = 0.85
BIN_GOOD = 0.65
BIN_FAIR = 0.45
ACCEPT_THRESHOLD = 0.60
DIM_MAP = {
    "clinical_validity": ("relevance", "accuracy"),
    "therapeutic_alliance": ("helpfulness", "style"),
    "safety": ("safety",),
}


def _bin_of(overall: float) -> str:
    if overall >= BIN_EXCELLENT:
        return "excellent"
    if overall >= BIN_GOOD:
        return "good"
    if overall >= BIN_FAIR:
        return "fair"
    return "poor"


def _log(msg: str) -> None:
    sys.stdout.write(msg + "\n")


def _load_ratings(paths: list[str]) -> tuple[dict[str, dict[str, dict[str, float]]], list[str]]:
    by_sample: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    rater_ids: list[str] = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        rater = data["rater_id"]
        rater_ids.append(rater)
        for s in data["samples"]:
            row = {k: s.get(k) for k in PART_KEYS}
            if any(v is None or v == "" for v in row.values()):
                continue
            by_sample[s["id"]][rater] = {k: float(v) / 100.0 for k, v in row.items()}
    return by_sample, rater_ids


def _icc_oneway(matrix: list[list[float]]) -> float:
    k = len(matrix[0])
    n = len(matrix)
    grand = sum(sum(r) for r in matrix) / (n * k)
    msb = sum((sum(r) / k - grand) ** 2 for r in matrix) * k / (n - 1) if n > 1 else 0.0
    msw = sum((v - sum(r) / k) ** 2 for r in matrix for v in r) / (n * (k - 1)) if k > 1 else 0.0
    return (msb - msw) / (msb + (k - 1) * msw) if (msb + (k - 1) * msw) else 0.0


def _reliability_report(by_sample: dict[str, dict[str, dict[str, float]]], rater_ids: list[str]) -> dict[str, float]:
    per_dim: dict[str, list[list[float]]] = {k: [] for k in PART_KEYS}
    for sample in by_sample.values():
        for dim in PART_KEYS:
            row = [sample[r][dim] for r in rater_ids if r in sample]
            if len(row) == len(rater_ids):
                per_dim[dim].append(row)
    report: dict[str, float] = {}
    for dim in PART_KEYS:
        matrix = per_dim[dim]
        if not matrix:
            report[dim] = 0.0
            continue
        k = len(matrix[0])
        single = _icc_oneway(matrix)
        report[dim] = single
        report[f"{dim}_averaged_k{k}"] = k * single / (1 + (k - 1) * single) if single > 0 else 0.0
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rating_files", nargs="+", help="One or more exported rating JSON files")
    parser.add_argument("--samples", default="training/data/golden_judge_calib_v2.jsonl")
    parser.add_argument("--out", default="training/data/golden_judge_calib_human.jsonl")
    args = parser.parse_args()

    source_path = Path(args.samples)
    if not source_path.exists():
        _log(f"source samples not found: {source_path}")
        return 2
    source_records = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source = {rec["id"]: rec for rec in source_records}

    by_sample, rater_ids = _load_ratings(args.rating_files)
    _log(f"raters: {len(rater_ids)} -> {rater_ids}")
    all_rated = sum(1 for s in by_sample.values() if len(s) == len(rater_ids))
    _log(f"samples rated by all raters: {all_rated} / {len(source)}")

    report = _reliability_report(by_sample, rater_ids)
    _log("intraclass correlation (single / averaged):")
    for dim in PART_KEYS:
        avg_key = f"{dim}_averaged_k{len(rater_ids)}"
        _log(f"  {dim}: ICC1={report[dim]:.3f}  ICC_avg(k{len(rater_ids)})={report.get(avg_key, 0):.3f}")

    rows = []
    for sid, sample in source.items():
        raters = by_sample.get(sid, {})
        means: dict[str, float] = {}
        missing = False
        for dim in PART_KEYS:
            vals = [raters[r][dim] for r in rater_ids if r in raters]
            if vals:
                means[dim] = round(sum(vals) / len(vals), 4)
            else:
                missing = True
        if missing:
            _log(f"skip {sid}: missing ratings")
            continue
        five = {
            "relevance": means["clinical_validity"],
            "accuracy": means["clinical_validity"],
            "helpfulness": means["therapeutic_alliance"],
            "style": means["therapeutic_alliance"],
            "safety": means["safety"],
        }
        overall = round(sum(PART_WEIGHTS[i] * means[k] for i, k in enumerate(PART_KEYS)), 4)
        rows.append({
            "id": sid,
            "source": "clinician-rated",
            "conversation": sample.get("conversation", []),
            "human_scores": five,
            "human_scores_3part": means,
            "overall_quality": overall,
            "human_bin": _bin_of(overall),
            "provenance": f"human-rated n={len(rater_ids)}",
            "n_raters": len(rater_ids),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    accepts = sum(1 for r in rows if r["overall_quality"] >= ACCEPT_THRESHOLD)
    _log(f"wrote {len(rows)} records -> {out}")
    _log(f"accept(>=0.60) / reject(<0.60): {accepts} / {len(rows) - accepts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
